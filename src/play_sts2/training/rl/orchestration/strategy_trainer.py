"""组合 GiGPO backbone 与 terminal Tree 更新战略 residual。"""

import json
import math
import random
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from play_sts2.training.sft import publish_adapter, resolve_device

from ..battle_trainer import (
    FrozenAnchor,
    backward_grpo_group,
    evaluate_grpo_group,
    grpo_base_model_dtype,
)
from ..gigpo import load_gigpo_training_group
from ..learner import GrpoTrainingError
from ..treegpo import load_terminal_tree_training_group


@dataclass(frozen=True, slots=True)
class StrategyGrpoConfig:
    """保存一轮组合战略更新配置。

    Args:
        base_model (Path): 已合并最终 SFT 的冻结 Hugging Face base。
        init_adapter (Path): 本轮冻结 ``S_n`` residual。
        policy_model (str): backbone 与 Tree 绑定的 vLLM 模型名。
        run_role (str): ``engineering_smoke`` 或 ``formal``。
        gigpo_path (Path): 当前同种子八局 GiGPO group。
        tree_root (Path): 零至多个 terminal Tree group 文件或目录。
        adapter_root (Path): ``S_(n+1)`` 输出父目录。
        runs_root (Path): 指标与配置输出父目录。
        device (str): 单张 CUDA 设备。
        learning_rate (float): AdamW 学习率。
        max_length (int): 单步 stateless 最大 token 数。
        max_grad_norm (float): 梯度裁剪上限。
        logits_chunk_size (int): 词表投影分块大小。
        seed (int): 训练随机种子。
        clip (float): PPO 对称裁剪半径。
        kl_beta (float): 冻结父 residual sampled-KL 权重。
        backbone_loss_weight (float): 有 Tree 时 GiGPO 数据源权重。
        tree_loss_weight (float): 有 Tree 时全部 Tree 数据源总权重。
    """

    base_model: Path
    init_adapter: Path
    policy_model: str
    run_role: str
    gigpo_path: Path
    tree_root: Path
    adapter_root: Path
    runs_root: Path
    device: str
    learning_rate: float
    max_length: int
    max_grad_norm: float
    logits_chunk_size: int
    seed: int
    clip: float
    kl_beta: float
    backbone_loss_weight: float = 0.5
    tree_loss_weight: float = 0.5
    dagger_weight: float = 0.0
    dagger_samples_per_group: int = 0


def load_strategy_grpo_config(path: Path) -> StrategyGrpoConfig:
    """从 TOML 加载组合战略训练配置。

    Args:
        path (Path): 配置文件路径。

    Raises:
        GrpoTrainingError: 字段缺失、设备或超参数无效。

    Returns:
        StrategyGrpoConfig: 已验证的单卡训练配置。
    """
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    try:
        config = StrategyGrpoConfig(
            base_model=Path(str(data["base_model"])),
            init_adapter=Path(str(data["init_adapter"])),
            policy_model=str(data["policy_model"]),
            run_role=str(data["run_role"]),
            gigpo_path=Path(str(data["gigpo_path"])),
            tree_root=Path(str(data["tree_root"])),
            adapter_root=Path(str(data["adapter_root"])),
            runs_root=Path(str(data["runs_root"])),
            device=str(data["device"]),
            learning_rate=float(data["learning_rate"]),
            max_length=int(data["max_length"]),
            max_grad_norm=float(data.get("max_grad_norm", 1.0)),
            logits_chunk_size=int(data.get("logits_chunk_size", 128)),
            seed=int(data["seed"]),
            clip=float(data.get("clip", 0.2)),
            kl_beta=float(data.get("kl_beta", 0.02)),
            backbone_loss_weight=float(data.get("backbone_loss_weight", 0.5)),
            tree_loss_weight=float(data.get("tree_loss_weight", 0.5)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GrpoTrainingError("组合战略 GRPO 配置字段无效") from exc
    weights = config.backbone_loss_weight + config.tree_loss_weight
    if (
        config.run_role not in {"engineering_smoke", "formal"}
        or not re.fullmatch(r"cuda:\d+", config.device)
        or not config.policy_model
        or config.learning_rate <= 0
        or config.max_length <= 0
        or config.max_grad_norm <= 0
        or config.logits_chunk_size <= 0
        or not 0 <= config.clip < 1
        or config.kl_beta < 0
        or config.backbone_loss_weight <= 0
        or config.tree_loss_weight <= 0
        or not math.isclose(weights, 1.0, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise GrpoTrainingError("组合战略 GRPO 配置值无效")
    return config


def train_strategy_grpo(
    config: StrategyGrpoConfig,
    run_name: str,
) -> dict[str, Any]:
    """在同一 optimizer step 中组合 GiGPO 与 terminal Tree loss。

    没有有效 Tree group 时只使用 backbone，且不会复制数据凑 0.5 权重。所有
    rollout 必须绑定同一个冻结 ``S_n``；训练完成后只发布一个 ``S_(n+1)``。

    Args:
        config (StrategyGrpoConfig): 已验证训练配置。
        run_name (str): 新 residual 与运行目录名称。

    Raises:
        GrpoTrainingError: 输出、policy、数据、模型或梯度不满足合同。

    Returns:
        dict[str, Any]: 组合 loss、ratio、KL、梯度与输出路径摘要。
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise GrpoTrainingError("组合战略 GRPO run_name 无效")
    adapter_path = config.adapter_root / run_name
    run_path = config.runs_root / run_name
    if adapter_path.exists() or run_path.exists():
        raise GrpoTrainingError(f"组合战略 GRPO 输出已存在: {run_name}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.base_model),
        local_files_only=True,
        trust_remote_code=False,
    )
    backbone = load_gigpo_training_group(
        config.gigpo_path,
        tokenizer,
        max_length=config.max_length,
    )
    tree_groups = tuple(
        load_terminal_tree_training_group(
            path,
            tokenizer,
            max_length=config.max_length,
        )
        for path in _terminal_tree_paths(config.tree_root)
    )
    groups = (backbone, *tree_groups)
    _validate_strategy_group_receipts(groups, expected_strategy=config.policy_model)

    base_model = AutoModelForCausalLM.from_pretrained(
        str(config.base_model),
        dtype=grpo_base_model_dtype(device),
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
    from peft import PeftModel

    model = PeftModel.from_pretrained(
        base_model,
        str(config.init_adapter),
        is_trainable=True,
        autocast_adapter_dtype=True,
    )
    model.to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise GrpoTrainingError("组合战略模型没有可训练 residual 参数")
    import torch

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    anchor = FrozenAnchor(model)
    optimizer.zero_grad(set_to_none=True)
    has_tree = bool(tree_groups)
    backbone_weight = config.backbone_loss_weight if has_tree else 1.0
    backbone_metric = backward_grpo_group(
        model,
        anchor,
        backbone.arms,
        config=config,
        device=device,
        loss_scale=backbone_weight,
    )
    tree_metrics = []
    for group in tree_groups:
        tree_metrics.append(
            backward_grpo_group(
                model,
                anchor,
                group.arms,
                config=config,
                device=device,
                loss_scale=config.tree_loss_weight / len(tree_groups),
            )
        )
    invalid_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and not torch.isfinite(parameter.grad).all().item()
    ]
    if invalid_gradients:
        raise GrpoTrainingError(
            f"组合战略 GRPO 梯度不是有限数值: {invalid_gradients[0]}"
        )
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
    if not torch.isfinite(grad_norm).item():
        raise GrpoTrainingError("组合战略 GRPO 梯度范数无效")
    optimizer.step()
    post_backbone = evaluate_grpo_group(
        model,
        anchor,
        backbone.arms,
        config=config,
        device=device,
    )
    post_tree = [
        evaluate_grpo_group(
            model,
            anchor,
            group.arms,
            config=config,
            device=device,
        )
        for group in tree_groups
    ]
    post_metrics = (post_backbone, *post_tree)
    if any(not _post_update_metrics_safe(metric) for metric in post_metrics):
        anchor.restore()
        raise GrpoTrainingError("组合战略更新超过 post-step KL/ratio 硬门槛，已回滚")

    summary = {
        "run_name": run_name,
        "status": "completed",
        "run_role": config.run_role,
        "engineering_smoke": config.run_role == "engineering_smoke",
        "base_model": str(config.base_model),
        "init_adapter": str(config.init_adapter),
        "policy_model": config.policy_model,
        "battle_policy_model": backbone.battle_policy_version,
        "generation_profile": dict(backbone.generation_profile or {}),
        "environment": dict(backbone.environment or {}),
        "gigpo_path": str(config.gigpo_path),
        "tree_groups": len(tree_groups),
        "backbone_loss_weight": backbone_weight,
        "tree_loss_weight": config.tree_loss_weight if has_tree else 0.0,
        "backbone_metric": backbone_metric,
        "tree_metric": _mean_metrics(tree_metrics),
        "post_backbone_metric": post_backbone,
        "post_tree_metric": _mean_metrics(post_tree),
        "grad_norm": float(grad_norm.detach().cpu()),
        "optimizer_steps": 1,
        "adapter": str(adapter_path),
        "device": device,
    }
    run_path.mkdir(parents=True)
    (run_path / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    publish_adapter(model, tokenizer, adapter_path, summary)
    (run_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    from .telemetry import TensorboardMetricsWriter

    with TensorboardMetricsWriter(run_path / "tensorboard") as writer:
        writer.write(strategy_tensorboard_payload(summary), step=1)
    return summary


def strategy_tensorboard_payload(summary: dict[str, Any]) -> dict[str, Any]:
    """把组合战略摘要拆成 backbone、Tree 与 post-step 曲线。

    Args:
        summary (dict[str, Any]): ``train_strategy_grpo`` 的完成摘要。

    Returns:
        dict[str, Any]: 可由 TensorBoard writer 递归展开的数值对象。
    """
    return {
        "strategy": {"grad_norm": summary.get("grad_norm")},
        "strategy/backbone": summary.get("backbone_metric"),
        "strategy/tree": summary.get("tree_metric"),
        "strategy/post_backbone": summary.get("post_backbone_metric"),
        "strategy/post_tree": summary.get("post_tree_metric"),
    }


def _terminal_tree_paths(root: Path) -> tuple[Path, ...]:
    """返回目录中明确声明为 terminal Tree 的 JSON 文件。

    Args:
        root (Path): 单个 JSON、目录或尚不存在的可选路径。

    Returns:
        tuple[Path, ...]: 按路径排序的 terminal Tree group。
    """
    path = Path(root)
    candidates = (
        (path,)
        if path.is_file()
        else tuple(sorted(path.rglob("*.json")))
        if path.is_dir()
        else ()
    )
    selected = []
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("format") == "terminal_tree_group":
            selected.append(candidate)
    return tuple(selected)


def _validate_strategy_group_receipts(
    groups: tuple[Any, ...],
    *,
    expected_strategy: str,
) -> None:
    """要求同一步组合的 GiGPO 与 Tree 数据来自同一冻结运行条件。

    Args:
        groups (tuple[Any, ...]): 一个 backbone 与零至多个 Tree 训练组。
        expected_strategy (str): 配置声明的冻结战略 policy。

    Raises:
        GrpoTrainingError: 双 policy、生成参数或运行环境任一不一致。

    Returns:
        None: 全部 group 可进入同一 optimizer step 时返回。
    """
    if not groups:
        raise GrpoTrainingError("组合战略训练缺少 rollout group")
    baseline = groups[0]
    if (
        baseline.policy_version != expected_strategy
        or not baseline.battle_policy_version
        or baseline.generation_profile is None
        or baseline.environment is None
    ):
        raise GrpoTrainingError("组合战略 backbone 缺少完整冻结运行收据")
    for group in groups[1:]:
        if (
            group.policy_version != expected_strategy
            or group.battle_policy_version != baseline.battle_policy_version
            or group.generation_profile != baseline.generation_profile
            or group.environment != baseline.environment
        ):
            raise GrpoTrainingError("组合战略 group 混入不同 B_n、生成参数或运行环境")


def _post_update_metrics_safe(metric: dict[str, float]) -> bool:
    """判断候选战略更新是否仍位于父策略附近。

    Args:
        metric (dict[str, float]): 更新后重新前向得到的 KL 与 ratio。

    Returns:
        bool: KL 不超过 0.05 且平均 ratio 位于 0.5 到 2.0 时返回真。
    """
    kl = metric.get("kl", math.inf)
    ratio = metric.get("ratio_mean", math.inf)
    return math.isfinite(kl) and kl <= 0.05 and 0.5 <= ratio <= 2.0


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float] | None:
    """对多个 Tree group 指标求简单均值。

    Args:
        metrics (list[dict[str, float]]): 每个 Tree group 的可读指标。

    Returns:
        dict[str, float] | None: 有输入时返回逐键均值，否则为空。
    """
    if not metrics:
        return None
    return {
        key: sum(metric[key] for metric in metrics) / len(metrics) for key in metrics[0]
    }
