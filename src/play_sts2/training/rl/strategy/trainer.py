"""使用共用 token-level GRPO 优化器执行一次战略 Tree smoke。"""

import json
import random
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from play_sts2.training.sft import publish_adapter, resolve_device

from ..battle_trainer import (
    FrozenAnchor,
    grpo_base_model_dtype,
    optimize_grpo_group,
)
from ..learner import GrpoTrainingError
from .learner import load_tree_training_group


@dataclass(frozen=True, slots=True)
class TreeGrpoConfig:
    """保存 Tree-GRPO 单卡可行性训练配置。

    Args:
        base_model (Path): Hugging Face Qwen3.5-4B 基座。
        init_adapter (Path): 冻结父 adapter。
        policy_model (str): rollout 与 learner 绑定的模型名。
        run_role (str): 本阶段只允许 ``engineering_smoke``。
        rollout_path (Path): 一个已准入 K=8 Tree group。
        adapter_root (Path): smoke adapter 输出父目录。
        runs_root (Path): 指标与配置输出父目录。
        device (str): 单张 CUDA 设备。
        learning_rate (float): AdamW 学习率。
        max_length (int): 单步对话最大 token 数。
        max_grad_norm (float): 梯度裁剪上限。
        logits_chunk_size (int): 词表投影分块大小。
        seed (int): 训练随机种子。
        clip (float): PPO 裁剪半径。
        kl_beta (float): 冻结父模型 sampled-KL 权重。
    """

    base_model: Path
    init_adapter: Path
    policy_model: str
    run_role: str
    rollout_path: Path
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
    dagger_weight: float = 0.0
    dagger_samples_per_group: int = 0


def load_tree_grpo_config(path: Path) -> TreeGrpoConfig:
    """读取只允许单卡工程 smoke 的 Tree 配置。

    Args:
        path (Path): TOML 配置文件。

    Raises:
        GrpoTrainingError: 字段缺失、数值无效或试图标成正式训练。
        OSError: 配置无法读取。
        tomllib.TOMLDecodeError: TOML 语法无效。

    Returns:
        TreeGrpoConfig: 已验证配置。
    """
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    try:
        config = TreeGrpoConfig(
            base_model=Path(str(data["base_model"])),
            init_adapter=Path(str(data["init_adapter"])),
            policy_model=str(data["policy_model"]),
            run_role=str(data["run_role"]),
            rollout_path=Path(str(data["rollout_path"])),
            adapter_root=Path(str(data["adapter_root"])),
            runs_root=Path(str(data["runs_root"])),
            device=str(data["device"]),
            learning_rate=float(data["learning_rate"]),
            max_length=int(data["max_length"]),
            max_grad_norm=float(data["max_grad_norm"]),
            logits_chunk_size=int(data["logits_chunk_size"]),
            seed=int(data["seed"]),
            clip=float(data["clip"]),
            kl_beta=float(data["kl_beta"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GrpoTrainingError("Tree-GRPO 配置字段无效") from exc
    if (
        config.run_role != "engineering_smoke"
        or not re.fullmatch(r"cuda:\d+", config.device)
        or not config.policy_model
        or config.learning_rate <= 0
        or config.max_length <= 0
        or config.max_grad_norm <= 0
        or config.logits_chunk_size <= 0
        or not 0 <= config.clip < 1
        or config.kl_beta < 0
    ):
        raise GrpoTrainingError(
            "Tree-GRPO 可行性配置必须为 engineering_smoke、单张 CUDA 和有效超参数"
        )
    return config


def train_tree_grpo(
    config: TreeGrpoConfig,
    run_name: str,
) -> dict[str, Any]:
    """在一个 K=8 Tree group 上执行一次 LoRA 更新并发布 smoke adapter。

    Args:
        config (TreeGrpoConfig): 单卡可行性配置。
        run_name (str): smoke adapter 与指标目录名称。

    Raises:
        GrpoTrainingError: 输出、policy、group、模型或梯度不符合工程契约。
        OSError: 模型或输出文件无法读取写入。

    Returns:
        dict[str, Any]: 含 ratio、KL、梯度和 adapter 路径的可行性摘要。
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise GrpoTrainingError("Tree-GRPO run_name 无效")
    adapter_path = config.adapter_root / run_name
    run_path = config.runs_root / run_name
    if adapter_path.exists() or run_path.exists():
        raise GrpoTrainingError(f"Tree-GRPO smoke 输出已存在: {run_name}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.base_model),
        local_files_only=True,
        trust_remote_code=False,
    )
    group = load_tree_training_group(
        config.rollout_path,
        tokenizer,
        max_length=config.max_length,
    )
    if group.policy_version != config.policy_model:
        raise GrpoTrainingError("Tree group 与配置冻结 policy 不一致")
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
        raise GrpoTrainingError("Tree-GRPO 模型没有可训练 adapter 参数")
    torch = __import__("torch")
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    anchor = FrozenAnchor(model)
    metric, _cursor = optimize_grpo_group(
        model,
        optimizer,
        anchor,
        group.arms,
        (),
        dagger_cursor=0,
        config=config,  # type: ignore[arg-type]
        device=device,
    )
    if any(not math_is_finite(value) for value in metric.values()):
        raise GrpoTrainingError("Tree-GRPO smoke 指标不是有限值")
    summary = {
        "run_name": run_name,
        "status": "completed",
        "run_role": config.run_role,
        "engineering_smoke": True,
        "base_model": str(config.base_model),
        "init_adapter": str(config.init_adapter),
        "policy_model": config.policy_model,
        "rollout": str(config.rollout_path),
        "adapter": str(adapter_path),
        "device": device,
        "optimizer_steps": 1,
        "reward_scheme": group.reward_scheme,
        "return_mean": sum(group.rewards) / len(group.rewards),
        **metric,
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
    return summary


def math_is_finite(value: float) -> bool:
    """判断一个训练指标是否为有限数值。

    Args:
        value (float): 待检查指标。

    Returns:
        bool: 指标有限时为真。
    """
    import math

    return math.isfinite(value)
