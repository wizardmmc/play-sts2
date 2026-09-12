"""提供单卡战斗 GRPO、独立 DAgger loss 与精确恢复训练器。"""

from __future__ import annotations

import json
import math
import random
import re
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from play_sts2.training.sft import (
    encode_messages,
    publish_adapter,
    resolve_device,
    save_last_checkpoint,
)

from .contracts import RL_GAME_VERSION
from .dagger import load_dagger_sft_rows
from .learner import (
    GrpoSequence,
    GrpoTrainingError,
    _grpo_loss_tensors,
    compare_battle_reward_schemes,
    constrained_completion_logprobs,
    load_grpo_training_group,
    stratified_importance_weight,
)
from .reward import BattleRewardScheme


@dataclass(frozen=True, slots=True)
class BattleGrpoConfig:
    """保存一次战斗 GRPO 工程训练的稳定配置。

    Args:
        base_model (Path): 本地 Hugging Face 基座模型。
        init_adapter (Path): 冻结 rollout policy 对应的父 LoRA adapter。
        policy_model (str): rollout group 必须绑定的精确 policy 名。
        run_role (str): ``engineering_smoke`` 或 ``formal``。
        rollout_root (Path): 单个 group JSON 或递归 group 目录。
        dagger_root (Path | None): 可选的 CombatSolver DAgger 标签源。
        adapter_root (Path): 最终 adapter 父目录。
        runs_root (Path): checkpoint 与指标父目录。
        device (str): 单个 PyTorch 训练设备。
        learning_rate (float): AdamW 学习率。
        max_length (int): 单步 stateless 对话最大长度。
        max_grad_norm (float): 梯度裁剪上限。
        logits_chunk_size (int): completion 词表投影分块大小。
        checkpoint_groups (int): 每隔多少 group 保存精确 checkpoint。
        seed (int): 随机种子。
        clip (float): PPO 对称裁剪半径。
        kl_beta (float): 冻结父 adapter sampled-KL 权重。
        dagger_weight (float): 独立 DAgger 监督 loss 权重。
        dagger_samples_per_group (int): 每个优化步加入的 DAgger 样本数。
        reward_scheme (BattleRewardScheme): 当前奖励消融方案。
        potion_cost (float): 固定药水成本方案的单瓶成本。
        loss_normalization (str): per_arm 保留历史逐轨迹平均；group_tokens 使用组内共同分母。
    """

    base_model: Path
    init_adapter: Path
    policy_model: str
    run_role: str
    rollout_root: Path
    dagger_root: Path | None
    adapter_root: Path
    runs_root: Path
    device: str
    learning_rate: float
    max_length: int
    max_grad_norm: float
    logits_chunk_size: int
    checkpoint_groups: int
    seed: int
    clip: float
    kl_beta: float
    dagger_weight: float
    dagger_samples_per_group: int
    reward_scheme: BattleRewardScheme
    potion_cost: float
    loss_normalization: str = "per_arm"


@dataclass(frozen=True, slots=True)
class BattleGrpoCheckpoint:
    """保存 group 边界精确恢复需要的动态训练状态。

    Args:
        next_group_index (int): 下一次处理的有序 group 下标。
        optimizer_steps (int): 已完成的优化器更新数。
        dagger_cursor (int): 下一批 DAgger 样本起点。
        torch_rng_state (Any): PyTorch CPU RNG 状态。
        device_rng_state (Any | None): 当前 CUDA/MPS RNG 状态。
        optimizer_state (dict[str, Any]): AdamW 完整状态。
        anchor_state (dict[str, Any]): 冻结父 adapter 的可训练参数副本。
    """

    next_group_index: int
    optimizer_steps: int
    dagger_cursor: int
    torch_rng_state: Any
    device_rng_state: Any | None
    optimizer_state: dict[str, Any]
    anchor_state: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        """转换为可由 ``torch.save`` 保存的字典。

        Returns:
            dict[str, Any]: 带格式标签的完整 checkpoint。
        """
        return {
            "format": "battle_grpo_checkpoint",
            "next_group_index": self.next_group_index,
            "optimizer_steps": self.optimizer_steps,
            "dagger_cursor": self.dagger_cursor,
            "torch_rng_state": self.torch_rng_state,
            "device_rng_state": self.device_rng_state,
            "optimizer_state": self.optimizer_state,
            "anchor_state": self.anchor_state,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> BattleGrpoCheckpoint:
        """校验并恢复 checkpoint 字典。

        Args:
            value (Mapping[str, Any]): ``torch.load`` 读取的状态对象。

        Raises:
            GrpoTrainingError: 格式或游标字段无效。

        Returns:
            BattleGrpoCheckpoint: 已类型化的精确恢复状态。
        """
        try:
            checkpoint = cls(
                next_group_index=int(value["next_group_index"]),
                optimizer_steps=int(value["optimizer_steps"]),
                dagger_cursor=int(value["dagger_cursor"]),
                torch_rng_state=value["torch_rng_state"],
                device_rng_state=value.get("device_rng_state"),
                optimizer_state=dict(value["optimizer_state"]),
                anchor_state=dict(value["anchor_state"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GrpoTrainingError("战斗 GRPO checkpoint 字段无效") from exc
        if (
            value.get("format") != "battle_grpo_checkpoint"
            or checkpoint.next_group_index < 0
            or checkpoint.optimizer_steps < 0
            or checkpoint.dagger_cursor < 0
            or not checkpoint.anchor_state
        ):
            raise GrpoTrainingError("战斗 GRPO checkpoint 格式或游标无效")
        return checkpoint


class FrozenAnchor:
    """保存并临时应用训练开始时的冻结父 adapter 参数。"""

    def __init__(
        self,
        model: Any,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        """复制当前可训练参数或载入 checkpoint 中的冻结锚。

        Args:
            model (Any): 含可训练 LoRA 参数的模型。
            state (Mapping[str, Any] | None): 可选的已保存锚状态。

        Raises:
            GrpoTrainingError: 模型没有可训练参数或锚键不一致。
        """
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not parameters:
            raise GrpoTrainingError("冻结锚模型没有可训练 adapter 参数")
        if state is not None and set(state) != set(parameters):
            raise GrpoTrainingError("checkpoint 冻结锚与当前 adapter 参数不一致")
        self._parameters = parameters
        self._state = {
            name: (state[name] if state is not None else parameter.detach())
            .clone()
            .to(device=parameter.device, dtype=parameter.dtype)
            for name, parameter in parameters.items()
        }

    @contextmanager
    def applied(self) -> Iterator[None]:
        """临时切到冻结锚并在退出时恢复当前 learner 参数。

        Yields:
            None: 上下文内部模型使用冻结锚参数。
        """
        current = {
            name: parameter.detach().clone()
            for name, parameter in self._parameters.items()
        }
        try:
            with _no_grad():
                for name, parameter in self._parameters.items():
                    parameter.copy_(self._state[name])
            yield
        finally:
            with _no_grad():
                for name, parameter in self._parameters.items():
                    parameter.copy_(current[name])

    def state_dict(self) -> dict[str, Any]:
        """返回可安全写入 checkpoint 的 CPU 锚参数。

        Returns:
            dict[str, Any]: 参数名到独立 CPU tensor 的映射。
        """
        return {
            name: tensor.detach().cpu().clone() for name, tensor in self._state.items()
        }

    def restore(self) -> None:
        """把当前可训练参数永久恢复为训练开始时的父 adapter。

        Returns:
            None: 所有可训练参数恢复完成。
        """
        with _no_grad():
            for name, parameter in self._parameters.items():
                parameter.copy_(self._state[name])


def validate_training_policy_identity(config: Any, *, check_base: bool = False) -> None:
    """在正式更新前拒绝把旧 adapter 路径标成新 policy。

    Args:
        config (Any): 含 run_role、init_adapter 和 policy_model 的训练配置。
        check_base (bool): 在模型加载入口检查 base 的原生 SFT 合并收据。

    Raises:
        GrpoTrainingError: 正式初始化错配，或 base 没有 SFT 合并来源。
    """
    if config.run_role == "formal" and config.init_adapter.name != config.policy_model:
        raise GrpoTrainingError(
            f"init_adapter 与 policy_model 不一致: {config.init_adapter} != {config.policy_model}"
        )
    if check_base and config.run_role == "formal":
        try:
            manifest = json.loads(
                (config.base_model / "merge_manifest.json").read_text()
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise GrpoTrainingError("正式 RL base 缺少有效 SFT 合并收据") from exc
        if not isinstance(manifest, dict) or not all(
            isinstance(manifest.get(key), str) and manifest[key]
            for key in ("adapter", "merge")
        ):
            raise GrpoTrainingError("正式 RL base 的 SFT 合并来源不完整")


def validate_initial_policy_ratio(metrics: Mapping[str, float]) -> None:
    """在首个 optimizer.step 前拒绝明显不匹配的真实行为分布。

    此检查用于发现明显错配，不能替代实际权重路径和 serving 收据校验。

    Args:
        metrics (Mapping[str, float]): 当前未更新 learner 在首组上的指标。

    Raises:
        GrpoTrainingError: 初始 ratio 非有限或超出 BF16 等价容差。
    """
    ratio = metrics["ratio_mean"]
    if not math.isfinite(ratio) or not 0.95 <= ratio <= 1.05:
        raise GrpoTrainingError(f"初始 policy ratio 错配，未执行更新: {ratio}")


def load_battle_grpo_config(path: Path) -> BattleGrpoConfig:
    """从 TOML 加载战斗 GRPO 配置。

    Args:
        path (Path): 配置文件路径。

    Raises:
        GrpoTrainingError: 字段缺失、类型或数值无效。
        OSError: 配置文件无法读取。
        tomllib.TOMLDecodeError: TOML 语法无效。

    Returns:
        BattleGrpoConfig: 已验证的训练配置。
    """
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    try:
        raw_dagger = data.get("dagger_root")
        scheme = str(data["reward_scheme"])
        if scheme not in {
            "core",
            "core_no_turn",
            "core_no_turn_boss_progress",
            "potion_cost",
            "legacy_remaining_potion",
        }:
            raise ValueError("reward_scheme 无效")
        config = BattleGrpoConfig(
            base_model=Path(data["base_model"]),
            init_adapter=Path(data["init_adapter"]),
            policy_model=str(data["policy_model"]),
            run_role=str(data["run_role"]),
            rollout_root=Path(data["rollout_root"]),
            dagger_root=Path(raw_dagger) if raw_dagger else None,
            adapter_root=Path(data["adapter_root"]),
            runs_root=Path(data["runs_root"]),
            device=str(data["device"]),
            learning_rate=float(data["learning_rate"]),
            max_length=int(data["max_length"]),
            max_grad_norm=float(data.get("max_grad_norm", 1.0)),
            logits_chunk_size=int(data.get("logits_chunk_size", 128)),
            checkpoint_groups=int(data.get("checkpoint_groups", 50)),
            seed=int(data["seed"]),
            clip=float(data.get("clip", 0.2)),
            kl_beta=float(data.get("kl_beta", 0.02)),
            dagger_weight=float(data.get("dagger_weight", 0.0)),
            dagger_samples_per_group=int(data.get("dagger_samples_per_group", 1)),
            reward_scheme=scheme,  # type: ignore[arg-type]
            potion_cost=float(data.get("potion_cost", 0.25)),
            loss_normalization=str(data.get("loss_normalization", "per_arm")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GrpoTrainingError(f"无效战斗 GRPO 配置: {exc}") from exc
    positive = {
        "learning_rate": config.learning_rate,
        "max_length": config.max_length,
        "max_grad_norm": config.max_grad_norm,
        "logits_chunk_size": config.logits_chunk_size,
        "checkpoint_groups": config.checkpoint_groups,
        "potion_cost": config.potion_cost,
        "dagger_samples_per_group": config.dagger_samples_per_group,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    cuda_device = config.device == "cuda" or re.fullmatch(r"cuda:\d+", config.device)
    if (
        invalid
        or config.device not in {"auto", "cpu", "mps"}
        and cuda_device is None
        or not 0 <= config.clip < 1
        or config.kl_beta < 0
        or config.dagger_weight < 0
        or config.dagger_weight > 0
        and config.dagger_root is None
        or not config.policy_model.strip()
        or config.run_role not in {"engineering_smoke", "formal"}
        or config.loss_normalization not in {"per_arm", "group_tokens"}
        or config.run_role == "formal"
        and config.checkpoint_groups < 20
    ):
        detail = (
            ", ".join(invalid)
            if invalid
            else "checkpoint_groups"
            if config.run_role == "formal" and config.checkpoint_groups < 20
            else "组合字段"
        )
        raise GrpoTrainingError(f"战斗 GRPO 配置值无效: {detail}")
    validate_training_policy_identity(config)
    return config


def train_battle_grpo(
    config: BattleGrpoConfig,
    run_name: str,
    *,
    max_groups: int | None = None,
    exact_resume: bool = False,
) -> dict[str, Any]:
    """执行一轮单卡战斗 GRPO 工程训练或精确续训。

    一个完整 rollout group 对应一个 optimizer step。八条学生 arm 的组内相对优势
    只进入 GRPO；DAgger 标签按独立 teacher-forced CE 加入同一步，绝不参与奖励
    归一化。``max_groups`` 只限制本次调用新增的 group 数，提前停止时保留可恢复
    checkpoint 而不发布 adapter。

    Args:
        config (BattleGrpoConfig): 已验证的训练配置。
        run_name (str): adapter 与运行目录名称。
        max_groups (int | None): 本次调用最多更新的 group 数。
        exact_resume (bool): 是否从同名 ``checkpoint-last`` 精确恢复。

    Raises:
        GrpoTrainingError: 输入、policy 血缘、恢复状态或数值不符合契约。
        OSError: 模型、rollout、checkpoint 或指标无法读写。

    Returns:
        dict[str, Any]: 可序列化的完成或暂停摘要。
    """
    validate_training_policy_identity(config, check_base=True)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise GrpoTrainingError(
            "战斗 GRPO run_name 只能包含字母、数字、点、横线和下划线"
        )
    if max_groups is not None and max_groups <= 0:
        raise GrpoTrainingError("max_groups 必须为正数")
    if config.run_role == "formal" and max_groups is not None:
        raise GrpoTrainingError("formal 训练不能使用 max_groups 提前发布 smoke")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rollout_paths = _rollout_paths(config.rollout_root)
    adapter_path = config.adapter_root / run_name
    run_path = config.runs_root / run_name
    checkpoint_path = run_path / "checkpoint-last"
    if exact_resume:
        if adapter_path.exists() or not checkpoint_path.is_dir():
            raise GrpoTrainingError(
                "精确续训要求未发布 adapter 且 checkpoint-last 存在"
            )
        _validate_resume_inputs(config, run_path, rollout_paths)
    elif adapter_path.exists() or run_path.exists():
        raise GrpoTrainingError(f"战斗 GRPO 输出已存在: {adapter_path} 或 {run_path}")

    device = resolve_device(config.device)
    dtype = grpo_base_model_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.base_model),
        local_files_only=True,
        trust_remote_code=False,
    )
    groups = tuple(
        load_grpo_training_group(
            path,
            tokenizer,
            reward_scheme=config.reward_scheme,
            max_length=config.max_length,
            potion_cost=config.potion_cost,
            allow_zero_variance=config.dagger_weight > 0,
        )
        for path in rollout_paths
    )
    if any(group.policy_version != config.policy_model for group in groups):
        raise GrpoTrainingError("rollout group 与配置的冻结 policy_model 不一致")
    dagger_samples = _load_dagger_samples(config, tokenizer)

    base_model = AutoModelForCausalLM.from_pretrained(
        str(config.base_model),
        dtype=dtype,
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
    from peft import PeftModel

    resume_state = None
    adapter_source = checkpoint_path if exact_resume else config.init_adapter
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_source),
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
        raise GrpoTrainingError("战斗 GRPO 模型没有可训练 adapter 参数")
    optimizer = __import__("torch").optim.AdamW(parameters, lr=config.learning_rate)
    if exact_resume:
        resume_state = _load_battle_checkpoint(checkpoint_path, device)
        if resume_state.next_group_index > len(groups):
            raise GrpoTrainingError("checkpoint group 游标超出当前输入")
        optimizer.load_state_dict(resume_state.optimizer_state)
        _restore_rng_state(device, resume_state)
        anchor = FrozenAnchor(model, resume_state.anchor_state)
        next_group_index = resume_state.next_group_index
        optimizer_steps = resume_state.optimizer_steps
        dagger_cursor = resume_state.dagger_cursor
        _prepare_trace(run_path / "metrics.jsonl", optimizer_steps)
    else:
        import torch

        torch.manual_seed(config.seed)
        random.seed(config.seed)
        anchor = FrozenAnchor(model)
        next_group_index = 0
        optimizer_steps = 0
        dagger_cursor = 0
        run_path.mkdir(parents=True)
        _write_json(run_path / "config.json", _config_json(config))
        _write_json(
            run_path / "inputs.json",
            {
                "policy_model": config.policy_model,
                "rollout_paths": [str(path) for path in rollout_paths],
            },
        )

    trace_path = run_path / "metrics.jsonl"
    trace_mode = "a" if exact_resume else "w"
    groups_this_call = 0
    from .orchestration.telemetry import TensorboardMetricsWriter

    purge_step = optimizer_steps + 1 if exact_resume else None
    with (
        trace_path.open(trace_mode, encoding="utf-8") as trace,
        TensorboardMetricsWriter(
            run_path / "tensorboard",
            purge_step=purge_step,
        ) as tensorboard,
    ):
        while next_group_index < len(groups):
            if max_groups is not None and groups_this_call >= max_groups:
                break
            group = groups[next_group_index]
            metric, dagger_cursor = optimize_grpo_group(
                model,
                optimizer,
                anchor,
                group.arms,
                dagger_samples,
                dagger_cursor=dagger_cursor,
                config=config,
                device=device,
                verify_initial_policy=config.run_role == "formal"
                and next_group_index == 0,
            )
            optimizer_steps += 1
            next_group_index += 1
            groups_this_call += 1
            record = {
                "step": optimizer_steps,
                "group_index": next_group_index - 1,
                "group_id": group.group_id,
                "reward_scheme": group.reward_scheme,
                "reward_mean": sum(group.rewards) / len(group.rewards),
                **metric,
            }
            _record_battle_training_metrics(trace, tensorboard, record)
            pausing = (
                max_groups is not None
                and groups_this_call >= max_groups
                and next_group_index < len(groups)
            )
            if _checkpoint_due(
                next_group_index,
                total_groups=len(groups),
                interval=config.checkpoint_groups,
                pausing=pausing,
            ):
                save_last_checkpoint(
                    model,
                    checkpoint_path,
                    _checkpoint_state(
                        optimizer,
                        anchor,
                        device=device,
                        next_group_index=next_group_index,
                        optimizer_steps=optimizer_steps,
                        dagger_cursor=dagger_cursor,
                    ).to_payload(),
                )

    completed = next_group_index == len(groups)
    post_update_metric = None
    if completed and config.run_role == "formal":
        post_update_metric = verify_final_battle_update(
            model,
            anchor,
            groups[-1],
            config=config,
            device=device,
        )
    summary = {
        "run_name": run_name,
        "status": "completed" if completed else "paused",
        "run_role": config.run_role,
        "engineering_smoke": config.run_role == "engineering_smoke",
        "base_model": str(config.base_model),
        "init_adapter": str(config.init_adapter),
        "policy_model": config.policy_model,
        "adapter": str(adapter_path) if completed else None,
        "device": device,
        "reward_scheme": config.reward_scheme,
        "groups_total": len(groups),
        "groups_completed": next_group_index,
        "optimizer_steps": optimizer_steps,
        "loss_normalization": config.loss_normalization,
        "post_update_metric": post_update_metric,
        "post_update_group_id": groups[-1].group_id
        if post_update_metric is not None
        else None,
        "dagger_samples": len(dagger_samples),
        "dagger_weight": config.dagger_weight,
        "exact_resume": exact_resume,
        "resumed_from_group": (
            resume_state.next_group_index if resume_state is not None else None
        ),
    }
    if completed:
        publish_adapter(model, tokenizer, adapter_path, summary)
    _write_json(run_path / "summary.json", summary)
    return summary


def verify_final_battle_update(
    model: Any,
    anchor: FrozenAnchor,
    group: Any,
    *,
    config: Any,
    device: str,
) -> dict[str, float]:
    """在发布前对最后一个完整组重算 KL/ratio，越界时恢复父权重。

    这是最后一步后的局部门禁，不能冒充全数据或未训练场景回归。

    Args:
        model (Any): 已执行本轮全部更新的候选模型。
        anchor (FrozenAnchor): 本轮冻结父 residual。
        group (Any): 本轮最后一个完整八臂组。
        config (Any): 同一训练配置。
        device (str): 模型所在设备。

    Raises:
        GrpoTrainingError: 候选在最后一组上的更新后漂移越界。

    Returns:
        dict[str, float]: 真正更新后的 KL 与 ratio 指标。
    """
    metrics = evaluate_grpo_group(
        model, anchor, group.arms, config=config, device=device
    )
    kl = metrics.get("kl", math.inf)
    ratio = metrics.get("ratio_mean", math.inf)
    if not math.isfinite(kl) or kl > 0.05 or not 0.5 <= ratio <= 2.0:
        anchor.restore()
        raise GrpoTrainingError(
            f"战斗 post-step KL/ratio 越界，已恢复父权重: {metrics}"
        )
    return metrics


def _checkpoint_due(
    next_group_index: int,
    *,
    total_groups: int,
    interval: int,
    pausing: bool,
) -> bool:
    """判断当前完整 group 边界是否必须发布 checkpoint。

    Args:
        next_group_index (int): 下一 group 游标。
        total_groups (int): 本轮全部 group 数。
        interval (int): 周期 checkpoint 间隔。
        pausing (bool): 本次调用是否会在当前边界主动暂停。

    Returns:
        bool: 非终局周期命中或主动暂停时为真；正常完成由最终 adapter 承担。
    """
    return pausing or (
        next_group_index < total_groups and next_group_index % interval == 0
    )


def _record_battle_training_metrics(
    trace: Any,
    tensorboard: Any,
    record: Mapping[str, Any],
) -> None:
    """把同一 optimizer step 指标写入 JSONL 与 TensorBoard。

    Args:
        trace (Any): 当前 ``metrics.jsonl`` 文本 writer。
        tensorboard (Any): ``TensorboardMetricsWriter`` 或测试替身。
        record (Mapping[str, Any]): 带非负 ``step`` 的完整训练指标。

    Raises:
        GrpoTrainingError: step 不是非负整数。

    Returns:
        None: JSONL 已刷新，标量已加入 TensorBoard 队列。
    """
    step = record.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise GrpoTrainingError("战斗训练指标缺少有效 optimizer step")
    trace.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    trace.flush()
    tensorboard.write({"battle": dict(record)}, step=step)


def write_battle_reward_comparison(
    config: BattleGrpoConfig,
    output_path: Path,
    *,
    rollout_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """重算固定奖励消融并写出 arm 级审计报告。

    Args:
        config (BattleGrpoConfig): 提供 tokenizer、rollout 与最大长度的训练配置。
        output_path (Path): 报告 JSON 路径。
        rollout_paths (Sequence[Path] | None): 可选的显式 group 文件列表。

    Returns:
        dict[str, Any]: 已写入的奖励比较报告。
    """
    report = compare_battle_reward_schemes(
        tuple(rollout_paths)
        if rollout_paths is not None
        else _rollout_paths(config.rollout_root),
    )
    report["run_role"] = config.run_role
    report["engineering_smoke"] = config.run_role == "engineering_smoke"
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, report)
    return report


def optimize_grpo_group(
    model: Any,
    optimizer: Any,
    anchor: FrozenAnchor,
    arms: Sequence[Any],
    dagger_samples: Sequence[Any],
    *,
    dagger_cursor: int,
    config: BattleGrpoConfig,
    device: str,
    verify_initial_policy: bool = False,
) -> tuple[dict[str, float], int]:
    """对一个八臂 group 及独立 DAgger 小批执行一次更新。

    Args:
        model (Any): 当前可训练 policy。
        optimizer (Any): AdamW 优化器。
        anchor (FrozenAnchor): 冻结父 adapter。
        arms (Sequence[Any]): 当前 group 的完整学生 arms。
        dagger_samples (Sequence[Any]): 已编码 DAgger 样本池。
        dagger_cursor (int): 下一批 DAgger 起点。
        config (BattleGrpoConfig): 损失与裁剪配置。
        device (str): 单个训练设备。
        verify_initial_policy (bool): 正式首组在 optimizer.step 前核对行为分布。

    Raises:
        GrpoTrainingError: group、loss 或梯度出现无效数值。

    Returns:
        tuple[dict[str, float], int]: 本步指标与更新后的 DAgger 游标。
    """
    import torch

    if len(arms) != 8:
        raise GrpoTrainingError("战斗 GRPO learner 只接受严格八臂 group")
    optimizer.zero_grad(set_to_none=True)
    totals = backward_grpo_group(
        model,
        anchor,
        arms,
        config=config,
        device=device,
    )
    if verify_initial_policy:
        validate_initial_policy_ratio(totals)

    dagger_loss_value = 0.0
    if config.dagger_weight > 0:
        selected, dagger_cursor = _next_dagger_batch(
            dagger_samples,
            dagger_cursor,
            config.dagger_samples_per_group,
        )
        for sample in selected:
            logprobs = _sample_logprobs(
                model,
                sample,
                device=device,
                logits_chunk_size=config.logits_chunk_size,
            )
            loss = -logprobs.mean()
            (loss * config.dagger_weight / len(selected)).backward()
            dagger_loss_value += float(loss.detach().cpu()) / len(selected)
    invalid_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and not torch.isfinite(parameter.grad).all().item()
    ]
    if invalid_gradients:
        raise GrpoTrainingError(f"战斗 GRPO 梯度不是有限数值: {invalid_gradients[0]}")
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        config.max_grad_norm,
    )
    optimizer.step()
    return (
        {
            **totals,
            "dagger_loss": dagger_loss_value,
            "grad_norm": float(grad_norm.detach().cpu()),
        },
        dagger_cursor,
    )


def compute_grpo_anchor_values(
    model: Any,
    anchor: FrozenAnchor,
    arms: Sequence[Any],
    *,
    config: Any,
    device: str,
) -> list[list[Any]]:
    """在冻结锚权重上一次算清整组逐序列 log-prob。

    多 epoch 更新复用同一批锚值；锚值只依赖冻结权重，与当前 learner 无关。

    Args:
        model (Any): 当前可训练 policy（锚值计算时临时切到冻结锚）。
        anchor (FrozenAnchor): 冻结父 adapter。
        arms (Sequence[Any]): 同组训练 arms。
        config (Any): 提供 logits 分块大小的配置。
        device (str): 单个训练设备。

    Raises:
        GrpoTrainingError: arm 没有可训练 token。

    Returns:
        list[list[Any]]: 每个 arm 内逐序列的锚 log-prob 张量。
    """
    import torch

    values: list[list[Any]] = []
    for arm in arms:
        if arm.supervised_tokens <= 0:
            raise GrpoTrainingError("GRPO arm 没有可训练 token")
        arm_anchors = []
        for sequence in arm.sequences:
            with anchor.applied(), torch.no_grad():
                arm_anchors.append(
                    _sequence_logprobs(
                        model,
                        sequence,
                        device=device,
                        logits_chunk_size=config.logits_chunk_size,
                    )
                )
        values.append(arm_anchors)
    return values


def backward_grpo_group(
    model: Any,
    anchor: FrozenAnchor,
    arms: Sequence[Any],
    *,
    config: Any,
    device: str,
    loss_scale: float = 1.0,
    precomputed_anchor_values: Sequence[Sequence[Any]] | None = None,
) -> dict[str, float]:
    """累积一个 GRPO/Tree/GiGPO group 的梯度但不更新优化器。

    普通 on-policy arm 的权重为一。分层 terminal Tree 的强制根动作先在冻结
    父策略和完整 xgrammar 支持集上重算 ``pi_old``，PPO ratio 也以该概率为
    behavior；整个入口计划的 policy loss 再乘 ``pi_old / q``。返回指标不乘
    ``loss_scale``，便于比较不同数据源本身的数值。

    Args:
        model (Any): 当前可训练 policy。
        anchor (FrozenAnchor): 冻结父 adapter。
        arms (Sequence[Any]): 二条以上同组训练 arms。
        config (Any): 提供 logits 分块、clip 与 KL 权重的配置。
        device (str): 单个训练设备。
        loss_scale (float): 当前数据源在组合战略 loss 中的固定系数。
        precomputed_anchor_values (Sequence[Sequence[Any]] | None): 可选的
            ``compute_grpo_anchor_values`` 结果；多 epoch 复用时不再重算。

    Raises:
        GrpoTrainingError: arm、重要性权重或 loss 数值无效。

    Returns:
        dict[str, float]: policy loss、KL、ratio 与重要性权重摘要。
    """
    import torch

    if len(arms) < 2:
        raise GrpoTrainingError("相对策略 group 至少需要两条 arms")
    if not 0 < loss_scale <= 1:
        raise GrpoTrainingError("组合 GRPO loss_scale 必须位于 (0, 1]")
    anchor_values: list[list[Any]] = []
    importance_weights: list[float] = []
    for arm_index, arm in enumerate(arms):
        if arm.supervised_tokens <= 0:
            raise GrpoTrainingError("GRPO arm 没有可训练 token")
        if precomputed_anchor_values is not None:
            arm_anchors = list(precomputed_anchor_values[arm_index])
        else:
            arm_anchors = []
            for sequence in arm.sequences:
                with anchor.applied(), torch.no_grad():
                    arm_anchors.append(
                        _sequence_logprobs(
                            model,
                            sequence,
                            device=device,
                            logits_chunk_size=config.logits_chunk_size,
                        )
                    )
        anchor_values.append(arm_anchors)
        if arm.recompute_root_probability:
            try:
                importance = stratified_importance_weight(
                    arm_anchors[0],
                    proposal_probability=arm.proposal_probability,
                )
            except ValueError as exc:
                raise GrpoTrainingError("Tree importance correction 无效") from exc
        else:
            importance = 1.0
        importance_weights.append(importance)
    importance_total = sum(importance_weights)
    if not math.isfinite(importance_total) or importance_total <= 0:
        raise GrpoTrainingError("GRPO importance weight 总和无效")

    totals = {"policy_loss": 0.0, "kl": 0.0, "ratio_mean": 0.0}
    metric_weight_total = 0.0
    estimator_weights = _importance_estimator_weights(importance_weights)
    denominators = _loss_normalization_denominators(arms, config, estimator_weights)
    for arm_index, arm in enumerate(arms):
        arm_weight = estimator_weights[arm_index]
        for sequence_index, sequence in enumerate(arm.sequences):
            anchor_logprobs = anchor_values[arm_index][sequence_index]
            new_logprobs = _sequence_logprobs(
                model,
                sequence,
                device=device,
                logits_chunk_size=config.logits_chunk_size,
            )
            if arm.recompute_root_probability and sequence_index == 0:
                behavior = anchor_logprobs
            else:
                behavior = torch.tensor(
                    sequence.behavior_logprobs,
                    dtype=new_logprobs.dtype,
                    device=device,
                )
            total, policy, kl, ratio = _grpo_loss_tensors(
                new_logprobs,
                behavior,
                anchor_logprobs,
                advantage=sequence.advantage,
                clip=config.clip,
                kl_beta=config.kl_beta,
            )
            token_count = len(sequence.behavior_logprobs)
            weight = arm_weight * token_count / denominators[arm_index]
            (total * weight * loss_scale).backward()
            totals["policy_loss"] += float(policy.detach().cpu()) * weight
            totals["kl"] += float(kl.detach().cpu()) * weight
            totals["ratio_mean"] += float(ratio.detach().cpu()) * weight
            metric_weight_total += weight
    if metric_weight_total <= 0:
        raise GrpoTrainingError("GRPO group 没有有效 loss 权重")
    return {
        "policy_loss": totals["policy_loss"] / metric_weight_total,
        "kl": totals["kl"] / metric_weight_total,
        "ratio_mean": totals["ratio_mean"] / metric_weight_total,
        "importance_mean": sum(importance_weights) / len(importance_weights),
        "importance_max": max(importance_weights),
    }


def evaluate_grpo_group(
    model: Any,
    anchor: FrozenAnchor,
    arms: Sequence[Any],
    *,
    config: Any,
    device: str,
) -> dict[str, float]:
    """在不反向传播时复算更新后 group 的 ratio 与父策略 KL。

    Args:
        model (Any): 已完成候选更新的 policy。
        anchor (FrozenAnchor): 同一步冻结父 adapter。
        arms (Sequence[Any]): 当前 GiGPO 或 Tree arms。
        config (Any): 提供 logits 分块、clip 与 KL 配置。
        device (str): 单个训练设备。

    Raises:
        GrpoTrainingError: group 或重要性权重无效。

    Returns:
        dict[str, float]: 候选参数上的 post-step KL、ratio 与权重摘要。
    """
    import torch

    if len(arms) < 2:
        raise GrpoTrainingError("相对策略 group 至少需要两条 arms")
    anchor_values: list[list[Any]] = []
    importance_weights = []
    for arm in arms:
        if arm.supervised_tokens <= 0:
            raise GrpoTrainingError("GRPO arm 没有可训练 token")
        arm_anchors = []
        for sequence in arm.sequences:
            with anchor.applied(), torch.no_grad():
                arm_anchors.append(
                    _sequence_logprobs(
                        model,
                        sequence,
                        device=device,
                        logits_chunk_size=config.logits_chunk_size,
                    )
                )
        anchor_values.append(arm_anchors)
        if arm.recompute_root_probability:
            importance = stratified_importance_weight(
                arm_anchors[0],
                proposal_probability=arm.proposal_probability,
            )
        else:
            importance = 1.0
        importance_weights.append(importance)
    estimator_weights = _importance_estimator_weights(importance_weights)
    denominators = _loss_normalization_denominators(arms, config, estimator_weights)
    totals = {"kl": 0.0, "ratio_mean": 0.0}
    metric_weight_total = 0.0
    with torch.no_grad():
        for arm_index, arm in enumerate(arms):
            for sequence_index, sequence in enumerate(arm.sequences):
                anchor_logprobs = anchor_values[arm_index][sequence_index]
                new_logprobs = _sequence_logprobs(
                    model,
                    sequence,
                    device=device,
                    logits_chunk_size=config.logits_chunk_size,
                )
                if arm.recompute_root_probability and sequence_index == 0:
                    behavior = anchor_logprobs
                else:
                    behavior = torch.tensor(
                        sequence.behavior_logprobs,
                        dtype=new_logprobs.dtype,
                        device=device,
                    )
                _total, _policy, kl, ratio = _grpo_loss_tensors(
                    new_logprobs,
                    behavior,
                    anchor_logprobs,
                    advantage=sequence.advantage,
                    clip=config.clip,
                    kl_beta=config.kl_beta,
                )
                token_count = len(sequence.behavior_logprobs)
                weight = (
                    estimator_weights[arm_index] * token_count / denominators[arm_index]
                )
                totals["kl"] += float(kl.detach().cpu()) * weight
                totals["ratio_mean"] += float(ratio.detach().cpu()) * weight
                metric_weight_total += weight
    if metric_weight_total <= 0:
        raise GrpoTrainingError("GRPO group 没有有效 post-step 指标权重")
    return {
        "kl": totals["kl"] / metric_weight_total,
        "ratio_mean": totals["ratio_mean"] / metric_weight_total,
        "importance_mean": sum(importance_weights) / len(importance_weights),
        "importance_max": max(importance_weights),
    }


def _loss_normalization_denominators(
    arms: Sequence[Any], config: Any, estimator_weights: Sequence[float]
) -> tuple[float, ...]:
    """选择逐轨迹或全组共同 token 分母，保留原来的 importance estimator。

    Args:
        arms (Sequence[Any]): 同一冻结策略的完整轨迹。
        config (Any): 含可选 loss_normalization 的训练配置。
        estimator_weights (Sequence[float]): 已校验的 pi_old/q/K，保持原值不自归一化。

    Raises:
        GrpoTrainingError: 轨迹无监督 token 或归一化方式无效。

    Returns:
        tuple[float, ...]: 每条轨迹使用的固定分母；group_tokens 下全部相同。
    """
    lengths = tuple(float(arm.supervised_tokens) for arm in arms)
    if not lengths or any(length <= 0 for length in lengths):
        raise GrpoTrainingError("GRPO arm 没有可训练 token")
    mode = getattr(config, "loss_normalization", "per_arm")
    if mode == "per_arm":
        return lengths
    if mode == "group_tokens":
        denominator = sum(
            weight * length
            for weight, length in zip(estimator_weights, lengths, strict=True)
        ) / sum(estimator_weights)
        return (denominator,) * len(lengths)
    raise GrpoTrainingError(f"未知 loss_normalization: {mode}")


def _importance_estimator_weights(values: Sequence[float]) -> tuple[float, ...]:
    """返回无偏 ``1/K`` importance estimator 的 branch 权重。

    Args:
        values (Sequence[float]): 每条 branch 的 ``pi_old/q``；普通 arm 为一。

    Raises:
        GrpoTrainingError: 数组为空或包含非有限正数。

    Returns:
        tuple[float, ...]: 每项为原始 importance 除以固定采样数 K，不按当前
        样本权重和做 self-normalization。
    """
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise GrpoTrainingError("GRPO importance estimator 权重无效")
    return tuple(value / len(values) for value in values)


def _sequence_logprobs(
    model: Any,
    sequence: GrpoSequence,
    *,
    device: str,
    logits_chunk_size: int,
) -> Any:
    """计算一条 rollout 序列在同一 structured-choice 分布下的概率。

    Args:
        model (Any): 当前或已临时切锚的因果语言模型。
        sequence (GrpoSequence): stateless rollout 序列。
        device (str): 模型所在设备。
        logits_chunk_size (int): 每次词表投影的最大 completion token 数。

    Returns:
        Any: 与 behavior log-prob 等长的可微条件 log-prob。
    """
    import torch

    input_ids = torch.tensor([sequence.input_ids], dtype=torch.long, device=device)
    hidden, lm_head = _hidden_and_head(model, input_ids)
    completion_positions = torch.tensor(
        [index - 1 for index, keep in enumerate(sequence.assistant_mask) if keep],
        dtype=torch.long,
        device=device,
    )
    selected_hidden = hidden[0].index_select(0, completion_positions)
    logits = _project_logits(lm_head, selected_hidden, logits_chunk_size)
    completion_ids = tuple(
        token
        for token, keep in zip(sequence.input_ids, sequence.assistant_mask)
        if keep
    )
    return constrained_completion_logprobs(
        logits,
        completion_ids=completion_ids,
        allowed_token_ids=sequence.allowed_token_ids,
        temperature=sequence.temperature,
    )


def _sample_logprobs(
    model: Any,
    sample: Any,
    *,
    device: str,
    logits_chunk_size: int,
) -> Any:
    """计算一条 DAgger teacher-forced 样本的 assistant token 原始概率。

    Args:
        model (Any): 当前可训练因果语言模型。
        sample (Any): ``encode_messages`` 生成的 TokenizedSample。
        device (str): 模型所在设备。
        logits_chunk_size (int): 每次词表投影的最大 assistant token 数。

    Raises:
        GrpoTrainingError: 样本没有 assistant token。

    Returns:
        Any: 保持梯度的 assistant token log-prob。
    """
    import torch

    input_ids = torch.tensor([sample.input_ids], dtype=torch.long, device=device)
    hidden, lm_head = _hidden_and_head(model, input_ids)
    positions = [
        index - 1
        for index, label in enumerate(sample.labels)
        if label != -100 and index > 0
    ]
    targets = [
        label
        for index, label in enumerate(sample.labels)
        if label != -100 and index > 0
    ]
    if not positions:
        raise GrpoTrainingError("DAgger 样本没有 assistant token")
    selected_hidden = hidden[0].index_select(
        0,
        torch.tensor(positions, dtype=torch.long, device=device),
    )
    logits = _project_logits(lm_head, selected_hidden, logits_chunk_size).float()
    target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
    return (
        torch.log_softmax(logits, dim=-1).gather(1, target_tensor[:, None]).squeeze(1)
    )


def _project_logits(lm_head: Any, hidden: Any, chunk_size: int) -> Any:
    """按 assistant token 位置分块执行词表投影。

    Args:
        lm_head (Any): 模型词表投影层。
        hidden (Any): 选中的 assistant predictor hidden state。
        chunk_size (int): 单次最大位置数。

    Returns:
        Any: 按原顺序拼接的全词表 logits。
    """
    import torch

    return torch.cat(
        [
            lm_head(hidden[begin : begin + chunk_size])
            for begin in range(0, len(hidden), chunk_size)
        ],
        dim=0,
    )


def _hidden_and_head(model: Any, input_ids: Any) -> tuple[Any, Any]:
    """执行 decoder 并返回 hidden state 与词表投影层。

    Args:
        model (Any): PEFT 包装后的 Qwen 因果语言模型。
        input_ids (Any): 单条完整序列 token tensor。

    Raises:
        GrpoTrainingError: 模型没有可用 decoder 或 lm_head。

    Returns:
        tuple[Any, Any]: decoder hidden state 与 lm_head。
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = getattr(base, "model", None)
    lm_head = getattr(base, "lm_head", None)
    if decoder is None or lm_head is None:
        raise GrpoTrainingError("战斗 GRPO 需要可分离的 decoder 与 lm_head")
    hidden = decoder(input_ids=input_ids, attention_mask=None).last_hidden_state
    return hidden, lm_head


def _load_dagger_samples(config: BattleGrpoConfig, tokenizer: Any) -> tuple[Any, ...]:
    """读取并编码独立 DAgger 监督样本。

    Args:
        config (BattleGrpoConfig): 当前训练配置。
        tokenizer (Any): 基座 tokenizer。

    Returns:
        tuple[Any, ...]: 可为空的 assistant-only 样本。
    """
    if config.dagger_root is None:
        return ()
    rows = load_dagger_sft_rows(
        config.dagger_root,
        expected_policy_version=config.policy_model,
    )
    _validate_dagger_versions(rows)
    return tuple(
        encode_messages(
            tokenizer,
            row["messages"],
            sample_id=str(row["sample_id"]),
            source="dagger_label",
            max_length=config.max_length,
        )
        for row in rows
    )


def _validate_dagger_versions(rows: Sequence[Mapping[str, Any]]) -> None:
    """要求 DAgger 标签来自同一套固定 RL 运行时。

    Args:
        rows (Sequence[Mapping[str, Any]]): 已通过标签结构与 policy 校验的行。

    Raises:
        GrpoTrainingError: 游戏版本不是 v0.111.0，或运行时版本在批内混杂。
    """
    fields = ("game_version", "mod_version", "protocol_version", "harness_version")
    if any(row.get("game_version") != RL_GAME_VERSION for row in rows):
        raise GrpoTrainingError(f"DAgger 标签必须来自 {RL_GAME_VERSION}")
    for field in fields:
        values = {row.get(field) for row in rows}
        if len(values) != 1 or None in values:
            raise GrpoTrainingError(f"DAgger 标签混入不同 {field}")


def _next_dagger_batch(
    samples: Sequence[Any],
    cursor: int,
    batch_size: int,
) -> tuple[tuple[Any, ...], int]:
    """从稳定样本顺序中循环取得下一批 DAgger 样本。

    Args:
        samples (Sequence[Any]): 非空 DAgger 样本池。
        cursor (int): 下一样本游标。
        batch_size (int): 本组所需样本数。

    Raises:
        GrpoTrainingError: 样本池为空或游标无效。

    Returns:
        tuple[tuple[Any, ...], int]: 选中样本与新游标。
    """
    if not samples or cursor < 0:
        raise GrpoTrainingError("启用 DAgger loss 时样本池和游标必须有效")
    chosen = tuple(
        samples[(cursor + offset) % len(samples)] for offset in range(batch_size)
    )
    return chosen, (cursor + batch_size) % len(samples)


def _rollout_paths(root: Path) -> tuple[Path, ...]:
    """解析稳定排序的 rollout group JSON 路径。

    Args:
        root (Path): 单文件或递归目录。

    Raises:
        GrpoTrainingError: 路径不存在或没有 JSON。

    Returns:
        tuple[Path, ...]: 非空有序 group 路径。
    """
    source = Path(root)
    paths = tuple(sorted(source.rglob("*.json"))) if source.is_dir() else (source,)
    if not paths or any(not path.is_file() for path in paths):
        raise GrpoTrainingError(f"战斗 rollout 路径不可用: {source}")
    return paths


def _checkpoint_state(
    optimizer: Any,
    anchor: FrozenAnchor,
    *,
    device: str,
    next_group_index: int,
    optimizer_steps: int,
    dagger_cursor: int,
) -> BattleGrpoCheckpoint:
    """读取当前 group 边界的完整训练状态。

    Args:
        optimizer (Any): 当前 AdamW 优化器。
        anchor (FrozenAnchor): 冻结父 adapter。
        device (str): 当前训练设备。
        next_group_index (int): 下一 group 游标。
        optimizer_steps (int): 已完成更新数。
        dagger_cursor (int): 下一 DAgger 样本游标。

    Returns:
        BattleGrpoCheckpoint: 可原子保存的状态。
    """
    import torch

    return BattleGrpoCheckpoint(
        next_group_index=next_group_index,
        optimizer_steps=optimizer_steps,
        dagger_cursor=dagger_cursor,
        torch_rng_state=torch.get_rng_state(),
        device_rng_state=_device_rng_state(device),
        optimizer_state=optimizer.state_dict(),
        anchor_state=anchor.state_dict(),
    )


def _load_battle_checkpoint(path: Path, device: str) -> BattleGrpoCheckpoint:
    """从已发布 adapter checkpoint 加载精确训练状态。

    Args:
        path (Path): ``checkpoint-last`` 目录。
        device (str): 优化器 tensor 映射设备。

    Raises:
        GrpoTrainingError: 状态缺失或无法解析。

    Returns:
        BattleGrpoCheckpoint: 已验证的精确状态。
    """
    import torch

    state_path = Path(path) / "training_state.pt"
    if not state_path.is_file():
        raise GrpoTrainingError(f"战斗 GRPO checkpoint 状态不存在: {state_path}")
    try:
        value = torch.load(state_path, map_location=device, weights_only=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise GrpoTrainingError(f"无法读取战斗 GRPO checkpoint: {state_path}") from exc
    if not isinstance(value, Mapping):
        raise GrpoTrainingError("战斗 GRPO checkpoint 不是对象")
    return BattleGrpoCheckpoint.from_payload(value)


def _device_rng_state(device: str) -> Any | None:
    """读取当前加速设备 RNG 状态。

    Args:
        device (str): 当前设备。

    Returns:
        Any | None: CPU 外设备状态；CPU 返回空。
    """
    import torch

    if device == "mps":
        return torch.mps.get_rng_state()
    if device.startswith("cuda"):
        return torch.cuda.get_rng_state(device)
    return None


def _restore_rng_state(device: str, checkpoint: BattleGrpoCheckpoint) -> None:
    """恢复 CPU 与当前加速设备 RNG 状态。

    Args:
        device (str): 当前设备。
        checkpoint (BattleGrpoCheckpoint): 已加载的恢复状态。

    Raises:
        GrpoTrainingError: 加速设备 checkpoint 缺少 RNG 状态。
    """
    import torch

    torch.set_rng_state(checkpoint.torch_rng_state.cpu())
    if device == "cpu":
        return
    if checkpoint.device_rng_state is None:
        raise GrpoTrainingError("加速设备 checkpoint 缺少 RNG 状态")
    if device == "mps":
        torch.mps.set_rng_state(checkpoint.device_rng_state.cpu())
    elif device.startswith("cuda"):
        torch.cuda.set_rng_state(checkpoint.device_rng_state.cpu(), device)


def _prepare_trace(path: Path, optimizer_steps: int) -> None:
    """把续训指标轨迹回滚到 checkpoint 对应的完整步数。

    Args:
        path (Path): 指标 JSONL。
        optimizer_steps (int): checkpoint 已完成步数。

    Raises:
        GrpoTrainingError: 轨迹缺失或前缀步号无效。
    """
    if not path.is_file():
        raise GrpoTrainingError(f"精确续训缺少指标轨迹: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    retained = []
    for expected in range(1, optimizer_steps + 1):
        try:
            line = lines[expected - 1]
            actual = int(json.loads(line)["step"])
        except (
            IndexError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise GrpoTrainingError("战斗 GRPO 指标轨迹无效") from exc
        if actual != expected:
            raise GrpoTrainingError("战斗 GRPO 指标步号与 checkpoint 不一致")
        retained.append(line)
    path.write_text("".join(f"{line}\n" for line in retained), encoding="utf-8")


def _validate_resume_inputs(
    config: BattleGrpoConfig,
    run_path: Path,
    rollout_paths: Sequence[Path],
) -> None:
    """确认精确续训仍使用相同配置和有序 rollout 路径。

    Args:
        config (BattleGrpoConfig): 当前配置。
        run_path (Path): 既有运行目录。
        rollout_paths (Sequence[Path]): 当前解析的有序输入。

    Raises:
        GrpoTrainingError: 配置或输入清单发生变化。
    """
    saved_config = _read_json(run_path / "config.json")
    saved_config.setdefault("loss_normalization", "per_arm")
    if saved_config != _config_json(config):
        raise GrpoTrainingError("精确续训配置已变化")
    expected = {
        "policy_model": config.policy_model,
        "rollout_paths": [str(path) for path in rollout_paths],
    }
    if _read_json(run_path / "inputs.json") != expected:
        raise GrpoTrainingError("精确续训 rollout 输入已变化")


def _config_json(config: BattleGrpoConfig) -> dict[str, Any]:
    """把配置转换为 JSON 兼容字典。

    Args:
        config (BattleGrpoConfig): 当前训练配置。

    Returns:
        dict[str, Any]: Path 已转换为字符串的稳定配置。
    """
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def grpo_base_model_dtype(device: str) -> Any:
    """根据单卡设备选择基座模型精度。

    Args:
        device (str): 已解析的训练设备。

    Returns:
        Any: CUDA/MPS 使用 BF16，CPU 使用 FP32。
    """
    import torch

    return (
        torch.bfloat16
        if device == "mps" or device.startswith("cuda")
        else torch.float32
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """写入缩进 JSON 文件。

    Args:
        path (Path): 目标文件。
        value (Mapping[str, Any]): 可序列化对象。
    """
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _read_json(path: Path) -> Any:
    """读取一个 JSON 文件。

    Args:
        path (Path): 来源文件。

    Returns:
        Any: JSON 解码值。
    """
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def _no_grad() -> Iterator[None]:
    """延迟导入并进入 PyTorch no-grad 上下文。

    Yields:
        None: 上下文内不记录梯度。
    """
    import torch

    with torch.no_grad():
        yield
