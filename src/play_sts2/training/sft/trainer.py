"""提供内存受控的 Qwen LoRA 训练与 adapter 发布。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import validate_sft_dataset
from .encoding import (
    IGNORE_LABEL,
    SftConfig,
    SftTrainingError,
    TokenizedSample,
    config_as_json,
    load_tokenized_samples,
)
from .provenance import (
    model_source_files,
    sha256_files,
    snapshot_files,
    sources_unchanged,
)

LORA_TARGET_MODULES = (
    "in_proj_a",
    "in_proj_b",
    "in_proj_qkv",
    "in_proj_z",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
)


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """描述核心优化循环完成的工作量。

    Args:
        optimizer_steps (int): 实际执行的参数更新次数。
        samples_seen (int): 参与训练的样本累计数。
        mean_loss (float): 各样本未缩放 loss 的算术平均值。
    """

    optimizer_steps: int
    samples_seen: int
    mean_loss: float


class ChunkedCrossEntropy:
    """按序列块执行词表投影和 prompt-loss-masked 交叉熵。

    Qwen 词表较大，若一次物化完整长战斗的 logits，峰值内存会随
    ``sequence_length × vocabulary_size`` 增长。此实现只保留一个块的 logits，
    同时保持因果语言模型“位置 j 预测 token j+1”的标准移位口径。

    Args:
        lm_head (Any): 把 decoder hidden state 投影到词表的 PyTorch 模块。
        chunk_size (int): 每次参与词表投影的最大序列位置数。
    """

    def __init__(self, lm_head: Any, chunk_size: int) -> None:
        """保存词表投影层与正数分块长度。

        Args:
            lm_head (Any): 模型词表投影层。
            chunk_size (int): 正数序列分块长度。

        Raises:
            SftTrainingError: 分块长度不是正数。
        """
        if chunk_size <= 0:
            raise SftTrainingError("logits chunk_size 必须为正数")
        self._lm_head = lm_head
        self._chunk_size = chunk_size

    def __call__(self, hidden: Any, input_ids: Any, labels: Any) -> Any:
        """计算所有 assistant token 的平均交叉熵。

        Args:
            hidden (Any): 形状为 ``[batch, sequence, hidden]`` 的 decoder 输出。
            input_ids (Any): 完整对话 token ID。
            labels (Any): 非 assistant 位置为 ``-100`` 的标签。

        Raises:
            SftTrainingError: 当前序列没有任何可监督 token。

        Returns:
            Any: 保持反向传播图的标量平均 loss。
        """
        import torch.utils.checkpoint as torch_checkpoint

        loss_sum = None
        supervised = None
        last_predictor = hidden.shape[1] - 1
        for start in range(0, last_predictor, self._chunk_size):
            end = min(start + self._chunk_size, last_predictor)
            chunk_sum, chunk_count = torch_checkpoint.checkpoint(
                self._chunk_loss,
                hidden[:, start:end],
                input_ids[:, start + 1 : end + 1],
                labels[:, start + 1 : end + 1],
                use_reentrant=False,
            )
            loss_sum = chunk_sum if loss_sum is None else loss_sum + chunk_sum
            supervised = chunk_count if supervised is None else supervised + chunk_count
        if supervised is None or int(supervised.detach().cpu()) == 0:
            raise SftTrainingError("当前序列没有 assistant 监督 token")
        return loss_sum / supervised

    def _chunk_loss(
        self,
        hidden: Any,
        targets: Any,
        labels: Any,
    ) -> tuple[Any, Any]:
        """计算单个 hidden 块的 loss 总和与监督 token 数。

        Args:
            hidden (Any): 当前序列块的 decoder hidden state。
            targets (Any): 与当前预测位置右移一位后的 token ID。
            labels (Any): 与 targets 对齐的 prompt-loss-mask 标签。

        Returns:
            tuple[Any, Any]: 保持梯度的 loss 总和与监督 token 数。
        """
        from torch.nn import functional

        logits = self._lm_head(hidden).float()
        keep = labels.ne(IGNORE_LABEL).reshape(-1).float()
        losses = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1).to(logits.device),
            reduction="none",
        )
        return (losses * keep).sum(), keep.sum()


def optimize(
    model: Any,
    samples: Sequence[TokenizedSample],
    *,
    device: str,
    epochs: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    seed: int,
    trace_path: Path,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 0,
    logits_chunk_size: int = 2048,
    checkpoint_steps: int = 0,
    checkpoint: Callable[[int], None] | None = None,
    max_steps: int | None = None,
) -> OptimizationResult:
    """用 batch=1、梯度累积和可选分块 CE 执行可靠训练。

    Args:
        model (Any): PyTorch 因果语言模型；Qwen 使用 decoder 与 lm_head 分块。
        samples (Sequence[TokenizedSample]): 已完成 assistant 掩码的样本。
        device (str): PyTorch 设备字符串。
        epochs (int): 数据遍历次数。
        learning_rate (float): AdamW 峰值学习率。
        gradient_accumulation_steps (int): 每次优化累积的样本数。
        seed (int): 洗牌与 PyTorch 随机种子。
        trace_path (Path): 每次优化写入一行的 JSONL 路径。
        max_grad_norm (float): 梯度裁剪上限。
        warmup_steps (int): 线性学习率预热的优化步数。
        logits_chunk_size (int): Qwen 词表投影的序列分块长度。
        checkpoint_steps (int): checkpoint 的优化步间隔；零为关闭。
        checkpoint (Callable[[int], None] | None): 接收当前步数的保存回调。
        max_steps (int | None): 可选优化步上限，主要用于真实模型冒烟。

    Raises:
        SftTrainingError: 输入无效、模型无可训练参数或出现非有限数值。
        OSError: 无法写入训练轨迹或 checkpoint。

    Returns:
        OptimizationResult: 优化步数、样本数和平均 loss。
    """
    if max_steps is not None and max_steps <= 0:
        raise SftTrainingError("max_steps 必须为正数")
    if checkpoint_steps < 0 or (checkpoint_steps and checkpoint is None):
        raise SftTrainingError("checkpoint_steps 需要非负数和保存回调")

    import torch

    if not samples:
        raise SftTrainingError("训练样本为空")
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise SftTrainingError("模型没有可训练参数")
    torch.manual_seed(seed)
    randomizer = random.Random(seed)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    decoder, lm_head = _decoder_and_head(model)
    chunked_loss = (
        ChunkedCrossEntropy(lm_head, logits_chunk_size)
        if decoder is not None and lm_head is not None
        else None
    )
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer_steps = 0
    samples_seen = 0
    loss_sum = 0.0
    with trace_path.open("w", encoding="utf-8") as trace:
        for epoch in range(epochs):
            order = list(range(len(samples)))
            randomizer.shuffle(order)
            for begin in range(0, len(order), gradient_accumulation_steps):
                group = order[begin : begin + gradient_accumulation_steps]
                optimizer.zero_grad(set_to_none=True)
                current_lr = _learning_rate_at(
                    optimizer_steps,
                    learning_rate,
                    warmup_steps,
                )
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = current_lr
                group_loss = 0.0
                for index in group:
                    sample = samples[index]
                    input_ids = torch.tensor(
                        [sample.input_ids],
                        dtype=torch.long,
                        device=device,
                    )
                    labels = torch.tensor(
                        [sample.labels],
                        dtype=torch.long,
                        device=device,
                    )
                    if chunked_loss is None:
                        output = model(
                            input_ids=input_ids,
                            attention_mask=torch.ones_like(input_ids),
                            labels=labels,
                            use_cache=False,
                        )
                        loss = output.loss
                    else:
                        hidden = decoder(
                            input_ids=input_ids,
                            attention_mask=None,
                        ).last_hidden_state
                        loss = chunked_loss(hidden, input_ids, labels)
                    loss_value = float(loss.detach().cpu())
                    if not math.isfinite(loss_value):
                        raise SftTrainingError(f"{sample.sample_id}: loss 不是有限数值")
                    (loss / len(group)).backward()
                    group_loss += loss_value
                    loss_sum += loss_value
                    samples_seen += 1
                _require_finite_gradients(model, optimizer_steps + 1)
                torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                optimizer.step()
                optimizer_steps += 1
                trace.write(
                    json.dumps(
                        {
                            "step": optimizer_steps,
                            "epoch": epoch + 1,
                            "samples": len(group),
                            "loss": group_loss / len(group),
                            "learning_rate": current_lr,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                trace.flush()
                if device == "mps":
                    torch.mps.empty_cache()
                if checkpoint_steps and optimizer_steps % checkpoint_steps == 0:
                    assert checkpoint is not None
                    checkpoint(optimizer_steps)
                if max_steps is not None and optimizer_steps >= max_steps:
                    return OptimizationResult(
                        optimizer_steps,
                        samples_seen,
                        loss_sum / samples_seen,
                    )
    return OptimizationResult(optimizer_steps, samples_seen, loss_sum / samples_seen)


def resolve_device(requested: str) -> str:
    """把 ``auto`` 解析为当前 PyTorch 可用设备。

    Args:
        requested (str): ``auto``、``mps`` 或 ``cpu``。

    Raises:
        SftTrainingError: 显式请求 MPS 但当前 PyTorch 不可用。

    Returns:
        str: 可直接传给 PyTorch 的设备字符串。
    """
    import torch

    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SftTrainingError("当前 PyTorch 无法使用 MPS")
    return requested


def attach_lora(
    base_model: Any,
    *,
    rank: int,
    alpha: int,
    seed: int,
    target_modules: Sequence[str] = LORA_TARGET_MODULES,
) -> Any:
    """在确定性随机种子下为因果语言模型装配 LoRA。

    Args:
        base_model (Any): 已加载的 Hugging Face 因果语言模型。
        rank (int): LoRA 矩阵秩。
        alpha (int): LoRA 缩放参数。
        seed (int): adapter 随机初始化种子。
        target_modules (Sequence[str]): 需要注入 LoRA 的模块末级名称。

    Returns:
        Any: PEFT 包装后的可训练模型。
    """
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    return get_peft_model(
        base_model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            target_modules=list(target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )


def publish_adapter(
    model: Any,
    tokenizer: Any,
    destination: Path,
    manifest: Mapping[str, object],
) -> None:
    """完整写入暂存目录后原子发布最终 adapter。

    Args:
        model (Any): 已训练的 PEFT 模型。
        tokenizer (Any): 与训练模型配套的 tokenizer。
        destination (Path): 最终 adapter 目录。
        manifest (Mapping[str, object]): 需要随 adapter 保存的训练清单。

    Raises:
        FileExistsError: 最终目录已经存在。
        OSError: 暂存写入或同文件系统重命名失败。

    Returns:
        None: 完整 adapter 已发布到最终目录。
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"adapter 输出已存在: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}-",
            dir=destination.parent,
        )
    )
    try:
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        _write_json(staging / "train_manifest.json", manifest)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def save_last_checkpoint(model: Any, destination: Path) -> None:
    """用原子符号链接切换覆盖一个 LoRA ``checkpoint-last``。

    Args:
        model (Any): 当前可训练的 PEFT 模型。
        destination (Path): 固定 checkpoint 目录。

    Raises:
        SftTrainingError: 目标是旧式实体目录，无法保证无缺口切换。
        OSError: 权重保存或符号链接原子替换失败。

    Returns:
        None: 新 checkpoint 完整发布后返回。
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not destination.is_symlink():
        raise SftTrainingError(f"checkpoint 目标必须不存在或为符号链接: {destination}")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}-generation-",
            dir=destination.parent,
        )
    )
    pointer = destination.with_name(f".{destination.name}-pointer-{uuid.uuid4().hex}")
    published = False
    try:
        model.save_pretrained(staging, safe_serialization=True)
        pointer.symlink_to(staging.name, target_is_directory=True)
        os.replace(pointer, destination)
        published = True
        for generation in destination.parent.glob(f".{destination.name}-generation-*"):
            if generation != staging:
                shutil.rmtree(generation, ignore_errors=True)
    except BaseException:
        if pointer.is_symlink():
            pointer.unlink()
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
        raise


def train_sft(
    config: SftConfig,
    run_name: str,
    *,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """加载本地 Qwen，训练或续训 LoRA，并保存 adapter 与运行记录。

    Args:
        config (SftConfig): 已校验的训练配置。
        run_name (str): 同时用于 adapter 和 runs 子目录的运行名称。
        max_steps (int | None): 可选的优化步上限。

    Raises:
        SftTrainingError: 名称、输入或输出目录不满足训练约定。
        OSError: 无法读取模型数据或写入产物。

    Returns:
        dict[str, Any]: 可序列化的最终训练清单。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_name):
        raise SftTrainingError(f"无效训练名称: {run_name}")
    adapter_path = config.adapter_root / run_name
    run_path = config.runs_root / run_name
    if adapter_path.exists() or run_path.exists():
        raise SftTrainingError(f"训练输出已存在: {adapter_path} 或 {run_path}")
    if config.init_adapter is not None:
        _validate_init_adapter(config)
    dataset_manifest = validate_sft_dataset(config.dataset_root)
    dataset_manifest_sha256 = _sha256(config.dataset_root / "manifest.json")
    base_model_files = model_source_files(config.base_model)
    base_model_snapshot = snapshot_files(base_model_files)
    base_model_sha256 = sha256_files(base_model_files)
    _require_unchanged_model_sources(
        config.base_model,
        base_model_files,
        base_model_snapshot,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.base_model),
        local_files_only=True,
        trust_remote_code=False,
    )
    samples = load_tokenized_samples(
        config.dataset_root / "train.jsonl",
        tokenizer,
        max_length=config.max_length,
    )
    device = resolve_device(config.device)
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    _require_unchanged_model_sources(
        config.base_model,
        base_model_files,
        base_model_snapshot,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        str(config.base_model),
        dtype=dtype,
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    _require_unchanged_model_sources(
        config.base_model,
        base_model_files,
        base_model_snapshot,
    )
    base_model.config.use_cache = False
    if config.init_adapter is None:
        model = attach_lora(
            base_model,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            seed=config.seed,
        )
    else:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            base_model,
            str(config.init_adapter),
            is_trainable=True,
        )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    run_path.mkdir(parents=True)
    _write_json(run_path / "config.json", config_as_json(config))
    result = optimize(
        model,
        samples,
        device=device,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        seed=config.seed,
        trace_path=run_path / "metrics.jsonl",
        max_grad_norm=config.max_grad_norm,
        warmup_steps=config.warmup_steps,
        logits_chunk_size=config.logits_chunk_size,
        checkpoint_steps=config.checkpoint_steps,
        checkpoint=lambda _step: save_last_checkpoint(
            model,
            run_path / "checkpoint-last",
        ),
        max_steps=max_steps,
    )
    manifest = {
        "run_name": run_name,
        "base_model": str(config.base_model),
        "base_model_sha256": base_model_sha256,
        "init_adapter": (
            str(config.init_adapter) if config.init_adapter is not None else None
        ),
        "approximate_resume": config.init_adapter is not None,
        "dataset": str(config.dataset_root),
        "dataset_files": dataset_manifest["files"],
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "init_adapter_manifest_sha256": _optional_sha256(
            config.init_adapter / "train_manifest.json"
            if config.init_adapter is not None
            else None
        ),
        "adapter": str(adapter_path),
        "device": device,
        "samples": len(samples),
        "samples_seen": result.samples_seen,
        "optimizer_steps": result.optimizer_steps,
        "mean_loss": result.mean_loss,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "max_length": config.max_length,
        "logits_chunk_size": config.logits_chunk_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "checkpoint_steps": config.checkpoint_steps,
        "seed": config.seed,
        "lora": {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "target_modules": list(LORA_TARGET_MODULES),
        },
        "max_steps": max_steps,
    }
    publish_adapter(model, tokenizer, adapter_path, manifest)
    _write_json(run_path / "summary.json", manifest)
    return manifest


def _require_unchanged_model_sources(
    root: Path,
    expected_files: Mapping[str, Path],
    expected_snapshot: Mapping[str, tuple[int, int, int, int, int]],
) -> None:
    """确认训练取摘要与加载前后的基座文件身份保持一致。

    Args:
        root (Path): 本地 Hugging Face 基座模型目录。
        expected_files (Mapping[str, Path]): 取摘要时的模型来源文件。
        expected_snapshot (Mapping[str, tuple[int, int, int, int, int]]): 初始状态。

    Raises:
        SftTrainingError: 文件集合或任一文件状态发生变化。

    Returns:
        None: 基座来源保持不变时返回。
    """
    current_files = model_source_files(root)
    if current_files != dict(expected_files) or not sources_unchanged(
        expected_files,
        expected_snapshot,
    ):
        raise SftTrainingError("训练基座发生变化，已拒绝使用过时指纹")


def _decoder_and_head(model: Any) -> tuple[Any | None, Any | None]:
    """发现可绕过整段 logits 的 decoder 与词表投影层。

    Args:
        model (Any): 裸 Transformers 或 PEFT 包装模型。

    Returns:
        tuple[Any | None, Any | None]: 两者都可用时返回，否则均为 ``None``。
    """
    decoder_getter = getattr(model, "get_decoder", None)
    decoder = decoder_getter() if callable(decoder_getter) else None
    lm_head = getattr(model, "lm_head", None)
    return (
        (decoder, lm_head)
        if decoder is not None and lm_head is not None
        else (None, None)
    )


def _learning_rate_at(step: int, peak: float, warmup_steps: int) -> float:
    """返回当前优化步的线性预热学习率。

    Args:
        step (int): 从零开始的优化步索引。
        peak (float): 预热结束后的峰值学习率。
        warmup_steps (int): 预热步数；零表示直接使用峰值。

    Returns:
        float: 当前优化步使用的学习率。
    """
    if warmup_steps and step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    return peak


def _require_finite_gradients(model: Any, step: int) -> None:
    """拒绝带非有限梯度的参数更新。

    Args:
        model (Any): 当前训练模型。
        step (int): 用于错误定位的优化步数。

    Raises:
        SftTrainingError: 任一可训练参数的梯度包含 NaN 或 Inf。

    Returns:
        None: 所有可见梯度均为有限值后返回。
    """
    import torch

    invalid = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and not torch.isfinite(parameter.grad).all().item()
    ]
    if invalid:
        raise SftTrainingError(f"step {step} 梯度非有限值: {invalid[:3]}")


def _validate_init_adapter(config: SftConfig) -> None:
    """确认近似续训 adapter 与当前 LoRA 形状一致。

    Args:
        config (SftConfig): 含 ``init_adapter`` 的训练配置。

    Raises:
        SftTrainingError: adapter 配置缺失或秩、缩放、目标模块不一致。
        OSError: adapter 配置无法读取。

    Returns:
        None: adapter 可以安全续训时返回。
    """
    assert config.init_adapter is not None
    adapter_config = config.init_adapter / "adapter_config.json"
    if not adapter_config.is_file():
        raise SftTrainingError(f"续训 adapter 配置不存在: {adapter_config}")
    value = json.loads(adapter_config.read_text(encoding="utf-8"))
    actual_targets = set(value.get("target_modules", ()))
    expected_targets = set(LORA_TARGET_MODULES)
    if (
        value.get("r") != config.lora_rank
        or value.get("lora_alpha") != config.lora_alpha
        or actual_targets != expected_targets
    ):
        raise SftTrainingError("续训 adapter 的 rank/alpha/target_modules 与配置不一致")


def _write_json(path: Path, value: object) -> None:
    """以统一缩进写入 UTF-8 JSON。

    Args:
        path (Path): 目标 JSON 路径。
        value (object): 可 JSON 序列化的值。

    Returns:
        None: 文件写入完成后返回。
    """
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    """计算训练输入文件的 SHA-256 摘要。

    Args:
        path (Path): 待读取的文件。

    Returns:
        str: 64 位小写十六进制摘要。
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_sha256(path: Path | None) -> str | None:
    """存在时计算父 adapter 清单摘要。

    Args:
        path (Path | None): 可选的父 adapter 清单。

    Returns:
        str | None: 文件存在时的 SHA-256，否则为 ``None``。
    """
    return _sha256(path) if path is not None and path.is_file() else None
