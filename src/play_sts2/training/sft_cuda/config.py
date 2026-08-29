"""读取 CUDA SFT 专用配置。"""

import re
import tomllib
from pathlib import Path

from ..sft import SftConfig, SftTrainingError

CudaSftConfig = SftConfig


def load_cuda_sft_config(path: Path) -> CudaSftConfig:
    """读取并校验一份只允许 CUDA 设备的 SFT 配置。

    Args:
        path (Path): CUDA TOML 配置路径。

    Raises:
        SftTrainingError: 字段缺失、数值无效或设备不是 CUDA。
        OSError: 配置无法读取。
        tomllib.TOMLDecodeError: TOML 语法无效。

    Returns:
        CudaSftConfig: 与共享训练核心兼容的已校验配置。
    """
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    try:
        init_adapter = data.get("init_adapter")
        expand_init_adapter = data.get("expand_init_adapter", False)
        if not isinstance(expand_init_adapter, bool):
            raise TypeError("expand_init_adapter 必须是布尔值")
        config = SftConfig(
            base_model=Path(data["base_model"]),
            dataset_root=Path(data["dataset_root"]),
            adapter_root=Path(data["adapter_root"]),
            runs_root=Path(data["runs_root"]),
            device=str(data["device"]),
            epochs=int(data["epochs"]),
            learning_rate=float(data["learning_rate"]),
            max_length=int(data["max_length"]),
            gradient_accumulation_steps=int(data["gradient_accumulation_steps"]),
            seed=int(data["seed"]),
            lora_rank=int(data["lora_rank"]),
            lora_alpha=int(data["lora_alpha"]),
            max_grad_norm=float(data.get("max_grad_norm", 1.0)),
            eval_max_new_tokens=int(data.get("eval_max_new_tokens", 256)),
            warmup_steps=int(data.get("warmup_steps", 4)),
            logits_chunk_size=int(data.get("logits_chunk_size", 2048)),
            checkpoint_steps=int(data.get("checkpoint_steps", 2000)),
            init_adapter=Path(init_adapter) if init_adapter else None,
            knowledge_epoch_start=int(data.get("knowledge_epoch_start", 1)),
            expand_init_adapter=expand_init_adapter,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SftTrainingError(f"无效 CUDA SFT 配置: {exc}") from exc
    positive = {
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "max_length": config.max_length,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "max_grad_norm": config.max_grad_norm,
        "eval_max_new_tokens": config.eval_max_new_tokens,
        "logits_chunk_size": config.logits_chunk_size,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if config.warmup_steps < 0:
        invalid.append("warmup_steps")
    if config.checkpoint_steps < 0:
        invalid.append("checkpoint_steps")
    if not 1 <= config.knowledge_epoch_start <= 5:
        invalid.append("knowledge_epoch_start")
    if config.knowledge_epoch_start + config.epochs - 1 > 5:
        invalid.append("knowledge_epoch_range")
    if config.expand_init_adapter and config.init_adapter is None:
        invalid.append("expand_init_adapter")
    if invalid or re.fullmatch(r"cuda(?::\d+)?", config.device) is None:
        detail = ", ".join(invalid) if invalid else f"device={config.device}"
        raise SftTrainingError(f"CUDA SFT 配置值无效: {detail}")
    return config
