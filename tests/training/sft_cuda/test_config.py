"""验证 CUDA SFT 只接受显式 CUDA 配置。"""

from pathlib import Path

import pytest

from play_sts2.training.sft import SftTrainingError
from play_sts2.training.sft_cuda import load_cuda_sft_config


def _write_config(path: Path, device: str) -> None:
    """写入一份最小 CUDA 配置。

    Args:
        path (Path): TOML 输出路径。
        device (str): 待验证设备。

    Returns:
        None: 文件写完后返回。
    """
    path.write_text(
        f'''base_model = "models/base/qwen3.5-4b"
dataset_root = "data/datasets/sft"
adapter_root = "models/adapters"
runs_root = "runs/sft"
device = "{device}"
epochs = 1
learning_rate = 0.0001
max_length = 12288
gradient_accumulation_steps = 1
seed = 20260828
lora_rank = 16
lora_alpha = 32
knowledge_epoch_start = 3
init_adapter = "models/adapters/e4"
expand_init_adapter = true
''',
        encoding="utf-8",
    )


def test_load_cuda_sft_config_accepts_numbered_cuda_device(tmp_path: Path) -> None:
    """CUDA 配置应保留显式设备并使用 BF16 基座约定。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Raises:
        AssertionError: 配置字段没有正确解析。

    Returns:
        None: 此测试不访问 GPU。
    """
    path = tmp_path / "cuda.toml"
    _write_config(path, "cuda:0")

    config = load_cuda_sft_config(path)

    assert config.device == "cuda:0"
    assert config.base_model == Path("models/base/qwen3.5-4b")
    assert config.lora_rank == 16
    assert config.knowledge_epoch_start == 3
    assert config.expand_init_adapter is True


def test_load_cuda_sft_config_rejects_cpu(tmp_path: Path) -> None:
    """CUDA 配置不能静默回退到 CPU。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Raises:
        AssertionError: 非 CUDA 设备未被拒绝。

    Returns:
        None: 此测试不访问 GPU。
    """
    path = tmp_path / "cpu.toml"
    _write_config(path, "cpu")

    with pytest.raises(SftTrainingError, match="CUDA"):
        load_cuda_sft_config(path)
