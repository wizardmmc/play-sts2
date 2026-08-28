"""验证 CUDA 入口复用训练核心但保持独立路由。"""

from pathlib import Path

import pytest

from play_sts2.training.sft import SftConfig
from play_sts2.training.sft import trainer as shared_trainer
from play_sts2.training.sft_cuda import trainer


def test_train_sft_cuda_calls_shared_core_with_cuda_config(monkeypatch: object) -> None:
    """CUDA 包应把已校验配置和续训参数交给共享训练核心。

    Args:
        monkeypatch (object): Pytest 提供的属性替换工具。

    Raises:
        AssertionError: CUDA 入口丢失运行参数或改用其他实现。

    Returns:
        None: 此测试不访问真实模型或 GPU。
    """
    config = SftConfig(
        base_model=Path("base"),
        dataset_root=Path("data"),
        adapter_root=Path("adapters"),
        runs_root=Path("runs"),
        device="cuda:0",
        epochs=1,
        learning_rate=1e-4,
        max_length=128,
        gradient_accumulation_steps=1,
        seed=7,
        lora_rank=16,
        lora_alpha=32,
    )
    calls: list[tuple[object, str, int | None, bool]] = []
    monkeypatch.setattr(
        trainer,
        "train_sft",
        lambda value, name, max_steps=None, exact_resume=False: (
            calls.append((value, name, max_steps, exact_resume))
            or {"device": value.device}
        ),
    )

    result = trainer.train_sft_cuda(
        config,
        "20260828-cuda",
        max_steps=1,
        exact_resume=True,
    )

    assert calls == [(config, "20260828-cuda", 1, True)]
    assert result == {"device": "cuda:0"}


def test_cuda_rng_state_round_trips_for_exact_resume(monkeypatch: object) -> None:
    """CUDA checkpoint 应保存并恢复指定设备的随机状态。

    Args:
        monkeypatch (object): Pytest 提供的属性替换工具。

    Raises:
        AssertionError: CUDA 设备身份或随机状态在续训时丢失。

    Returns:
        None: 此测试用替身方法，不需要真实 GPU。
    """
    import torch

    state = torch.tensor([1, 2, 3], dtype=torch.uint8)
    restored: list[tuple[object, str]] = []
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: state)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda value, device: restored.append((value, device)),
    )

    captured = shared_trainer._device_rng_state("cuda:1")
    shared_trainer._restore_device_rng_state("cuda:1", captured)

    assert captured is state
    assert restored == [(state, "cuda:1")]


def test_resolve_cuda_device_canonicalizes_and_rejects_out_of_range(
    monkeypatch: object,
) -> None:
    """CUDA 设备应规范成有效逻辑编号，并在模型加载前拒绝越界。

    Args:
        monkeypatch (object): Pytest 提供的属性替换工具。

    Raises:
        AssertionError: 裸设备未解析或越界设备被接受。

    Returns:
        None: 此测试不访问真实 GPU。
    """
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert shared_trainer.resolve_device("cuda") == "cuda:1"
    assert shared_trainer.resolve_device("cuda:0") == "cuda:0"
    with pytest.raises(shared_trainer.SftTrainingError, match="cuda:2"):
        shared_trainer.resolve_device("cuda:2")


def test_cuda_base_model_dtype_is_bfloat16() -> None:
    """CUDA 基座必须使用 BF16，不能跟随 CPU 回退到 FP32。

    Raises:
        AssertionError: CUDA 与 CPU 的基座精度约定不正确。

    Returns:
        None: 此测试只比较 PyTorch dtype 常量。
    """
    import torch

    assert shared_trainer._base_model_dtype("cuda:0") is torch.bfloat16
    assert shared_trainer._base_model_dtype("cpu") is torch.float32
