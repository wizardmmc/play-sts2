"""执行 CUDA 专用 LoRA SFT。"""

from typing import Any

from ..sft import train_sft
from .config import CudaSftConfig


def train_sft_cuda(
    config: CudaSftConfig,
    run_name: str,
    *,
    max_steps: int | None = None,
    exact_resume: bool = False,
) -> dict[str, Any]:
    """执行 CUDA LoRA 训练或精确续训。

    Args:
        config (CudaSftConfig): CUDA 专用配置。
        run_name (str): 日期开头的运行名称。
        max_steps (int | None): 可选优化步上限。
        exact_resume (bool): 是否从同名 checkpoint 精确续训。

    Returns:
        dict[str, Any]: 可序列化训练清单。
    """
    return train_sft(
        config,
        run_name,
        max_steps=max_steps,
        exact_resume=exact_resume,
    )
