"""从独立文本资源加载 Harness 系统提示词。"""

from importlib import resources

from ..ownership import HarnessLayer


def system_prompt(layer: HarnessLayer) -> str:
    """读取指定决策层的系统提示词。

    Args:
        layer (HarnessLayer): 战斗或战略 Harness 层。

    Raises:
        ValueError: 过渡层不应调用模型，因此没有系统提示词。

    Returns:
        str: 保留资源文件原始换行的系统提示词。
    """
    if layer is HarnessLayer.TRANSIENT:
        raise ValueError("过渡层不应请求模型提示词")
    prompt_file = resources.files(__package__).joinpath(f"{layer.value}.txt")
    return prompt_file.read_text(encoding="utf-8")


__all__ = ["system_prompt"]
