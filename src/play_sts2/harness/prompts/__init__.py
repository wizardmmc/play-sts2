"""从独立文本资源加载 Harness 系统提示词。"""

import re
from collections.abc import Mapping
from importlib import resources
from typing import Any

from ..ownership import HarnessLayer

_MARKUP_PATTERN = re.compile(r"\[/?[A-Za-z_]+(?:=[^\]]+)?\]")
_RESOURCE_PATTERN = re.compile(r"res://\S+?\.png")


def system_prompt(
    layer: HarnessLayer,
    state: Mapping[str, Any] | None = None,
) -> str:
    """读取指定决策层的系统提示词。

    Args:
        layer (HarnessLayer): 战斗或战略 Harness 层。
        state (Mapping[str, Any] | None): 当前完整状态；战斗层据此补充遗物效果。

    Raises:
        ValueError: 过渡层不应调用模型，因此没有系统提示词。

    Returns:
        str: 保留资源文件换行并按需附加动态上下文的系统提示词。
    """
    if layer is HarnessLayer.TRANSIENT:
        raise ValueError("过渡层不应请求模型提示词")
    prompt_file = resources.files(__package__).joinpath(f"{layer.value}.txt")
    prompt = prompt_file.read_text(encoding="utf-8").rstrip()
    if layer is HarnessLayer.BATTLE and state is not None:
        prompt = (
            f"{prompt}\n\n【当前回合】{state.get('turn', 0)}\n\n{_render_relics(state)}"
        )
    return prompt


def _render_relics(state: Mapping[str, Any]) -> str:
    """把战斗中持续生效的遗物说明放入 system 上下文。

    Args:
        state (Mapping[str, Any]): 当前完整游戏状态。

    Returns:
        str: 带稳定索引的遗物列表；空列表明确显示为无。
    """
    run = state.get("run")
    relics = run.get("relics") if isinstance(run, Mapping) else None
    lines = ["【遗物】"]
    if not isinstance(relics, list) or not relics:
        return "\n".join((*lines, "- 无"))
    for fallback_index, relic in enumerate(relics):
        if not isinstance(relic, Mapping):
            continue
        index = relic.get("index", fallback_index)
        name = _clean_text(relic.get("name")) or "未知遗物"
        if relic.get("is_melted") is True:
            lines.append(f"- [{index}] {name}（已熔毁，效果失效）")
            continue
        description = _clean_text(relic.get("description"))
        suffix = f": {description}" if description else ""
        lines.append(f"- [{index}] {name}{suffix}")
    return "\n".join(lines if len(lines) > 1 else (*lines, "- 无"))


def _clean_text(value: Any) -> str:
    """移除游戏富文本标记和资源路径。

    Args:
        value (Any): Mod 返回的可空文本值。

    Returns:
        str: 适合直接进入模型提示词的单行文本。
    """
    text = _RESOURCE_PATTERN.sub("", str(value or ""))
    return " ".join(_MARKUP_PATTERN.sub("", text).split())


__all__ = ["system_prompt"]
