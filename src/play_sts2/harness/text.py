"""统一清洗游戏富文本，并转换玩家可见的名称与编号。"""

import re
from typing import Any

_MARKUP_PATTERN = re.compile(r"\[/?[A-Za-z_]+(?:=[^\]]+)?\]")
_RESOURCE_ICON_PATTERN = re.compile(
    r"(?:\[img\])?"
    r"(res://[^\[\]\s]+?\.(?:png|svg|webp|jpe?g))"
    r"(?:\[/img\])?",
    re.IGNORECASE,
)
_LOCALIZATION_KEY_PATTERN = re.compile(
    r"[A-Za-z0-9_]+\.pages\.(?:[A-Za-z0-9_]+\.)+"
    r"(?:description|option|title)",
    re.IGNORECASE,
)
_ENERGY_TOKEN = "\ue000"
_STAR_TOKEN = "\ue001"


def clean_game_text(value: Any) -> str:
    """移除富文本，并把连续资源图标转换成带数量的文本。

    Args:
        value (Any): Mod 返回的描述、名称或其他可读字段。

    Returns:
        str: 不含资源路径、保留能量与星能语义的单行文本。
    """
    text = _RESOURCE_ICON_PATTERN.sub(_replace_resource_icon, str(value or ""))
    text = _MARKUP_PATTERN.sub("", text)
    text = _expand_resource_tokens(text, _ENERGY_TOKEN, "能量")
    text = _expand_resource_tokens(text, _STAR_TOKEN, "星能")
    return " ".join(text.split())


def _replace_resource_icon(match: re.Match[str]) -> str:
    """把资源图标替换成等待数量化的内部 token。

    Args:
        match (re.Match[str]): 包含资源路径及可选 ``[img]`` 的匹配。

    Returns:
        str: 能量/星能 token，或保留文件名的未知图标提示。
    """
    filename = match.group(1).rsplit("/", 1)[-1].casefold()
    stem = filename.rsplit(".", 1)[0]
    if stem.endswith("_energy_icon"):
        return _ENERGY_TOKEN
    if stem == "star_icon":
        return _STAR_TOKEN
    return f"〔未知图标: {stem}〕"


def _expand_resource_tokens(text: str, token: str, resource_name: str) -> str:
    """用显式数字或连续图标数生成资源数量。

    Args:
        text (str): 已替换资源路径、尚未展开内部 token 的文本。
        token (str): 当前资源的单字符内部 token。
        resource_name (str): 输出使用的中文资源名称。

    Returns:
        str: 数字覆盖完整连续图标段后的玩家可读文本。
    """
    return re.sub(
        rf"(?:(\d+)\s*点?\s*)?({token}+)",
        lambda match: f"{match.group(1) or len(match.group(2))}点{resource_name}",
        text,
    )


def card_display_name(value: Any, *, upgraded: bool) -> str:
    """返回不会重复追加升级符号的玩家可见卡名。

    Args:
        value (Any): Mod 返回的卡牌名称。
        upgraded (bool): 结构化升级标记。

    Returns:
        str: 恰好包含一个升级后缀的卡牌名称。
    """
    name = clean_game_text(value) or "未知卡牌"
    return f"{name}+" if upgraded and not name.endswith("+") else name


def human_act_number(value: Any) -> str:
    """把 Mod 的零基幕索引转换成人类的一基幕编号。

    Args:
        value (Any): 数字、数字字符串或 ``ACT_01`` 风格幕 ID。

    Returns:
        str: 面向玩家的幕编号；未知非数字值原样保留。
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value + 1)
    text = str(value or "?")
    if text.isdecimal():
        return str(int(text) + 1)
    if text.startswith("ACT_") and text[4:].isdecimal():
        return str(int(text[4:]))
    return text


def is_unresolved_localization_key(value: Any) -> bool:
    """判断字段是否仍是游戏事件本地化键。

    Args:
        value (Any): 待检查的事件描述或选项说明。

    Returns:
        bool: 完整匹配 ``*.pages.*`` 内部键时为真。
    """
    return _LOCALIZATION_KEY_PATTERN.fullmatch(str(value or "")) is not None
