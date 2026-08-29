"""应用固定版本、可审计且可重复生成的少量事实修正。"""

import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

from .markdown import KnowledgeEntry

FIXED_GAME_VERSION = "v0.107.1"
SUPPORTED_GAME_VERSIONS = frozenset({FIXED_GAME_VERSION, "v0.111.0"})
_CURATED_ROOT = Path(__file__).with_name("curated")


@cache
def load_overrides(game_version: str = FIXED_GAME_VERSION) -> dict[str, Any]:
    """读取随代码版本管理的指定版本事实修正。

    Args:
        game_version (str): 带 ``v`` 前缀的受支持游戏版本。

    Returns:
        dict[str, Any]: 按类别和对象 ID 组织的修正规则。

    Raises:
        ValueError: 版本不受支持或修正规则声明了其他游戏版本。
        OSError: 修正规则文件不可读。
    """
    if game_version not in SUPPORTED_GAME_VERSIONS:
        raise ValueError(f"不支持的 curated 版本: {game_version}")
    path = _CURATED_ROOT / f"{game_version}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("game_version") != game_version:
        raise ValueError(f"curated override 版本必须为 {game_version}")
    return payload


def curate_entity(
    category: str,
    entity: Mapping[str, Any],
    game_version: str = FIXED_GAME_VERSION,
) -> dict[str, Any]:
    """把 curated 文本替换递归应用到 Mod 原始实体副本。

    Args:
        category (str): 实体类别。
        entity (Mapping[str, Any]): Mod 原始实体。
        game_version (str): 实体所属的受支持游戏版本。

    Returns:
        dict[str, Any]: 不修改输入对象的修正后实体。
    """
    object_id = str(entity.get("id") or "")
    override = (
        load_overrides(game_version)
        .get("entities", {})
        .get(category, {})
        .get(object_id, {})
    )
    replacements = override.get("text_replacements", [])
    return _replace_value(dict(entity), replacements)


def curate_entry(
    category: str,
    entry: KnowledgeEntry,
    game_version: str | None = None,
) -> KnowledgeEntry:
    """把同一组 curated 文本替换应用到已渲染 Markdown 条目。

    Args:
        category (str): 实体类别。
        entry (KnowledgeEntry): 待修正的规范事实 Markdown 条目。
        game_version (str | None): 显式版本；省略时读取条目元数据。

    Returns:
        KnowledgeEntry: 元数据不变、正文已应用固定版本替换的新条目。
    """
    resolved_version = game_version or str(
        entry.metadata.get("game_version", FIXED_GAME_VERSION)
    )
    override = (
        load_overrides(resolved_version)
        .get("entities", {})
        .get(category, {})
        .get(entry.object_id, {})
    )
    body = _replace_text(entry.body, override.get("text_replacements", []))
    return KnowledgeEntry(metadata=dict(entry.metadata), body=body)


def is_question_excluded(
    category: str,
    object_id: str,
    game_version: str = FIXED_GAME_VERSION,
) -> bool:
    """判断固定版本对象是否被明确排除在监督问法之外。

    该判断只匹配版本化清单中的完整 ID，不按 ``FAKE_``、``MOCK_`` 等
    前缀做推断。

    Args:
        category (str): 实体类别。
        object_id (str): 游戏实体稳定 ID。
        game_version (str): 实体所属的受支持游戏版本。

    Returns:
        bool: 完整 ID 出现在固定版本排除清单中时为真。
    """
    exclusions = (
        load_overrides(game_version).get("question_exclusions", {}).get(category, [])
    )
    return object_id in exclusions


def _replace_value(value: Any, replacements: list[dict[str, str]]) -> Any:
    """递归替换字符串、列表和字典中的固定版本文本。

    Args:
        value (Any): 待修正的任意 JSON 兼容值。
        replacements (list[dict[str, str]]): 有序的原文与替换文本规则。

    Returns:
        Any: 保持原容器结构的修正后副本。
    """
    if isinstance(value, str):
        return _replace_text(value, replacements)
    if isinstance(value, list):
        return [_replace_value(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_value(item, replacements) for key, item in value.items()}
    return value


def _replace_text(value: str, replacements: list[dict[str, str]]) -> str:
    """按声明顺序应用精确字符串替换。

    Args:
        value (str): 待修正文本。
        replacements (list[dict[str, str]]): 有序的 ``from``、``to`` 规则。

    Returns:
        str: 应用全部规则后的文本。
    """
    for replacement in replacements:
        value = value.replace(str(replacement["from"]), str(replacement["to"]))
    return value
