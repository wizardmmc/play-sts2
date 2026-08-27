"""构建 Web Wiki 与 Mod 实测两类 Markdown 游戏知识。"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..client import Health
from .markdown import KnowledgeEntry, KnowledgeFormatError, parse_knowledge_entry

_COLLECTIONS = (
    "cards",
    "relics",
    "potions",
    "powers",
    "characters",
    "monsters",
    "events",
    "keywords",
)
_ENTITY_TYPES = {
    "cards": "card",
    "relics": "relic",
    "potions": "potion",
    "powers": "power",
    "characters": "character",
    "monsters": "monster",
    "events": "event",
    "keywords": "keyword",
}
_LOCAL_MARKER = "<!-- praxis:local -->"
_MARKUP = re.compile(r"\[/?[A-Za-z_]+(?:=[^\]]+)?\]")
_RESOURCE = re.compile(r"res://\S+?\.png")


class GameDataClient(Protocol):
    """描述知识导出所需的只读 Mod 客户端接口。"""

    def health(self) -> Health:
        """返回当前 Mod 与游戏版本信息。

        Returns:
            Health: 当前游戏实例的健康信息。
        """
        ...

    def data_collection(self, collection: str) -> list[dict[str, Any]]:
        """读取一个游戏实体集合。

        Args:
            collection (str): Mod 支持的集合名称。

        Returns:
            list[dict[str, Any]]: Mod 原样导出的实体列表。
        """
        ...


@dataclass(frozen=True, slots=True)
class KnowledgeBuildResult:
    """描述一次知识导入或导出的落盘结果。

    Args:
        output_root (Path): 当前来源的知识根目录。
        entry_count (int): 成功写入的单实体 Markdown 数量。
        categories (dict[str, int]): 每个实体类别的写入数量。
    """

    output_root: Path
    entry_count: int
    categories: dict[str, int]


def import_web_wiki(source_root: Path, output_root: Path) -> KnowledgeBuildResult:
    """导入结构化 Web Wiki，并将来源统一标为 ``web_wiki``。

    只迁移来源文件分隔线以上的站点事实，不把其他工程的本地注记误标为
    Web 数据。

    Args:
        source_root (Path): 按类别存放单实体 Markdown 的 Wiki 根目录。
        output_root (Path): 当前项目的 ``data/game_knowledge`` 根目录。

    Raises:
        KnowledgeFormatError: 来源条目不符合最小 Markdown 契约。
        OSError: 无法读取来源或写入目标目录。

    Returns:
        KnowledgeBuildResult: Web Wiki 导入目录及实体计数。
    """
    destination = Path(output_root) / "web_wiki"
    categories: dict[str, int] = {}
    expected: dict[str, set[str]] = {}
    for category_dir in sorted(
        path for path in Path(source_root).iterdir() if path.is_dir()
    ):
        count = 0
        expected[category_dir.name] = set()
        for source_file in sorted(category_dir.glob("*.md")):
            entry = parse_knowledge_entry(source_file.read_text(encoding="utf-8"))
            body = entry.body.partition(_LOCAL_MARKER)[0].strip()
            metadata = dict(entry.metadata)
            original_source = metadata["source"]
            metadata["type"] = metadata["type"].removeprefix("wiki/")
            metadata["source"] = "web_wiki"
            metadata["source_detail"] = original_source
            imported = KnowledgeEntry(metadata=metadata, body=body)
            target = destination / category_dir.name / f"{_safe_id(entry.object_id)}.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(imported.to_markdown(), encoding="utf-8")
            expected[category_dir.name].add(target.stem)
            count += 1
        if count:
            categories[category_dir.name] = count
    _remove_stale_entries(destination, expected)
    return _write_result(destination, "web_wiki", categories)


def export_mod_knowledge(
    client: GameDataClient,
    output_root: Path,
) -> KnowledgeBuildResult:
    """从正在运行的游戏导出原始快照和单实体 Markdown。

    Args:
        client (GameDataClient): 已连接真实 Mod 的只读数据客户端。
        output_root (Path): 当前项目的 ``data/game_knowledge`` 根目录。

    Raises:
        KnowledgeFormatError: Mod 实体缺少 ID、名称或版本不可用作目录名。
        OSError: 无法创建目录或写入快照。
        httpx.HTTPError: Mod 健康检查或数据端点请求失败。

    Returns:
        KnowledgeBuildResult: 以游戏版本分隔的 Mod 导出结果。
    """
    health = client.health()
    game_version = _safe_version(health.game_version)
    destination = Path(output_root) / "mod_export" / game_version
    raw_root = destination / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    categories: dict[str, int] = {}
    expected: dict[str, set[str]] = {}
    for collection in _COLLECTIONS:
        entities = client.data_collection(collection)
        (raw_root / f"{collection}.json").write_text(
            json.dumps(entities, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        expected[collection] = set()
        for entity in entities:
            entry = _mod_entry(collection, entity, game_version)
            target = destination / collection / f"{_safe_id(entry.object_id)}.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(entry.to_markdown(), encoding="utf-8")
            expected[collection].add(target.stem)
        categories[collection] = len(entities)
    _remove_stale_entries(destination, expected)
    return _write_result(
        destination,
        "mod_export",
        categories,
        game_version=game_version,
    )


def _mod_entry(
    collection: str,
    entity: Mapping[str, Any],
    game_version: str,
) -> KnowledgeEntry:
    """把一个 Mod 实体转换成 Praxis 风格的 Markdown 条目。

    Args:
        collection (str): 实体所属的 Mod 集合。
        entity (Mapping[str, Any]): Mod 原样导出的单个实体。
        game_version (str): 实测数据所属游戏版本。

    Raises:
        KnowledgeFormatError: 实体缺少可用的 ID 或显示名称。

    Returns:
        KnowledgeEntry: 可写入对应类别目录的知识条目。
    """
    object_id = _required_text(entity, "id")
    name = _required_text(entity, "name")
    metadata = {
        "id": object_id,
        "name": name,
        "type": _ENTITY_TYPES[collection],
        "source": "mod_export",
        "game_version": game_version,
    }
    body = _MOD_RENDERERS[collection](entity, metadata)
    return KnowledgeEntry(metadata=metadata, body=body)


def _render_card(entity: Mapping[str, Any], metadata: dict[str, str]) -> str:
    """渲染卡牌属性、效果和升级文本。

    Args:
        entity (Mapping[str, Any]): Mod 导出的卡牌实体。
        metadata (dict[str, str]): 将写入 frontmatter 的公共字段。

    Returns:
        str: 卡牌 Markdown 正文。
    """
    metadata.update(
        _scalar_fields(
            entity,
            {
                "color": "character",
                "type": "card_type",
                "rarity": "rarity",
                "target": "target",
            },
        )
    )
    metadata["cost"] = (
        "X" if entity.get("is_x_cost") is True else str(entity.get("cost", ""))
    )
    description = _clean_text(entity.get("description"))
    lines = ["## 效果", description or "——"]
    upgrade = entity.get("upgrade")
    if isinstance(upgrade, Mapping):
        upgrade_description = _clean_text(upgrade.get("description"))
        if upgrade_description and upgrade_description != description:
            lines.extend(("## 升级", upgrade_description))
    return "\n".join(lines)


def _render_relic(entity: Mapping[str, Any], metadata: dict[str, str]) -> str:
    """渲染遗物属性与效果。

    Args:
        entity (Mapping[str, Any]): Mod 导出的遗物实体。
        metadata (dict[str, str]): 将写入 frontmatter 的公共字段。

    Returns:
        str: 遗物 Markdown 正文。
    """
    metadata.update(_scalar_fields(entity, {"rarity": "rarity", "pool": "pool"}))
    return f"## 效果\n{_clean_text(entity.get('description')) or '——'}"


def _render_potion(entity: Mapping[str, Any], metadata: dict[str, str]) -> str:
    """渲染药水属性与效果。

    Args:
        entity (Mapping[str, Any]): Mod 导出的药水实体。
        metadata (dict[str, str]): 将写入 frontmatter 的公共字段。

    Returns:
        str: 药水 Markdown 正文。
    """
    metadata.update(
        _scalar_fields(
            entity,
            {
                "rarity": "rarity",
                "pool": "pool",
                "usage": "usage",
                "target_type": "target",
            },
        )
    )
    return f"## 效果\n{_clean_text(entity.get('description')) or '——'}"


def _render_power(entity: Mapping[str, Any], metadata: dict[str, str]) -> str:
    """渲染能力属性与效果。

    Args:
        entity (Mapping[str, Any]): Mod 导出的能力实体。
        metadata (dict[str, str]): 将写入 frontmatter 的公共字段。

    Returns:
        str: 能力 Markdown 正文。
    """
    metadata.update(
        _scalar_fields(
            entity,
            {"type": "power_type", "stack_type": "stack_type"},
        )
    )
    return f"## 效果\n{_clean_text(entity.get('description')) or '——'}"


def _render_character(entity: Mapping[str, Any], metadata: dict[str, str]) -> str:
    """渲染角色初始属性、牌组和物品。

    Args:
        entity (Mapping[str, Any]): Mod 导出的角色实体。
        metadata (dict[str, str]): 将写入 frontmatter 的公共字段。

    Returns:
        str: 角色 Markdown 正文。
    """
    metadata.update(
        _scalar_fields(
            entity,
            {
                "starting_hp": "hp",
                "starting_gold": "gold",
                "max_energy": "energy",
                "orb_slots": "orb_slots",
            },
        )
    )
    lines = ["## 起始牌组", *_bullet_ids(entity.get("starting_deck"))]
    lines.extend(("", "## 起始遗物", *_bullet_ids(entity.get("starting_relics"))))
    potions = _bullet_ids(entity.get("starting_potions"))
    if potions:
        lines.extend(("", "## 起始药水", *potions))
    description = _clean_text(entity.get("description"))
    if description:
        lines.extend(("", "## 简介", description))
    return "\n".join(lines)


def _render_monster(entity: Mapping[str, Any], metadata: dict[str, str]) -> str:
    """渲染敌人生命范围与已导出的招式名称。

    Args:
        entity (Mapping[str, Any]): Mod 导出的敌人实体。
        metadata (dict[str, str]): 将写入 frontmatter 的公共字段。

    Returns:
        str: 敌人 Markdown 正文。
    """
    metadata.update(
        _scalar_fields(
            entity,
            {"type": "room", "min_hp": "min_hp", "max_hp": "max_hp"},
        )
    )
    moves = entity.get("moves")
    lines = ["## 招式"]
    if isinstance(moves, Sequence) and not isinstance(moves, (str, bytes)):
        for move in moves:
            if not isinstance(move, Mapping):
                continue
            move_id = _clean_text(move.get("id"))
            move_name = _clean_text(move.get("name"))
            if move_id or move_name:
                lines.append(f"- {move_id or '?'}：{move_name or '——'}")
    if len(lines) == 1:
        lines.append("（无招式数据）")
    return "\n".join(lines)


def _render_event(entity: Mapping[str, Any], metadata: dict[str, str]) -> str:
    """渲染事件初始文本与可见选项。

    Args:
        entity (Mapping[str, Any]): Mod 导出的事件实体。
        metadata (dict[str, str]): 将写入 frontmatter 的公共字段。

    Returns:
        str: 事件 Markdown 正文。
    """
    metadata.update(_scalar_fields(entity, {"type": "event_type", "act": "act"}))
    lines = ["## 文本", _clean_text(entity.get("description")) or "——", "## 选项"]
    options = entity.get("options")
    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
        for option in options:
            if not isinstance(option, Mapping):
                continue
            option_id = _clean_text(option.get("id"))
            title = _clean_text(option.get("title"))
            description = _clean_text(option.get("description"))
            lines.append(f"- [{option_id}] {title} — {description}".rstrip(" —"))
    if lines[-1] == "## 选项":
        lines.append("（无）")
    return "\n".join(lines)


def _render_keyword(entity: Mapping[str, Any], _metadata: dict[str, str]) -> str:
    """渲染游戏关键词释义。

    Args:
        entity (Mapping[str, Any]): Mod 导出的关键词实体。
        _metadata (dict[str, str]): 当前类别没有额外 frontmatter 字段。

    Returns:
        str: 关键词 Markdown 正文。
    """
    return f"## 释义\n{_clean_text(entity.get('description')) or '——'}"


_MOD_RENDERERS = {
    "cards": _render_card,
    "relics": _render_relic,
    "potions": _render_potion,
    "powers": _render_power,
    "characters": _render_character,
    "monsters": _render_monster,
    "events": _render_event,
    "keywords": _render_keyword,
}


def _scalar_fields(
    entity: Mapping[str, Any],
    fields: Mapping[str, str],
) -> dict[str, str]:
    """选择非空标量字段并转换为扁平 frontmatter 字符串。

    Args:
        entity (Mapping[str, Any]): Mod 原始实体。
        fields (Mapping[str, str]): 原字段到目标字段的名称映射。

    Returns:
        dict[str, str]: 可直接合并进 frontmatter 的字段。
    """
    result: dict[str, str] = {}
    for source_name, target_name in fields.items():
        value = entity.get(source_name)
        if value is not None and not isinstance(value, (Mapping, list, tuple)):
            result[target_name] = (
                str(value).lower() if isinstance(value, bool) else str(value)
            )
    return result


def _bullet_ids(value: object) -> list[str]:
    """把实体 ID 序列渲染为 Markdown 项目符号。

    Args:
        value (object): 预期为字符串序列的 Mod 字段。

    Returns:
        list[str]: 保留顺序的项目符号行；无有效值时为空。
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [f"- {item}" for item in value if isinstance(item, str) and item]


def _clean_text(value: object) -> str:
    """清理游戏富文本标签和资源路径，但不改写事实。

    Args:
        value (object): Mod 返回的可能文本字段。

    Returns:
        str: 保留原句的纯文本；非字符串返回空串。
    """
    if not isinstance(value, str):
        return ""
    text = value.replace(
        "res://images/packed/sprite_fonts/defect_energy_icon.png", "能量"
    )
    text = _RESOURCE.sub("", text)
    return _MARKUP.sub("", text).strip()


def _required_text(entity: Mapping[str, Any], field: str) -> str:
    """读取 Mod 实体的必需文本字段。

    Args:
        entity (Mapping[str, Any]): Mod 原始实体。
        field (str): 必需字段名称。

    Raises:
        KnowledgeFormatError: 字段缺失、为空或不是字符串。

    Returns:
        str: 校验通过的非空文本。
    """
    value = entity.get(field)
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeFormatError(f"Mod 实体缺少字段: {field}")
    return value.strip()


def _safe_id(object_id: str) -> str:
    """验证实体 ID 可以安全作为单个文件名。

    Args:
        object_id (str): 游戏实体稳定 ID。

    Raises:
        KnowledgeFormatError: ID 为空、为目录别名或包含路径分隔符。

    Returns:
        str: 可直接作为 Markdown 文件名主体的 ID。
    """
    if object_id in {"", ".", ".."} or "/" in object_id or "\\" in object_id:
        raise KnowledgeFormatError(f"实体 ID 不能作为文件名: {object_id}")
    return object_id


def _safe_version(game_version: str) -> str:
    """验证游戏版本可以作为知识快照目录名。

    Args:
        game_version (str): Mod 健康检查返回的游戏版本。

    Raises:
        KnowledgeFormatError: 版本包含路径字符或为空。

    Returns:
        str: 可直接作为目录名的游戏版本。
    """
    if not re.fullmatch(r"[A-Za-z0-9._-]+", game_version) or game_version in {
        ".",
        "..",
    }:
        raise KnowledgeFormatError(f"无效游戏版本: {game_version}")
    return game_version


def _remove_stale_entries(
    destination: Path,
    expected: Mapping[str, set[str]],
) -> None:
    """删除同一生成目录中已不属于本次采样的旧 Markdown。

    只处理来源根下的直接类别目录和 ``*.md``，不会删除原始 JSON、清单或
    非 Markdown 文件。

    Args:
        destination (Path): ``web_wiki`` 或某个 Mod 版本的生成根目录。
        expected (Mapping[str, set[str]]): 各类别本次实际写入的实体 ID。

    Returns:
        None: 过期生成条目清理完成后返回。
    """
    if not destination.is_dir():
        return
    for category_dir in destination.iterdir():
        if not category_dir.is_dir() or category_dir.name == "raw":
            continue
        current_ids = expected.get(category_dir.name, set())
        for knowledge_file in category_dir.glob("*.md"):
            if knowledge_file.stem not in current_ids:
                knowledge_file.unlink()


def _write_result(
    destination: Path,
    source: str,
    categories: dict[str, int],
    *,
    game_version: str | None = None,
) -> KnowledgeBuildResult:
    """写入最小知识清单并返回构建结果。

    Args:
        destination (Path): 当前来源的知识根目录。
        source (str): 所有条目共享的来源类别。
        categories (dict[str, int]): 各类别的实体数量。
        game_version (str | None): Mod 实测来源的可选游戏版本。

    Returns:
        KnowledgeBuildResult: 写盘目录和实体计数。
    """
    destination.mkdir(parents=True, exist_ok=True)
    entry_count = sum(categories.values())
    manifest: dict[str, object] = {
        "source": source,
        "entries": entry_count,
        "categories": categories,
    }
    if game_version is not None:
        manifest["game_version"] = game_version
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return KnowledgeBuildResult(destination, entry_count, categories)
