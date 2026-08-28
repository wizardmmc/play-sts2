"""构建 Web Wiki 与 Mod 实测两类 Markdown 游戏知识。"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..client import Health
from .curation import FIXED_GAME_VERSION, curate_entity
from .markdown import KnowledgeEntry, KnowledgeFormatError, parse_knowledge_entry

_COLLECTIONS = (
    "acts",
    "cards",
    "relics",
    "potions",
    "enchantments",
    "powers",
    "characters",
    "monsters",
    "encounters",
    "events",
    "keywords",
)
_ENTITY_TYPES = {
    "acts": "act",
    "cards": "card",
    "relics": "relic",
    "potions": "potion",
    "enchantments": "enchantment",
    "powers": "power",
    "characters": "character",
    "monsters": "monster",
    "encounters": "encounter",
    "events": "event",
    "keywords": "keyword",
}
_FIXED_GAME_VERSION = FIXED_GAME_VERSION
_REQUIRED_ACT_IDS = ("OVERGROWTH", "UNDERDOCKS", "HIVE", "GLORY")
_REQUIRED_ACT_POOLS = (
    "weak_encounters",
    "regular_encounters",
    "elite_encounters",
    "boss_encounters",
)
_DEFECT_ORB_IDS = (
    "LIGHTNING_ORB",
    "FROST_ORB",
    "DARK_ORB",
    "GLASS_ORB",
    "PLASMA_ORB",
)
_DEFECT_ORB_SUPPLEMENT = Path("supplements/v0.107.1/characters/defect_orbs.json")
_DEFECT_ORB_ORIGIN = "human_rl:data/mod_information/snapshots/v0.107.1/orbs.json"
_LOCAL_MARKER = "<!-- praxis:local -->"
_MARKUP = re.compile(r"\[/?[A-Za-z_]+(?:=[^\]]+)?\]")
_RESOURCE = re.compile(r"res://\S+?\.png")
_RESOURCE_ICON_RUN = re.compile(
    r"(?:res://images/packed/sprite_fonts/"
    r"(?:colorless|necrobinder|defect|ironclad|regent|silent)_energy_icon\.png"
    r"|res://images/packed/sprite_fonts/star_icon\.png)+"
)
_UNRESOLVED_TEXT = re.compile(r"\{[^{}]+\}|\bTODO\b", re.IGNORECASE)


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
    game_version = _fixed_version(health.game_version)
    destination = Path(output_root) / "mod_export" / game_version
    raw_root = destination / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    categories: dict[str, int] = {}
    expected: dict[str, set[str]] = {}
    gaps: list[str] = []
    for collection in _COLLECTIONS:
        entities = client.data_collection(collection)
        (raw_root / f"{collection}.json").write_text(
            json.dumps(entities, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        expected[collection] = set()
        for entity in entities:
            gaps.extend(_entity_gaps(collection, curate_entity(collection, entity)))
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
        gaps=sorted(set(gaps)),
    )


def rebuild_mod_knowledge(
    raw_root: Path,
    output_root: Path,
    *,
    wiki_root: Path | None = None,
    cycles_root: Path | None = None,
    event_entries_root: Path | None = None,
) -> KnowledgeBuildResult:
    """从固定原始快照离线重建规范事实 Markdown。

    Wiki 只允许补充怪物的招式与循环；实跳 cycle 只作为待复核观察附在
    正文中。两者都不会取代 Mod 导出的基础 HP 和身份。

    Args:
        raw_root (Path): ``mod_export/v0.107.1/raw`` 目录。
        output_root (Path): ``data/game_knowledge`` 根目录。
        wiki_root (Path | None): 可选的 Web Wiki 根目录。
        cycles_root (Path | None): 可选的 human-rl cycle 记录目录。
        event_entries_root (Path | None): 可选的固定版本实机事件 UI 快照目录。

    Raises:
        ValueError: 原始快照不属于固定版本。
        TypeError: 原始集合不是 JSON 数组。
        KnowledgeFormatError: 实体、Wiki 或补充快照不满足知识格式契约。
        OSError: 快照或补充来源不可读、产物不可写。

    Returns:
        KnowledgeBuildResult: 重建目录、条目数量与类别计数。
    """
    raw_root = Path(raw_root)
    if raw_root.name != "raw" or raw_root.parent.name != _FIXED_GAME_VERSION:
        raise ValueError(f"离线重建只接受 {_FIXED_GAME_VERSION}/raw")
    acts_path = raw_root / "acts.json"
    if not acts_path.is_file():
        raise KnowledgeFormatError(f"固定版本重建缺少地图集合: {acts_path}")
    destination = Path(output_root) / "mod_export" / _FIXED_GAME_VERSION
    observations = _load_cycle_observations(cycles_root)
    event_snapshots = _load_event_snapshots(event_entries_root)
    reference_names = _load_reference_names(raw_root)
    orb_snapshot_path = Path(output_root) / _DEFECT_ORB_SUPPLEMENT
    defect_orbs: list[dict[str, Any]] = []
    defect_orb_metadata: dict[str, str] = {}
    if orb_snapshot_path.is_file():
        defect_orbs, defect_orb_metadata = _load_defect_orbs(orb_snapshot_path)
    categories: dict[str, int] = {}
    expected: dict[str, set[str]] = {}
    gaps: list[str] = []
    for collection in _COLLECTIONS:
        path = raw_root / f"{collection}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"原始集合必须是 JSON 数组: {path}")
        entities = [dict(row) for row in payload if isinstance(row, Mapping)]
        if collection == "acts":
            _validate_required_acts(entities, path)
        elif collection == "monsters":
            entities = [
                _supplement_monster(entity, wiki_root, observations)
                for entity in entities
            ]
        elif collection == "events":
            entities = [
                _supplement_event(entity, event_snapshots) for entity in entities
            ]
        elif collection == "characters":
            entities = [
                _supplement_character(
                    entity,
                    reference_names,
                    defect_orbs,
                    defect_orb_metadata,
                )
                for entity in entities
            ]
        expected[collection] = set()
        for entity in entities:
            gaps.extend(_entity_gaps(collection, curate_entity(collection, entity)))
            entry = _mod_entry(collection, entity, _FIXED_GAME_VERSION)
            target = destination / collection / f"{_safe_id(entry.object_id)}.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(entry.to_markdown(), encoding="utf-8")
            expected[collection].add(target.stem)
        categories[collection] = len(entities)
    _remove_stale_entries(destination, expected)
    supplement_sources = []
    if wiki_root is not None:
        supplement_sources.append("web_wiki:monster_moves_cycles")
    if cycles_root is not None:
        supplement_sources.append("human_rl_cycle_lab:pending_observation")
    if event_entries_root is not None:
        supplement_sources.append("game_ui_snapshot:resolved_options")
    if defect_orbs:
        supplement_sources.append(_DEFECT_ORB_SUPPLEMENT.as_posix())
    return _write_result(
        destination,
        "mod_export",
        categories,
        game_version=_FIXED_GAME_VERSION,
        gaps=sorted(set(gaps)),
        supplements=supplement_sources,
    )


def _validate_required_acts(
    entities: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    """要求固定版本四张地图及四类遭遇池完整存在。

    Args:
        entities (Sequence[Mapping[str, Any]]): ``acts.json`` 中的地图对象。
        path (Path): 用于错误定位的原始集合路径。

    Raises:
        KnowledgeFormatError: 地图 ID 集合或任一遭遇池不完整。

    Returns:
        None: 四张地图和所有池均可生成知识时返回。
    """
    ids = [entity.get("id") for entity in entities]
    if len(entities) != len(_REQUIRED_ACT_IDS) or set(ids) != set(_REQUIRED_ACT_IDS):
        raise KnowledgeFormatError(
            f"{path} 必须且只能包含固定四张地图: {list(_REQUIRED_ACT_IDS)}"
        )
    for entity in entities:
        act_id = str(entity["id"])
        for field in _REQUIRED_ACT_POOLS:
            pool = entity.get(field)
            if not isinstance(pool, list) or not pool:
                raise KnowledgeFormatError(f"地图 {act_id} 缺少非空 {field}")
            if any(
                not isinstance(encounter, Mapping)
                or not isinstance(encounter.get("id"), str)
                or not str(encounter.get("id")).strip()
                or not isinstance(encounter.get("name"), str)
                or not str(encounter.get("name")).strip()
                for encounter in pool
            ):
                raise KnowledgeFormatError(f"地图 {act_id} 的 {field} 条目无效")


def _load_reference_names(raw_root: Path) -> dict[str, str]:
    """从固定快照读取角色起始卡牌、遗物和药水的显示名称。

    Args:
        raw_root (Path): ``mod_export/v0.107.1/raw`` 目录。

    Raises:
        TypeError: 任一存在的引用集合不是 JSON 数组。
        OSError: 原始集合无法读取。

    Returns:
        dict[str, str]: 稳定 ID 到当前固定版本中文显示名的映射。
    """
    names: dict[str, str] = {}
    for collection in ("cards", "relics", "potions"):
        path = Path(raw_root) / f"{collection}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"原始集合必须是 JSON 数组: {path}")
        for row in payload:
            if not isinstance(row, Mapping):
                continue
            object_id = _clean_text(row.get("id"))
            name = _clean_text(row.get("name"))
            if object_id and name:
                names[object_id] = name
    return names


def _load_defect_orbs(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """读取并严格校验固定版本故障机器人充能球实测快照。

    Args:
        path (Path): 受控 ``defect_orbs.json`` 补录文件。

    Raises:
        KnowledgeFormatError: 版本、球种集合或机制字段不完整。
        OSError: 补录文件无法读取。

    Returns:
        tuple[list[dict[str, Any]], dict[str, str]]: 按固定球种顺序保存的机制
            事实，以及写入角色 frontmatter 的受控来源元数据。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise KnowledgeFormatError(f"充能球快照不是 JSON 对象: {path}")
    if str(payload.get("game_version") or "").removeprefix("v") != "0.107.1":
        raise KnowledgeFormatError(f"充能球快照版本不是 {_FIXED_GAME_VERSION}: {path}")
    supplement_metadata: dict[str, str] = {
        "orb_supplement_source": _DEFECT_ORB_SUPPLEMENT.as_posix(),
        "orb_supplement_origin": _DEFECT_ORB_ORIGIN,
    }
    for source_field, target_field in (
        ("captured_at_utc", "orb_supplement_captured_at"),
        ("mod_version", "orb_supplement_mod_version"),
        ("provenance", "orb_supplement_provenance"),
    ):
        value = payload.get(source_field)
        if not isinstance(value, str) or not value.strip():
            raise KnowledgeFormatError(f"充能球快照缺少字段 {source_field}: {path}")
        supplement_metadata[target_field] = value.strip()
    raw_orbs = payload.get("orbs")
    if not isinstance(raw_orbs, list):
        raise KnowledgeFormatError(f"充能球快照缺少 orbs 数组: {path}")
    by_id: dict[str, dict[str, Any]] = {}
    for raw_orb in raw_orbs:
        if not isinstance(raw_orb, Mapping):
            raise KnowledgeFormatError(f"充能球条目不是对象: {path}")
        orb = dict(raw_orb)
        orb_id = orb.get("id")
        if not isinstance(orb_id, str) or orb_id in by_id:
            raise KnowledgeFormatError(f"充能球 ID 缺失或重复: {orb_id}")
        for field in ("name", "description", "trigger", "focus_effect"):
            if not isinstance(orb.get(field), str) or not str(orb[field]).strip():
                raise KnowledgeFormatError(f"充能球 {orb_id} 缺少字段 {field}")
        if type(orb.get("passive_base")) is not int:
            raise KnowledgeFormatError(f"充能球 {orb_id} 的 passive_base 无效")
        evoke = orb.get("evoke_base")
        if orb_id == "DARK_ORB" and evoke is not None:
            raise KnowledgeFormatError(f"充能球 {orb_id} 的 evoke_base 无效")
        if orb_id != "DARK_ORB" and type(evoke) is not int:
            raise KnowledgeFormatError(f"充能球 {orb_id} 的 evoke_base 无效")
        if orb["trigger"] not in {"turn_end", "turn_start"}:
            raise KnowledgeFormatError(f"充能球 {orb_id} 的 trigger 无效")
        by_id[orb_id] = orb
    if set(by_id) != set(_DEFECT_ORB_IDS):
        raise KnowledgeFormatError(
            f"充能球快照必须且只能包含固定五种球: {list(_DEFECT_ORB_IDS)}"
        )
    return [by_id[orb_id] for orb_id in _DEFECT_ORB_IDS], supplement_metadata


def _supplement_character(
    entity: dict[str, Any],
    reference_names: Mapping[str, str],
    defect_orbs: Sequence[Mapping[str, Any]],
    defect_orb_metadata: Mapping[str, str],
) -> dict[str, Any]:
    """解析角色起始物品名称，并只给故障机器人附加充能球事实。

    Args:
        entity (dict[str, Any]): Mod 导出的角色实体。
        reference_names (Mapping[str, str]): 起始物品 ID 到显示名的映射。
        defect_orbs (Sequence[Mapping[str, Any]]): 固定版本五种充能球实测事实。
        defect_orb_metadata (Mapping[str, str]): 补充文件、上游来源与采样元数据。

    Raises:
        KnowledgeFormatError: 故障机器人缺少补充文件或起始物品显示名。

    Returns:
        dict[str, Any]: 可直接渲染、仍保留稳定 ID 的角色实体副本。
    """
    result = dict(entity)
    is_defect = entity.get("id") == "DEFECT"
    for field in ("starting_deck", "starting_relics", "starting_potions"):
        values = entity.get(field)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        unresolved = [
            value
            for value in values
            if isinstance(value, str) and value and value not in reference_names
        ]
        if is_defect and unresolved:
            raise KnowledgeFormatError(
                f"故障机器人 {field} 无法解析显示名: {unresolved}"
            )
        result[field] = [
            {"id": value, "name": reference_names.get(value, value)}
            for value in values
            if isinstance(value, str) and value
        ]
    if is_defect and not defect_orbs:
        raise KnowledgeFormatError(
            f"缺少故障机器人充能球补充: {_DEFECT_ORB_SUPPLEMENT.as_posix()}"
        )
    if is_defect:
        result["orbs"] = [dict(orb) for orb in defect_orbs]
        result.update(defect_orb_metadata)
    return result


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
    entity = curate_entity(collection, entity)
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
    if entity.get("is_x_star_cost") is True:
        metadata["star_cost"] = "X"
    elif entity.get("star_cost") is not None:
        metadata["star_cost"] = str(entity["star_cost"])
    description = _clean_text(entity.get("description"))
    lines = ["## 效果", description or "（缺少已解析效果数据）"]
    upgrade = entity.get("upgrade")
    if isinstance(upgrade, Mapping):
        changes: list[str] = []
        base_cost = _display_cost(entity.get("cost"), entity.get("is_x_cost"))
        upgrade_cost = _display_cost(
            upgrade.get("cost"),
            upgrade.get("is_x_cost"),
        )
        if upgrade_cost and upgrade_cost != base_cost:
            changes.append(f"- 费用：{base_cost} → {upgrade_cost}")
        base_star_cost = _display_cost(
            entity.get("star_cost"),
            entity.get("is_x_star_cost"),
        )
        upgrade_star_cost = _display_cost(
            upgrade.get("star_cost"),
            upgrade.get("is_x_star_cost"),
        )
        if upgrade_star_cost and upgrade_star_cost != base_star_cost:
            changes.append(f"- 星能费用：{base_star_cost} → {upgrade_star_cost}")
        upgrade_description = _clean_text(upgrade.get("description"))
        if upgrade_description and upgrade_description != description:
            changes.append(f"- 效果：{upgrade_description}")
        if changes:
            lines.extend(("## 升级", *changes))
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


def _render_enchantment(entity: Mapping[str, Any], metadata: dict[str, str]) -> str:
    """渲染附魔的单层效果、可叠加性与卡面附加文本。

    Args:
        entity (Mapping[str, Any]): Mod 导出的附魔实体。
        metadata (dict[str, str]): 将写入 Markdown frontmatter 的基础字段。

    Returns:
        str: 包含附魔属性、效果和可选卡面附加文本的 Markdown。
    """
    metadata.update(
        _scalar_fields(
            entity,
            {
                "is_stackable": "stackable",
                "show_amount": "show_amount",
                "sample_amount": "sample_amount",
            },
        )
    )
    lines = ["## 效果", _clean_text(entity.get("description")) or "——"]
    extra = _clean_text(entity.get("extra_card_text"))
    if extra:
        lines.extend(("", "## 卡面附加", extra))
    return "\n".join(lines)


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
            {
                "type": "power_type",
                "stack_type": "stack_type",
                "sample_amount": "sample_amount",
                "uses_amount": "uses_amount",
            },
        )
    )
    return f"## 效果\n{_clean_text(entity.get('description')) or '——'}"


def _render_character(entity: Mapping[str, Any], metadata: dict[str, str]) -> str:
    """渲染角色初始属性、物品、简介和可选充能球机制。

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
                "orb_supplement_source": "supplement_source",
                "orb_supplement_origin": "supplement_origin",
                "orb_supplement_captured_at": "supplement_captured_at",
                "orb_supplement_mod_version": "supplement_mod_version",
            },
        )
    )
    lines = ["## 起始牌组", *_named_reference_bullets(entity.get("starting_deck"))]
    lines.extend(
        ("", "## 起始遗物", *_named_reference_bullets(entity.get("starting_relics")))
    )
    potions = _named_reference_bullets(entity.get("starting_potions"))
    if potions:
        lines.extend(("", "## 起始药水", *potions))
    description = _clean_text(entity.get("description"))
    if description:
        lines.extend(("", "## 简介", description))
    orbs = entity.get("orbs")
    if isinstance(orbs, Sequence) and not isinstance(orbs, (str, bytes)):
        orb_rows = [orb for orb in orbs if isinstance(orb, Mapping)]
        if orb_rows:
            lines.extend(("", "## 充能球", *_named_reference_bullets(orb_rows)))
            for orb in orb_rows:
                name = _clean_text(orb.get("name"))
                description = _clean_text(orb.get("description"))
                if not name or not description:
                    continue
                lines.extend(
                    (
                        "",
                        f"## 充能球：{name}",
                        f"- 游戏描述：{description}",
                        f"- 基础数值与集中：{_orb_mechanics(orb)}",
                    )
                )
    return "\n".join(lines)


def _orb_mechanics(orb: Mapping[str, Any]) -> str:
    """把充能球被动、激发与集中关系渲染为中文短句。

    Args:
        orb (Mapping[str, Any]): 已通过固定版本校验的充能球事实。

    Returns:
        str: 不把未知黑暗激发基值伪装成零的机制说明。
    """
    trigger = {
        "turn_end": "回合结束",
        "turn_start": "回合开始",
    }[str(orb["trigger"])]
    values = [f"被动（{trigger}）{orb['passive_base']}"]
    if orb.get("evoke_base") is not None:
        values.append(f"激发{orb['evoke_base']}")
    focus = _clean_text(orb.get("focus_effect")).replace(",", "，")
    if focus and focus[-1] not in "。！？":
        focus += "。"
    return f"{'，'.join(values)}。{focus}"


def _render_act(entity: Mapping[str, Any], metadata: dict[str, str]) -> str:
    """渲染地图的弱、常规、精英和 Boss 遭遇池。

    Args:
        entity (Mapping[str, Any]): Mod 导出的地图实体。
        metadata (dict[str, str]): 将写入 frontmatter 的公共字段。

    Returns:
        str: 可区分前期弱池与常规池的地图 Markdown 正文。
    """
    metadata.update(
        _scalar_fields(
            entity,
            {
                "index": "index",
                "is_default": "is_default",
            },
        )
    )
    sections = (
        ("弱遭遇池", "weak_encounters"),
        ("常规遭遇池", "regular_encounters"),
        ("精英遭遇池", "elite_encounters"),
        ("Boss 遭遇池", "boss_encounters"),
    )
    lines: list[str] = []
    for title, field in sections:
        entries = _named_reference_bullets(entity.get(field))
        if lines:
            lines.append("")
        lines.extend((f"## {title}", *(entries or ["（空）"])))
    return "\n".join(lines)


def _render_monster(entity: Mapping[str, Any], metadata: dict[str, str]) -> str:
    """渲染敌人基础属性、关联遭遇、招式、循环与实跳观察。

    Args:
        entity (Mapping[str, Any]): Mod 导出的敌人实体。
        metadata (dict[str, str]): 将写入 frontmatter 的公共字段。

    Returns:
        str: 敌人 Markdown 正文。
    """
    metadata.update(
        _scalar_fields(
            entity,
            {
                "type": "room",
                "min_hp": "min_hp",
                "max_hp": "max_hp",
                "act": "act",
                "hp_ascension": "hp_ascension",
                "supplement_source": "supplement_source",
            },
        )
    )
    for source, target in (("innate", "innate"), ("encounters", "encounters")):
        value = entity.get(source)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            metadata[target] = ",".join(str(item) for item in value)
    supplement_body = entity.get("supplement_body")
    if isinstance(supplement_body, str) and supplement_body.strip():
        lines = [supplement_body.strip()]
    else:
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
            lines.append("（缺少已解析招式数据）")
    acts = entity.get("acts")
    if isinstance(acts, Sequence) and not isinstance(acts, (str, bytes)):
        act_lines = []
        for act in acts:
            if not isinstance(act, Mapping):
                continue
            act_id = _clean_text(act.get("id"))
            act_name = _clean_text(act.get("name"))
            act_index = act.get("index")
            if act_id and act_name and isinstance(act_index, int):
                act_lines.append(f"- Act {act_index}：{act_name}（{act_id}）")
        if act_lines:
            lines.extend(("", "## 所在 Act", *act_lines))
    encounter_ids = _bullet_ids(entity.get("encounters"))
    if encounter_ids:
        lines.extend(("", "## 可能关联的遭遇", *encounter_ids))
    observations = entity.get("cycle_observations")
    if isinstance(observations, Sequence) and observations:
        lines.extend(("", "## 实跳观察（待复核）"))
        for observation in observations:
            if isinstance(observation, str) and observation:
                lines.append(f"- {observation}")
    return "\n".join(lines)


def _render_encounter(entity: Mapping[str, Any], metadata: dict[str, str]) -> str:
    """渲染遭遇属性和全部可能出现的敌人。

    Args:
        entity (Mapping[str, Any]): Mod 导出的遭遇实体。
        metadata (dict[str, str]): 将写入 frontmatter 的公共字段。

    Returns:
        str: 遭遇 Markdown 正文。
    """
    metadata.update(
        _scalar_fields(
            entity,
            {
                "room_type": "room",
                "is_weak": "is_weak",
                "is_debug": "is_debug",
                "should_give_rewards": "should_give_rewards",
                "monster_list_kind": "monster_list_kind",
            },
        )
    )
    lines = [
        "## 可能出现的敌人类型",
        "以下为去重后的可能类型，不表示同时出现或数量。",
    ]
    monsters = entity.get("monsters")
    if isinstance(monsters, Sequence) and not isinstance(monsters, (str, bytes)):
        for monster in monsters:
            if not isinstance(monster, Mapping):
                continue
            monster_id = _clean_text(monster.get("id"))
            name = _clean_text(monster.get("name"))
            if monster_id or name:
                lines.append(f"- {name or '未知敌人'}（{monster_id or '?'}）")
    if len(lines) == 2:
        lines.append("（缺少已解析敌人类型）")
    return "\n".join(lines)


def _render_event(entity: Mapping[str, Any], metadata: dict[str, str]) -> str:
    """渲染事件初始文本与可见选项。

    Args:
        entity (Mapping[str, Any]): Mod 导出的事件实体。
        metadata (dict[str, str]): 将写入 frontmatter 的公共字段。

    Returns:
        str: 事件 Markdown 正文。
    """
    metadata.update(
        _scalar_fields(
            entity,
            {
                "type": "event_type",
                "act": "act",
                "supplement_source": "supplement_source",
                "snapshot_scope": "snapshot_scope",
            },
        )
    )
    lines = ["## 文本", _clean_text(entity.get("description")) or "——", "## 选项"]
    options = entity.get("options")
    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
        for option in options:
            if not isinstance(option, Mapping):
                continue
            option_id = _clean_text(option.get("id"))
            title = _clean_text(option.get("title"))
            description = _clean_text(option.get("description"))
            if not option_id or not title or not description:
                continue
            lines.append(f"- [{option_id}] {title} — {description}")
    if lines[-1] == "## 选项":
        lines.append("（缺少已解析选项数据）")
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
    "acts": _render_act,
    "cards": _render_card,
    "relics": _render_relic,
    "potions": _render_potion,
    "enchantments": _render_enchantment,
    "powers": _render_power,
    "characters": _render_character,
    "monsters": _render_monster,
    "encounters": _render_encounter,
    "events": _render_event,
    "keywords": _render_keyword,
}


def _named_reference_bullets(value: object) -> list[str]:
    """把带 ID 与显示名称的引用数组渲染成项目统一列表。

    Args:
        value (object): Mod 导出的引用对象数组。

    Returns:
        list[str]: ``- 名称（ID）`` 形式的 Markdown 列表。
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        object_id = _clean_text(item.get("id"))
        name = _clean_text(item.get("name"))
        if object_id and name:
            result.append(f"- {name}（{object_id}）")
    return result


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
    if _UNRESOLVED_TEXT.search(value):
        return ""
    text = _MARKUP.sub("", value)

    def replace_icon_run(match: re.Match[str]) -> str:
        """把连续能量或星能资源图标转换为可读费用文本。

        Args:
            match (re.Match[str]): 已知资源图标的连续匹配。

        Returns:
            str: 与相邻数字或图标数量对应的中文费用单位。
        """
        raw = match.group(0)
        unit = "星能" if "star_icon" in raw else "能量"
        previous = text[match.start() - 1] if match.start() else ""
        if previous.isdigit() or previous in {"X", "x"}:
            return f"点{unit}"
        return f"{raw.count('res://')}点{unit}"

    text = _RESOURCE_ICON_RUN.sub(replace_icon_run, text)
    if _RESOURCE.search(text):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _display_cost(value: object, is_x_cost: object) -> str:
    """把普通或 X 费用渲染成稳定的短文本。

    Args:
        value (object): 数字费用或空值。
        is_x_cost (object): 是否为 X 费用。

    Returns:
        str: ``X``、数字字符串或空串。
    """
    if is_x_cost is True:
        return "X"
    return "" if value is None else str(value)


def _entity_gaps(collection: str, entity: Mapping[str, Any]) -> list[str]:
    """收集实体中不能安全写入知识的未解析文本字段。

    Args:
        collection (str): 实体类别。
        entity (Mapping[str, Any]): Mod 原始实体。

    Returns:
        list[str]: 可供 manifest 和审计报告使用的稳定 gap 标识。
    """
    object_id = str(entity.get("id") or "?")
    gaps: list[str] = []
    # description_raw 是保留给调试的本地化模板，天然含 {Damage} 等占位符；
    # 只要 Mod 已给出可用的 resolved description，就不应把整批对象误报为缺口。
    for field in ("description",):
        value = entity.get(field)
        if isinstance(value, str) and (
            _UNRESOLVED_TEXT.search(value) or _unknown_resource(value)
        ):
            gaps.append(f"{collection}:{object_id}:{field}:unresolved_text")
    options = entity.get("options")
    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
        for index, option in enumerate(options):
            if not isinstance(option, Mapping):
                continue
            option_id = str(option.get("id") or index)
            for field in ("title", "description"):
                value = option.get(field)
                if isinstance(value, str) and (
                    _UNRESOLVED_TEXT.search(value) or _unknown_resource(value)
                ):
                    gaps.append(
                        f"{collection}:{object_id}:options.{option_id}.{field}:"
                        "unresolved_text"
                    )
    return gaps


def _unknown_resource(value: str) -> bool:
    """判断文本是否含有清洗器尚不理解的资源图标。

    Args:
        value (str): 游戏富文本。

    Returns:
        bool: 已知能量/星能图标替换后仍有资源路径时为真。
    """
    return bool(_RESOURCE.search(_RESOURCE_ICON_RUN.sub("", value)))


def _supplement_monster(
    entity: dict[str, Any],
    wiki_root: Path | None,
    observations: Mapping[str, list[str]],
) -> dict[str, Any]:
    """用受控 Wiki 字段和实跳观察补充单个 Mod 怪物。

    Args:
        entity (dict[str, Any]): Mod 导出的怪物实体副本。
        wiki_root (Path | None): 只允许读取招式与循环章节的 Wiki 根目录。
        observations (Mapping[str, list[str]]): 按怪物显示名称索引的实跳摘要。

    Raises:
        KnowledgeFormatError: Wiki 条目的实体 ID 或来源与目标怪物不匹配。
        OSError: Wiki 文件存在但无法读取。

    Returns:
        dict[str, Any]: 附带受控正文和逐项来源的怪物实体。
    """
    object_id = str(entity.get("id") or "")
    name = str(entity.get("name") or "")
    sources: list[str] = []
    if wiki_root is not None:
        wiki_path = Path(wiki_root) / "monsters" / f"{object_id}.md"
        if wiki_path.is_file():
            wiki = parse_knowledge_entry(wiki_path.read_text(encoding="utf-8"))
            if wiki.object_id != object_id or wiki.source != "web_wiki":
                raise KnowledgeFormatError(f"怪物 Wiki 身份或来源不匹配: {wiki_path}")
            sections = []
            for heading in ("招式", "循环"):
                body = _extract_markdown_section(wiki.body, heading)
                if body:
                    sections.extend((f"## {heading}", body))
            if sections:
                entity["supplement_body"] = "\n".join(sections)
                sources.append("web_wiki:monster_moves_cycles")
    if name in observations:
        entity["cycle_observations"] = observations[name]
        sources.append("human_rl_cycle_lab:pending_observation")
    if sources:
        entity["supplement_source"] = ";".join(sources)
    return entity


def _extract_markdown_section(body: str, heading: str) -> str:
    """提取受控 Wiki 正文中的单个二级章节。

    Args:
        body (str): 不含 frontmatter 的 Markdown 正文。
        heading (str): 不含 ``##`` 前缀的二级标题。

    Returns:
        str: 去除首尾空白的章节正文；标题不存在时返回空串。
    """
    match = re.search(
        rf"(?ms)^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)",
        body,
    )
    return "" if match is None else match.group(1).strip()


def _load_cycle_observations(
    cycles_root: Path | None,
) -> dict[str, list[str]]:
    """把 cycle_lab 记录压成不冒充完整循环的观察摘要。

    Args:
        cycles_root (Path | None): cycle_lab JSON 记录目录；空值表示不加载。

    Returns:
        dict[str, list[str]]: 按怪物显示名称索引的待复核观察摘要。
    """
    if cycles_root is None:
        return {}
    result: dict[str, list[str]] = {}
    for path in sorted(Path(cycles_root).glob("*.json")):
        if path.name == "manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        encounter_id = str(payload.get("encounter_id") or path.stem)
        status = str(payload.get("status") or "unknown")
        for monster in payload.get("monsters") or []:
            if not isinstance(monster, Mapping):
                continue
            name = str(monster.get("name") or "")
            if not name:
                continue
            for band_name, band in (monster.get("bands") or {}).items():
                if not isinstance(band, Mapping):
                    continue
                moves = []
                for move in band.get("moves") or []:
                    if not isinstance(move, Mapping):
                        continue
                    move_id = str(move.get("move") or "?")
                    damage = move.get("damage")
                    hits = move.get("hits")
                    if damage is not None and hits is not None:
                        moves.append(f"{move_id}：{damage}×{hits}")
                    else:
                        intent = str(move.get("intent_type") or "未知意图")
                        moves.append(f"{move_id}：{intent}")
                turns = band.get("turns_observed")
                closed = "已闭环" if band.get("closed") is True else "未闭环"
                summary = (
                    f"{encounter_id}/{band_name}（{turns}回合，{closed}，{status}）："
                    + " → ".join(moves)
                )
                result.setdefault(name, []).append(summary)
    return result


def _load_event_snapshots(
    event_entries_root: Path | None,
) -> dict[str, dict[str, Any]]:
    """读取已经真正进入事件页面后记录的已解析 UI 文本。

    Args:
        event_entries_root (Path | None): 固定版本事件快照目录；空值表示不加载。

    Returns:
        dict[str, dict[str, Any]]: 按事件 ID 索引的 UI 事件载荷，包含快照文件名。
    """
    if event_entries_root is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    event_entries_root = Path(event_entries_root)
    roots = [event_entries_root]
    sibling_ancients = event_entries_root.parent / "ancients"
    if sibling_ancients.is_dir():
        roots.append(sibling_ancients)
    paths = sorted(path for root in roots for path in root.glob("*.json"))
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = payload.get("state") if isinstance(payload, Mapping) else None
        event = state.get("event") if isinstance(state, Mapping) else None
        if not isinstance(event, Mapping):
            continue
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            continue
        result[event_id] = {**event, "_snapshot_file": path.name}
    return result


def _supplement_event(
    entity: dict[str, Any],
    snapshots: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """以同版本实机 UI 快照替换含运行期变量的初始事件选项。

    Args:
        entity (dict[str, Any]): Mod 导出的事件实体副本。
        snapshots (Mapping[str, Mapping[str, Any]]): 按事件 ID 索引的 UI 快照。

    Returns:
        dict[str, Any]: 已补入可见文本、来源及可选状态范围标记的事件实体。
    """
    object_id = str(entity.get("id") or "")
    snapshot = snapshots.get(object_id)
    if snapshot is None:
        return entity
    raw_description = entity.get("description")
    description = snapshot.get("description")
    if isinstance(description, str) and description.strip():
        entity["description"] = description
    raw_options = entity.get("options")
    fallback_ids = [
        str(option.get("id") or index)
        for index, option in enumerate(raw_options or [])
        if isinstance(option, Mapping)
    ]
    resolved_options: list[dict[str, str]] = []
    for index, option in enumerate(snapshot.get("options") or []):
        if not isinstance(option, Mapping):
            continue
        text_key = str(option.get("text_key") or "")
        option_id = text_key.rpartition(".options.")[2]
        if not option_id and index < len(fallback_ids):
            option_id = fallback_ids[index]
        title = option.get("title")
        option_description = option.get("description")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (option_id, title, option_description)
        ):
            continue
        resolved_options.append(
            {
                "id": option_id,
                "title": str(title),
                "description": str(option_description),
            }
        )
    if resolved_options:
        entity["options"] = resolved_options
        entity["supplement_source"] = "game_ui_snapshot:" + str(
            snapshot.get("_snapshot_file") or ""
        )
        if _event_snapshot_is_state_dependent(
            raw_description, raw_options, resolved_options
        ):
            entity["snapshot_scope"] = "single_state"
    return entity


def _event_snapshot_is_state_dependent(
    raw_description: object,
    raw_options: object,
    resolved_options: Sequence[Mapping[str, str]],
) -> bool:
    """判断一次实机事件快照能否安全代表固定版本的通用选项。

    Args:
        raw_description (object): Mod 导出的初始事件描述模板。
        raw_options (object): Mod 导出的初始事件选项模板。
        resolved_options (Sequence[Mapping[str, str]]): 实机快照中的完整选项。

    Returns:
        bool: 模板集合不同或原始文本含运行期变量时为真。
    """
    if not isinstance(raw_options, Sequence) or isinstance(raw_options, (str, bytes)):
        return True
    raw_rows = [option for option in raw_options if isinstance(option, Mapping)]
    if not raw_rows:
        return True
    raw_ids = {str(option.get("id") or "") for option in raw_rows}
    resolved_ids = {str(option.get("id") or "") for option in resolved_options}
    if raw_ids != resolved_ids:
        return True
    candidate_texts = [raw_description]
    candidate_texts.extend(
        option.get(field) for option in raw_rows for field in ("title", "description")
    )
    return any(
        isinstance(value, str)
        and (_UNRESOLVED_TEXT.search(value) or _unknown_resource(value))
        for value in candidate_texts
    )


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


def _fixed_version(game_version: str) -> str:
    """规范化并锁定本项目唯一允许的游戏知识版本。

    Args:
        game_version (str): Mod 健康检查返回的版本。

    Raises:
        KnowledgeFormatError: 游戏版本不是固定的 ``v0.107.1``。

    Returns:
        str: 固定目录名 ``v0.107.1``。
    """
    safe = _safe_version(game_version)
    normalized = safe if safe.startswith("v") else f"v{safe}"
    if normalized != _FIXED_GAME_VERSION:
        raise KnowledgeFormatError(
            f"知识导出只接受固定游戏版本 {_FIXED_GAME_VERSION}，实际为 {game_version}"
        )
    return normalized


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
    gaps: Sequence[str] = (),
    supplements: Sequence[str] = (),
) -> KnowledgeBuildResult:
    """写入最小知识清单并返回构建结果。

    Args:
        destination (Path): 当前来源的知识根目录。
        source (str): 所有条目共享的来源类别。
        categories (dict[str, int]): 各类别的实体数量。
        game_version (str | None): Mod 实测来源的可选游戏版本。
        gaps (Sequence[str]): 未解析但保留在原始快照中的字段记录。
        supplements (Sequence[str]): 受控补充来源的简短说明。

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
    if gaps:
        manifest["gaps"] = list(gaps)
    if supplements:
        manifest["supplements"] = list(supplements)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return KnowledgeBuildResult(destination, entry_count, categories)
