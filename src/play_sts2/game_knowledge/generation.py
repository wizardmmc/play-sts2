"""从固定版本规范事实 Markdown 生成多问法知识和人工审计报告。"""

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .curation import FIXED_GAME_VERSION, curate_entry, is_question_excluded
from .markdown import KnowledgeEntry, parse_knowledge_entry
from .pipeline import KnowledgeBuildResult

_FIXED_VERSION = FIXED_GAME_VERSION
_REQUIRED_MAP_IDS = ("OVERGROWTH", "UNDERDOCKS", "HIVE", "GLORY")
_BAD_TEXT = re.compile(r"\{[^{}]+\}|\bTODO\b|res://", re.IGNORECASE)
_INCOMPLETE_MARKERS = (
    "缺少已解析",
    "——",
    "（循环未知）",
    "Unknown",
    "None",
    "？？？？",
    "????",
    "分支 None",
)
_TYPE_NAMES = {
    "Attack": "攻击",
    "Skill": "技能",
    "Power": "能力",
    "Status": "状态",
    "Curse": "诅咒",
    "Quest": "任务",
}
_RARITY_NAMES = {
    "Basic": "基础",
    "Common": "普通",
    "Uncommon": "罕见",
    "Rare": "稀有",
    "Ancient": "先古",
    "Status": "状态",
    "Curse": "诅咒",
    "Quest": "任务",
    "Event": "事件",
    "Token": "衍生",
    "Shop": "商店",
    "Starter": "起始",
}
_CHARACTER_NAMES = {
    "ironclad": "铁甲战士",
    "silent": "静默猎手",
    "defect": "故障机器人",
    "necrobinder": "亡灵契约师",
    "regent": "储君",
    "colorless": "无色",
    "status": "",
    "curse": "",
    "quest": "",
}
_TARGET_NAMES = {
    "Self": "自身",
    "AnyEnemy": "任一敌人",
    "RandomEnemy": "随机敌人",
    "AllEnemies": "全体敌人",
    "AnyAlly": "任一友方",
    "AllAllies": "所有友方",
    "AnyPlayer": "任一玩家",
    "TargetedNoCreature": "无需生物目标",
    "None": "无",
}
_POWER_TYPE_NAMES = {"Buff": "增益", "Debuff": "减益"}
_POWER_STACK_NAMES = {"Counter": "计数叠加", "Single": "单实例", "None": ""}
_ROOM_NAMES = {"Normal": "普通", "Elite": "精英", "Boss": "首领"}
_POTION_USAGE_NAMES = {
    "AnyTime": "任意时机",
    "Automatic": "自动触发",
    "CombatOnly": "仅战斗中",
}


def generate_question_variants(
    snapshot_root: Path,
    output_root: Path,
) -> KnowledgeBuildResult:
    """把固定版本单实体知识展开成可重建的多问法 JSONL。

    该函数只消费 ``mod_export/v0.107.1``，不会读取 ``web_wiki``，也不会
    构建训练、验证或测试分卷。

    Args:
        snapshot_root (Path): 固定版本规范事实 Markdown 根目录。
        output_root (Path): 按类别和实体写入 JSONL 的目标目录。

    Raises:
        ValueError: 输入不是固定版本、含 Web Wiki 条目或生成了冲突问法。
        OSError: 无法读取输入或写入输出。

    Returns:
        KnowledgeBuildResult: 输出目录、问答总数和各类别问答数量。
    """
    snapshot_root = Path(snapshot_root)
    if snapshot_root.name != _FIXED_VERSION:
        raise ValueError(f"多问法生成只接受固定版本 {_FIXED_VERSION}")
    output_root = Path(output_root)
    _validate_map_catalog(snapshot_root)
    categories: dict[str, int] = {}
    expected: dict[str, set[str]] = {}
    seen_prompts: dict[str, str] = {}
    skipped: list[str] = []

    for category_dir in sorted(
        path for path in snapshot_root.iterdir() if path.is_dir()
    ):
        if category_dir.name == "raw":
            continue
        category = category_dir.name
        output_category = "encounters" if category == "acts" else category
        expected.setdefault(category, set())
        expected.setdefault(output_category, set())
        entries = [
            parse_knowledge_entry(path.read_text(encoding="utf-8"))
            for path in sorted(category_dir.glob("*.md"))
        ]
        name_counts: dict[str, int] = {}
        for entry in entries:
            name_counts[entry.name] = name_counts.get(entry.name, 0) + 1
        for entry in entries:
            if entry.source != "mod_export":
                raise ValueError(
                    f"多问法输入只能来自 mod_export: {category}/{entry.object_id}"
                )
            if is_question_excluded(category, entry.object_id):
                skipped.append(f"{category}:{entry.object_id}:curated_exclusion")
                continue
            patched = curate_entry(category, entry)
            if name_counts[patched.name] > 1:
                patched = _disambiguate_entry(category, patched)
            if (
                category == "events"
                and patched.metadata.get("snapshot_scope") == "single_state"
            ):
                skipped.append(f"{category}:{entry.object_id}:state_dependent_snapshot")
                continue
            rows = _rows_for_entry(category, patched)
            if not rows:
                skipped.append(f"{category}:{entry.object_id}:no_verified_facts")
                continue
            if output_category != category:
                for row in rows:
                    row["category"] = output_category
            for row in rows:
                prompt = str(row["prompt"])
                completion = str(row["completion"])
                previous = seen_prompts.get(prompt)
                if previous is not None and previous != completion:
                    raise ValueError(f"相同问题存在不同答案: {prompt}")
                seen_prompts[prompt] = completion
            target = output_root / output_category / f"{entry.object_id}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            expected[output_category].add(target.name)
            categories[output_category] = categories.get(output_category, 0) + len(rows)

    _remove_stale_generated(output_root, expected)
    entry_count = sum(categories.values())
    manifest = {
        "game_version": _FIXED_VERSION,
        "source": "mod_export + curated_override",
        "samples": entry_count,
        "categories": categories,
        "skipped": skipped,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "_knowledge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return KnowledgeBuildResult(output_root, entry_count, categories)


def _validate_map_catalog(snapshot_root: Path) -> None:
    """在写产物前验证四张地图及四类怪池完整存在。

    生产生成器必须始终看到固定版本完整地图目录，防止局部输入在 stale 清理时
    删除旧的有效地图文件。

    Args:
        snapshot_root (Path): 固定版本规范事实 Markdown 根目录。

    Raises:
        ValueError: 地图 ID、来源、怪池或生成问法数量不符合固定契约。

    Returns:
        None: 四张地图及其怪池均完整时返回。
    """
    acts_root = Path(snapshot_root) / "acts"
    if not acts_root.is_dir():
        raise ValueError(
            f"地图知识必须且只能包含固定四张地图: {list(_REQUIRED_MAP_IDS)}"
        )
    entries = [
        parse_knowledge_entry(path.read_text(encoding="utf-8"))
        for path in sorted(acts_root.glob("*.md"))
    ]
    ids = [entry.object_id for entry in entries]
    if len(entries) != len(_REQUIRED_MAP_IDS) or set(ids) != set(_REQUIRED_MAP_IDS):
        raise ValueError(
            f"地图知识必须且只能包含固定四张地图: {list(_REQUIRED_MAP_IDS)}"
        )
    headings = ("弱遭遇池", "常规遭遇池", "精英遭遇池", "Boss 遭遇池")
    for entry in entries:
        if entry.source != "mod_export":
            raise ValueError(f"地图知识只能来自 mod_export: {entry.object_id}")
        missing = [
            heading for heading in headings if not _section_items(entry.body, heading)
        ]
        if missing:
            raise ValueError(f"地图 {entry.object_id} 缺少怪池: {missing}")
        if len(_map_encounter_questions(entry)) != 8:
            raise ValueError(f"地图 {entry.object_id} 必须生成 8 条怪池问答")


def generate_review_report(snapshot_root: Path, output_path: Path) -> Path:
    """生成便于人工决定补录与保留范围的 Markdown 审计报告。

    Args:
        snapshot_root (Path): 包含 ``raw/*.json`` 的固定版本快照。
        output_path (Path): 报告文件路径。

    Raises:
        OSError: 原始快照无法读取或报告无法写入。
        TypeError: 原始集合不是 JSON 数组。

    Returns:
        Path: 已写入的报告路径。
    """
    snapshot_root = Path(snapshot_root)
    raw_root = snapshot_root / "raw"
    collections: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(raw_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"原始集合必须是 JSON 数组: {path}")
        collections[path.stem] = [row for row in payload if isinstance(row, dict)]

    manifest_path = snapshot_root / "manifest.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            manifest = payload
    manifest_gaps = [
        gap for gap in (manifest or {}).get("gaps", []) if isinstance(gap, str) and gap
    ]
    gap_objects: dict[str, set[str]] = {}
    for gap in manifest_gaps:
        category, separator, remainder = gap.partition(":")
        object_id = remainder.partition(":")[0] if separator else "?"
        gap_objects.setdefault(category, set()).add(object_id)

    lines = [
        f"# 游戏知识审计：{snapshot_root.name}",
        "",
        "## 覆盖概览",
        "",
        "| 类别 | 原始对象 | 缺核心文本/字段 |",
        "| --- | ---: | ---: |",
    ]
    for category, rows in sorted(collections.items()):
        missing = (
            len(gap_objects.get(category, set()))
            if manifest is not None
            else sum(_has_missing_core(category, row) for row in rows)
        )
        lines.append(f"| {category} | {len(rows)} | {missing} |")

    supplements = [
        source
        for source in (manifest or {}).get("supplements", [])
        if isinstance(source, str) and source
    ]
    if supplements:
        lines.extend(("", "## 已控制的补充来源", ""))
        lines.extend(f"- `{source}`" for source in supplements)

    state_dependent_events = []
    event_root = snapshot_root / "events"
    if event_root.is_dir():
        for path in sorted(event_root.glob("*.md")):
            entry = parse_knowledge_entry(path.read_text(encoding="utf-8"))
            if entry.metadata.get("snapshot_scope") == "single_state":
                state_dependent_events.append(entry.object_id)
    if state_dependent_events:
        lines.extend(
            (
                "",
                "## 状态依赖事件快照（不进入通用监督问法）",
                "",
                "一次实机进入只能证明该局可见文本，不能冒充事件的穷尽规则：",
                "",
                "- "
                + "、".join(f"`{object_id}`" for object_id in state_dependent_events),
            )
        )

    lines.extend(("", "## 特殊对象审查", ""))
    special_rows = [
        (category, row)
        for category, rows in sorted(collections.items())
        for row in rows
        if _is_special_object(str(row.get("id") or ""))
    ]
    if not special_rows:
        lines.append("- 无")
    for category, row in special_rows:
        object_id = str(row.get("id") or "?")
        name = str(row.get("name") or "?")
        decision = _special_object_decision(category, row)
        lines.append(f"- `{category}:{object_id}`（{name}）：{decision}")

    lines.extend(("", "## 待补录或补证", ""))
    needs_evidence = []
    if manifest is not None:
        grouped_gaps: dict[tuple[str, str], list[str]] = {}
        for gap in manifest_gaps:
            category, _, remainder = gap.partition(":")
            object_id, _, field = remainder.partition(":")
            grouped_gaps.setdefault((category, object_id), []).append(field)
        for (category, object_id), fields in sorted(grouped_gaps.items()):
            advice = _gap_advice(category, object_id)
            field_list = "、".join(sorted(fields))
            needs_evidence.append(
                f"- `{category}:{object_id}`：{advice}（字段：`{field_list}`）"
            )
    else:
        for category, rows in sorted(collections.items()):
            for row in rows:
                object_id = str(row.get("id") or "?")
                if _has_unresolved(row):
                    needs_evidence.append(f"- `{category}:{object_id}`：含未解析富文本")
                if category == "monsters" and not row.get("moves"):
                    needs_evidence.append(f"- `monsters:{object_id}`：缺少招式/循环")
                if category == "events" and not _has_complete_option(row):
                    needs_evidence.append(f"- `events:{object_id}`：缺少已解析事件选项")
    lines.extend(needs_evidence or ["- 无"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def _gap_advice(category: str, object_id: str) -> str:
    """给已知实机缺口提供最小、可执行的补录条件。

    Args:
        category (str): 缺口所属实体类别。
        object_id (str): 缺口实体的稳定游戏 ID。

    Returns:
        str: 面向人工补录的操作建议。
    """
    if (category, object_id) == ("events", "RELIC_TRADER"):
        return "需携带足够的可交换遗物重新进入事件，记录三组实际交换项"
    if (category, object_id) == ("events", "THE_FUTURE_OF_POTIONS"):
        return "需携带符合条件的药水重新进入事件，记录实际药水与产牌选项"
    return "curated 与实机补充后仍含未解析文本"


def _disambiguate_entry(category: str, entry: KnowledgeEntry) -> KnowledgeEntry:
    """为同名实体添加稳定的人类可读限定，避免冲突问法。

    Args:
        category (str): 实体类别。
        entry (KnowledgeEntry): 名称发生冲突的规范事实条目。

    Returns:
        KnowledgeEntry: 名称附带角色或稳定 ID、正文保持不变的新条目。
    """
    metadata = dict(entry.metadata)
    qualifier = ""
    if category == "cards":
        qualifier = _CHARACTER_NAMES.get(
            metadata.get("character", ""), metadata.get("character", "")
        )
    metadata["name"] = f"{entry.name}（{qualifier or entry.object_id}）"
    return KnowledgeEntry(metadata=metadata, body=entry.body)


def _rows_for_entry(category: str, entry: KnowledgeEntry) -> list[dict[str, str]]:
    """按实体类别生成事实对齐的多问法行。

    Args:
        category (str): 实体类别。
        entry (KnowledgeEntry): 已修正并消歧的规范事实条目。

    Returns:
        list[dict[str, str]]: 去重且不含空问答的训练候选行。
    """
    if _BAD_TEXT.search(entry.body):
        return []
    builders = {
        "acts": _map_encounter_questions,
        "cards": _card_questions,
        "relics": _relic_questions,
        "potions": _potion_questions,
        "enchantments": _enchantment_questions,
        "powers": _power_questions,
        "characters": _character_questions,
        "monsters": _monster_questions,
        "events": _event_questions,
        "keywords": _keyword_questions,
    }
    builder = builders.get(category)
    if builder is None:
        return []
    pairs = builder(entry)
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for question, answer in pairs:
        question = question.strip()
        answer = _normalize_generated_answer(answer)
        if (
            not question
            or not answer
            or _BAD_TEXT.search(answer)
            or any(marker in answer for marker in _INCOMPLETE_MARKERS)
            or (question, answer) in seen
        ):
            continue
        seen.add((question, answer))
        row = {
            "category": category,
            "object_id": entry.object_id,
            "source": "mod_export+curated_override",
            "prompt": f"Q: {question}\nA:",
            "completion": f" {answer}",
        }
        supplement_source = _question_supplement_source(category, entry, question)
        if supplement_source:
            row["supplement_source"] = supplement_source
        rows.append(row)
    return rows


def _normalize_generated_answer(value: str) -> str:
    """移除只对 Markdown 浏览有意义的链接路径，保留可读标签。

    Args:
        value (str): 由规范事实 Markdown 提取的候选答案。

    Returns:
        str: 删除链接目标并压缩空白后的单行答案。
    """
    value = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", value.strip())
    return re.sub(r"\s+", " ", value).strip()


def _question_supplement_source(
    category: str,
    entry: KnowledgeEntry,
    question: str,
) -> str:
    """只给实际消费补充事实的问答附行级来源。

    Args:
        category (str): 当前问答的实体类别。
        entry (KnowledgeEntry): 带补充来源元数据的规范事实条目。
        question (str): 已生成的问题文本。

    Returns:
        str: 当前问题实际使用的补充来源；未使用时返回空串。
    """
    raw = entry.metadata.get("supplement_source", "").strip()
    if category == "characters":
        return raw if "充能球" in question else ""
    if category != "monsters":
        return raw
    if "有哪些招式" not in question and "行动循环" not in question:
        return ""
    sources = [source for source in raw.split(";") if source]
    return ";".join(
        source for source in sources if source == "web_wiki:monster_moves_cycles"
    )


def _card_questions(entry: KnowledgeEntry) -> list[tuple[str, str]]:
    """生成卡牌身份、费用、效果与升级问法。

    Args:
        entry (KnowledgeEntry): 已应用 curated 修正的卡牌条目。

    Returns:
        list[tuple[str, str]]: 保持事实等价的问答对。
    """
    effect = _section(entry.body, "效果")
    if not effect:
        return []
    metadata = entry.metadata
    cost = metadata.get("cost", "?")
    is_unplayable = _is_negative_number(cost)
    cost_text = _card_cost_text(cost, metadata.get("star_cost", ""))
    card_type = _TYPE_NAMES.get(
        metadata.get("card_type", ""), metadata.get("card_type", "")
    )
    rarity = _RARITY_NAMES.get(metadata.get("rarity", ""), metadata.get("rarity", ""))
    character = _CHARACTER_NAMES.get(
        metadata.get("character", ""), metadata.get("character", "")
    )
    target = _TARGET_NAMES.get(metadata.get("target", ""), metadata.get("target", ""))
    identity = f"{'不可打出' if is_unplayable else cost_text}；{card_type}"
    if character or rarity:
        identity += f"({character}{'·' if character and rarity else ''}{rarity})"
    full = f"{identity}。{effect}"
    if target:
        full += f"目标:{target}。"
    if is_unplayable:
        detail_prompt = (
            f"请给出'{entry.name}'的类别、归属和完整效果。"
            if character
            else f"请给出'{entry.name}'的类别和完整效果。"
        )
        rows = [
            (f"'{entry.name}'是什么牌？", full),
            (f"'{entry.name}'能否直接打出？", "不能直接打出。"),
            (detail_prompt, full),
        ]
    else:
        rows = [
            (f"'{entry.name}'是什么牌？", full),
            (
                f"'{entry.name}'在没有外界影响时的费用和效果是什么？",
                f"{cost_text}。{effect}",
            ),
            (f"请给出'{entry.name}'的类别、归属、稀有度、费用和完整效果。", full),
            (
                f"'{entry.name}'这张牌的费用是多少？",
                f"{cost_text}，{card_type}({character}·{rarity})。",
            ),
            (f"打出'{entry.name}'需要几点费用？", f"{cost_text}。"),
        ]
    upgrade = _section(entry.body, "升级")
    if upgrade:
        compact_upgrade = _compact_upgrade(upgrade)
        rows.extend(
            (
                (f"'{entry.name}'升级后有什么变化？", compact_upgrade),
                (
                    f"升级后的'{entry.name}'怎么描述？",
                    f"基础效果:{effect}升级变化:{compact_upgrade}",
                ),
            )
        )
    return rows


def _relic_questions(entry: KnowledgeEntry) -> list[tuple[str, str]]:
    """生成遗物效果和可用稀有度问法。

    Args:
        entry (KnowledgeEntry): 已应用 curated 修正的遗物条目。

    Returns:
        list[tuple[str, str]]: 遗物知识问答对。
    """
    effect = _section(entry.body, "效果")
    if not effect:
        return []
    rarity = _RARITY_NAMES.get(
        entry.metadata.get("rarity", ""), entry.metadata.get("rarity", "")
    )
    if rarity == "None":
        rarity = ""
    full = f"{rarity}。{effect}" if rarity else effect
    rows = [
        (f"遗物'{entry.name}'的效果是什么？", full),
        (f"'{entry.name}'有什么作用？", effect),
    ]
    if rarity:
        rows.extend(
            (
                (f"遗物'{entry.name}'是什么稀有度？", f"{rarity}。"),
                (f"请给出遗物'{entry.name}'的稀有度和完整效果。", full),
            )
        )
    return rows


def _potion_questions(entry: KnowledgeEntry) -> list[tuple[str, str]]:
    """生成药水效果、稀有度、使用时机和目标问法。

    Args:
        entry (KnowledgeEntry): 已应用 curated 修正的药水条目。

    Returns:
        list[tuple[str, str]]: 药水知识问答对。
    """
    effect = _section(entry.body, "效果")
    if not effect:
        return []
    rarity = _RARITY_NAMES.get(
        entry.metadata.get("rarity", ""), entry.metadata.get("rarity", "")
    )
    usage_raw = entry.metadata.get("usage", "")
    usage = _POTION_USAGE_NAMES.get(usage_raw, usage_raw)
    target = _TARGET_NAMES.get(
        entry.metadata.get("target", ""), entry.metadata.get("target", "")
    )
    return [
        (f"药水'{entry.name}'的效果是什么？", effect),
        (f"使用'{entry.name}'会发生什么？", effect),
        (f"药水'{entry.name}'是什么稀有度？", f"{rarity}。"),
        (f"'{entry.name}'什么时候可以使用？", f"使用方式:{usage}。目标:{target}。"),
        (f"请给出药水'{entry.name}'的使用方式和完整效果。", f"{usage}。{effect}"),
    ]


def _power_questions(entry: KnowledgeEntry) -> list[tuple[str, str]]:
    """生成能力类别、叠加语义与样例强度问法。

    Args:
        entry (KnowledgeEntry): 已应用 curated 修正的能力条目。

    Returns:
        list[tuple[str, str]]: 能力知识问答对。
    """
    effect = _section(entry.body, "效果")
    if not effect:
        return []
    kind_raw = entry.metadata.get("power_type", "")
    kind = _POWER_TYPE_NAMES.get(kind_raw, kind_raw)
    stack_raw = entry.metadata.get("stack_type", "")
    stack = _POWER_STACK_NAMES.get(stack_raw, stack_raw)
    classification = kind if not stack else f"{kind}（{stack}）"
    if entry.metadata.get("uses_amount") == "true":
        amount = entry.metadata.get("sample_amount", "1")
        subject = f"'{entry.name}'强度为{amount}时"
        return [
            (f"{subject}是什么效果？", f"{classification}。{effect}"),
            (f"{subject}会产生什么作用？", effect),
            (f"{subject}的类别和具体效果是什么？", f"{classification}。{effect}"),
            (f"{subject}是增益还是减益？", f"{kind}。"),
        ]
    return [
        (f"'{entry.name}'是什么效果？", f"{classification}。{effect}"),
        (f"'{entry.name}'会产生什么作用？", effect),
        (f"'{entry.name}'的类别和具体效果是什么？", f"{classification}。{effect}"),
        (f"'{entry.name}'是增益还是减益？", f"{kind}。"),
    ]


def _enchantment_questions(entry: KnowledgeEntry) -> list[tuple[str, str]]:
    """生成附魔效果、卡面附加文本和叠加问法。

    Args:
        entry (KnowledgeEntry): 已应用 curated 修正的附魔条目。

    Returns:
        list[tuple[str, str]]: 附魔知识问答对。
    """
    effect = _section(entry.body, "效果")
    if not effect:
        return []
    extra = _section(entry.body, "卡面附加")
    stackable = entry.metadata.get("stackable", "")
    detail = effect
    if extra:
        detail += f"卡面附加:{extra}"
    return [
        (f"附魔'{entry.name}'的效果是什么？", detail),
        (f"卡牌获得'{entry.name}'后会怎样？", effect),
        (f"附魔'{entry.name}'能否叠加？", f"可叠加:{stackable}。"),
    ]


def _character_questions(entry: KnowledgeEntry) -> list[tuple[str, str]]:
    """生成角色初始配置、简介和故障机器人充能球问法。

    Args:
        entry (KnowledgeEntry): 角色规范事实知识条目。

    Returns:
        list[tuple[str, str]]: 角色知识问答对。
    """
    deck = _compact_reference_counts(_section_items(entry.body, "起始牌组"))
    relics = _compact_reference_counts(_section_items(entry.body, "起始遗物"))
    metadata = entry.metadata
    orb_slots = metadata.get("orb_slots", "0")
    base = (
        f"初始HP{metadata.get('hp', '?')}，金币{metadata.get('gold', '?')}，"
        f"每回合能量{metadata.get('energy', '?')}，充能球槽{orb_slots}。"
        f"初始遗物：{'、'.join(relics)}。"
        f"初始牌组（{sum(_reference_count(item) for item in deck)}张）："
        f"{'、'.join(deck)}。"
    )
    rows = [(f"'{entry.name}'的初始配置是什么？", base)]
    description = _section(entry.body, "简介")
    if description:
        rows.append((f"'{entry.name}'是什么角色？", description))
    orbs = _section_items(entry.body, "充能球")
    if orbs:
        names = [_reference_name(orb) for orb in orbs]
        rows.append((f"'{entry.name}'的充能球有哪几种？", "、".join(names) + "。"))
        rows.extend(_character_orb_questions(entry.body))
    return rows


def _compact_reference_counts(items: Sequence[str]) -> list[str]:
    """按首次出现顺序合并角色初始物品中的重复名称。

    Args:
        items (Sequence[str]): Markdown 列表中的名称与可选 ID。

    Returns:
        list[str]: 名称保留首次顺序，重复项写成 ``名称×数量``。
    """
    ordered: list[str] = []
    counts: dict[str, int] = {}
    for item in items:
        name = _reference_name(item)
        if name not in counts:
            ordered.append(name)
            counts[name] = 0
        counts[name] += 1
    return [f"{name}×{counts[name]}" if counts[name] > 1 else name for name in ordered]


def _reference_name(value: str) -> str:
    """从 ``显示名（稳定 ID）`` 中提取供回答使用的显示名。

    Args:
        value (str): 规范事实中的引用文本。

    Returns:
        str: 去掉稳定 ID 后的显示名称。
    """
    matched = re.fullmatch(r"(.+?)（[^（）]+）", value.strip())
    return matched.group(1) if matched is not None else value.strip()


def _reference_count(value: str) -> int:
    """读取合并后引用末尾的数量，单项按一张计算。

    Args:
        value (str): ``名称`` 或 ``名称×数量``。

    Returns:
        int: 当前合并项代表的原始条目数。
    """
    matched = re.search(r"×(\d+)$", value)
    return int(matched.group(1)) if matched is not None else 1


def _character_orb_questions(body: str) -> list[tuple[str, str]]:
    """从故障机器人充能球章节生成描述与机制问答。

    Args:
        body (str): 角色规范事实 Markdown 正文。

    Returns:
        list[tuple[str, str]]: 每种充能球各三条描述问法和三条数值问法。
    """
    rows: list[tuple[str, str]] = []
    for matched in re.finditer(r"(?ms)^## 充能球：([^\n]+)\s*\n(.*?)(?=^## |\Z)", body):
        name = matched.group(1).strip()
        section = matched.group(2)
        description = re.search(r"(?m)^- 游戏描述：(.+)$", section)
        mechanics = re.search(r"(?m)^- 基础数值与集中：(.+)$", section)
        if description is not None:
            answer = description.group(1)
            rows.extend(
                (
                    (f"游戏如何描述'{name}'充能球？", answer),
                    (f"'{name}'充能球的效果是什么？", answer),
                    (f"生成'{name}'充能球后，它会提供什么效果？", answer),
                )
            )
        if mechanics is not None:
            answer = mechanics.group(1)
            rows.extend(
                (
                    (
                        f"实机测得'{name}'充能球的基础数值和集中关系是什么？",
                        answer,
                    ),
                    (f"'{name}'充能球的被动、激发和集中加成分别怎样？", answer),
                    (f"集中会如何影响'{name}'充能球的基础数值？", answer),
                )
            )
    return rows


def _monster_questions(entry: KnowledgeEntry) -> list[tuple[str, str]]:
    """生成怪物生命、类型、招式和循环问法。

    Args:
        entry (KnowledgeEntry): 已合并受控补充的怪物条目。

    Returns:
        list[tuple[str, str]]: 怪物知识问答对。
    """
    metadata = entry.metadata
    minimum = metadata.get("min_hp", metadata.get("hp", ""))
    maximum = metadata.get("max_hp", metadata.get("hp", ""))
    room_raw = metadata.get("room", "")
    room = _ROOM_NAMES.get(room_raw, room_raw)
    moves = _section(entry.body, "招式")
    cycle = _section(entry.body, "循环")
    hp = _hp_range(minimum, maximum)
    rows = [
        (
            f"怪物'{entry.name}'在A0下的初始生命值范围是什么？",
            hp,
        ),
        (f"'{entry.name}'在0进阶时开场可能有多少生命？", hp),
        (f"'{entry.name}'属于普通、精英还是Boss？", f"{room}。"),
        (
            f"不考虑进阶难度，'{entry.name}'的基础信息是什么？",
            f"怪物类型:{room}。基础HP:{hp}",
        ),
    ]
    if moves:
        rows.append((f"怪物'{entry.name}'有哪些招式？", moves))
    if cycle:
        rows.append((f"怪物'{entry.name}'的行动循环是什么？", cycle))
    return rows


def _map_encounter_questions(entry: KnowledgeEntry) -> list[tuple[str, str]]:
    """生成地图级全部、普通、精英和 Boss 怪池问法。

    普通房间会先从弱遭遇池取若干场，再使用常规遭遇池，因此普通怪池答案
    保留两个子池，避免把早期弱池错误描述为全幕常规池。

    Args:
        entry (KnowledgeEntry): 地图规范事实知识条目。

    Returns:
        list[tuple[str, str]]: 四类地图怪池各两种等价问法。
    """
    weak = _reference_section_answer(entry.body, "弱遭遇池")
    regular = _reference_section_answer(entry.body, "常规遭遇池")
    elite = _reference_section_answer(entry.body, "精英遭遇池")
    boss = _reference_section_answer(entry.body, "Boss 遭遇池")
    rows: list[tuple[str, str]] = []
    ordinary = f"前期弱遭遇池：{weak or '空'}；常规遭遇池：{regular or '空'}。"
    if weak or regular or elite or boss:
        complete = (
            f"普通怪池：{ordinary}精英怪池：{elite or '空'}。Boss池：{boss or '空'}。"
        )
        rows.extend(
            (
                (f"请汇总地图'{entry.name}'的全部怪池。", complete),
                (
                    f"地图'{entry.name}'的普通、精英和Boss怪池分别有哪些遭遇？",
                    complete,
                ),
            )
        )
    if weak or regular:
        rows.extend(
            (
                (f"地图'{entry.name}'的普通怪池有哪些遭遇？", ordinary),
                (
                    f"地图'{entry.name}'普通战斗的前期弱池和常规池分别是什么？",
                    ordinary,
                ),
            )
        )
    if elite:
        rows.extend(
            (
                (f"地图'{entry.name}'的精英怪池有哪些遭遇？", elite),
                (f"在地图'{entry.name}'进入精英房可能遇到哪些遭遇？", elite),
            )
        )
    if boss:
        rows.extend(
            (
                (f"地图'{entry.name}'的Boss池有哪些遭遇？", boss),
                (f"地图'{entry.name}'末尾可能是哪几场Boss战？", boss),
            )
        )
    return rows


def _reference_section_answer(body: str, heading: str) -> str:
    """把引用列表章节转换为不含内部 ID 的中文顿号列表。

    Args:
        body (str): 规范事实 Markdown 正文。
        heading (str): 引用列表所属的二级标题。

    Returns:
        str: 只保留显示名称的单行答案；无列表项时返回空串。
    """
    return "、".join(_reference_name(item) for item in _section_items(body, heading))


def _event_questions(entry: KnowledgeEntry) -> list[tuple[str, str]]:
    """生成仅基于已验证静态选项的事件问法。

    Args:
        entry (KnowledgeEntry): 已解析且通过状态范围检查的事件条目。

    Returns:
        list[tuple[str, str]]: 事件知识问答对。
    """
    options = _section(entry.body, "选项")
    if not options:
        return []
    description = _section(entry.body, "文本")
    return [
        (f"事件'{entry.name}'有哪些已解析选项？", options),
        (f"在'{entry.name}'事件中可以选择什么？", options),
        (
            f"请说明事件'{entry.name}'当前已验证的文本和选项。",
            f"{description}{options}",
        ),
    ]


def _keyword_questions(entry: KnowledgeEntry) -> list[tuple[str, str]]:
    """生成游戏关键词释义问法。

    Args:
        entry (KnowledgeEntry): 关键词规范事实知识条目。

    Returns:
        list[tuple[str, str]]: 关键词知识问答对。
    """
    definition = _section(entry.body, "释义")
    if not definition:
        return []
    return [
        (f"游戏关键词'{entry.name}'是什么意思？", definition),
        (f"解释一下'{entry.name}'。", definition),
    ]


def _section(body: str, heading: str) -> str:
    """提取二级 Markdown 标题下的正文并压成单行。

    Args:
        body (str): 规范事实 Markdown 正文。
        heading (str): 不含 ``##`` 前缀的二级标题。

    Returns:
        str: 删除项目符号并压缩空白后的章节文本；不存在时返回空串。
    """
    match = re.search(
        rf"(?ms)^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)",
        body,
    )
    if match is None:
        return ""
    value = re.sub(r"^[-*]\s*", "", match.group(1).strip(), flags=re.MULTILINE)
    return re.sub(r"\s+", " ", value).strip()


def _section_items(body: str, heading: str) -> list[str]:
    """读取二级 Markdown 章节中的项目列表并保留顺序。

    Args:
        body (str): 规范事实 Markdown 正文。
        heading (str): 不含 ``##`` 前缀的二级标题。

    Returns:
        list[str]: 去掉项目符号的非空列表项。
    """
    matched = re.search(rf"(?ms)^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)", body)
    if matched is None:
        return []
    return [
        item.group(1).strip()
        for line in matched.group(1).splitlines()
        if (item := re.match(r"^[-*]\s+(.+)$", line.strip())) is not None
    ]


def _compact_upgrade(value: str) -> str:
    """把升级项目符号转换成适合短答案的连续文本。

    Args:
        value (str): 卡牌升级章节的 Markdown 文本。

    Returns:
        str: 保留费用和效果变化的连续中文短答案。
    """
    value = re.sub(r"^[-*]\s*", "", value, flags=re.MULTILINE)
    value = re.sub(r"费用：([^ ]+) → ([^ ]+)", r"费用\1→\2", value)
    value = value.replace("效果：", "升级效果:")
    return re.sub(r"\s+", "。", value).strip("。") + "。"


def _hp_range(minimum: str, maximum: str) -> str:
    """把最小和最大生命值渲染为中文范围。

    Args:
        minimum (str): 最小初始生命值。
        maximum (str): 最大初始生命值。

    Returns:
        str: 单值、范围或空串。
    """
    if not minimum and not maximum:
        return ""
    if minimum == maximum or not maximum:
        return f"{minimum}。"
    return f"{minimum}～{maximum}。"


def _is_negative_number(value: str) -> bool:
    """识别游戏用负费用表达的不可打出哨兵。

    Args:
        value (str): 规范事实卡牌费用文本。

    Returns:
        bool: 文本可解析为负数时为真。
    """
    try:
        return float(value) < 0
    except ValueError:
        return False


def _card_cost_text(energy_cost: str, star_cost: str) -> str:
    """组合能量与星能成本，避免把储君卡误写成免费。

    Args:
        energy_cost (str): 能量费用或 ``X``。
        star_cost (str): 可选星能费用或 ``X``。

    Returns:
        str: 同时保留两种资源的中文费用文本。
    """
    energy = f"{energy_cost}点能量"
    if not star_cost:
        return energy
    return f"{energy} + {star_cost}点星能"


def _remove_stale_generated(
    root: Path,
    expected: Mapping[str, set[str]],
) -> None:
    """删除本生成器负责但本轮不再产生的 JSONL。

    Args:
        root (Path): 多问法生成目录。
        expected (Mapping[str, set[str]]): 各类别本轮应保留的 JSONL 文件名。

    Returns:
        None: 过期文件清理完成后返回。
    """
    if not root.is_dir():
        return
    for category_dir in root.iterdir():
        if not category_dir.is_dir():
            continue
        current = expected.get(category_dir.name, set())
        for path in category_dir.glob("*.jsonl"):
            if path.name not in current:
                path.unlink()


def _is_special_object(object_id: str) -> bool:
    """判断实体是否需要在人工报告中展示保留决策。

    Args:
        object_id (str): 游戏实体稳定 ID。

    Returns:
        bool: 命中特殊前缀或明确辅助对象时为真。
    """
    return object_id.startswith(
        ("FAKE_", "MOCK_", "TEST_", "DEPRECATED_")
    ) or object_id in {
        "ARCHITECT",
        "BYRDPIP",
        "OSTY",
        "PAELS_LEGION",
        "THE_ARCHITECT",
    }


def _special_object_decision(category: str, row: Mapping[str, Any]) -> str:
    """说明一个特殊命名对象在监督知识中的处理依据。

    Args:
        category (str): 实体类别。
        row (Mapping[str, Any]): Mod 原始实体。

    Returns:
        str: 面向人工复核的保留或排除说明。
    """
    object_id = str(row.get("id") or "")
    description = str(row.get("description") or "")
    if object_id == "FAKE_MERCHANT_EVENT_ENCOUNTER":
        return "用户确认真实存在，保留"
    if object_id == "FAKE_MERCHANT_MONSTER":
        return "由 FAKE_MERCHANT_EVENT_ENCOUNTER 实际生成，保留"
    if (category, object_id) == ("encounters", "TEST_SUBJECT_BOSS"):
        return "当前版本非调试、可奖励 Boss 遭遇；raw 保留，不生成现场可见的组合题"
    if (category, object_id) == ("monsters", "TEST_SUBJECT"):
        return "当前版本图鉴可见 Boss 且招式完整；保留并进入监督问法"
    if (category, object_id) == ("events", "FAKE_MERCHANT"):
        return "事件入口是结构占位文本；raw 保留，监督问法排除，真实战斗由遭遇条目覆盖"
    if (category, object_id) == ("events", "THE_ARCHITECT"):
        return "结局动态对话，无稳定初始选项；raw 保留，监督问法排除"
    if object_id == "ARCHITECT":
        return "结局对话角色，不是常规敌人；raw 保留，监督问法排除"
    if category == "relics" and object_id in {"BYRDPIP", "PAELS_LEGION"}:
        return "真实遗物模型，与同名辅助 MonsterModel 区分；保留"
    if category == "monsters" and object_id in {"BYRDPIP", "OSTY", "PAELS_LEGION"}:
        return "友方/演出用 MonsterModel，不作为敌人知识；raw 保留，监督问法排除"
    if object_id.startswith("DEPRECATED_") or "已经从游戏中被移除" in description:
        return "明确标注已移除；原始快照保留，训练知识候选排除"
    if object_id.startswith("FAKE_") and description.strip():
        return "保留：有完整游戏文本；不能按 FAKE_ 前缀删除"
    if object_id.startswith("TEST_"):
        return "待确认自然流程可达性；raw 保留，监督问法暂时排除"
    if object_id.startswith("MOCK_"):
        return "当前版本内部测试辅助对象；raw 保留，监督问法排除"
    return f"{category} 特殊命名对象；等待实机证据"


def _has_unresolved(value: object) -> bool:
    """递归检查 JSON 值中是否仍含未解析模板或资源文本。

    Args:
        value (object): 待审计的任意 JSON 兼容值。

    Returns:
        bool: 任一嵌套字符串命中污染规则时为真。
    """
    if isinstance(value, str):
        return bool(_BAD_TEXT.search(value))
    if isinstance(value, Mapping):
        return any(_has_unresolved(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_has_unresolved(item) for item in value)
    return False


def _has_complete_option(row: Mapping[str, Any]) -> bool:
    """判断事件是否至少有一个标题和描述均完整的选项。

    Args:
        row (Mapping[str, Any]): 事件原始实体。

    Returns:
        bool: 存在可用于知识渲染的完整选项时为真。
    """
    options = row.get("options")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        return False
    return any(
        isinstance(option, Mapping)
        and isinstance(option.get("title"), str)
        and option["title"].strip()
        and isinstance(option.get("description"), str)
        and option["description"].strip()
        and not _BAD_TEXT.search(option["description"])
        for option in options
    )


def _has_missing_core(category: str, row: Mapping[str, Any]) -> bool:
    """按类别检查实体是否缺少生成问答所需的核心事实。

    Args:
        category (str): 实体类别。
        row (Mapping[str, Any]): Mod 原始实体。

    Returns:
        bool: 描述、招式、敌人或事件选项缺失时为真。
    """
    if category in {
        "cards",
        "relics",
        "potions",
        "enchantments",
        "powers",
        "keywords",
    }:
        return not bool(str(row.get("description") or "").strip())
    if category == "monsters":
        return not bool(row.get("moves"))
    if category == "encounters":
        return not bool(row.get("monsters"))
    if category == "events":
        return not _has_complete_option(row)
    return False
