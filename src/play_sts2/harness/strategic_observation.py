"""渲染战略模型进行整局规划所需的长期状态与地图。"""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

_MARKUP_PATTERN = re.compile(r"\[/?[A-Za-z_]+(?:=[^\]]+)?\]")
_RESOURCE_PATTERN = re.compile(r"res://\S+?\.png")
_CARD_TYPE_NAMES = {
    "Attack": "攻击",
    "Curse": "诅咒",
    "Power": "能力",
    "Skill": "技能",
    "Status": "状态",
}
_NODE_GLYPHS = {
    "Ancient": "古",
    "Boss": "王",
    "Elite": "精",
    "Monster": "敌",
    "RestSite": "火",
    "Shop": "商",
    "Treasure": "宝",
    "Unknown": "?",
}
_GLYPH_ORDER = {glyph: index for index, glyph in enumerate("?商宝敌火王精古")}
_BOSS_NAMES = {
    "AEONGLASS_BOSS": "永世沙漏",
    "CEREMONIAL_BEAST_BOSS": "仪式兽",
    "KAISER_CRAB_BOSS": "帝皇蟹",
    "KNOWLEDGE_DEMON_BOSS": "知识恶魔",
    "LAGAVULIN_MATRIARCH_BOSS": "乐加维林族母",
    "QUEEN_BOSS": "女王",
    "SOUL_FYSH_BOSS": "灵魂异鱼",
    "TEST_SUBJECT_BOSS": "实验体",
    "THE_INSATIABLE_BOSS": "无厌沙虫",
    "THE_KIN_BOSS": "同族小队",
    "VANTOM_BOSS": "墨影幻灵",
    "WATERFALL_GIANT_BOSS": "瀑布巨兽",
}

NodeKey = tuple[int, int]


def render_strategic_context(state: Mapping[str, Any]) -> str:
    """渲染每次战略决策都必须携带的整局上下文。

    Args:
        state (Mapping[str, Any]): Mod 返回的完整当前游戏状态。

    Returns:
        str: 幕、Boss、进阶效果、资源、遗物、药水与牌组摘要。
    """
    run = state.get("run") or {}
    act_id = _as_int(run.get("act_id"))
    boss_id = _clean_text(run.get("boss_id")) or "未知"
    boss_name = _BOSS_NAMES.get(boss_id)
    boss = f"{boss_name} ({boss_id})" if boss_name else boss_id
    lines = [f"【第{act_id}幕】", f"本幕Boss: {boss}", _render_ascension(run)]
    lines.extend(
        (
            "【当前状态】",
            (
                f"HP {run.get('current_hp', 0)}/{run.get('max_hp', 0)} | "
                f"金币{run.get('gold', 0)} | 第{run.get('floor', 0)}层"
            ),
            _render_potions(run.get("potions") or []),
            _render_relics(run.get("relics") or []),
            _render_deck(run.get("deck") or []),
        )
    )
    position = _render_position(state.get("map") or {})
    if position:
        lines.append(position)
    return "\n".join(lines)


def render_map(state: Mapping[str, Any]) -> str:
    """渲染候选路线、可达节点统计与未走地图邻接图。

    Args:
        state (Mapping[str, Any]): 当前地图状态。

    Returns:
        str: 可供战略模型比较完整路线的地图文本。
    """
    map_state = state.get("map") or {}
    nodes = map_state.get("nodes") or []
    if not nodes:
        return _render_simple_map(map_state)

    node_index = {_node_key(node): node for node in nodes}
    current = map_state.get("current_node") or {}
    lines = [
        "=== 地图 ===",
        f"当前位置: 行{current.get('row')} 列{current.get('col')}",
        "可前往:",
    ]
    for option in map_state.get("available_nodes") or []:
        key = _node_key(option)
        node = node_index.get(key, option)
        distance = _distance_to_boss(key, node_index)
        preview = _route_preview(key, node_index)
        suffix = f" | 距Boss {distance}步" if distance is not None else ""
        if preview:
            suffix += f" | 后续: {preview}"
        lines.append(
            f"  [{option.get('index')}] 行{key[0]}列{key[1]} "
            f"{_node_glyph(node)}{suffix}"
        )

    reachable = _reachable_nodes(map_state.get("available_nodes") or [], node_index)
    counts = Counter(_node_glyph(node) for node in reachable.values())
    count_text = " ".join(
        f"{glyph}×{counts[glyph]}"
        for glyph in sorted(counts, key=lambda item: _GLYPH_ORDER.get(item, 99))
    )
    start_row = min(
        (_node_key(node)[0] for node in map_state.get("available_nodes") or []),
        default=_as_int(current.get("row")),
    )
    deepest = max((row - start_row for row, _col in reachable), default=0)
    lines.append(f"本幕可达: {count_text or '无'} | 最深可达 {deepest}步")
    lines.extend(
        ("=== 全图(坐标邻接, 未走层) ===", _render_adjacency(nodes, node_index))
    )
    return "\n".join(lines)


def _render_ascension(run: Mapping[str, Any]) -> str:
    """渲染进阶等级及当前生效的规则。

    Args:
        run (Mapping[str, Any]): 当前整局状态。

    Returns:
        str: 单行进阶摘要。
    """
    ascension = run.get("ascension", 0)
    effects = []
    for effect in run.get("ascension_effects") or []:
        if isinstance(effect, Mapping):
            name = _clean_text(effect.get("name"))
            description = _clean_text(effect.get("description"))
            effects.append(f"{name}（{description}）" if description else name)
        else:
            effects.append(_clean_text(effect))
    suffix = f": {'；'.join(item for item in effects if item)}" if effects else ""
    return f"难度{ascension}{suffix}"


def _render_potions(potions: Sequence[Mapping[str, Any]]) -> str:
    """渲染药水栏占用情况和药水效果。

    Args:
        potions (Sequence[Mapping[str, Any]]): 按槽位排序的药水状态。

    Returns:
        str: 单行药水栏摘要。
    """
    occupied = sum(bool(potion.get("occupied")) for potion in potions)
    values = []
    for potion in potions:
        index = potion.get("index")
        if not potion.get("occupied"):
            values.append(f"[{index}] -")
            continue
        name = _clean_text(potion.get("name")) or "未知药水"
        description = _clean_text(potion.get("description"))
        values.append(f"[{index}] {name}{f'（{description}）' if description else ''}")
    return f"药水栏 {occupied}/{len(potions)}: {' '.join(values) if values else '无'}"


def _render_relics(relics: Sequence[Mapping[str, Any]]) -> str:
    """渲染持有遗物及其效果。

    Args:
        relics (Sequence[Mapping[str, Any]]): 当前持有的遗物。

    Returns:
        str: 单行遗物摘要。
    """
    values = []
    for relic in relics:
        name = _clean_text(relic.get("name")) or "未知遗物"
        description = _clean_text(relic.get("description"))
        stack = relic.get("stack")
        if stack is not None:
            name = f"{name}×{stack}"
        values.append(f"{name}（{description}）" if description else name)
    return f"遗物: {', '.join(values) if values else '无'}"


def _render_deck(deck: Sequence[Mapping[str, Any]]) -> str:
    """按名称、升级、费用和类型分组渲染牌组。

    Args:
        deck (Sequence[Mapping[str, Any]]): 当前牌组中的全部卡牌。

    Returns:
        str: 多行牌组摘要。
    """
    groups: dict[tuple[str, str, str], int] = {}
    for card in deck:
        name = _clean_text(card.get("name")) or "未知卡牌"
        if card.get("upgraded"):
            name += "+"
        cost = _card_cost(card)
        card_type = _CARD_TYPE_NAMES.get(
            str(card.get("card_type") or ""),
            _clean_text(card.get("card_type")) or "未知",
        )
        key = (name, cost, card_type)
        groups[key] = groups.get(key, 0) + 1
    upgraded = sum(bool(card.get("upgraded")) for card in deck)
    lines = [f"牌组 {len(deck)} 张（升级{upgraded}）"]
    lines.extend(
        f"- {name} x{count}（{cost}{card_type}）"
        for (name, cost, card_type), count in groups.items()
    )
    return "\n".join(lines)


def _card_cost(card: Mapping[str, Any]) -> str:
    """渲染战略牌组列表中的紧凑费用。

    Args:
        card (Mapping[str, Any]): 一张牌的结构化状态。

    Returns:
        str: ``1费``、``X费`` 或空字符串。
    """
    if card.get("costs_x"):
        return "X费"
    cost = card.get("energy_cost")
    return f"{cost}费" if isinstance(cost, int) and cost >= 0 else ""


def _render_position(map_state: Mapping[str, Any]) -> str:
    """渲染当前位置以及按行排序的已走节点。

    Args:
        map_state (Mapping[str, Any]): 当前幕地图。

    Returns:
        str: 位置与路径摘要；没有地图信息时为空。
    """
    current = map_state.get("current_node") or {}
    if current.get("row") is None or current.get("col") is None:
        return ""
    visited = sorted(
        (node for node in map_state.get("nodes") or [] if node.get("visited")),
        key=_node_key,
    )
    path = "→".join(f"{_node_glyph(node)}(行{node.get('row')})" for node in visited)
    suffix = f" | 已走: {path}" if path else ""
    return f"位置: 行{current.get('row')} 列{current.get('col')}{suffix}"


def _render_simple_map(map_state: Mapping[str, Any]) -> str:
    """在旧状态没有完整节点图时渲染候选节点。

    Args:
        map_state (Mapping[str, Any]): 缺少 ``nodes`` 的地图状态。

    Returns:
        str: 兼容旧录制数据的候选节点列表。
    """
    lines = ["=== 地图 ==="]
    for node in map_state.get("available_nodes") or []:
        node_type = str(node.get("node_type") or "Unknown")
        names = {
            "Ancient": "先古之民",
            "Boss": "Boss",
            "Elite": "精英敌人",
            "Monster": "普通敌人",
            "RestSite": "休息处",
            "Shop": "商店",
            "Treasure": "宝箱",
            "Unknown": "未知地点",
        }
        lines.append(
            f"[{node.get('index')}] 第 {node.get('row')} 行，第 {node.get('col')} 列"
            f" | {names.get(node_type, node_type)}"
        )
    return "\n".join(lines)


def _reachable_nodes(
    starts: Sequence[Mapping[str, Any]],
    node_index: Mapping[NodeKey, Mapping[str, Any]],
) -> dict[NodeKey, Mapping[str, Any]]:
    """返回从全部候选节点可以到达的唯一节点。

    Args:
        starts (Sequence[Mapping[str, Any]]): 当前可选择的下一层节点。
        node_index (Mapping[NodeKey, Mapping[str, Any]]): 坐标到节点的索引。

    Returns:
        dict[NodeKey, Mapping[str, Any]]: 包含起点的可达子图。
    """
    pending = [_node_key(node) for node in starts]
    reachable: dict[NodeKey, Mapping[str, Any]] = {}
    while pending:
        key = pending.pop()
        if key in reachable or key not in node_index:
            continue
        node = node_index[key]
        reachable[key] = node
        pending.extend(_child_keys(node))
    return reachable


def _distance_to_boss(
    start: NodeKey,
    node_index: Mapping[NodeKey, Mapping[str, Any]],
) -> int | None:
    """计算候选节点到任一 Boss 节点的最短边数。

    Args:
        start (NodeKey): 候选节点坐标。
        node_index (Mapping[NodeKey, Mapping[str, Any]]): 坐标到节点的索引。

    Returns:
        int | None: 最短步数；不可达 Boss 时为 ``None``。
    """
    frontier = [(start, 0)]
    visited: set[NodeKey] = set()
    while frontier:
        key, distance = frontier.pop(0)
        if key in visited:
            continue
        visited.add(key)
        node = node_index.get(key)
        if node is None:
            continue
        if node.get("is_boss") or node.get("node_type") == "Boss":
            return distance
        frontier.extend((child, distance + 1) for child in _child_keys(node))
    return None


def _route_preview(
    start: NodeKey,
    node_index: Mapping[NodeKey, Mapping[str, Any]],
    *,
    depth: int = 3,
) -> str:
    """按后续深度汇总一条候选路线可能遇到的节点类型。

    Args:
        start (NodeKey): 候选节点坐标。
        node_index (Mapping[NodeKey, Mapping[str, Any]]): 坐标到节点的索引。
        depth (int): 最多展示的后续层数。

    Returns:
        str: 以箭头分隔、同层以斜线分隔的节点类型预览。
    """
    frontier = [start]
    levels = []
    for _level in range(depth):
        children = {
            child
            for key in frontier
            if (node := node_index.get(key)) is not None
            for child in _child_keys(node)
        }
        if not children:
            break
        glyphs = {_node_glyph(node_index[key]) for key in children if key in node_index}
        levels.append(
            "/".join(sorted(glyphs, key=lambda item: _GLYPH_ORDER.get(item, 99)))
        )
        frontier = sorted(children)
    return " → ".join(levels)


def _render_adjacency(
    nodes: Sequence[Mapping[str, Any]],
    node_index: Mapping[NodeKey, Mapping[str, Any]],
) -> str:
    """按行渲染当前节点和所有未走节点的坐标邻接关系。

    Args:
        nodes (Sequence[Mapping[str, Any]]): 地图中的全部节点。
        node_index (Mapping[NodeKey, Mapping[str, Any]]): 坐标到节点的索引。

    Returns:
        str: 每个地图行占一行的紧凑邻接图。
    """
    visible = [
        node
        for node in nodes
        if (not node.get("visited") or node.get("is_current"))
        and (node.get("children") or node.get("is_current"))
    ]
    rows: dict[int, list[str]] = {}
    for node in sorted(visible, key=_node_key):
        row, col = _node_key(node)
        marker = "@" if node.get("is_current") else ""
        children = " ".join(
            f"({key[0]},{key[1]}){_node_glyph(node_index[key])}"
            for key in _child_keys(node)
            if key in node_index
        )
        rows.setdefault(row, []).append(
            f"({row},{col}){marker}{_node_glyph(node)}→{children}"
        )
    return "\n".join(" | ".join(rows[row]) for row in sorted(rows))


def _child_keys(node: Mapping[str, Any]) -> list[NodeKey]:
    """提取一个地图节点的全部子节点坐标。

    Args:
        node (Mapping[str, Any]): 一个地图节点。

    Returns:
        list[NodeKey]: 保留原始顺序的子节点坐标。
    """
    return [_node_key(child) for child in node.get("children") or []]


def _node_key(node: Mapping[str, Any]) -> NodeKey:
    """把地图节点转换为可哈希坐标。

    Args:
        node (Mapping[str, Any]): 含 ``row`` 和 ``col`` 的节点。

    Returns:
        NodeKey: 整数行列坐标。
    """
    return (_as_int(node.get("row")), _as_int(node.get("col")))


def _node_glyph(node: Mapping[str, Any]) -> str:
    """返回地图节点的单字紧凑标记。

    Args:
        node (Mapping[str, Any]): 一个地图节点。

    Returns:
        str: ``敌``、``火``、``?`` 等稳定标记。
    """
    node_type = str(node.get("node_type") or "Unknown")
    return _NODE_GLYPHS.get(node_type, node_type)


def _as_int(value: Any) -> int:
    """把 Mod 的数字或数字字符串统一转换为整数。

    Args:
        value (Any): 可能为 ``None``、整数或数字字符串的值。

    Returns:
        int: 转换后的整数；无效值为零。
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _clean_text(value: Any) -> str:
    """移除游戏富文本标签和资源路径并合并空白。

    Args:
        value (Any): 可能包含游戏标记的字段值。

    Returns:
        str: 适合直接放入模型观测的纯文本。
    """
    text = _MARKUP_PATTERN.sub("", str(value or ""))
    text = _RESOURCE_PATTERN.sub("", text)
    return " ".join(text.split())
