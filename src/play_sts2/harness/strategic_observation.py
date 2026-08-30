"""渲染战略模型进行整局规划所需的长期状态与地图。"""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .text import card_display_name, clean_game_text, human_act_number

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
_BOSS_MEMBERS = {
    "KAISER_CRAB_BOSS": ("碾碎爪", "火箭"),
    "QUEEN_BOSS": ("女王", "火炬头聚合体"),
    "THE_KIN_BOSS": ("同族信徒", "同族神官"),
}
_ENCHANTMENT_NAMES = {
    "ADROIT": "伶俐",
    "CLONE": "克隆",
    "CORRUPTED": "腐化",
    "DEPRECATED_ENCHANTMENT": "弃用",
    "GLAM": "华彩",
    "GOOPY": "黏糊",
    "IMBUED": "注能",
    "INKY": "墨影",
    "INSTINCT": "本能",
    "MOCK_FREE_ENCHANTMENT": "临时免费附魔",
    "MOMENTUM": "动量",
    "NIMBLE": "灵巧",
    "PERFECT_FIT": "完美契合",
    "ROYALLY_APPROVED": "王室认证",
    "SHARP": "锋利",
    "SLITHER": "蛇行",
    "SLUMBERING_ESSENCE": "沉眠精华",
    "SOULS_POWER": "灵魂之力",
    "SOWN": "播种",
    "SPIRAL": "涡旋",
    "STEADY": "稳定",
    "SWIFT": "迅速",
    "TEZCATARAS_EMBER": "特兹卡塔拉的余烬",
    "VIGOROUS": "活力",
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
    act_id = human_act_number(run.get("act_id"))
    boss_id = _clean_text(run.get("boss_id")) or "未知"
    boss = _BOSS_NAMES.get(boss_id, "未知")
    boss_members = _BOSS_MEMBERS.get(boss_id)
    if boss_members:
        boss += f" | 组成: {'、'.join(boss_members)}"
    lines = [f"【第{act_id}幕】", f"本幕Boss: {boss}", _render_ascension(run)]
    lines.extend(
        (
            "【当前状态】",
            _render_character(run),
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


def _render_character(run: Mapping[str, Any]) -> str:
    """渲染角色身份、最大能量与玩家可见的角色容量。

    Args:
        run (Mapping[str, Any]): 当前整局状态。

    Returns:
        str: 不依赖角色 ID 特判的稳定资源摘要。
    """
    name = _clean_text(run.get("character_name")) or "未知角色"
    parts = [f"角色: {name}"]
    max_energy = run.get("max_energy")
    if isinstance(max_energy, int) and not isinstance(max_energy, bool):
        parts.append(f"最大能量{max_energy}")
    base_orb_slots = run.get("base_orb_slots")
    if (
        isinstance(base_orb_slots, int)
        and not isinstance(base_orb_slots, bool)
        and base_orb_slots > 0
    ):
        parts.append(f"基础充能球槽{base_orb_slots}")
    return " | ".join(parts)


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
    ]
    available_nodes = map_state.get("available_nodes") or []
    route_starts: Sequence[Mapping[str, Any]] = available_nodes
    if available_nodes:
        lines.append("可前往:")
        for option in available_nodes:
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
    else:
        current_node = node_index.get(_node_key(current))
        route_starts = (
            [node_index[key] for key in _child_keys(current_node) if key in node_index]
            if current_node is not None
            else []
        )
        next_text = "、".join(
            f"行{row}列{col} {_node_glyph(node)}"
            for node in route_starts
            for row, col in [_node_key(node)]
        )
        lines.append(f"当前节点后续: {next_text or '无'}")

    reachable = _reachable_nodes(route_starts, node_index)
    counts = Counter(_node_glyph(node) for node in reachable.values())
    count_text = " ".join(
        f"{glyph}×{counts[glyph]}"
        for glyph in sorted(counts, key=lambda item: _GLYPH_ORDER.get(item, 99))
    )
    start_row = min(
        (_node_key(node)[0] for node in route_starts),
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
    lines = [f"难度{ascension}:"]
    for effect in run.get("ascension_effects") or []:
        if isinstance(effect, Mapping):
            name = _clean_text(effect.get("name"))
            description = _clean_text(effect.get("description"))
            suffix = f"：{description}" if description else ""
            if name:
                lines.append(f"- {name}{suffix}")
        else:
            name = _clean_text(effect)
            if name:
                lines.append(f"- {name}")
    return "\n".join(lines) if len(lines) > 1 else f"难度{ascension}"


def _render_potions(potions: Sequence[Mapping[str, Any]]) -> str:
    """渲染药水栏占用情况和药水效果。

    Args:
        potions (Sequence[Mapping[str, Any]]): 按槽位排序的药水状态。

    Returns:
        str: 单行药水栏摘要。
    """
    occupied = sum(bool(potion.get("occupied")) for potion in potions)
    if not potions:
        return "药水栏 0/0: 无"
    lines = [f"药水栏 {occupied}/{len(potions)}:"]
    for potion in potions:
        index = potion.get("index")
        if not potion.get("occupied"):
            lines.append(f"- [{index}] 空")
            continue
        name = _clean_text(potion.get("name")) or "未知药水"
        description = _clean_text(potion.get("description"))
        suffix = f"：{description}" if description else ""
        lines.append(f"- [{index}] {name}{suffix}")
    return "\n".join(lines)


def _render_relics(relics: Sequence[Mapping[str, Any]]) -> str:
    """渲染持有遗物及其效果。

    Args:
        relics (Sequence[Mapping[str, Any]]): 当前持有的遗物。

    Returns:
        str: 每件遗物一行的动态状态摘要。
    """
    if not relics:
        return "遗物: 无"
    lines = [f"遗物 {len(relics)} 件:"]
    for fallback_index, relic in enumerate(relics):
        index = relic.get("index", fallback_index)
        name = _clean_text(relic.get("name")) or "未知遗物"
        stack_count = relic.get("stack_count", relic.get("stack"))
        if isinstance(stack_count, int) and stack_count > 1:
            name = f"{name}×{stack_count}"
        states = []
        counter_value = relic.get("counter_value")
        if relic.get("show_counter") is True and isinstance(counter_value, int):
            states.append(f"计数{counter_value}")
        is_used_up = relic.get("is_used_up") is True
        is_melted = relic.get("is_melted") is True
        if is_used_up:
            states.append("已耗尽")
        if is_melted:
            states.extend(("已熔毁", "效果失效"))
        status = str(relic.get("status") or "")
        if status == "Active":
            states.append("已激活")
        elif status == "Disabled":
            states.append("当前禁用")
        state_text = f"〔{'；'.join(states)}〕" if states else ""
        description = (
            "" if is_used_up or is_melted else _clean_text(relic.get("description"))
        )
        suffix = f"：{description}" if description else ""
        lines.append(f"- [{index}] {name}{state_text}{suffix}")
    return "\n".join(lines)


def _render_deck(deck: Sequence[Mapping[str, Any]]) -> str:
    """按实例渲染牌组的升级、附魔和永久数值。

    Args:
        deck (Sequence[Mapping[str, Any]]): 当前牌组中的全部卡牌。

    Returns:
        str: 保留稳定实例索引的多行牌组摘要。
    """
    upgraded = sum(_upgrade_level(card) > 0 for card in deck)
    enchanted = sum(bool(card.get("enchantment_id")) for card in deck)
    lines = [f"牌组 {len(deck)} 张（升级{upgraded}，附魔{enchanted}）:"]
    lines.extend(
        format_strategic_card(card, fallback_index=fallback_index)
        for fallback_index, card in enumerate(deck)
    )
    return "\n".join(lines)


def format_strategic_card(
    card: Mapping[str, Any],
    *,
    fallback_index: int = 0,
    extra_states: Sequence[str] = (),
) -> str:
    """按牌组样式渲染一个战略卡牌实例或候选项。

    Args:
        card (Mapping[str, Any]): Mod 返回的卡牌实例、候选或升级预览。
        fallback_index (int): 状态缺少稳定索引时使用的列表位置。
        extra_states (Sequence[str]): 商店价格、不可购买等附加玩家可见状态。

    Returns:
        str: 不含英文卡牌 ID 和重复升级等级的单行卡牌文本。
    """
    index = card.get("index", fallback_index)
    name = card_display_name(
        card.get("name"),
        upgraded=card.get("upgraded") is True,
    )
    cost = _card_cost(card)
    card_type = _CARD_TYPE_NAMES.get(
        str(card.get("card_type") or ""),
        _clean_text(card.get("card_type")) or "未知",
    )
    details = f"{cost}{card_type}" or "未知"
    states = []
    enchantment = _render_enchantment(card)
    if enchantment:
        states.append(enchantment)
    if card.get("selected") is True:
        states.append("已选择")
    states.extend(item for item in extra_states if item)
    state_text = f"〔{'；'.join(states)}〕" if states else ""
    rules_text = _clean_text(card.get("resolved_rules_text") or card.get("rules_text"))
    suffix = f"：{rules_text}" if rules_text else ""
    return f"- [{index}] {name}（{details}）{state_text}{suffix}"


def _upgrade_level(card: Mapping[str, Any]) -> int:
    """读取卡牌实例的真实升级等级并兼容旧录制。

    Args:
        card (Mapping[str, Any]): 一张牌的结构化状态。

    Returns:
        int: 非负升级次数。
    """
    value = card.get("upgrade_level")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 1 if card.get("upgraded") is True else 0


def _render_enchantment(card: Mapping[str, Any]) -> str:
    """渲染卡牌实例附魔名称、层数和动态说明。

    Args:
        card (Mapping[str, Any]): 一张牌的结构化状态。

    Returns:
        str: ``附魔：中文名×层数``；未附魔时为空。
    """
    enchantment_id = _clean_text(card.get("enchantment_id"))
    if not enchantment_id:
        return ""
    name = _clean_text(card.get("enchantment_name")) or _ENCHANTMENT_NAMES.get(
        enchantment_id,
        "未知附魔",
    )
    amount = card.get("enchantment_amount")
    amount_text = (
        f"×{amount}"
        if isinstance(amount, int) and not isinstance(amount, bool) and amount > 1
        else ""
    )
    return f"附魔：{name}{amount_text}"


def _card_cost(card: Mapping[str, Any]) -> str:
    """渲染战略牌组列表中的紧凑费用。

    Args:
        card (Mapping[str, Any]): 一张牌的结构化状态。

    Returns:
        str: ``1费``、``X费+2星能`` 或空字符串。
    """
    costs = []
    if card.get("costs_x"):
        costs.append("X费")
    else:
        energy_cost = card.get("energy_cost")
        if isinstance(energy_cost, int) and energy_cost >= 0:
            costs.append(f"{energy_cost}费")
    if card.get("star_costs_x"):
        costs.append("X星能")
    else:
        star_cost = card.get("star_cost")
        if isinstance(star_cost, int) and star_cost > 0:
            costs.append(f"{star_cost}星能")
    return "+".join(costs)


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
    return clean_game_text(value)
