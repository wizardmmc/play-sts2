"""把 Mod 状态投影为在线推理与 SFT 共用的可读观测。"""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .actions import model_actions
from .ownership import HarnessLayer, state_layer

_MARKUP_PATTERN = re.compile(r"\[/?[A-Za-z_]+(?:=[^\]]+)?\]")
_RESOURCE_PATTERN = re.compile(r"res://\S+?\.png")
_NODE_NAMES = {
    "Ancient": "先古之民",
    "Boss": "Boss",
    "Elite": "精英敌人",
    "Monster": "普通敌人",
    "RestSite": "休息处",
    "Shop": "商店",
    "Treasure": "宝箱",
    "Unknown": "未知地点",
}
_INTENT_NAMES = {
    "Attack": "攻击",
    "Buff": "强化",
    "Debuff": "弱化",
    "Defend": "防御",
    "Escape": "逃跑",
    "Sleep": "睡眠",
    "StatusCard": "塞入状态牌",
    "Stun": "眩晕",
    "Unknown": "未知",
}


class ObservationError(ValueError):
    """表示当前状态无法生成可靠的模型观测。"""


@dataclass(frozen=True, slots=True)
class Observation:
    """表示一次可直接交给模型的决策观测。

    Args:
        layer (HarnessLayer): 负责当前决策的 Harness 层。
        text (str): 已去除游戏富文本标记的可读状态。
        available_actions (tuple[str, ...]): 模型可以选择的动作名称。
    """

    layer: HarnessLayer
    text: str
    available_actions: tuple[str, ...]


def build_observation(state: Mapping[str, Any]) -> Observation:
    """把当前 Mod 状态转换为一个模型决策观测。

    Args:
        state (Mapping[str, Any]): Mod 返回的完整当前游戏状态。

    Raises:
        ObservationError: 当前状态无需模型决策或屏幕尚未支持。

    Returns:
        Observation: 包含归属、可读文本和合法动作的共享观测。
    """
    layer = state_layer(state)
    actions = model_actions(state)
    if layer is HarnessLayer.TRANSIENT or not actions:
        raise ObservationError("当前状态无需模型决策")

    screen = str(state.get("screen") or "")
    renderers = {
        "CARD_SELECTION": _render_card_selection,
        "COMBAT": _render_combat,
        "EVENT": _render_event,
        "MAP": _render_map,
        "REWARD": _render_reward,
    }
    renderer = renderers.get(screen)
    if renderer is None:
        raise ObservationError(f"尚未支持的决策屏幕: {screen}")

    sections = [_render_run(state)]
    if layer is HarnessLayer.STRATEGIC:
        sections.append(_render_inventory(state))
    sections.extend((renderer(state), _render_actions(actions)))
    return Observation(
        layer=layer,
        text="\n\n".join(section for section in sections if section),
        available_actions=actions,
    )


def _render_run(state: Mapping[str, Any]) -> str:
    """渲染角色、进阶、幕数、楼层和整局资源。

    Args:
        state (Mapping[str, Any]): Mod 返回的完整当前游戏状态。

    Returns:
        str: 两行整局摘要。
    """
    run = state.get("run") or {}
    act = int(run.get("act_id", 0)) + 1
    return (
        f"角色: {run.get('character_name', '未知')} | "
        f"进阶: {run.get('ascension', 0)} | 第 {act} 幕 | "
        f"第 {run.get('floor', 0)} 层\n"
        f"生命: {run.get('current_hp', 0)}/{run.get('max_hp', 0)} | "
        f"金币: {run.get('gold', 0)}"
    )


def _render_inventory(state: Mapping[str, Any]) -> str:
    """渲染战略决策长期依赖的遗物、药水和牌组。

    Args:
        state (Mapping[str, Any]): Mod 返回的完整当前游戏状态。

    Returns:
        str: 战略层的整局物品与牌组摘要。
    """
    run = state.get("run") or {}
    relics = run.get("relics") or []
    potions = run.get("potions") or []
    deck = run.get("deck") or []
    relic_text = (
        "，".join(
            f"[{relic.get('index')}] {_clean_text(relic.get('name'))}"
            for relic in relics
        )
        or "无"
    )
    potion_text = (
        "，".join(
            f"[{potion.get('index')}] "
            f"{_clean_text(potion.get('name')) if potion.get('occupied') else '空'}"
            for potion in potions
        )
        or "无"
    )

    card_counts = Counter(_card_name(card) for card in deck)
    deck_text = (
        "，".join(
            name if count == 1 else f"{name}×{count}"
            for name, count in card_counts.items()
        )
        or "空"
    )
    return f"遗物: {relic_text}\n药水: {potion_text}\n牌组: {deck_text}"


def _render_combat(state: Mapping[str, Any]) -> str:
    """渲染一次战斗决策需要的玩家、敌人、手牌和药水状态。

    Args:
        state (Mapping[str, Any]): 当前战斗状态。

    Returns:
        str: 带稳定索引的战斗观测正文。
    """
    combat = state.get("combat") or {}
    player = combat.get("player") or {}
    player_parts = [
        f"生命 {player.get('current_hp', 0)}/{player.get('max_hp', 0)}",
        f"格挡 {player.get('block', 0)}",
        f"能量 {player.get('energy', 0)}",
        f"星能 {player.get('stars', 0)}",
    ]
    if player.get("focus") is not None:
        player_parts.append(f"集中 {player.get('focus')}")
    orbs = player.get("orbs") or []
    if player.get("orb_capacity") is not None:
        player_parts.append(f"充能球槽 {len(orbs)}/{player.get('orb_capacity')}")
    lines = [
        f"=== 战斗（回合 {state.get('turn', 0)}）===",
        f"玩家: {' | '.join(player_parts)}",
    ]
    powers = _format_powers(player.get("powers") or [])
    if powers:
        lines.append(f"玩家状态: {powers}")
    if orbs:
        lines.append("充能球:")
        lines.extend(_format_orb(orb) for orb in orbs)

    lines.append("敌人:")
    lines.extend(_format_enemy(enemy) for enemy in combat.get("enemies") or [])
    lines.append("手牌:")
    lines.extend(_format_card(card) for card in combat.get("hand") or [])
    lines.append(
        f"牌堆: 抽牌 {combat.get('draw_count', 0)} | "
        f"弃牌 {combat.get('discard_count', 0)}"
    )

    potions = (state.get("run") or {}).get("potions") or []
    if potions:
        lines.append("药水:")
        lines.extend(_format_potion(potion) for potion in potions)
    return "\n".join(lines)


def _render_event(state: Mapping[str, Any]) -> str:
    """渲染事件正文及所有带索引的选项。

    Args:
        state (Mapping[str, Any]): 当前事件状态。

    Returns:
        str: 可供战略模型选择的事件观测。
    """
    event = state.get("event") or {}
    lines = [f"=== 事件: {_clean_text(event.get('title'))} ==="]
    description = _clean_text(event.get("description"))
    if description:
        lines.append(description)
    for option in event.get("options") or []:
        parts = [
            f"[{option.get('index')}] {_clean_text(option.get('title'))}",
        ]
        if option.get("is_locked"):
            parts.append("已锁定")
        description = _clean_text(option.get("description"))
        if description:
            parts.append(description)
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _render_map(state: Mapping[str, Any]) -> str:
    """渲染当前可以前往的地图节点。

    Args:
        state (Mapping[str, Any]): 当前地图状态。

    Returns:
        str: 带选择索引和节点类型的地图观测。
    """
    map_state = state.get("map") or {}
    lines = ["=== 地图 ==="]
    for node in map_state.get("available_nodes") or []:
        node_type = str(node.get("node_type") or "Unknown")
        lines.append(
            f"[{node.get('index')}] 第 {node.get('row')} 行，第 {node.get('col')} 列"
            f" | {_NODE_NAMES.get(node_type, node_type)}"
        )
    return "\n".join(lines)


def _render_reward(state: Mapping[str, Any]) -> str:
    """渲染可领取奖励、卡牌候选和奖励替代项。

    Args:
        state (Mapping[str, Any]): 当前奖励状态。

    Returns:
        str: 带稳定索引的奖励观测。
    """
    reward = state.get("reward") or {}
    lines = ["=== 奖励 ==="]
    for item in reward.get("rewards") or []:
        parts = [f"[{item.get('index')}] {_clean_text(item.get('name'))}"]
        if item.get("claimable") is False:
            parts.append("暂不可领取")
        description = _clean_text(
            item.get("effect_description") or item.get("description")
        )
        if description:
            parts.append(description)
        lines.append(" | ".join(parts))

    cards = reward.get("card_options") or []
    if cards:
        lines.append("卡牌候选:")
        lines.extend(_format_card(card) for card in cards)
    alternatives = reward.get("alternatives") or []
    if alternatives:
        lines.append("其他选项:")
        lines.extend(
            f"[{item.get('index')}] "
            f"{_clean_text(item.get('label') or item.get('name'))}"
            for item in alternatives
        )
    return "\n".join(lines)


def _render_card_selection(state: Mapping[str, Any]) -> str:
    """渲染奖励、事件或战斗机制要求的选牌状态。

    Args:
        state (Mapping[str, Any]): 当前选牌状态。

    Returns:
        str: 选择进度及全部卡牌候选。
    """
    selection = state.get("selection") or {}
    lines = ["=== 选择卡牌 ==="]
    prompt = _clean_text(selection.get("prompt"))
    if prompt:
        lines.append(prompt)
    lines.append(
        f"已选 {selection.get('selected_count', 0)} | "
        f"至少 {selection.get('min_select', 0)} | "
        f"至多 {selection.get('max_select', 0)}"
    )
    lines.extend(_format_card(card) for card in selection.get("cards") or [])
    return "\n".join(lines)


def _render_actions(actions: Sequence[str]) -> str:
    """渲染观测末尾唯一可信的模型动作列表。

    Args:
        actions (Sequence[str]): 已按 Harness 权限过滤的动作名称。

    Returns:
        str: 每行一个动作的可执行动作菜单。
    """
    return "可执行动作:\n" + "\n".join(f"- {action}" for action in actions)


def _format_card(card: Mapping[str, Any]) -> str:
    """把一张卡牌格式化为带索引、费用、效果和目标的单行文本。

    Args:
        card (Mapping[str, Any]): Mod 返回的卡牌描述。

    Returns:
        str: 模型可据此选择的卡牌文本。
    """
    parts = [f"[{card.get('index')}] {_card_name(card)}", _format_cost(card)]
    rules = _clean_text(card.get("resolved_rules_text") or card.get("rules_text"))
    if rules:
        parts.append(rules)
    if card.get("selected"):
        parts.append("已选择")
    if card.get("playable") is False:
        reason = _clean_text(card.get("unplayable_reason"))
        parts.append(f"不可使用{f': {reason}' if reason else ''}")
    targets = card.get("valid_target_indices") or []
    if targets:
        parts.append(f"目标: {list(targets)}")
    return " | ".join(part for part in parts if part)


def _card_name(card: Mapping[str, Any]) -> str:
    """返回包含升级标记的卡牌名称。

    Args:
        card (Mapping[str, Any]): Mod 返回的卡牌描述。

    Returns:
        str: 去除富文本标记后的卡牌名称。
    """
    name = _clean_text(card.get("name")) or "未知卡牌"
    return f"{name}+" if card.get("upgraded") else name


def _format_cost(card: Mapping[str, Any]) -> str:
    """把卡牌的能量与星能费用压缩为一段文本。

    Args:
        card (Mapping[str, Any]): Mod 返回的卡牌描述。

    Returns:
        str: 卡牌的可读费用；没有费用字段时为空。
    """
    costs = []
    if card.get("costs_x"):
        costs.append("X 能量")
    elif card.get("energy_cost") is not None and card.get("energy_cost") >= 0:
        costs.append(f"{card.get('energy_cost')} 能量")
    if card.get("star_costs_x"):
        costs.append("X 星能")
    elif card.get("star_cost", 0):
        costs.append(f"{card.get('star_cost')} 星能")
    return " + ".join(costs)


def _format_enemy(enemy: Mapping[str, Any]) -> str:
    """把一个敌人的生存资源、意图和状态格式化为单行文本。

    Args:
        enemy (Mapping[str, Any]): Mod 返回的敌人描述。

    Returns:
        str: 带目标索引的敌人文本。
    """
    parts = [
        f"[{enemy.get('index')}] {_clean_text(enemy.get('name'))}",
        f"生命 {enemy.get('current_hp', 0)}/{enemy.get('max_hp', 0)}",
        f"格挡 {enemy.get('block', 0)}",
    ]
    intents = "，".join(_format_intent(item) for item in enemy.get("intents") or [])
    if intents:
        parts.append(f"意图: {intents}")
    powers = _format_powers(enemy.get("powers") or [])
    if powers:
        parts.append(f"状态: {powers}")
    return " | ".join(parts)


def _format_intent(intent: Mapping[str, Any]) -> str:
    """把一个敌人意图转换为简洁中文。

    Args:
        intent (Mapping[str, Any]): Mod 返回的单个意图描述。

    Returns:
        str: 包含攻击次数或意图标签的中文文本。
    """
    intent_type = str(intent.get("intent_type") or "Unknown")
    name = _INTENT_NAMES.get(intent_type, intent_type)
    if intent_type == "Attack":
        hits = int(intent.get("hits") or 1)
        total = intent.get("total_damage", intent.get("label", "?"))
        if hits > 1:
            damage = intent.get("damage", "?")
            return f"{name} {damage}×{hits}（共 {total}）"
        return f"{name} {total}"
    label = _clean_text(intent.get("label"))
    return f"{name} {label}" if label else name


def _format_powers(powers: Sequence[Mapping[str, Any]]) -> str:
    """把角色或敌人的能力列表压缩为一行。

    Args:
        powers (Sequence[Mapping[str, Any]]): Mod 返回的能力描述序列。

    Returns:
        str: 以中文逗号分隔的能力名称与层数。
    """
    values = []
    for power in powers:
        name = _clean_text(power.get("name"))
        amount = power.get("amount")
        values.append(f"{name} {amount}" if amount is not None else name)
    return "，".join(values)


def _format_orb(orb: Mapping[str, Any]) -> str:
    """把一个充能球格式化为带槽位索引的单行文本。

    Args:
        orb (Mapping[str, Any]): Mod 返回的充能球描述。

    Returns:
        str: 充能球的被动值与激发值。
    """
    return (
        f"[{orb.get('slot_index')}] {_clean_text(orb.get('name'))} | "
        f"被动 {orb.get('passive_value', 0)} | 激发 {orb.get('evoke_value', 0)}"
    )


def _format_potion(potion: Mapping[str, Any]) -> str:
    """把一个药水槽格式化为带可用状态和目标索引的单行文本。

    Args:
        potion (Mapping[str, Any]): Mod 返回的药水槽描述。

    Returns:
        str: 战斗模型可用于选择或丢弃药水的文本。
    """
    name = _clean_text(potion.get("name")) if potion.get("occupied") else "空"
    parts = [f"[{potion.get('index')}] {name}"]
    if potion.get("occupied"):
        parts.append("可使用" if potion.get("can_use") else "不可使用")
    targets = potion.get("valid_target_indices") or []
    if targets:
        parts.append(f"目标: {list(targets)}")
    return " | ".join(parts)


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
