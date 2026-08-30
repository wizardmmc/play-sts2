"""把 Mod 状态投影为在线推理与 SFT 共用的可读观测。"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .actions import action_signature, model_actions
from .ownership import HarnessLayer, state_layer
from .strategic_observation import (
    format_strategic_card,
    render_map,
    render_strategic_context,
)
from .text import (
    card_display_name,
    clean_game_text,
    is_unresolved_localization_key,
)

_VISIBLE_PILE_COST_PATTERN = re.compile(r"\s*\[([^\]]+费)\]\s*[：:]\s*")
_INTENT_NAMES = {
    "Attack": "攻击",
    "Buff": "增强",
    "CardDebuff": "诅咒卡牌",
    "DeathBlow": "处决",
    "Debuff": "削弱",
    "DebuffStrong": "强力削弱",
    "Defend": "防御",
    "Escape": "逃跑",
    "Heal": "治疗",
    "Hidden": "隐藏",
    "Sleep": "沉睡",
    "StatusCard": "塞状态牌",
    "Stun": "眩晕",
    "Summon": "召唤",
    "Unknown": "未知",
}
_TARGET_TYPE_NAMES = {
    "AnyEnemy": "任一敌人",
    "Enemy": "敌人",
    "AnyPlayer": "任一角色",
    "SelfAndEnemy": "自身与敌人",
    "AnyAlly": "任一友方",
    "AllAllies": "所有友方",
    "TargetedNoCreature": "指定位置",
}
_UNPLAYABLE_REASON_NAMES = {
    "unplayable": "牌本身不能被打出",
    "not_enough_energy": "能量不足",
    "not_enough_stars": "星能不足",
    "blocked_by_hook": "当前效果禁止打出",
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
        "BUNDLE_SELECTION": _render_bundle_selection,
        "CARD_SELECTION": _render_card_selection,
        "CARDS_VIEW": _render_cards_view,
        "CHEST": _render_chest,
        "COMBAT": _render_combat,
        "CRYSTAL_SPHERE": _render_crystal_sphere,
        "EVENT": _render_event,
        "MAP": render_map,
        "MODAL": _render_modal,
        "REST": _render_rest,
        "REWARD": _render_reward,
        "SHOP": _render_shop,
        "TIMELINE": _render_timeline,
        "UNKNOWN": _render_unknown_proceed,
    }
    renderer = renderers.get(screen)
    if renderer is None:
        raise ObservationError(f"尚未支持的决策屏幕: {screen}")

    sections = []
    if layer is HarnessLayer.STRATEGIC:
        sections.append(render_strategic_context(state))
        if screen != "MAP" and actions != ("proceed",):
            map_state = state.get("map")
            if isinstance(map_state, Mapping) and map_state.get("nodes"):
                sections.append(render_map(state))
    sections.extend((renderer(state), _render_actions(state, actions)))
    return Observation(
        layer=layer,
        text="\n\n".join(section for section in sections if section),
        available_actions=actions,
    )


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
        f"HP {player.get('current_hp', 0)}/{player.get('max_hp', 0)}",
        f"格挡{player.get('block', 0)}",
        f"能量{player.get('energy', 0)}",
        f"星能{player.get('stars', 0)}",
    ]
    orbs = player.get("orbs") or []
    lines = [f"玩家: {' | '.join(player_parts)}"]
    if player.get("card_play_counters_reliable") is True:
        lines.append(
            "本回合已打出: "
            f"卡牌 {player.get('cards_played_this_turn', 0)} | "
            f"攻击 {player.get('attacks_played_this_turn', 0)} | "
            f"技能 {player.get('skills_played_this_turn', 0)}"
        )
    player_buffs = []
    if player.get("focus") is not None:
        player_buffs.append(f"集中{player.get('focus')}")
    power_items = player.get("powers") or []
    if player.get("focus") is not None:
        power_items = [power for power in power_items if not _is_focus_power(power)]
    powers = _format_powers(power_items)
    if powers:
        player_buffs.append(powers)
    if player_buffs:
        lines.append(f"    buff: {' | '.join(player_buffs)}")
    if player.get("orb_capacity") is not None:
        lines.append(
            "    "
            + _format_orbs(
                orbs,
                capacity=player.get("orb_capacity"),
            )
        )

    relic_ui_states = _format_relic_ui_states(
        (state.get("run") or {}).get("relics") or []
    )
    if relic_ui_states:
        lines.append("遗物UI:")
        lines.extend(f"  {relic_state}" for relic_state in relic_ui_states)

    lines.append("敌人:")
    lines.extend(
        _format_enemy(enemy)
        for enemy in combat.get("enemies") or []
        if enemy.get("is_alive") is not False
    )
    lines.append("手牌:")
    lines.extend(f"  {_format_card(card)}" for card in combat.get("hand") or [])
    agent_combat = (state.get("agent_view") or {}).get("combat") or {}
    lines.extend(
        _format_visible_pile(
            "抽牌堆",
            agent_combat.get("draw") or [],
            combat.get("draw_count"),
        )
    )
    lines.extend(
        _format_visible_pile(
            "弃牌堆",
            agent_combat.get("discard") or [],
            combat.get("discard_count"),
        )
    )
    lines.extend(
        _format_visible_pile(
            "消耗牌堆",
            agent_combat.get("exhaust") or [],
            _visible_pile_count(
                agent_combat.get("exhaust") or [],
                agent_combat.get("exhaust_cards") or [],
            ),
        )
    )

    risks = combat.get("lethal_risks") or []
    lines.extend(f"危险: {_format_risk(risk)}" for risk in risks)

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
    raw_title = event.get("title")
    title = "" if is_unresolved_localization_key(raw_title) else _clean_text(raw_title)
    title = title or _clean_text(event.get("event_id")) or "未知事件"
    lines = [f"=== 事件: {title} ==="]
    description = _clean_text(event.get("description"))
    if is_unresolved_localization_key(event.get("description")):
        description = ""
    if description:
        lines.append(description)
    for option in event.get("options") or []:
        raw_option_title = option.get("title")
        option_title = (
            ""
            if is_unresolved_localization_key(raw_option_title)
            else _clean_text(raw_option_title)
        )
        option_title = option_title or f"选项 {option.get('index')}"
        parts = [
            f"[{option.get('index')}] {option_title}",
        ]
        if option.get("is_locked"):
            parts.append("已锁定")
        description = _clean_text(option.get("description"))
        if is_unresolved_localization_key(option.get("description")):
            description = ""
        if description:
            parts.append(description)
        lines.append(" | ".join(parts))
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
        name = _clean_text(item.get("name"))
        description = _clean_text(
            item.get("effect_description") or item.get("description")
        )
        if description and description != name:
            parts.append(description)
        lines.append(" | ".join(parts))

    cards = reward.get("card_options") or []
    if cards:
        lines.append("卡牌候选:")
        lines.extend(
            format_strategic_card(card, fallback_index=index)
            for index, card in enumerate(cards)
        )
    alternatives = reward.get("alternatives") or []
    if alternatives:
        lines.append("其他选项:")
        lines.extend(
            f"[{item.get('index')}] "
            f"{_clean_text(item.get('label') or item.get('name'))}"
            for item in alternatives
        )
    return "\n".join(lines)


def _render_rest(state: Mapping[str, Any]) -> str:
    """渲染休息处的可用选项及其效果。

    Args:
        state (Mapping[str, Any]): 当前休息处状态。

    Returns:
        str: 带选项索引、可用性和目标的休息处观测。
    """
    rest = state.get("rest") or {}
    lines = ["=== 休息处 ==="]
    for option in rest.get("options") or []:
        parts = [f"[{option.get('index')}] {_clean_text(option.get('title'))}"]
        description = _clean_text(option.get("description"))
        if description:
            parts.append(description)
        if option.get("is_enabled") is False:
            parts.append("不可选择")
        targets = option.get("valid_target_indices") or []
        if targets:
            parts.append(f"目标: {list(targets)}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _render_shop(state: Mapping[str, Any]) -> str:
    """渲染商店库存、价格、购买状态和删牌费用。

    Args:
        state (Mapping[str, Any]): 当前商店状态。

    Returns:
        str: 模型可以据此购买或离开商店的观测。
    """
    shop = state.get("shop") or {}
    status = "库存已打开" if shop.get("is_open") else "库存未打开"
    lines = [f"=== 商店（{status}）==="]

    cards = shop.get("cards") or []
    if cards:
        lines.append("卡牌:")
        lines.extend(_format_shop_card(card) for card in cards)
    relics = shop.get("relics") or []
    if relics:
        lines.append("遗物:")
        lines.extend(_format_shop_relic(relic) for relic in relics)
    potions = shop.get("potions") or []
    if potions:
        lines.append("药水:")
        lines.extend(_format_shop_potion(potion) for potion in potions)

    removal = shop.get("card_removal")
    if isinstance(removal, Mapping):
        states = [f"{removal.get('price', 0)}金币"]
        if removal.get("used"):
            states.append("已使用")
        elif removal.get("available") is False:
            states.append("不可用")
        elif removal.get("enough_gold") is False:
            states.append("金币不足")
        else:
            states.append("可用")
        lines.append(f"删牌〔{'；'.join(states)}〕")
    if shop_purchase_available(shop) is False:
        message = "当前没有任何可购买项目；重新打开库存不会刷新商品。"
        if "proceed" in (state.get("available_actions") or []):
            message += "请输出 `ACTION: proceed` 离开商店。"
        else:
            message += "若不再购买，请关闭库存后离开商店。"
        lines.append(message)
    return "\n".join(lines)


def shop_purchase_available(shop: Mapping[str, Any]) -> bool | None:
    """判断已知商店库存中是否仍有可购买项目。

    Args:
        shop (Mapping[str, Any]): Mod 返回的商店库存状态。

    Returns:
        bool | None: 有项目可购买时为 ``True``，完整库存均不可购买时为
        ``False``，库存尚未加载时为 ``None``。
    """
    inventory_items = [
        *(shop.get("cards") or []),
        *(shop.get("relics") or []),
        *(shop.get("potions") or []),
    ]
    removal = shop.get("card_removal")
    if not inventory_items and not isinstance(removal, Mapping):
        return None
    return any(
        item.get("is_stocked") is not False and item.get("enough_gold") is True
        for item in inventory_items
    ) or (
        isinstance(removal, Mapping)
        and removal.get("available") is True
        and removal.get("enough_gold") is True
    )


def _render_chest(state: Mapping[str, Any]) -> str:
    """渲染宝箱开关状态和可选择的遗物。

    Args:
        state (Mapping[str, Any]): 当前宝箱状态。

    Returns:
        str: 带遗物索引、稀有度和描述的宝箱观测。
    """
    chest = state.get("chest") or {}
    status = "已打开" if chest.get("is_opened") else "未打开"
    lines = [f"=== 宝箱（{status}）==="]
    for relic in chest.get("relic_options") or []:
        parts = [f"[{relic.get('index')}] {_clean_text(relic.get('name'))}"]
        rarity = _clean_text(relic.get("rarity"))
        if rarity:
            parts.append(rarity)
        description = _clean_text(relic.get("description"))
        if description:
            parts.append(description)
        lines.append(" | ".join(parts))
    if chest.get("has_relic_been_claimed"):
        lines.append("遗物已领取")
    return "\n".join(lines)


def _render_bundle_selection(state: Mapping[str, Any]) -> str:
    """渲染开局或事件中的卡牌包候选。

    Args:
        state (Mapping[str, Any]): 当前卡牌包选择状态。

    Returns:
        str: 每个卡牌包及其卡牌内容。
    """
    lines = ["=== 选择卡牌包 ==="]
    for bundle in state.get("bundles") or []:
        lines.append(f"卡牌包 [{bundle.get('index')}]:")
        lines.extend(
            format_strategic_card(card, fallback_index=index)
            for index, card in enumerate(bundle.get("cards") or [])
        )
    return "\n".join(lines)


def _render_crystal_sphere(state: Mapping[str, Any]) -> str:
    """渲染水晶球剩余次数、可点击格和已揭示物品。

    Args:
        state (Mapping[str, Any]): 当前水晶球小游戏状态。

    Returns:
        str: 不泄漏迷雾内容的水晶球观测。
    """
    sphere = state.get("crystal_sphere") or {}
    lines = [
        "=== 水晶球 ===",
        (
            f"剩余占卜: {sphere.get('divinations_remaining', 0)} | "
            f"工具: {_clean_text(sphere.get('tool'))}"
        ),
    ]
    cells = sphere.get("clickable_cells") or []
    if cells:
        lines.append("可点击格:")
        lines.extend(
            f"[{cell.get('index')}] 坐标 ({cell.get('x')}, {cell.get('y')})"
            for cell in cells
        )
    items = sphere.get("revealed_items") or []
    if items:
        lines.append("已揭示物品:")
        lines.extend(
            f"{_clean_text(item.get('kind'))} @ ({item.get('x')}, {item.get('y')})"
            for item in items
        )
    return "\n".join(lines)


def _render_modal(state: Mapping[str, Any]) -> str:
    """渲染覆盖当前页面的确认或取消弹窗。

    Args:
        state (Mapping[str, Any]): 当前弹窗状态。

    Returns:
        str: 弹窗类型、来源页和按钮标签。
    """
    modal = state.get("modal") or {}
    lines = ["=== 确认弹窗 ==="]
    type_name = _clean_text(modal.get("type_name"))
    if type_name:
        lines.append(f"类型: {type_name}")
    underlying = _clean_text(modal.get("underlying_screen"))
    if underlying:
        lines.append(f"来源页面: {underlying}")
    confirm = _clean_text(modal.get("confirm_label"))
    if confirm:
        lines.append(f"确认: {confirm}")
    dismiss = _clean_text(modal.get("dismiss_label"))
    if dismiss:
        lines.append(f"取消: {dismiss}")
    return "\n".join(lines)


def _render_cards_view(state: Mapping[str, Any]) -> str:
    """渲染只读的牌组查看覆盖层。

    Args:
        state (Mapping[str, Any]): 当前牌组查看状态。

    Returns:
        str: 覆盖层提示和全部可见卡牌。
    """
    cards_view = state.get("cards_view") or {}
    lines = ["=== 查看牌组 ==="]
    prompt = _clean_text(cards_view.get("prompt"))
    if prompt:
        lines.append(prompt)
    cards = cards_view.get("cards") or []
    if state_layer(state) is HarnessLayer.STRATEGIC:
        lines.extend(
            format_strategic_card(card, fallback_index=index)
            for index, card in enumerate(cards)
        )
    else:
        lines.extend(_format_card(card) for card in cards)
    return "\n".join(lines)


def _render_timeline(state: Mapping[str, Any]) -> str:
    """渲染时间线槽位及其是否可以选择。

    Args:
        state (Mapping[str, Any]): 当前时间线状态。

    Returns:
        str: 带槽位索引、名称、状态和可用性的时间线观测。
    """
    timeline = state.get("timeline") or {}
    lines = ["=== 时间线 ==="]
    for slot in timeline.get("slots") or []:
        parts = [
            f"[{slot.get('index')}] {_clean_text(slot.get('title'))}",
            _clean_text(slot.get("state")),
            "可选择" if slot.get("is_actionable") else "不可选择",
        ]
        lines.append(" | ".join(part for part in parts if part))
    return "\n".join(lines)


def _render_unknown_proceed(_state: Mapping[str, Any]) -> str:
    """渲染真实房间结算中只有 ``proceed`` 的匿名页面。

    Args:
        _state (Mapping[str, Any]): 当前匿名结算状态。

    Returns:
        str: 不猜测页面类型的继续提示。
    """
    return "=== 房间结算 ===\n当前只需继续进入下一状态。"


def _render_card_selection(state: Mapping[str, Any]) -> str:
    """渲染奖励、事件或战斗机制要求的选牌状态。

    战斗层选牌会保留当前完整战斗观测，使从弃牌堆取回卡牌等决策仍能看到
    剩余能量、手牌、敌人和牌堆；战略层只展示战略上下文与候选牌。

    Args:
        state (Mapping[str, Any]): 当前选牌状态。

    Returns:
        str: 当前层所需上下文、页面提示及全部卡牌候选。
    """
    selection = state.get("selection") or {}
    prompt = _clean_text(selection.get("prompt"))
    cards = selection.get("cards") or []
    if state_layer(state) is HarnessLayer.STRATEGIC:
        lines = ["=== 选择卡牌 ==="]
        card_rules = {
            _clean_text(card.get("resolved_rules_text") or card.get("rules_text"))
            for card in cards
            if isinstance(card, Mapping)
        }
        is_upgrade = selection.get("kind") == "deck_upgrade_select"
        has_upgrade_previews = (
            is_upgrade
            and bool(cards)
            and all(isinstance(card.get("upgrade_preview"), Mapping) for card in cards)
        )
        if prompt and (is_upgrade or prompt not in card_rules):
            if has_upgrade_previews:
                prompt_suffix = "以下展示升级后效果："
            elif is_upgrade:
                prompt_suffix = "旧录像未保存升级预览，以下展示当前效果："
            else:
                prompt_suffix = ""
            lines.append(f"{prompt}{prompt_suffix}")
        elif is_upgrade:
            lines.append(
                "以下展示升级后效果："
                if has_upgrade_previews
                else "旧录像未保存升级预览，以下展示当前效果："
            )
        for index, card in enumerate(cards):
            rendered_card = card
            if has_upgrade_previews:
                preview = card.get("upgrade_preview")
                rendered_card = dict(preview)
                rendered_card["index"] = card.get("index", index)
                rendered_card["selected"] = card.get("selected") is True
            lines.append(
                format_strategic_card(
                    rendered_card,
                    fallback_index=index,
                )
            )
        return "\n".join(lines)

    lines = ["=== 选择卡牌 ==="]
    card_rules = {
        _clean_text(card.get("resolved_rules_text") or card.get("rules_text"))
        for card in cards
        if isinstance(card, Mapping)
    }
    if prompt and prompt not in card_rules:
        lines.append(prompt)
    lines.extend(_format_card(card) for card in cards)
    selection_text = "\n".join(lines)
    if state_layer(state) is HarnessLayer.BATTLE:
        return f"{_render_combat(state)}\n\n{selection_text}"
    return selection_text


def _render_actions(state: Mapping[str, Any], actions: Sequence[str]) -> str:
    """渲染观测末尾唯一可信的模型动作列表。

    Args:
        state (Mapping[str, Any]): 用于判断可选目标参数是否实际需要的状态。
        actions (Sequence[str]): 已按 Harness 权限过滤的动作名称。

    Returns:
        str: 每行一个动作的可执行动作菜单。
    """
    combat = state.get("combat") or {}
    run = state.get("run") or {}
    rest = state.get("rest") or {}
    target_sources = {
        "play_card": combat.get("hand") or [],
        "use_potion": run.get("potions") or [],
        "choose_rest_option": rest.get("options") or [],
    }
    targeted = {
        action
        for action, items in target_sources.items()
        if any(
            item.get("requires_target") is True
            or bool(item.get("valid_target_indices"))
            for item in items
        )
    }
    return "可执行动作:\n" + "\n".join(
        f"- {action_signature(action, include_optional=action in targeted)}"
        for action in actions
    )


def _format_card(card: Mapping[str, Any]) -> str:
    """把一张卡牌格式化为紧凑、稳定的单行文本。

    Args:
        card (Mapping[str, Any]): Mod 返回的卡牌描述。

    Returns:
        str: 模型可据此选择的卡牌文本。
    """
    prefix = f"[{card.get('index')}]{_card_name(card)}"
    cost = _format_cost(card)
    if cost:
        prefix += f"({cost})"
    target_type = _clean_text(card.get("target_type"))
    requires_target = card.get("requires_target") is True or bool(
        card.get("valid_target_indices")
    )
    if requires_target and target_type not in {"", "Self", "None"}:
        prefix += f"<目标:{_TARGET_TYPE_NAMES.get(target_type, '需指定目标')}>"
    parts = [prefix]
    rules = _clean_text(card.get("resolved_rules_text") or card.get("rules_text"))
    if rules:
        parts.append(rules)
    highlight = _format_card_highlight(card)
    if highlight:
        parts.append(highlight)
    if card.get("selected"):
        parts.append("(已选择)")
    if card.get("playable") is False:
        raw_reason = _clean_text(card.get("unplayable_reason"))
        reason = _UNPLAYABLE_REASON_NAMES.get(raw_reason, "当前不可使用")
        parts.append(f"(不可使用{f': {reason}' if reason else ''})")
    return " ".join(part for part in parts if part)


def _format_card_highlight(card: Mapping[str, Any]) -> str:
    """把游戏手牌 UI 的金红高亮翻译为通用语义。

    Args:
        card (Mapping[str, Any]): Mod 返回的手牌描述。

    Returns:
        str: 红光优先于金光；没有特殊高亮时为空。
    """
    if card.get("should_glow_red") is True:
        return "〔红光：不利条件生效〕"
    if card.get("playable") is True and card.get("should_glow_gold") is True:
        return "〔金光：有利条件满足〕"
    return ""


def _card_name(card: Mapping[str, Any]) -> str:
    """返回包含升级标记的卡牌名称。

    Args:
        card (Mapping[str, Any]): Mod 返回的卡牌描述。

    Returns:
        str: 去除富文本标记后的卡牌名称。
    """
    return card_display_name(
        card.get("name"),
        upgraded=card.get("upgraded") is True,
    )


def _format_cost(card: Mapping[str, Any]) -> str:
    """把卡牌的能量与星能费用压缩为一段文本。

    Args:
        card (Mapping[str, Any]): Mod 返回的卡牌描述。

    Returns:
        str: 卡牌的可读费用；没有费用字段时为空。
    """
    costs = []
    if card.get("costs_x"):
        costs.append("X费")
    elif card.get("energy_cost") is not None and card.get("energy_cost") >= 0:
        costs.append(f"{card.get('energy_cost')}费")
    if card.get("star_costs_x"):
        costs.append("X星能")
    elif card.get("star_cost", 0):
        costs.append(f"{card.get('star_cost')}星能")
    return "+".join(costs)


def _format_enemy(enemy: Mapping[str, Any]) -> str:
    """把一个敌人的生存资源、状态和意图格式化为缩进文本。

    Args:
        enemy (Mapping[str, Any]): Mod 返回的敌人描述。

    Returns:
        str: 带目标索引且将状态、意图分行的敌人文本。
    """
    lines = [
        (
            f"  敌[{enemy.get('index')}] {_clean_text(enemy.get('name'))}: "
            f"HP {enemy.get('current_hp', 0)}/{enemy.get('max_hp', 0)} | "
            f"格挡{enemy.get('block', 0)}"
        )
    ]
    powers = _format_powers(enemy.get("powers") or [])
    if powers:
        lines.append(f"    buff: {powers}")
    intents = "，".join(_format_intent(item) for item in enemy.get("intents") or [])
    if intents:
        lines.append(f"    意图:{intents}")
    return "\n".join(lines)


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
            return f"{name}{damage}×{hits}={total}"
        return f"{name}{total}"
    label = _clean_text(intent.get("label"))
    return f"{name}{label}" if label else name


def _format_powers(powers: Sequence[Mapping[str, Any]]) -> str:
    """把角色或敌人的能力列表压缩为一行。

    Args:
        powers (Sequence[Mapping[str, Any]]): Mod 返回的能力描述序列。

    Returns:
        str: 以竖线分隔的能力名称与层数。
    """
    values = []
    for power in powers:
        name = _clean_text(power.get("name"))
        amount = power.get("amount")
        values.append(f"{name}{amount}" if amount is not None else name)
    return " | ".join(values)


def _is_focus_power(power: Mapping[str, Any]) -> bool:
    """判断能力项是否重复表达玩家的结构化集中数值。

    Args:
        power (Mapping[str, Any]): Mod 返回的单个能力项。

    Returns:
        bool: 能力 ID 或历史名称表示集中时为 ``True``。
    """
    return (
        power.get("power_id") == "FOCUS_POWER"
        or _clean_text(power.get("name")) == "集中"
    )


def _format_orb(orb: Mapping[str, Any]) -> str:
    """把一个充能球格式化为带槽位索引的单行文本。

    Args:
        orb (Mapping[str, Any]): Mod 返回的充能球描述。

    Returns:
        str: 充能球的被动值与激发值。
    """
    return (
        f"[{orb.get('slot_index')}]{_clean_text(orb.get('name'))}"
        f"(被动{orb.get('passive_value', 0)}/激发{orb.get('evoke_value', 0)})"
    )


def _format_orbs(orbs: Sequence[Mapping[str, Any]], *, capacity: Any) -> str:
    """按 FIFO 顺序渲染全部充能球和下一次激发位置。

    Args:
        orbs (Sequence[Mapping[str, Any]]): 当前已占用的充能球槽。
        capacity (Any): 当前充能球总容量。

    Returns:
        str: 含容量、FIFO 次序与激发值的单行摘要。
    """
    ordered = sorted(orbs, key=lambda orb: int(orb.get("slot_index", 0)))
    if not ordered:
        return f"充能球(FIFO最旧→最新;下一激发=无): 0/{capacity} (空;无可激发球)"
    next_index = ordered[0].get("slot_index", 0)
    values = ",".join(_format_orb(orb) for orb in ordered)
    return (
        f"充能球(FIFO最旧→最新;下一激发=[{next_index}]): "
        f"{len(ordered)}/{capacity} {values}"
    )


def _format_visible_pile(
    name: str,
    entries: Sequence[Any],
    count: Any,
) -> list[str]:
    """渲染模型实际可见的战斗牌堆内容。

    Args:
        name (str): 牌堆的中文名称。
        entries (Sequence[Any]): Mod ``agent_view`` 中已分组的可见牌行。
        count (Any): 结构化状态给出的准确牌数。

    Returns:
        list[str]: 首行带牌数、后续行缩进的牌堆文本。
    """
    lines = [_visible_pile_line(entry) for entry in entries]
    lines = [line for line in lines if line]
    total = count if isinstance(count, int) else _visible_pile_count(entries, [])
    if not lines and not total:
        return []
    if not lines:
        return [f"{name}（{total}张）: 内容不可见"]
    return [f"{name}（{total}张）:", *(f"  {line}" for line in lines)]


def _format_relic_ui_states(
    relics: Sequence[Mapping[str, Any]],
) -> list[str]:
    """渲染游戏实际显示的遗物计数、激活和禁用状态。

    Args:
        relics (Sequence[Mapping[str, Any]]): 当前玩家持有的遗物。

    Returns:
        list[str]: 仅包含存在动态 UI 状态的遗物行。
    """
    lines = []
    status_names = {
        "active": "已高亮",
        "disabled": "已禁用",
    }
    for fallback_index, relic in enumerate(relics):
        states = []
        counter_value = relic.get("counter_value")
        if (
            relic.get("show_counter") is True
            and isinstance(counter_value, int)
            and not isinstance(counter_value, bool)
        ):
            states.append(f"计数 {counter_value}")
        status = _clean_text(relic.get("status"))
        normalized_status = status.lower()
        if normalized_status in status_names:
            states.append(status_names[normalized_status])
        elif normalized_status and normalized_status != "normal":
            states.append(f"状态 {status}")
        if relic.get("is_used_up") is True:
            states.append("已耗尽")
        if not states:
            continue
        index = relic.get("index", fallback_index)
        name = _clean_text(relic.get("name")) or "未知遗物"
        lines.append(f"[{index}] {name}〔{'；'.join(states)}〕")
    return lines


def _visible_pile_line(entry: Any) -> str:
    """提取一个 ``agent_view`` 牌堆条目的可读行。

    Args:
        entry (Any): 字符串或含 ``line`` 字段的映射。

    Returns:
        str: 清理富文本后的牌堆条目。
    """
    if isinstance(entry, Mapping):
        line = _clean_text(entry.get("line") or entry.get("name"))
    else:
        line = _clean_text(entry)
    return _VISIBLE_PILE_COST_PATTERN.sub(r"(\1) ", line)


def _visible_pile_count(
    entries: Sequence[Any],
    structured_cards: Sequence[Any],
) -> int:
    """从结构化卡牌或分组文本计算牌堆中的实际牌数。

    Args:
        entries (Sequence[Any]): 可能含 ``名称*数量`` 的可见牌行。
        structured_cards (Sequence[Any]): Mod 提供的逐卡牌列表。

    Returns:
        int: 牌堆中的卡牌总数。
    """
    if structured_cards:
        return len(structured_cards)
    total = 0
    for entry in entries:
        line = _visible_pile_line(entry)
        match = re.search(r"\*(\d+)", line)
        total += int(match.group(1)) if match else 1
    return total


def _format_risk(risk: Any) -> str:
    """把 Mod 的致命风险提示统一为单行文本。

    Args:
        risk (Any): 字符串或结构化风险对象。

    Returns:
        str: 战斗模型可直接理解的危险提示。
    """
    if isinstance(risk, Mapping):
        if risk.get("risk_id") == "incoming_damage":
            damage = risk.get("damage_after_block", risk.get("incoming_damage", "?"))
            suffix = "，足以致命。" if risk.get("will_kill_player") else "。"
            return f"预计承受{damage}点未格挡伤害{suffix}"
        return _clean_text(
            risk.get("line")
            or risk.get("description")
            or risk.get("message")
            or risk.get("reason")
        )
    return _clean_text(risk)


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
        description = _clean_text(potion.get("description"))
        if description:
            parts.append(description)
    targets = potion.get("valid_target_indices") or []
    if targets:
        parts.append(f"目标: {list(targets)}")
    return " | ".join(parts)


def _format_shop_card(card: Mapping[str, Any]) -> str:
    """把商店卡牌格式化为含价格和购买状态的单行文本。

    Args:
        card (Mapping[str, Any]): Mod 返回的商店卡牌描述。

    Returns:
        str: 可用于 ``buy_card`` 选择的卡牌文本。
    """
    if card.get("is_stocked") is False and not _clean_text(card.get("name")):
        return f"- [{card.get('index')}] 已售出"
    states = [f"{card.get('price', 0)}金币"]
    status = _purchase_status(card)
    if status:
        states.append(status)
    return format_strategic_card(card, extra_states=states)


def _format_shop_relic(relic: Mapping[str, Any]) -> str:
    """把商店遗物格式化为含索引、说明、价格和状态的文本。

    Args:
        relic (Mapping[str, Any]): Mod 返回的商店遗物描述。

    Returns:
        str: 可用于 ``buy_relic`` 选择的遗物文本。
    """
    if relic.get("is_stocked") is False and not _clean_text(relic.get("name")):
        return f"- [{relic.get('index')}] 已售出"
    states = [f"{relic.get('price', 0)}金币"]
    status = _purchase_status(relic)
    if status:
        states.append(status)
    state_text = f"〔{'；'.join(states)}〕"
    prefix = f"- [{relic.get('index')}] {_clean_text(relic.get('name'))}{state_text}"
    description = _clean_text(relic.get("description"))
    return f"{prefix}：{description}" if description else prefix


def _format_shop_potion(potion: Mapping[str, Any]) -> str:
    """把商店药水格式化为含索引、说明、价格和状态的文本。

    Args:
        potion (Mapping[str, Any]): Mod 返回的商店药水描述。

    Returns:
        str: 可用于 ``buy_potion`` 选择的药水文本。
    """
    if potion.get("is_stocked") is False and not _clean_text(potion.get("name")):
        return f"- [{potion.get('index')}] 已售出"
    states = [f"{potion.get('price', 0)}金币"]
    status = _purchase_status(potion)
    if status:
        states.append(status)
    state_text = f"〔{'；'.join(states)}〕"
    prefix = f"- [{potion.get('index')}] {_clean_text(potion.get('name'))}{state_text}"
    description = _clean_text(potion.get("description"))
    return f"{prefix}：{description}" if description else prefix


def _purchase_status(item: Mapping[str, Any]) -> str:
    """返回一个商店物品当前不可购买的原因。

    Args:
        item (Mapping[str, Any]): 含库存和金币状态的商店物品。

    Returns:
        str: ``已售出``、``金币不足`` 或空字符串。
    """
    if item.get("is_stocked") is False:
        return "已售出"
    if item.get("enough_gold") is False:
        return "金币不足"
    return ""


def _clean_text(value: Any) -> str:
    """移除游戏富文本标签和资源路径并合并空白。

    Args:
        value (Any): 可能包含游戏标记的字段值。

    Returns:
        str: 适合直接放入模型观测的纯文本。
    """
    return clean_game_text(value)
