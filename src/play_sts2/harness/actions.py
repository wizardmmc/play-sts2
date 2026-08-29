"""筛选模型可见动作，并在规范动作行与 Mod 参数之间转换。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .ownership import HarnessLayer, state_layer

_BATTLE_ACTIONS = {
    "confirm_modal",
    "confirm_selection",
    "dismiss_modal",
    "end_turn",
    "play_card",
    "select_deck_card",
    "skip_card_selection",
    "use_potion",
}
_PLAYER_TURN_ACTIONS = {"end_turn", "play_card", "use_potion"}
_STRATEGIC_EXCLUDED_ACTIONS = {
    "abandon_run",
    "choose_capstone_option",
    "collect_rewards_and_proceed",
    "end_turn",
    "play_card",
    "resolve_rewards",
    "return_to_main_menu",
    "return_to_menu",
    "save_and_quit",
}

_OPTION_ACTIONS = {
    "buy_card",
    "buy_potion",
    "buy_relic",
    "choose_bundle",
    "choose_capstone_option",
    "choose_crystal_sphere_cell",
    "choose_event_option",
    "choose_map_node",
    "choose_rest_option",
    "choose_reward_alternative",
    "choose_reward_card",
    "choose_timeline_epoch",
    "choose_treasure_relic",
    "claim_reward",
    "discard_potion",
    "select_character",
    "select_deck_card",
}
_NO_PARAMETER_ACTIONS = {
    "abandon_run",
    "close_cards_view",
    "close_main_menu_submenu",
    "close_shop_inventory",
    "confirm_bundle",
    "confirm_modal",
    "confirm_selection",
    "confirm_timeline_overlay",
    "continue_run",
    "dismiss_modal",
    "decrease_ascension",
    "embark",
    "end_turn",
    "increase_ascension",
    "open_character_select",
    "open_chest",
    "open_shop_inventory",
    "open_timeline",
    "proceed",
    "remove_card_at_shop",
    "return_to_main_menu",
    "save_and_quit",
    "skip_card_selection",
    "skip_reward_cards",
    "unready",
}


class ActionParseError(ValueError):
    """表示模型输出不符合当前 Harness 动作契约。"""


@dataclass(frozen=True, slots=True)
class HarnessAction:
    """表示已经通过语法和可用动作校验的模型决策。

    Args:
        name (str): Agent Mod 使用的稳定动作名称。
        parameters (dict[str, int]): 动作需要的非空整数参数。
    """

    name: str
    parameters: dict[str, int]


def model_actions(state: Mapping[str, Any]) -> tuple[str, ...]:
    """返回当前状态中应向模型公开的动作。

    结果保留 Mod 给出的顺序。正常出牌阶段允许模型主动弃药；奖励选牌则
    隐藏 Mod 内部复用的选牌动作，避免同一决策出现两套动作语义。

    Args:
        state (Mapping[str, Any]): Mod 返回的完整当前游戏状态。

    Returns:
        tuple[str, ...]: 当前决策层允许模型选择的动作名称。
    """
    layer = state_layer(state)
    available = tuple(str(action) for action in state.get("available_actions") or ())
    if layer is HarnessLayer.TRANSIENT:
        return ()
    if layer is HarnessLayer.BATTLE:
        allowed = set(_BATTLE_ACTIONS)
        if state.get("screen") == "COMBAT" and set(available) & _PLAYER_TURN_ACTIONS:
            allowed.add("discard_potion")
        return tuple(action for action in available if action in allowed)

    excluded = set(_STRATEGIC_EXCLUDED_ACTIONS)
    if "choose_reward_card" in available:
        excluded.add("select_deck_card")
    return tuple(action for action in available if action not in excluded)


def legal_action_lines(state: Mapping[str, Any]) -> tuple[str, ...]:
    """枚举当前模型可见且参数落在真实状态边界内的规范动作行。

    无参数动作直接展开；卡牌、药水和页面选项只保留当前可用索引，必须指定
    目标的动作会为每个合法目标生成一行。该集合用于离线数据校验和严格评测，
    不负责猜测 Mod 没有公开的索引。

    Args:
        state (Mapping[str, Any]): Mod 返回的完整当前游戏状态。

    Raises:
        ActionParseError: 当前开放的参数动作尚无可枚举的状态来源。

    Returns:
        tuple[str, ...]: 按模型动作和页面项目顺序排列的规范动作行。
    """
    lines: list[str] = []
    for name in model_actions(state):
        if name in _NO_PARAMETER_ACTIONS:
            lines.append(format_action(HarnessAction(name=name, parameters={})))
            continue
        items = _action_items(state, name)
        for item in items:
            if not _item_is_available(state, name, item, items):
                continue
            lines.extend(_item_action_lines(name, item))
    return tuple(lines)


def parse_action(
    text: str,
    available_actions: Sequence[str],
    *,
    reasoning: str | None = None,
) -> HarnessAction:
    """解析严格的单行 ``ACTION:`` 输出并校验动作可用性。

    Args:
        text (str): 模型返回的完整文本。
        available_actions (Sequence[str]): 当前状态明确开放的动作名称。
        reasoning (str | None): 思考模板单独返回的推理文本；仅当最终文本为空时，
            可用其最后一个非空行承接严格动作协议。

    Raises:
        ActionParseError: 输出不是单行规范动作、动作不可用或参数无效。

    Returns:
        HarnessAction: 可直接映射到 Mod 请求的结构化动作。
    """
    body = text.strip()
    if not body and reasoning:
        reasoning_lines = [
            line.strip() for line in reasoning.splitlines() if line.strip()
        ]
        if reasoning_lines:
            body = reasoning_lines[-1]
    if "\n" in body or "\r" in body:
        raise ActionParseError("模型输出必须只有一行 ACTION")
    tokens = body.split()
    if len(tokens) < 2 or tokens[0] != "ACTION:":
        raise ActionParseError("模型输出必须以 ACTION: 开头")

    name = tokens[1]
    if name not in available_actions:
        raise ActionParseError(f"动作不在当前可用动作中: {name}")
    try:
        values = [int(token) for token in tokens[2:]]
    except ValueError as exc:
        raise ActionParseError("动作参数必须是整数索引") from exc
    if any(value < 0 for value in values):
        raise ActionParseError("动作索引不能为负数")
    return _build_action(name, values)


def format_action(action: HarnessAction) -> str:
    """把结构化动作格式化成模型与 SFT 共用的规范动作行。

    Args:
        action (HarnessAction): 待格式化的结构化动作。

    Raises:
        ActionParseError: 动作未知、参数名称错误或参数值无效。

    Returns:
        str: 单行 ``ACTION: <动作> <参数>`` 文本。
    """
    parameter_names, required_count = _signature(action.name)
    supplied_count = len(action.parameters)
    expected_names = set(parameter_names[:supplied_count])
    if (
        supplied_count < required_count
        or supplied_count > len(parameter_names)
        or set(action.parameters) != expected_names
    ):
        raise ActionParseError(f"动作参数不符合签名: {action.name}")

    values = [action.parameters[name] for name in parameter_names[:supplied_count]]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ActionParseError("动作参数必须是非负整数索引")
    suffix = "" if not values else " " + " ".join(str(value) for value in values)
    return f"ACTION: {action.name}{suffix}"


def action_signature(name: str, *, include_optional: bool = False) -> str:
    """把动作的解析签名渲染为模型可读格式。

    Args:
        name (str): Agent Mod 使用的稳定动作名称。
        include_optional (bool): 是否显示当前状态实际需要的可选目标参数。

    Raises:
        ActionParseError: 动作尚未纳入 Harness 输出契约。

    Returns:
        str: 标明必需参数与可选参数的动作签名。
    """
    parameter_names, required_count = _signature(name)
    if not parameter_names:
        return name
    count = len(parameter_names) if include_optional else required_count
    return f"{name}({', '.join(parameter_names[:count])})"


def _build_action(name: str, values: list[int]) -> HarnessAction:
    """依据动作签名把有序索引映射成命名参数。

    Args:
        name (str): Agent Mod 的稳定动作名称。
        values (list[int]): 模型按顺序给出的非负整数索引。

    Raises:
        ActionParseError: 动作未知或参数数量不符合动作签名。

    Returns:
        HarnessAction: 参数已经命名的结构化动作。
    """
    parameter_names, required_count = _signature(name)
    if not required_count <= len(values) <= len(parameter_names):
        raise ActionParseError(f"动作参数数量错误: {name}")
    return HarnessAction(
        name=name,
        parameters=dict(zip(parameter_names[: len(values)], values, strict=True)),
    )


def _action_items(
    state: Mapping[str, Any],
    name: str,
) -> Sequence[Mapping[str, Any]]:
    """返回参数动作在当前状态中的候选项目。

    Args:
        state (Mapping[str, Any]): Mod 返回的完整当前游戏状态。
        name (str): 当前模型可见的参数动作名称。

    Raises:
        ActionParseError: 动作没有稳定的状态来源或候选项目结构无效。

    Returns:
        Sequence[Mapping[str, Any]]: 带稳定 ``index`` 的候选项目。
    """
    combat = state.get("combat") or {}
    run = state.get("run") or {}
    shop = state.get("shop") or {}
    sources: dict[str, Any] = {
        "buy_card": shop.get("cards"),
        "buy_potion": shop.get("potions"),
        "buy_relic": shop.get("relics"),
        "choose_bundle": state.get("bundles"),
        "choose_crystal_sphere_cell": (state.get("crystal_sphere") or {}).get(
            "clickable_cells"
        ),
        "choose_event_option": (state.get("event") or {}).get("options"),
        "choose_map_node": (state.get("map") or {}).get("available_nodes"),
        "choose_rest_option": (state.get("rest") or {}).get("options"),
        "choose_reward_alternative": (state.get("reward") or {}).get("alternatives"),
        "choose_reward_card": (state.get("selection") or {}).get("cards"),
        "choose_timeline_epoch": (state.get("timeline") or {}).get("slots"),
        "choose_treasure_relic": (state.get("chest") or {}).get("relic_options"),
        "claim_reward": (state.get("reward") or {}).get("rewards"),
        "discard_potion": run.get("potions"),
        "play_card": combat.get("hand"),
        "select_character": (state.get("character_select") or {}).get("characters"),
        "select_deck_card": (state.get("selection") or {}).get("cards"),
        "use_potion": run.get("potions"),
    }
    if name not in sources:
        raise ActionParseError(f"无法枚举动作的合法索引: {name}")
    raw_items = sources[name] or []
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        raise ActionParseError(f"动作候选项目结构无效: {name}")
    items = tuple(item for item in raw_items if isinstance(item, Mapping))
    if len(items) != len(raw_items):
        raise ActionParseError(f"动作候选项目结构无效: {name}")
    return items


def _item_is_available(
    state: Mapping[str, Any],
    name: str,
    item: Mapping[str, Any],
    peers: Sequence[Mapping[str, Any]],
) -> bool:
    """判断页面项目能否用于指定动作。

    Args:
        state (Mapping[str, Any]): Mod 返回的完整当前游戏状态。
        name (str): 当前参数动作名称。
        item (Mapping[str, Any]): Mod 返回的单个候选项目。
        peers (Sequence[Mapping[str, Any]]): 同一动作的全部候选项目。

    Returns:
        bool: 项目是否属于当前动作的合法参数域。
    """
    if name == "play_card":
        return item.get("playable") is not False
    if name == "use_potion":
        if any(peer.get("can_use") is True for peer in peers):
            return item.get("can_use") is True
        return item.get("occupied") is True
    if name == "discard_potion":
        return item.get("can_discard") is True
    if name in {"buy_card", "buy_potion", "buy_relic"}:
        return item.get("is_stocked") is not False and item.get("enough_gold") is True
    if name == "choose_event_option":
        return item.get("is_locked") is not True
    if name == "choose_reward_alternative":
        return str(item.get("option_id") or "").casefold() != "skip"
    if name == "choose_rest_option":
        return item.get("is_enabled") is not False
    if name == "claim_reward":
        return item.get("claimable") is not False
    if name == "choose_timeline_epoch":
        return item.get("is_actionable") is True
    if name == "select_character":
        return item.get("is_locked") is not True
    if name == "select_deck_card":
        selection = state.get("selection")
        if not isinstance(selection, Mapping):
            return True
        selected_count = selection.get("selected_count")
        max_select = selection.get("max_select")
        grid_kinds = {
            "deck_card_select",
            "deck_enchant_select",
            "deck_transform_select",
            "deck_upgrade_select",
        }
        at_selection_limit = (
            selection.get("kind") in grid_kinds
            and isinstance(selected_count, int)
            and not isinstance(selected_count, bool)
            and isinstance(max_select, int)
            and not isinstance(max_select, bool)
            and max_select > 0
            and selected_count >= max_select
        )
        return not at_selection_limit or item.get("selected") is True
    return True


def _item_action_lines(name: str, item: Mapping[str, Any]) -> tuple[str, ...]:
    """把一个可用项目展开为一个或多个规范动作行。

    Args:
        name (str): 当前参数动作名称。
        item (Mapping[str, Any]): 带索引和可选目标集合的候选项目。

    Raises:
        ActionParseError: 项目索引或目标索引不符合非负整数契约。

    Returns:
        tuple[str, ...]: 当前项目对应的全部合法动作行。
    """
    option_index = _nonnegative_index(item.get("index"), name=name)
    targets = item.get("valid_target_indices") or []
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        raise ActionParseError(f"动作目标索引结构无效: {name}")
    target_indices = tuple(_nonnegative_index(target, name=name) for target in targets)
    if name in {"play_card", "use_potion", "choose_rest_option"} and (
        item.get("requires_target") is True or target_indices
    ):
        return tuple(
            format_action(
                HarnessAction(
                    name=name,
                    parameters={
                        "option_index" if name != "play_card" else "card_index": (
                            option_index
                        ),
                        "target_index": target_index,
                    },
                )
            )
            for target_index in target_indices
        )
    parameter_name = "card_index" if name == "play_card" else "option_index"
    return (
        format_action(
            HarnessAction(name=name, parameters={parameter_name: option_index})
        ),
    )


def _nonnegative_index(value: Any, *, name: str) -> int:
    """校验并返回 Mod 候选项目中的非负整数索引。

    Args:
        value (Any): 待校验的索引值。
        name (str): 用于错误信息定位的动作名称。

    Raises:
        ActionParseError: 索引不是非负整数。

    Returns:
        int: 已通过严格类型和范围校验的索引。
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ActionParseError(f"动作候选缺少合法索引: {name}")
    return value


def _signature(name: str) -> tuple[tuple[str, ...], int]:
    """返回动作的有序参数名及必需参数数量。

    Args:
        name (str): Agent Mod 的稳定动作名称。

    Raises:
        ActionParseError: 动作尚未纳入 Harness 输出契约。

    Returns:
        tuple[tuple[str, ...], int]: 参数名顺序和必需参数数量。
    """
    if name == "play_card":
        return ("card_index", "target_index"), 1
    if name == "use_potion":
        return ("option_index", "target_index"), 1
    if name == "choose_rest_option":
        return ("option_index", "target_index"), 1
    if name in _OPTION_ACTIONS:
        return ("option_index",), 1
    if name in _NO_PARAMETER_ACTIONS:
        return (), 0
    raise ActionParseError(f"动作尚未纳入 Harness 契约: {name}")
