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
    "use_potion",
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
    "decrease_ascension",
    "discard_potion",
    "increase_ascension",
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
    "embark",
    "end_turn",
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


def parse_action(text: str, available_actions: Sequence[str]) -> HarnessAction:
    """解析严格的单行 ``ACTION:`` 输出并校验动作可用性。

    Args:
        text (str): 模型返回的完整文本。
        available_actions (Sequence[str]): 当前状态明确开放的动作名称。

    Raises:
        ActionParseError: 输出不是单行规范动作、动作不可用或参数无效。

    Returns:
        HarnessAction: 可直接映射到 Mod 请求的结构化动作。
    """
    body = text.strip()
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
