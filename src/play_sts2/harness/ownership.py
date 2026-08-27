"""判断当前游戏状态应由哪个 Harness 层处理。"""

from collections.abc import Mapping
from enum import Enum
from typing import Any

_REWARD_ACTIONS = {
    "choose_reward_alternative",
    "choose_reward_card",
    "skip_reward_cards",
}
_COMBAT_SELECTION_ACTIONS = {
    "confirm_selection",
    "select_deck_card",
    "skip_card_selection",
}


class HarnessLayer(str, Enum):
    """表示战略、战斗或无需模型介入的过渡层。"""

    BATTLE = "battle"
    STRATEGIC = "strategic"
    TRANSIENT = "transient"


def state_layer(state: Mapping[str, Any]) -> HarnessLayer:
    """返回当前游戏状态所属的 Harness 层。

    奖励选牌始终属于战略决策；其他战斗内选牌只有在 Mod 明确开放对应
    动作时才交给战斗层，避免在动画过渡帧唤醒模型。

    Args:
        state (Mapping[str, Any]): Mod 返回的完整当前游戏状态。

    Returns:
        HarnessLayer: 当前状态应交给的 Harness 层。
    """
    screen = str(state.get("screen") or "")
    if screen == "CAPSTONE_SELECTION":
        return HarnessLayer.TRANSIENT
    if screen == "CARD_SELECTION":
        return _card_selection_layer(state)
    if screen == "COMBAT" or state.get("in_combat") is True:
        return HarnessLayer.BATTLE
    return HarnessLayer.STRATEGIC


def _card_selection_layer(state: Mapping[str, Any]) -> HarnessLayer:
    """区分奖励选牌、战斗机制选牌和战斗过渡帧。

    Args:
        state (Mapping[str, Any]): 处于 ``CARD_SELECTION`` 的游戏状态。

    Returns:
        HarnessLayer: 当前选牌状态所属的 Harness 层。
    """
    actions = set(state.get("available_actions") or [])
    if actions & _REWARD_ACTIONS:
        return HarnessLayer.STRATEGIC
    if state.get("in_combat") is not True:
        return HarnessLayer.STRATEGIC
    if actions & _COMBAT_SELECTION_ACTIONS:
        return HarnessLayer.BATTLE
    return HarnessLayer.TRANSIENT
