"""把完整游戏状态分类为整局运行器的下一处理路径。"""

from collections.abc import Mapping
from enum import Enum
from typing import Any

from ..harness import HarnessLayer, model_actions, state_layer

_STRATEGIC_ACTIONS_BY_SCREEN = {
    "BUNDLE_SELECTION": {"choose_bundle", "confirm_bundle"},
    "CARDS_VIEW": {"close_cards_view"},
    "CHEST": {"choose_treasure_relic", "open_chest", "proceed"},
    "CRYSTAL_SPHERE": {"choose_crystal_sphere_cell", "confirm_selection", "proceed"},
    "EVENT": {"choose_event_option"},
    "MAP": {"choose_map_node"},
    "MODAL": {"confirm_modal", "dismiss_modal"},
    "REST": {"choose_rest_option", "proceed"},
    "REWARD": {
        "choose_reward_alternative",
        "choose_reward_card",
        "claim_reward",
        "proceed",
        "skip_reward_cards",
    },
    "SHOP": {
        "buy_card",
        "buy_potion",
        "buy_relic",
        "close_shop_inventory",
        "open_shop_inventory",
        "proceed",
        "remove_card_at_shop",
    },
    "TIMELINE": {
        "choose_timeline_epoch",
        "confirm_timeline_overlay",
    },
}


class RunRoute(str, Enum):
    """表示整局运行器处理当前状态的方式。"""

    BATTLE = "battle"
    STRATEGIC = "strategic"
    TRANSIENT = "transient"
    TERMINAL = "terminal"
    UNKNOWN = "unknown"


def classify_run_state(state: Mapping[str, Any]) -> RunRoute:
    """依据屏幕、归属和动作面返回确定的整局路由。

    已知页面尚未开放决策动作时属于动画过渡帧。未枚举页面不会根据相似
    动作猜测，只有真实录制中出现过的 ``UNKNOWN + proceed`` 结算层例外。

    Args:
        state (Mapping[str, Any]): Mod 返回的完整当前游戏状态。

    Returns:
        RunRoute: 战斗、战略、过渡、终局或未知路由。
    """
    screen = str(state.get("screen") or "UNKNOWN")
    if screen == "GAME_OVER" or state.get("game_over"):
        return RunRoute.TERMINAL

    layer = state_layer(state)
    actions = set(model_actions(state))
    if layer is HarnessLayer.BATTLE:
        return RunRoute.BATTLE if actions else RunRoute.TRANSIENT
    if layer is HarnessLayer.TRANSIENT:
        return RunRoute.TRANSIENT

    if screen == "CARD_SELECTION":
        return RunRoute.STRATEGIC if actions else RunRoute.TRANSIENT
    if screen == "UNKNOWN":
        if not actions:
            return RunRoute.TRANSIENT
        return RunRoute.STRATEGIC if actions == {"proceed"} else RunRoute.UNKNOWN

    expected = _STRATEGIC_ACTIONS_BY_SCREEN.get(screen)
    if expected is None:
        return RunRoute.UNKNOWN
    return RunRoute.STRATEGIC if expected & actions else RunRoute.TRANSIENT
