"""识别战略宏 checkpoint 并自动快进纯 UI 页面。"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from ....checkpoint import checkpoint_option_ids
from ....harness import HarnessAction, model_actions

MacroCheckpointKind = Literal[
    "map",
    "card_selection",
    "card_reward",
    "event",
    "rest",
    "shop",
]


@dataclass(frozen=True, slots=True)
class MacroCheckpoint:
    """表示一个存在至少两个语义后继的战略分支点。

    Args:
        kind (MacroCheckpointKind): 宏节点类型。
        option_ids (tuple[str, ...]): 玩家可见候选的稳定身份。
        scope_id (tuple[str, ...]): 当前 run、幕与 floor 的可读房间身份。
    """

    kind: MacroCheckpointKind
    option_ids: tuple[str, ...]
    scope_id: tuple[str, ...]


def classify_macro_checkpoint(
    state: Mapping[str, Any],
) -> MacroCheckpoint | None:
    """判断当前页面是否值得消耗 Tree K=8 预算。

    Args:
        state (Mapping[str, Any]): 当前完整玩家可见状态。

    Returns:
        MacroCheckpoint | None: 至少有两个语义后继时返回宏节点。
    """
    screen = str(state.get("screen") or "")
    if screen == "REWARD":
        reward = state.get("reward")
        items = reward.get("rewards") if isinstance(reward, Mapping) else None
        card_rewards = [
            item
            for item in items or []
            if isinstance(item, Mapping)
            and item.get("reward_type") == "Card"
            and item.get("claimable") is not False
        ]
        if len(card_rewards) == 1 and "proceed" in model_actions(state):
            index = card_rewards[0].get("index")
            return MacroCheckpoint(
                kind="card_reward",
                option_ids=(f"reward_card:{index}", "reward_card:skip"),
                scope_id=_macro_scope_id(state),
            )
        return None

    kinds: dict[str, MacroCheckpointKind] = {
        "MAP": "map",
        "CARD_SELECTION": "card_selection",
        "EVENT": "event",
        "REST": "rest",
        "SHOP": "shop",
    }
    kind = kinds.get(screen)
    if kind is None:
        return None
    option_ids = checkpoint_option_ids(state)
    if screen == "EVENT":
        event = state.get("event")
        options = event.get("options") if isinstance(event, Mapping) else None
        semantic_count = sum(
            isinstance(item, Mapping)
            and item.get("is_locked") is not True
            and item.get("is_proceed") is not True
            for item in options or []
        )
        has_exit = any(
            isinstance(item, Mapping)
            and item.get("is_locked") is not True
            and item.get("is_proceed") is True
            for item in options or []
        )
        if semantic_count + int(has_exit) < 2:
            return None
    elif len(option_ids) < 2:
        return None
    return MacroCheckpoint(
        kind=kind,
        option_ids=option_ids,
        scope_id=_macro_scope_id(state),
    )


def same_macro_scope(
    root: MacroCheckpoint,
    candidate: MacroCheckpoint,
) -> bool:
    """判断候选页面是否仍属于同一宏计划。

    事件、商店、火堆与卡牌奖励可能打开 ``CARD_SELECTION`` 子页面；这些子页面
    仍和父房间共享一个宏计划，直到进入地图、战斗或另一房间。

    Args:
        root (MacroCheckpoint): 当前宏计划的入口。
        candidate (MacroCheckpoint): 后续稳定战略页面。

    Returns:
        bool: 同一房间内的同类页面或选牌子页面返回真。
    """
    if root.scope_id != candidate.scope_id:
        return False
    if root.kind == candidate.kind:
        return True
    return candidate.kind == "card_selection" and root.kind in {
        "card_reward",
        "event",
        "rest",
        "shop",
    }


def _macro_scope_id(state: Mapping[str, Any]) -> tuple[str, ...]:
    """从玩家可见 run 状态构造无哈希房间身份。

    Args:
        state (Mapping[str, Any]): 当前稳定战略状态。

    Returns:
        tuple[str, ...]: run、幕和 floor 的可读值。
    """
    run = state.get("run")
    if not isinstance(run, Mapping):
        return ("", "", "")
    return (
        str(state.get("run_id") or run.get("run_id") or ""),
        str(run.get("act_id") or run.get("act") or ""),
        str(run.get("floor") or ""),
    )


def automatic_strategic_action(
    state: Mapping[str, Any],
) -> HarnessAction | None:
    """返回无需模型判断的唯一 Proceed、确认或纯 UI 动作。

    Args:
        state (Mapping[str, Any]): 当前完整玩家可见状态。

    Returns:
        HarnessAction | None: 可安全自动执行的唯一动作；否则为空。
    """
    actions = model_actions(state)
    event = state.get("event")
    options = event.get("options") if isinstance(event, Mapping) else None
    if (
        actions == ("choose_event_option",)
        and isinstance(options, list)
        and len(options) == 1
        and isinstance(options[0], Mapping)
        and options[0].get("is_proceed") is True
    ):
        return HarnessAction(
            "choose_event_option",
            {"option_index": int(options[0].get("index", 0))},
        )
    no_parameter = {
        "close_cards_view",
        "confirm_bundle",
        "confirm_modal",
        "confirm_selection",
        "dismiss_modal",
        "open_chest",
        "proceed",
    }
    if len(actions) == 1 and actions[0] in no_parameter:
        return HarnessAction(actions[0], {})
    return None
