"""验证 Harness 只向模型公开属于当前决策层的动作。"""

import importlib
from typing import Any

import pytest


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {
                "screen": "COMBAT",
                "in_combat": True,
                "available_actions": [
                    "save_and_quit",
                    "play_card",
                    "discard_potion",
                    "end_turn",
                ],
            },
            ("play_card", "discard_potion", "end_turn"),
        ),
        (
            {
                "screen": "CARD_SELECTION",
                "in_combat": True,
                "available_actions": [
                    "select_deck_card",
                    "discard_potion",
                    "confirm_selection",
                ],
            },
            ("select_deck_card", "confirm_selection"),
        ),
    ],
)
def test_model_actions_filters_battle_actions(
    state: dict[str, Any],
    expected: tuple[str, ...],
) -> None:
    """战斗层只保留模型决策，并仅在正常出牌阶段允许弃药。

    Args:
        state (dict[str, Any]): Mod 返回的战斗状态。
        expected (tuple[str, ...]): 按 Mod 原顺序排列的模型可见动作。

    Raises:
        AssertionError: 过滤结果包含越权动作或破坏了原始顺序。

    Returns:
        None: 此测试只验证战斗动作过滤结果。
    """
    harness = importlib.import_module("play_sts2.harness")

    assert harness.model_actions(state) == expected


def test_model_actions_filters_strategic_actions() -> None:
    """战略层隐藏控制动作、战斗动作和奖励选牌的底层动作。

    Raises:
        AssertionError: 战略模型看见了不属于当前决策的动作。

    Returns:
        None: 此测试只验证战略动作过滤结果。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "CARD_SELECTION",
        "in_combat": False,
        "available_actions": [
            "save_and_quit",
            "return_to_menu",
            "choose_reward_card",
            "select_deck_card",
            "skip_reward_cards",
            "end_turn",
        ],
    }

    assert harness.model_actions(state) == (
        "choose_reward_card",
        "skip_reward_cards",
    )


def test_model_actions_returns_empty_for_transient_state() -> None:
    """无需模型介入的过渡状态不公开任何动作。

    Raises:
        AssertionError: 过渡状态错误地唤醒了模型。

    Returns:
        None: 此测试只验证过渡层没有模型动作。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "CAPSTONE_SELECTION",
        "in_combat": False,
        "available_actions": ["choose_capstone_option"],
    }

    assert harness.model_actions(state) == ()
