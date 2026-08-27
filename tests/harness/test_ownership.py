"""验证游戏状态在战略、战斗和过渡 Harness 之间的归属。"""

import importlib
from typing import Any

import pytest


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"screen": "COMBAT", "in_combat": True}, "battle"),
        ({"screen": "EVENT", "in_combat": False}, "strategic"),
        ({"screen": "MAP", "in_combat": False}, "strategic"),
        ({"screen": "CAPSTONE_SELECTION", "in_combat": False}, "transient"),
    ],
)
def test_state_layer_assigns_regular_screens(
    state: dict[str, Any],
    expected: str,
) -> None:
    """普通游戏屏幕按是否处于战斗分配给正确 Harness。

    Args:
        state (dict[str, Any]): 与 Mod ``GET /state`` 同形状的最小状态。
        expected (str): 期望的 Harness 层名称。

    Raises:
        AssertionError: 普通屏幕被分配给错误的 Harness 层。

    Returns:
        None: 此测试只验证状态归属结果。
    """
    ownership = importlib.import_module("play_sts2.harness.ownership")

    assert ownership.state_layer(state).value == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {
                "screen": "CARD_SELECTION",
                "in_combat": True,
                "available_actions": [
                    "choose_reward_card",
                    "select_deck_card",
                ],
            },
            "strategic",
        ),
        (
            {
                "screen": "CARD_SELECTION",
                "in_combat": True,
                "available_actions": ["select_deck_card", "confirm_selection"],
            },
            "battle",
        ),
        (
            {
                "screen": "CARD_SELECTION",
                "in_combat": True,
                "available_actions": [],
            },
            "transient",
        ),
        (
            {
                "screen": "CARD_SELECTION",
                "in_combat": False,
                "available_actions": ["select_deck_card"],
            },
            "strategic",
        ),
    ],
)
def test_state_layer_resolves_card_selection_owner(
    state: dict[str, Any],
    expected: str,
) -> None:
    """奖励选牌优先归战略层，战斗机制选牌归战斗层。

    Args:
        state (dict[str, Any]): 处于 ``CARD_SELECTION`` 的完整关键字段。
        expected (str): 期望的 Harness 层名称。

    Raises:
        AssertionError: 选牌屏幕归属没有遵守奖励与战斗例外。

    Returns:
        None: 此测试只验证选牌屏幕归属结果。
    """
    ownership = importlib.import_module("play_sts2.harness.ownership")

    assert ownership.state_layer(state).value == expected
