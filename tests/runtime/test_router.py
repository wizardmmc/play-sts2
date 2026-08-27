"""验证整局运行时会把游戏状态交给正确的处理层。"""

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
                "available_actions": ["end_turn", "save_and_quit"],
            },
            "BATTLE",
        ),
        (
            {
                "screen": "COMBAT",
                "in_combat": True,
                "available_actions": ["save_and_quit"],
            },
            "TRANSIENT",
        ),
        (
            {
                "screen": "CARD_SELECTION",
                "in_combat": False,
                "available_actions": ["choose_reward_card"],
            },
            "STRATEGIC",
        ),
        (
            {
                "screen": "MAP",
                "in_combat": False,
                "available_actions": ["save_and_quit"],
            },
            "TRANSIENT",
        ),
        (
            {
                "screen": "UNKNOWN",
                "in_combat": False,
                "available_actions": ["proceed"],
            },
            "STRATEGIC",
        ),
        (
            {
                "screen": "GAME_OVER",
                "available_actions": ["return_to_main_menu"],
                "game_over": {"is_victory": False},
            },
            "TERMINAL",
        ),
        (
            {
                "screen": "MAIN_MENU",
                "available_actions": ["open_character_select"],
            },
            "UNKNOWN",
        ),
        (
            {
                "screen": "UNKNOWN",
                "available_actions": ["new_mod_action"],
            },
            "UNKNOWN",
        ),
    ],
)
def test_classify_run_state_routes_real_mod_states(
    state: dict[str, Any],
    expected: str,
) -> None:
    """依据屏幕归属和决策动作区分四种运行状态。

    Args:
        state (dict[str, Any]): 与 Mod `/state` 响应一致的最小状态。
        expected (str): 期望得到的路由枚举成员名。

    Raises:
        AssertionError: 状态被交给错误层，或未知屏幕被擅自猜测。

    Returns:
        None: 此测试只验证纯路由结果。
    """
    runtime = importlib.import_module("play_sts2.runtime")

    assert runtime.classify_run_state(state).name == expected
