"""验证真实游戏中的战斗场景与遭遇 RNG 复现。"""

import time
from collections.abc import Mapping
from typing import Any

import pytest

from play_sts2.client import GameClient
from play_sts2.scenario import (
    BattleResetter,
    BattleScenario,
    capture_battle_snapshot,
)

from .conftest import RunningGame

pytestmark = pytest.mark.e2e


def test_battle_scenario_repeats_entry_and_fixed_second_turn(
    running_game: RunningGame,
) -> None:
    """同一场景重复创建时复现入口及固定动作后的第二回合。

    Args:
        running_game (RunningGame): 测试进程启动的隔离游戏实例。

    Raises:
        AssertionError: 敌人、HP、手牌或意图在两次固定轨迹中不同。
        TimeoutError: 结束回合后未到达第二回合决策点。

    Returns:
        None: 此测试仅验证真实游戏中的场景复现契约。
    """
    scenario = BattleScenario(
        character_id="DEFECT",
        seed="ABCDEF1234",
        floor=7,
        encounter_id="CULTISTS_NORMAL",
        deck=(
            "ZAP",
            "DUALCAST",
            "STRIKE_DEFECTx4",
            "DEFEND_DEFECTx4",
        ),
        relics=("CRACKED_CORE",),
        potion_slots=3,
        current_hp=70,
        max_hp=70,
    )

    with GameClient(running_game.base_url) as client:
        resetter = BattleResetter(client)
        first = resetter.reset(scenario)
        assert "end_turn" in first.state["available_actions"], first.state[
            "available_actions"
        ]
        first_second_turn = capture_battle_snapshot(_end_turn(client))
        second = resetter.reset(
            scenario,
            expected_snapshot=first.snapshot,
        )
        second_second_turn = capture_battle_snapshot(_end_turn(client))

    assert first.snapshot == second.snapshot
    assert first_second_turn.turn == 2
    assert first_second_turn == second_second_turn


def _end_turn(client: GameClient) -> dict[str, Any]:
    """执行固定的空过回合动作并等待第二回合决策状态。

    Args:
        client (GameClient): 已连接到当前场景战斗的客户端。

    Raises:
        TimeoutError: 游戏未在动作超时内到达第二回合。

    Returns:
        dict[str, Any]: 第二回合的稳定战斗状态。
    """
    result = client.execute_action("end_turn")
    candidate = result.get("state")
    deadline = time.monotonic() + client.action_timeout
    while time.monotonic() < deadline:
        state = dict(candidate) if isinstance(candidate, Mapping) else client.state()
        if (
            state.get("screen") == "COMBAT"
            and state.get("in_combat") is True
            and state.get("turn") == 2
            and "end_turn" in (state.get("available_actions") or [])
        ):
            return state
        time.sleep(0.2)
        candidate = None
    raise TimeoutError("等待第二回合决策状态超时")
