"""验证真实游戏中的战斗场景与遭遇 RNG 复现。"""

import time
from collections.abc import Mapping
from typing import Any

import pytest

from play_sts2.client import GameClient
from play_sts2.harness import HarnessLayer, build_observation, system_prompt
from play_sts2.scenario import (
    BattleResetter,
    BattleScenario,
    capture_battle_snapshot,
)

from ..conftest import RunningGame

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


def test_combat_state_exposes_card_glow_and_relic_ui_counter(
    running_game: RunningGame,
) -> None:
    """Mod 忠实导出游戏用于手牌和遗物 UI 的通用运行时状态。

    Args:
        running_game (RunningGame): 启用调试场景的隔离真实游戏实例。

    Raises:
        AssertionError: FTL 高亮或双截棍计数、状态、触发结果未被导出。
        TimeoutError: 打牌或跨回合后没有回到稳定玩家阶段。

    Returns:
        None: 此测试验证真实游戏模型到 HTTP payload 的边界。
    """
    scenario = BattleScenario(
        character_id="DEFECT",
        seed="ABCDEF1234",
        floor=7,
        encounter_id="MOCK_MONSTER_ENCOUNTER",
        deck=("FTLx10",),
        relics=("NUNCHAKU",),
        current_hp=70,
        max_hp=70,
    )

    with GameClient(running_game.base_url) as client:
        state = BattleResetter(client).reset(scenario).state
        relic = state["run"]["relics"][0]
        first_card = state["combat"]["hand"][0]
        prompt = system_prompt(HarnessLayer.BATTLE, state)
        observation = build_observation(state).text

        assert first_card["should_glow_gold"] is True
        assert first_card["should_glow_red"] is False
        assert relic["show_counter"] is True
        assert relic["counter_value"] == 0
        assert relic["status"] == "Normal"
        assert relic["stack_count"] == 1
        assert relic["is_used_up"] is False
        assert "【当前回合】" not in prompt
        assert (
            "- [0] 双截棍: 你每打出10张攻击牌，获得1点能量。" in prompt
        )
        assert "遗物UI:\n  [0] 双截棍〔计数 0〕" in observation
        assert "你每打出10张攻击牌" not in observation
        assert "〔金光：有利条件满足〕" in observation
        assert "抽牌堆（5张）:\n  超越光速*5" in observation

        for played in range(1, 10):
            state = _play_first_card(client, state)
            relic = state["run"]["relics"][0]
            assert relic["counter_value"] == played
            if played == 3:
                assert state["combat"]["hand"][0]["should_glow_gold"] is False
            if not state["combat"]["hand"]:
                state = _end_turn(client)

        assert relic["status"] == "Active"
        energy_before = state["combat"]["player"]["energy"]
        state = _play_first_card(client, state)
        state = _await_relic_counter(client, state, expected=0)
        relic = state["run"]["relics"][0]

    assert state["combat"]["player"]["energy"] == energy_before + 1
    assert relic["counter_value"] == 0
    assert relic["status"] == "Normal"


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


def _play_first_card(
    client: GameClient,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """打出手牌第一张卡并等待重新进入稳定玩家阶段。

    Args:
        client (GameClient): 已连接到当前场景战斗的客户端。
        state (Mapping[str, Any]): 打牌前的稳定战斗状态。

    Raises:
        TimeoutError: 动作后未回到可结束回合的稳定状态。

    Returns:
        dict[str, Any]: 动作完成后的稳定战斗状态。
    """
    card = state["combat"]["hand"][0]
    result = client.execute_action(
        "play_card",
        card_index=card["index"],
        target_index=0,
    )
    return _await_player_turn(client, result.get("state"))


def _await_player_turn(
    client: GameClient,
    candidate: object,
) -> dict[str, Any]:
    """等待动作响应重新暴露玩家战斗动作。

    Args:
        client (GameClient): 已连接到当前场景战斗的客户端。
        candidate (object): 动作响应可能携带的首份状态。

    Raises:
        TimeoutError: 动作超时内没有回到玩家阶段。

    Returns:
        dict[str, Any]: 可继续打牌或结束回合的稳定状态。
    """
    deadline = time.monotonic() + client.action_timeout
    state = dict(candidate) if isinstance(candidate, Mapping) else client.state()
    while time.monotonic() < deadline:
        if (
            state.get("screen") == "COMBAT"
            and state.get("in_combat") is True
            and "end_turn" in (state.get("available_actions") or [])
        ):
            return state
        time.sleep(0.2)
        state = client.state()
    raise TimeoutError("等待打牌后玩家阶段超时")


def _await_relic_counter(
    client: GameClient,
    candidate: Mapping[str, Any],
    *,
    expected: int,
) -> dict[str, Any]:
    """等待遗物短暂触发动画结束并稳定到指定 UI 计数。

    Args:
        client (GameClient): 已连接到当前场景战斗的客户端。
        candidate (Mapping[str, Any]): 触发动作后的首份稳定状态。
        expected (int): 动画结束后预期的遗物 UI 计数。

    Raises:
        TimeoutError: 动作超时内遗物计数未稳定。

    Returns:
        dict[str, Any]: 遗物计数达到预期值的最新战斗状态。
    """
    deadline = time.monotonic() + client.action_timeout
    state = dict(candidate)
    while time.monotonic() < deadline:
        relics = (state.get("run") or {}).get("relics") or []
        if relics and relics[0].get("counter_value") == expected:
            return state
        time.sleep(0.2)
        state = client.state()
    raise TimeoutError(f"等待遗物计数稳定到 {expected} 超时")
