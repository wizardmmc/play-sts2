"""验证真实 STS2、Mod HTTP API 与 Python 客户端的完整动作链路。"""

import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from play_sts2 import start_run
from play_sts2.client import GameClient
from play_sts2.scenario import BattleResetter, BattleScenario

from .conftest import RunningGame

pytestmark = pytest.mark.e2e


def test_game_mod_action_contract(running_game: RunningGame) -> None:
    """使用专用测试档在真实游戏中开启故障机器人新局。

    Args:
        running_game (RunningGame): 测试进程启动的隔离游戏实例。

    Raises:
        AssertionError: 真实 Mod 响应、开局状态或存档隔离不满足预期。

    Returns:
        None: 此测试仅验证完整游戏动作链路。
    """
    with GameClient(running_game.base_url) as client:
        health = client.health()
        state = client.state()
        unchanged_state = client.state()
        actions = client.available_actions()
        run_state = start_run(client, "DEFECT")

    assert health.service == "sts2-ai-agent"
    assert health.status == "ready"
    assert isinstance(state.get("state_version"), int)
    assert isinstance(state.get("state_revision"), int)
    assert unchanged_state["state_revision"] == state["state_revision"]
    assert isinstance(state.get("screen"), str)
    assert isinstance(state.get("available_actions"), list)
    assert actions.screen == state["screen"]
    assert [action.name for action in actions.actions] == state["available_actions"]
    assert run_state["run"]["character_id"] == "DEFECT"

    data_root = running_game.home / "Library/Application Support/SlayTheSpire2"
    expected_run_save = data_root / "default/1/modded/profile1/saves/current_run.save"
    assert set(data_root.rglob("current_run.save")) == {expected_run_save}
    assert not (data_root / "steam").exists()


def test_game_mod_rejects_stale_state_revision(
    running_game: RunningGame,
) -> None:
    """在游戏线程执行前拒绝基于过期 revision 的动作且不改变页面。"""
    with httpx.Client(base_url=running_game.base_url) as client:
        response = client.post(
            "/action",
            json={
                "action": "open_character_select",
                "expected_state_revision": 9_223_372_036_854_775_000,
            },
        )
        state = client.get("/state").json()["data"]

    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "stale_state"
    assert error["retryable"] is True
    assert (
        error["details"]["current_state"]["state_revision"] == state["state_revision"]
    )
    assert state["screen"] == "MAIN_MENU"


def test_game_mod_requires_revision_for_state_dependent_action(
    running_game: RunningGame,
) -> None:
    """模型可执行动作缺少 revision 时必须在改变页面前被拒绝。"""
    with httpx.Client(base_url=running_game.base_url) as client:
        response = client.post(
            "/action",
            json={"action": "open_character_select"},
        )
        state = client.get("/state").json()["data"]

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_state_revision"
    assert state["screen"] == "MAIN_MENU"


def test_game_mod_pushes_new_state_without_state_polling(
    running_game: RunningGame,
) -> None:
    """真实状态变化经 SSE 推送，等待期间不产生额外 ``GET /state``。"""
    with (
        GameClient(running_game.base_url) as observer,
        GameClient(running_game.base_url) as actor,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        before = observer.state()
        timeout_started = time.monotonic()
        with pytest.raises(TimeoutError):
            observer.wait_for_state(
                after_revision=before["state_revision"],
                timeout=0.2,
            )
        assert time.monotonic() - timeout_started < 1.0

        log_before = running_game.log_path.read_text(encoding="utf-8")
        state_requests_before = log_before.count("GET /state")
        stream_requests_before = log_before.count("GET /events/stream")

        future = executor.submit(
            observer.wait_for_state,
            after_revision=before["state_revision"],
            timeout=5.0,
        )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            log = running_game.log_path.read_text(encoding="utf-8")
            if log.count("GET /events/stream") > stream_requests_before:
                break
            time.sleep(0.01)
        else:
            pytest.fail("真实游戏未建立 SSE 状态等待连接")

        actor.execute_action(
            "open_character_select",
            expected_state_revision=before["state_revision"],
        )
        updated = future.result(timeout=6.0)

    log_after = running_game.log_path.read_text(encoding="utf-8")
    assert updated["state_revision"] > before["state_revision"]
    assert updated["screen"] == "CHARACTER_SELECT"
    assert log_after.count("GET /state") == state_requests_before


def test_game_mod_pushes_combat_turn_until_exact_ready_boundary(
    running_game: RunningGame,
) -> None:
    """结束回合后仅靠状态事件跨过敌方动画，到达准确的第二回合边界。"""
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
        current_hp=70,
        max_hp=70,
    )

    with (
        GameClient(running_game.base_url) as observer,
        GameClient(running_game.base_url) as actor,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        first_turn = BattleResetter(actor).reset(scenario).state
        log_before = running_game.log_path.read_text(encoding="utf-8")
        state_requests_before = log_before.count("GET /state")
        stream_requests_before = log_before.count("GET /events/stream")
        revision = first_turn["state_revision"]

        future = executor.submit(
            observer.wait_for_state,
            after_revision=revision,
            timeout=15.0,
        )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            log = running_game.log_path.read_text(encoding="utf-8")
            if log.count("GET /events/stream") > stream_requests_before:
                break
            time.sleep(0.01)
        else:
            pytest.fail("真实战斗未建立 SSE 状态等待连接")

        actor.execute_action(
            "end_turn",
            expected_state_revision=revision,
        )
        state = future.result(timeout=16.0)
        delivered_states = [state]
        deadline = time.monotonic() + 15.0
        while not (
            state.get("screen") == "COMBAT"
            and state.get("turn") == 2
            and "end_turn" in (state.get("available_actions") or [])
        ):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pytest.fail("状态事件未在时限内交付第二回合决策边界")
            state = observer.wait_for_state(
                after_revision=state["state_revision"],
                timeout=remaining,
            )
            delivered_states.append(state)

    log_after = running_game.log_path.read_text(encoding="utf-8")
    assert state["turn"] == 2
    assert all(
        "end_turn" not in (delivered.get("available_actions") or [])
        or delivered.get("turn") == 2
        for delivered in delivered_states
    )
    assert log_after.count("GET /state") == state_requests_before


def test_game_mod_pushes_combat_rewards_after_victory_without_state_polling(
    running_game: RunningGame,
) -> None:
    """战斗胜利后仅靠页面事件交付已就绪的奖励状态。

    Args:
        running_game (RunningGame): 已启动并安装测试 Mod 的固定版本游戏。

    Raises:
        AssertionError: 奖励页未由事件流交付或发生了额外轮询。

    Returns:
        None: 此测试只验证战斗胜利后的状态交付契约。
    """
    scenario = BattleScenario(
        character_id="DEFECT",
        seed="ABCDEF1234",
        floor=7,
        encounter_id="MOCK_MONSTER_ENCOUNTER",
        deck=("STRIKE_DEFECTx10",),
        relics=("CRACKED_CORE",),
        current_hp=70,
        max_hp=70,
    )

    with (
        GameClient(running_game.base_url) as observer,
        GameClient(running_game.base_url) as actor,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        combat = BattleResetter(actor).reset(scenario).state
        enemy_hp = combat["combat"]["enemies"][0]["current_hp"]
        damaged = actor.execute_action(
            "run_console_command",
            command=f"damage {enemy_hp - 1}",
        )
        combat = damaged["state"]
        revision = combat["state_revision"]
        card_index = next(
            index
            for index, card in enumerate(combat["combat"]["hand"])
            if card["card_id"] == "STRIKE_DEFECT"
        )
        log_before = running_game.log_path.read_text(encoding="utf-8")
        state_requests_before = log_before.count("GET /state")
        stream_requests_before = log_before.count("GET /events/stream")

        def wait_for_rewards() -> tuple[dict[str, object], list[dict[str, object]]]:
            """持续消费状态事件，直到真实奖励页到达。

            Raises:
                TimeoutError: 事件流未在时限内交付奖励页。

            Returns:
                tuple[dict[str, object], list[dict[str, object]]]:
                    最终奖励状态与期间收到的全部状态。
            """
            deadline = time.monotonic() + 15.0
            delivered_states: list[dict[str, object]] = []
            next_revision = revision
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("状态事件未在时限内交付战斗奖励页")
                state = observer.wait_for_state(
                    after_revision=next_revision,
                    timeout=remaining,
                )
                delivered_states.append(state)
                if state.get("screen") == "REWARD":
                    return state, delivered_states
                next_revision = state["state_revision"]

        future = executor.submit(wait_for_rewards)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            log = running_game.log_path.read_text(encoding="utf-8")
            if log.count("GET /events/stream") > stream_requests_before:
                break
            time.sleep(0.01)
        else:
            pytest.fail("真实战斗未建立 SSE 奖励状态等待连接")

        actor.execute_action(
            "play_card",
            card_index=card_index,
            target_index=0,
            expected_state_revision=revision,
        )
        reward_state, delivered_states = future.result(timeout=16.0)

    log_after = running_game.log_path.read_text(encoding="utf-8")
    assert reward_state["state_revision"] > revision
    assert "claim_reward" in reward_state["available_actions"]
    assert delivered_states[-1] == reward_state
    assert log_after.count("GET /state") == state_requests_before
