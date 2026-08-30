"""验证 v0.111.0 原生整局 checkpoint 与跨 HOME 确定性。"""

import os
import time
from collections import deque
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import httpx
import pytest

from play_sts2 import start_run
from play_sts2.checkpoint import (
    CheckpointEntry,
    StrategicCheckpoint,
    capture_strategic_checkpoint,
    derive_strategic_checkpoint,
    observe_checkpoint_entry,
    restore_strategic_checkpoint,
)
from play_sts2.client import GameClient
from play_sts2.game_launcher import DEFAULT_APP, DEFAULT_PROFILE, launch_game
from play_sts2.harness import HarnessAction

from .conftest import RunningGame

pytestmark = pytest.mark.e2e


def test_checkpoint_audit_is_hidden_from_policy_state(
    running_game: RunningGame,
) -> None:
    """隐藏 RNG 与完整牌堆只能从开发审计端点读取。

    Args:
        running_game (RunningGame): v0.111.0 隔离无头游戏实例。

    Returns:
        None: 普通 `/state` 保持玩家可见，审计端点保留恢复验收事实。
    """
    with GameClient(running_game.base_url) as game:
        health = game.health()
        state = start_run(game, "DEFECT", seed="CKPTAUDIT1", ascension=0)
        audit = game.checkpoint_audit()

    assert health.game_version == "v0.111.0"
    assert health.mod_version == "0.8.0-rlsts2.50"
    assert audit["run_id"] == state["run_id"] == "CKPTAUD1T1"
    assert len(audit["run_rng"]) == 12
    assert len(audit["players"][0]["player_rng"]) == 3
    assert audit["players"][0]["piles"][0]["pile_type"] == "Deck"
    assert "run_rng" not in state
    assert "player_rng" not in state["run"]["players"][0]
    assert "piles" not in state["run"]["players"][0]


def test_checkpoint_audit_is_disabled_without_development_switch(
    running_game_without_debug: RunningGame,
) -> None:
    """普通游戏进程不得开放隐藏 RNG 与完整牌堆端点。

    Args:
        running_game_without_debug (RunningGame): 未开放开发动作的真实实例。

    Returns:
        None: 未设置开发开关时端点返回明确冲突错误。
    """
    with GameClient(running_game_without_debug.base_url) as game:
        start_run(game, "DEFECT", seed="ABCDEF1234", ascension=0)
        with pytest.raises(httpx.HTTPStatusError) as error:
            game.checkpoint_audit()

    assert error.value.response.status_code == 409
    assert error.value.response.json()["error"]["code"] == "checkpoint_audit_disabled"


def test_event_checkpoint_restores_same_entry_in_two_homes(
    running_game: RunningGame,
    tmp_path: Path,
) -> None:
    """同一事件 checkpoint 应经原生 continue 恢复到两个完全相同的入口。

    Args:
        running_game (RunningGame): 用于捕获原生存档的源游戏实例。
        tmp_path (Path): checkpoint 与两个恢复 HOME 的隔离根目录。

    Returns:
        None: 两次恢复和相同事件选项后的状态都通过全字段核对。
    """
    with GameClient(running_game.base_url) as source:
        state = start_run(source, "DEFECT", seed="CKPTEVENT1", ascension=0)
        assert state["screen"] == "EVENT"
        checkpoint = capture_strategic_checkpoint(
            source,
            home=running_game.home,
            destination=tmp_path / "checkpoint-event",
        )

    app_path = Path(os.environ.get("STS2_APP_PATH", DEFAULT_APP))
    executable = app_path / "Contents/MacOS/Slay the Spire 2"
    base_port = int(running_game.base_url.rsplit(":", 1)[1]) + 1
    restored_states = []
    suffix_entries = []
    with ExitStack() as stack:
        for index in range(2):
            restored_game = stack.enter_context(
                launch_game(
                    executable,
                    port=base_port + index,
                    home=tmp_path / f"restore-home-{index}",
                    profile=DEFAULT_PROFILE,
                    mode="headless",
                    enable_debug_actions=True,
                    run_save=checkpoint.save_path,
                )
            )
            client = stack.enter_context(GameClient(restored_game.base_url))
            restored = restore_strategic_checkpoint(client, checkpoint)
            restored_states.append(restored)
            option = next(
                item
                for item in restored["event"]["options"]
                if item.get("is_locked") is not True
            )
            after = _execute_stable(
                client,
                restored,
                "choose_event_option",
                option_index=option["index"],
            )
            suffix_entries.append(observe_checkpoint_entry(client, after))

    assert [state["screen"] for state in restored_states] == ["EVENT", "EVENT"]
    assert suffix_entries[0] == suffix_entries[1]


def test_map_checkpoint_replays_battle_and_preserves_negative_control(
    running_game: RunningGame,
    tmp_path: Path,
) -> None:
    """四个 HOME 应分别复现两种地图动作，且兄弟动作允许真实分化。

    Args:
        running_game (RunningGame): 捕获地图 checkpoint 的源游戏实例。
        tmp_path (Path): checkpoint 与四个恢复 HOME 的隔离根目录。

    Returns:
        None: A/A、B/B 各自一致，A/B 因地图坐标和战斗入口自然不同。
    """
    with GameClient(running_game.base_url) as source:
        state = _start_to_map(source)
        assert len(state["map"]["available_nodes"]) >= 2
        checkpoint = capture_strategic_checkpoint(
            source,
            home=running_game.home,
            destination=tmp_path / "checkpoint-map",
        )

    app_path = Path(os.environ.get("STS2_APP_PATH", DEFAULT_APP))
    executable = app_path / "Contents/MacOS/Slay the Spire 2"
    base_port = int(running_game.base_url.rsplit(":", 1)[1]) + 1
    arm_entries = []
    with ExitStack() as stack:
        for worker, option_index in enumerate((0, 0, 1, 1)):
            restored_game = stack.enter_context(
                launch_game(
                    executable,
                    port=base_port + worker,
                    home=tmp_path / f"map-home-{worker}",
                    profile=DEFAULT_PROFILE,
                    mode="headless",
                    enable_debug_actions=True,
                    run_save=checkpoint.save_path,
                )
            )
            game = stack.enter_context(GameClient(restored_game.base_url))
            restored = restore_strategic_checkpoint(game, checkpoint)
            battle = _execute_and_wait_for_screen(
                game,
                restored,
                "choose_map_node",
                target_screen="COMBAT",
                option_index=option_index,
            )
            arm_entries.append(observe_checkpoint_entry(game, battle))

    assert arm_entries[0] == arm_entries[1]
    assert arm_entries[2] == arm_entries[3]
    assert arm_entries[0] != arm_entries[2]
    for entry in arm_entries:
        combat = entry.audit["combat"]
        assert combat["enemies"]
        pile_types = {pile["pile_type"] for pile in entry.audit["players"][0]["piles"]}
        assert {"Hand", "Draw", "Discard", "Exhaust", "Play", "Deck"} <= pile_types
        assert "敌人:" in entry.policy_text


def test_reward_checkpoint_ignores_finished_combat_residue(
    running_game: RunningGame,
    tmp_path: Path,
) -> None:
    """奖励 checkpoint 不应把已经结束且不会存盘的 CombatState 当作环境状态。

    Args:
        running_game (RunningGame): 创建真实战斗奖励的源游戏实例。
        tmp_path (Path): checkpoint 与恢复 HOME 的隔离根目录。

    Returns:
        None: 奖励页捕获与恢复都没有陈旧 combat 审计。
    """
    with GameClient(running_game.base_url) as source:
        start_run(source, "DEFECT", seed="ABCDEF1234", ascension=0)
        state = source.execute_action(
            "run_console_command",
            command="scenariofight MOCK_MONSTER_ENCOUNTER floor=7",
        )["state"]
        assert state["screen"] == "COMBAT"
        state = source.execute_action(
            "run_console_command",
            command="win",
        )["state"]
        assert state["screen"] == "REWARD"
        checkpoint = capture_strategic_checkpoint(
            source,
            home=running_game.home,
            destination=tmp_path / "checkpoint-reward",
        )

    assert checkpoint.entry.audit["combat"] is None
    app_path = Path(os.environ.get("STS2_APP_PATH", DEFAULT_APP))
    with (
        launch_game(
            app_path / "Contents/MacOS/Slay the Spire 2",
            port=int(running_game.base_url.rsplit(":", 1)[1]) + 1,
            home=tmp_path / "reward-home",
            profile=DEFAULT_PROFILE,
            mode="headless",
            enable_debug_actions=True,
            run_save=checkpoint.save_path,
        ) as restored_game,
        GameClient(restored_game.base_url) as game,
    ):
        restored = restore_strategic_checkpoint(game, checkpoint)
        audit = game.checkpoint_audit()

    assert restored["screen"] == "REWARD"
    assert audit["combat"] is None


def test_reward_selection_shop_and_rest_replay_across_two_homes(
    running_game: RunningGame,
    tmp_path: Path,
) -> None:
    """四类战略 checkpoint 应在两个 HOME 中执行相同 suffix 后保持一致。

    Args:
        running_game (RunningGame): 沿真实地图捕获 checkpoint 的源游戏实例。
        tmp_path (Path): checkpoint 和恢复 HOME 的隔离根目录。

    Returns:
        None: REWARD、CARD_SELECTION、SHOP、REST 都通过成对重放。
    """
    with GameClient(running_game.base_url) as source:
        state = _start_to_map(source)
        shop_path = _path_to_node_type(state, "Shop")
        first_coord = shop_path[0]
        first_node = next(
            item
            for item in state["map"]["available_nodes"]
            if (item["row"], item["col"]) == first_coord
        )
        state = _execute_and_wait_for_screen(
            source,
            state,
            "choose_map_node",
            target_screen="COMBAT",
            option_index=first_node["index"],
        )
        state = source.execute_action(
            "run_console_command",
            command="win",
        )["state"]
        assert state["screen"] == "REWARD"

        reward_checkpoint = capture_strategic_checkpoint(
            source,
            home=running_game.home,
            destination=tmp_path / "checkpoint-reward-pair",
        )
        state = restore_strategic_checkpoint(source, reward_checkpoint)
        card_reward = next(
            item for item in state["reward"]["rewards"] if item["reward_type"] == "Card"
        )
        state = source.execute_action(
            "claim_reward",
            expected_state_revision=state["state_revision"],
            option_index=card_reward["index"],
        )["state"]
        assert state["screen"] == "CARD_SELECTION"
        selection_checkpoint = derive_strategic_checkpoint(
            source,
            base=reward_checkpoint,
            destination=tmp_path / "checkpoint-selection",
            resume_actions=(
                HarnessAction(
                    "claim_reward",
                    {"option_index": card_reward["index"]},
                ),
            ),
        )
        state = source.execute_action(
            "choose_reward_card",
            expected_state_revision=state["state_revision"],
            option_index=state["selection"]["cards"][0]["index"],
        )["state"]
        state = _finish_room_to_map(source, state)

        state = _navigate_to_node_type(source, state, "Shop")
        assert state["screen"] == "SHOP"
        shop_checkpoint = capture_strategic_checkpoint(
            source,
            home=running_game.home,
            destination=tmp_path / "checkpoint-shop",
        )
        state = restore_strategic_checkpoint(source, shop_checkpoint)
        state = _finish_room_to_map(source, state)

        state = _navigate_to_node_type(source, state, "RestSite")
        assert state["screen"] == "REST"
        rest_checkpoint = capture_strategic_checkpoint(
            source,
            home=running_game.home,
            destination=tmp_path / "checkpoint-rest",
        )

    app_path = Path(os.environ.get("STS2_APP_PATH", DEFAULT_APP))
    executable = app_path / "Contents/MacOS/Slay the Spire 2"
    base_port = int(running_game.base_url.rsplit(":", 1)[1]) + 1
    cases: tuple[
        tuple[
            str,
            StrategicCheckpoint,
            Callable[[GameClient, dict[str, object]], dict[str, object]],
        ],
        ...,
    ] = (
        ("reward", reward_checkpoint, _claim_gold_suffix),
        ("selection", selection_checkpoint, _choose_first_card_suffix),
        ("shop", shop_checkpoint, _buy_first_shop_card_suffix),
        ("rest", rest_checkpoint, _heal_suffix),
    )
    for label, checkpoint, suffix in cases:
        entries = _restore_pair_and_apply_suffix(
            checkpoint,
            suffix,
            executable=executable,
            profile=DEFAULT_PROFILE,
            base_port=base_port,
            homes_root=tmp_path / f"pair-{label}",
        )
        assert entries[0] == entries[1]


def _start_to_map(game: GameClient) -> dict[str, object]:
    """从固定新局完成涅奥事件并进入第一个真实地图决策。

    Args:
        game (GameClient): 连接源游戏实例的客户端。

    Returns:
        dict[str, object]: 至少有两个可选节点的地图状态。
    """
    state = start_run(game, "DEFECT", seed="ABCDEF1234", ascension=0)
    while state["screen"] != "MAP":
        option = next(
            item
            for item in state["event"]["options"]
            if item.get("is_locked") is not True
        )
        state = game.execute_action(
            "choose_event_option",
            expected_state_revision=state["state_revision"],
            option_index=option["index"],
        )["state"]
    return state


def _finish_room_to_map(
    game: GameClient,
    state: dict[str, object],
) -> dict[str, object]:
    """用真实动作结束当前房间并回到地图。

    Args:
        game (GameClient): 当前源游戏客户端。
        state (dict[str, object]): 当前房间稳定状态。

    Returns:
        dict[str, object]: 下一次地图决策状态。
    """
    current = state
    for _ in range(60):
        screen = current["screen"]
        actions = current.get("available_actions") or []
        if screen == "MAP":
            return current
        if screen == "COMBAT":
            current = game.execute_action(
                "run_console_command",
                command="win",
            )["state"]
            continue
        if screen == "REWARD" and "proceed" in actions:
            current = _execute_stable(game, current, "proceed")
            continue
        if screen == "CARD_SELECTION":
            if "skip_reward_cards" in actions:
                current = _execute_stable(game, current, "skip_reward_cards")
            elif "skip_card_selection" in actions:
                current = _execute_stable(game, current, "skip_card_selection")
            elif "select_deck_card" in actions:
                current = _execute_stable(
                    game,
                    current,
                    "select_deck_card",
                    option_index=current["selection"]["cards"][0]["index"],
                )
            elif "confirm_selection" in actions:
                current = _execute_stable(game, current, "confirm_selection")
            else:
                raise AssertionError(f"无法结束选牌页面: {actions}")
            continue
        if screen == "EVENT":
            options = [
                item
                for item in current["event"]["options"]
                if item.get("is_locked") is not True
            ]
            option = next(
                (item for item in options if item.get("is_proceed") is True),
                options[0],
            )
            current = _execute_stable(
                game,
                current,
                "choose_event_option",
                option_index=option["index"],
            )
            continue
        if screen == "CHEST" and "open_chest" in actions:
            current = _execute_stable(game, current, "open_chest")
            continue
        if screen == "CHEST" and "choose_treasure_relic" in actions:
            current = _execute_stable(
                game,
                current,
                "choose_treasure_relic",
                option_index=current["chest"]["relic_options"][0]["index"],
            )
            continue
        if "proceed" in actions:
            current = _execute_stable(game, current, "proceed")
            continue
        raise AssertionError(f"无法结束房间: {screen} {actions}")
    raise AssertionError("结束房间的动作数超过验收预算")


def _path_to_node_type(
    state: dict[str, object],
    target_type: str,
) -> list[tuple[int, int]]:
    """从当前真实可达节点中寻找第一个指定类型房间。

    Args:
        state (dict[str, object]): 当前完整地图状态。
        target_type (str): ``Shop`` 或 ``RestSite``。

    Returns:
        list[tuple[int, int]]: 从下一节点到目标房间的坐标路径。
    """
    map_state = state["map"]
    nodes = {(item["row"], item["col"]): item for item in map_state["nodes"]}
    queue = deque(
        (
            (item["row"], item["col"]),
            [(item["row"], item["col"])],
        )
        for item in map_state["available_nodes"]
    )
    visited = set()
    while queue:
        coord, path = queue.popleft()
        if coord in visited:
            continue
        visited.add(coord)
        node = nodes[coord]
        if node["node_type"] == target_type:
            return path
        for child in node["children"]:
            child_coord = (child["row"], child["col"])
            queue.append((child_coord, [*path, child_coord]))
    raise AssertionError(f"当前剩余地图没有节点: {target_type}")


def _navigate_to_node_type(
    game: GameClient,
    state: dict[str, object],
    target_type: str,
) -> dict[str, object]:
    """沿真实地图与房间同步器前进到指定节点类型。

    Args:
        game (GameClient): 当前源游戏客户端。
        state (dict[str, object]): 当前地图决策状态。
        target_type (str): 目标地图节点类型。

    Returns:
        dict[str, object]: 目标 SHOP 或 REST 页面。
    """
    current = state
    for coord in _path_to_node_type(current, target_type):
        option = next(
            item
            for item in current["map"]["available_nodes"]
            if (item["row"], item["col"]) == coord
        )
        current = _execute_stable(
            game,
            current,
            "choose_map_node",
            option_index=option["index"],
        )
        if target_type == "Shop" and current["screen"] == "SHOP":
            return current
        if target_type == "RestSite" and current["screen"] == "REST":
            return current
        current = _finish_room_to_map(game, current)
    raise AssertionError(f"没有进入目标页面: {target_type}")


def _restore_pair_and_apply_suffix(
    checkpoint: StrategicCheckpoint,
    suffix: Callable[[GameClient, dict[str, object]], dict[str, object]],
    *,
    executable: Path,
    profile: Path,
    base_port: int,
    homes_root: Path,
) -> tuple[CheckpointEntry, CheckpointEntry]:
    """在两个隔离 HOME 中恢复同一 checkpoint 并执行固定 suffix。

    Args:
        checkpoint (StrategicCheckpoint): 待复制的战略 checkpoint。
        suffix (Callable): 在恢复入口执行的固定动作序列。
        executable (Path): v0.111.0 游戏可执行文件。
        profile (Path): 受控测试 profile。
        base_port (int): 两个实例使用的起始端口。
        homes_root (Path): 两个恢复 HOME 的父目录。

    Returns:
        tuple[CheckpointEntry, CheckpointEntry]: 两次 suffix 后的完整比较入口。
    """
    entries = []
    with ExitStack() as stack:
        for index in range(2):
            restored_game = stack.enter_context(
                launch_game(
                    executable,
                    port=base_port + index,
                    home=homes_root / str(index),
                    profile=profile,
                    mode="headless",
                    enable_debug_actions=True,
                    run_save=checkpoint.save_path,
                )
            )
            game = stack.enter_context(GameClient(restored_game.base_url))
            state = restore_strategic_checkpoint(game, checkpoint)
            state = suffix(game, state)
            entries.append(observe_checkpoint_entry(game, state))
    return entries[0], entries[1]


def _claim_gold_suffix(
    game: GameClient,
    state: dict[str, object],
) -> dict[str, object]:
    """领取当前奖励页的金币。

    Args:
        game (GameClient): 当前恢复实例客户端。
        state (dict[str, object]): REWARD checkpoint 入口。

    Returns:
        dict[str, object]: 领取金币后的奖励页。
    """
    reward = next(
        item for item in state["reward"]["rewards"] if item["reward_type"] == "Gold"
    )
    return _execute_stable(
        game,
        state,
        "claim_reward",
        option_index=reward["index"],
    )


def _choose_first_card_suffix(
    game: GameClient,
    state: dict[str, object],
) -> dict[str, object]:
    """选择卡牌奖励中的第一张牌。

    Args:
        game (GameClient): 当前恢复实例客户端。
        state (dict[str, object]): CARD_SELECTION checkpoint 入口。

    Returns:
        dict[str, object]: 选择后的奖励页。
    """
    return _execute_stable(
        game,
        state,
        "choose_reward_card",
        option_index=state["selection"]["cards"][0]["index"],
    )


def _buy_first_shop_card_suffix(
    game: GameClient,
    state: dict[str, object],
) -> dict[str, object]:
    """打开真实商店库存并购买第一张可负担卡牌。

    Args:
        game (GameClient): 当前恢复实例客户端。
        state (dict[str, object]): SHOP checkpoint 入口。

    Returns:
        dict[str, object]: 购买后的商店状态。
    """
    current = _execute_stable(game, state, "open_shop_inventory")
    card = next(
        item
        for item in current["shop"]["cards"]
        if item["is_stocked"] and item["enough_gold"]
    )
    return _execute_stable(
        game,
        current,
        "buy_card",
        option_index=card["index"],
    )


def _heal_suffix(
    game: GameClient,
    state: dict[str, object],
) -> dict[str, object]:
    """执行真实休息选项并返回随后的稳定页面。

    Args:
        game (GameClient): 当前恢复实例客户端。
        state (dict[str, object]): REST checkpoint 入口。

    Returns:
        dict[str, object]: 休息结算后的稳定状态。
    """
    option = next(
        item for item in state["rest"]["options"] if item["option_id"] == "HEAL"
    )
    return _execute_stable(
        game,
        state,
        "choose_rest_option",
        option_index=option["index"],
    )


def _execute_stable(
    game: GameClient,
    state: dict[str, object],
    action: str,
    **parameters: int,
) -> dict[str, object]:
    """执行动作并提取 Mod 已等待完成的稳定状态。

    Args:
        game (GameClient): 当前游戏客户端。
        state (dict[str, object]): 动作前稳定状态。
        action (str): 待执行动作。
        parameters (int): 动作整数参数。

    Returns:
        dict[str, object]: 动作响应中的状态。
    """
    return game.execute_action(
        action,
        expected_state_revision=state["state_revision"],
        **parameters,
    )["state"]


def _execute_and_wait_for_screen(
    game: GameClient,
    state: dict[str, object],
    action: str,
    *,
    target_screen: str,
    **parameters: int,
) -> dict[str, object]:
    """执行真实动作并等待目标页面开放模型动作窗口。

    Args:
        game (GameClient): 当前恢复实例客户端。
        state (dict[str, object]): 动作前稳定状态。
        action (str): 待执行的 Mod 动作。
        target_screen (str): 期望到达的稳定页面。
        parameters (int): 动作需要的整数参数。

    Raises:
        AssertionError: 三十秒内没有到达目标模型动作窗口。

    Returns:
        dict[str, object]: 目标页面的稳定状态。
    """
    revision = state["state_revision"]
    assert isinstance(revision, int)
    result = game.execute_action(
        action,
        expected_state_revision=revision,
        **parameters,
    )
    current = result["state"]
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        actions = current.get("available_actions") or []
        if current.get("screen") == target_screen and any(
            candidate not in {"save_and_quit"} for candidate in actions
        ):
            return current
        latest_revision = current.get("state_revision")
        assert isinstance(latest_revision, int)
        try:
            current = game.wait_for_state(
                after_revision=latest_revision,
                timeout=min(5.0, deadline - time.monotonic()),
            )
        except TimeoutError:
            current = game.state()
    raise AssertionError(f"未到达目标页面: {target_screen}")
