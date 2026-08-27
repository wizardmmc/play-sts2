"""验证模型在战略与战斗之间切换并完成整局。"""

import importlib
from collections.abc import Sequence
from typing import Any

import pytest

from play_sts2.inference import ChatMessage, ModelReply


class WholeRunProvider:
    """按整局决策顺序返回动作并记录上下文。"""

    def __init__(self, replies: Sequence[str]) -> None:
        """保存预设动作和空请求记录。

        Args:
            replies (Sequence[str]): 战略与战斗共用的有序模型回复。

        Returns:
            None: 此方法只初始化测试替身。
        """
        self._replies = iter(replies)
        self.requests: list[tuple[ChatMessage, ...]] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> ModelReply:
        """保存模型输入并返回下一条动作。

        Args:
            messages (Sequence[ChatMessage]): 当前决策的完整消息。
            max_tokens (int): 本次回复的最大 token 数。
            temperature (float): 本次回复的采样温度。

        Returns:
            ModelReply: 下一条预设回复。
        """
        self.requests.append(tuple(messages))
        return ModelReply(next(self._replies))


class WholeRunGame:
    """模拟地图、战斗、奖励、过渡地图和胜利终局。"""

    def __init__(self) -> None:
        """初始化动作、轮询状态与地图选择计数。

        Returns:
            None: 此方法只初始化测试游戏。
        """
        self.actions: list[tuple[str, dict[str, int]]] = []
        self.state_calls = 0
        self._map_choices = 0

    def execute_action(self, action: str, **parameters: int) -> dict[str, Any]:
        """按照整局脚本推进到下一个真实形态状态。

        Args:
            action (str): Runtime 提交的动作名称。
            parameters (int): Runtime 提交的动作索引参数。

        Raises:
            AssertionError: Runtime 执行了脚本之外的动作。

        Returns:
            dict[str, Any]: 当前动作的 Mod 结果。
        """
        self.actions.append((action, parameters))
        if action == "choose_map_node":
            self._map_choices += 1
            state = (
                _combat_state() if self._map_choices == 1 else _game_over_state(True)
            )
            return _completed(state)
        if action == "end_turn":
            return _completed(_reward_state())
        if action == "claim_reward":
            return {"status": "pending", "stable": False}
        raise AssertionError(f"预期外动作: {action}")

    def state(self) -> dict[str, Any]:
        """奖励结算后先返回过渡地图帧，再返回可选节点地图。

        Returns:
            dict[str, Any]: 当前轮询次数对应的地图状态。
        """
        self.state_calls += 1
        if self.state_calls == 1:
            return _map_state(available_actions=["save_and_quit"])
        return _map_state()


class StaticGame:
    """始终返回同一个状态，用于验证运行器边界。"""

    def __init__(self, state: dict[str, Any]) -> None:
        """保存固定状态。

        Args:
            state (dict[str, Any]): 每次读取都返回的游戏状态。

        Returns:
            None: 此方法只初始化测试游戏。
        """
        self._state = state

    def state(self) -> dict[str, Any]:
        """返回固定状态的副本。

        Returns:
            dict[str, Any]: 固定游戏状态。
        """
        return dict(self._state)

    def execute_action(self, action: str, **_parameters: int) -> dict[str, Any]:
        """拒绝边界测试中不应出现的游戏动作。

        Args:
            action (str): 意外提交的动作名称。
            _parameters (int): 意外提交的动作参数。

        Raises:
            AssertionError: 此方法被调用时始终抛出。

        Returns:
            dict[str, Any]: 此方法不会正常返回。
        """
        raise AssertionError(f"不应执行动作: {action}")


def test_run_runner_completes_strategy_battle_and_transient_loop() -> None:
    """整局 Runner 按顺序处理战略、战斗、过渡状态和胜利终局。

    Raises:
        AssertionError: 任一路由、动作顺序、轮询或结果记录不正确。

    Returns:
        None: 此测试验证最小完整整局闭环。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    game = WholeRunGame()
    provider = WholeRunProvider(
        [
            "ACTION: choose_map_node 0",
            "ACTION: end_turn",
            "ACTION: claim_reward 0",
            "ACTION: choose_map_node 0",
        ]
    )

    result = runtime.RunRunner(
        game,
        provider,
        poll_interval=0,
        state_timeout=1,
    ).run(_map_state())

    assert result.outcome is runtime.RunOutcome.VICTORY
    assert result.battle_count == 1
    assert [decision.layer.value for decision in result.decisions] == [
        "strategic",
        "battle",
        "strategic",
        "strategic",
    ]
    assert [decision.step.action.name for decision in result.decisions] == [
        "choose_map_node",
        "end_turn",
        "claim_reward",
        "choose_map_node",
    ]
    assert result.final_state == _game_over_state(True)
    assert game.state_calls == 2
    assert all(
        tuple(message.role for message in provider.requests[index])
        == ("system", "user")
        for index in (0, 2, 3)
    )


def test_run_runner_reports_terminal_loss_without_model_call() -> None:
    """已经失败的终局直接返回死亡，不再请求模型动作。

    Raises:
        AssertionError: 失败终局被误判，或 Runtime 继续调用模型。

    Returns:
        None: 此测试只验证终局收口。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    state = _game_over_state(False)
    provider = WholeRunProvider([])

    result = runtime.RunRunner(StaticGame(state), provider).run(state)

    assert result.outcome is runtime.RunOutcome.DIED
    assert result.decisions == ()
    assert result.battle_count == 0
    assert provider.requests == []


def test_run_runner_fails_fast_on_unknown_screen() -> None:
    """局中出现未枚举页面时明确失败而不猜动作。

    Raises:
        AssertionError: 未知页面触发了模型请求或没有明确报错。

    Returns:
        None: 此测试只验证未知协议边界。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    state = {"screen": "MAIN_MENU", "available_actions": ["open_character_select"]}
    provider = WholeRunProvider([])

    with pytest.raises(runtime.RunError, match="未知游戏屏幕: MAIN_MENU"):
        runtime.RunRunner(StaticGame(state), provider).run(state)

    assert provider.requests == []


def test_run_runner_times_out_while_state_stays_transient() -> None:
    """持续没有决策动作的已知页面在限定时间后停止。

    Raises:
        AssertionError: Runtime 把过渡帧交给模型或无限等待。

    Returns:
        None: 此测试只验证过渡状态的超时边界。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    state = _map_state(available_actions=["save_and_quit"])
    provider = WholeRunProvider([])

    with pytest.raises(runtime.RunError, match="等待下一可处理状态超时"):
        runtime.RunRunner(
            StaticGame(state),
            provider,
            poll_interval=0,
            state_timeout=0,
        ).run(state)

    assert provider.requests == []


def _completed(state: dict[str, Any]) -> dict[str, Any]:
    """把状态包装成 Mod 的稳定动作结果。

    Args:
        state (dict[str, Any]): 动作完成后的游戏状态。

    Returns:
        dict[str, Any]: 与 `/action` 一致的稳定结果。
    """
    return {"status": "completed", "stable": True, "state": state}


def _run_state() -> dict[str, Any]:
    """构造所有页面共用的最小整局资源。

    Returns:
        dict[str, Any]: 可供战略和战斗观测使用的运行状态。
    """
    return {
        "character_name": "故障机器人",
        "current_hp": 70,
        "max_hp": 75,
        "gold": 99,
        "deck": [],
        "relics": [],
        "potions": [],
    }


def _map_state(
    *,
    available_actions: Sequence[str] = ("choose_map_node",),
) -> dict[str, Any]:
    """构造一个最小地图状态。

    Args:
        available_actions (Sequence[str]): 当前地图开放的动作。

    Returns:
        dict[str, Any]: 与战略 Harness 兼容的地图状态。
    """
    return {
        "screen": "MAP",
        "in_combat": False,
        "available_actions": list(available_actions),
        "run": _run_state(),
        "map": {
            "available_nodes": [
                {"index": 0, "row": 1, "col": 2, "node_type": "Monster"}
            ]
        },
    }


def _combat_state() -> dict[str, Any]:
    """构造一个可以立即结束回合的最小战斗状态。

    Returns:
        dict[str, Any]: 与战斗 Harness 兼容的战斗状态。
    """
    return {
        "screen": "COMBAT",
        "in_combat": True,
        "turn": 1,
        "available_actions": ["end_turn"],
        "run": _run_state(),
        "combat": {
            "player": {
                "current_hp": 70,
                "max_hp": 75,
                "block": 0,
                "energy": 0,
                "stars": 0,
            },
            "enemies": [],
            "hand": [],
            "draw_count": 0,
            "discard_count": 0,
        },
    }


def _reward_state() -> dict[str, Any]:
    """构造一个可以领取金币的战后奖励状态。

    Returns:
        dict[str, Any]: 与战略 Harness 兼容的奖励状态。
    """
    return {
        "screen": "REWARD",
        "in_combat": False,
        "available_actions": ["claim_reward"],
        "run": _run_state(),
        "reward": {"rewards": [{"index": 0, "name": "金币", "claimable": True}]},
    }


def _game_over_state(victory: bool) -> dict[str, Any]:
    """构造胜利或死亡的游戏结束状态。

    Args:
        victory (bool): 是否为通关胜利。

    Returns:
        dict[str, Any]: 与 Mod 原始字段一致的终局状态。
    """
    return {
        "screen": "GAME_OVER",
        "available_actions": ["return_to_main_menu"],
        "run": _run_state(),
        "game_over": {"is_victory": victory},
    }
