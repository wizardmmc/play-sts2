"""验证模型在战略与战斗之间切换并完成整局。"""

import importlib
from collections.abc import Sequence
from typing import Any

import httpx
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


class ConflictingStrategicGame:
    """模拟战略动作窗口短暂关闭后重新开放的游戏。"""

    def __init__(self) -> None:
        """初始化动作和状态读取计数。

        Returns:
            None: 此方法只初始化测试游戏。
        """
        self.action_calls = 0
        self.state_calls = 0

    def execute_action(self, action: str, **_parameters: int) -> dict[str, Any]:
        """首次拒绝地图动作，第二次返回胜利终局。

        Args:
            action (str): Runtime 提交的地图动作。
            _parameters (int): 模型提交的节点索引。

        Raises:
            httpx.HTTPStatusError: 首次动作落在暂不可用窗口时抛出。

        Returns:
            dict[str, Any]: 第二次动作完成后的胜利终局。
        """
        assert action == "choose_map_node"
        self.action_calls += 1
        if self.action_calls == 1:
            raise _action_unavailable("choose_map_node", "MAP")
        return _completed(_game_over_state(True))

    def state(self) -> dict[str, Any]:
        """返回重新开放动作的地图状态。

        Returns:
            dict[str, Any]: 可再次交给战略模型的地图状态。
        """
        self.state_calls += 1
        return _map_state()


class ShopLoopGame:
    """模拟金币耗尽后反复开关同一商店库存的游戏。"""

    def __init__(self, *, state_changes: bool = False) -> None:
        """初始化关闭的库存和空动作记录。

        Args:
            state_changes (bool): 是否让每次动作后的观测资源发生变化。

        Returns:
            None: 此方法只初始化商店循环测试替身。
        """
        self.actions: list[str] = []
        self._is_open = False
        self._state_changes = state_changes

    def execute_action(self, action: str, **_parameters: int) -> dict[str, Any]:
        """切换库存开关，或在模型主动离开时返回终局。

        Args:
            action (str): 模型提交的商店动作。
            _parameters (int): 商店开关与离开动作不使用的参数。

        Raises:
            AssertionError: 模型动作与当前库存状态不一致。

        Returns:
            dict[str, Any]: 下一份商店状态或胜利终局。
        """
        self.actions.append(action)
        if action == "open_shop_inventory":
            assert self._is_open is False
            self._is_open = True
            return _completed(self._shop_state())
        if action == "close_shop_inventory":
            assert self._is_open is True
            self._is_open = False
            return _completed(self._shop_state())
        if action == "proceed":
            assert self._is_open is False
            return _completed(_game_over_state(True))
        raise AssertionError(f"预期外动作: {action}")

    def state(self) -> dict[str, Any]:
        """返回当前库存开关对应的商店状态。

        Returns:
            dict[str, Any]: 当前商店状态。
        """
        return self._shop_state()

    def _shop_state(self) -> dict[str, Any]:
        """构造当前库存状态和可见动作。

        Returns:
            dict[str, Any]: 金币不足的商店观测。
        """
        gold = len(self.actions) if self._state_changes else 0
        return _shop_state(is_open=self._is_open, gold=gold)


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


def test_run_runner_retries_temporary_strategic_action_conflict() -> None:
    """战略动作窗口短暂关闭时刷新状态并重新决策。

    Raises:
        AssertionError: 合法动作的短暂 409 终止整局或被记录成成功决策。

    Returns:
        None: 此测试只验证战略层的输入窗口竞争。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    game = ConflictingStrategicGame()
    provider = WholeRunProvider(
        ["ACTION: choose_map_node 0", "ACTION: choose_map_node 0"]
    )

    result = runtime.RunRunner(
        game,
        provider,
        poll_interval=0,
        state_timeout=1,
    ).run(_map_state())

    assert result.outcome is runtime.RunOutcome.VICTORY
    assert len(result.decisions) == 1
    assert game.action_calls == 2
    assert game.state_calls == 1
    assert len(provider.requests) == 2


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


def test_run_runner_warns_once_then_accepts_model_shop_exit() -> None:
    """商店开关循环触发一次提示后仍由模型主动离开。

    Raises:
        AssertionError: Harness 自动代打、没有提示循环或阻止模型纠偏。

    Returns:
        None: 此测试验证不代打的商店循环纠偏路径。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    game = ShopLoopGame()
    provider = WholeRunProvider(
        [
            "ACTION: open_shop_inventory",
            "ACTION: close_shop_inventory",
            "ACTION: open_shop_inventory",
            "ACTION: close_shop_inventory",
            "ACTION: open_shop_inventory",
            "ACTION: close_shop_inventory",
            "ACTION: proceed",
        ]
    )

    result = runtime.RunRunner(game, provider).run(_shop_state())

    assert result.outcome is runtime.RunOutcome.VICTORY
    assert game.actions == [
        "open_shop_inventory",
        "close_shop_inventory",
        "open_shop_inventory",
        "close_shop_inventory",
        "open_shop_inventory",
        "close_shop_inventory",
        "proceed",
    ]
    assert "立刻输出 `ACTION: proceed`" in provider.requests[0][-1].content
    corrective_message = provider.requests[6][-1].content
    assert "动作循环" in corrective_message
    assert "商店库存不会" in corrective_message
    assert "harness 不会替你操作" in corrective_message


def test_run_runner_stops_repeated_shop_loop_without_harness_action() -> None:
    """模型忽略一次循环提示后明确停止本局且不替它离开。

    Raises:
        AssertionError: 循环耗尽全局步数、Harness 自动离开或错误继续运行。

    Returns:
        None: 此测试验证循环被归类为模型失败。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    game = ShopLoopGame()
    provider = WholeRunProvider(
        ["ACTION: open_shop_inventory", "ACTION: close_shop_inventory"] * 6
    )

    with pytest.raises(runtime.RunError, match="纠偏提示后仍重复战略动作循环"):
        runtime.RunRunner(game, provider).run(_shop_state())

    assert game.actions == ["open_shop_inventory", "close_shop_inventory"] * 6
    assert len(provider.requests) == 12


def test_run_runner_does_not_flag_actions_when_observation_progresses() -> None:
    """动作名称重复但模型观测持续变化时不误判为循环。

    Raises:
        AssertionError: Harness 只按动作名检测而忽略真实状态进展。

    Returns:
        None: 此测试验证循环检测绑定模型实际观测。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    game = ShopLoopGame(state_changes=True)
    provider = WholeRunProvider(
        ["ACTION: open_shop_inventory", "ACTION: close_shop_inventory"] * 3
        + ["ACTION: proceed"]
    )

    result = runtime.RunRunner(game, provider).run(_shop_state())

    assert result.outcome is runtime.RunOutcome.VICTORY
    assert all("动作循环" not in request[-1].content for request in provider.requests)


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


def _shop_state(*, is_open: bool = False, gold: int = 0) -> dict[str, Any]:
    """构造库存固定且没有任何可购买项目的商店状态。

    Args:
        is_open (bool): 当前是否打开商店库存。
        gold (int): 用于区分测试观测的当前金币数。

    Returns:
        dict[str, Any]: 与战略 Harness 兼容的商店状态。
    """
    run = _run_state()
    run["gold"] = gold
    return {
        "screen": "SHOP",
        "in_combat": False,
        "available_actions": (
            ["close_shop_inventory"] if is_open else ["open_shop_inventory", "proceed"]
        ),
        "run": run,
        "shop": {
            "is_open": is_open,
            "cards": [
                {
                    "index": 0,
                    "name": "眼部攻击",
                    "price": 45,
                    "is_stocked": True,
                    "enough_gold": False,
                }
            ],
            "relics": [],
            "potions": [],
            "card_removal": {
                "price": 75,
                "available": True,
                "used": False,
                "enough_gold": False,
            },
        },
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


def _action_unavailable(action: str, screen: str) -> httpx.HTTPStatusError:
    """构造与真实 Mod 输入窗口竞争一致的 409 异常。

    Args:
        action (str): 被暂时拒绝的动作名称。
        screen (str): 动作所属的当前页面。

    Returns:
        httpx.HTTPStatusError: 包含真实错误外壳的 HTTP 异常。
    """
    request = httpx.Request("POST", "http://127.0.0.1:8080/action")
    response = httpx.Response(
        409,
        request=request,
        json={
            "ok": False,
            "error": {
                "code": "invalid_action",
                "message": "Action is not available in the current state.",
                "details": {"action": action, "screen": screen},
                "retryable": False,
            },
        },
    )
    return httpx.HTTPStatusError(
        "409 Conflict",
        request=request,
        response=response,
    )
