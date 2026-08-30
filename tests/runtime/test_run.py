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
                _combat_state(state_revision=2)
                if self._map_choices == 1
                else _game_over_state(True, state_revision=6)
            )
            return _completed(state)
        if action == "end_turn":
            return _completed(_reward_state(state_revision=3))
        if action == "claim_reward":
            return {"status": "pending", "stable": False}
        raise AssertionError(f"预期外动作: {action}")

    def wait_for_state(
        self,
        *,
        after_revision: int,
        timeout: float,
    ) -> dict[str, Any]:
        """奖励结算后先接收过渡地图事件，再接收可选节点地图。

        Returns:
            dict[str, Any]: 当前事件次数对应的地图状态。
        """
        assert timeout > 0
        self.state_calls += 1
        if self.state_calls == 1:
            assert after_revision == 3
            return _map_state(
                available_actions=["save_and_quit"],
                state_revision=4,
            )
        assert after_revision == 4
        return _map_state(state_revision=5)


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


class MissedEventGame:
    """模拟 SSE 漏失但只读状态已经推进的游戏。"""

    def __init__(self, latest_state: dict[str, Any]) -> None:
        """保存超时后 `/state` 能取得的最新快照。

        Args:
            latest_state (dict[str, Any]): SSE 未交付但 Mod 已持有的状态。

        Returns:
            None: 此方法只初始化测试计数。
        """
        self._latest_state = latest_state
        self.wait_calls = 0
        self.state_calls = 0

    def state(self) -> dict[str, Any]:
        """返回当前 Mod 的只读最新状态。

        Returns:
            dict[str, Any]: 最新状态副本。
        """
        self.state_calls += 1
        return dict(self._latest_state)

    def wait_for_state(
        self,
        *,
        after_revision: int,
        timeout: float,
    ) -> dict[str, Any]:
        """模拟 SSE 在新 revision 已发生后仍超时。

        Args:
            after_revision (int): Runtime 已处理的状态 revision。
            timeout (float): SSE 等待预算。

        Raises:
            TimeoutError: 固定模拟漏失事件。

        Returns:
            dict[str, Any]: 此方法不会正常返回。
        """
        assert after_revision >= 0
        assert timeout > 0
        self.wait_calls += 1
        raise TimeoutError("missed state event")


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
        return _completed(_game_over_state(True, state_revision=3))

    def wait_for_state(
        self,
        *,
        after_revision: int,
        timeout: float,
    ) -> dict[str, Any]:
        """返回重新开放动作的地图状态。

        Returns:
            dict[str, Any]: 可再次交给战略模型的地图状态。
        """
        assert timeout > 0
        self.state_calls += 1
        return _map_state(state_revision=after_revision + 1)


class TwoShopVisitsGame:
    """模拟关闭第一家商店后离开，并进入第二家商店的游戏。"""

    def __init__(self) -> None:
        """初始化动作记录与两次独立商店访问状态。"""
        self.actions: list[str] = []
        self._visit = 1
        self._is_open = False

    def execute_action(self, action: str, **_parameters: int) -> dict[str, Any]:
        """按库存开关、离店和下一节点顺序推进两次商店访问。

        Args:
            action (str): 模型提交的战略动作。
            _parameters (int): 当前动作的索引参数。

        Raises:
            AssertionError: 动作绕过商店访问边界或顺序不符合测试契约。

        Returns:
            dict[str, Any]: 下一份商店、地图或终局状态。
        """
        self.actions.append(action)
        revision = len(self.actions) + 1
        if action == "open_shop_inventory":
            assert self._is_open is False
            self._is_open = True
            return _completed(_shop_state(is_open=True, state_revision=revision))
        if action == "close_shop_inventory":
            assert self._is_open is True
            self._is_open = False
            return _completed(_shop_state(is_open=False, state_revision=revision))
        if action == "proceed" and self._visit == 1:
            assert self._is_open is False
            return _completed(_map_state(state_revision=revision))
        if action == "choose_map_node":
            assert self._visit == 1
            self._visit = 2
            return _completed(_shop_state(is_open=False, state_revision=revision))
        assert action == "proceed" and self._visit == 2
        assert self._is_open is False
        return _completed(_game_over_state(True, state_revision=revision))


class PurchasingShopGame:
    """模拟连续购买两类商品、主动关店并离开的游戏。"""

    def __init__(self) -> None:
        """初始化动作记录和购买阶段。"""
        self.actions: list[str] = []
        self._stage = 0

    def execute_action(self, action: str, **parameters: int) -> dict[str, Any]:
        """按购买、关闭和离店顺序推进商店状态。

        Args:
            action (str): 模型提交的商店动作。
            parameters (int): 当前动作的 revision 和商品索引。

        Raises:
            AssertionError: 购买被商店访问屏蔽提前阻断或动作顺序错误。

        Returns:
            dict[str, Any]: 下一购买阶段、关闭库存页或终局状态。
        """
        assert parameters.pop("expected_state_revision") == len(self.actions) + 1
        self.actions.append(action)
        revision = len(self.actions) + 1
        if self._stage == 0:
            assert action == "buy_card"
            assert parameters == {"option_index": 0}
            self._stage = 1
            return _completed(_purchasable_shop_state(stage=1, revision=revision))
        if self._stage == 1:
            assert action == "buy_relic"
            assert parameters == {"option_index": 0}
            self._stage = 2
            return _completed(_purchasable_shop_state(stage=2, revision=revision))
        if self._stage == 2:
            assert action == "close_shop_inventory"
            assert parameters == {}
            self._stage = 3
            return _completed(_shop_state(is_open=False, state_revision=revision))
        assert action == "proceed"
        assert parameters == {}
        return _completed(_game_over_state(True, state_revision=revision))


class RewardSkipGame:
    """模拟跳过卡牌奖励后原奖励入口仍留在总页的游戏。"""

    def __init__(self) -> None:
        """初始化动作记录和卡牌、金币并存的奖励页。"""
        self.actions: list[tuple[str, dict[str, int]]] = []

    def execute_action(self, action: str, **parameters: int) -> dict[str, Any]:
        """按卡牌奖励跳过、领取金币和离场顺序推进状态。

        Args:
            action (str): 模型提交的奖励动作。
            parameters (int): 奖励索引参数。

        Raises:
            AssertionError: 模型重新打开已跳过的卡牌奖励或动作顺序错误。

        Returns:
            dict[str, Any]: 下一份奖励选择、奖励总页或终局状态。
        """
        expected_revision = parameters.pop("expected_state_revision")
        assert expected_revision == len(self.actions) + 1
        self.actions.append((action, parameters))
        if len(self.actions) == 1:
            assert (action, parameters) == ("claim_reward", {"option_index": 1})
            return _completed(_reward_card_selection_state(state_revision=2))
        if len(self.actions) == 2:
            assert (action, parameters) == ("skip_reward_cards", {})
            return _completed(_reward_with_card_and_gold(state_revision=3))
        if len(self.actions) == 3:
            assert (action, parameters) == ("claim_reward", {"option_index": 0})
            return _completed(_reward_with_card_only(state_revision=4))
        if len(self.actions) == 4:
            assert (action, parameters) == ("proceed", {})
            return _completed(_map_state(state_revision=5))
        if len(self.actions) == 5:
            assert (action, parameters) == (
                "choose_map_node",
                {"option_index": 0},
            )
            return _completed(_reward_with_card_only(state_revision=6))
        if len(self.actions) == 6:
            assert (action, parameters) == ("claim_reward", {"option_index": 0})
            return _completed(_reward_card_selection_state(state_revision=7))
        assert len(self.actions) == 7
        assert (action, parameters) == (
            "choose_reward_card",
            {"option_index": 0},
        )
        return _completed(_game_over_state(True, state_revision=8))


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
    assert result.final_state == _game_over_state(True, state_revision=6)
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
            state_timeout=0,
        ).run(state)

    assert provider.requests == []


def test_run_runner_recovers_newer_routable_state_after_missed_event() -> None:
    """SSE 超时后应只读补取已经推进的可路由状态。

    Raises:
        AssertionError: 更高 revision 的终局仍被误报为等待超时。

    Returns:
        None: 此测试复现宝箱自动领取落在订阅窗口中的竞态。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    initial = _map_state(
        available_actions=["save_and_quit"],
        state_revision=10,
    )
    game = MissedEventGame(_game_over_state(True, state_revision=11))
    provider = WholeRunProvider([])

    result = runtime.RunRunner(game, provider, state_timeout=1).run(initial)

    assert result.outcome is runtime.RunOutcome.VICTORY
    assert result.decisions == ()
    assert game.wait_calls == 1
    assert game.state_calls == 1
    assert provider.requests == []


def test_run_runner_rejects_fallback_without_newer_revision() -> None:
    """SSE 超时后的同 revision 快照不能被当成状态推进。

    Raises:
        AssertionError: 未推进状态绕过原有超时错误。

    Returns:
        None: 此测试固定补取逻辑的严格 revision 边界。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    initial = _map_state(
        available_actions=["save_and_quit"],
        state_revision=10,
    )
    game = MissedEventGame(initial)
    provider = WholeRunProvider([])

    with pytest.raises(runtime.RunError, match="等待下一可处理状态超时"):
        runtime.RunRunner(game, provider, state_timeout=1).run(initial)

    assert game.wait_calls == 1
    assert game.state_calls == 1
    assert provider.requests == []


def test_run_runner_blocks_shop_reopen_until_next_shop_visit() -> None:
    """库存关闭后隐藏重开动作，离开商店后下一次访问重新开放。

    Raises:
        AssertionError: 同一次访问仍可重开，或下一家商店没有恢复动作。

    Returns:
        None: 此测试验证商店访问级硬屏蔽和离店重置。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    game = TwoShopVisitsGame()
    provider = WholeRunProvider(
        [
            "ACTION: open_shop_inventory",
            "ACTION: close_shop_inventory",
            "ACTION: open_shop_inventory",
            "ACTION: proceed",
            "ACTION: choose_map_node 0",
            "ACTION: open_shop_inventory",
            "ACTION: close_shop_inventory",
            "ACTION: open_shop_inventory",
            "ACTION: proceed",
        ]
    )

    result = runtime.RunRunner(game, provider).run(_shop_state())

    assert result.outcome is runtime.RunOutcome.VICTORY
    assert game.actions == [
        "open_shop_inventory",
        "close_shop_inventory",
        "proceed",
        "choose_map_node",
        "open_shop_inventory",
        "close_shop_inventory",
        "proceed",
    ]
    first_retry = provider.requests[3][-1].content
    assert "动作不在当前可用动作中: open_shop_inventory" in first_retry
    second_visit = provider.requests[5][-1].content
    assert "- open_shop_inventory" in second_visit
    second_retry = provider.requests[8][-1].content
    assert "动作不在当前可用动作中: open_shop_inventory" in second_retry


def test_run_runner_keeps_shop_purchases_available_until_inventory_is_closed() -> None:
    """连续购买不会触发访问屏蔽，只有主动关闭库存后才禁止重开。

    Raises:
        AssertionError: 一次购买后无法继续浏览，或关闭前错误隐藏购买动作。

    Returns:
        None: 此测试覆盖商店硬屏蔽不替模型结束购买流程。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    game = PurchasingShopGame()
    provider = WholeRunProvider(
        [
            "ACTION: buy_card 0",
            "ACTION: buy_relic 0",
            "ACTION: close_shop_inventory",
            "ACTION: open_shop_inventory",
            "ACTION: proceed",
        ]
    )

    result = runtime.RunRunner(game, provider).run(_purchasable_shop_state(stage=0))

    assert result.outcome is runtime.RunOutcome.VICTORY
    assert game.actions == [
        "buy_card",
        "buy_relic",
        "close_shop_inventory",
        "proceed",
    ]
    after_first_purchase = provider.requests[1][-1].content
    assert "buy_relic" in after_first_purchase
    retry_after_close = provider.requests[4][-1].content
    assert "动作不在当前可用动作中: open_shop_inventory" in retry_after_close


def test_run_runner_hides_skipped_card_reward_until_leaving_reward_screen() -> None:
    """跳过卡牌后只屏蔽该入口，并保留其他奖励和真实离场动作。

    Raises:
        AssertionError: 已跳过的卡牌入口再次暴露，或金币和离场动作被误删。

    Returns:
        None: 此测试验证奖励页局部遮蔽而非自动代打。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    game = RewardSkipGame()
    provider = WholeRunProvider(
        [
            "ACTION: claim_reward 1",
            "ACTION: skip_reward_cards",
            "ACTION: claim_reward 1",
            "ACTION: claim_reward 0",
            "ACTION: proceed",
            "ACTION: choose_map_node 0",
            "ACTION: claim_reward 0",
            "ACTION: choose_reward_card 0",
        ]
    )

    result = runtime.RunRunner(game, provider).run(_reward_with_card_and_gold())

    assert result.outcome is runtime.RunOutcome.VICTORY
    assert game.actions == [
        ("claim_reward", {"option_index": 1}),
        ("skip_reward_cards", {}),
        ("claim_reward", {"option_index": 0}),
        ("proceed", {}),
        ("choose_map_node", {"option_index": 0}),
        ("claim_reward", {"option_index": 0}),
        ("choose_reward_card", {"option_index": 0}),
    ]
    after_skip = provider.requests[2][-1].content
    assert "13 金币" in after_skip
    assert "将一张牌添加到你的牌组" not in after_skip
    retry_after_skip = provider.requests[3][-1].content
    assert "option_index 1 不在当前奖励选项" in retry_after_skip
    card_only = provider.requests[4][-1].content
    assert "claim_reward" not in card_only
    assert "proceed" in card_only
    next_reward = provider.requests[6][-1].content
    assert "将一张牌添加到你的牌组" in next_reward
    assert "claim_reward" in next_reward


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
    state_revision: int = 1,
) -> dict[str, Any]:
    """构造一个最小地图状态。

    Args:
        available_actions (Sequence[str]): 当前地图开放的动作。
        state_revision (int): 当前地图状态的 revision。

    Returns:
        dict[str, Any]: 与战略 Harness 兼容的地图状态。
    """
    return {
        "state_revision": state_revision,
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


def _combat_state(*, state_revision: int = 1) -> dict[str, Any]:
    """构造一个可以立即结束回合的最小战斗状态。

    Returns:
        dict[str, Any]: 与战斗 Harness 兼容的战斗状态。
    """
    return {
        "state_revision": state_revision,
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


def _reward_state(*, state_revision: int = 1) -> dict[str, Any]:
    """构造一个可以领取金币的战后奖励状态。

    Returns:
        dict[str, Any]: 与战略 Harness 兼容的奖励状态。
    """
    return {
        "state_revision": state_revision,
        "screen": "REWARD",
        "in_combat": False,
        "available_actions": ["claim_reward"],
        "run": _run_state(),
        "reward": {"rewards": [{"index": 0, "name": "金币", "claimable": True}]},
    }


def _reward_with_card_and_gold(*, state_revision: int = 1) -> dict[str, Any]:
    """构造金币和卡牌入口并存的真实战后奖励页。

    Args:
        state_revision (int): 当前奖励状态版本。

    Returns:
        dict[str, Any]: 可领取金币、可打开卡牌且可离场的奖励状态。
    """
    return {
        "state_revision": state_revision,
        "screen": "REWARD",
        "in_combat": False,
        "available_actions": ["claim_reward", "proceed"],
        "run": _run_state(),
        "reward": {
            "can_proceed": True,
            "rewards": [
                {
                    "index": 0,
                    "reward_type": "Gold",
                    "name": "13 金币",
                    "claimable": True,
                },
                {
                    "index": 1,
                    "reward_type": "Card",
                    "name": "将一张牌添加到你的牌组。",
                    "claimable": True,
                },
            ],
        },
    }


def _reward_with_card_only(*, state_revision: int = 1) -> dict[str, Any]:
    """构造只剩已跳过卡牌入口的奖励总页。

    Args:
        state_revision (int): 当前奖励状态版本。

    Returns:
        dict[str, Any]: 卡牌入口仍由游戏保留但允许离场的奖励状态。
    """
    state = _reward_with_card_and_gold(state_revision=state_revision)
    state["reward"]["rewards"] = [state["reward"]["rewards"][1]]
    state["reward"]["rewards"][0]["index"] = 0
    return state


def _reward_card_selection_state(*, state_revision: int = 1) -> dict[str, Any]:
    """构造允许选择或跳过卡牌奖励的候选页。

    Args:
        state_revision (int): 当前选牌状态版本。

    Returns:
        dict[str, Any]: 一张卡牌候选和跳过动作组成的战略状态。
    """
    return {
        "state_revision": state_revision,
        "screen": "CARD_SELECTION",
        "in_combat": False,
        "available_actions": ["choose_reward_card", "skip_reward_cards"],
        "run": _run_state(),
        "selection": {
            "cards": [
                {
                    "index": 0,
                    "card_id": "COOLHEADED",
                    "name": "冷静头脑",
                    "card_type": "Skill",
                    "energy_cost": 1,
                    "resolved_rules_text": "生成1个冰霜充能球。抽1张牌。",
                }
            ]
        },
    }


def _shop_state(
    *,
    is_open: bool = False,
    gold: int = 0,
    state_revision: int = 1,
) -> dict[str, Any]:
    """构造库存固定且没有任何可购买项目的商店状态。

    Args:
        is_open (bool): 当前是否打开商店库存。
        gold (int): 用于区分测试观测的当前金币数。
        state_revision (int): 当前商店状态的 revision。

    Returns:
        dict[str, Any]: 与战略 Harness 兼容的商店状态。
    """
    run = _run_state()
    run["gold"] = gold
    return {
        "state_revision": state_revision,
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


def _purchasable_shop_state(
    *,
    stage: int,
    revision: int = 1,
) -> dict[str, Any]:
    """构造依次可购买卡牌、遗物并最终关闭的库存状态。

    Args:
        stage (int): 已完成的购买数量，支持 ``0``、``1`` 或 ``2``。
        revision (int): 当前商店状态版本。

    Raises:
        AssertionError: 测试传入未定义的购买阶段。

    Returns:
        dict[str, Any]: 仍处于同一次 SHOP 访问的打开库存状态。
    """
    assert stage in {0, 1, 2}
    actions = ["close_shop_inventory"]
    if stage == 0:
        actions.insert(0, "buy_card")
    elif stage == 1:
        actions.insert(0, "buy_relic")
    return {
        "state_revision": revision,
        "screen": "SHOP",
        "in_combat": False,
        "available_actions": actions,
        "run": {**_run_state(), "gold": 200},
        "shop": {
            "is_open": True,
            "cards": [
                {
                    "index": 0,
                    "name": "眼部攻击",
                    "price": 45,
                    "is_stocked": stage == 0,
                    "enough_gold": stage == 0,
                }
            ],
            "relics": [
                {
                    "index": 0,
                    "name": "草莓",
                    "price": 75,
                    "is_stocked": stage <= 1,
                    "enough_gold": stage <= 1,
                }
            ],
            "potions": [],
        },
    }


def _game_over_state(
    victory: bool,
    *,
    state_revision: int = 1,
) -> dict[str, Any]:
    """构造胜利或死亡的游戏结束状态。

    Args:
        victory (bool): 是否为通关胜利。
        state_revision (int): 当前终局状态的 revision。

    Returns:
        dict[str, Any]: 与 Mod 原始字段一致的终局状态。
    """
    return {
        "state_revision": state_revision,
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
