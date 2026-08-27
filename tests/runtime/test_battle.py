"""验证模型在一场战斗内持续观察、决策并执行动作。"""

import importlib
from collections.abc import Sequence
from typing import Any

import pytest

from play_sts2.inference import ChatMessage, ModelReply


class BattleProvider:
    """按顺序生成动作并保存每次收到的完整战斗对话。

    Args:
        replies (Sequence[str]): 每个战斗步骤依次使用的动作回复。
    """

    def __init__(self, replies: Sequence[str]) -> None:
        """初始化回复队列与消息记录。

        Args:
            replies (Sequence[str]): 每个战斗步骤依次使用的动作回复。

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
        """记录消息并返回下一条动作。

        Args:
            messages (Sequence[ChatMessage]): 当前战斗的完整对话。
            max_tokens (int): 本次生成允许使用的最大输出 token 数。
            temperature (float): 本次生成使用的采样温度。

        Returns:
            ModelReply: 下一条预设动作回复。
        """
        self.requests.append(tuple(messages))
        return ModelReply(next(self._replies))


class PendingBattleGame:
    """模拟首个动作异步结算、第二个动作结束战斗的游戏。"""

    def __init__(self) -> None:
        """初始化动作与轮询计数。

        Returns:
            None: 此方法只初始化测试替身。
        """
        self.actions: list[str] = []
        self.state_calls = 0

    def execute_action(self, action: str, **_parameters: int) -> dict[str, Any]:
        """首次返回 pending，第二次返回稳定奖励屏。

        Args:
            action (str): Runtime 提交的动作名称。
            _parameters (int): 当前测试不会使用的动作参数。

        Returns:
            dict[str, Any]: 当前动作的 Mod 结果。
        """
        self.actions.append(action)
        if len(self.actions) == 1:
            return {"status": "pending", "stable": False}
        return {
            "status": "completed",
            "stable": True,
            "state": _reward_state(),
        }

    def state(self) -> dict[str, Any]:
        """先返回不可决策过渡帧，再返回下一回合稳定状态。

        Returns:
            dict[str, Any]: 当前轮询位置对应的游戏状态。
        """
        self.state_calls += 1
        if self.state_calls == 1:
            return _combat_state(turn=2, available_actions=[])
        return _combat_state(turn=2)


class FinishingBattleGame:
    """模拟一次动作后进入指定离场状态的游戏。

    Args:
        final_state (Mapping[str, Any]): 动作完成后返回的离场状态。
    """

    def __init__(self, final_state: dict[str, Any]) -> None:
        """保存动作完成后的稳定状态。

        Args:
            final_state (dict[str, Any]): 动作完成后返回的离场状态。

        Returns:
            None: 此方法只初始化测试游戏。
        """
        self._final_state = final_state

    def execute_action(self, action: str, **_parameters: int) -> dict[str, Any]:
        """返回配置好的稳定离场状态。

        Args:
            action (str): Runtime 提交的动作名称。
            _parameters (int): 当前测试不会使用的动作参数。

        Returns:
            dict[str, Any]: 含配置离场状态的动作结果。
        """
        assert action == "end_turn"
        return {
            "status": "completed",
            "stable": True,
            "state": self._final_state,
        }

    def state(self) -> dict[str, Any]:
        """返回初始战斗状态。

        Returns:
            dict[str, Any]: 第一回合的稳定战斗状态。
        """
        return _combat_state()


class SelectionBattleGame:
    """模拟战斗内选牌先过渡、后稳定并最终结束战斗的游戏。"""

    def __init__(self) -> None:
        """初始化动作记录。

        Returns:
            None: 此方法只初始化测试游戏。
        """
        self.actions: list[tuple[str, dict[str, int]]] = []

    def execute_action(self, action: str, **parameters: int) -> dict[str, Any]:
        """首次进入选牌过渡帧，第二次选择卡牌后进入奖励屏。

        Args:
            action (str): Runtime 提交的动作名称。
            parameters (int): Runtime 提交的动作参数。

        Returns:
            dict[str, Any]: 当前动作完成后的稳定状态。
        """
        self.actions.append((action, parameters))
        if len(self.actions) == 1:
            return {
                "status": "completed",
                "stable": True,
                "state": _combat_selection_state(available_actions=[]),
            }
        return {
            "status": "completed",
            "stable": True,
            "state": _reward_state(),
        }

    def state(self) -> dict[str, Any]:
        """返回已经开放选牌动作的稳定战斗状态。

        Returns:
            dict[str, Any]: 可以选择第 0 张牌的战斗内选牌状态。
        """
        return _combat_selection_state()


def test_battle_runner_keeps_history_and_waits_for_pending_action() -> None:
    """战斗循环等待下一可决策帧，并把前一步动作留在后续对话中。

    Raises:
        AssertionError: Runtime 读取过渡帧、丢失历史或未正确结束战斗。

    Returns:
        None: 此测试只验证两步战斗的完整闭环。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    game = PendingBattleGame()
    provider = BattleProvider(["ACTION: end_turn", "ACTION: end_turn"])

    result = runtime.BattleRunner(
        game,
        provider,
        poll_interval=0,
        state_timeout=1,
    ).run(_combat_state())

    assert result.outcome is runtime.BattleOutcome.CLEARED
    assert len(result.steps) == 2
    assert result.final_state == _reward_state()
    assert game.actions == ["end_turn", "end_turn"]
    assert game.state_calls == 2
    assert tuple(message.role for message in provider.requests[1]) == (
        "system",
        "user",
        "assistant",
        "user",
    )
    assert provider.requests[1][2].content == "ACTION: end_turn"
    assert "回合 2" in provider.requests[1][-1].content


def test_battle_runner_reports_player_death() -> None:
    """角色生命归零离开战斗时返回死亡结果。

    Raises:
        AssertionError: Runtime 把死亡误判为普通战斗胜利。

    Returns:
        None: 此测试只验证战斗出口分类。
    """
    runtime = importlib.import_module("play_sts2.runtime")

    result = runtime.BattleRunner(
        FinishingBattleGame(_game_over_state()),
        BattleProvider(["ACTION: end_turn"]),
    ).run(_combat_state())

    assert result.outcome is runtime.BattleOutcome.DIED
    assert len(result.steps) == 1
    assert result.final_state == _game_over_state()


def test_battle_runner_reports_final_boss_victory() -> None:
    """最终 Boss 战直接进入胜利终局时仍返回战斗通关。

    Raises:
        AssertionError: Runtime 未按 Mod 的 ``is_victory`` 字段识别胜利。

    Returns:
        None: 此测试只验证最终 Boss 战的终局分类。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    final_state = _victory_game_over_state()

    result = runtime.BattleRunner(
        FinishingBattleGame(final_state),
        BattleProvider(["ACTION: end_turn"]),
    ).run(_combat_state())

    assert result.outcome is runtime.BattleOutcome.CLEARED
    assert result.final_state == final_state


def test_battle_runner_leaves_combat_for_strategic_card_reward() -> None:
    """战后卡牌奖励即使仍带战斗标记，也应交还战略层。

    Raises:
        AssertionError: Runtime 把奖励选牌误判为仍需战斗模型处理。

    Returns:
        None: 此测试只验证 Harness 层级优先于残留战斗标记。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    final_state = _card_reward_state()

    result = runtime.BattleRunner(
        FinishingBattleGame(final_state),
        BattleProvider(["ACTION: end_turn"]),
        poll_interval=0,
        state_timeout=0,
    ).run(_combat_state())

    assert result.outcome is runtime.BattleOutcome.CLEARED
    assert result.final_state == final_state


def test_battle_runner_leaves_combat_for_any_strategic_screen() -> None:
    """战斗结束后进入事件页面时应立即交还整局调度。

    Raises:
        AssertionError: BattleRunner 只承认固定的少数战斗出口。

    Returns:
        None: 此测试验证战斗层与战略层的通用交接。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    final_state = {
        "screen": "EVENT",
        "in_combat": False,
        "available_actions": ["choose_event_option"],
        "run": {"current_hp": 65, "max_hp": 75},
        "event": {
            "title": "战后事件",
            "options": [{"index": 0, "title": "继续"}],
        },
    }

    result = runtime.BattleRunner(
        FinishingBattleGame(final_state),
        BattleProvider(["ACTION: end_turn"]),
        poll_interval=0,
        state_timeout=0,
    ).run(_combat_state())

    assert result.outcome is runtime.BattleOutcome.CLEARED
    assert result.final_state == final_state


def test_battle_runner_waits_for_transient_combat_card_selection() -> None:
    """战斗内选牌的空动作过渡帧应等待，而不是提前结束战斗。

    Raises:
        AssertionError: Runtime 没有等待并执行稳定后的选牌动作。

    Returns:
        None: 此测试只验证战斗内选牌过渡状态。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    game = SelectionBattleGame()
    provider = BattleProvider(["ACTION: end_turn", "ACTION: select_deck_card 0"])

    result = runtime.BattleRunner(
        game,
        provider,
        poll_interval=0,
        state_timeout=1,
    ).run(_combat_state())

    assert result.outcome is runtime.BattleOutcome.CLEARED
    assert game.actions == [
        ("end_turn", {}),
        ("select_deck_card", {"option_index": 0}),
    ]
    assert "选择卡牌" in provider.requests[1][-1].content


def test_battle_runner_allows_three_retries_by_default() -> None:
    """默认策略允许首次失败后的三次重试。

    Raises:
        AssertionError: 默认重试次数不是三次或非法回复触发了游戏动作。

    Returns:
        None: 此测试只验证 BattleRunner 的默认重试边界。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    provider = BattleProvider(["无效一", "无效二", "无效三", "ACTION: end_turn"])

    result = runtime.BattleRunner(
        FinishingBattleGame(_reward_state()),
        provider,
    ).run(_combat_state())

    assert len(provider.requests) == 4
    assert len(result.steps[0].retry_errors) == 3


def test_battle_runner_stops_at_action_limit() -> None:
    """达到动作上限后在请求下一次模型决策前停止。

    Raises:
        AssertionError: Runtime 超过配置动作上限后仍继续调用模型。

    Returns:
        None: 此测试使用较小注入上限验证同一熔断语义。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    provider = BattleProvider(["ACTION: end_turn"])

    with pytest.raises(runtime.BattleRunError, match="战斗动作数超过上限: 1"):
        runtime.BattleRunner(
            FinishingBattleGame(_combat_state(turn=2)),
            provider,
            max_steps=1,
        ).run(_combat_state())

    assert len(provider.requests) == 1


def _combat_state(
    *,
    turn: int = 1,
    available_actions: Sequence[str] = ("end_turn",),
) -> dict[str, Any]:
    """构造一个可以结束回合的最小战斗状态。

    Args:
        turn (int): 状态所属的战斗回合。
        available_actions (Sequence[str]): Mod 当前开放的动作。

    Returns:
        dict[str, Any]: 与 Harness 观测字段兼容的战斗状态。
    """
    return {
        "screen": "COMBAT",
        "in_combat": True,
        "turn": turn,
        "available_actions": list(available_actions),
        "run": {
            "character_name": "故障机器人",
            "current_hp": 70,
            "max_hp": 75,
            "potions": [],
        },
        "combat": {
            "player": {
                "current_hp": 70,
                "max_hp": 75,
                "block": 0,
                "energy": 3,
                "stars": 0,
                "orbs": [],
            },
            "enemies": [],
            "hand": [],
        },
    }


def _reward_state() -> dict[str, Any]:
    """构造战斗胜利后进入的奖励状态。

    Returns:
        dict[str, Any]: 已离开战斗的奖励状态。
    """
    return {
        "screen": "REWARD",
        "in_combat": False,
        "available_actions": ["claim_reward"],
        "run": {"current_hp": 65, "max_hp": 75},
    }


def _game_over_state() -> dict[str, Any]:
    """构造角色死亡后的游戏结束状态。

    Returns:
        dict[str, Any]: 生命为零的游戏结束状态。
    """
    return {
        "screen": "GAME_OVER",
        "in_combat": False,
        "available_actions": ["return_to_main_menu"],
        "run": {"current_hp": 0, "max_hp": 75},
        "game_over": {"is_victory": False},
    }


def _victory_game_over_state() -> dict[str, Any]:
    """构造最终 Boss 战胜利后的游戏结束状态。

    Returns:
        dict[str, Any]: 使用 Mod 真实字段表示胜利的终局状态。
    """
    return {
        "screen": "GAME_OVER",
        "in_combat": False,
        "available_actions": ["return_to_main_menu"],
        "run": {"current_hp": 18, "max_hp": 75},
        "game_over": {"is_victory": True},
    }


def _card_reward_state() -> dict[str, Any]:
    """构造仍带战斗标记的战后卡牌奖励状态。

    Returns:
        dict[str, Any]: Harness 明确归属战略层的奖励选牌状态。
    """
    return {
        "screen": "CARD_SELECTION",
        "in_combat": True,
        "available_actions": ["choose_reward_card", "skip_reward_cards"],
        "run": {"current_hp": 65, "max_hp": 75},
    }


def _combat_selection_state(
    *,
    available_actions: Sequence[str] = ("select_deck_card",),
) -> dict[str, Any]:
    """构造战斗机制产生的选牌状态。

    Args:
        available_actions (Sequence[str]): Mod 当前开放的选牌动作。

    Returns:
        dict[str, Any]: 过渡中或已可决策的战斗选牌状态。
    """
    return {
        "screen": "CARD_SELECTION",
        "in_combat": True,
        "available_actions": list(available_actions),
        "run": {
            "character_name": "故障机器人",
            "current_hp": 70,
            "max_hp": 75,
        },
        "selection": {
            "prompt": "选择一张牌",
            "selected_count": 0,
            "min_select": 1,
            "max_select": 1,
            "cards": [
                {
                    "index": 0,
                    "name": "打击",
                    "energy_cost": 1,
                    "resolved_rules_text": "造成6点伤害。",
                }
            ],
        },
    }
