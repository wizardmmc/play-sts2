"""验证模型在一场战斗内持续观察、决策并执行动作。"""

import importlib
from collections.abc import Sequence
from typing import Any

import httpx
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
            "state": _reward_state(state_revision=4),
        }

    def wait_for_state(
        self,
        *,
        after_revision: int,
        timeout: float,
    ) -> dict[str, Any]:
        """先返回不可决策事件，再返回下一回合稳定状态。

        Returns:
            dict[str, Any]: 当前事件位置对应的游戏状态。
        """
        assert timeout > 0
        self.state_calls += 1
        if self.state_calls == 1:
            assert after_revision == 1
            return _combat_state(
                turn=2,
                available_actions=[],
                state_revision=2,
            )
        assert after_revision == 2
        state = _combat_state(turn=2, state_revision=3)
        state["combat"]["player"]["energy"] = 2
        return state


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
                "state": _combat_selection_state(
                    available_actions=[],
                    state_revision=2,
                ),
            }
        return {
            "status": "completed",
            "stable": True,
            "state": _reward_state(state_revision=4),
        }

    def wait_for_state(
        self,
        *,
        after_revision: int,
        timeout: float,
    ) -> dict[str, Any]:
        """返回已经开放选牌动作的稳定战斗状态。

        Returns:
            dict[str, Any]: 可以选择第 0 张牌的战斗内选牌状态。
        """
        assert after_revision == 2
        assert timeout > 0
        return _combat_selection_state(state_revision=3)


class ConflictingBattleGame:
    """模拟动作窗口短暂关闭后重新开放的真实战斗状态。"""

    def __init__(self, conflicts: int = 1) -> None:
        """初始化冲突数量与调用计数。

        Args:
            conflicts (int): 开始时连续返回的动作窗口冲突数。

        Returns:
            None: 此方法只初始化测试游戏。
        """
        self._conflicts = conflicts
        self.action_calls = 0
        self.state_calls = 0

    def execute_action(self, action: str, **_parameters: int) -> dict[str, Any]:
        """在配置次数内返回动作窗口冲突，随后结束战斗。

        Args:
            action (str): Runtime 提交的动作名称。
            _parameters (int): 当前测试不会使用的动作参数。

        Raises:
            httpx.HTTPStatusError: 动作仍在配置的冲突次数内时抛出。

        Returns:
            dict[str, Any]: 冲突结束后动作完成的奖励状态。
        """
        assert action == "end_turn"
        self.action_calls += 1
        if self.action_calls <= self._conflicts:
            raise _action_unavailable("end_turn")
        return {
            "status": "completed",
            "stable": True,
            "state": _reward_state(state_revision=3),
        }

    def wait_for_state(
        self,
        *,
        after_revision: int,
        timeout: float,
    ) -> dict[str, Any]:
        """返回重新开放动作的稳定战斗状态。

        Returns:
            dict[str, Any]: 可再次交给模型的战斗状态。
        """
        assert timeout > 0
        self.state_calls += 1
        return _combat_state(state_revision=after_revision + 1)


class EventDrivenBattleGame:
    """只通过事件等待交付下一决策状态，拒绝主动状态轮询。"""

    def __init__(self) -> None:
        """初始化动作与事件等待记录。

        Returns:
            None: 此方法只初始化测试替身。
        """
        self.actions: list[tuple[str, dict[str, int]]] = []
        self.waits: list[tuple[int, float]] = []

    def execute_action(self, action: str, **parameters: int) -> dict[str, Any]:
        """首次返回过渡状态，第二次结束模拟战斗。

        Args:
            action (str): Harness 提交的战斗动作。
            **parameters (int): 动作参数，包括期望状态 revision。

        Returns:
            dict[str, Any]: 过渡战斗状态或最终奖励状态。
        """
        self.actions.append((action, parameters))
        if len(self.actions) == 1:
            return {
                "status": "pending",
                "stable": False,
                "state": _combat_state(
                    turn=2,
                    available_actions=[],
                    state_revision=2,
                ),
            }
        return {
            "status": "completed",
            "stable": True,
            "state": _reward_state(state_revision=4),
        }

    def wait_for_state(
        self,
        *,
        after_revision: int,
        timeout: float,
    ) -> dict[str, Any]:
        """返回事件流交付的下一稳定战斗状态。

        Args:
            after_revision (int): 已消费的最新状态 revision。
            timeout (float): 事件等待超时秒数。

        Returns:
            dict[str, Any]: 事件流交付的下一决策状态。
        """
        self.waits.append((after_revision, timeout))
        state = _combat_state(turn=2, state_revision=3)
        state["combat"]["player"]["energy"] = 2
        return state

    def state(self) -> dict[str, Any]:
        """拒绝事件等待期间的主动状态轮询。

        Raises:
            AssertionError: 调用方在应使用事件流时轮询了状态。
        """
        raise AssertionError("事件等待期间不应轮询 /state")


class StaleRevisionBattleGame:
    """首次提交时报告观测已过期，随后接受新 revision 的动作。"""

    def __init__(self) -> None:
        """初始化过期观测后的最新战斗状态。

        Returns:
            None: 此方法只初始化测试替身。
        """
        self.actions: list[tuple[str, dict[str, int]]] = []
        self.current = _combat_state(turn=2, state_revision=2)
        self.current["combat"]["player"]["energy"] = 2

    def execute_action(self, action: str, **parameters: int) -> dict[str, Any]:
        """首次报告 revision 过期，重试时结束模拟战斗。

        Args:
            action (str): Harness 提交的战斗动作。
            **parameters (int): 包含期望状态 revision 的动作参数。

        Raises:
            httpx.HTTPStatusError: 首次动作使用了过期 revision。

        Returns:
            dict[str, Any]: 重试成功后的最终奖励状态。
        """
        self.actions.append((action, parameters))
        if len(self.actions) == 1:
            raise _stale_state(parameters["expected_state_revision"], self.current)
        return {
            "status": "completed",
            "stable": True,
            "state": _reward_state(state_revision=3),
        }


def test_battle_runner_waits_on_state_events_without_polling() -> None:
    """pending 动作通过 SSE 状态事件继续，而不是定时调用 ``/state``。"""
    runtime = importlib.import_module("play_sts2.runtime")
    game = EventDrivenBattleGame()
    provider = BattleProvider(["ACTION: end_turn", "ACTION: end_turn"])

    result = runtime.BattleRunner(game, provider, state_timeout=5).run(
        _combat_state(state_revision=1)
    )

    assert result.outcome is runtime.BattleOutcome.CLEARED
    assert [revision for revision, _ in game.waits] == [2]
    assert game.waits[0][1] == pytest.approx(5, abs=0.1)
    assert [
        parameters["expected_state_revision"] for _, parameters in game.actions
    ] == [
        1,
        3,
    ]


def test_battle_runner_discards_stale_reply_and_regenerates_from_latest_state() -> None:
    """revision 冲突不在旧 prompt 上重试，而是用 Mod 返回的新状态重新推理。"""
    runtime = importlib.import_module("play_sts2.runtime")
    game = StaleRevisionBattleGame()
    provider = BattleProvider(["ACTION: end_turn", "ACTION: end_turn"])

    result = runtime.BattleRunner(game, provider).run(_combat_state(state_revision=1))

    assert result.outcome is runtime.BattleOutcome.CLEARED
    assert len(result.steps) == 1
    assert [
        parameters["expected_state_revision"] for _, parameters in game.actions
    ] == [
        1,
        2,
    ]
    assert "能量2" in provider.requests[1][-1].content


def test_battle_runner_uses_stateless_steps_and_waits_for_pending_action() -> None:
    """战斗循环等待下一可决策帧，但不把前一步带入下一次请求。

    Raises:
        AssertionError: Runtime 读取过渡帧、保留历史或未正确结束战斗。

    Returns:
        None: 此测试只验证两步战斗的完整闭环。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    game = PendingBattleGame()
    provider = BattleProvider(["ACTION: end_turn", "ACTION: end_turn"])

    result = runtime.BattleRunner(
        game,
        provider,
        state_timeout=1,
    ).run(_combat_state())

    assert result.outcome is runtime.BattleOutcome.CLEARED
    assert len(result.steps) == 2
    assert result.final_state == _reward_state(state_revision=4)
    assert game.actions == ["end_turn", "end_turn"]
    assert game.state_calls == 2
    assert tuple(message.role for message in provider.requests[1]) == ("system", "user")
    assert "对话历史" not in provider.requests[1][0].content
    assert provider.requests[1][0].content == provider.requests[0][0].content
    assert "【当前回合】" not in provider.requests[1][0].content
    assert provider.requests[1][-1].content.startswith("玩家:")
    assert "能量2" in provider.requests[1][-1].content
    assert "角色:" not in provider.requests[1][-1].content


def test_battle_runner_retries_temporary_action_window_conflict() -> None:
    """Mod 暂时关闭输入窗口时重新等待状态并再次决策。

    Raises:
        AssertionError: 合法动作的短暂 409 直接终止战斗或污染成功步骤。

    Returns:
        None: 此测试只验证真实出现过的输入窗口竞争。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    game = ConflictingBattleGame()
    provider = BattleProvider(["ACTION: end_turn", "ACTION: end_turn"])

    result = runtime.BattleRunner(
        game,
        provider,
        state_timeout=1,
    ).run(_combat_state())

    assert result.outcome is runtime.BattleOutcome.CLEARED
    assert len(result.steps) == 1
    assert game.action_calls == 2
    assert game.state_calls == 1
    assert len(provider.requests) == 2


def test_battle_runner_stops_after_action_window_retry_limit() -> None:
    """动作窗口持续冲突时在既有重试上限处停止。

    Raises:
        AssertionError: Runtime 超过配置上限继续请求模型或提交动作。

    Returns:
        None: 此测试只验证瞬时冲突的有限重试边界。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    game = ConflictingBattleGame(conflicts=3)
    provider = BattleProvider(["ACTION: end_turn", "ACTION: end_turn"])

    with pytest.raises(httpx.HTTPStatusError, match="409 Conflict"):
        runtime.BattleRunner(
            game,
            provider,
            max_retries=1,
            state_timeout=1,
        ).run(_combat_state())

    assert game.action_calls == 2
    assert game.state_calls == 1
    assert len(provider.requests) == 2


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
        "state_revision": 2,
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
        state_timeout=1,
    ).run(_combat_state())

    assert result.outcome is runtime.BattleOutcome.CLEARED
    assert game.actions == [
        ("end_turn", {"expected_state_revision": 1}),
        (
            "select_deck_card",
            {"expected_state_revision": 3, "option_index": 0},
        ),
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
    state_revision: int = 1,
) -> dict[str, Any]:
    """构造一个可以结束回合的最小战斗状态。

    Args:
        turn (int): 状态所属的战斗回合。
        available_actions (Sequence[str]): Mod 当前开放的动作。
        state_revision (int): Mod 为动作并发控制分配的状态修订号。

    Returns:
        dict[str, Any]: 与 Harness 观测字段兼容的战斗状态。
    """
    return {
        "state_revision": state_revision,
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


def _reward_state(*, state_revision: int = 2) -> dict[str, Any]:
    """构造战斗胜利后进入的奖励状态。

    Returns:
        dict[str, Any]: 已离开战斗的奖励状态。
    """
    return {
        "state_revision": state_revision,
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
        "state_revision": 2,
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
        "state_revision": 2,
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
        "state_revision": 2,
        "screen": "CARD_SELECTION",
        "in_combat": True,
        "available_actions": ["choose_reward_card", "skip_reward_cards"],
        "run": {"current_hp": 65, "max_hp": 75},
    }


def _combat_selection_state(
    *,
    available_actions: Sequence[str] = ("select_deck_card",),
    state_revision: int = 2,
) -> dict[str, Any]:
    """构造战斗机制产生的选牌状态。

    Args:
        available_actions (Sequence[str]): Mod 当前开放的选牌动作。
        state_revision (int): 当前选牌状态的 revision。

    Returns:
        dict[str, Any]: 过渡中或已可决策的战斗选牌状态。
    """
    return {
        "state_revision": state_revision,
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


def _action_unavailable(action: str) -> httpx.HTTPStatusError:
    """构造与真实 Mod 输入窗口竞争一致的 409 异常。

    Args:
        action (str): 被暂时拒绝的动作名称。

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
                "details": {"action": action, "screen": "COMBAT"},
                "retryable": False,
            },
        },
    )
    return httpx.HTTPStatusError(
        "409 Conflict",
        request=request,
        response=response,
    )


def _stale_state(
    expected_revision: int,
    current_state: dict[str, Any],
) -> httpx.HTTPStatusError:
    """构造 Mod 对过期状态动作返回的结构化 409。"""
    request = httpx.Request("POST", "http://127.0.0.1:8080/action")
    response = httpx.Response(
        409,
        request=request,
        json={
            "ok": False,
            "error": {
                "code": "stale_state",
                "message": "Observed state revision is no longer current.",
                "details": {
                    "expected_state_revision": expected_revision,
                    "actual_state_revision": current_state["state_revision"],
                    "current_state": current_state,
                },
                "retryable": True,
            },
        },
    )
    return httpx.HTTPStatusError(
        "409 Conflict",
        request=request,
        response=response,
    )
