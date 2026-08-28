"""在战略与战斗之间调度模型，直到当前一局结束。"""

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from ..client import GameClient
from ..harness import HarnessLayer, format_action, shop_purchase_available
from ..inference import DecisionProvider
from .battle import BattleRunner
from .decision import (
    DecisionStep,
    is_action_window_conflict,
    stale_state_from_conflict,
    state_revision,
)
from .router import RunRoute, classify_run_state
from .strategic import StrategicRunner

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_MAX_STRATEGIC_STEPS = 400
_DEFAULT_STATE_TIMEOUT = 30.0
_SHOP_LOOP_ACTIONS = {"close_shop_inventory", "open_shop_inventory"}
_SHOP_LOOP_REPETITIONS = 3


class RunError(RuntimeError):
    """表示当前整局无法继续形成可靠的模型闭环。"""


class RunOutcome(str, Enum):
    """表示一局游戏的正常终局。"""

    VICTORY = "victory"
    DIED = "died"


@dataclass(frozen=True, slots=True)
class RunDecision:
    """表示整局时间线中的一次分层模型决策。

    Args:
        layer (HarnessLayer): 本次决策属于战略层或战斗层。
        step (DecisionStep): 本次决策的观测、消息、回复、动作和结果。
    """

    layer: HarnessLayer
    step: DecisionStep


@dataclass(frozen=True, slots=True)
class RunResult:
    """表示一局结束后的结果与完整在线决策时间线。

    Args:
        outcome (RunOutcome): 本局通关或角色死亡。
        decisions (tuple[RunDecision, ...]): 按真实执行顺序保存的全部决策。
        battle_count (int): 本局进入并完成的战斗数量。
        final_state (dict[str, Any]): Mod 返回的终局状态。
    """

    outcome: RunOutcome
    decisions: tuple[RunDecision, ...]
    battle_count: int
    final_state: dict[str, Any]


class RunRunner:
    """持续调度战略单步与单场战斗，直到 ``GAME_OVER``。"""

    def __init__(
        self,
        game: GameClient,
        provider: DecisionProvider,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        max_strategic_steps: int = _DEFAULT_MAX_STRATEGIC_STEPS,
        state_timeout: float = _DEFAULT_STATE_TIMEOUT,
    ) -> None:
        """初始化不拥有游戏与模型连接生命周期的整局 Runner。

        Args:
            game (GameClient): 已连接到当前游戏实例的客户端。
            provider (DecisionProvider): 已连接到推理服务的模型提供者。
            max_tokens (int): 每次模型回复允许生成的最大 token 数。
            temperature (float): 每次模型回复使用的采样温度。
            max_retries (int): 每个动作首次失败后允许的重试次数。
            max_strategic_steps (int): 单局允许执行的最大战略动作数。
            state_timeout (float): 单次等待可处理状态的最长秒数。

        Returns:
            None: 此方法只组合整局闭环需要的依赖与边界。
        """
        self._game = game
        self._battle = BattleRunner(
            game,
            provider,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            state_timeout=state_timeout,
        )
        self._strategic = StrategicRunner(
            game,
            provider,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
        )
        self._max_retries = max_retries
        self._max_strategic_steps = max_strategic_steps
        self._state_timeout = state_timeout

    def run(self, initial_state: Mapping[str, Any] | None = None) -> RunResult:
        """从当前新局状态开始，持续执行模型决策直到终局。

        Args:
            initial_state (Mapping[str, Any] | None): 可选的开局或续局状态；
                省略时从 Mod 读取当前状态。

        Raises:
            RunError: 状态等待超时、页面未知或战略动作数超过上限。
            BattleRunError: 单场战斗无法可靠完成。
            DecisionRetriesExhausted: 模型输出耗尽重试仍然非法。
            httpx.HTTPStatusError: 推理服务或 Mod 拒绝 HTTP 请求。

        Returns:
            RunResult: 本局结果、分层决策时间线、战斗数和终局状态。
        """
        state = dict(initial_state) if initial_state is not None else self._game.state()
        decisions: list[RunDecision] = []
        battle_count = 0
        strategic_steps = 0
        conflict_retries = 0
        shop_trace: list[tuple[str, str]] = []
        warned_shop_loops: set[tuple[tuple[str, str], ...]] = set()
        shop_notice_active = False

        while True:
            route = classify_run_state(state)
            if route is RunRoute.TRANSIENT:
                state = self._wait_for_route(
                    state,
                    after_revision=state_revision(state),
                )
                continue
            if route is RunRoute.UNKNOWN:
                raise RunError(f"未知游戏屏幕: {state.get('screen')}")
            if route is RunRoute.TERMINAL:
                return RunResult(
                    outcome=_terminal_outcome(state),
                    decisions=tuple(decisions),
                    battle_count=battle_count,
                    final_state=state,
                )
            if route is RunRoute.BATTLE:
                shop_trace.clear()
                shop_notice_active = False
                battle = self._battle.run(state)
                battle_count += 1
                decisions.extend(
                    RunDecision(HarnessLayer.BATTLE, step) for step in battle.steps
                )
                state = battle.final_state
                continue

            if strategic_steps >= self._max_strategic_steps:
                raise RunError(f"战略动作数超过上限: {self._max_strategic_steps}")
            try:
                notice = (
                    _shop_loop_notice(state)
                    if shop_notice_active
                    else _shop_exit_notice(state)
                )
                step = self._strategic.step(state, notice=notice)
            except httpx.HTTPStatusError as exc:
                latest = stale_state_from_conflict(exc)
                if latest is None and not is_action_window_conflict(exc):
                    raise
                if conflict_retries >= self._max_retries:
                    raise
                conflict_retries += 1
                state = self._wait_for_route(
                    latest,
                    after_revision=state_revision(state),
                )
                continue
            conflict_retries = 0
            strategic_steps += 1
            decisions.append(RunDecision(HarnessLayer.STRATEGIC, step))
            state = self._state_after(
                step.action_result,
                after_revision=state_revision(state),
            )
            shop_trace.append((step.observation.text, format_action(step.action)))
            del shop_trace[: -2 * _SHOP_LOOP_REPETITIONS]
            loop = _repeated_shop_loop(shop_trace)
            if loop is not None:
                shop_trace.clear()
                if loop in warned_shop_loops:
                    raise RunError(
                        "模型在纠偏提示后仍重复战略动作循环: "
                        "open_shop_inventory ↔ close_shop_inventory"
                    )
                warned_shop_loops.add(loop)
                shop_notice_active = True
            elif step.action.name not in _SHOP_LOOP_ACTIONS:
                shop_notice_active = False

    def _state_after(
        self,
        action_result: Mapping[str, Any],
        *,
        after_revision: int,
    ) -> dict[str, Any]:
        """从战略动作结果或后续事件取得下一份可处理状态。

        Args:
            action_result (Mapping[str, Any]): Mod 返回的原始动作结果。

        Raises:
            RunError: 在限定时间内没有取得可处理状态。

        Returns:
            dict[str, Any]: 可路由的战略、战斗、终局或未知状态。
        """
        raw_state = action_result.get("state")
        candidate = dict(raw_state) if isinstance(raw_state, Mapping) else None
        if action_result.get("stable") is True and candidate is not None:
            route = classify_run_state(candidate)
            if route is not RunRoute.TRANSIENT:
                return candidate
        return self._wait_for_route(
            candidate,
            after_revision=after_revision,
        )

    def _wait_for_route(
        self,
        candidate: Mapping[str, Any] | None = None,
        *,
        after_revision: int | None = None,
    ) -> dict[str, Any]:
        """等待动画过渡结束并取得下一份可分类状态。

        Args:
            candidate (Mapping[str, Any] | None): 可先检查的动作结果内状态。
            after_revision (int | None): 没有候选状态时已经处理的 revision。

        Raises:
            RunError: 在限定时间内状态始终属于过渡帧。

        Returns:
            dict[str, Any]: 首份不属于过渡帧的完整游戏状态。
        """
        deadline = time.monotonic() + self._state_timeout
        state = dict(candidate) if candidate is not None else None
        revision = state_revision(state) if state is not None else after_revision
        while True:
            if (
                state is not None
                and classify_run_state(state) is not RunRoute.TRANSIENT
            ):
                return state
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RunError("等待下一可处理状态超时")
            if revision is None:
                raise RunError("等待状态缺少 state_revision")
            try:
                state = self._game.wait_for_state(
                    after_revision=revision,
                    timeout=remaining,
                )
            except TimeoutError as exc:
                raise RunError("等待下一可处理状态超时") from exc
            revision = state_revision(state)


def _terminal_outcome(state: Mapping[str, Any]) -> RunOutcome:
    """依据 Mod 的原始终局字段区分通关与死亡。

    Args:
        state (Mapping[str, Any]): 已进入 ``GAME_OVER`` 的游戏状态。

    Returns:
        RunOutcome: 明确胜利时为 ``VICTORY``，否则为 ``DIED``。
    """
    game_over = state.get("game_over") or {}
    victory = game_over.get("is_victory", game_over.get("victory"))
    return RunOutcome.VICTORY if victory is True else RunOutcome.DIED


def _repeated_shop_loop(
    trace: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...] | None:
    """识别同两份模型观测之间连续三次开关商店的周期。

    Args:
        trace (Sequence[tuple[str, str]]): 按执行顺序记录的观测与动作行。

    Returns:
        tuple[tuple[str, str], ...] | None: 与起点无关的两步循环键；
        尚未形成真实商店周期时返回 ``None``。
    """
    window_size = 2 * _SHOP_LOOP_REPETITIONS
    if len(trace) < window_size:
        return None
    window = tuple(trace[-window_size:])
    cycle = window[:2]
    if window != cycle * _SHOP_LOOP_REPETITIONS:
        return None
    actions = {entry[1].split()[1] for entry in cycle}
    if actions != _SHOP_LOOP_ACTIONS:
        return None
    reversed_cycle = (cycle[1], cycle[0])
    return min(cycle, reversed_cycle)


def _shop_loop_notice(state: Mapping[str, Any]) -> str:
    """根据当前商店页面给出不代打的循环出口提示。

    Args:
        state (Mapping[str, Any]): 待交给战略模型的当前商店状态。

    Returns:
        str: 说明库存语义、当前真实出口和再次循环后果的提示。
    """
    actions = {str(action) for action in state.get("available_actions") or []}
    if "proceed" in actions:
        exit_hint = "立刻输出 `ACTION: proceed` 离开商店。"
    elif "close_shop_inventory" in actions:
        exit_hint = (
            "若不再购买，先输出 `ACTION: close_shop_inventory`；"
            "回到商店后再使用真实离开动作。"
        )
    else:
        exit_hint = "请使用当前页面真实的离开动作。"
    return (
        "注意：你已经连续重复商店开关动作循环。商店库存不会因关闭并重新打开而刷新；"
        f"{exit_hint}harness 不会替你操作；再次重复该循环将终止本局。"
    )


def _shop_exit_notice(state: Mapping[str, Any]) -> str | None:
    """在零购买力商店存在真实出口时给模型精确提示。

    Args:
        state (Mapping[str, Any]): 待交给战略模型的当前商店状态。

    Returns:
        str | None: 应附在观测后的退出提示；仍可购买或没有出口时为 ``None``。
    """
    actions = {str(action) for action in state.get("available_actions") or []}
    shop = state.get("shop")
    if (
        state.get("screen") != "SHOP"
        or "proceed" not in actions
        or not isinstance(shop, Mapping)
        or shop_purchase_available(shop) is not False
    ):
        return None
    return (
        "决策提示：当前没有任何可购买项目，关闭并重新打开库存也不会刷新商品。"
        "立刻输出 `ACTION: proceed` 离开商店。"
    )
