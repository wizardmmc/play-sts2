"""驱动模型完成当前一场战斗。"""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..client import GameClient
from ..harness import (
    HarnessLayer,
    ObservationError,
    build_observation,
    format_action,
    state_layer,
)
from ..inference import ChatMessage, DecisionProvider
from .decision import DecisionEngine, DecisionStep
from .router import RunRoute, classify_run_state

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_MAX_STEPS = 300
_DEFAULT_POLL_INTERVAL = 0.2
_DEFAULT_STATE_TIMEOUT = 30.0


class BattleRunError(RuntimeError):
    """表示当前战斗无法继续形成可靠的模型决策闭环。"""


class BattleOutcome(str, Enum):
    """表示战斗循环的正常离场原因。"""

    CLEARED = "cleared"
    DIED = "died"


@dataclass(frozen=True, slots=True)
class BattleResult:
    """表示一场已经结束的战斗及其全部在线决策产物。

    Args:
        outcome (BattleOutcome): 战斗胜利离场或角色死亡。
        steps (tuple[DecisionStep, ...]): 按执行顺序保存的模型决策步骤。
        final_state (dict[str, Any]): 离开战斗后的第一份稳定状态。
    """

    outcome: BattleOutcome
    steps: tuple[DecisionStep, ...]
    final_state: dict[str, Any]


class BattleRunner:
    """维护一场战斗的对话，并持续驱动模型动作直到离场。"""

    def __init__(
        self,
        game: GameClient,
        provider: DecisionProvider,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        max_steps: int = _DEFAULT_MAX_STEPS,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        state_timeout: float = _DEFAULT_STATE_TIMEOUT,
    ) -> None:
        """初始化不拥有游戏与模型连接生命周期的战斗 Runner。

        Args:
            game (GameClient): 已连接到当前游戏实例的客户端。
            provider (DecisionProvider): 已连接到 Qwen 等推理服务的提供者。
            max_tokens (int): 每次模型生成允许使用的最大输出 token 数。
            temperature (float): 每次模型生成使用的采样温度。
            max_retries (int): 每个动作首次失败后允许的重试次数。
            max_steps (int): 单场战斗允许执行的最大动作数。
            poll_interval (float): 等待下一可决策状态时的轮询间隔秒数。
            state_timeout (float): 单次动作后等待稳定状态的最长秒数。

        Returns:
            None: 此方法只组合战斗闭环所需依赖。
        """
        self._game = game
        self._engine = DecisionEngine(
            game,
            provider,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._max_retries = max_retries
        self._max_steps = max_steps
        self._poll_interval = poll_interval
        self._state_timeout = state_timeout

    def run(self, initial_state: Mapping[str, Any] | None = None) -> BattleResult:
        """从当前战斗状态开始循环，直到战斗胜利或角色死亡。

        Args:
            initial_state (Mapping[str, Any] | None): 可选的初始状态；省略时
                从 Mod 读取当前状态。

        Raises:
            BattleRunError: 初始状态不在战斗中、状态等待超时或动作数超限。
            DecisionRetriesExhausted: 某一步的模型输出耗尽重试仍然非法。
            httpx.HTTPStatusError: 推理服务或 Mod 拒绝 HTTP 请求。

        Returns:
            BattleResult: 战斗出口、全部步骤与最终状态。
        """
        state = dict(initial_state) if initial_state is not None else self._game.state()
        if not _fight_ongoing(state):
            raise BattleRunError("当前状态不在战斗中")
        state = self._wait_for_state(state)
        history: tuple[ChatMessage, ...] = ()
        steps: list[DecisionStep] = []

        while _fight_ongoing(state):
            if len(steps) >= self._max_steps:
                raise BattleRunError(f"战斗动作数超过上限: {self._max_steps}")
            step = self._engine.step(
                state,
                history=history,
                max_retries=self._max_retries,
            )
            steps.append(step)
            history = (
                *step.messages[1:],
                ChatMessage(role="assistant", content=format_action(step.action)),
            )
            state = self._state_after(step.action_result)

        return BattleResult(
            outcome=_outcome(state),
            steps=tuple(steps),
            final_state=state,
        )

    def _state_after(self, action_result: Mapping[str, Any]) -> dict[str, Any]:
        """从动作结果或后续轮询中取得下一份可决策状态。

        Args:
            action_result (Mapping[str, Any]): Mod 返回的原始动作结果。

        Raises:
            BattleRunError: 在时限内没有等到可决策状态或战斗出口。

        Returns:
            dict[str, Any]: 下一份战斗决策状态或战斗离场状态。
        """
        raw_state = action_result.get("state")
        candidate = dict(raw_state) if isinstance(raw_state, Mapping) else None
        if action_result.get("stable") is True and candidate is not None:
            return self._wait_for_state(candidate)
        return self._wait_for_state()

    def _wait_for_state(
        self,
        candidate: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """等待下一份可生成观测的战斗状态或明确的战斗出口。

        Args:
            candidate (Mapping[str, Any] | None): 可先检查的动作结果内状态。

        Raises:
            BattleRunError: 在配置时限内没有取得可用状态。

        Returns:
            dict[str, Any]: 可交给模型的战斗状态或正常离场状态。
        """
        deadline = time.monotonic() + self._state_timeout
        state = dict(candidate) if candidate is not None else None
        while True:
            if state is not None and (
                _decision_ready(state) or _battle_finished(state)
            ):
                return state
            if time.monotonic() >= deadline:
                raise BattleRunError("等待下一可决策状态超时")
            if self._poll_interval:
                time.sleep(self._poll_interval)
            state = self._game.state()


def _fight_ongoing(state: Mapping[str, Any]) -> bool:
    """判断状态是否仍属于当前战斗。

    Args:
        state (Mapping[str, Any]): 待判断的完整游戏状态。

    Returns:
        bool: 屏幕或 Mod 标记仍在战斗中时为 ``True``。
    """
    if state.get("screen") == "CARD_SELECTION":
        return state_layer(state) is not HarnessLayer.STRATEGIC
    return state.get("screen") == "COMBAT" or state.get("in_combat") is True


def _decision_ready(state: Mapping[str, Any]) -> bool:
    """判断当前战斗状态能否生成一份合法模型观测。

    Args:
        state (Mapping[str, Any]): 待判断的战斗状态。

    Returns:
        bool: 状态属于战斗层且存在模型可见动作时为 ``True``。
    """
    if state_layer(state) is not HarnessLayer.BATTLE:
        return False
    try:
        build_observation(state)
    except ObservationError:
        return False
    return True


def _battle_finished(state: Mapping[str, Any]) -> bool:
    """判断状态是否已经进入可交还给整局调度的非战斗页面。

    Args:
        state (Mapping[str, Any]): 待判断的完整游戏状态。

    Returns:
        bool: 状态可由战略、终局或未知页面路由处理时为 ``True``。
    """
    route = classify_run_state(state)
    return not _fight_ongoing(state) and route in {
        RunRoute.STRATEGIC,
        RunRoute.TERMINAL,
        RunRoute.UNKNOWN,
    }


def _outcome(state: Mapping[str, Any]) -> BattleOutcome:
    """依据战斗离场状态区分胜利与角色死亡。

    Args:
        state (Mapping[str, Any]): 已经离开当前战斗的稳定状态。

    Returns:
        BattleOutcome: ``DIED`` 或 ``CLEARED``。
    """
    run = state.get("run") or {}
    game_over = state.get("game_over") or {}
    if run.get("current_hp") == 0:
        return BattleOutcome.DIED
    victory = game_over.get("is_victory", game_over.get("victory"))
    if state.get("screen") == "GAME_OVER" and victory is not True:
        return BattleOutcome.DIED
    return BattleOutcome.CLEARED
