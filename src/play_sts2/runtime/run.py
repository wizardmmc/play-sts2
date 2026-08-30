"""在战略与战斗之间调度模型，直到当前一局结束。"""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from ..client import GameClient
from ..harness import HarnessLayer, shop_purchase_available
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
        card_reward_skipped = False
        shop_inventory_closed = False

        while True:
            screen = str(state.get("screen") or "")
            if screen not in {"REWARD", "CARD_SELECTION"}:
                card_reward_skipped = False
            if screen != "SHOP":
                shop_inventory_closed = False
            model_state = project_strategic_model_state(
                state,
                card_reward_skipped=card_reward_skipped,
                shop_inventory_closed=shop_inventory_closed,
            )
            route = classify_run_state(model_state)
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
                step = self._strategic.step(
                    model_state,
                    notice=_shop_exit_notice(model_state),
                )
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
            next_state = self._state_after(
                step.action_result,
                after_revision=state_revision(state),
            )
            if step.action.name == "skip_reward_cards":
                card_reward_skipped = True
            if step.action.name == "close_shop_inventory":
                shop_inventory_closed = True
            state = next_state

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
                latest = self._game.state()
                latest_revision = state_revision(latest)
                latest_route = classify_run_state(latest)
                if (
                    latest_revision is not None
                    and latest_revision > revision
                    and latest_route not in {RunRoute.TRANSIENT, RunRoute.UNKNOWN}
                ):
                    return latest
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


def _hide_skipped_card_rewards(
    state: Mapping[str, Any],
    *,
    hide_card_rewards: bool,
) -> dict[str, Any]:
    """在当前奖励流程内隐藏模型已经明确跳过的卡牌入口。

    只改变交给模型的状态副本，不执行游戏动作，也不隐藏金币、药水、遗物
    等其他奖励。奖励索引会在领取其他奖励后重排，因此这里按奖励类型遮蔽，
    离开奖励流程后由调用方恢复显示。

    Args:
        state (Mapping[str, Any]): Mod 返回的当前完整状态。
        hide_card_rewards (bool): 本次奖励流程是否已经跳过卡牌选择。

    Returns:
        dict[str, Any]: 可安全交给战略 Harness 的状态副本。
    """
    model_state = dict(state)
    if state.get("screen") != "REWARD" or not hide_card_rewards:
        return model_state
    reward = state.get("reward")
    if not isinstance(reward, Mapping):
        return model_state
    rewards = reward.get("rewards") or []
    visible_rewards = [
        item
        for item in rewards
        if not (
            isinstance(item, Mapping)
            and str(item.get("reward_type") or "").casefold() == "card"
        )
    ]
    if len(visible_rewards) == len(rewards):
        return model_state
    model_reward = dict(reward)
    model_reward["rewards"] = visible_rewards
    model_state["reward"] = model_reward
    if not any(
        isinstance(item, Mapping) and item.get("claimable") is not False
        for item in visible_rewards
    ):
        model_state["available_actions"] = [
            action
            for action in state.get("available_actions") or []
            if action != "claim_reward"
        ]
    return model_state


def project_strategic_model_state(
    state: Mapping[str, Any],
    *,
    card_reward_skipped: bool,
    shop_inventory_closed: bool,
) -> dict[str, Any]:
    """应用跨战略动作的奖励与商店可见性记忆。

    Args:
        state (Mapping[str, Any]): Mod 返回的当前完整状态。
        card_reward_skipped (bool): 当前奖励流程是否已跳过卡牌入口。
        shop_inventory_closed (bool): 当前商店访问是否已主动关闭库存。

    Returns:
        dict[str, Any]: 可交给整局或 Tree 战略 policy 的状态副本。
    """
    model_state = _hide_skipped_card_rewards(
        state,
        hide_card_rewards=card_reward_skipped,
    )
    return _hide_closed_shop_inventory(
        model_state,
        inventory_closed=shop_inventory_closed,
    )


def _hide_closed_shop_inventory(
    state: Mapping[str, Any],
    *,
    inventory_closed: bool,
) -> dict[str, Any]:
    """在同一次商店访问关闭库存后隐藏重新打开动作。

    Args:
        state (Mapping[str, Any]): Mod 返回或已做奖励遮蔽的模型状态。
        inventory_closed (bool): 当前商店访问是否已经主动关闭过库存。

    Returns:
        dict[str, Any]: 仅移除无新信息重开动作的状态副本。
    """
    model_state = dict(state)
    if state.get("screen") != "SHOP" or not inventory_closed:
        return model_state
    model_state["available_actions"] = [
        action
        for action in state.get("available_actions") or []
        if action != "open_shop_inventory"
    ]
    return model_state


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
