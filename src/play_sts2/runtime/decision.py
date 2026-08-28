"""编排一次状态到游戏动作的在线决策闭环。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from ..client import GameClient, ProtocolError
from ..harness import (
    ActionParseError,
    HarnessAction,
    Observation,
    build_observation,
    parse_action,
    system_prompt,
)
from ..inference import ChatMessage, DecisionProvider, ModelReply

_RETRY_NOTE = (
    "注意: 上次动作无效({error})。立刻重新输出一行合法的 `ACTION: ...`，不要解释。"
)
_ACTION_WINDOW_MESSAGE = "Action is not available in the current state."


class DecisionRetriesExhausted(ActionParseError):
    """表示模型在有限重试内始终没有完成合法动作。"""

    def __init__(
        self,
        errors: Sequence[str],
        replies: Sequence[ModelReply],
    ) -> None:
        """保存每次失败原因与原始模型回复。

        Args:
            errors (Sequence[str]): 每次动作解析或执行被拒绝的错误信息。
            replies (Sequence[ModelReply]): 模型依次返回的原始回复。

        Returns:
            None: 此方法初始化可供上层记录的失败异常。
        """
        self.errors = tuple(errors)
        self.replies = tuple(replies)
        super().__init__(f"模型动作重试耗尽: {self.errors[-1]}")


@dataclass(frozen=True, slots=True)
class DecisionStep:
    """表示一次在线决策留下的完整可重放产物。

    Args:
        observation (Observation): 从动作前状态生成的模型观测。
        messages (tuple[ChatMessage, ...]): 本次实际发送给 Provider 的消息。
        replies (tuple[ModelReply, ...]): Provider 每次尝试返回的原始回复。
        retry_errors (tuple[str, ...]): 成功前各次动作失败或拒绝原因。
        action (HarnessAction): 通过 Harness 校验的结构化动作。
        action_result (dict[str, Any]): Mod 返回的原始动作结果。
    """

    observation: Observation
    messages: tuple[ChatMessage, ...]
    replies: tuple[ModelReply, ...]
    retry_errors: tuple[str, ...]
    action: HarnessAction
    action_result: dict[str, Any]

    @property
    def reply(self) -> ModelReply:
        """返回最终通过 Harness 校验的模型回复。

        Returns:
            ModelReply: 本步骤最后一次、也是成功的模型回复。
        """
        return self.replies[-1]


class DecisionEngine:
    """使用既有 Harness、Provider 和 GameClient 执行单个决策步骤。"""

    def __init__(
        self,
        game: GameClient,
        provider: DecisionProvider,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> None:
        """初始化不拥有外部资源生命周期的单步决策引擎。

        Args:
            game (GameClient): 已连接到目标游戏实例的客户端。
            provider (DecisionProvider): 为 Harness 消息生成原始回复的提供者。
            max_tokens (int): 每次模型生成允许使用的最大输出 token 数。
            temperature (float): 每次模型生成使用的采样温度。

        Returns:
            None: 此方法只保存闭环所需的依赖与生成参数。
        """
        self._game = game
        self._provider = provider
        self._max_tokens = max_tokens
        self._temperature = temperature

    def step(
        self,
        state: Mapping[str, Any],
        *,
        history: Sequence[ChatMessage] = (),
        notice: str | None = None,
        max_retries: int = 0,
    ) -> DecisionStep:
        """根据一个稳定游戏状态生成、校验并执行一次模型动作。

        Args:
            state (Mapping[str, Any]): Mod 返回的动作前稳定游戏状态。
            history (Sequence[ChatMessage]): 当前战斗中已经完成的消息历史，
                不包含固定的 system 消息。
            notice (str | None): 附加到当前观测后的运行时纠偏提示。
            max_retries (int): 首次输出失败后允许追加的重试次数。

        Raises:
            ObservationError: 当前状态无需模型决策或屏幕尚未支持。
            InferenceProtocolError: 推理服务成功响应违反协议。
            DecisionRetriesExhausted: 模型在有限尝试内没有生成合法动作。
            httpx.HTTPStatusError: 推理服务或 Mod 拒绝 HTTP 请求。
            ProtocolError: Mod 动作结果违反客户端协议。

        Returns:
            DecisionStep: 本次闭环的消息、回复、动作和 Mod 结果。
        """
        revision = state_revision(state)
        observation = build_observation(state)
        user_content = observation.text
        if notice:
            user_content = f"{user_content}\n\n{notice}"
        messages = [
            ChatMessage(
                role="system",
                content=system_prompt(observation.layer, state),
            ),
            *history,
            ChatMessage(role="user", content=user_content),
        ]
        replies: list[ModelReply] = []
        errors: list[str] = []
        for attempt in range(max_retries + 1):
            reply = self._provider.chat(
                messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            replies.append(reply)
            try:
                action = parse_action(reply.text, observation.available_actions)
                _validate_action_for_state(action, state)
            except ActionParseError as exc:
                errors.append(str(exc))
                if attempt == max_retries:
                    raise DecisionRetriesExhausted(errors, replies) from exc
                messages.append(
                    ChatMessage(
                        role="user",
                        content=(f"{user_content}\n\n{_RETRY_NOTE.format(error=exc)}"),
                    )
                )
                continue

            try:
                action_result = self._game.execute_action(
                    action.name,
                    expected_state_revision=revision,
                    **action.parameters,
                )
            except httpx.HTTPStatusError as exc:
                error = _model_action_rejection(exc)
                if error is None:
                    raise
                errors.append(error)
                if attempt == max_retries:
                    raise DecisionRetriesExhausted(errors, replies) from exc
                messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            f"{user_content}\n\n{_RETRY_NOTE.format(error=error)}"
                        ),
                    )
                )
                continue
            return DecisionStep(
                observation=observation,
                messages=tuple(messages),
                replies=tuple(replies),
                retry_errors=tuple(errors),
                action=action,
                action_result=action_result,
            )

        raise RuntimeError("模型动作循环意外结束")


def _validate_action_for_state(
    action: HarnessAction,
    state: Mapping[str, Any],
) -> None:
    """在提交 Mod 前核对模型动作引用的当前状态索引。"""
    if action.name != "play_card":
        return

    card_index = action.parameters["card_index"]
    hand = (state.get("combat") or {}).get("hand") or []
    card = next(
        (
            candidate
            for fallback_index, candidate in enumerate(hand)
            if isinstance(candidate, Mapping)
            and candidate.get("index", fallback_index) == card_index
        ),
        None,
    )
    if card is None:
        raise ActionParseError(f"card_index {card_index} 不在当前手牌中")
    if card.get("playable") is False:
        reason = card.get("unplayable_reason") or "未知原因"
        raise ActionParseError(f"卡牌 [{card_index}] 当前不可使用: {reason}")
    if card.get("requires_target") is not True:
        return

    valid_targets = [
        target
        for target in card.get("valid_target_indices") or []
        if isinstance(target, int) and not isinstance(target, bool)
    ]
    if not valid_targets:
        raise ActionParseError(f"卡牌 [{card_index}] 当前没有合法目标")
    target_index = action.parameters.get("target_index")
    if target_index is None:
        raise ActionParseError(
            f"卡牌 [{card_index}] 需要目标，合法目标 {valid_targets}"
        )
    if target_index not in valid_targets:
        raise ActionParseError(
            f"target_index {target_index} 不在合法目标 {valid_targets}"
        )


def is_action_window_conflict(exc: httpx.HTTPStatusError) -> bool:
    """判断 Mod 是否因输入窗口短暂关闭而拒绝动作。

    Args:
        exc (httpx.HTTPStatusError): Mod 返回的 HTTP 错误。

    Returns:
        bool: 仅对真实观测到的瞬时动作窗口冲突返回 ``True``。
    """
    error = _mod_error(exc)
    return (
        isinstance(error, Mapping)
        and error.get("code") == "invalid_action"
        and error.get("message") == _ACTION_WINDOW_MESSAGE
    )


def stale_state_from_conflict(
    exc: httpx.HTTPStatusError,
) -> dict[str, Any] | None:
    """从 Mod 的 revision 冲突中取得已在游戏线程读取的当前状态。"""
    error = _mod_error(exc)
    if not isinstance(error, Mapping) or error.get("code") != "stale_state":
        return None
    details = error.get("details")
    if not isinstance(details, Mapping):
        return None
    current_state = details.get("current_state")
    if not isinstance(current_state, Mapping):
        return None
    state = dict(current_state)
    state_revision(state)
    return state


def state_revision(state: Mapping[str, Any]) -> int:
    """读取动作并发控制所需的非负状态 revision。"""
    revision = state.get("state_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ProtocolError("game state is missing a valid state_revision")
    return revision


def _model_action_rejection(exc: httpx.HTTPStatusError) -> str | None:
    """提取适合反馈给模型自行纠正的 Mod 业务拒绝。

    Args:
        exc (httpx.HTTPStatusError): 执行模型动作时收到的 HTTP 错误。

    Returns:
        str | None: 可反馈的明确拒绝原因；瞬时窗口或其他 HTTP 错误返回
        ``None``。
    """
    error = _mod_error(exc)
    if not isinstance(error, Mapping):
        return None
    code = error.get("code")
    message = error.get("message")
    if (
        code not in {"invalid_action", "invalid_target"}
        or not isinstance(message, str)
        or (code == "invalid_action" and message == _ACTION_WINDOW_MESSAGE)
    ):
        return None
    return f"Mod 拒绝动作: {message}"


def _mod_error(exc: httpx.HTTPStatusError) -> Mapping[object, object] | None:
    """读取 Mod 409 响应中的结构化错误对象。

    Args:
        exc (httpx.HTTPStatusError): 待检查的 HTTP 错误。

    Returns:
        Mapping[object, object] | None: 结构合法的错误对象；状态码或响应体
        不匹配时返回 ``None``。
    """
    if exc.response.status_code != 409:
        return None
    try:
        payload = exc.response.json()
    except ValueError:
        return None
    if not isinstance(payload, Mapping):
        return None
    error = payload.get("error")
    return error if isinstance(error, Mapping) else None
