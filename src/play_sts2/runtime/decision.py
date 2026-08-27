"""编排一次状态到游戏动作的在线决策闭环。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..client import GameClient
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


class DecisionRetriesExhausted(ActionParseError):
    """表示模型在有限重试内始终没有给出合法动作。"""

    def __init__(
        self,
        errors: Sequence[str],
        replies: Sequence[ModelReply],
    ) -> None:
        """保存每次失败原因与原始模型回复。

        Args:
            errors (Sequence[str]): 每次动作解析失败的错误信息。
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
        retry_errors (tuple[str, ...]): 成功前各次失败的动作解析错误。
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
        max_retries: int = 0,
    ) -> DecisionStep:
        """根据一个稳定游戏状态生成、校验并执行一次模型动作。

        Args:
            state (Mapping[str, Any]): Mod 返回的动作前稳定游戏状态。
            history (Sequence[ChatMessage]): 当前战斗中已经完成的消息历史，
                不包含固定的 system 消息。
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
        observation = build_observation(state)
        messages = [
            ChatMessage(
                role="system",
                content=system_prompt(observation.layer),
            ),
            *history,
            ChatMessage(role="user", content=observation.text),
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
            except ActionParseError as exc:
                errors.append(str(exc))
                if attempt == max_retries:
                    raise DecisionRetriesExhausted(errors, replies) from exc
                messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            f"{observation.text}\n\n{_RETRY_NOTE.format(error=exc)}"
                        ),
                    )
                )
                continue

            action_result = self._game.execute_action(
                action.name,
                **action.parameters,
            )
            return DecisionStep(
                observation=observation,
                messages=tuple(messages),
                replies=tuple(replies),
                retry_errors=tuple(errors),
                action=action,
                action_result=action_result,
            )

        raise RuntimeError("模型动作循环意外结束")
