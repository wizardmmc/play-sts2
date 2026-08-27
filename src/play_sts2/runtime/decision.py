"""编排一次状态到游戏动作的在线决策闭环。"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..client import GameClient
from ..harness import (
    HarnessAction,
    Observation,
    build_observation,
    parse_action,
    system_prompt,
)
from ..inference import ChatMessage, DecisionProvider, ModelReply


@dataclass(frozen=True, slots=True)
class DecisionStep:
    """表示一次在线决策留下的完整可重放产物。

    Args:
        observation (Observation): 从动作前状态生成的模型观测。
        messages (tuple[ChatMessage, ...]): 本次实际发送给 Provider 的消息。
        reply (ModelReply): Provider 返回的原始文本与用量信息。
        action (HarnessAction): 通过 Harness 校验的结构化动作。
        action_result (dict[str, Any]): Mod 返回的原始动作结果。
    """

    observation: Observation
    messages: tuple[ChatMessage, ...]
    reply: ModelReply
    action: HarnessAction
    action_result: dict[str, Any]


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

    def step(self, state: Mapping[str, Any]) -> DecisionStep:
        """根据一个稳定游戏状态生成、校验并执行一次模型动作。

        Args:
            state (Mapping[str, Any]): Mod 返回的动作前稳定游戏状态。

        Raises:
            ObservationError: 当前状态无需模型决策或屏幕尚未支持。
            InferenceProtocolError: 推理服务成功响应违反协议。
            ActionParseError: 模型文本不是当前状态允许的规范动作。
            httpx.HTTPStatusError: 推理服务或 Mod 拒绝 HTTP 请求。
            ProtocolError: Mod 动作结果违反客户端协议。

        Returns:
            DecisionStep: 本次闭环的消息、回复、动作和 Mod 结果。
        """
        observation = build_observation(state)
        messages = (
            ChatMessage(
                role="system",
                content=system_prompt(observation.layer),
            ),
            ChatMessage(role="user", content=observation.text),
        )
        reply = self._provider.chat(
            messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        action = parse_action(reply.text, observation.available_actions)
        action_result = self._game.execute_action(
            action.name,
            **action.parameters,
        )
        return DecisionStep(
            observation=observation,
            messages=messages,
            reply=reply,
            action=action,
            action_result=action_result,
        )
