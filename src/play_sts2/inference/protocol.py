"""声明整局 Runner 所依赖的最小推理协议。"""

from collections.abc import Sequence
from typing import Protocol

from .models import ChatMessage, ModelReply


class DecisionProvider(Protocol):
    """表示可以为一段对话生成下一条原始回复的提供者。"""

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        response_choices: Sequence[str] | None = None,
    ) -> ModelReply:
        """依据完整消息序列生成下一条回复。

        Args:
            messages (Sequence[ChatMessage]): 按时间顺序排列的完整对话。
            max_tokens (int): 本次生成允许使用的最大输出 token 数。
            temperature (float): 本次生成使用的采样温度。
            response_choices (Sequence[str] | None): 可选的完整回复候选集合。

        Returns:
            ModelReply: 尚未经过 Harness 动作解析的原始回复。
        """
        ...
