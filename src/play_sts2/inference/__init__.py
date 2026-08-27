"""提供与具体模型或外部 Agent 解耦的推理契约。"""

from .models import ChatMessage, ModelReply
from .openai_compatible import InferenceProtocolError, OpenAICompatibleProvider
from .protocol import DecisionProvider

__all__ = [
    "ChatMessage",
    "DecisionProvider",
    "InferenceProtocolError",
    "ModelReply",
    "OpenAICompatibleProvider",
]
