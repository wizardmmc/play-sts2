"""提供与具体模型或外部 Agent 解耦的推理契约。"""

from .local_model import prepare_model, serve_model
from .models import ChatMessage, ModelReply
from .openai_compatible import InferenceProtocolError, OpenAICompatibleProvider
from .protocol import DecisionProvider

__all__ = [
    "ChatMessage",
    "DecisionProvider",
    "InferenceProtocolError",
    "ModelReply",
    "OpenAICompatibleProvider",
    "prepare_model",
    "serve_model",
]
