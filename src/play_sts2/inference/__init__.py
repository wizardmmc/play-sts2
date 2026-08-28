"""提供与具体模型或外部 Agent 解耦的推理契约。"""

from .local_model import (
    ServingModelIdentityError,
    prepare_model,
    serve_model,
    validate_serving_model,
)
from .models import ChatMessage, ModelReply
from .openai_compatible import (
    InferenceGenerationTruncated,
    InferenceModelIdentityError,
    InferenceProtocolError,
    OpenAICompatibleProvider,
)
from .protocol import DecisionProvider

__all__ = [
    "ChatMessage",
    "DecisionProvider",
    "InferenceGenerationTruncated",
    "InferenceModelIdentityError",
    "InferenceProtocolError",
    "ModelReply",
    "OpenAICompatibleProvider",
    "ServingModelIdentityError",
    "prepare_model",
    "serve_model",
    "validate_serving_model",
]
