"""定义推理提供者之间共享的不可变数据结构。"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """表示一条与具体推理后端无关的对话消息。

    Args:
        role (Literal["system", "user", "assistant"]): 消息在对话中的角色。
        content (str): 消息的纯文本内容。
    """

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class ModelReply:
    """表示推理后端返回的文本及可选用量信息。

    Args:
        text (str): 模型生成的原始文本。
        prompt_tokens (int | None): 服务端报告的输入 token 数。
        completion_tokens (int | None): 服务端报告的输出 token 数。
        cached_tokens (int | None): 命中前缀缓存的输入 token 数。
    """

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
