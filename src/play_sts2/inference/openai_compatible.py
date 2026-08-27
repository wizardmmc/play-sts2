"""通过 OpenAI-compatible chat completions 协议调用推理服务。"""

from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Self

import httpx

from .models import ChatMessage, ModelReply

_DEFAULT_TIMEOUT_SECONDS = 120.0
_CHAT_PATH = "/v1/chat/completions"


class InferenceProtocolError(RuntimeError):
    """表示推理服务的成功响应不符合约定结构。"""


class OpenAICompatibleProvider:
    """持有与一个 OpenAI-compatible 推理服务通信的 HTTP 会话。"""

    def __init__(
        self,
        base_url: str,
        *,
        model: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """初始化推理服务连接。

        Args:
            base_url (str): 推理服务根地址。
            model (str | None): 服务端需要的模型名称；本地服务允许省略。
            timeout (float): 单次生成请求允许等待的秒数。
            transport (httpx.BaseTransport | None): 可选的 HTTP 传输实现。

        Returns:
            None: 此方法在当前实例上完成初始化。
        """
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            transport=transport,
        )
        self._model = model

    def __enter__(self) -> Self:
        """进入持有底层 HTTP 会话的上下文。

        Returns:
            Self: 当前 Provider 实例。
        """
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """离开上下文时关闭 HTTP 会话。

        Args:
            _exc_type (type[BaseException] | None): 上下文失败时的异常类型。
            _exc (BaseException | None): 上下文失败时的异常实例。
            _traceback (TracebackType | None): 上下文失败时的异常调用栈。

        Returns:
            None: 此方法释放资源，但不屏蔽上下文中的异常。
        """
        self.close()

    def close(self) -> None:
        """释放 Provider 持有的网络资源。

        Returns:
            None: 此方法关闭底层 HTTP 会话。
        """
        self._http.close()

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> ModelReply:
        """请求一次非流式 chat completion。

        Args:
            messages (Sequence[ChatMessage]): 按时间顺序排列的完整对话。
            max_tokens (int): 本次生成允许使用的最大输出 token 数。
            temperature (float): 本次生成使用的采样温度。

        Raises:
            httpx.HTTPStatusError: 推理服务返回非成功 HTTP 状态码。
            InferenceProtocolError: 成功响应不是合法 JSON 或结构不完整。

        Returns:
            ModelReply: 服务端生成的原始文本及可选用量信息。
        """
        body: dict[str, Any] = {
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if self._model is not None:
            body["model"] = self._model

        response = self._http.post(_CHAT_PATH, json=body)
        response.raise_for_status()
        return _parse_reply(response)


def _parse_reply(response: httpx.Response) -> ModelReply:
    """把 chat completion 响应解析为统一回复对象。

    Args:
        response (httpx.Response): 状态码已经校验成功的 HTTP 响应。

    Raises:
        InferenceProtocolError: 响应 JSON 或必需字段不符合协议。

    Returns:
        ModelReply: 已投影为稳定字段的模型回复。
    """
    try:
        payload = response.json()
    except ValueError as exc:
        raise InferenceProtocolError("invalid chat completion response") from exc
    if not isinstance(payload, Mapping):
        raise InferenceProtocolError("invalid chat completion response")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise InferenceProtocolError("invalid chat completion response")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise InferenceProtocolError("invalid chat completion response")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise InferenceProtocolError("invalid chat completion response")
    text = message.get("content")
    if not isinstance(text, str):
        raise InferenceProtocolError("invalid chat completion response")

    usage = payload.get("usage")
    if usage is None:
        usage = {}
    if not isinstance(usage, Mapping):
        raise InferenceProtocolError("invalid chat completion response")
    details = usage.get("prompt_tokens_details")
    if details is None:
        details = {}
    if not isinstance(details, Mapping):
        raise InferenceProtocolError("invalid chat completion response")
    return ModelReply(
        text=text,
        prompt_tokens=_optional_int(usage, "prompt_tokens"),
        completion_tokens=_optional_int(usage, "completion_tokens"),
        cached_tokens=_optional_int(details, "cached_tokens"),
    )


def _optional_int(data: Mapping[object, object], field: str) -> int | None:
    """读取一个可缺省但出现时必须为整数的用量字段。

    Args:
        data (Mapping[object, object]): 包含用量信息的响应对象。
        field (str): 待读取的字段名称。

    Raises:
        InferenceProtocolError: 字段存在但不是整数。

    Returns:
        int | None: 服务端报告的整数，或字段缺省时的 ``None``。
    """
    value = data.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InferenceProtocolError("invalid chat completion response")
    return value
