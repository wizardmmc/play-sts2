"""通过 OpenAI-compatible chat completions 协议调用推理服务。"""

import logging
import math
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Self

import httpx

from .models import ChatMessage, ModelReply

_DEFAULT_TIMEOUT_SECONDS = 120.0
_CHAT_PATH = "/v1/chat/completions"
_THINKING_OUTPUT_INSTRUCTION = (
    "重要：ACTION 绝不能写在思考区内。完成分析后先输出 `</think>`，"
    "然后输出 `ACTION: <动作> <参数>`。"
)


class InferenceProtocolError(RuntimeError):
    """表示推理服务的成功响应不符合约定结构。"""


class InferenceModelIdentityError(InferenceProtocolError):
    """表示推理服务没有按请求确认同一个模型身份。"""


class InferenceGenerationTruncated(RuntimeError):
    """表示服务因 token 上限停止，回复不能安全交给动作解析器。"""

    def __init__(self, reply: ModelReply) -> None:
        """保存被截断的模型回复，供调用方输出诊断信息。

        Args:
            reply (ModelReply): 已解析但不能安全执行的截断回复。

        Returns:
            None: 此方法只初始化异常实例。
        """
        self.reply = reply
        super().__init__("chat completion generation truncated")


class OpenAICompatibleProvider:
    """持有与一个 OpenAI-compatible 推理服务通信的 HTTP 会话。"""

    def __init__(
        self,
        base_url: str,
        *,
        model: str | None = None,
        enable_thinking: bool | None = None,
        capture_token_metadata: bool = False,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
        retry_remote_disconnect: bool = False,
    ) -> None:
        """初始化推理服务连接。

        Args:
            base_url (str): 推理服务根地址。
            model (str | None): 服务端需要的模型名称；设置后同时要求响应回报
                完全相同的模型标识。
            enable_thinking (bool | None): 显式传给支持该扩展的 chat template；
                ``None`` 表示保持通用 OpenAI-compatible 请求。
            capture_token_metadata (bool): 是否请求 vLLM 扩展的生成 token ID
                和逐 token 行为策略 log-prob。
            timeout (float): 单次生成请求允许等待的秒数。
            transport (httpx.BaseTransport | None): 可选的 HTTP 传输实现。
            retry_remote_disconnect (bool): 未收到响应便断连时，是否允许同一
                无状态推理请求重发一次；不重试模型回复或游戏动作。

        Returns:
            None: 此方法在当前实例上完成初始化。
        """
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            transport=transport,
        )
        self._model = model
        self._enable_thinking = enable_thinking
        self._capture_token_metadata = capture_token_metadata
        self._retry_remote_disconnect = retry_remote_disconnect
        self._remote_disconnect_retries = 0

    @property
    def remote_disconnect_retries(self) -> int:
        """返回本实例已重发的无响应推理请求数，供采样收据审计。

        Returns:
            int: 累计发生的单次传输重试数量。
        """
        return self._remote_disconnect_retries

    @property
    def thinking_enabled(self) -> bool | None:
        """返回当前 Provider 实际发送的 thinking 配置。

        Returns:
            bool | None: 显式启用或禁用时返回对应布尔值，未指定时返回
            ``None``。
        """
        return self._enable_thinking

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
        response_choices: Sequence[str] | None = None,
    ) -> ModelReply:
        """请求一次非流式 chat completion。

        Args:
            messages (Sequence[ChatMessage]): 按时间顺序排列的完整对话。
            max_tokens (int): 本次生成允许使用的最大输出 token 数。
            temperature (float): 本次生成使用的采样温度。
            response_choices (Sequence[str] | None): 可选的完整回复候选；设置后
                使用 vLLM ``structured_outputs.choice`` 约束生成。

        Raises:
            httpx.HTTPStatusError: 推理服务返回非成功 HTTP 状态码。
            InferenceProtocolError: 成功响应不是合法 JSON 或结构不完整。
            InferenceModelIdentityError: 响应未确认请求绑定的模型标识。
            InferenceGenerationTruncated: 服务因 token 上限停止生成。

        Returns:
            ModelReply: 服务端生成的原始文本及可选用量信息。
        """
        body: dict[str, Any] = {
            "messages": _request_messages(
                messages,
                enable_thinking=self._enable_thinking is True,
            ),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if self._model is not None:
            body["model"] = self._model
        if self._enable_thinking is not None:
            body["chat_template_kwargs"] = {
                "enable_thinking": self._enable_thinking,
            }
        if self._capture_token_metadata:
            body.update(
                {
                    "logprobs": True,
                    "top_logprobs": 0,
                    "return_token_ids": True,
                }
            )
        if response_choices is not None:
            if not response_choices:
                raise ValueError("结构化回复候选不能为空")
            body["structured_outputs"] = {"choice": list(response_choices)}

        try:
            response = self._http.post(_CHAT_PATH, json=body)
        except httpx.RemoteProtocolError as exc:
            if not self._retry_remote_disconnect or str(exc) != (
                "Server disconnected without sending a response."
            ):
                raise
            self._remote_disconnect_retries += 1
            logging.getLogger(__name__).warning(
                "推理连接未返回响应，重发同一请求一次；累计次数=%s",
                self._remote_disconnect_retries,
            )
            response = self._http.post(_CHAT_PATH, json=body)
        response.raise_for_status()
        reply = _parse_reply(
            response,
            require_token_metadata=self._capture_token_metadata,
        )
        if self._model is not None and reply.model != self._model:
            raise InferenceModelIdentityError(
                f"chat completion model 不一致: expected={self._model!r}, "
                f"actual={reply.model!r}"
            )
        if reply.finish_reason == "length":
            raise InferenceGenerationTruncated(reply)
        return reply


def _request_messages(
    messages: Sequence[ChatMessage],
    *,
    enable_thinking: bool,
) -> list[dict[str, str]]:
    """构造请求消息，并为显式思考模式补充最终输出约束。

    Args:
        messages (Sequence[ChatMessage]): 调用方提供的完整对话。
        enable_thinking (bool): 是否启用推理服务的隐藏思考模板。

    Returns:
        list[dict[str, str]]: 可直接放入 chat completion 请求的消息列表。
    """
    request_messages = [
        {"role": message.role, "content": message.content} for message in messages
    ]
    if not enable_thinking:
        return request_messages
    for message in reversed(request_messages):
        if message["role"] == "user":
            message["content"] = (
                f"{message['content']}\n\n{_THINKING_OUTPUT_INSTRUCTION}"
            )
            break
    return request_messages


def _parse_reply(
    response: httpx.Response,
    *,
    require_token_metadata: bool = False,
) -> ModelReply:
    """把 chat completion 响应解析为统一回复对象。

    Args:
        response (httpx.Response): 状态码已经校验成功的 HTTP 响应。
        require_token_metadata (bool): 是否要求并解析 vLLM rollout 元数据。

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
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise InferenceProtocolError("invalid chat completion response")
    reasoning = message.get("reasoning", message.get("reasoning_content"))
    if reasoning is not None and not isinstance(reasoning, str):
        raise InferenceProtocolError("invalid chat completion response")
    text = message.get("content")
    if (
        text is None
        and message.get("role") == "assistant"
        and finish_reason in {"stop", "length"}
    ):
        text = ""
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
    model = payload.get("model")
    if model is not None and not isinstance(model, str):
        raise InferenceProtocolError("invalid chat completion response")
    token_ids, behavior_logprobs = _token_metadata(
        choice,
        required=require_token_metadata,
    )
    reply = ModelReply(
        text=text,
        prompt_tokens=_optional_int(usage, "prompt_tokens"),
        completion_tokens=_optional_int(usage, "completion_tokens"),
        cached_tokens=_optional_int(details, "cached_tokens"),
        reasoning=reasoning,
        finish_reason=finish_reason,
        model=model,
        token_ids=token_ids,
        behavior_logprobs=behavior_logprobs,
    )
    return reply


def _token_metadata(
    choice: Mapping[object, object],
    *,
    required: bool,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """读取并核对 vLLM chat completion 的 rollout token 元数据。

    Args:
        choice (Mapping[object, object]): 已校验为对象的首个生成 choice。
        required (bool): 缺少元数据时是否按协议错误处理。

    Raises:
        InferenceProtocolError: token ID、log-prob 或两者长度不合法。

    Returns:
        tuple[tuple[int, ...], tuple[float, ...]]: 对齐的 token ID 与
        行为策略 log-prob；普通推理不要求时返回两个空 tuple。
    """
    raw_ids = choice.get("token_ids")
    raw_logprobs = choice.get("logprobs")
    if raw_ids is None and raw_logprobs is None and not required:
        return (), ()
    if not isinstance(raw_ids, list) or not isinstance(raw_logprobs, Mapping):
        raise InferenceProtocolError("invalid chat completion response")
    content = raw_logprobs.get("content")
    if not isinstance(content, list) or len(content) != len(raw_ids):
        raise InferenceProtocolError("invalid chat completion response")

    token_ids: list[int] = []
    behavior_logprobs: list[float] = []
    for raw_id, entry in zip(raw_ids, content, strict=True):
        if isinstance(raw_id, bool) or not isinstance(raw_id, int) or raw_id < 0:
            raise InferenceProtocolError("invalid chat completion response")
        if not isinstance(entry, Mapping):
            raise InferenceProtocolError("invalid chat completion response")
        raw_logprob = entry.get("logprob")
        if (
            isinstance(raw_logprob, bool)
            or not isinstance(raw_logprob, (int, float))
            or not math.isfinite(raw_logprob)
        ):
            raise InferenceProtocolError("invalid chat completion response")
        token_ids.append(raw_id)
        behavior_logprobs.append(float(raw_logprob))
    return tuple(token_ids), tuple(behavior_logprobs)


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
