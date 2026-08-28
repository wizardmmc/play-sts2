"""验证 OpenAI-compatible 推理服务的最小通信契约。"""

import importlib
import json

import httpx
import pytest


def test_openai_provider_posts_messages_and_reads_reply() -> None:
    """发送标准 chat 请求，并读取文本、用量和前缀缓存命中数。

    Raises:
        AssertionError: 请求协议或回复投影不符合推理边界。

    Returns:
        None: 此测试只验证一次成功的模型调用。
    """

    def respond(request: httpx.Request) -> httpx.Response:
        """校验模型请求并返回生产等价的成功响应。

        Args:
            request (httpx.Request): Provider 发出的 HTTP 请求。

        Raises:
            AssertionError: 请求方法、路径、超时或 JSON 正文不正确。

        Returns:
            httpx.Response: OpenAI-compatible chat completion 响应。
        """
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        assert request.extensions["timeout"]["read"] == 45.0
        assert json.loads(request.content) == {
            "model": "/models/serving/demo-e3-mlx-8bit",
            "messages": [
                {"role": "system", "content": "只输出 ACTION。"},
                {"role": "user", "content": "可执行动作:\n- end_turn"},
            ],
            "max_tokens": 64,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,
        }
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "reasoning": "先确认唯一合法动作。",
                            "content": "ACTION: end_turn",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 321,
                    "completion_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 256},
                },
                "model": "/models/serving/demo-e3-mlx-8bit",
            },
        )

    inference = importlib.import_module("play_sts2.inference")
    messages = (
        inference.ChatMessage(role="system", content="只输出 ACTION。"),
        inference.ChatMessage(
            role="user",
            content="可执行动作:\n- end_turn",
        ),
    )

    with inference.OpenAICompatibleProvider(
        "http://127.0.0.1:8900",
        model="/models/serving/demo-e3-mlx-8bit",
        enable_thinking=False,
        timeout=45.0,
        transport=httpx.MockTransport(respond),
    ) as provider:
        reply = provider.chat(messages, max_tokens=64)

    assert reply == inference.ModelReply(
        text="ACTION: end_turn",
        prompt_tokens=321,
        completion_tokens=4,
        cached_tokens=256,
        reasoning="先确认唯一合法动作。",
        finish_reason="stop",
        model="/models/serving/demo-e3-mlx-8bit",
    )


@pytest.mark.parametrize("reported_model", [None, "default_model", "/models/e2"])
def test_openai_provider_rejects_unbound_response_model(
    reported_model: str | None,
) -> None:
    """请求绑定模型后，服务必须在响应中确认同一个模型标识。"""

    def respond(_request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = {
            "choices": [
                {
                    "message": {"content": "ACTION: end_turn"},
                    "finish_reason": "stop",
                }
            ]
        }
        if reported_model is not None:
            payload["model"] = reported_model
        return httpx.Response(200, json=payload)

    inference = importlib.import_module("play_sts2.inference")
    provider = inference.OpenAICompatibleProvider(
        "http://127.0.0.1:8900",
        model="/models/serving/demo-e3-mlx-8bit",
        transport=httpx.MockTransport(respond),
    )

    with provider, pytest.raises(inference.InferenceModelIdentityError):
        provider.chat((inference.ChatMessage(role="user", content="状态"),))


def test_openai_provider_checks_model_identity_before_length_classification() -> None:
    """错误模型的截断响应不能被 readiness 当作已成功加载目标模型。"""

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "/models/serving/demo-e2-mlx-8bit",
                "choices": [
                    {
                        "message": {"role": "assistant"},
                        "finish_reason": "length",
                    }
                ],
            },
        )

    inference = importlib.import_module("play_sts2.inference")
    provider = inference.OpenAICompatibleProvider(
        "http://127.0.0.1:8900",
        model="/models/serving/demo-e3-mlx-8bit",
        transport=httpx.MockTransport(respond),
    )

    with provider, pytest.raises(inference.InferenceModelIdentityError):
        provider.chat((inference.ChatMessage(role="user", content="状态"),))


def test_openai_provider_can_explicitly_enable_thinking() -> None:
    """思考 profile 必须通过 MLX 请求参数显式打开，而不是依赖模板默认值。"""

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["chat_template_kwargs"] == {"enable_thinking": True}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "reasoning": "分析局面",
                            "content": "ACTION: end_turn",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    inference = importlib.import_module("play_sts2.inference")
    provider = inference.OpenAICompatibleProvider(
        "http://127.0.0.1:8900",
        enable_thinking=True,
        transport=httpx.MockTransport(respond),
    )

    with provider:
        reply = provider.chat((inference.ChatMessage(role="user", content="状态"),))

    assert reply.reasoning == "分析局面"


def test_openai_provider_classifies_generation_truncated_during_thinking() -> None:
    """思考耗尽 token 时保留诊断信息并抛出专门错误，不能伪装成协议错误。"""

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "demo-e3",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "reasoning": "尚未完成的分析",
                        },
                        "finish_reason": "length",
                    }
                ],
                "usage": {"completion_tokens": 512},
            },
        )

    inference = importlib.import_module("play_sts2.inference")
    provider = inference.OpenAICompatibleProvider(
        "http://127.0.0.1:8900",
        enable_thinking=True,
        transport=httpx.MockTransport(respond),
    )

    with (
        provider,
        pytest.raises(inference.InferenceGenerationTruncated) as error,
    ):
        provider.chat((inference.ChatMessage(role="user", content="状态"),))

    assert error.value.reply.text == ""
    assert error.value.reply.reasoning == "尚未完成的分析"
    assert error.value.reply.finish_reason == "length"
    assert error.value.reply.model == "demo-e3"
    assert error.value.reply.completion_tokens == 512


def test_openai_provider_preserves_reasoning_only_stopped_reply() -> None:
    """MLX 正常停止但没有最终文本时，保留 reasoning 并交给 Harness 判空。"""

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "default_model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "reasoning": "分析结束但没有输出动作",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    inference = importlib.import_module("play_sts2.inference")
    provider = inference.OpenAICompatibleProvider(
        "http://127.0.0.1:8900",
        enable_thinking=True,
        transport=httpx.MockTransport(respond),
    )

    with provider:
        reply = provider.chat((inference.ChatMessage(role="user", content="状态"),))

    assert reply.text == ""
    assert reply.reasoning == "分析结束但没有输出动作"
    assert reply.finish_reason == "stop"


@pytest.mark.parametrize("text", ["", "  \n"])
def test_openai_provider_preserves_blank_model_reply(text: str) -> None:
    """原样返回空白模型文本，留给 Harness 判断动作格式错误。

    Args:
        text (str): 模型返回的空字符串或纯空白字符串。

    Raises:
        AssertionError: Provider 越权拒绝或改写了模型原始文本。

    Returns:
        None: 此测试只验证 inference 与 Harness 的错误归属边界。
    """

    def respond(_request: httpx.Request) -> httpx.Response:
        """返回结构合法但没有有效动作内容的模型回复。

        Args:
            _request (httpx.Request): Provider 发出的 HTTP 请求。

        Returns:
            httpx.Response: ``content`` 为测试文本的成功响应。
        """
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": text}}]},
        )

    inference = importlib.import_module("play_sts2.inference")
    provider = inference.OpenAICompatibleProvider(
        "http://127.0.0.1:8900",
        transport=httpx.MockTransport(respond),
    )

    with provider:
        reply = provider.chat((inference.ChatMessage(role="user", content="状态"),))

    assert reply.text == text


def test_openai_provider_omits_unconfigured_model() -> None:
    """本地服务未配置模型名时不发送 ``model: null``。

    Raises:
        AssertionError: 可选模型字段被错误地写入请求。

    Returns:
        None: 此测试只验证可选模型字段的请求行为。
    """

    def respond(request: httpx.Request) -> httpx.Response:
        """校验请求没有模型字段并返回最小成功响应。

        Args:
            request (httpx.Request): Provider 发出的 HTTP 请求。

        Raises:
            AssertionError: 请求正文仍然包含模型字段。

        Returns:
            httpx.Response: 不含用量信息的最小合法响应。
        """
        assert "model" not in json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ACTION: end_turn"}}]},
        )

    inference = importlib.import_module("play_sts2.inference")
    provider = inference.OpenAICompatibleProvider(
        "http://127.0.0.1:8900",
        transport=httpx.MockTransport(respond),
    )

    with provider:
        reply = provider.chat((inference.ChatMessage(role="user", content="状态"),))

    assert reply.prompt_tokens is None
    assert reply.completion_tokens is None
    assert reply.cached_tokens is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": "ACTION: end_turn"}}], "usage": []},
    ],
)
def test_openai_provider_rejects_malformed_success_response(payload: object) -> None:
    """拒绝 HTTP 成功但不符合 chat completion 结构的响应。

    Args:
        payload (object): 缺少必需字段或类型错误的响应 JSON。

    Returns:
        None: 此测试只验证协议错误会被明确分类。
    """

    def respond(_request: httpx.Request) -> httpx.Response:
        """返回待验证的畸形成功响应。

        Args:
            _request (httpx.Request): Provider 发出的 HTTP 请求。

        Returns:
            httpx.Response: 状态码成功但正文协议非法的响应。
        """
        return httpx.Response(200, json=payload)

    inference = importlib.import_module("play_sts2.inference")
    provider = inference.OpenAICompatibleProvider(
        "http://127.0.0.1:8900",
        transport=httpx.MockTransport(respond),
    )

    with (
        provider,
        pytest.raises(
            inference.InferenceProtocolError,
            match="invalid chat completion response",
        ),
    ):
        provider.chat((inference.ChatMessage(role="user", content="状态"),))


def test_openai_provider_preserves_http_error() -> None:
    """模型服务拒绝请求时保留 httpx 的状态码异常。

    Returns:
        None: 此测试只验证 HTTP 故障不会伪装成响应协议错误。
    """

    def respond(_request: httpx.Request) -> httpx.Response:
        """返回模型尚未就绪时的服务错误。

        Args:
            _request (httpx.Request): Provider 发出的 HTTP 请求。

        Returns:
            httpx.Response: HTTP 503 响应。
        """
        return httpx.Response(503, json={"error": "model loading"})

    inference = importlib.import_module("play_sts2.inference")
    provider = inference.OpenAICompatibleProvider(
        "http://127.0.0.1:8900",
        transport=httpx.MockTransport(respond),
    )

    with provider, pytest.raises(httpx.HTTPStatusError) as error:
        provider.chat((inference.ChatMessage(role="user", content="状态"),))

    assert error.value.response.status_code == 503
