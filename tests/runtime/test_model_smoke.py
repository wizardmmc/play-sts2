"""验证模型服务与真实 Harness 之间的最小协议冒烟。"""

import json

import httpx


def test_smoke_model_checks_health_and_generates_legal_action() -> None:
    """冒烟检查先访问健康端点，再用真实 Harness 契约生成合法动作。

    Raises:
        AssertionError: 请求顺序、提示词内容或生成参数不符合约定。

    Returns:
        None: 此测试只验证 OpenAI-compatible 服务协议。
    """
    from play_sts2.runtime.model_smoke import smoke_model

    paths: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """返回 MLX 健康响应和合法 chat completion。

        Args:
            request (httpx.Request): 本地模型检查发出的 HTTP 请求。

        Raises:
            AssertionError: 请求不是健康检查或 Harness 生成请求。

        Returns:
            httpx.Response: 对应端点的生产等价响应。
        """
        paths.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["max_tokens"] == 48
        assert body["temperature"] == 0.0
        assert "战斗决策模型" in body["messages"][0]["content"]
        assert body["messages"][1]["content"].endswith("可执行动作:\n- end_turn")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ACTION: end_turn"}}]},
        )

    reply = smoke_model(
        "http://127.0.0.1:8900",
        transport=httpx.MockTransport(respond),
    )

    assert paths == ["/health", "/v1/chat/completions"]
    assert reply.text == "ACTION: end_turn"
