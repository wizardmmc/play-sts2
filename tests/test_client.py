"""根据上游响应协议验证 STS2 Mod HTTP 客户端。"""

import json
from threading import Event

import httpx
import pytest

from play_sts2.client import (
    AvailableAction,
    AvailableActions,
    GameClient,
    Health,
    ProtocolError,
)


def test_health_reads_mod_contract() -> None:
    """将成功的健康检查响应解析为不可变值对象。"""

    def respond(request: httpx.Request) -> httpx.Response:
        """为健康检查请求返回与生产环境结构一致的响应。

        Args:
            request (httpx.Request): 游戏客户端发出的请求。

        Raises:
            AssertionError: 客户端使用了预期外的请求方法或路径。

        Returns:
            httpx.Response: 成功的 Mod 健康检查响应。
        """
        assert request.method == "GET"
        assert request.url.path == "/health"
        assert request.extensions["timeout"]["read"] == 5.0

        return httpx.Response(
            200,
            json={
                "ok": True,
                "request_id": "req_test",
                "data": {
                    "service": "sts2-ai-agent",
                    "mod_version": "0.8.0",
                    "protocol_version": "2026-03-11-v1",
                    "game_version": "0.107.1",
                    "status": "ready",
                },
            },
        )

    transport = httpx.MockTransport(respond)

    with GameClient(
        "http://127.0.0.1:8080",
        transport=transport,
    ) as client:
        health = client.health()

    assert health == Health(
        service="sts2-ai-agent",
        mod_version="0.8.0",
        protocol_version="2026-03-11-v1",
        game_version="0.107.1",
        status="ready",
    )


def test_health_rejects_invalid_contract() -> None:
    """拒绝状态成功但缺少必需字段的响应。"""

    def respond(_request: httpx.Request) -> httpx.Response:
        """返回不完整响应以触发协议校验。

        Args:
            _request (httpx.Request): 游戏客户端发出的请求。

        Returns:
            httpx.Response: 缺少必需健康检查字段的响应。
        """
        return httpx.Response(
            200,
            json={
                "ok": True,
                "data": {
                    "status": "ready",
                },
            },
        )

    transport = httpx.MockTransport(respond)

    with (
        GameClient(
            "http://127.0.0.1:8080",
            transport=transport,
        ) as client,
        pytest.raises(
            ProtocolError,
            match="invalid /health response",
        ),
    ):
        client.health()


def test_state_returns_complete_mod_payload() -> None:
    """原样返回 Mod 提供的完整游戏状态对象。"""
    state_data = {
        "state_version": 10,
        "run_id": "TEST-SEED",
        "screen": "COMBAT",
        "session": {
            "mode": "singleplayer",
            "phase": "run",
            "control_scope": "local_player",
        },
        "in_combat": True,
        "turn": 1,
        "available_actions": ["end_turn", "play_card"],
        "combat": {"energy": 3},
        "run": {"floor": 1},
        "agent_view": {"format_version": 4},
    }

    def respond(request: httpx.Request) -> httpx.Response:
        """返回与上游根结构一致的游戏状态响应。

        Args:
            request (httpx.Request): 游戏客户端发出的请求。

        Raises:
            AssertionError: 客户端请求了预期外的路径。

        Returns:
            httpx.Response: 包含完整状态对象的 Mod 响应。
        """
        assert request.url.path == "/state"
        return httpx.Response(
            200,
            json={
                "ok": True,
                "request_id": "req_state",
                "data": state_data,
            },
        )

    transport = httpx.MockTransport(respond)

    with GameClient(
        "http://127.0.0.1:8080",
        transport=transport,
    ) as client:
        state = client.state()

    assert state == state_data


def test_iter_events_parses_multiline_sse_payload() -> None:
    """忽略 SSE 注释，并把多行 data 解析成一个 Mod 事件。

    Raises:
        AssertionError: 客户端请求或 SSE 解析结果不符合真实协议。

    Returns:
        None: 此测试仅验证只读事件流协议。
    """

    def respond(request: httpx.Request) -> httpx.Response:
        """返回包含心跳注释和多行 JSON 的有限事件流。

        Args:
            request (httpx.Request): 游戏客户端发出的流式请求。

        Raises:
            AssertionError: 请求方法、路径或 Accept 头不正确。

        Returns:
            httpx.Response: 可由测试完整消费的 SSE 响应。
        """
        assert request.method == "GET"
        assert request.url.path == "/events/stream"
        assert request.headers["accept"] == "text/event-stream"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b": heartbeat\n\n"
                b"event: stream_ready\n"
                b'data: {"event_id": 1,\n'
                b'data: "type": "stream_ready", "data": {}}\n\n'
            ),
        )

    with GameClient(
        "http://127.0.0.1:8080",
        transport=httpx.MockTransport(respond),
    ) as client:
        events = list(client.iter_events(Event()))

    assert events == [{"event_id": 1, "type": "stream_ready", "data": {}}]


def test_iter_events_rejects_non_object_payload() -> None:
    """拒绝根节点不是对象的 SSE data。

    Raises:
        AssertionError: 非对象事件没有触发 ``ProtocolError``。

    Returns:
        None: 此测试仅验证事件流的数据边界。
    """

    def respond(_request: httpx.Request) -> httpx.Response:
        """返回根节点为数组的非法 SSE 事件。

        Args:
            _request (httpx.Request): 游戏客户端发出的流式请求。

        Returns:
            httpx.Response: 包含非法事件的 SSE 响应。
        """
        return httpx.Response(200, content=b"data: []\n\n")

    with (
        GameClient(
            "http://127.0.0.1:8080",
            transport=httpx.MockTransport(respond),
        ) as client,
        pytest.raises(ProtocolError, match="invalid /events/stream response"),
    ):
        list(client.iter_events(Event()))


def test_available_actions_reads_typed_descriptors() -> None:
    """将动作端点解析为包含屏幕归属的不可变动作集合。"""

    def respond(request: httpx.Request) -> httpx.Response:
        """返回与上游动作描述符一致的响应。

        Args:
            request (httpx.Request): 游戏客户端发出的请求。

        Raises:
            AssertionError: 客户端请求了预期外的路径。

        Returns:
            httpx.Response: 包含两个合法动作描述符的 Mod 响应。
        """
        assert request.url.path == "/actions/available"
        return httpx.Response(
            200,
            json={
                "ok": True,
                "request_id": "req_actions",
                "data": {
                    "screen": "COMBAT",
                    "actions": [
                        {
                            "name": "end_turn",
                            "requires_target": False,
                            "requires_index": False,
                        },
                        {
                            "name": "play_card",
                            "requires_target": False,
                            "requires_index": True,
                        },
                    ],
                },
            },
        )

    transport = httpx.MockTransport(respond)

    with GameClient(
        "http://127.0.0.1:8080",
        transport=transport,
    ) as client:
        actions = client.available_actions()

    assert actions == AvailableActions(
        screen="COMBAT",
        actions=(
            AvailableAction(
                name="end_turn",
                requires_target=False,
                requires_index=False,
            ),
            AvailableAction(
                name="play_card",
                requires_target=False,
                requires_index=True,
            ),
        ),
    )


def test_execute_action_posts_parameters_and_returns_result() -> None:
    """发送动作参数并原样返回 Mod 的动作结果对象。

    Raises:
        AssertionError: 客户端没有遵守动作端点的请求协议。

    Returns:
        None: 此测试仅验证动作请求和响应行为。
    """
    action_data = {
        "action": "select_character",
        "status": "completed",
        "stable": True,
        "message": "Action completed.",
        "state": {"screen": "CHARACTER_SELECT"},
    }

    def respond(request: httpx.Request) -> httpx.Response:
        """校验写请求并返回与真实 Mod 一致的动作结果。

        Args:
            request (httpx.Request): 游戏客户端发出的动作请求。

        Raises:
            AssertionError: 请求方法、路径或 JSON 参数不正确。

        Returns:
            httpx.Response: 包含完整动作结果的成功响应。
        """
        assert request.method == "POST"
        assert request.url.path == "/action"
        assert request.extensions["timeout"]["read"] == 30.0
        assert json.loads(request.content) == {
            "action": "select_character",
            "option_index": 2,
        }
        return httpx.Response(
            200,
            json={
                "ok": True,
                "request_id": "req_action",
                "data": action_data,
            },
        )

    transport = httpx.MockTransport(respond)

    with GameClient(
        "http://127.0.0.1:8080",
        transport=transport,
    ) as client:
        result = client.execute_action("select_character", option_index=2)

    assert result == action_data


def test_execute_action_rejects_non_object_result() -> None:
    """拒绝 ``data`` 不是对象的动作成功响应。

    Raises:
        AssertionError: 客户端没有将非法动作响应转换为协议异常。

    Returns:
        None: 此测试仅验证动作响应的对象边界。
    """

    def respond(_request: httpx.Request) -> httpx.Response:
        """返回非对象动作结果以触发协议校验。

        Args:
            _request (httpx.Request): 游戏客户端发出的动作请求。

        Returns:
            httpx.Response: ``data`` 为列表的非法成功响应。
        """
        return httpx.Response(
            200,
            json={
                "ok": True,
                "request_id": "req_action",
                "data": [],
            },
        )

    transport = httpx.MockTransport(respond)

    with (
        GameClient(
            "http://127.0.0.1:8080",
            transport=transport,
        ) as client,
        pytest.raises(ProtocolError, match="invalid /action response"),
    ):
        client.execute_action("open_character_select")


def test_execute_action_timeout_is_configurable() -> None:
    """只为动作请求应用调用方配置的长超时。

    Raises:
        AssertionError: 动作请求没有使用指定的读取超时。

    Returns:
        None: 此测试仅验证动作超时配置。
    """

    def respond(request: httpx.Request) -> httpx.Response:
        """校验动作超时并返回成功响应。

        Args:
            request (httpx.Request): 游戏客户端发出的动作请求。

        Raises:
            AssertionError: 请求没有使用指定超时。

        Returns:
            httpx.Response: 最小合法动作响应。
        """
        assert request.extensions["timeout"]["read"] == 12.0
        return httpx.Response(
            200,
            json={
                "ok": True,
                "data": {"action": "end_turn"},
            },
        )

    with GameClient(
        "http://127.0.0.1:8080",
        transport=httpx.MockTransport(respond),
        action_timeout=12.0,
    ) as client:
        assert client.action_timeout == 12.0
        client.execute_action("end_turn")
