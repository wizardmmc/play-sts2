"""验证从主菜单选择角色并开始新局的最小流程。"""

import json

import httpx
import pytest

import play_sts2
from play_sts2.client import GameClient


def test_start_run_selects_requested_character_and_embarks() -> None:
    """按角色稳定 ID 选择对应索引，并在身份校验后返回新局状态。

    Raises:
        AssertionError: 流程使用了错误动作、角色索引或返回了错误新局状态。

    Returns:
        None: 此测试仅验证完整的最小开局流程。
    """
    initial_state = {
        "screen": "MAIN_MENU",
        "available_actions": ["open_character_select", "open_timeline"],
    }
    character_state = {
        "screen": "CHARACTER_SELECT",
        "available_actions": ["select_character", "embark"],
        "character_select": {
            "selected_character_id": "IRONCLAD",
            "characters": [
                {
                    "character_id": "IRONCLAD",
                    "index": 0,
                    "is_locked": False,
                },
                {
                    "character_id": "DEFECT",
                    "index": 4,
                    "is_locked": False,
                },
            ],
        },
    }
    selected_state = {
        **character_state,
        "character_select": {
            **character_state["character_select"],
            "selected_character_id": "DEFECT",
        },
    }
    run_state = {
        "screen": "MAP",
        "run_id": "TEST-RUN",
        "available_actions": ["choose_map_node"],
        "run": {
            "character_id": "DEFECT",
            "floor": 1,
        },
    }
    action_states = {
        "open_character_select": character_state,
        "select_character": selected_state,
        "embark": run_state,
    }
    requests: list[dict[str, object]] = []
    embark_pending = False

    def respond(request: httpx.Request) -> httpx.Response:
        """返回与真实 Mod 一致的菜单和动作状态。

        Args:
            request (httpx.Request): 开局流程发出的 HTTP 请求。

        Raises:
            AssertionError: 流程请求了预期外的端点或角色索引。

        Returns:
            httpx.Response: 当前步骤对应的 Mod 协议响应。
        """
        nonlocal embark_pending
        if request.method == "GET" and request.url.path == "/state":
            data = run_state if embark_pending else initial_state
        else:
            assert request.method == "POST"
            assert request.url.path == "/action"
            body = json.loads(request.content)
            requests.append(body)
            if body["action"] == "select_character":
                assert body["option_index"] == 4
            if body["action"] == "embark":
                embark_pending = True
                data = {
                    "action": "embark",
                    "status": "pending",
                    "stable": False,
                    "message": "Action queued but state is still transitioning.",
                    "state": selected_state,
                }
            else:
                data = {
                    "action": body["action"],
                    "status": "completed",
                    "stable": True,
                    "message": "Action completed.",
                    "state": action_states[body["action"]],
                }

        return httpx.Response(
            200,
            json={
                "ok": True,
                "request_id": "req_run_start",
                "data": data,
            },
        )

    transport = httpx.MockTransport(respond)

    with GameClient(
        "http://127.0.0.1:8080",
        transport=transport,
    ) as client:
        result = play_sts2.start_run(client, "DEFECT")

    assert result == run_state
    assert requests == [
        {"action": "open_character_select"},
        {"action": "select_character", "option_index": 4},
        {"action": "embark"},
    ]


def test_start_run_rejects_locked_character_before_selection() -> None:
    """目标角色锁定时停止流程，不发送选择或开局动作。

    Raises:
        AssertionError: 流程没有在锁定角色边界停止。

    Returns:
        None: 此测试仅验证锁定角色的失败关闭行为。
    """
    requests: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """返回包含锁定角色的真实协议形态。

        Args:
            request (httpx.Request): 开局流程发出的 HTTP 请求。

        Raises:
            AssertionError: 锁定检查前出现预期外的动作请求。

        Returns:
            httpx.Response: 主菜单或锁定角色选择状态响应。
        """
        if request.method == "GET":
            data = {
                "screen": "MAIN_MENU",
                "available_actions": ["open_character_select"],
            }
        else:
            body = json.loads(request.content)
            requests.append(body)
            assert body == {"action": "open_character_select"}
            data = {
                "action": "open_character_select",
                "status": "completed",
                "stable": True,
                "message": "Action completed.",
                "state": {
                    "screen": "CHARACTER_SELECT",
                    "available_actions": ["select_character", "embark"],
                    "character_select": {
                        "selected_character_id": "IRONCLAD",
                        "characters": [
                            {
                                "character_id": "DEFECT",
                                "index": 4,
                                "is_locked": True,
                            }
                        ],
                    },
                },
            }

        return httpx.Response(
            200,
            json={
                "ok": True,
                "request_id": "req_locked_character",
                "data": data,
            },
        )

    transport = httpx.MockTransport(respond)

    with (
        GameClient(
            "http://127.0.0.1:8080",
            transport=transport,
        ) as client,
        pytest.raises(play_sts2.RunStartError, match="角色已锁定"),
    ):
        play_sts2.start_run(client, "DEFECT")

    assert requests == [{"action": "open_character_select"}]


def test_start_run_sets_ascension_and_seed_before_embark() -> None:
    """在开局前把进阶和种子调整为明确目标。

    Raises:
        AssertionError: 开局动作顺序、参数或最终身份核对不正确。

    Returns:
        None: 此测试仅验证场景开局需要的确定性参数。
    """
    state = {
        "screen": "MAIN_MENU",
        "available_actions": ["open_character_select"],
    }
    requests: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """按动作推进一个包含进阶和种子的角色选择状态。

        Args:
            request (httpx.Request): 开局流程发出的 HTTP 请求。

        Raises:
            AssertionError: 流程请求了预期外的端点或动作。

        Returns:
            httpx.Response: 当前动作完成后的稳定 Mod 响应。
        """
        nonlocal state
        if request.method == "GET":
            data = state
        else:
            body = json.loads(request.content)
            requests.append(body)
            action = body["action"]
            if action == "open_character_select":
                state = {
                    "screen": "CHARACTER_SELECT",
                    "available_actions": [
                        "select_character",
                        "increase_ascension",
                        "set_seed",
                        "embark",
                    ],
                    "character_select": {
                        "selected_character_id": "IRONCLAD",
                        "ascension": 0,
                        "max_ascension": 20,
                        "seed": None,
                        "characters": [
                            {
                                "character_id": "DEFECT",
                                "index": 4,
                                "is_locked": False,
                            }
                        ],
                    },
                }
            elif action == "select_character":
                state["character_select"]["selected_character_id"] = "DEFECT"
            elif action == "increase_ascension":
                state["character_select"]["ascension"] += 1
            elif action == "set_seed":
                state["character_select"]["seed"] = body["game_seed"]
            else:
                assert action == "embark"
                state = {
                    "screen": "EVENT",
                    "available_actions": ["choose_event_option", "save_and_quit"],
                    "run": {
                        "character_id": "DEFECT",
                        "ascension": 2,
                        "floor": 0,
                    },
                }
            data = {
                "action": action,
                "status": "completed",
                "stable": True,
                "message": "Action completed.",
                "state": state,
            }
        return httpx.Response(200, json={"ok": True, "data": data})

    with GameClient(
        "http://127.0.0.1:8080",
        transport=httpx.MockTransport(respond),
    ) as client:
        result = play_sts2.start_run(
            client,
            "DEFECT",
            seed="ABCDEF1234",
            ascension=2,
        )

    assert result["run"] == {
        "character_id": "DEFECT",
        "ascension": 2,
        "floor": 0,
    }
    assert requests == [
        {"action": "open_character_select"},
        {"action": "select_character", "option_index": 4},
        {"action": "increase_ascension"},
        {"action": "increase_ascension"},
        {"action": "set_seed", "game_seed": "ABCDEF1234"},
        {"action": "embark"},
    ]
