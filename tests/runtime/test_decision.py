"""验证 Runtime 把稳定状态闭环为一次真实游戏动作。"""

import importlib
import json
from typing import Any

import httpx
import pytest

from play_sts2.client import GameClient
from play_sts2.harness import ActionParseError
from play_sts2.inference import OpenAICompatibleProvider


def test_decision_engine_executes_model_action() -> None:
    """将战斗状态依次转换为消息、模型动作和 Mod 请求。

    Raises:
        AssertionError: 闭环遗漏提示词、错误映射动作参数或丢失步骤产物。

    Returns:
        None: 此测试只验证一个成功决策步骤。
    """
    state = _combat_state()
    action_result = {
        "action": "play_card",
        "status": "completed",
        "stable": True,
        "state": {"screen": "COMBAT", "turn": 1},
    }

    def respond_model(request: httpx.Request) -> httpx.Response:
        """校验 Runtime 交给模型的完整消息并返回规范动作。

        Args:
            request (httpx.Request): Provider 发出的模型请求。

        Raises:
            AssertionError: 模型请求缺少正确层级的提示词或可读观测。

        Returns:
            httpx.Response: 要求打出第 0 张牌并选择敌人 1 的回复。
        """
        body = json.loads(request.content)
        assert body["max_tokens"] == 128
        assert body["temperature"] == 0.0
        assert body["messages"][0]["role"] == "system"
        assert "战斗决策模型" in body["messages"][0]["content"]
        assert body["messages"][1]["role"] == "user"
        assert (
            "[0] 打击 | 1 能量 | 造成6点伤害。 | 目标: [1]"
            in (body["messages"][1]["content"])
        )
        assert body["messages"][1]["content"].endswith(
            "可执行动作:\n- play_card\n- end_turn"
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ACTION: play_card 0 1"}}]},
        )

    def respond_game(request: httpx.Request) -> httpx.Response:
        """校验解析后的动作参数并返回 Mod 动作结果。

        Args:
            request (httpx.Request): GameClient 发出的动作请求。

        Raises:
            AssertionError: Runtime 使用了错误的端点或动作参数。

        Returns:
            httpx.Response: 与真实 Mod 外壳一致的成功响应。
        """
        assert request.method == "POST"
        assert request.url.path == "/action"
        assert json.loads(request.content) == {
            "action": "play_card",
            "card_index": 0,
            "target_index": 1,
        }
        return httpx.Response(200, json={"ok": True, "data": action_result})

    runtime = importlib.import_module("play_sts2.runtime")
    with (
        GameClient(
            "http://127.0.0.1:8080",
            transport=httpx.MockTransport(respond_game),
        ) as game,
        OpenAICompatibleProvider(
            "http://127.0.0.1:8900",
            transport=httpx.MockTransport(respond_model),
        ) as provider,
    ):
        step = runtime.DecisionEngine(game, provider).step(state)

    assert tuple(message.role for message in step.messages) == ("system", "user")
    assert step.observation.available_actions == ("play_card", "end_turn")
    assert step.reply.text == "ACTION: play_card 0 1"
    assert step.action.name == "play_card"
    assert step.action.parameters == {"card_index": 0, "target_index": 1}
    assert step.action_result == action_result


@pytest.mark.parametrize(
    "reply_text",
    ["我建议结束回合。", "ACTION: save_and_quit"],
)
def test_decision_engine_does_not_execute_invalid_model_output(
    reply_text: str,
) -> None:
    """模型输出不符合 ACTION 契约时不向游戏发送任何动作。

    Args:
        reply_text (str): 非规范文本或当前决策层不可见的规范动作。

    Returns:
        None: 此测试只验证动作解析失败发生在游戏写入之前。
    """

    def respond_model(_request: httpx.Request) -> httpx.Response:
        """返回结构合法但不符合 Harness 动作契约的文本。

        Args:
            _request (httpx.Request): Provider 发出的模型请求。

        Returns:
            httpx.Response: 含解释性自然语言的模型回复。
        """
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": reply_text}}]},
        )

    def reject_game_request(_request: httpx.Request) -> httpx.Response:
        """在错误路径触发游戏写入时立即让测试失败。

        Args:
            _request (httpx.Request): 不应出现的游戏 HTTP 请求。

        Raises:
            AssertionError: Runtime 在动作解析失败后仍然写入了游戏。

        Returns:
            httpx.Response: 此函数始终在返回前失败。
        """
        raise AssertionError("非法模型输出不得执行游戏动作")

    runtime = importlib.import_module("play_sts2.runtime")
    with (
        GameClient(
            "http://127.0.0.1:8080",
            transport=httpx.MockTransport(reject_game_request),
        ) as game,
        OpenAICompatibleProvider(
            "http://127.0.0.1:8900",
            transport=httpx.MockTransport(respond_model),
        ) as provider,
        pytest.raises(ActionParseError),
    ):
        runtime.DecisionEngine(game, provider).step(_combat_state())


def _combat_state() -> dict[str, Any]:
    """构造与真实 Mod 字段一致的最小稳定战斗状态。

    Returns:
        dict[str, Any]: 可以打出第 0 张牌攻击敌人 1 的战斗状态。
    """
    return {
        "screen": "COMBAT",
        "in_combat": True,
        "turn": 1,
        "available_actions": ["save_and_quit", "play_card", "end_turn"],
        "run": {
            "character_name": "故障机器人",
            "ascension": 0,
            "act_id": "0",
            "floor": 2,
            "current_hp": 70,
            "max_hp": 75,
            "gold": 99,
            "potions": [],
        },
        "combat": {
            "player": {
                "current_hp": 70,
                "max_hp": 75,
                "block": 0,
                "energy": 3,
                "stars": 0,
                "focus": 0,
                "orb_capacity": 3,
                "orbs": [],
            },
            "enemies": [
                {
                    "index": 1,
                    "name": "邪教徒",
                    "current_hp": 48,
                    "max_hp": 48,
                    "block": 0,
                    "intents": [{"intent_type": "Buff", "label": None}],
                    "powers": [],
                }
            ],
            "hand": [
                {
                    "index": 0,
                    "name": "打击",
                    "upgraded": False,
                    "energy_cost": 1,
                    "star_cost": 0,
                    "resolved_rules_text": "造成6点伤害。",
                    "playable": True,
                    "valid_target_indices": [1],
                }
            ],
            "draw_count": 5,
            "discard_count": 0,
        },
    }
