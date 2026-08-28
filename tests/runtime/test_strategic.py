"""验证战略页面采用相互独立的单步模型决策。"""

import importlib
from collections.abc import Sequence
from typing import Any

import pytest

from play_sts2.inference import ChatMessage, ModelReply


class StrategicGame:
    """记录战略 Runner 执行的全部动作。"""

    def __init__(self) -> None:
        """初始化空动作记录。

        Returns:
            None: 此方法只初始化测试替身。
        """
        self.actions: list[tuple[str, dict[str, int]]] = []

    def execute_action(self, action: str, **parameters: int) -> dict[str, Any]:
        """记录动作并返回稳定完成结果。

        Args:
            action (str): Runtime 提交的战略动作名。
            parameters (int): Runtime 提交的动作索引参数。

        Returns:
            dict[str, Any]: 最小稳定动作结果。
        """
        self.actions.append((action, parameters))
        return {"status": "completed", "stable": True, "state": {}}


class StrategicProvider:
    """按顺序回复，并保留每次收到的消息。"""

    def __init__(self, replies: Sequence[str]) -> None:
        """保存预设回复和空请求记录。

        Args:
            replies (Sequence[str]): 每个战略页面依次返回的动作。

        Returns:
            None: 此方法只初始化测试替身。
        """
        self._replies = iter(replies)
        self.requests: list[tuple[ChatMessage, ...]] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> ModelReply:
        """记录本页完整消息并返回下一条动作。

        Args:
            messages (Sequence[ChatMessage]): 当前页面发送给模型的消息。
            max_tokens (int): 本次回复的最大 token 数。
            temperature (float): 本次回复的采样温度。

        Returns:
            ModelReply: 下一条预设模型回复。
        """
        self.requests.append(tuple(messages))
        return ModelReply(next(self._replies))


def test_strategic_runner_does_not_keep_history_between_screens() -> None:
    """连续战略决策都只包含 system 和当前页面观测。

    Raises:
        AssertionError: 战略动作错误，或前一页面消息泄漏到后一页面。

    Returns:
        None: 此测试验证战略上下文的生命周期。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    game = StrategicGame()
    provider = StrategicProvider(
        ["ACTION: choose_map_node 0", "ACTION: choose_event_option 1"]
    )
    runner = runtime.StrategicRunner(game, provider)

    map_step = runner.step(_map_state())
    event_step = runner.step(_event_state())

    assert map_step.action.name == "choose_map_node"
    assert event_step.action.name == "choose_event_option"
    assert game.actions == [
        (
            "choose_map_node",
            {"option_index": 0, "expected_state_revision": 1},
        ),
        (
            "choose_event_option",
            {"option_index": 1, "expected_state_revision": 2},
        ),
    ]
    assert [
        tuple(message.role for message in request) for request in provider.requests
    ] == [
        ("system", "user"),
        ("system", "user"),
    ]


def test_strategic_runner_rejects_battle_state() -> None:
    """战略 Runner 在请求模型前拒绝战斗状态。

    Raises:
        AssertionError: 战斗状态触发了模型请求或没有明确失败。

    Returns:
        None: 此测试只验证层级边界。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    provider = StrategicProvider(["ACTION: end_turn"])

    with pytest.raises(runtime.StrategicRunError, match="不属于战略决策"):
        runtime.StrategicRunner(StrategicGame(), provider).step(
            {
                "state_revision": 1,
                "screen": "COMBAT",
                "in_combat": True,
                "available_actions": ["end_turn"],
            }
        )

    assert provider.requests == []


def _map_state() -> dict[str, Any]:
    """构造可选择一个节点的地图状态。

    Returns:
        dict[str, Any]: 可由战略 Harness 渲染的地图状态。
    """
    return {
        "state_revision": 1,
        "screen": "MAP",
        "available_actions": ["choose_map_node"],
        "run": {"character_name": "故障机器人", "current_hp": 70, "max_hp": 75},
        "map": {"available_nodes": [{"index": 0, "row": 1, "col": 2}]},
    }


def _event_state() -> dict[str, Any]:
    """构造可选择第二个选项的事件状态。

    Returns:
        dict[str, Any]: 可由战略 Harness 渲染的事件状态。
    """
    return {
        "state_revision": 2,
        "screen": "EVENT",
        "available_actions": ["choose_event_option"],
        "run": {"character_name": "故障机器人", "current_hp": 70, "max_hp": 75},
        "event": {
            "title": "测试事件",
            "options": [
                {"index": 0, "title": "离开"},
                {"index": 1, "title": "接受"},
            ],
        },
    }
