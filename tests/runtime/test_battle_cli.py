"""验证 Qwen 战斗 CLI 的用户可观察行为。"""

import importlib
from collections.abc import Sequence
from types import TracebackType
from typing import Any, Self

import pytest

from play_sts2.inference import ChatMessage, ModelReply


class CliGameClient:
    """提供一动作即可获胜的命令行测试游戏。

    Args:
        base_url (str): CLI 使用的 Agent Mod 地址。
    """

    def __init__(self, base_url: str) -> None:
        """校验 CLI 传入的游戏地址。

        Args:
            base_url (str): CLI 使用的 Agent Mod 地址。

        Raises:
            AssertionError: CLI 没有传入测试指定地址。

        Returns:
            None: 此方法只初始化测试游戏。
        """
        assert base_url == "http://127.0.0.1:8082"

    def __enter__(self) -> Self:
        """进入测试游戏上下文。

        Returns:
            Self: 当前测试游戏。
        """
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """离开无需释放资源的测试游戏上下文。

        Args:
            _exc_type (type[BaseException] | None): 上下文异常类型。
            _exc (BaseException | None): 上下文异常实例。
            _traceback (TracebackType | None): 上下文异常调用栈。

        Returns:
            None: 此测试替身没有外部资源。
        """

    def state(self) -> dict[str, Any]:
        """返回当前稳定战斗状态。

        Returns:
            dict[str, Any]: 可以结束回合的第一回合状态。
        """
        return _combat_state()

    def execute_action(self, action: str, **_parameters: int) -> dict[str, Any]:
        """执行唯一动作并返回战斗奖励状态。

        Args:
            action (str): Runtime 提交的动作名称。
            _parameters (int): 当前动作未使用的参数。

        Raises:
            AssertionError: 模型没有选择结束回合。

        Returns:
            dict[str, Any]: 含稳定奖励状态的动作结果。
        """
        assert action == "end_turn"
        return {
            "status": "completed",
            "stable": True,
            "state": {
                "screen": "REWARD",
                "in_combat": False,
                "available_actions": ["claim_reward"],
                "run": {"current_hp": 65, "max_hp": 75},
            },
        }


class CliProvider:
    """模拟 OpenAI-compatible Qwen 服务。

    Args:
        base_url (str): CLI 使用的推理服务地址。
        model (str | None): CLI 传给服务端的模型名。
    """

    def __init__(self, base_url: str, *, model: str | None = None) -> None:
        """校验 CLI 传入的模型连接参数。

        Args:
            base_url (str): CLI 使用的推理服务地址。
            model (str | None): CLI 传给服务端的模型名。

        Raises:
            AssertionError: CLI 没有传入测试指定的服务或模型。

        Returns:
            None: 此方法只初始化测试 Provider。
        """
        assert base_url == "http://127.0.0.1:8901"
        assert model == "Qwen/Qwen3.5-4B"

    def __enter__(self) -> Self:
        """进入测试 Provider 上下文。

        Returns:
            Self: 当前测试 Provider。
        """
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """离开无需释放资源的测试 Provider 上下文。

        Args:
            _exc_type (type[BaseException] | None): 上下文异常类型。
            _exc (BaseException | None): 上下文异常实例。
            _traceback (TracebackType | None): 上下文异常调用栈。

        Returns:
            None: 此测试替身没有外部资源。
        """

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> ModelReply:
        """校验收到战斗提示词并返回结束回合动作。

        Args:
            messages (Sequence[ChatMessage]): Runtime 发送的完整消息。
            max_tokens (int): 本次生成允许使用的最大输出 token 数。
            temperature (float): 本次生成使用的采样温度。

        Raises:
            AssertionError: Runtime 没有发送战斗 system 消息。

        Returns:
            ModelReply: 合法的结束回合动作。
        """
        assert messages[0].role == "system"
        assert "战斗决策模型" in messages[0].content
        return ModelReply("ACTION: end_turn")


def test_main_connects_qwen_and_finishes_current_battle(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """命令行连接指定游戏与 Qwen 服务并打印战斗结果。

    Args:
        monkeypatch (pytest.MonkeyPatch): 用于替换真实网络客户端。
        capsys (pytest.CaptureFixture[str]): 用于读取命令标准输出。

    Raises:
        AssertionError: CLI 参数、闭环动作或完成输出不符合约定。

    Returns:
        None: 此测试只验证命令入口。
    """
    cli = importlib.import_module("play_sts2.runtime.cli")
    monkeypatch.setattr(cli, "GameClient", CliGameClient)
    monkeypatch.setattr(cli, "OpenAICompatibleProvider", CliProvider)

    exit_code = cli.main(
        [
            "--game-url",
            "http://127.0.0.1:8082",
            "--model-url",
            "http://127.0.0.1:8901",
            "--model",
            "Qwen/Qwen3.5-4B",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == "战斗结束: cleared，执行 1 个动作\n"


def _combat_state() -> dict[str, Any]:
    """构造命令行测试使用的最小稳定战斗状态。

    Returns:
        dict[str, Any]: 可以结束回合的战斗状态。
    """
    return {
        "screen": "COMBAT",
        "in_combat": True,
        "turn": 1,
        "available_actions": ["end_turn"],
        "run": {
            "character_name": "故障机器人",
            "current_hp": 70,
            "max_hp": 75,
            "potions": [],
        },
        "combat": {
            "player": {
                "current_hp": 70,
                "max_hp": 75,
                "block": 0,
                "energy": 3,
                "stars": 0,
                "orbs": [],
            },
            "enemies": [],
            "hand": [],
        },
    }
