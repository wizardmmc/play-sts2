"""验证 Qwen 战斗 CLI 的用户可观察行为。"""

import importlib
import json
from collections.abc import Sequence
from pathlib import Path
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

    def execute_action(self, action: str, **parameters: int) -> dict[str, Any]:
        """执行唯一动作并返回战斗奖励状态。

        Args:
            action (str): Runtime 提交的动作名称。
            parameters (int): Runtime 提交的状态版本参数。

        Raises:
            AssertionError: 模型没有选择结束回合。

        Returns:
            dict[str, Any]: 含稳定奖励状态的动作结果。
        """
        assert action == "end_turn"
        assert parameters == {"expected_state_revision": 1}
        return {
            "status": "completed",
            "stable": True,
            "state": {
                "state_revision": 2,
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

    def __init__(
        self,
        base_url: str,
        *,
        model: str | None = None,
        enable_thinking: bool | None = None,
    ) -> None:
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
        assert model is not None
        assert Path(model).name == "demo-e3-mlx-8bit"
        assert enable_thinking is True

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
        assert max_tokens == 512
        assert temperature == 0.2
        return ModelReply("ACTION: end_turn")


def test_main_connects_qwen_and_finishes_current_battle(
    tmp_path: Path,
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
    preflight_calls: list[tuple[str, str, Path, Path]] = []

    def validate(
        base_url: str,
        *,
        artifact_id: str,
        merged_model: Path,
        serving_model: Path,
        enable_thinking: bool,
    ) -> None:
        assert enable_thinking is True
        preflight_calls.append((base_url, artifact_id, merged_model, serving_model))

    def game(base_url: str) -> CliGameClient:
        assert preflight_calls, "模型身份预检必须发生在连接游戏之前"
        return CliGameClient(base_url)

    monkeypatch.setattr(cli, "validate_model_service", validate, raising=False)
    monkeypatch.setattr(cli, "GameClient", game)
    monkeypatch.setattr(cli, "OpenAICompatibleProvider", CliProvider)
    config_path = _write_inference_config(tmp_path)

    exit_code = cli.main(
        [
            "--game-url",
            "http://127.0.0.1:8082",
            "--inference-config",
            str(config_path),
            "--profile",
            "think",
        ]
    )

    assert exit_code == 0
    assert preflight_calls == [
        (
            "http://127.0.0.1:8901",
            "demo-e3",
            tmp_path / "demo-e3-merged",
            tmp_path / "demo-e3-mlx-8bit",
        )
    ]
    assert capsys.readouterr().out == "战斗结束: cleared，执行 1 个动作\n"


def test_battle_cli_rejects_request_model_override() -> None:
    """战斗入口不允许请求级 model 绕过推理配置。"""
    cli = importlib.import_module("play_sts2.runtime.cli")

    with pytest.raises(SystemExit):
        cli._parser().parse_args(["--model", "wrong-model"])


def _write_inference_config(tmp_path: Path) -> Path:
    """写入运行时 CLI 测试使用的双模式配置。"""
    artifact_id = "demo-e3"
    config_path = tmp_path / "inference.toml"
    config_path.write_text(
        f"""
artifact_id = "{artifact_id}"
merged_model = {json.dumps(str(tmp_path / f"{artifact_id}-merged"))}
serving_model = {json.dumps(str(tmp_path / f"{artifact_id}-mlx-8bit"))}
default_profile = "no-think"

[server]
base_url = "http://127.0.0.1:8901"
port = 8901
prompt_cache_size = 10

[profiles.no-think]
enable_thinking = false
max_tokens = 128
temperature = 0.0

[profiles.think]
enable_thinking = true
max_tokens = 512
temperature = 0.2
""".strip(),
        encoding="utf-8",
    )
    return config_path


def _combat_state() -> dict[str, Any]:
    """构造命令行测试使用的最小稳定战斗状态。

    Returns:
        dict[str, Any]: 可以结束回合的战斗状态。
    """
    return {
        "state_revision": 1,
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
