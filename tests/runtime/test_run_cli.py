"""验证完整一局模型 CLI 的开局与续局入口。"""

import importlib
from collections.abc import Sequence
from types import TracebackType
from typing import Any, Self

import pytest

from play_sts2.inference import ChatMessage, ModelReply


class CliRunProvider:
    """为命令行测试返回一次地图选择动作。"""

    def __init__(self, base_url: str, *, model: str | None = None) -> None:
        """校验 CLI 传入的模型连接参数。

        Args:
            base_url (str): CLI 使用的推理服务地址。
            model (str | None): CLI 传给服务端的模型名。

        Raises:
            AssertionError: CLI 传入了错误的连接参数。

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
        """确认战略提示词并选择唯一地图节点。

        Args:
            messages (Sequence[ChatMessage]): Runtime 发送的完整消息。
            max_tokens (int): 本次生成允许使用的最大 token 数。
            temperature (float): 本次生成使用的采样温度。

        Raises:
            AssertionError: Runtime 没有使用战略系统提示词。

        Returns:
            ModelReply: 合法的地图选择动作。
        """
        assert "战略决策模型" in messages[0].content
        return ModelReply("ACTION: choose_map_node 0")


class NewRunCliGame:
    """模拟从主菜单开局并在一次地图选择后通关。"""

    latest: "NewRunCliGame | None" = None

    def __init__(self, base_url: str) -> None:
        """校验游戏地址并初始化主菜单状态。

        Args:
            base_url (str): CLI 使用的 Agent Mod 地址。

        Raises:
            AssertionError: CLI 传入了错误的游戏地址。

        Returns:
            None: 此方法初始化测试游戏并保存最近实例。
        """
        assert base_url == "http://127.0.0.1:8082"
        self.actions: list[dict[str, Any]] = []
        self._state = {
            "screen": "MAIN_MENU",
            "available_actions": ["open_character_select"],
        }
        type(self).latest = self

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

    @property
    def action_timeout(self) -> float:
        """返回开局流程共享的动作超时。

        Returns:
            float: 测试使用的一秒超时。
        """
        return 1.0

    def state(self) -> dict[str, Any]:
        """返回当前测试游戏状态。

        Returns:
            dict[str, Any]: 当前状态的副本。
        """
        return dict(self._state)

    def execute_action(self, action: str, **parameters: Any) -> dict[str, Any]:
        """执行开局动作或唯一地图选择动作。

        Args:
            action (str): Runtime 提交的动作名称。
            parameters (Any): 当前动作的协议参数。

        Raises:
            AssertionError: Runtime 执行了预期之外的动作。

        Returns:
            dict[str, Any]: 含下一稳定状态的动作结果。
        """
        self.actions.append({"action": action, **parameters})
        if action == "open_character_select":
            self._state = _character_state(selected="IRONCLAD")
        elif action == "select_character":
            assert parameters == {"option_index": 4}
            self._state = _character_state(selected="DEFECT")
        elif action == "set_seed":
            assert parameters == {"game_seed": "TEST-SEED"}
        elif action == "embark":
            self._state = _map_state()
        elif action == "choose_map_node":
            assert parameters == {"option_index": 0}
            self._state = _game_over_state()
        else:
            raise AssertionError(f"预期外动作: {action}")
        return {"status": "completed", "stable": True, "state": self._state}


class ResumeCliGame(NewRunCliGame):
    """模拟主菜单存在保存局的续局游戏。"""

    latest: "ResumeCliGame | None" = None

    def __init__(self, base_url: str) -> None:
        """初始化无需执行开局动作的地图状态。

        Args:
            base_url (str): CLI 使用的 Agent Mod 地址。

        Returns:
            None: 此方法保存地图状态和最近实例。
        """
        super().__init__(base_url)
        self._state = {
            "screen": "MAIN_MENU",
            "available_actions": ["continue_run"],
        }
        type(self).latest = self

    def execute_action(self, action: str, **parameters: Any) -> dict[str, Any]:
        """先恢复保存局，再沿用父类执行地图选择。

        Args:
            action (str): Runtime 提交的动作名称。
            parameters (Any): 当前动作的协议参数。

        Returns:
            dict[str, Any]: 含恢复后地图或终局状态的动作结果。
        """
        if action == "continue_run":
            self.actions.append({"action": action, **parameters})
            self._state = _map_state()
            return {"status": "completed", "stable": True, "state": self._state}
        return super().execute_action(action, **parameters)


def test_main_starts_new_run_and_plays_to_victory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """默认命令从主菜单按参数开局并运行到终局。

    Args:
        monkeypatch (pytest.MonkeyPatch): 用于替换真实网络客户端。
        capsys (pytest.CaptureFixture[str]): 用于读取标准输出。

    Raises:
        AssertionError: 开局参数、动作顺序或结果输出不正确。

    Returns:
        None: 此测试验证新局命令的用户行为。
    """
    cli = importlib.import_module("play_sts2.runtime.run_cli")
    monkeypatch.setattr(cli, "GameClient", NewRunCliGame)
    monkeypatch.setattr(cli, "OpenAICompatibleProvider", CliRunProvider)

    exit_code = cli.main(
        [
            "--game-url",
            "http://127.0.0.1:8082",
            "--model-url",
            "http://127.0.0.1:8901",
            "--model",
            "Qwen/Qwen3.5-4B",
            "--character",
            "DEFECT",
            "--seed",
            "TEST-SEED",
        ]
    )

    assert exit_code == 0
    assert NewRunCliGame.latest is not None
    assert NewRunCliGame.latest.actions == [
        {"action": "open_character_select"},
        {"action": "select_character", "option_index": 4},
        {"action": "set_seed", "game_seed": "TEST-SEED"},
        {"action": "embark"},
        {"action": "choose_map_node", "option_index": 0},
    ]
    assert capsys.readouterr().out == (
        "整局结束: victory，经历 0 场战斗，执行 1 个动作\n"
    )


def test_main_resumes_current_run_without_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--resume`` 直接使用当前局状态，不触发主菜单开局动作。

    Args:
        monkeypatch (pytest.MonkeyPatch): 用于替换真实网络客户端。

    Raises:
        AssertionError: 续局仍执行了角色选择或开局动作。

    Returns:
        None: 此测试只验证续局分支。
    """
    cli = importlib.import_module("play_sts2.runtime.run_cli")
    monkeypatch.setattr(cli, "GameClient", ResumeCliGame)
    monkeypatch.setattr(cli, "OpenAICompatibleProvider", CliRunProvider)

    exit_code = cli.main(
        [
            "--game-url",
            "http://127.0.0.1:8082",
            "--model-url",
            "http://127.0.0.1:8901",
            "--model",
            "Qwen/Qwen3.5-4B",
            "--resume",
        ]
    )

    assert exit_code == 0
    assert ResumeCliGame.latest is not None
    assert ResumeCliGame.latest.actions == [
        {"action": "continue_run"},
        {"action": "choose_map_node", "option_index": 0},
    ]


def _character_state(*, selected: str) -> dict[str, Any]:
    """构造角色选择页面。

    Args:
        selected (str): 当前选中的角色稳定 ID。

    Returns:
        dict[str, Any]: 与开局流程兼容的角色选择状态。
    """
    return {
        "screen": "CHARACTER_SELECT",
        "available_actions": ["select_character", "set_seed", "embark"],
        "character_select": {
            "selected_character_id": selected,
            "ascension": 0,
            "max_ascension": 20,
            "characters": [{"index": 4, "character_id": "DEFECT", "is_locked": False}],
        },
    }


def _map_state() -> dict[str, Any]:
    """构造开局后的唯一节点地图。

    Returns:
        dict[str, Any]: 可供战略模型选择的地图状态。
    """
    return {
        "screen": "MAP",
        "available_actions": ["choose_map_node"],
        "run": {
            "character_id": "DEFECT",
            "character_name": "故障机器人",
            "ascension": 0,
            "current_hp": 70,
            "max_hp": 75,
            "deck": [],
            "relics": [],
            "potions": [],
        },
        "map": {
            "available_nodes": [
                {"index": 0, "row": 1, "col": 1, "node_type": "Monster"}
            ]
        },
    }


def _game_over_state() -> dict[str, Any]:
    """构造通关终局。

    Returns:
        dict[str, Any]: Mod 原始胜利状态。
    """
    return {
        "screen": "GAME_OVER",
        "available_actions": ["return_to_main_menu"],
        "game_over": {"is_victory": True},
    }
