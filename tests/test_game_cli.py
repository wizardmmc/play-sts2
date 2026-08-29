"""验证正式 STS2 启动命令的参数与进程生命周期。"""

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest


class WaitingGame:
    """记录命令行是否持续等待已经就绪的游戏进程。"""

    def __init__(self, home: Path) -> None:
        """保存隔离目录并初始化等待标记。

        Args:
            home (Path): CLI 为当前游戏创建的隔离 HOME。

        Returns:
            None: 此方法只初始化测试替身。
        """
        self.base_url = "http://127.0.0.1:8082"
        self.home = home
        self.log_path = home / "headed.log"
        self.waited = False

    def wait(self) -> int:
        """记录 CLI 已经等待游戏进程退出。

        Returns:
            int: 模拟游戏正常退出的状态码。
        """
        self.waited = True
        return 0


def test_default_profile_does_not_depend_on_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """默认存档模板从项目源码位置解析，不依赖当前目录。

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest 提供的工作目录替换工具。
        tmp_path (Path): 用作任意启动目录的临时路径。

    Raises:
        AssertionError: 默认模板仍是相对路径或缺少必需存档。

    Returns:
        None: 此测试只验证默认模板的定位边界。
    """
    cli = importlib.import_module("play_sts2.game_cli")
    monkeypatch.chdir(tmp_path)

    profile = cli._parser().parse_args([]).profile

    assert profile.is_absolute()
    assert (profile / "settings.save").is_file()
    assert (profile / "prefs.save").is_file()
    assert (profile / "progress.save").is_file()


def test_parser_allows_explicit_rl_debug_actions() -> None:
    """RL 场景 worker 必须由用户显式开放受保护的调试动作。

    Raises:
        AssertionError: 正式游戏 CLI 无法为 scenario collector 开启调试入口。

    Returns:
        None: 此测试只验证显式开关，默认值由启动测试继续覆盖。
    """
    cli = importlib.import_module("play_sts2.game_cli")

    args = cli._parser().parse_args(["--enable-debug-actions"])

    assert args.enable_debug_actions is True


def test_resolve_game_executable_makes_relative_app_absolute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """相对应用路径必须在游戏切换到隔离 HOME 前固定为绝对路径。

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest 提供的工作目录替换工具。
        tmp_path (Path): 模拟用户执行命令的项目目录。

    Raises:
        AssertionError: 可执行文件仍依赖子进程的工作目录。

    Returns:
        None: 此测试只验证路径解析时点。
    """
    cli = importlib.import_module("play_sts2.game_cli")
    monkeypatch.chdir(tmp_path)

    executable = cli._resolve_game_executable(Path(".runtime/teacher.app"))

    assert executable == (
        tmp_path / ".runtime/teacher.app/Contents/MacOS/Slay the Spire 2"
    )
    assert executable.is_absolute()


def test_main_launches_isolated_game_and_waits_for_exit(
    monkeypatch: Any,
    tmp_path: Path,
    capsys: Any,
) -> None:
    """CLI 把用户参数交给启动器并保持进程到游戏退出。

    Args:
        monkeypatch (Any): Pytest 提供的属性替换工具。
        tmp_path (Path): Pytest 提供的单测临时目录。
        capsys (Any): Pytest 提供的标准输出捕获工具。

    Raises:
        AssertionError: CLI 丢失参数、未等待游戏或未清理临时 HOME。

    Returns:
        None: 此测试只验证命令行编排行为。
    """
    cli = importlib.import_module("play_sts2.game_cli")
    captured: dict[str, Any] = {}

    @contextmanager
    def fake_launch_game(
        executable: Path,
        *,
        port: int,
        home: Path,
        profile: Path,
        mode: str,
        enable_debug_actions: bool = False,
    ) -> Iterator[WaitingGame]:
        """记录 CLI 传入的启动配置并返回可等待的游戏替身。

        Args:
            executable (Path): CLI 解析出的游戏可执行文件。
            port (int): Mod HTTP 服务端口。
            home (Path): 当前游戏的临时隔离 HOME。
            profile (Path): 用户选择的隔离存档模板。
            mode (str): 用户选择的显示模式。
            enable_debug_actions (bool): 是否开放调试动作。

        Yields:
            WaitingGame: 可供 CLI 阻塞等待的游戏替身。
        """
        captured.update(
            executable=executable,
            port=port,
            home=home,
            profile=profile,
            mode=mode,
            enable_debug_actions=enable_debug_actions,
        )
        game = WaitingGame(home)
        captured["game"] = game
        yield game

    monkeypatch.setattr(cli, "launch_game", fake_launch_game)
    app_path = tmp_path / "SlayTheSpire2.app"
    profile = tmp_path / "profile"

    result = cli.main(
        [
            "--mode",
            "headed",
            "--port",
            "8082",
            "--app-path",
            str(app_path),
            "--profile",
            str(profile),
        ]
    )

    home = captured["home"]
    assert result == 0
    assert captured == {
        "executable": app_path / "Contents/MacOS/Slay the Spire 2",
        "port": 8082,
        "home": home,
        "profile": profile,
        "mode": "headed",
        "enable_debug_actions": False,
        "game": captured["game"],
    }
    assert captured["game"].waited is True
    assert not home.exists()
    assert "游戏已就绪: http://127.0.0.1:8082" in capsys.readouterr().out
