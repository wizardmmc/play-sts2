"""为真实游戏 E2E 提供隔离 STS2 fixture。"""

import os
import sys
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path

import pytest

from play_sts2.game_launcher import DEFAULT_APP as _DEFAULT_APP
from play_sts2.game_launcher import DEFAULT_PROFILE as _DEFAULT_PROFILE
from play_sts2.game_launcher import RunningGame, launch_game

_DEFAULT_PORT = 18081


def pytest_addoption(parser: pytest.Parser) -> None:
    """注册真实游戏 E2E 的启用参数与启动模式。

    Args:
        parser (pytest.Parser): Pytest 命令行参数解析器。

    Returns:
        None: 此函数注册 E2E 开关与无头、有头模式选项。
    """
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="启动真实 STS2 并运行 E2E 测试",
    )
    parser.addoption(
        "--sts2-mode",
        choices=("headless", "headed"),
        default="headless",
        help="选择 STS2 启动模式，默认为 headless",
    )


@pytest.fixture
def running_game(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Iterator[RunningGame]:
    """为当前测试提供使用专用存档的独立游戏实例。

    Args:
        request (pytest.FixtureRequest): 当前 Pytest 测试请求。
        tmp_path (Path): 当前测试独占的临时目录。

    Raises:
        ValueError: ``STS2_E2E_PORT`` 不是有效整数或存档配置不安全。
        FileNotFoundError: 找不到游戏文件或专用测试存档。
        RuntimeError: 端口占用、Mod 隔离失败或游戏提前退出。
        TimeoutError: 游戏未能在限定时间内到达可操作主菜单。

    Yields:
        RunningGame: 游戏到达可操作主菜单后的实例信息。
    """
    if not request.config.getoption("--run-e2e"):
        pytest.skip("使用 --run-e2e 才会启动真实游戏")
    if sys.platform != "darwin":
        pytest.skip("当前 E2E 启动器仅支持 macOS")

    app_path = Path(os.environ.get("STS2_APP_PATH", _DEFAULT_APP))
    executable = app_path / "Contents/MacOS/Slay the Spire 2"
    port = int(os.environ.get("STS2_E2E_PORT", _DEFAULT_PORT))
    mode = request.config.getoption("--sts2-mode")
    isolated_home = tmp_path / "home"
    isolated_home.mkdir()

    with launch_game(
        executable,
        port=port,
        home=isolated_home,
        profile=_DEFAULT_PROFILE,
        mode=mode,
        enable_debug_actions=True,
    ) as game:
        yield game


@pytest.fixture
def running_games(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Iterator[tuple[RunningGame, RunningGame]]:
    """同时启动两个使用独立 HOME 与端口的真实游戏实例。

    Args:
        request (pytest.FixtureRequest): 当前 Pytest 测试请求。
        tmp_path (Path): 当前测试独占的临时目录。

    Raises:
        ValueError: ``STS2_E2E_PORT`` 不是有效整数或存档配置不安全。
        FileNotFoundError: 找不到游戏文件或专用测试存档。
        RuntimeError: 任一端口占用、Mod 隔离失败或游戏提前退出。
        TimeoutError: 任一游戏未能在限定时间内到达可操作主菜单。

    Yields:
        tuple[RunningGame, RunningGame]: 两个同时存活的隔离游戏实例。
    """
    if not request.config.getoption("--run-e2e"):
        pytest.skip("使用 --run-e2e 才会启动真实游戏")
    if sys.platform != "darwin":
        pytest.skip("当前 E2E 启动器仅支持 macOS")

    app_path = Path(os.environ.get("STS2_APP_PATH", _DEFAULT_APP))
    executable = app_path / "Contents/MacOS/Slay the Spire 2"
    base_port = int(os.environ.get("STS2_E2E_PORT", _DEFAULT_PORT))
    mode = request.config.getoption("--sts2-mode")
    games: list[RunningGame] = []
    with ExitStack() as stack:
        for index in range(2):
            isolated_home = tmp_path / f"home-{index}"
            isolated_home.mkdir()
            games.append(
                stack.enter_context(
                    launch_game(
                        executable,
                        port=base_port + index,
                        home=isolated_home,
                        profile=_DEFAULT_PROFILE,
                        mode=mode,
                        enable_debug_actions=True,
                    )
                )
            )
        yield games[0], games[1]
