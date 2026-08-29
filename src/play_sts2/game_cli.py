"""提供启动隔离 STS2 进程的正式命令行入口。"""

import argparse
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .game_launcher import DEFAULT_APP, DEFAULT_PORT, DEFAULT_PROFILE, launch_game


def main(argv: Sequence[str] | None = None) -> int:
    """启动隔离游戏并保持前台运行直到游戏退出。

    Args:
        argv (Sequence[str] | None): 不含程序名的命令行参数。

    Returns:
        int: 游戏退出状态码；用户中断时返回 ``130``。
    """
    args = _parser().parse_args(argv)
    executable = _resolve_game_executable(args.app_path)
    with tempfile.TemporaryDirectory(prefix="play-sts2-game-") as temporary:
        home = Path(temporary)
        with launch_game(
            executable,
            port=args.port,
            home=home,
            profile=args.profile,
            mode=args.mode,
            enable_debug_actions=args.enable_debug_actions,
        ) as game:
            print(f"游戏已就绪: {game.base_url}", flush=True)
            print(f"隔离 HOME: {game.home}", flush=True)
            print(f"游戏日志: {game.log_path}", flush=True)
            try:
                return game.wait()
            except KeyboardInterrupt:
                return 130


def _resolve_game_executable(app_path: Path) -> Path:
    """在切换子进程工作目录前固定游戏可执行文件的绝对路径。

    Args:
        app_path (Path): 用户传入的 ``SlayTheSpire2.app`` 路径。

    Returns:
        Path: 不再依赖后续工作目录的 macOS 可执行文件绝对路径。
    """
    return (app_path / "Contents/MacOS/Slay the Spire 2").resolve()


def _parser() -> argparse.ArgumentParser:
    """构造游戏启动命令的参数解析器。

    Returns:
        argparse.ArgumentParser: 包含模式、端口、程序与存档模板参数的解析器。
    """
    default_app = Path(os.environ.get("STS2_APP_PATH", DEFAULT_APP))
    parser = argparse.ArgumentParser(
        description="使用隔离存档启动加载 Agent Mod 的 STS2。",
    )
    parser.add_argument(
        "--mode",
        choices=("headless", "headed"),
        default="headless",
        help="游戏显示模式，默认为 headless",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Agent Mod HTTP 端口，默认为 {DEFAULT_PORT}",
    )
    parser.add_argument(
        "--app-path",
        type=Path,
        default=default_app,
        help="SlayTheSpire2.app 路径；默认读取 STS2_APP_PATH 或项目固定副本",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help=f"隔离存档模板目录，默认为 {DEFAULT_PROFILE}",
    )
    parser.add_argument(
        "--enable-debug-actions",
        action="store_true",
        help="显式开放 scenario 重置动作，仅用于合成 RL 采样",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
