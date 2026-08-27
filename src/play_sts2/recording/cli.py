"""提供从命令行启动人类轨迹录制的入口。"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from play_sts2.client import GameClient

from .recorder import HumanRunRecorder

_DEFAULT_BASE_URL = "http://127.0.0.1:8080"
_DEFAULT_OUTPUT_ROOT = Path("data/raw")
_DEFAULT_POLL_INTERVAL = 0.1


def main(argv: Sequence[str] | None = None) -> int:
    """连接已运行的 STS2，并把遇到的第一局保存为人类轨迹。

    Args:
        argv (Sequence[str] | None): 不含程序名的命令行参数；为 ``None`` 时
            使用当前进程参数。

    Returns:
        int: 命令成功时返回 ``0``。
    """
    args = _parser().parse_args(argv)
    with GameClient(args.base_url) as client:
        result = HumanRunRecorder(
            client,
            args.output_root,
            poll_interval=args.poll_interval,
        ).record()

    if result is not None:
        print(f"录制完成: {result.run_dir.resolve()}")
    return 0


def _parser() -> argparse.ArgumentParser:
    """创建人类轨迹录制命令的参数解析器。

    Returns:
        argparse.ArgumentParser: 包含连接、目录和轮询参数的解析器。
    """
    parser = argparse.ArgumentParser(
        description="连接已运行的 STS2，并录制遇到的第一局人类游玩轨迹。",
    )
    parser.add_argument(
        "--base-url",
        default=_DEFAULT_BASE_URL,
        help=f"Agent Mod 服务地址，默认为 {_DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
        help=f"原始数据根目录，默认为 {_DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=_DEFAULT_POLL_INTERVAL,
        help=f"状态轮询间隔秒数，默认为 {_DEFAULT_POLL_INTERVAL}",
    )
    return parser
