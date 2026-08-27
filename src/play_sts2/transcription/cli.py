"""提供从命令行转录一局人类轨迹的入口。"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from .transcriber import transcribe_run

_DEFAULT_OUTPUT_ROOT = Path("data/transcripts")


def main(argv: Sequence[str] | None = None) -> int:
    """把命令行指定的一局原始轨迹转换为精确人类决策。

    Args:
        argv (Sequence[str] | None): 不含程序名的命令行参数；为 ``None`` 时
            使用当前进程参数。

    Returns:
        int: 命令成功时返回 ``0``。
    """
    args = _parser().parse_args(argv)
    result = transcribe_run(args.run_dir, args.output_root)
    print(f"转录完成: {result.output_path.resolve()}")
    print(f"人类决策: {result.decision_count}")
    return 0


def _parser() -> argparse.ArgumentParser:
    """创建精确人类决策转录命令的参数解析器。

    Returns:
        argparse.ArgumentParser: 包含输入局目录和输出目录的解析器。
    """
    parser = argparse.ArgumentParser(
        description="从 recorder 原始事件中提取精确的人类 UI 决策。",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="包含 meta.json 和 events.jsonl 的原始局目录",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
        help=f"精确决策输出目录，默认为 {_DEFAULT_OUTPUT_ROOT}",
    )
    return parser
