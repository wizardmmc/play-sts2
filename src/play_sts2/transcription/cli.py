"""提供从命令行重新生成一局 transcript 的入口。"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from .transcriber import render_run

_DEFAULT_OUTPUT_ROOT = Path("data/transcripts")


def main(argv: Sequence[str] | None = None) -> int:
    """把命令行指定的一局 raw 转换为可读 transcript。

    Args:
        argv (Sequence[str] | None): 不含程序名的命令行参数；为 ``None`` 时
            使用当前进程参数。

    Returns:
        int: 命令成功时返回 ``0``。
    """
    args = _parser().parse_args(argv)
    result = render_run(args.run_dir, args.output_root)
    print(f"Transcript 完成: {result.output_dir}")
    print(f"战斗决策: {result.battle_decision_count}")
    print(f"战略决策: {result.strategic_decision_count}")
    return 0


def _parser() -> argparse.ArgumentParser:
    """创建 raw 到 transcript 的命令参数解析器。

    Returns:
        argparse.ArgumentParser: 包含输入局目录和输出目录的解析器。
    """
    parser = argparse.ArgumentParser(
        description="从当前 raw 重新生成按录制来源分组的可读 transcript。",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="包含 meta.json、combat 和 strategy 的 raw 局目录",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
        help=f"Transcript 输出根目录，默认为 {_DEFAULT_OUTPUT_ROOT}",
    )
    return parser
