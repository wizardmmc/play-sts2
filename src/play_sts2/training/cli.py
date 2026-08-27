"""提供可读 SFT 数据集构建命令。"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .dataset import build_sft_dataset


def build_parser() -> argparse.ArgumentParser:
    """构建后训练命令行参数解析器。

    Returns:
        argparse.ArgumentParser: 当前包含 ``build-sft`` 子命令的解析器。
    """
    parser = argparse.ArgumentParser(prog="play-sts2-train")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-sft", help="构建可读 SFT messages")
    build.add_argument(
        "--knowledge-root",
        type=Path,
        default=Path("data/game_knowledge/web_wiki"),
    )
    build.add_argument(
        "--transcripts-root",
        type=Path,
        default=Path("data/transcripts"),
    )
    build.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/datasets/sft/baseline-v1"),
    )
    build.add_argument("--dev-run", action="append", default=[])
    build.add_argument("--test-run", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行可读 SFT 数据集构建。

    Args:
        argv (Sequence[str] | None): 可选命令行参数，省略时读取进程参数。

    Returns:
        int: 数据集成功写入时返回 ``0``。
    """
    args = build_parser().parse_args(argv)
    result = build_sft_dataset(
        knowledge_root=args.knowledge_root,
        transcripts_root=args.transcripts_root,
        output_root=args.output_root,
        dev_run_ids=args.dev_run,
        test_run_ids=args.test_run,
    )
    print(
        json.dumps(
            {
                "output_root": str(result.output_root),
                "train": result.train_count,
                "dev": result.dev_count,
                "test": result.test_count,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
