"""提供游戏知识迁移与实测采样命令。"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ..client import GameClient
from .pipeline import export_mod_knowledge, import_web_wiki


def build_parser() -> argparse.ArgumentParser:
    """构建知识命令行参数解析器。

    Returns:
        argparse.ArgumentParser: 包含 ``import-wiki`` 与 ``export`` 子命令。
    """
    parser = argparse.ArgumentParser(prog="play-sts2-knowledge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    importer = subparsers.add_parser("import-wiki", help="导入单实体 Markdown Wiki")
    importer.add_argument("source", type=Path, help="Wiki 根目录")
    importer.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/game_knowledge"),
        help="游戏知识根目录",
    )

    exporter = subparsers.add_parser("export", help="从运行中的 Mod 实测导出")
    exporter.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
        help="Agent Mod 服务地址",
    )
    exporter.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/game_knowledge"),
        help="游戏知识根目录",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行知识导入或 Mod 实测导出。

    Args:
        argv (Sequence[str] | None): 可选的命令行参数，省略时读取进程参数。

    Returns:
        int: 成功时返回 ``0``。
    """
    args = build_parser().parse_args(argv)
    if args.command == "import-wiki":
        result = import_web_wiki(args.source, args.output_root)
    else:
        with GameClient(args.base_url) as client:
            result = export_mod_knowledge(client, args.output_root)
    print(
        json.dumps(
            {
                "output_root": str(result.output_root),
                "entries": result.entry_count,
                "categories": result.categories,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
