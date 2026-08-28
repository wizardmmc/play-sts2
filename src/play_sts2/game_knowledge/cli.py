"""提供游戏知识迁移与实测采样命令。"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ..client import GameClient
from .arithmetic import generate_arithmetic_candidates
from .generation import generate_question_variants, generate_review_report
from .pipeline import export_mod_knowledge, import_web_wiki, rebuild_mod_knowledge


def build_parser() -> argparse.ArgumentParser:
    """构建知识命令行参数解析器。

    Returns:
        argparse.ArgumentParser: 游戏知识在线采样与离线重建命令。
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

    rebuilder = subparsers.add_parser(
        "rebuild",
        help="从 v0.107.1 原始快照离线重建规范事实 Markdown",
    )
    rebuilder.add_argument("raw_root", type=Path, help="固定版本 raw 目录")
    rebuilder.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/game_knowledge"),
        help="游戏知识根目录",
    )
    rebuilder.add_argument(
        "--wiki-root",
        type=Path,
        help="仅用于补充怪物招式与循环的 Web Wiki 根目录",
    )
    rebuilder.add_argument(
        "--cycles-root",
        type=Path,
        help="human-rl 的实跳怪物循环记录目录",
    )
    rebuilder.add_argument(
        "--event-entries-root",
        type=Path,
        help="human-rl 已进入事件界面后保存的已解析 UI 快照目录",
    )
    generator = subparsers.add_parser(
        "generate-questions",
        help="生成按实体保存的多问法知识，不划分 E3 数据集",
    )
    generator.add_argument("snapshot_root", type=Path, help="固定版本知识快照")
    generator.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/game_knowledge/generated-v0.107.1"),
        help="多问法 JSONL 输出目录",
    )

    arithmetic = subparsers.add_parser(
        "generate-arithmetic",
        help="从当前项目实战意图生成互斥的训练与验证算术候选",
    )
    arithmetic.add_argument(
        "--human-root",
        type=Path,
        default=Path("data/raw/human"),
        help="当前项目的人类精确战斗目录",
    )
    arithmetic.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/game_knowledge/generated-v0.107.1"),
        help="知识与算术候选输出目录",
    )
    arithmetic.add_argument(
        "--probes-root",
        type=Path,
        default=Path("data/datasets/sft/eval/knowledge"),
        help="生成时必须排除的最终知识考试目录",
    )

    reviewer = subparsers.add_parser(
        "review",
        help="生成人工可读的缺口与特殊对象审计报告",
    )
    reviewer.add_argument("snapshot_root", type=Path, help="固定版本知识快照")
    reviewer.add_argument(
        "--output",
        type=Path,
        default=Path("data/game_knowledge/reports/v0.107.1.md"),
        help="Markdown 报告路径",
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
    if args.command == "review":
        report = generate_review_report(args.snapshot_root, args.output)
        print(json.dumps({"report": str(report)}, ensure_ascii=False))
        return 0
    if args.command == "generate-arithmetic":
        arithmetic_result = generate_arithmetic_candidates(
            human_root=args.human_root,
            output_root=args.output_root,
            probe_root=args.probes_root,
        )
        print(
            json.dumps(
                {
                    "output_root": str(arithmetic_result.output_root),
                    "train": arithmetic_result.train_count,
                    "validation": arithmetic_result.validation_count,
                    "buckets": arithmetic_result.buckets,
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "import-wiki":
        result = import_web_wiki(args.source, args.output_root)
    elif args.command == "export":
        with GameClient(args.base_url) as client:
            result = export_mod_knowledge(client, args.output_root)
    elif args.command == "rebuild":
        result = rebuild_mod_knowledge(
            args.raw_root,
            args.output_root,
            wiki_root=args.wiki_root,
            cycles_root=args.cycles_root,
            event_entries_root=args.event_entries_root,
        )
    else:
        result = generate_question_variants(
            args.snapshot_root,
            args.output_root,
        )
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
