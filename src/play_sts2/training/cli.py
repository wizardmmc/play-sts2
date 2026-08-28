"""提供可读 SFT 数据集构建命令。"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .sft import (
    build_sft_dataset,
    evaluate_sft,
    evaluate_sft_loss,
    load_sft_config,
    merge_sft_adapter,
    run_knowledge_evaluation,
    train_sft,
)


def build_parser() -> argparse.ArgumentParser:
    """构建后训练命令行参数解析器。

    Returns:
        argparse.ArgumentParser: 包含数据构建、SFT 训练和评测子命令的解析器。
    """
    parser = argparse.ArgumentParser(prog="play-sts2-train")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-sft", help="构建可读 SFT messages")
    build.add_argument(
        "--knowledge-root",
        type=Path,
        default=Path("data/game_knowledge/generated-v0.107.1"),
    )
    build.add_argument(
        "--human-root",
        type=Path,
        default=Path("data/raw/human"),
    )
    build.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/datasets/sft"),
    )
    build.add_argument("--train-run", action="append", default=[])
    build.add_argument("--dev-run", action="append", default=[])
    build.add_argument("--test-run", action="append", default=[])
    build.add_argument(
        "--knowledge-probes-root",
        type=Path,
        default=Path("data/datasets/sft/eval/knowledge"),
    )

    train = subparsers.add_parser("sft", help="训练 Qwen LoRA adapter")
    train.add_argument("--config", type=Path, default=Path("configs/sft.toml"))
    train.add_argument("--name", required=True, help="adapter 与 run 的目录名称")
    train.add_argument("--max-steps", type=int, help="限制优化步数，用于真实冒烟")
    train.add_argument(
        "--resume",
        action="store_true",
        help="从同名运行的 checkpoint-last 精确恢复",
    )

    merge = subparsers.add_parser("merge-sft", help="把 LoRA 合并为独立 HF 模型")
    merge.add_argument("--config", type=Path, default=Path("configs/sft.toml"))
    merge.add_argument("--adapter", type=Path, required=True)
    merge.add_argument("--output", type=Path)

    evaluate = subparsers.add_parser("eval-sft", help="生成式验证 LoRA adapter")
    evaluate.add_argument("--config", type=Path, default=Path("configs/sft.toml"))
    evaluate.add_argument("--adapter", type=Path, required=True)
    evaluate.add_argument("--split", choices=("dev", "test"), default="dev")
    evaluate.add_argument("--max-samples", type=int)
    evaluate.add_argument("--output", type=Path)

    evaluate_loss = subparsers.add_parser(
        "eval-sft-loss",
        help="计算 assistant-only teacher-forced loss 与 token accuracy",
    )
    evaluate_loss.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sft.toml"),
    )
    evaluate_loss.add_argument("--adapter", type=Path, required=True)
    evaluate_loss.add_argument("--split", choices=("dev", "test"), default="dev")
    evaluate_loss.add_argument("--max-samples", type=int)
    evaluate_loss.add_argument("--output", type=Path)

    knowledge = subparsers.add_parser(
        "eval-sft-knowledge",
        help="运行未见问法知识召回与组合算术探针",
    )
    knowledge.add_argument(
        "--model",
        type=Path,
        default=Path("models/merged/sft-clean-20260827-native-r16-e2-merged"),
    )
    knowledge.add_argument(
        "--probes-root",
        type=Path,
        default=Path("data/datasets/sft/eval/knowledge"),
    )
    knowledge.add_argument("--output", type=Path)
    knowledge.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    knowledge.add_argument("--limit", type=int)
    knowledge.add_argument("--minimum-new-tokens", type=int, default=96)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行 SFT 数据构建、训练、合并或评测命令。

    Args:
        argv (Sequence[str] | None): 可选命令行参数，省略时读取进程参数。

    Returns:
        int: 所选命令成功完成时返回 ``0``。
    """
    args = build_parser().parse_args(argv)
    if args.command == "build-sft":
        result = build_sft_dataset(
            knowledge_root=args.knowledge_root,
            human_root=args.human_root,
            output_root=args.output_root,
            train_run_ids=args.train_run,
            dev_run_ids=args.dev_run,
            test_run_ids=args.test_run,
            knowledge_probe_root=args.knowledge_probes_root,
        )
        output = {
            "output_root": str(result.output_root),
            "train": result.train_count,
            "dev": result.dev_count,
            "test": result.test_count,
        }
    elif args.command == "sft":
        config = load_sft_config(args.config)
        output = train_sft(
            config,
            args.name,
            max_steps=args.max_steps,
            exact_resume=args.resume,
        )
    elif args.command == "merge-sft":
        config = load_sft_config(args.config)
        output = merge_sft_adapter(config, args.adapter, args.output)
    elif args.command == "eval-sft":
        config = load_sft_config(args.config)
        output = evaluate_sft(
            config,
            args.adapter,
            args.split,
            max_samples=args.max_samples,
            output_path=args.output,
        )
    elif args.command == "eval-sft-loss":
        config = load_sft_config(args.config)
        output = evaluate_sft_loss(
            config,
            args.adapter,
            args.split,
            max_samples=args.max_samples,
            output_path=args.output,
        )
    else:
        report_path = args.output or (
            Path("runs/eval") / f"{args.model.name}-knowledge.json"
        )
        output = run_knowledge_evaluation(
            args.model,
            args.probes_root,
            report_path,
            device=args.device,
            limit=args.limit,
            minimum_new_tokens=args.minimum_new_tokens,
        )
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
