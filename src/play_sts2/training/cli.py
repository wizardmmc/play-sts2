"""提供可读 SFT 数据集构建命令。"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .rl import (
    collect_battle_rollout_group,
    evaluate_battle_rollout_file,
    label_dagger_rollout_group,
    load_battle_grpo_config,
    train_battle_grpo,
    write_battle_reward_comparison,
)
from .sft import (
    build_sft_dataset,
    evaluate_sft,
    evaluate_sft_loss,
    load_sft_config,
    merge_sft_adapter,
    run_knowledge_evaluation,
    train_sft,
)
from .sft_cuda import load_cuda_sft_config, train_sft_cuda


def build_parser() -> argparse.ArgumentParser:
    """构建后训练命令行参数解析器。

    Returns:
        argparse.ArgumentParser: 包含数据构建、SFT 训练和评测子命令的解析器。
    """
    parser = argparse.ArgumentParser(prog="play-sts2-train")
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_rl = subparsers.add_parser(
        "collect-rl-battle",
        help="从本地游戏 workers 收集严格同入口的战斗 rollout group",
    )
    collect_rl.add_argument("--scenario", type=Path, required=True)
    collect_rl.add_argument("--game-url", action="append", required=True)
    collect_rl.add_argument("--model-url", required=True)
    collect_rl.add_argument("--policy-model", required=True)
    collect_rl.add_argument(
        "--vllm-logprobs-mode",
        choices=("processed_logprobs",),
        required=True,
        help="确认 vLLM 已用 --logprobs-mode processed_logprobs 启动",
    )
    collect_rl.add_argument(
        "--structured-output-backend",
        choices=("xgrammar",),
        required=True,
        help="确认 vLLM 当前 structured choice 使用 xgrammar backend",
    )
    collect_rl.add_argument(
        "--structured-output-version",
        required=True,
        help="确认服务端 xgrammar 精确包版本，例如 0.1.33",
    )
    collect_rl.add_argument("--group-id", required=True)
    collect_rl.add_argument("--output", type=Path, required=True)
    collect_rl.add_argument("--group-size", type=int, default=8)
    collect_rl.add_argument("--max-tokens", type=int, default=128)
    collect_rl.add_argument("--temperature", type=float, default=0.8)
    collect_rl.add_argument("--infrastructure-attempts", type=int, default=3)
    label_dagger = subparsers.add_parser(
        "label-rl-dagger",
        help="在教师游戏中重放学生状态并收集 CombatSolver 旁路标签",
    )
    label_dagger.add_argument("--rollout", type=Path, required=True)
    label_dagger.add_argument("--game-url", required=True)
    label_dagger.add_argument("--output", type=Path, required=True)
    label_dagger.add_argument("--max-labels", type=int, default=8)
    label_dagger.add_argument("--selection-seed", type=int, default=0)
    label_dagger.add_argument("--search-timeout", type=float, default=135.0)
    train_grpo = subparsers.add_parser(
        "battle-grpo",
        help="用八臂 rollout 和独立 DAgger loss 训练战斗 LoRA",
    )
    train_grpo.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rl/battle-grpo.toml"),
    )
    train_grpo.add_argument("--name", required=True)
    train_grpo.add_argument(
        "--max-groups",
        type=int,
        help="限制本次新增优化 group 数，用于工程冒烟或中断恢复验证",
    )
    train_grpo.add_argument(
        "--resume",
        action="store_true",
        help="从同名运行的 checkpoint-last 精确恢复",
    )
    compare_rewards = subparsers.add_parser(
        "compare-rl-rewards",
        help="在相同八臂 rollout 上离线比较第三阶段奖励方案",
    )
    compare_rewards.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rl/battle-grpo.toml"),
    )
    compare_rewards.add_argument("--output", type=Path, required=True)
    compare_rewards.add_argument(
        "--rollout",
        type=Path,
        action="append",
        help="显式指定一个 group JSON；可重复，省略时使用配置 rollout_root",
    )
    evaluate_rollout = subparsers.add_parser(
        "eval-rl-rollout",
        help="从一个落盘 battle group 复算固定五项回归指标",
    )
    evaluate_rollout.add_argument("--rollout", type=Path, required=True)
    evaluate_rollout.add_argument("--output", type=Path, required=True)
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
        "--additional-human-root",
        type=Path,
        action="append",
        default=[],
        help="按附加 raw 根自己的 splits.json 合并行为样本",
    )
    build.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/datasets/sft"),
    )
    build.add_argument("--train-run", action="append", default=[])
    build.add_argument("--validation-run", dest="dev_run", action="append", default=[])
    build.add_argument("--eval-run", dest="test_run", action="append", default=[])
    build.add_argument(
        "--run-splits",
        type=Path,
        help="覆盖所有 human raw 根默认名册的独立 splits.json",
    )
    build.add_argument(
        "--dagger-root",
        type=Path,
        help="可选的学生状态 CombatSolver 标签文件或目录",
    )
    build.add_argument(
        "--dagger-policy-model",
        help="--dagger-root 中唯一允许的学生父 policy",
    )
    build.add_argument(
        "--mix",
        type=Path,
        help="可选的训练高频人类动作上限 TOML",
    )

    train = subparsers.add_parser("sft", help="训练 Qwen LoRA adapter")
    train.add_argument("--config", type=Path, default=Path("configs/sft/sft.toml"))
    train.add_argument("--name", required=True, help="adapter 与 run 的目录名称")
    train.add_argument("--max-steps", type=int, help="限制优化步数，用于真实冒烟")
    train.add_argument(
        "--resume",
        action="store_true",
        help="从同名运行的 checkpoint-last 精确恢复",
    )

    train_cuda = subparsers.add_parser("sft-cuda", help="使用独立 CUDA 实现训练 LoRA")
    train_cuda.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sft/sft-cuda.toml"),
    )
    train_cuda.add_argument("--name", required=True, help="adapter 与 run 的目录名称")
    train_cuda.add_argument("--max-steps", type=int, help="限制优化步数，用于真实冒烟")
    train_cuda.add_argument(
        "--resume",
        action="store_true",
        help="从同名运行的 checkpoint-last 精确恢复",
    )

    merge = subparsers.add_parser("merge-sft", help="把 LoRA 合并为独立 HF 模型")
    merge.add_argument("--config", type=Path, default=Path("configs/sft/sft.toml"))
    merge.add_argument("--adapter", type=Path, required=True)
    merge.add_argument("--output", type=Path)

    evaluate = subparsers.add_parser("eval-sft", help="生成式验证 LoRA adapter")
    evaluate.add_argument("--config", type=Path, default=Path("configs/sft/sft.toml"))
    evaluate.add_argument("--adapter", type=Path, required=True)
    evaluate.add_argument(
        "--split",
        choices=("validation", "eval"),
        default="validation",
    )
    evaluate.add_argument("--max-samples", type=int)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--temperature", type=float, default=0.0)

    evaluate_loss = subparsers.add_parser(
        "eval-sft-loss",
        help="计算 assistant-only teacher-forced loss 与 token accuracy",
    )
    evaluate_loss.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sft/sft.toml"),
    )
    evaluate_loss.add_argument("--adapter", type=Path, required=True)
    evaluate_loss.add_argument(
        "--split",
        choices=("validation", "eval"),
        default="validation",
    )
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
        "--eval-root",
        type=Path,
        default=Path("data/datasets/sft/eval"),
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.command == "build-sft"
        and args.run_splits is not None
        and any((args.train_run, args.dev_run, args.test_run))
    ):
        parser.error("--run-splits 不能与 --train-run/--validation-run/--eval-run 混用")
    if args.command == "build-sft" and (
        (args.dagger_root is None) != (args.dagger_policy_model is None)
    ):
        parser.error("--dagger-root 与 --dagger-policy-model 必须同时提供")
    if args.command == "collect-rl-battle":
        output = collect_battle_rollout_group(
            scenario_path=args.scenario,
            game_urls=tuple(args.game_url),
            model_url=args.model_url,
            policy_model=args.policy_model,
            vllm_logprobs_mode=args.vllm_logprobs_mode,
            structured_output_backend=args.structured_output_backend,
            structured_output_version=args.structured_output_version,
            group_id=args.group_id,
            output_path=args.output,
            group_size=args.group_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            infrastructure_attempts=args.infrastructure_attempts,
        )
    elif args.command == "label-rl-dagger":
        output = label_dagger_rollout_group(
            rollout_path=args.rollout,
            game_url=args.game_url,
            output_path=args.output,
            max_labels=args.max_labels,
            selection_seed=args.selection_seed,
            search_timeout=args.search_timeout,
        )
    elif args.command == "battle-grpo":
        config = load_battle_grpo_config(args.config)
        output = train_battle_grpo(
            config,
            args.name,
            max_groups=args.max_groups,
            exact_resume=args.resume,
        )
    elif args.command == "compare-rl-rewards":
        config = load_battle_grpo_config(args.config)
        output = write_battle_reward_comparison(
            config,
            args.output,
            rollout_paths=args.rollout,
        )
    elif args.command == "eval-rl-rollout":
        output = evaluate_battle_rollout_file(args.rollout)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif args.command == "build-sft":
        result = build_sft_dataset(
            knowledge_root=args.knowledge_root,
            human_root=args.human_root,
            additional_human_roots=tuple(args.additional_human_root),
            output_root=args.output_root,
            train_run_ids=args.train_run,
            dev_run_ids=args.dev_run,
            test_run_ids=args.test_run,
            run_splits_path=args.run_splits,
            dagger_root=args.dagger_root,
            dagger_policy_version=args.dagger_policy_model,
            mix_config_path=args.mix,
        )
        output = {
            "output_root": str(result.output_root),
            "train": result.train_count,
            "validation": result.dev_count,
            "eval": result.test_count,
        }
    elif args.command == "sft":
        config = load_sft_config(args.config)
        output = train_sft(
            config,
            args.name,
            max_steps=args.max_steps,
            exact_resume=args.resume,
        )
    elif args.command == "sft-cuda":
        config = load_cuda_sft_config(args.config)
        output = train_sft_cuda(
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
            temperature=args.temperature,
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
            args.eval_root,
            report_path,
            device=args.device,
            limit=args.limit,
            minimum_new_tokens=args.minimum_new_tokens,
        )
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
