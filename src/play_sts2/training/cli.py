"""提供可读 SFT 数据集构建命令。"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .rl import (
    CycleJournal,
    CyclePhase,
    LongRunCurriculumState,
    TensorboardMetricsWriter,
    TrainingAssignment,
    TrainingCycleResult,
    build_promotion_receipt,
    collect_battle_rollout_group,
    collect_gigpo_group,
    collect_terminal_tree_group,
    collect_tree_rollout_group,
    compose_serving_adapter,
    evaluate_battle_rollout_file,
    evaluate_policy_pair,
    evaluate_tree_branch_regret,
    initialize_residual_adapters,
    label_dagger_rollout_group,
    load_battle_grpo_config,
    load_cycle_journal,
    load_long_run_curriculum,
    load_strategy_grpo_config,
    load_tree_grpo_config,
    new_long_run_curriculum,
    next_training_assignment,
    record_training_result,
    select_battle_candidates,
    select_tree_checkpoints,
    summarize_cycle_timing,
    train_battle_grpo,
    train_strategy_grpo,
    train_tree_grpo,
    write_battle_reward_comparison,
    write_cycle_journal,
    write_long_run_curriculum,
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
    collect_rl.add_argument(
        "--entry-snapshot",
        type=Path,
        help="可选的完整游戏原始战斗入口快照",
    )
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
    collect_rl.add_argument(
        "--allow-zero-variance",
        action="store_true",
        help="为独立 DAgger 保存零优势完整八臂组",
    )
    collect_tree = subparsers.add_parser(
        "collect-rl-tree",
        help="从原生战略 checkpoint 收集两个本地 worker 的 K=8 Tree group",
    )
    collect_tree.add_argument("--checkpoint", type=Path, required=True)
    collect_tree.add_argument("--executable", type=Path, required=True)
    collect_tree.add_argument("--profile", type=Path, required=True)
    collect_tree.add_argument("--home-root", type=Path, required=True)
    collect_tree.add_argument("--port", type=int, action="append", required=True)
    collect_tree.add_argument("--strategy-model-url", required=True)
    collect_tree.add_argument("--battle-model-url", required=True)
    collect_tree.add_argument("--strategy-policy-model", required=True)
    collect_tree.add_argument("--battle-policy-model", required=True)
    collect_tree.add_argument(
        "--vllm-logprobs-mode",
        choices=("processed_logprobs",),
        required=True,
    )
    collect_tree.add_argument(
        "--structured-output-backend",
        choices=("xgrammar",),
        required=True,
    )
    collect_tree.add_argument("--structured-output-version", required=True)
    collect_tree.add_argument("--group-id", required=True)
    collect_tree.add_argument("--output", type=Path, required=True)
    collect_tree.add_argument("--group-size", type=int, default=8)
    collect_tree.add_argument("--max-tokens", type=int, default=128)
    collect_tree.add_argument("--temperature", type=float, default=0.8)
    collect_tree.add_argument("--max-macro-checkpoints", type=int, default=2)
    collect_gigpo = subparsers.add_parser(
        "collect-rl-gigpo",
        help="用分层双 policy 收集同种子八条完整游戏",
    )
    collect_gigpo.add_argument("--executable", type=Path, required=True)
    collect_gigpo.add_argument("--profile", type=Path, required=True)
    collect_gigpo.add_argument("--home-root", type=Path, required=True)
    collect_gigpo.add_argument("--checkpoint-root", type=Path, required=True)
    collect_gigpo.add_argument("--port", type=int, action="append", required=True)
    collect_gigpo.add_argument("--strategy-model-url", required=True)
    collect_gigpo.add_argument("--battle-model-url", required=True)
    collect_gigpo.add_argument("--strategy-policy-model", required=True)
    collect_gigpo.add_argument("--battle-policy-model", required=True)
    collect_gigpo.add_argument("--seed", required=True)
    collect_gigpo.add_argument("--character", default="DEFECT")
    collect_gigpo.add_argument("--ascension", type=int, default=0)
    collect_gigpo.add_argument(
        "--vllm-logprobs-mode",
        choices=("processed_logprobs",),
        required=True,
    )
    collect_gigpo.add_argument(
        "--structured-output-backend",
        choices=("xgrammar",),
        required=True,
    )
    collect_gigpo.add_argument("--structured-output-version", required=True)
    collect_gigpo.add_argument("--group-id", required=True)
    collect_gigpo.add_argument("--output", type=Path, required=True)
    collect_gigpo.add_argument("--candidates", type=Path, required=True)
    collect_gigpo.add_argument("--group-size", type=int, default=8)
    collect_gigpo.add_argument("--max-tokens", type=int, default=128)
    collect_gigpo.add_argument("--temperature", type=float, default=0.8)
    collect_gigpo.add_argument("--lambda-milestone", type=float, default=1.0)
    collect_gigpo.add_argument(
        "--normalization",
        choices=("one", "std"),
        default="one",
    )
    collect_terminal = subparsers.add_parser(
        "collect-rl-terminal-tree",
        help="从一个原生宏 checkpoint 收集分层 terminal Tree",
    )
    collect_terminal.add_argument("--checkpoint", type=Path, required=True)
    collect_terminal.add_argument("--executable", type=Path, required=True)
    collect_terminal.add_argument("--profile", type=Path, required=True)
    collect_terminal.add_argument("--home-root", type=Path, required=True)
    collect_terminal.add_argument("--port", type=int, action="append", required=True)
    collect_terminal.add_argument("--strategy-model-url", required=True)
    collect_terminal.add_argument("--battle-model-url", required=True)
    collect_terminal.add_argument("--strategy-policy-model", required=True)
    collect_terminal.add_argument("--battle-policy-model", required=True)
    collect_terminal.add_argument(
        "--vllm-logprobs-mode",
        choices=("processed_logprobs",),
        required=True,
    )
    collect_terminal.add_argument(
        "--structured-output-backend",
        choices=("xgrammar",),
        required=True,
    )
    collect_terminal.add_argument("--structured-output-version", required=True)
    collect_terminal.add_argument("--group-id", required=True)
    collect_terminal.add_argument("--output", type=Path, required=True)
    collect_terminal.add_argument("--max-tokens", type=int, default=128)
    collect_terminal.add_argument("--temperature", type=float, default=0.8)
    collect_terminal.add_argument("--max-model-steps", type=int, default=400)
    select_battles = subparsers.add_parser(
        "select-rl-battles",
        help="从完整游戏候选中按结果事实选择战斗刷新场景",
    )
    select_battles.add_argument(
        "--candidates",
        type=Path,
        action="append",
        required=True,
        help="先给 backbone candidates，再重复传入 terminal Tree group",
    )
    select_battles.add_argument("--output-root", type=Path, required=True)
    select_battles.add_argument("--history", type=Path, required=True)
    select_battles.add_argument("--max-scenarios", type=int, required=True)
    select_battles.add_argument("--selection-seed", type=int, default=0)
    select_tree_stage7 = subparsers.add_parser(
        "select-rl-tree",
        help="从完整游戏候选中选择早/中期与较晚 terminal Tree 节点",
    )
    select_tree_stage7.add_argument("--candidates", type=Path, required=True)
    select_tree_stage7.add_argument("--output", type=Path, required=True)
    select_tree_stage7.add_argument("--max-checkpoints", type=int, default=2)
    select_tree_stage7.add_argument("--selection-seed", type=int, default=0)
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
    train_tree = subparsers.add_parser(
        "tree-grpo",
        help="在一个 K=8 战略兄弟组上执行单卡可行性更新",
    )
    train_tree.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rl/tree-grpo.toml"),
    )
    train_tree.add_argument("--name", required=True)
    train_strategy = subparsers.add_parser(
        "strategy-grpo",
        help="组合 GiGPO 与 terminal Tree 更新战略 residual",
    )
    train_strategy.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rl/strategy-grpo.toml"),
    )
    train_strategy.add_argument("--name", required=True)
    init_policies = subparsers.add_parser(
        "init-rl-policies",
        help="从 LoRA 结构创建函数为零的战略与战斗 residual",
    )
    init_policies.add_argument("--template", type=Path, required=True)
    init_policies.add_argument("--output-root", type=Path, required=True)
    init_policies.add_argument("--strategy-name", required=True)
    init_policies.add_argument("--battle-name", required=True)
    init_policies.add_argument("--seed", type=int, default=20260830)
    compose_policy = subparsers.add_parser(
        "compose-rl-policy",
        help="把 SFT LoRA 与一个 residual 精确拼成 vLLM serving LoRA",
    )
    compose_policy.add_argument("--parent", type=Path, required=True)
    compose_policy.add_argument("--residual", type=Path, required=True)
    compose_policy.add_argument("--output", type=Path, required=True)
    compose_policy.add_argument("--name", required=True)
    evaluate_policy = subparsers.add_parser(
        "eval-rl-policy",
        help="用双 residual 运行三个 frozen seed 加一个 fresh seed",
    )
    evaluate_policy.add_argument("--executable", type=Path, required=True)
    evaluate_policy.add_argument("--profile", type=Path, required=True)
    evaluate_policy.add_argument("--home-root", type=Path, required=True)
    evaluate_policy.add_argument("--port", type=int, action="append", required=True)
    evaluate_policy.add_argument("--model-url", required=True)
    evaluate_policy.add_argument("--strategy-policy", required=True)
    evaluate_policy.add_argument("--battle-policy", required=True)
    evaluate_policy.add_argument(
        "--frozen-seed",
        action="append",
        required=True,
    )
    evaluate_policy.add_argument("--fresh-seed", required=True)
    evaluate_policy.add_argument("--output", type=Path, required=True)
    evaluate_policy.add_argument("--character", default="DEFECT")
    evaluate_policy.add_argument("--ascension", type=int, default=0)
    evaluate_policy.add_argument("--max-tokens", type=int, default=128)
    evaluate_policy.add_argument("--temperature", type=float, default=0.0)
    evaluate_policy.add_argument("--tensorboard-dir", type=Path)
    evaluate_policy.add_argument("--cycle-index", type=int)
    summarize_cycle = subparsers.add_parser(
        "summarize-rl-cycle",
        help="按阶段七口径汇总完整轮次墙钟预算",
    )
    summarize_cycle.add_argument("--backbone-seconds", type=float, required=True)
    summarize_cycle.add_argument("--tree-seconds", type=float, required=True)
    summarize_cycle.add_argument("--battle-seconds", type=float, required=True)
    summarize_cycle.add_argument("--solver-seconds", type=float, required=True)
    summarize_cycle.add_argument(
        "--strategy-update-seconds",
        type=float,
        required=True,
    )
    summarize_cycle.add_argument(
        "--battle-update-seconds",
        type=float,
        required=True,
    )
    summarize_cycle.add_argument(
        "--validation-seconds",
        type=float,
        required=True,
    )
    summarize_cycle.add_argument("--output", type=Path, required=True)
    record_cycle = subparsers.add_parser(
        "record-rl-cycle",
        help="在完整 group、optimizer 或验证边界持久化阶段七轮次",
    )
    record_cycle.add_argument("--journal", type=Path, required=True)
    record_cycle.add_argument(
        "--phase",
        choices=tuple(phase.value for phase in CyclePhase),
        required=True,
    )
    record_cycle.add_argument("--cycle-id")
    record_cycle.add_argument("--strategy-policy")
    record_cycle.add_argument("--battle-policy")
    record_cycle.add_argument("--train-seed")
    record_cycle.add_argument("--last-promoted-strategy-policy")
    record_cycle.add_argument("--last-promoted-battle-policy")
    record_cycle.add_argument("--new-strategy-policy")
    record_cycle.add_argument("--new-battle-policy")
    record_cycle.add_argument("--promotion-receipt", type=Path)
    promotion = subparsers.add_parser(
        "decide-rl-promotion",
        help="用三次滚动验证与固定回归生成双 policy 原子晋升收据",
    )
    promotion.add_argument("--parent-report", type=Path, required=True)
    promotion.add_argument(
        "--candidate-report",
        type=Path,
        action="append",
        required=True,
    )
    promotion.add_argument("--battle-regression-passed", action="store_true")
    promotion.add_argument("--branch-regret-passed", action="store_true")
    promotion.add_argument("--illegal-action-passed", action="store_true")
    promotion.add_argument("--potion-guard-passed", action="store_true")
    promotion.add_argument("--output", type=Path, required=True)
    init_curriculum = subparsers.add_parser(
        "init-rl-curriculum",
        help="创建长期 RL 的 cold-start 课程状态",
    )
    init_curriculum.add_argument("--state", type=Path, required=True)
    init_curriculum.add_argument("--target-seed", required=True)
    init_curriculum.add_argument("--ascension", type=int, default=0)
    plan_curriculum = subparsers.add_parser(
        "plan-rl-curriculum",
        help="从长期课程状态选择下一轮 seed 与进阶",
    )
    plan_curriculum.add_argument("--state", type=Path, required=True)
    plan_curriculum.add_argument("--fresh-seed", required=True)
    update_curriculum = subparsers.add_parser(
        "update-rl-curriculum",
        help="记录一轮 K=8 结果并推进长期课程",
    )
    update_curriculum.add_argument("--state", type=Path, required=True)
    update_curriculum.add_argument("--seed", required=True)
    update_curriculum.add_argument(
        "--source",
        choices=("target", "cleared", "hard", "fresh"),
        required=True,
    )
    update_curriculum.add_argument("--ascension", type=int, required=True)
    update_curriculum.add_argument("--cycle-index", type=int, required=True)
    update_curriculum.add_argument("--wins", type=int, required=True)
    update_curriculum.add_argument("--attempts", type=int, default=8)
    update_curriculum.add_argument("--mean-floor", type=float, required=True)
    update_curriculum.add_argument("--bosses-cleared", type=int, required=True)
    update_curriculum.add_argument(
        "--rolling-validation-mean-floor",
        type=float,
        required=True,
    )
    update_curriculum.add_argument("--validation-stable", action="store_true")
    update_curriculum.add_argument("--validation-over-budget", action="store_true")
    update_curriculum.add_argument(
        "--battle-regression-passed",
        action="store_true",
    )
    update_curriculum.add_argument(
        "--tree-regression-passed",
        action="store_true",
    )
    update_curriculum.add_argument("--tensorboard-dir", type=Path)
    evaluate_tree = subparsers.add_parser(
        "eval-tree-regret",
        help="查询冻结战略 policy 并计算 held-out 兄弟计划 regret",
    )
    evaluate_tree.add_argument("--rollout", type=Path, required=True)
    evaluate_tree.add_argument("--model-url", required=True)
    evaluate_tree.add_argument("--policy-model", required=True)
    evaluate_tree.add_argument("--output", type=Path, required=True)
    evaluate_tree.add_argument("--max-tokens", type=int, default=128)
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
            expected_snapshot_path=args.entry_snapshot,
            allow_zero_variance=args.allow_zero_variance,
        )
    elif args.command == "collect-rl-tree":
        output = collect_tree_rollout_group(
            checkpoint_path=args.checkpoint,
            executable=args.executable,
            profile=args.profile,
            home_root=args.home_root,
            ports=tuple(args.port),
            strategy_model_url=args.strategy_model_url,
            battle_model_url=args.battle_model_url,
            strategy_policy_model=args.strategy_policy_model,
            battle_policy_model=args.battle_policy_model,
            vllm_logprobs_mode=args.vllm_logprobs_mode,
            structured_output_backend=args.structured_output_backend,
            structured_output_version=args.structured_output_version,
            group_id=args.group_id,
            output_path=args.output,
            group_size=args.group_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_macro_checkpoints=args.max_macro_checkpoints,
        )
    elif args.command == "collect-rl-gigpo":
        output = collect_gigpo_group(
            executable=args.executable,
            profile=args.profile,
            home_root=args.home_root,
            checkpoint_root=args.checkpoint_root,
            ports=tuple(args.port),
            strategy_model_url=args.strategy_model_url,
            battle_model_url=args.battle_model_url,
            strategy_policy_model=args.strategy_policy_model,
            battle_policy_model=args.battle_policy_model,
            seed=args.seed,
            character_id=args.character,
            ascension=args.ascension,
            vllm_logprobs_mode=args.vllm_logprobs_mode,
            structured_output_backend=args.structured_output_backend,
            structured_output_version=args.structured_output_version,
            group_id=args.group_id,
            output_path=args.output,
            candidates_path=args.candidates,
            group_size=args.group_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            lambda_milestone=args.lambda_milestone,
            normalization=args.normalization,
        )
    elif args.command == "collect-rl-terminal-tree":
        output = collect_terminal_tree_group(
            checkpoint_path=args.checkpoint,
            executable=args.executable,
            profile=args.profile,
            home_root=args.home_root,
            ports=tuple(args.port),
            strategy_model_url=args.strategy_model_url,
            battle_model_url=args.battle_model_url,
            strategy_policy_model=args.strategy_policy_model,
            battle_policy_model=args.battle_policy_model,
            vllm_logprobs_mode=args.vllm_logprobs_mode,
            structured_output_backend=args.structured_output_backend,
            structured_output_version=args.structured_output_version,
            group_id=args.group_id,
            output_path=args.output,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_model_steps=args.max_model_steps,
        )
    elif args.command == "select-rl-battles":
        output = select_battle_candidates(
            candidates_path=args.candidates[0],
            additional_candidates_paths=tuple(args.candidates[1:]),
            output_root=args.output_root,
            history_path=args.history,
            max_scenarios=args.max_scenarios,
            seed=args.selection_seed,
        )
    elif args.command == "select-rl-tree":
        output = select_tree_checkpoints(
            candidates_path=args.candidates,
            output_path=args.output,
            max_checkpoints=args.max_checkpoints,
            selection_seed=args.selection_seed,
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
    elif args.command == "tree-grpo":
        config = load_tree_grpo_config(args.config)
        output = train_tree_grpo(config, args.name)
    elif args.command == "strategy-grpo":
        config = load_strategy_grpo_config(args.config)
        output = train_strategy_grpo(config, args.name)
    elif args.command == "init-rl-policies":
        output = initialize_residual_adapters(
            template_adapter=args.template,
            output_root=args.output_root,
            strategy_name=args.strategy_name,
            battle_name=args.battle_name,
            seed=args.seed,
        )
    elif args.command == "compose-rl-policy":
        output = compose_serving_adapter(
            parent_adapter=args.parent,
            residual_adapter=args.residual,
            output=args.output,
            name=args.name,
        )
    elif args.command == "eval-rl-policy":
        if len(args.frozen_seed) != 3:
            parser.error("--frozen-seed 必须恰好提供三次")
        output = evaluate_policy_pair(
            executable=args.executable,
            profile=args.profile,
            home_root=args.home_root,
            ports=tuple(args.port),
            model_url=args.model_url,
            strategy_policy=args.strategy_policy,
            battle_policy=args.battle_policy,
            frozen_seeds=tuple(args.frozen_seed),
            fresh_seed=args.fresh_seed,
            output_path=args.output,
            character_id=args.character,
            ascension=args.ascension,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            tensorboard_dir=args.tensorboard_dir,
            cycle_index=args.cycle_index,
        )
    elif args.command == "summarize-rl-cycle":
        output = summarize_cycle_timing(
            backbone_seconds=args.backbone_seconds,
            tree_seconds=args.tree_seconds,
            battle_seconds=args.battle_seconds,
            solver_seconds=args.solver_seconds,
            strategy_update_seconds=args.strategy_update_seconds,
            battle_update_seconds=args.battle_update_seconds,
            validation_seconds=args.validation_seconds,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif args.command == "record-rl-cycle":
        phase = CyclePhase(args.phase)
        if args.journal.exists():
            journal = load_cycle_journal(args.journal)
            receipt = (
                json.loads(args.promotion_receipt.read_text(encoding="utf-8"))
                if args.promotion_receipt is not None
                else None
            )
            journal = journal.advance(
                phase,
                strategy_policy=args.new_strategy_policy,
                battle_policy=args.new_battle_policy,
                promotion_receipt=receipt,
            )
        else:
            if phase is not CyclePhase.CREATED or not all(
                (
                    args.cycle_id,
                    args.strategy_policy,
                    args.battle_policy,
                    args.train_seed,
                    args.last_promoted_strategy_policy,
                    args.last_promoted_battle_policy,
                )
            ):
                parser.error(
                    "新 journal 必须从 created、完整 sampling parent、"
                    "last promoted pair 和 seed 开始"
                )
            journal = CycleJournal.new(
                cycle_id=args.cycle_id,
                strategy_policy=args.strategy_policy,
                battle_policy=args.battle_policy,
                train_seed=args.train_seed,
                last_promoted_strategy_policy=args.last_promoted_strategy_policy,
                last_promoted_battle_policy=args.last_promoted_battle_policy,
            )
        write_cycle_journal(journal, args.journal)
        output = {
            "cycle_id": journal.cycle_id,
            "phase": journal.phase.value,
            "strategy_policy": journal.strategy_policy,
            "battle_policy": journal.battle_policy,
            "last_promoted_strategy_policy": journal.last_promoted_strategy_policy,
            "last_promoted_battle_policy": journal.last_promoted_battle_policy,
            "journal": str(args.journal),
        }
    elif args.command == "decide-rl-promotion":
        if len(args.candidate_report) != 3:
            parser.error("--candidate-report 必须恰好提供三次")
        parent_report = json.loads(args.parent_report.read_text(encoding="utf-8"))
        candidate_reports = tuple(
            json.loads(path.read_text(encoding="utf-8"))
            for path in args.candidate_report
        )
        output = build_promotion_receipt(
            parent_report=parent_report,
            rolling_candidate_reports=candidate_reports,
            battle_regression_passed=args.battle_regression_passed,
            branch_regret_passed=args.branch_regret_passed,
            illegal_action_passed=args.illegal_action_passed,
            potion_guard_passed=args.potion_guard_passed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif args.command == "init-rl-curriculum":
        if args.state.exists():
            parser.error("长期 curriculum 状态已存在")
        curriculum = new_long_run_curriculum(
            target_seed=args.target_seed,
            ascension=args.ascension,
        )
        write_long_run_curriculum(curriculum, args.state)
        output = _curriculum_output(curriculum, args.state)
    elif args.command == "plan-rl-curriculum":
        curriculum = load_long_run_curriculum(args.state)
        assignment = next_training_assignment(
            curriculum,
            fresh_seed=args.fresh_seed,
        )
        output = {
            "seed": assignment.seed,
            "source": assignment.source,
            "ascension": assignment.ascension,
            "cycle_index": assignment.cycle_index,
            "full_validation_due": assignment.full_validation_due,
        }
    elif args.command == "update-rl-curriculum":
        curriculum = load_long_run_curriculum(args.state)
        assignment = TrainingAssignment(
            seed=args.seed,
            source=args.source,
            ascension=args.ascension,
            cycle_index=args.cycle_index,
            full_validation_due=(
                curriculum.total_cycles % curriculum.validation_interval == 0
            ),
        )
        result = TrainingCycleResult(
            seed=args.seed,
            ascension=args.ascension,
            wins=args.wins,
            attempts=args.attempts,
            mean_floor=args.mean_floor,
            bosses_cleared=args.bosses_cleared,
        )
        curriculum = record_training_result(
            curriculum,
            assignment,
            result,
            rolling_validation_mean_floor=args.rolling_validation_mean_floor,
            validation_stable=args.validation_stable,
            battle_regression_passed=args.battle_regression_passed,
            tree_regression_passed=args.tree_regression_passed,
            validation_over_budget=args.validation_over_budget,
        )
        if args.tensorboard_dir is not None:
            with TensorboardMetricsWriter(args.tensorboard_dir) as writer:
                writer.write(
                    {
                        "rollout": {
                            "wins": result.wins,
                            "attempts": result.attempts,
                            "win_rate": result.wins / result.attempts,
                            "mean_floor": result.mean_floor,
                            "bosses_cleared": result.bosses_cleared,
                        },
                        "curriculum": {
                            "ascension": curriculum.ascension,
                            "phase": _curriculum_phase_number(curriculum.phase.value),
                            "full_validation_due": assignment.full_validation_due,
                        },
                        "validation": {
                            "rolling_mean_floor": (args.rolling_validation_mean_floor),
                            "stable": args.validation_stable,
                            "battle_regression": args.battle_regression_passed,
                            "tree_regression": args.tree_regression_passed,
                        },
                    },
                    step=assignment.cycle_index,
                )
        write_long_run_curriculum(curriculum, args.state)
        output = _curriculum_output(curriculum, args.state)
    elif args.command == "eval-tree-regret":
        output = evaluate_tree_branch_regret(
            rollout_path=args.rollout,
            model_url=args.model_url,
            policy_model=args.policy_model,
            output_path=args.output,
            max_tokens=args.max_tokens,
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


def _curriculum_output(
    curriculum: LongRunCurriculumState,
    path: Path,
) -> dict[str, object]:
    """构造长期课程 CLI 的紧凑可读摘要。

    Args:
        curriculum (object): ``LongRunCurriculumState`` 实例。
        path (Path): 当前持久化状态文件。

    Returns:
        dict[str, object]: 阶段、难度、池和下一轮游标。
    """
    return {
        "phase": curriculum.phase.value,
        "ascension": curriculum.ascension,
        "target_seed": curriculum.target_seed,
        "total_cycles": curriculum.total_cycles,
        "cleared_seeds": list(curriculum.cleared_seeds),
        "hard_seeds": list(curriculum.hard_seeds),
        "validation_interval": curriculum.validation_interval,
        "next_full_validation_due": (
            curriculum.total_cycles % curriculum.validation_interval == 0
        ),
        "state": str(path),
    }


def _curriculum_phase_number(value: str) -> int:
    """把课程阶段映射为 TensorBoard 可画的有序数值。

    Args:
        value (str): cold_start、transfer 或 formal。

    Raises:
        ValueError: 阶段未知。

    Returns:
        int: cold-start 为零，迁移为一，正式为二。
    """
    try:
        return {"cold_start": 0, "transfer": 1, "formal": 2}[value]
    except KeyError as exc:
        raise ValueError(f"未知长期 RL curriculum 阶段: {value}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
