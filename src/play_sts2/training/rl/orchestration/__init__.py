"""提供阶段七完整训练轮次的轻量状态机。"""

from .adapters import compose_serving_adapter, initialize_residual_adapters
from .curriculum import (
    CurriculumPhase,
    LongRunCurriculumState,
    TrainingAssignment,
    TrainingCycleResult,
    ascension_ready,
    formal_seed_bucket,
    load_long_run_curriculum,
    new_long_run_curriculum,
    next_training_assignment,
    record_training_result,
    write_long_run_curriculum,
)
from .cycle import (
    CycleJournal,
    CyclePhase,
    build_promotion_receipt,
    load_cycle_journal,
    write_cycle_journal,
)
from .evaluation import (
    PolicyRunEvaluation,
    build_policy_evaluation_report,
    evaluate_policy_pair,
)
from .selection import select_tree_checkpoints
from .strategy_trainer import (
    StrategyGrpoConfig,
    load_strategy_grpo_config,
    train_strategy_grpo,
)
from .telemetry import TensorboardMetricsWriter, flatten_tensorboard_metrics
from .timing import summarize_cycle_timing

__all__ = [
    "CurriculumPhase",
    "CycleJournal",
    "CyclePhase",
    "LongRunCurriculumState",
    "PolicyRunEvaluation",
    "StrategyGrpoConfig",
    "TensorboardMetricsWriter",
    "TrainingAssignment",
    "TrainingCycleResult",
    "ascension_ready",
    "build_policy_evaluation_report",
    "build_promotion_receipt",
    "compose_serving_adapter",
    "evaluate_policy_pair",
    "flatten_tensorboard_metrics",
    "formal_seed_bucket",
    "initialize_residual_adapters",
    "load_cycle_journal",
    "load_long_run_curriculum",
    "load_strategy_grpo_config",
    "new_long_run_curriculum",
    "next_training_assignment",
    "record_training_result",
    "select_tree_checkpoints",
    "summarize_cycle_timing",
    "train_strategy_grpo",
    "write_cycle_journal",
    "write_long_run_curriculum",
]
