"""提供战略宏 checkpoint、Tree-GRPO 组与工程 return。"""

from .collector import TreeGroupCollector, TreeRolloutWorker
from .contracts import (
    StrategicReturn,
    StrategicReturnComponent,
    TreeGroupRejected,
    TreeRolloutArm,
    TreeRolloutGroup,
    TreeRolloutStep,
    build_tree_rollout_group,
)
from .entrypoint import collect_tree_rollout_group
from .evaluation import evaluate_tree_branch_regret
from .io import write_tree_rollout_group
from .learner import load_tree_training_group
from .macro import (
    MacroCheckpoint,
    automatic_strategic_action,
    classify_macro_checkpoint,
)
from .reward import StrategicReturnInput, score_engineering_milestone_return
from .rollout import (
    TreeRolloutError,
    TreeSuffixResult,
    TreeSuffixRunner,
    build_tree_rollout_arm,
)
from .trainer import TreeGrpoConfig, load_tree_grpo_config, train_tree_grpo
from .worker import GameTreeRolloutWorker

__all__ = [
    "GameTreeRolloutWorker",
    "MacroCheckpoint",
    "StrategicReturn",
    "StrategicReturnComponent",
    "StrategicReturnInput",
    "TreeGroupCollector",
    "TreeGroupRejected",
    "TreeGrpoConfig",
    "TreeRolloutArm",
    "TreeRolloutError",
    "TreeRolloutGroup",
    "TreeRolloutStep",
    "TreeRolloutWorker",
    "TreeSuffixResult",
    "TreeSuffixRunner",
    "automatic_strategic_action",
    "build_tree_rollout_arm",
    "build_tree_rollout_group",
    "classify_macro_checkpoint",
    "collect_tree_rollout_group",
    "evaluate_tree_branch_regret",
    "load_tree_grpo_config",
    "load_tree_training_group",
    "score_engineering_milestone_return",
    "train_tree_grpo",
    "write_tree_rollout_group",
]
