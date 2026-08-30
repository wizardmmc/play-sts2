"""提供战斗 rollout 收集、奖励投影与同入口组准入。"""

from .collector import (
    BattleGroupCollector,
    BattleRolloutWorker,
    GameBattleRolloutWorker,
    RolloutInfrastructureError,
    RolloutModelError,
)
from .contracts import (
    ACTION_CONSTRAINT_MODE,
    BEHAVIOR_LOGPROBS_MODE,
    BattleGroupRejected,
    BattleReward,
    BattleRollout,
    BattleRolloutGroup,
    BattleRolloutStep,
    RewardComponent,
    RolloutContractError,
    build_battle_rollout_group,
)
from .dagger import (
    DaggerCandidate,
    DaggerContractError,
    DaggerLabel,
    DaggerReplayLabeler,
    DaggerRolloutSource,
    build_dagger_label,
    load_dagger_rollout_source,
    load_dagger_sft_rows,
    select_dagger_candidates,
    summarize_dagger_unsupported,
    write_dagger_labels,
)
from .entrypoint import collect_battle_rollout_group, label_dagger_rollout_group
from .io import load_battle_scenario, write_battle_rollout_group
from .metrics import (
    BattleEvaluationAttempt,
    BattleRegressionMetrics,
    evaluate_battle_regression_metrics,
)
from .rollout import build_battle_rollout, score_battle_reward

__all__ = [
    "ACTION_CONSTRAINT_MODE",
    "BEHAVIOR_LOGPROBS_MODE",
    "BattleEvaluationAttempt",
    "BattleGroupCollector",
    "BattleGroupRejected",
    "BattleRegressionMetrics",
    "BattleReward",
    "BattleRollout",
    "BattleRolloutGroup",
    "BattleRolloutStep",
    "BattleRolloutWorker",
    "DaggerCandidate",
    "DaggerContractError",
    "DaggerLabel",
    "DaggerReplayLabeler",
    "DaggerRolloutSource",
    "GameBattleRolloutWorker",
    "RewardComponent",
    "RolloutContractError",
    "RolloutInfrastructureError",
    "RolloutModelError",
    "build_battle_rollout",
    "build_battle_rollout_group",
    "build_dagger_label",
    "collect_battle_rollout_group",
    "evaluate_battle_regression_metrics",
    "label_dagger_rollout_group",
    "load_battle_scenario",
    "load_dagger_rollout_source",
    "load_dagger_sft_rows",
    "score_battle_reward",
    "select_dagger_candidates",
    "summarize_dagger_unsupported",
    "write_battle_rollout_group",
    "write_dagger_labels",
]
