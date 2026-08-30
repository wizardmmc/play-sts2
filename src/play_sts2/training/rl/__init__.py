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
from .entrypoint import collect_battle_rollout_group
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
    "GameBattleRolloutWorker",
    "RewardComponent",
    "RolloutContractError",
    "RolloutInfrastructureError",
    "RolloutModelError",
    "build_battle_rollout",
    "build_battle_rollout_group",
    "collect_battle_rollout_group",
    "evaluate_battle_regression_metrics",
    "load_battle_scenario",
    "score_battle_reward",
    "write_battle_rollout_group",
]
