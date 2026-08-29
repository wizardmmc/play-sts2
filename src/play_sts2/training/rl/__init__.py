"""提供战斗 rollout 收集、奖励投影与同入口组准入。"""

from .collector import (
    BattleGroupCollector,
    BattleRolloutWorker,
    GameBattleRolloutWorker,
    RolloutInfrastructureError,
    RolloutModelError,
)
from .contracts import (
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
from .rollout import build_battle_rollout, score_battle_reward_v0

__all__ = [
    "BEHAVIOR_LOGPROBS_MODE",
    "BattleGroupCollector",
    "BattleGroupRejected",
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
    "load_battle_scenario",
    "score_battle_reward_v0",
    "write_battle_rollout_group",
]
