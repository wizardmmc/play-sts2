"""提供可供模型闭环和 GRPO 复用的战斗场景接口。"""

from .models import (
    BattleScenario,
    BattleSnapshot,
    CardSnapshot,
    EnemySnapshot,
    IntentSnapshot,
    ModelInputSnapshot,
    ScenarioResetResult,
)
from .resetter import BattleResetError, BattleResetter
from .verifier import (
    ScenarioVerificationError,
    capture_battle_snapshot,
    verify_battle_scenario,
)

__all__ = [
    "BattleResetError",
    "BattleResetter",
    "BattleScenario",
    "BattleSnapshot",
    "CardSnapshot",
    "EnemySnapshot",
    "IntentSnapshot",
    "ModelInputSnapshot",
    "ScenarioResetResult",
    "ScenarioVerificationError",
    "capture_battle_snapshot",
    "verify_battle_scenario",
]
