"""计算不含构筑强度启发式的战略工程 return。"""

import math
from dataclasses import dataclass

from .contracts import StrategicReturn, StrategicReturnComponent


@dataclass(frozen=True, slots=True)
class StrategicReturnInput:
    """保存工程 milestone return 的真实结果字段。

    Args:
        entry_floor (int): checkpoint 入口总楼层。
        final_floor (int): suffix 结束总楼层。
        final_hp (int): suffix 结束生命值。
        max_hp (int): suffix 结束最大生命值。
        bosses_cleared (int): suffix 内实际击败的 Boss 数。
        victory (bool): suffix 是否到达整局胜利。
        died (bool): suffix 是否以死亡结束。
    """

    entry_floor: int
    final_floor: int
    final_hp: int
    max_hp: int
    bosses_cleared: int
    victory: bool
    died: bool


def score_engineering_milestone_return(
    inputs: StrategicReturnInput,
) -> StrategicReturn:
    """用终局、Boss、真实楼层进度和极小 HP tie-break 计算工程回报。

    该方案只用于可行性 smoke。它不包含卡牌、遗物、路线或商店强度表，
    也不作为正式战略奖励结论。

    Args:
        inputs (StrategicReturnInput): suffix 的真实环境结果。

    Raises:
        ValueError: 楼层、生命或 Boss 计数不符合游戏边界。

    Returns:
        StrategicReturn: 可审计的工程标量 return。
    """
    if (
        inputs.entry_floor < 0
        or inputs.final_floor < inputs.entry_floor
        or inputs.max_hp <= 0
        or inputs.final_hp < 0
        or inputs.final_hp > inputs.max_hp
        or inputs.bosses_cleared < 0
        or inputs.victory
        and inputs.died
    ):
        raise ValueError("战略工程 return 输入无效")
    victory = 4.0 if inputs.victory else 0.0
    boss = float(inputs.bosses_cleared)
    progress = (inputs.final_floor - inputs.entry_floor) * 0.01
    hp_tiebreak = 0.0 if inputs.died else inputs.final_hp / inputs.max_hp * 0.001
    total = victory + boss + progress + hp_tiebreak
    if not math.isfinite(total):
        raise ValueError("战略工程 return 必须有限")
    return StrategicReturn(
        scheme="engineering_terminal_milestone",
        total=total,
        components=(
            StrategicReturnComponent("victory", victory),
            StrategicReturnComponent("boss_milestones", boss),
            StrategicReturnComponent("floor_progress", progress),
            StrategicReturnComponent("hp_tiebreak", hp_tiebreak),
        ),
    )
