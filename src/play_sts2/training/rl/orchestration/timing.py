"""按阶段七文档口径汇总完整轮次墙钟。"""

import math


def summarize_cycle_timing(
    *,
    backbone_seconds: float,
    tree_seconds: float,
    battle_seconds: float,
    solver_seconds: float,
    strategy_update_seconds: float,
    battle_update_seconds: float,
    validation_seconds: float,
) -> dict[str, float | bool]:
    """计算训练游戏、完整轮次和两条墙钟预算比例。

    Args:
        backbone_seconds (float): 八局 backbone 环境墙钟。
        tree_seconds (float): terminal Tree 环境墙钟。
        battle_seconds (float): 战斗学生 rollout 环境墙钟。
        solver_seconds (float): CombatSolver 标注墙钟。
        strategy_update_seconds (float): A100 战略更新墙钟。
        battle_update_seconds (float): A100 战斗更新墙钟。
        validation_seconds (float): 3 frozen + 1 fresh 验证墙钟。

    Raises:
        ValueError: 任一阶段墙钟不是有限非负数，或分母为零。

    Returns:
        dict[str, float | bool]: 总墙钟、Tree/训练游戏比例、验证/轮次比例和
        是否同时满足 40%/20% 红线。
    """
    values = (
        backbone_seconds,
        tree_seconds,
        battle_seconds,
        solver_seconds,
        strategy_update_seconds,
        battle_update_seconds,
        validation_seconds,
    )
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("阶段七各阶段墙钟必须是有限非负数")
    train_game = backbone_seconds + tree_seconds + battle_seconds + solver_seconds
    cycle = (
        train_game
        + strategy_update_seconds
        + battle_update_seconds
        + validation_seconds
    )
    if train_game <= 0 or cycle <= 0:
        raise ValueError("阶段七墙钟摘要缺少有效训练游戏或完整轮次时间")
    tree_ratio = tree_seconds / train_game
    validation_ratio = validation_seconds / cycle
    return {
        "train_game_seconds": train_game,
        "cycle_seconds": cycle,
        "tree_train_game_ratio": tree_ratio,
        "validation_cycle_ratio": validation_ratio,
        "within_budget": tree_ratio <= 0.4 and validation_ratio <= 0.2,
    }
