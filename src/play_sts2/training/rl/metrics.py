"""计算固定口径的战斗回归指标。"""

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class BattleEvaluationAttempt:
    """保存一次冻结 policy 战斗评测的计数事实。

    基础设施故障不构成该对象，而由调用方重采。同一次战斗若因模型输出无法继续，
    必须以 ``model_error`` 保存，不能从评测分母删除。

    Args:
        outcome (Literal["cleared", "died", "model_error"]): 战斗离场或模型中止。
        final_hp (int): 正常离场或模型中止时观测到的当前生命。
        max_hp (int): 本次战斗的最大生命。
        turns (int): 正常离场或模型中止时达到的回合数。
        actions (tuple[str, ...]): 已成功解析并执行的规范动作。
        invalid_replies (int): 解析、动作域或执行校验失败的模型回复数。
    """

    outcome: Literal["cleared", "died", "model_error"]
    final_hp: int
    max_hp: int
    turns: int
    actions: tuple[str, ...]
    invalid_replies: int


@dataclass(frozen=True, slots=True)
class BattleRegressionMetrics:
    """保存固定战斗回归口径的五个微平均指标。

    Args:
        clear_rate (float): 通关次数除以全部 policy 战斗尝试数。
        final_hp_ratio (float): 每次尝试终局 HP 比例的均值；模型中止记零。
        mean_turns (float): 正常通关或死亡战斗的回合数均值。
        potion_use_rate (float): ``use_potion`` 动作数除以全部有效动作数。
        invalid_output_rate (float): 非法回复数除以全部有效和非法模型回复数。
    """

    clear_rate: float
    final_hp_ratio: float
    mean_turns: float
    potion_use_rate: float
    invalid_output_rate: float


def evaluate_battle_regression_metrics(
    attempts: Sequence[BattleEvaluationAttempt],
) -> BattleRegressionMetrics:
    """按固定战斗回归口径微平均一组战斗尝试。

    每个 split 中每个 scenario 必须先采相同数量的 policy 尝试，再把全部尝试交给
    本函数。基础设施故障重采且不进入分母；模型中止保留在分母中。若没有正常
    通关或死亡，``mean_turns`` 记为零；若没有有效动作或模型回复，对应动作级比率
    也记为零。

    Args:
        attempts (Sequence[BattleEvaluationAttempt]): 同一 split 的全部 policy 尝试。

    Raises:
        ValueError: 尝试为空，或任一计数字段不满足战斗评测边界。

    Returns:
        BattleRegressionMetrics: 五个固定口径的战斗回归指标。
    """
    if not attempts:
        raise ValueError("战斗回归指标至少需要一次 policy 尝试")

    for attempt in attempts:
        if attempt.outcome not in {"cleared", "died", "model_error"}:
            raise ValueError(f"未知战斗评测结果: {attempt.outcome}")
        if attempt.max_hp <= 0 or not 0 <= attempt.final_hp <= attempt.max_hp:
            raise ValueError("战斗评测生命值超出有效边界")
        if attempt.turns < 0 or attempt.invalid_replies < 0:
            raise ValueError("战斗评测计数不能为负数")

    attempt_count = len(attempts)
    completed = tuple(
        attempt for attempt in attempts if attempt.outcome in {"cleared", "died"}
    )
    valid_action_count = sum(len(attempt.actions) for attempt in attempts)
    invalid_reply_count = sum(attempt.invalid_replies for attempt in attempts)
    potion_action_count = sum(
        action.startswith("ACTION: use_potion ")
        for attempt in attempts
        for action in attempt.actions
    )
    reply_count = valid_action_count + invalid_reply_count
    return BattleRegressionMetrics(
        clear_rate=(
            sum(attempt.outcome == "cleared" for attempt in attempts) / attempt_count
        ),
        final_hp_ratio=(
            sum(
                attempt.final_hp / attempt.max_hp
                if attempt.outcome != "model_error"
                else 0.0
                for attempt in attempts
            )
            / attempt_count
        ),
        mean_turns=(
            statistics.fmean(attempt.turns for attempt in completed)
            if completed
            else 0.0
        ),
        potion_use_rate=(
            potion_action_count / valid_action_count if valid_action_count else 0.0
        ),
        invalid_output_rate=(invalid_reply_count / reply_count if reply_count else 0.0),
    )
