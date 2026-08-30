"""计算固定口径的战斗回归指标。"""

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .contracts import RL_GAME_VERSION, STRUCTURED_OUTPUT_BACKEND


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


def evaluate_battle_rollout_file(path: Path) -> dict[str, Any]:
    """从一个完整 rollout group 文件复算固定战斗回归指标。

    Args:
        path (Path): collector 写出的 group JSON。

    Raises:
        TypeError: group、arm 或步骤字段类型错误。
        ValueError: group、arm、状态或步骤字段不符合评测契约。
        OSError: 文件无法读取。
        json.JSONDecodeError: 文件不是合法 JSON。

    Returns:
        dict[str, Any]: policy、环境、尝试数和五项指标报告。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("战斗评测 group 必须是 JSON 对象")
    raw_rollouts = payload.get("rollouts")
    if not isinstance(raw_rollouts, list) or len(raw_rollouts) != 8:
        raise ValueError("战斗评测 group 必须恰好包含八条 rollouts")
    _validate_evaluation_environment(payload)
    attempts = tuple(_attempt_from_rollout(rollout) for rollout in raw_rollouts)
    metrics = evaluate_battle_regression_metrics(attempts)
    return {
        "group_id": payload.get("group_id"),
        "policy_version": payload.get("policy_version"),
        "environment": payload.get("environment"),
        "attempts": len(attempts),
        "metrics": asdict(metrics),
    }


def _validate_evaluation_environment(payload: Mapping[str, Any]) -> None:
    """验证评测文件带固定游戏与 grammar 运行时收据。

    Args:
        payload (Mapping[str, Any]): group 顶层对象。

    Raises:
        ValueError: 环境收据不符合固定评测契约。
    """
    environment = payload.get("environment")
    if (
        not isinstance(environment, Mapping)
        or environment.get("game_version") != RL_GAME_VERSION
        or environment.get("structured_output_backend") != STRUCTURED_OUTPUT_BACKEND
        or not isinstance(environment.get("structured_output_version"), str)
    ):
        raise ValueError("战斗评测 group 缺少固定游戏与 structured-output 收据")


def _attempt_from_rollout(value: object) -> BattleEvaluationAttempt:
    """把一条 JSON rollout 投影为固定指标计数事实。

    Args:
        value (object): JSON 解码后的单个 arm。

    Raises:
        TypeError: arm、步骤或计数字段类型错误。
        ValueError: 终局、生命、步骤或动作字段无效。

    Returns:
        BattleEvaluationAttempt: 可交给微平均 evaluator 的尝试。
    """
    if not isinstance(value, Mapping):
        raise TypeError("战斗评测 arm 必须是对象")
    outcome = value.get("outcome")
    if outcome not in {"cleared", "died", "model_error"}:
        raise ValueError("战斗评测 arm outcome 无效")
    final_state = value.get("final_state")
    final_run = final_state.get("run") if isinstance(final_state, Mapping) else None
    raw_steps = value.get("steps")
    if (
        not isinstance(final_run, Mapping)
        or not isinstance(raw_steps, list)
        or not raw_steps
    ):
        raise ValueError("战斗评测 arm 缺少终局或步骤")
    first_state = (
        raw_steps[0].get("before_state") if isinstance(raw_steps[0], Mapping) else None
    )
    entry_run = first_state.get("run") if isinstance(first_state, Mapping) else None
    max_hp = final_run.get("max_hp")
    if not isinstance(max_hp, int) and isinstance(entry_run, Mapping):
        max_hp = entry_run.get("max_hp")
    actions = []
    turns = []
    for step in raw_steps:
        if not isinstance(step, Mapping) or not isinstance(step.get("action"), str):
            raise TypeError("战斗评测 step 动作无效")
        before_state = step.get("before_state")
        turn = before_state.get("turn") if isinstance(before_state, Mapping) else None
        if isinstance(turn, bool) or not isinstance(turn, int):
            raise TypeError("战斗评测 step 回合无效")
        turns.append(turn)
        if step.get("is_model_failure") is not True:
            actions.append(step["action"])
    final_hp = final_run.get("current_hp")
    invalid_replies = value.get("invalid_replies", 0)
    if (
        isinstance(final_hp, bool)
        or not isinstance(final_hp, int)
        or isinstance(max_hp, bool)
        or not isinstance(max_hp, int)
        or isinstance(invalid_replies, bool)
        or not isinstance(invalid_replies, int)
    ):
        raise TypeError("战斗评测 arm 计数字段无效")
    return BattleEvaluationAttempt(
        outcome=outcome,
        final_hp=final_hp,
        max_hp=max_hp,
        turns=max(turns),
        actions=tuple(actions),
        invalid_replies=invalid_replies,
    )
