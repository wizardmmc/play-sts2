"""把在线战斗结果投影为包含行为概率的可训练 rollout。"""

import copy
import math
from collections.abc import Mapping
from typing import Any

from ...harness import format_action
from ...runtime import BattleOutcome, BattleResult
from ...scenario import BattleSnapshot
from .contracts import (
    BattleReward,
    BattleRollout,
    BattleRolloutStep,
    RewardComponent,
    RolloutContractError,
)

_CLEAR_REWARD = 3.0
_HP_DELTA_SCALE = 6.0
_DEATH_PENALTY = -3.0
_TURN_PENALTY = -0.05


def build_battle_rollout(
    *,
    arm_index: int,
    worker_id: str,
    entry_snapshot: BattleSnapshot,
    entry_state: Mapping[str, Any],
    result: BattleResult,
    behavior_logprobs_mode: str,
) -> BattleRollout:
    """把一场正常离场战斗转换为含行为概率的训练 arm。

    Args:
        arm_index (int): arm 在当前 group 内的零基序号。
        worker_id (str): 执行战斗的本地游戏 worker。
        entry_snapshot (BattleSnapshot): reset 后已经核对的统一入口。
        entry_state (Mapping[str, Any]): 战斗开始时的完整游戏状态。
        result (BattleResult): Runtime 返回的正常战斗结果。
        behavior_logprobs_mode (str): 服务端声明的行为概率计算模式。

    Raises:
        RolloutContractError: 战斗没有动作、发生模型重试、缺少模型版本，
            或 token ID 与 behavior log-prob 不完整。

    Returns:
        BattleRollout: 可进入同状态 group 准入检查的完整 arm。
    """
    if not result.steps:
        raise RolloutContractError("战斗结果没有任何模型动作")

    policy_versions: set[str] = set()
    rollout_steps: list[BattleRolloutStep] = []
    for index, step in enumerate(result.steps):
        if step.retry_errors or len(step.replies) != 1:
            raise RolloutContractError("训练 rollout 不允许混入动作内模型重试")
        reply = step.reply
        if not reply.model:
            raise RolloutContractError("模型响应缺少可绑定的 policy version")
        if not reply.token_ids or len(reply.token_ids) != len(reply.behavior_logprobs):
            raise RolloutContractError("模型响应缺少对齐的 token ID 与 log-prob")
        if any(not math.isfinite(value) for value in reply.behavior_logprobs):
            raise RolloutContractError("模型响应包含非有限 behavior log-prob")
        policy_versions.add(reply.model)
        raw_after_state = step.action_result.get("state")
        after_state = (
            copy.deepcopy(dict(raw_after_state))
            if isinstance(raw_after_state, Mapping)
            else {}
        )
        rollout_steps.append(
            BattleRolloutStep(
                index=index,
                before_state=copy.deepcopy(step.before_state),
                after_state=after_state,
                messages=step.messages,
                reply_text=reply.text,
                action=format_action(step.action),
                token_ids=reply.token_ids,
                behavior_logprobs=reply.behavior_logprobs,
            )
        )
    if len(policy_versions) != 1:
        raise RolloutContractError("同一战斗混入了不同 policy version")

    return BattleRollout(
        arm_index=arm_index,
        worker_id=worker_id,
        policy_version=policy_versions.pop(),
        behavior_logprobs_mode=behavior_logprobs_mode,
        entry_snapshot=entry_snapshot,
        steps=tuple(rollout_steps),
        outcome=result.outcome.value,
        final_state=copy.deepcopy(result.final_state),
        reward=score_battle_reward_v0(entry_state, result),
    )


def score_battle_reward_v0(
    entry_state: Mapping[str, Any],
    result: BattleResult,
) -> BattleReward:
    """计算不含药水项的首轮可审计战斗奖励。

    该奖励只用于验证同入口 group 数据链可以产生方差；正式训练仍需与前人
    v2 药水项做消融，不能把这里的权重当作最终结论。

    Args:
        entry_state (Mapping[str, Any]): reset 后的战斗入口状态。
        result (BattleResult): 已正常离场的战斗结果。

    Raises:
        RolloutContractError: 入口或终局缺少有效生命值，或步骤缺少回合数。

    Returns:
        BattleReward: 通关、相对 HP、死亡和回合四个分量的总和。
    """
    entry_hp = _current_hp(entry_state, "战斗入口")
    final_hp = _current_hp(result.final_state, "战斗终局")
    hp_delta_ratio = max(-1.0, min(0.25, (final_hp - entry_hp) / entry_hp))
    turns = _turn_count(result)
    components = (
        RewardComponent(
            name="clear",
            value=_CLEAR_REWARD if result.outcome is BattleOutcome.CLEARED else 0.0,
        ),
        RewardComponent(name="hp_delta", value=_HP_DELTA_SCALE * hp_delta_ratio),
        RewardComponent(
            name="death",
            value=_DEATH_PENALTY if result.outcome is BattleOutcome.DIED else 0.0,
        ),
        RewardComponent(name="turns", value=_TURN_PENALTY * turns),
    )
    return BattleReward(
        scheme="battle-v0",
        total=sum(component.value for component in components),
        components=components,
    )


def _current_hp(state: Mapping[str, Any], label: str) -> int:
    """读取战斗奖励所需的正整数或零生命值。

    Args:
        state (Mapping[str, Any]): 待读取的完整游戏状态。
        label (str): 用于错误信息的状态阶段。

    Raises:
        RolloutContractError: 状态缺少非负整数生命值。

    Returns:
        int: 当前生命值。
    """
    run = state.get("run")
    value = run.get("current_hp") if isinstance(run, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RolloutContractError(f"{label}缺少有效 current_hp")
    if label == "战斗入口" and value == 0:
        raise RolloutContractError("战斗入口 current_hp 必须大于零")
    return value


def _turn_count(result: BattleResult) -> int:
    """从每个动作前状态取得本场经历的最大回合数。

    Args:
        result (BattleResult): 含完整 DecisionStep 的战斗结果。

    Raises:
        RolloutContractError: 任一步骤缺少正整数回合。

    Returns:
        int: 本场战斗中出现的最大回合编号。
    """
    turns: list[int] = []
    for step in result.steps:
        value = step.before_state.get("turn")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RolloutContractError("战斗动作前状态缺少有效 turn")
        turns.append(value)
    if not turns:
        raise RolloutContractError("战斗结果没有可计算回合数的动作")
    return max(turns)
