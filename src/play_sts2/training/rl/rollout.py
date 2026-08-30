"""把在线战斗结果投影为包含行为概率的可训练 rollout。"""

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

from ...harness import (
    build_observation,
    format_action,
    legal_action_lines,
    system_prompt,
)
from ...inference import ChatMessage
from ...runtime import (
    BattleOutcome,
    BattlePolicyFailure,
    BattleResult,
    DecisionGenerationProfile,
    DecisionStep,
)
from ...scenario import BattleSnapshot
from .contracts import (
    ACTION_CONSTRAINT_MODE,
    BattleReward,
    BattleRollout,
    BattleRolloutStep,
    RolloutContractError,
)
from .reward import BattleRewardInput, BattleRewardScheme, score_battle_reward


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

    rollout_steps, policy_versions, generation_profile = _project_successful_steps(
        result.steps
    )
    if len(policy_versions) != 1:
        raise RolloutContractError("同一战斗混入了不同 policy version")

    return BattleRollout(
        arm_index=arm_index,
        worker_id=worker_id,
        policy_version=policy_versions.pop(),
        behavior_logprobs_mode=behavior_logprobs_mode,
        action_constraint_mode=ACTION_CONSTRAINT_MODE,
        generation_profile=generation_profile,
        entry_snapshot=entry_snapshot,
        steps=tuple(rollout_steps),
        outcome=result.outcome.value,
        final_state=copy.deepcopy(result.final_state),
        reward=score_battle_result(entry_state, result),
    )


def build_battle_failure_rollout(
    *,
    arm_index: int,
    worker_id: str,
    entry_snapshot: BattleSnapshot,
    entry_state: Mapping[str, Any],
    failure: BattlePolicyFailure,
    behavior_logprobs_mode: str,
) -> BattleRollout:
    """把模型可归因的中止轨迹转换为死亡等价 GRPO arm。

    成功执行的前缀步骤照常保留；格式失败或截断回复若带完整 token 元数据，则作为
    未执行的失败步骤加入。动作上限没有额外回复，但其已有前缀仍接受最差终局信用。

    Args:
        arm_index (int): arm 在当前 group 内的序号。
        worker_id (str): 执行本条采样的游戏 worker。
        entry_snapshot (BattleSnapshot): 已核对的统一入口。
        entry_state (Mapping[str, Any]): 战斗入口状态。
        failure (BattlePolicyFailure): Runtime 保存的策略失败事实。
        behavior_logprobs_mode (str): 行为概率模式。

    Raises:
        RolloutContractError: 失败轨迹没有可训练 token 或缺少统一策略事实。

    Returns:
        BattleRollout: outcome 为 ``model_error``、奖励按死亡等价计算的 arm。
    """
    rollout_steps, policy_versions, completed_profile = _project_successful_steps(
        failure.steps,
        allow_empty=True,
    )
    generation_profile = failure.generation_profile or completed_profile
    if generation_profile is None:
        raise RolloutContractError("模型失败 rollout 缺少生成参数")
    if failure.replies:
        if len(failure.replies) != 1:
            raise RolloutContractError("训练 rollout 不允许混入动作内模型重试")
        reply = failure.replies[0]
        _validate_reply(reply)
        if not reply.model:
            raise RolloutContractError("模型失败响应缺少 policy version")
        policy_versions.add(reply.model)
        state = copy.deepcopy(failure.failure_state)
        observation = build_observation(state)
        choices = legal_action_lines(state)
        rollout_steps.append(
            BattleRolloutStep(
                index=len(rollout_steps),
                before_state=state,
                after_state=copy.deepcopy(state),
                messages=(
                    ChatMessage(
                        role="system",
                        content=system_prompt(observation.layer, state),
                    ),
                    ChatMessage(role="user", content=observation.text),
                ),
                reply_text=reply.text,
                action="",
                token_ids=reply.token_ids,
                behavior_logprobs=reply.behavior_logprobs,
                response_choices=choices,
                finish_reason=reply.finish_reason or failure.kind,
                is_model_failure=True,
            )
        )
    if not rollout_steps or len(policy_versions) != 1:
        raise RolloutContractError("模型失败 rollout 没有可训练 token 或统一 policy")
    if completed_profile is not None and completed_profile != generation_profile:
        raise RolloutContractError("模型失败 rollout 混入不同生成参数")
    turns = max(
        _state_turn(failure.failure_state),
        *(_state_turn(step.before_state) for step in rollout_steps),
    )
    run = entry_state.get("run")
    potions = run.get("potions") if isinstance(run, Mapping) else None
    potions_entry = sum(
        isinstance(potion, Mapping)
        and (potion.get("occupied") is True or potion.get("potion_id") is not None)
        for potion in potions or ()
    )
    reward = score_battle_reward(
        BattleRewardInput(
            entry_hp=_current_hp(entry_state, "战斗入口"),
            max_hp=_max_hp(entry_state, "战斗入口"),
            final_hp=0,
            turns=turns,
            cleared=False,
            died=False,
            model_error=True,
            potions_entry=potions_entry,
            potions_used=sum(
                step.action.startswith("ACTION: use_potion ") for step in rollout_steps
            ),
        ),
        scheme="core",
    )
    return BattleRollout(
        arm_index=arm_index,
        worker_id=worker_id,
        policy_version=policy_versions.pop(),
        behavior_logprobs_mode=behavior_logprobs_mode,
        action_constraint_mode=ACTION_CONSTRAINT_MODE,
        generation_profile=generation_profile,
        entry_snapshot=entry_snapshot,
        steps=tuple(rollout_steps),
        outcome="model_error",
        final_state=copy.deepcopy(failure.failure_state),
        reward=reward,
        invalid_replies=len(failure.replies),
    )


def _project_successful_steps(
    steps: Sequence[DecisionStep],
    *,
    allow_empty: bool = False,
) -> tuple[list[BattleRolloutStep], set[str], DecisionGenerationProfile | None]:
    """投影已成功执行的 Runtime 决策步骤。

    Args:
        steps (Sequence[DecisionStep]): 按执行顺序排列的成功步骤。
        allow_empty (bool): 策略可能在首步失败时是否允许空前缀。

    Raises:
        RolloutContractError: 重试、约束、token、策略或生成参数无效。

    Returns:
        tuple[list[BattleRolloutStep], set[str], DecisionGenerationProfile | None]:
            已投影步骤、policy 集合与统一生成参数。
    """
    if not steps and not allow_empty:
        raise RolloutContractError("战斗结果没有任何模型动作")
    generation_profiles = {step.generation_profile for step in steps}
    if None in generation_profiles or len(generation_profiles) > 1:
        raise RolloutContractError("训练 rollout 缺少统一的实际生成参数记录")
    generation_profile = next(iter(generation_profiles), None)
    policy_versions: set[str] = set()
    rollout_steps: list[BattleRolloutStep] = []
    for index, step in enumerate(steps):
        if step.retry_errors or len(step.replies) != 1:
            raise RolloutContractError("训练 rollout 不允许混入动作内模型重试")
        expected_choices = legal_action_lines(step.before_state)
        if not step.response_choices or step.response_choices != expected_choices:
            raise RolloutContractError("训练 rollout 缺少实际结构化动作约束记录")
        reply = step.reply
        _validate_reply(reply)
        if not reply.model:
            raise RolloutContractError("模型响应缺少可绑定的 policy version")
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
                response_choices=step.response_choices,
                finish_reason=reply.finish_reason or "",
            )
        )
    return rollout_steps, policy_versions, generation_profile


def _validate_reply(reply: Any) -> None:
    """核对一个训练回复的 token 元数据。

    Args:
        reply (Any): ModelReply 或兼容测试替身。

    Raises:
        RolloutContractError: token、行为概率或 finish reason 无效。
    """
    if not reply.token_ids or len(reply.token_ids) != len(reply.behavior_logprobs):
        raise RolloutContractError("模型响应缺少对齐的 token ID 与 log-prob")
    if any(not math.isfinite(value) for value in reply.behavior_logprobs):
        raise RolloutContractError("模型响应包含非有限 behavior log-prob")
    if not reply.finish_reason:
        raise RolloutContractError("模型响应缺少 finish reason")


def _state_turn(state: Mapping[str, Any]) -> int:
    """读取失败奖励使用的正回合数。

    Args:
        state (Mapping[str, Any]): 动作前游戏状态。

    Raises:
        RolloutContractError: turn 不是正整数。

    Returns:
        int: 当前回合。
    """
    turn = state.get("turn")
    if isinstance(turn, bool) or not isinstance(turn, int) or turn < 1:
        raise RolloutContractError("模型失败状态缺少有效回合")
    return turn


def score_battle_result(
    entry_state: Mapping[str, Any],
    result: BattleResult,
    *,
    scheme: BattleRewardScheme = "core",
    potion_cost: float = 0.25,
) -> BattleReward:
    """计算不含药水项的可审计战斗奖励。

    该奖励只用于验证同入口 group 数据链可以产生方差；正式训练仍需与旧药水项
    做消融，不能把这里的权重当作最终结论。

    Args:
        entry_state (Mapping[str, Any]): reset 后的战斗入口状态。
        result (BattleResult): 已正常离场的战斗结果。
        scheme (BattleRewardScheme): 待计算的奖励方案。
        potion_cost (float): 每瓶药水固定成本。

    Raises:
        RolloutContractError: 入口或终局缺少有效生命值，或步骤缺少回合数。

    Returns:
        BattleReward: 通关、相对 HP、死亡和回合四个分量的总和。
    """
    entry_hp = _current_hp(entry_state, "战斗入口")
    final_hp = _current_hp(result.final_state, "战斗终局")
    turns = _turn_count(result)
    run = entry_state.get("run")
    potions = run.get("potions") if isinstance(run, Mapping) else None
    potions_entry = sum(
        isinstance(potion, Mapping)
        and (potion.get("occupied") is True or potion.get("potion_id") is not None)
        for potion in potions or ()
    )
    potions_used = sum(step.action.name == "use_potion" for step in result.steps)
    return score_battle_reward(
        BattleRewardInput(
            entry_hp=entry_hp,
            max_hp=_max_hp(entry_state, "战斗入口"),
            final_hp=final_hp,
            turns=turns,
            cleared=result.outcome is BattleOutcome.CLEARED,
            died=result.outcome is BattleOutcome.DIED,
            potions_entry=potions_entry,
            potions_used=potions_used,
        ),
        scheme=scheme,
        potion_cost=potion_cost,
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


def _max_hp(state: Mapping[str, Any], label: str) -> int:
    """读取战斗奖励所需的正整数最大生命。

    Args:
        state (Mapping[str, Any]): 待读取的完整游戏状态。
        label (str): 用于错误信息的状态阶段。

    Raises:
        RolloutContractError: 状态缺少正整数最大生命。

    Returns:
        int: 当前最大生命值。
    """
    run = state.get("run")
    value = run.get("max_hp") if isinstance(run, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RolloutContractError(f"{label}缺少有效 max_hp")
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
