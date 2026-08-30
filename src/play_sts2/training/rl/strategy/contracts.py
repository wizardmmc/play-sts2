"""定义宏 checkpoint Tree-GRPO rollout 与八臂组契约。"""

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ....inference import ChatMessage
from ....runtime import DecisionGenerationProfile
from ..contracts import ACTION_CONSTRAINT_MODE, BEHAVIOR_LOGPROBS_MODE


class TreeGroupRejected(ValueError):
    """表示 Tree rollout group 不满足同 checkpoint 相对学习条件。"""


@dataclass(frozen=True, slots=True)
class StrategicReturnComponent:
    """保存一个战略 continuation return 分量。

    Args:
        name (str): 分量名称。
        value (float): 对总 return 的贡献。
    """

    name: str
    value: float


@dataclass(frozen=True, slots=True)
class StrategicReturn:
    """保存一个战略 suffix 的标量 return 与审计分量。

    Args:
        scheme (str): 工程或正式奖励方案名称。
        total (float): 组内相对 advantage 使用的标量。
        components (tuple[StrategicReturnComponent, ...]): 可审计分量。
    """

    scheme: str
    total: float
    components: tuple[StrategicReturnComponent, ...]


@dataclass(frozen=True, slots=True)
class TreeRolloutStep:
    """保存一个由战略 policy 生成的可训练动作。

    Args:
        index (int): 当前 arm 内的战略步骤序号。
        messages (tuple[ChatMessage, ...]): 实际发送的 stateless system/user 消息。
        reply_text (str): 模型原始规范回复。
        action (str): 已执行规范动作。
        token_ids (tuple[int, ...]): assistant completion token。
        behavior_logprobs (tuple[float, ...]): processed 旧策略 log-prob。
        response_choices (tuple[str, ...]): 实际 xgrammar 完整候选。
        finish_reason (str): 服务端生成终止原因。
    """

    index: int
    messages: tuple[ChatMessage, ...]
    reply_text: str
    action: str
    token_ids: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    response_choices: tuple[str, ...]
    finish_reason: str


@dataclass(frozen=True, slots=True)
class TreeRolloutArm:
    """保存同一 checkpoint 下的一条完整宏 suffix。

    Args:
        arm_index (int): arm 序号。
        worker_id (str): 本地游戏 worker。
        strategy_policy_version (str): 冻结战略 policy。
        battle_policy_version (str): 冻结战斗 policy。
        behavior_logprobs_mode (str): 行为概率模式。
        action_constraint_mode (str): 结构化动作约束模式。
        generation_profile (DecisionGenerationProfile): 战略生成参数。
        max_macro_checkpoints (int): 统一的后继宏节点 horizon 预算。
        plan_id (str): 首个完整宏计划的规范动作轨迹。
        plan_step_count (int): ``steps`` 中属于首个宏计划的动作数。
        macro_successor_text (str): 首个宏计划完成后的玩家可见语义状态。
        steps (tuple[TreeRolloutStep, ...]): 仅战略 policy 的可训练步骤。
        final_state (Mapping[str, Any]): horizon 结束玩家可见状态。
        horizon_reason (str): 截断或终局原因。
        continuation_return (StrategicReturn): 当前 suffix return。
        elapsed_seconds (float): 当前 arm 游戏时间。
    """

    arm_index: int
    worker_id: str
    strategy_policy_version: str
    battle_policy_version: str
    behavior_logprobs_mode: str
    action_constraint_mode: str
    generation_profile: DecisionGenerationProfile
    max_macro_checkpoints: int
    plan_id: str
    plan_step_count: int
    macro_successor_text: str
    steps: tuple[TreeRolloutStep, ...]
    final_state: Mapping[str, Any]
    horizon_reason: str
    continuation_return: StrategicReturn
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class TreeRolloutGroup:
    """保存一个同 checkpoint、总预算 K=8 的兄弟 suffix 组。

    Args:
        group_id (str): 组标识。
        checkpoint_kind (str): 宏节点类型。
        checkpoint_option_ids (tuple[str, ...]): 入口语义候选。
        checkpoint_policy_text (str): 冻结入口 observation。
        strategy_policy_version (str): 统一战略 policy。
        battle_policy_version (str): 统一战斗 policy。
        generation_profile (DecisionGenerationProfile): 统一战略生成参数。
        max_macro_checkpoints (int): 统一的后继宏节点 horizon 预算。
        arms (tuple[TreeRolloutArm, ...]): 八条 suffix。
        returns (tuple[float, ...]): arm 标量 return。
        advantages (tuple[float, ...]): 组内标准化 branch advantage。
        plan_counts (tuple[tuple[str, int], ...]): 首计划重复覆盖。
        successor_counts (tuple[tuple[str, int], ...]): 语义后继去重后的代表计划与计数。
        successor_returns (tuple[tuple[str, float], ...]): 各语义后继的平均 continuation return。
    """

    group_id: str
    checkpoint_kind: str
    checkpoint_option_ids: tuple[str, ...]
    checkpoint_policy_text: str
    strategy_policy_version: str
    battle_policy_version: str
    generation_profile: DecisionGenerationProfile
    max_macro_checkpoints: int
    arms: tuple[TreeRolloutArm, ...]
    returns: tuple[float, ...]
    advantages: tuple[float, ...]
    plan_counts: tuple[tuple[str, int], ...]
    successor_counts: tuple[tuple[str, int], ...]
    successor_returns: tuple[tuple[str, float], ...]


def build_tree_rollout_group(
    *,
    group_id: str,
    checkpoint_kind: str,
    checkpoint_option_ids: Sequence[str],
    checkpoint_policy_text: str,
    arms: Sequence[TreeRolloutArm],
    expected_size: int,
) -> TreeRolloutGroup:
    """执行同 checkpoint、冻结 policy、计划覆盖和 return 方差准入。

    Args:
        group_id (str): 当前 Tree group 标识。
        checkpoint_kind (str): 宏 checkpoint 类型。
        checkpoint_option_ids (Sequence[str]): 入口语义候选。
        checkpoint_policy_text (str): 入口玩家可见 observation。
        arms (Sequence[TreeRolloutArm]): 已完成 suffix。
        expected_size (int): 固定总轨迹预算。

    Raises:
        TreeGroupRejected: K、policy、token、计划覆盖或 return 不满足训练条件。

    Returns:
        TreeRolloutGroup: 可进入 Tree learner 的八臂组。
    """
    ordered = tuple(sorted(arms, key=lambda arm: arm.arm_index))
    if expected_size != 8 or len(ordered) != expected_size:
        raise TreeGroupRejected("Tree-GRPO 必须使用总预算 K=8")
    if tuple(arm.arm_index for arm in ordered) != tuple(range(8)):
        raise TreeGroupRejected("Tree arm_index 必须连续且唯一")
    strategy_policies = {arm.strategy_policy_version for arm in ordered}
    if len(strategy_policies) != 1 or not ordered[0].strategy_policy_version:
        raise TreeGroupRejected("Tree group 混入不同 strategy policy")
    battle_policies = {arm.battle_policy_version for arm in ordered}
    if len(battle_policies) != 1 or not ordered[0].battle_policy_version:
        raise TreeGroupRejected("Tree group 混入不同 battle policy")
    profiles = {arm.generation_profile for arm in ordered}
    if len(profiles) != 1:
        raise TreeGroupRejected("Tree group 混入不同战略生成参数")
    horizons = {arm.max_macro_checkpoints for arm in ordered}
    if len(horizons) != 1 or ordered[0].max_macro_checkpoints <= 0:
        raise TreeGroupRejected("Tree group 混入不同或无效 horizon 配置")
    if any(
        arm.behavior_logprobs_mode != BEHAVIOR_LOGPROBS_MODE
        or arm.action_constraint_mode != ACTION_CONSTRAINT_MODE
        for arm in ordered
    ):
        raise TreeGroupRejected("Tree group 缺少 processed log-prob 或 xgrammar 约束")
    for arm in ordered:
        _validate_arm(arm)
    first_step = ordered[0].steps[0]
    if first_step.messages[1].content != checkpoint_policy_text or any(
        arm.steps[0].messages != first_step.messages
        or arm.steps[0].response_choices != first_step.response_choices
        for arm in ordered[1:]
    ):
        raise TreeGroupRejected("Tree group 的玩家可见入口或首步候选不一致")
    plan_counter = Counter(arm.plan_id for arm in ordered)
    successor_arms: dict[str, list[TreeRolloutArm]] = {}
    for arm in ordered:
        successor_arms.setdefault(arm.macro_successor_text, []).append(arm)
    successor_counts = tuple(
        (group[0].plan_id, len(group)) for group in successor_arms.values()
    )
    if len(successor_counts) < 2:
        raise TreeGroupRejected("Tree group 没有探索至少两个不同宏计划")
    repeated_plans = sum(count >= 2 for _plan, count in successor_counts)
    if len(checkpoint_option_ids) <= 4 and repeated_plans < 2:
        raise TreeGroupRejected("小动作域缺少两个重复覆盖的宏计划")
    returns = tuple(arm.continuation_return.total for arm in ordered)
    successor_means = {
        successor: statistics.fmean(
            arm.continuation_return.total for arm in successor_group
        )
        for successor, successor_group in successor_arms.items()
    }
    branch_values = tuple(successor_means.values())
    mean = statistics.fmean(branch_values)
    std = statistics.pstdev(branch_values)
    if std == 0.0:
        raise TreeGroupRejected("Tree group 的计划平均 continuation return 没有方差")
    advantages = tuple(
        (successor_means[arm.macro_successor_text] - mean) / std for arm in ordered
    )
    successor_returns = tuple(
        (successor_group[0].plan_id, successor_means[successor])
        for successor, successor_group in successor_arms.items()
    )
    return TreeRolloutGroup(
        group_id=group_id,
        checkpoint_kind=checkpoint_kind,
        checkpoint_option_ids=tuple(checkpoint_option_ids),
        checkpoint_policy_text=checkpoint_policy_text,
        strategy_policy_version=ordered[0].strategy_policy_version,
        battle_policy_version=ordered[0].battle_policy_version,
        generation_profile=ordered[0].generation_profile,
        max_macro_checkpoints=ordered[0].max_macro_checkpoints,
        arms=ordered,
        returns=returns,
        advantages=advantages,
        plan_counts=tuple(plan_counter.items()),
        successor_counts=successor_counts,
        successor_returns=successor_returns,
    )


def _validate_arm(arm: TreeRolloutArm) -> None:
    """核对一条 Tree arm 的战略 token 与 return 事实。

    Args:
        arm (TreeRolloutArm): 待准入 suffix。

    Raises:
        TreeGroupRejected: 步骤、token、动作或 return 无效。

    Returns:
        None: arm 可训练时返回。
    """
    if (
        not arm.steps
        or arm.max_macro_checkpoints <= 0
        or not arm.plan_id
        or arm.plan_step_count <= 0
        or arm.plan_step_count > len(arm.steps)
        or not arm.macro_successor_text
        or not arm.horizon_reason
        or not math.isfinite(arm.continuation_return.total)
        or not math.isfinite(arm.elapsed_seconds)
        or arm.elapsed_seconds < 0
    ):
        raise TreeGroupRejected("Tree arm 缺少步骤、计划、horizon 或有限 return")
    if tuple(step.index for step in arm.steps) != tuple(range(len(arm.steps))):
        raise TreeGroupRejected("Tree step index 必须连续")
    if (
        "\n".join(step.action for step in arm.steps[: arm.plan_step_count])
        != arm.plan_id
    ):
        raise TreeGroupRejected("Tree plan_id 必须等于首个完整宏动作轨迹")
    for step in arm.steps:
        if (
            tuple(message.role for message in step.messages) != ("system", "user")
            or not step.token_ids
            or len(step.token_ids) != len(step.behavior_logprobs)
            or any(
                not math.isfinite(value) or value > 0
                for value in step.behavior_logprobs
            )
            or step.reply_text != step.action
            or step.action not in step.response_choices
            or not step.finish_reason
        ):
            raise TreeGroupRejected("Tree step 缺少 stateless token 行为策略事实")
