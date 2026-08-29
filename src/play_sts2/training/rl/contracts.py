"""定义战斗 rollout 与同入口 GRPO group 的稳定数据契约。"""

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...inference import ChatMessage
from ...scenario import BattleScenario, BattleSnapshot

BEHAVIOR_LOGPROBS_MODE = "processed_logprobs"


class RolloutContractError(ValueError):
    """表示单条 rollout 缺少训练所需的行为策略事实。"""


class BattleGroupRejected(ValueError):
    """表示一组 rollout 不满足同状态相对学习的准入条件。"""


@dataclass(frozen=True, slots=True)
class RewardComponent:
    """保存一个可审计的战斗奖励分量。

    Args:
        name (str): 奖励分量的稳定名称。
        value (float): 该分量对总奖励的实际贡献。
    """

    name: str
    value: float


@dataclass(frozen=True, slots=True)
class BattleReward:
    """保存一条战斗 rollout 的奖励方案、总值与组成。

    Args:
        scheme (str): 奖励方案的可读稳定名称。
        total (float): 用于组内相对优势的标量奖励。
        components (tuple[RewardComponent, ...]): 可审计的奖励分量。
    """

    scheme: str
    total: float
    components: tuple[RewardComponent, ...]


@dataclass(frozen=True, slots=True)
class BattleRolloutStep:
    """保存一个已执行战斗动作及其行为策略概率。

    Args:
        index (int): 动作在本场战斗中的零基序号。
        before_state (Mapping[str, Any]): 生成观测时的完整游戏状态。
        after_state (Mapping[str, Any]): 动作响应携带的游戏状态。
        messages (tuple[ChatMessage, ...]): 本次实际发送给模型的消息。
        reply_text (str): 模型返回的原始文本。
        action (str): Harness 解析后的规范动作文本。
        token_ids (tuple[int, ...]): assistant 生成 token ID。
        behavior_logprobs (tuple[float, ...]): 与 token ID 对齐的行为策略
            log-prob。
    """

    index: int
    before_state: Mapping[str, Any]
    after_state: Mapping[str, Any]
    messages: tuple[ChatMessage, ...]
    reply_text: str
    action: str
    token_ids: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BattleRollout:
    """保存同一场景中的一条完整战斗 arm。

    Args:
        arm_index (int): arm 在 group 内的零基序号。
        worker_id (str): 执行本条采样的本地游戏 worker。
        policy_version (str): rollout 时冻结的模型版本。
        behavior_logprobs_mode (str): 行为概率是否包含温度和采样处理。
        entry_snapshot (BattleSnapshot): 模型实际看到的统一战斗入口。
        steps (tuple[BattleRolloutStep, ...]): 按执行顺序保存的动作。
        outcome (str): 战斗正常离场结果。
        final_state (Mapping[str, Any]): 离开战斗后的第一份稳定状态。
        reward (BattleReward): 本条 arm 的终局奖励。
    """

    arm_index: int
    worker_id: str
    policy_version: str
    behavior_logprobs_mode: str
    entry_snapshot: BattleSnapshot
    steps: tuple[BattleRolloutStep, ...]
    outcome: str
    final_state: Mapping[str, Any]
    reward: BattleReward

    @property
    def first_action(self) -> str:
        """返回用于检查组内探索的首个规范动作。

        Raises:
            RolloutContractError: rollout 没有任何已执行步骤。

        Returns:
            str: 本场战斗的首个规范动作。
        """
        if not self.steps:
            raise RolloutContractError("战斗 rollout 没有任何已执行动作")
        return self.steps[0].action


@dataclass(frozen=True, slots=True)
class BattleRolloutGroup:
    """保存一个已通过同状态 GRPO 准入的战斗组。

    Args:
        group_id (str): 当前组的稳定标识。
        scenario (BattleScenario): 所有 arm 共享的场景配置。
        policy_version (str): 所有 arm 共享的冻结模型版本。
        behavior_logprobs_mode (str): 所有 arm 共享的行为概率计算模式。
        entry_snapshot (BattleSnapshot): 所有 arm 共享的真实入口快照。
        rollouts (tuple[BattleRollout, ...]): 按 arm 序号排列的完整战斗。
        reward_mean (float): 组内奖励均值。
        reward_std (float): 组内奖励总体标准差。
        advantages (tuple[float, ...]): 每条 arm 的标准化相对优势。
    """

    group_id: str
    scenario: BattleScenario
    policy_version: str
    behavior_logprobs_mode: str
    entry_snapshot: BattleSnapshot
    rollouts: tuple[BattleRollout, ...]
    reward_mean: float
    reward_std: float
    advantages: tuple[float, ...]


def build_battle_rollout_group(
    *,
    group_id: str,
    scenario: BattleScenario,
    rollouts: Sequence[BattleRollout],
    expected_size: int,
) -> BattleRolloutGroup:
    """核对同入口、同策略和有效探索后计算组相对优势。

    Args:
        group_id (str): 当前组的稳定标识。
        scenario (BattleScenario): 所有 arm 请求使用的场景。
        rollouts (Sequence[BattleRollout]): 已完成的候选 arms。
        expected_size (int): 本轮采样要求的精确 arm 数。

    Raises:
        BattleGroupRejected: arm 数、入口、策略、动作或奖励方差不满足准入。
        RolloutContractError: 单条 rollout 缺少 token 级行为策略事实。

    Returns:
        BattleRolloutGroup: 可直接交给后续 GRPO loss 的组数据。
    """
    ordered = tuple(sorted(rollouts, key=lambda rollout: rollout.arm_index))
    if expected_size < 2 or len(ordered) != expected_size:
        raise BattleGroupRejected(
            f"战斗 group arm 数不一致: expected={expected_size}, actual={len(ordered)}"
        )
    if tuple(rollout.arm_index for rollout in ordered) != tuple(range(expected_size)):
        raise BattleGroupRejected("战斗 group 的 arm_index 必须连续且唯一")

    baseline = ordered[0]
    policy_versions = {rollout.policy_version for rollout in ordered}
    if len(policy_versions) != 1 or not baseline.policy_version:
        raise BattleGroupRejected("战斗 group 混入了不同或空的 policy version")
    logprobs_modes = {rollout.behavior_logprobs_mode for rollout in ordered}
    if logprobs_modes != {BEHAVIOR_LOGPROBS_MODE}:
        raise BattleGroupRejected("战斗 group 必须统一使用 processed_logprobs 行为概率")
    if any(rollout.entry_snapshot != baseline.entry_snapshot for rollout in ordered):
        raise BattleGroupRejected("战斗 group 的入口快照不一致")

    for rollout in ordered:
        _validate_rollout(rollout)
    if len({rollout.first_action for rollout in ordered}) < 2:
        raise BattleGroupRejected("战斗 group 没有探索至少两个不同首动作")

    rewards = tuple(rollout.reward.total for rollout in ordered)
    reward_mean = statistics.fmean(rewards)
    reward_std = statistics.pstdev(rewards)
    if reward_std == 0.0:
        raise BattleGroupRejected("战斗 group 的奖励没有方差")
    advantages = tuple((reward - reward_mean) / reward_std for reward in rewards)
    return BattleRolloutGroup(
        group_id=group_id,
        scenario=scenario,
        policy_version=baseline.policy_version,
        behavior_logprobs_mode=baseline.behavior_logprobs_mode,
        entry_snapshot=baseline.entry_snapshot,
        rollouts=ordered,
        reward_mean=reward_mean,
        reward_std=reward_std,
        advantages=advantages,
    )


def _validate_rollout(rollout: BattleRollout) -> None:
    """核对一条完整 rollout 的奖励与 token 训练事实。

    Args:
        rollout (BattleRollout): 待检查的单条战斗采样。

    Raises:
        RolloutContractError: 奖励非有限、步骤为空或 token 元数据不对齐。

    Returns:
        None: rollout 含有训练所需的完整行为策略事实时返回。
    """
    if not math.isfinite(rollout.reward.total):
        raise RolloutContractError("战斗 rollout 奖励必须是有限值")
    if not rollout.steps:
        raise RolloutContractError("战斗 rollout 没有任何已执行动作")
    for step in rollout.steps:
        if not step.token_ids or len(step.token_ids) != len(step.behavior_logprobs):
            raise RolloutContractError("战斗步骤的 token ID 与 log-prob 不对齐")
        if any(not math.isfinite(value) for value in step.behavior_logprobs):
            raise RolloutContractError("战斗步骤包含非有限 behavior log-prob")
