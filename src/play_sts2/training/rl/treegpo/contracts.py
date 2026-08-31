"""定义 terminal Tree 的分层 proposal 与 Monte Carlo branch 合同。"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ....runtime import DecisionGenerationProfile
from ..strategy.contracts import StrategicReturn, TreeRolloutStep


class TerminalTreeGroupRejected(ValueError):
    """表示 terminal Tree 分支不能形成可信的相对训练组。"""


@dataclass(frozen=True, slots=True)
class TerminalTreeBranch:
    """保存从同一宏 checkpoint 续玩到终局的一条分支。

    Args:
        arm_index (int): 分支序号。
        worker_id (str): 本地游戏 worker。
        strategy_policy_version (str): 冻结战略 residual。
        battle_policy_version (str): 冻结战斗 residual。
        generation_profile (DecisionGenerationProfile): 战略采样参数。
        plan_id (str): 入口完整宏计划动作轨迹。
        plan_step_count (int): 只训练的入口计划步骤数。
        steps (tuple[TreeRolloutStep, ...]): 全部战略步骤。
        horizon_reason (str): 必须为胜利或死亡。
        continuation_return (StrategicReturn): 完整 terminal 战略回报。
        proposal_probability (float): 根计划在 proposal ``q`` 下的概率。
        recompute_root_probability (bool): learner 是否用冻结父策略重算根概率。
        elapsed_seconds (float): 当前分支墙钟。
        battle_candidates (tuple[Any, ...]): terminal continuation 暴露的战斗入口。
        failure_reason (str | None): terminal policy 失败原因。
    """

    arm_index: int
    worker_id: str
    strategy_policy_version: str
    battle_policy_version: str
    generation_profile: DecisionGenerationProfile
    plan_id: str
    plan_step_count: int
    steps: tuple[TreeRolloutStep, ...]
    horizon_reason: str
    continuation_return: StrategicReturn
    proposal_probability: float
    recompute_root_probability: bool
    elapsed_seconds: float
    battle_candidates: tuple[Any, ...] = ()
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class TerminalTreeGroup:
    """保存一个分层 terminal Tree group。

    Args:
        group_id (str): 当前 Tree group 名称。
        checkpoint_kind (str): 宏节点类型。
        checkpoint_option_ids (tuple[str, ...]): 玩家可见根选项。
        checkpoint_policy_text (str): 入口战略观测。
        sampling_mode (Literal["stratified", "policy_iid"]): proposal 类型。
        branches (tuple[TerminalTreeBranch, ...]): 二至四条 terminal 分支。
        returns (tuple[float, ...]): 每条单次 Monte Carlo 回报。
        advantages (tuple[float, ...]): ``F_norm=1`` 的组中心化优势。
    """

    group_id: str
    checkpoint_kind: str
    checkpoint_option_ids: tuple[str, ...]
    checkpoint_policy_text: str
    sampling_mode: Literal["stratified", "policy_iid"]
    branches: tuple[TerminalTreeBranch, ...]
    returns: tuple[float, ...]
    advantages: tuple[float, ...]


def build_terminal_tree_group(
    *,
    group_id: str,
    checkpoint_kind: str,
    checkpoint_option_ids: Sequence[str],
    checkpoint_policy_text: str,
    branches: Sequence[TerminalTreeBranch],
    sampling_mode: Literal["stratified", "policy_iid"],
) -> TerminalTreeGroup:
    """核对 terminal 分支并直接计算单样本 branch advantage。

    Args:
        group_id (str): 当前 Tree group 名称。
        checkpoint_kind (str): 宏节点类型。
        checkpoint_option_ids (Sequence[str]): 玩家可见根选项。
        checkpoint_policy_text (str): 入口战略观测。
        branches (Sequence[TerminalTreeBranch]): 二至四条 terminal 分支。
        sampling_mode (Literal["stratified", "policy_iid"]): proposal 类型。

    Raises:
        TerminalTreeGroupRejected: proposal、策略、入口、终局或回报无效。

    Returns:
        TerminalTreeGroup: 不做同计划平均的 terminal 分支组。
    """
    ordered = tuple(sorted(branches, key=lambda branch: branch.arm_index))
    if not group_id or not checkpoint_kind or not checkpoint_policy_text:
        raise TerminalTreeGroupRejected("terminal Tree 入口字段不能为空")
    if not 2 <= len(ordered) <= 4:
        raise TerminalTreeGroupRejected("terminal Tree 总预算必须为二至四条")
    if tuple(branch.arm_index for branch in ordered) != tuple(range(len(ordered))):
        raise TerminalTreeGroupRejected("terminal Tree arm_index 必须连续且唯一")
    if sampling_mode not in {"stratified", "policy_iid"}:
        raise TerminalTreeGroupRejected("terminal Tree sampling mode 无效")
    strategies = {branch.strategy_policy_version for branch in ordered}
    battles = {branch.battle_policy_version for branch in ordered}
    profiles = {branch.generation_profile for branch in ordered}
    if len(strategies) != 1 or not ordered[0].strategy_policy_version:
        raise TerminalTreeGroupRejected("terminal Tree 混入不同 strategy policy")
    if len(battles) != 1 or not ordered[0].battle_policy_version:
        raise TerminalTreeGroupRejected("terminal Tree 混入不同 battle policy")
    if len(profiles) != 1:
        raise TerminalTreeGroupRejected("terminal Tree 混入不同战略生成参数")
    for branch in ordered:
        _validate_branch(branch, checkpoint_policy_text)
    if sampling_mode == "stratified":
        expected = 1.0 / len(ordered)
        if len(checkpoint_option_ids) != len(ordered) or any(
            not branch.recompute_root_probability
            or not math.isclose(
                branch.proposal_probability,
                expected,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            for branch in ordered
        ):
            raise TerminalTreeGroupRejected("stratified proposal 必须逐根均匀覆盖")
    elif any(
        branch.recompute_root_probability or not 0 < branch.proposal_probability <= 1
        for branch in ordered
    ):
        raise TerminalTreeGroupRejected("policy_iid proposal 必须直接来自冻结策略")
    returns = tuple(branch.continuation_return.total for branch in ordered)
    if any(not math.isfinite(value) for value in returns):
        raise TerminalTreeGroupRejected("terminal Tree return 必须有限")
    mean = statistics.fmean(returns)
    advantages = tuple(value - mean for value in returns)
    if not any(value != 0.0 for value in advantages):
        raise TerminalTreeGroupRejected("terminal Tree return 没有方差")
    return TerminalTreeGroup(
        group_id=group_id,
        checkpoint_kind=checkpoint_kind,
        checkpoint_option_ids=tuple(checkpoint_option_ids),
        checkpoint_policy_text=checkpoint_policy_text,
        sampling_mode=sampling_mode,
        branches=ordered,
        returns=returns,
        advantages=advantages,
    )


def _validate_branch(branch: TerminalTreeBranch, checkpoint_policy_text: str) -> None:
    """核对一条 terminal branch 的入口计划与终局事实。

    Args:
        branch (TerminalTreeBranch): 待核对分支。
        checkpoint_policy_text (str): 全组共享入口观测。

    Raises:
        TerminalTreeGroupRejected: 分支不是完整终局或缺少 token 事实。

    Returns:
        None: 分支可进入 group 时返回。
    """
    if (
        branch.horizon_reason not in {"victory", "died", "model_error"}
        or (branch.horizon_reason == "model_error")
        != (branch.failure_reason is not None)
        or not branch.steps
        or not 0 < branch.plan_step_count <= len(branch.steps)
        or "\n".join(step.action for step in branch.steps[: branch.plan_step_count])
        != branch.plan_id
        or not math.isfinite(branch.elapsed_seconds)
        or branch.elapsed_seconds < 0
    ):
        raise TerminalTreeGroupRejected("terminal Tree branch 未完整到达终局")
    first = branch.steps[0]
    if (
        len(first.messages) != 2
        or first.messages[1].content != checkpoint_policy_text
        or not first.response_choices
    ):
        raise TerminalTreeGroupRejected("terminal Tree branch 入口观测不一致")
    for step in branch.steps:
        if (
            not step.token_ids
            or len(step.token_ids) != len(step.behavior_logprobs)
            or any(
                not math.isfinite(value) or value > 0
                for value in step.behavior_logprobs
            )
            or step.reply_text != step.action
            or step.action not in step.response_choices
        ):
            raise TerminalTreeGroupRejected("terminal Tree branch token 事实无效")
