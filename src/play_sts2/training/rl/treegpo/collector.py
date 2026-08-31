"""并行收集二至四条分层 terminal Tree branches。"""

import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from ..strategy import TreeRolloutArm
from .contracts import (
    TerminalTreeBranch,
    TerminalTreeGroup,
    build_terminal_tree_group,
)


class TerminalTreeWorker(Protocol):
    """声明 terminal Tree collector 使用的本地 worker。"""

    worker_id: str

    def collect_arm(
        self,
        *,
        arm_index: int,
        forced_action: str | None = None,
    ) -> TreeRolloutArm:
        """从同一原生 checkpoint 续玩到终局。

        Args:
            arm_index (int): 当前分支序号。
            forced_action (str | None): simple macro 分层分配的根动作。

        Returns:
            TreeRolloutArm: 含完整战略步骤和 terminal return 的分支。
        """
        ...


class TerminalTreeCollector:
    """把 simple macro 或组合宏的终局分支分配给最多四个 workers。"""

    def __init__(self, workers: Sequence[TerminalTreeWorker]) -> None:
        """保存一至四个隔离游戏 workers。

        Args:
            workers (Sequence[TerminalTreeWorker]): 独占端口和 HOME 的 workers。

        Raises:
            ValueError: worker 数不在一至四之间。
        """
        self._workers = tuple(workers)
        if not 1 <= len(self._workers) <= 4:
            raise ValueError("terminal Tree collector 需要一至四个本地 worker")

    def collect(
        self,
        *,
        group_id: str,
        checkpoint_kind: str,
        checkpoint_policy_text: str,
        root_actions: Sequence[str],
        sampling_mode: str,
    ) -> TerminalTreeGroup:
        """收集一个宏 checkpoint 的全部 terminal branches。

        Args:
            group_id (str): 当前 group 名称。
            checkpoint_kind (str): 宏节点类型。
            checkpoint_policy_text (str): 入口玩家可见观测。
            root_actions (Sequence[str]): simple macro 的全部根动作；组合宏为空。
            sampling_mode (str): ``stratified`` 或 ``policy_iid``。

        Raises:
            ValueError: proposal 与根动作数量不符合阶段七预算。

        Returns:
            TerminalTreeGroup: 已计算单样本 branch advantage 的终局组。
        """
        if sampling_mode == "stratified":
            actions: tuple[str | None, ...] = tuple(root_actions)
            if not 2 <= len(actions) <= 4:
                raise ValueError("stratified terminal Tree 必须覆盖二至四个根动作")
        elif sampling_mode == "policy_iid":
            actions = (None,) * 4
        else:
            raise ValueError("terminal Tree sampling mode 无效")
        allocations = [
            list(range(index, len(actions), len(self._workers)))
            for index in range(len(self._workers))
        ]
        with ThreadPoolExecutor(max_workers=len(self._workers)) as executor:
            futures = [
                executor.submit(
                    _collect_batch,
                    worker,
                    indices,
                    actions,
                )
                for worker, indices in zip(self._workers, allocations, strict=True)
                if indices
            ]
            arms = tuple(arm for future in futures for arm in future.result())
        ordered = tuple(sorted(arms, key=lambda arm: arm.arm_index))
        branches = tuple(
            _terminal_branch(
                arm,
                sampling_mode=sampling_mode,
                branch_count=len(actions),
            )
            for arm in ordered
        )
        option_ids = (
            tuple(root_actions)
            if root_actions
            else tuple(f"policy-plan-{index}" for index in range(len(branches)))
        )
        return build_terminal_tree_group(
            group_id=group_id,
            checkpoint_kind=checkpoint_kind,
            checkpoint_option_ids=option_ids,
            checkpoint_policy_text=checkpoint_policy_text,
            branches=branches,
            sampling_mode=sampling_mode,  # type: ignore[arg-type]
        )


def _collect_batch(
    worker: TerminalTreeWorker,
    arm_indices: Sequence[int],
    actions: Sequence[str | None],
) -> tuple[TreeRolloutArm, ...]:
    """让一个 worker 顺序完成被分配的 terminal branches。

    Args:
        worker (TerminalTreeWorker): 当前独占游戏 worker。
        arm_indices (Sequence[int]): 被分配的分支序号。
        actions (Sequence[str | None]): 每条分支的可选强制根动作。

    Returns:
        tuple[TreeRolloutArm, ...]: 完成的原始 Tree arms。
    """
    return tuple(
        worker.collect_arm(
            arm_index=index,
            forced_action=actions[index],
        )
        for index in arm_indices
    )


def _terminal_branch(
    arm: TreeRolloutArm,
    *,
    sampling_mode: str,
    branch_count: int,
) -> TerminalTreeBranch:
    """把通用 Tree arm 投影为 proposal-aware terminal branch。

    Args:
        arm (TreeRolloutArm): 已续玩到整局终局的原始 arm。
        sampling_mode (str): ``stratified`` 或 ``policy_iid``。
        branch_count (int): 当前总分支数。

    Raises:
        ValueError: arm 没有到达胜负终局或 proposal 概率下溢。

    Returns:
        TerminalTreeBranch: 只训练入口宏计划的终局分支。
    """
    if arm.horizon_reason not in {"victory", "died", "model_error"}:
        raise ValueError("terminal Tree arm 没有到达整局终局")
    if sampling_mode == "stratified":
        proposal = 1.0 / branch_count
        recompute = True
    else:
        proposal = math.exp(
            sum(
                value
                for step in arm.steps[: arm.plan_step_count]
                for value in step.behavior_logprobs
            )
        )
        recompute = False
    if not math.isfinite(proposal) or proposal <= 0:
        raise ValueError("terminal Tree plan proposal probability 无效")
    return TerminalTreeBranch(
        arm_index=arm.arm_index,
        worker_id=arm.worker_id,
        strategy_policy_version=arm.strategy_policy_version,
        battle_policy_version=arm.battle_policy_version,
        generation_profile=arm.generation_profile,
        plan_id=arm.plan_id,
        plan_step_count=arm.plan_step_count,
        steps=arm.steps,
        horizon_reason=arm.horizon_reason,
        continuation_return=arm.continuation_return,
        proposal_probability=proposal,
        recompute_root_probability=recompute,
        elapsed_seconds=arm.elapsed_seconds,
        battle_candidates=arm.battle_candidates,
        failure_reason=arm.failure_reason,
    )
