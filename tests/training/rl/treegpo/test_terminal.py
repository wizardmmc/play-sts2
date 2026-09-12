"""验证分层 importance-corrected terminal Tree 数据合同。"""

from dataclasses import replace

import pytest


def test_terminal_tree_uses_each_branch_return_without_plan_averaging() -> None:
    """简单宏的一次幸运结果应直接形成 Monte Carlo branch advantage。

    Returns:
        None: 同动作分支不会在组内重复求均值，且使用 ``F_norm=1``。
    """
    from play_sts2.training.rl.treegpo import build_terminal_tree_group

    group = build_terminal_tree_group(
        group_id="tree-terminal",
        checkpoint_kind="event",
        checkpoint_option_ids=("event:0:A", "event:1:B", "event:2:C"),
        checkpoint_policy_text="事件入口",
        branches=(
            _branch(0, "ACTION: choose_event_option 0", 0.1, 1 / 3),
            _branch(1, "ACTION: choose_event_option 1", 1.1, 1 / 3),
            _branch(2, "ACTION: choose_event_option 2", 0.4, 1 / 3),
        ),
        sampling_mode="stratified",
    )

    assert group.returns == pytest.approx((0.1, 1.1, 0.4))
    assert group.advantages == pytest.approx((-13 / 30, 17 / 30, -4 / 30))
    assert all(branch.recompute_root_probability for branch in group.branches)


def test_terminal_tree_rejects_non_uniform_stratified_proposal() -> None:
    """简单宏强制每个根动作一条时 proposal 必须严格均匀。

    Returns:
        None: 错误的 q 在进入 A100 learner 前被拒绝。
    """
    from play_sts2.training.rl.treegpo import (
        TerminalTreeGroupRejected,
        build_terminal_tree_group,
    )

    with pytest.raises(TerminalTreeGroupRejected, match="proposal"):
        build_terminal_tree_group(
            group_id="tree-bad-q",
            checkpoint_kind="event",
            checkpoint_option_ids=("event:0:A", "event:1:B"),
            checkpoint_policy_text="事件入口",
            branches=(
                _branch(0, "ACTION: choose_event_option 0", 0.0, 0.8),
                _branch(1, "ACTION: choose_event_option 1", 1.0, 0.2),
            ),
            sampling_mode="stratified",
        )


def test_complete_plan_macros_use_policy_iid_and_keep_duplicates() -> None:
    """火堆等多步宏应采完整计划，并保留冻结策略重复采到的计划。

    Returns:
        None: 四条 iid plan 使用各自 ``q=pi_old`` 且不被去重。
    """
    from play_sts2.training.rl.treegpo import build_terminal_tree_group
    from play_sts2.training.rl.treegpo.entrypoint import _terminal_sampling_mode

    assert (
        _terminal_sampling_mode(
            "rest",
            ("ACTION: choose_rest_option 0", "ACTION: choose_rest_option 1"),
        )
        == "policy_iid"
    )
    assert (
        _terminal_sampling_mode(
            "card_reward",
            (
                "ACTION: claim_reward 0",
                "ACTION: claim_reward 1",
                "ACTION: proceed",
            ),
        )
        == "stratified"
    )
    assert (
        _terminal_sampling_mode(
            "card_reward",
            (
                "ACTION: claim_reward 0",
                "ACTION: claim_reward 1",
                "ACTION: claim_reward 2",
                "ACTION: claim_reward 3",
                "ACTION: proceed",
            ),
        )
        == "policy_iid"
    )
    repeated = "ACTION: choose_event_option 0"
    branches = tuple(
        replace(
            _branch(
                index,
                repeated if index < 2 else f"ACTION: choose_event_option {index - 1}",
                float(index),
                0.25,
            ),
            recompute_root_probability=False,
        )
        for index in range(4)
    )

    group = build_terminal_tree_group(
        group_id="tree-policy-iid",
        checkpoint_kind="rest",
        checkpoint_option_ids=tuple(f"policy-plan-{index}" for index in range(4)),
        checkpoint_policy_text="事件入口",
        branches=branches,
        sampling_mode="policy_iid",
    )

    assert [branch.plan_id for branch in group.branches].count(repeated) == 2
    assert all(
        branch.proposal_probability == pytest.approx(0.25) for branch in group.branches
    )


def _branch(index: int, action: str, value: float, proposal: float):
    """构造一个到终局的最小 Tree branch。

    Args:
        index (int): 分支序号。
        action (str): 被强制或采样的入口动作。
        value (float): 完整战略回报。
        proposal (float): 入口 plan 的采样概率。

    Returns:
        TerminalTreeBranch: 测试分支。
    """
    from play_sts2.inference import ChatMessage
    from play_sts2.runtime import DecisionGenerationProfile
    from play_sts2.training.rl.strategy import (
        StrategicReturn,
        StrategicReturnComponent,
        TreeRolloutStep,
    )
    from play_sts2.training.rl.treegpo import TerminalTreeBranch

    return TerminalTreeBranch(
        arm_index=index,
        worker_id=f"worker-{index}",
        strategy_policy_version="qwen3.5-s0",
        battle_policy_version="qwen3.5-b0",
        generation_profile=DecisionGenerationProfile(
            max_tokens=128,
            temperature=0.8,
            max_retries=0,
            thinking_enabled=False,
        ),
        plan_id=action,
        plan_step_count=1,
        steps=(
            TreeRolloutStep(
                index=0,
                messages=(
                    ChatMessage("system", "战略系统"),
                    ChatMessage("user", "事件入口"),
                ),
                reply_text=action,
                action=action,
                token_ids=(101 + index, 102),
                behavior_logprobs=(0.0, 0.0),
                response_choices=(
                    "ACTION: choose_event_option 0",
                    "ACTION: choose_event_option 1",
                    "ACTION: choose_event_option 2",
                ),
                finish_reason="stop",
            ),
        ),
        horizon_reason="victory" if index == 1 else "died",
        continuation_return=StrategicReturn(
            scheme="terminal_strategy",
            total=value,
            components=(StrategicReturnComponent("test", value),),
        ),
        proposal_probability=proposal,
        recompute_root_probability=True,
        elapsed_seconds=1.0,
    )
