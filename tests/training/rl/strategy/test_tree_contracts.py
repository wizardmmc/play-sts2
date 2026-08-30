"""验证 Tree-GRPO 八臂组、策略冻结与 branch advantage。"""

from dataclasses import replace

import pytest


def test_tree_group_computes_branch_advantage_from_eight_suffixes() -> None:
    """K=8 组应按 continuation return 计算相对优势并保留计划重复数。

    Returns:
        None: 两个计划各四条 suffix，优势均值为零且存在方差。
    """
    from play_sts2.training.rl.strategy.contracts import build_tree_rollout_group

    arms = tuple(
        _arm(
            index,
            "ACTION: choose_map_node 0" if index < 4 else "ACTION: choose_map_node 1",
            1.0 if index < 4 else 2.0,
        )
        for index in range(8)
    )

    group = build_tree_rollout_group(
        group_id="tree-test",
        checkpoint_kind="map",
        checkpoint_option_ids=("map:2:1", "map:2:3"),
        checkpoint_policy_text="地图入口",
        arms=arms,
        expected_size=8,
    )

    assert group.plan_counts == (
        ("ACTION: choose_map_node 0", 4),
        ("ACTION: choose_map_node 1", 4),
    )
    assert group.returns == pytest.approx((1.0,) * 4 + (2.0,) * 4)
    assert group.advantages == pytest.approx((-1.0,) * 4 + (1.0,) * 4)
    assert sum(group.advantages) == pytest.approx(0.0)


def test_tree_group_rejects_policy_drift_and_singleton_plan() -> None:
    """兄弟 arm 不能混 policy，且小动作域的已观察计划必须重复。

    Returns:
        None: 两种无效组分别在进入 learner 前被拒绝。
    """
    from play_sts2.training.rl.strategy.contracts import (
        TreeGroupRejected,
        build_tree_rollout_group,
    )

    arms = (
        _arm(0, "ACTION: choose_map_node 1", 1.0),
        *(
            _arm(index, "ACTION: choose_map_node 0", 1.0 + index)
            for index in range(1, 8)
        ),
    )
    with pytest.raises(TreeGroupRejected, match="重复覆盖"):
        build_tree_rollout_group(
            group_id="tree-singleton",
            checkpoint_kind="map",
            checkpoint_option_ids=("map:2:1", "map:2:3"),
            checkpoint_policy_text="地图入口",
            arms=arms,
            expected_size=8,
        )

    drifted = (*arms[:7], replace(arms[7], strategy_policy_version="other"))
    with pytest.raises(TreeGroupRejected, match="strategy policy"):
        build_tree_rollout_group(
            group_id="tree-drift",
            checkpoint_kind="map",
            checkpoint_option_ids=("map:2:1", "map:2:3"),
            checkpoint_policy_text="地图入口",
            arms=drifted,
            expected_size=8,
        )


def test_tree_group_keeps_tail_sample_when_two_plans_repeat() -> None:
    """两个计划已有重复覆盖时不应因第三个探索尾样本丢弃整组。

    Returns:
        None: 4+3+1 仍保留完整 K=8 和三个计划的真实样本。
    """
    from play_sts2.training.rl.strategy.contracts import build_tree_rollout_group

    plans = (
        *("ACTION: choose_map_node 0",) * 4,
        *("ACTION: choose_map_node 1",) * 3,
        "ACTION: choose_map_node 2",
    )
    group = build_tree_rollout_group(
        group_id="tree-tail-sample",
        checkpoint_kind="map",
        checkpoint_option_ids=("map:2:0", "map:2:1", "map:2:2"),
        checkpoint_policy_text="地图入口",
        arms=tuple(_arm(index, plan, float(index)) for index, plan in enumerate(plans)),
        expected_size=8,
    )

    assert group.plan_counts == (
        ("ACTION: choose_map_node 0", 4),
        ("ACTION: choose_map_node 1", 3),
        ("ACTION: choose_map_node 2", 1),
    )


def test_tree_group_rejects_different_visible_entry() -> None:
    """同 checkpoint 的第一步消息和完整候选必须逐 arm 一致。

    Returns:
        None: 玩家可见入口漂移在 collector 准入边界被拒绝。
    """
    from play_sts2.inference import ChatMessage
    from play_sts2.training.rl.strategy.contracts import (
        TreeGroupRejected,
        build_tree_rollout_group,
    )

    arms = tuple(
        _arm(
            index,
            "ACTION: choose_map_node 0" if index < 4 else "ACTION: choose_map_node 1",
            float(index),
        )
        for index in range(8)
    )
    first_step = arms[7].steps[0]
    drifted_step = replace(
        first_step,
        messages=(
            ChatMessage("system", "战略系统"),
            ChatMessage("user", "另一个地图入口"),
        ),
    )
    drifted = (*arms[:7], replace(arms[7], steps=(drifted_step,)))

    with pytest.raises(TreeGroupRejected, match="玩家可见入口"):
        build_tree_rollout_group(
            group_id="tree-entry-drift",
            checkpoint_kind="map",
            checkpoint_option_ids=("map:2:1", "map:2:3"),
            checkpoint_policy_text="地图入口",
            arms=drifted,
            expected_size=8,
        )


def test_tree_group_rejects_horizon_drift() -> None:
    """兄弟 arm 必须共享同一后继宏节点 horizon 配置。

    Returns:
        None: 自然终局可以不同，但配置预算漂移会整组拒绝。
    """
    from play_sts2.training.rl.strategy.contracts import (
        TreeGroupRejected,
        build_tree_rollout_group,
    )

    arms = tuple(
        _arm(
            index,
            "ACTION: choose_map_node 0" if index < 4 else "ACTION: choose_map_node 1",
            float(index),
        )
        for index in range(8)
    )
    drifted = (*arms[:7], replace(arms[7], max_macro_checkpoints=3))

    with pytest.raises(TreeGroupRejected, match="horizon"):
        build_tree_rollout_group(
            group_id="tree-horizon-drift",
            checkpoint_kind="map",
            checkpoint_option_ids=("map:2:1", "map:2:3"),
            checkpoint_policy_text="地图入口",
            arms=drifted,
            expected_size=8,
        )


def test_tree_group_averages_repeated_suffixes_before_advantage() -> None:
    """同一语义计划的幸运与倒霉 suffix 应先平均成一个 Q 值。

    Returns:
        None: 同计划 arms 获得相同 branch advantage。
    """
    from play_sts2.training.rl.strategy.contracts import build_tree_rollout_group

    values = (0.0, 2.0, 0.0, 2.0, 2.0, 4.0, 2.0, 4.0)
    arms = tuple(
        _arm(
            index,
            "ACTION: choose_map_node 0" if index < 4 else "ACTION: choose_map_node 1",
            value,
        )
        for index, value in enumerate(values)
    )

    group = build_tree_rollout_group(
        group_id="tree-plan-mean",
        checkpoint_kind="map",
        checkpoint_option_ids=("map:2:1", "map:2:3"),
        checkpoint_policy_text="地图入口",
        arms=arms,
        expected_size=8,
    )

    assert group.successor_returns == (
        ("ACTION: choose_map_node 0", 1.0),
        ("ACTION: choose_map_node 1", 3.0),
    )
    assert group.advantages == pytest.approx((-1.0,) * 4 + (1.0,) * 4)


def _arm(index: int, plan: str, value: float):
    """构造一个含真实 token 元数据的最小战略 suffix arm。

    Args:
        index (int): arm 序号。
        plan (str): 首个宏计划动作。
        value (float): continuation return。

    Returns:
        TreeRolloutArm: 可进入组准入的测试 arm。
    """
    from play_sts2.inference import ChatMessage
    from play_sts2.runtime import DecisionGenerationProfile
    from play_sts2.training.rl.strategy.contracts import (
        StrategicReturn,
        StrategicReturnComponent,
        TreeRolloutArm,
        TreeRolloutStep,
    )

    return TreeRolloutArm(
        arm_index=index,
        worker_id=f"worker-{index % 2}",
        strategy_policy_version="e6-feasibility",
        battle_policy_version="e6-feasibility",
        behavior_logprobs_mode="processed_logprobs",
        action_constraint_mode="vllm_structured_choice",
        generation_profile=DecisionGenerationProfile(
            max_tokens=128,
            temperature=0.8,
            max_retries=0,
            thinking_enabled=False,
        ),
        max_macro_checkpoints=2,
        plan_id=plan,
        plan_step_count=1,
        macro_successor_text=f"successor:{plan}",
        steps=(
            TreeRolloutStep(
                index=0,
                messages=(
                    ChatMessage("system", "战略系统"),
                    ChatMessage("user", "地图入口"),
                ),
                reply_text=plan,
                action=plan,
                token_ids=(100 + index, 200),
                behavior_logprobs=(-0.2, -0.1),
                response_choices=(
                    "ACTION: choose_map_node 0",
                    "ACTION: choose_map_node 1",
                    "ACTION: choose_map_node 2",
                ),
                finish_reason="stop",
            ),
        ),
        final_state={"run": {"floor": 3, "current_hp": 60, "max_hp": 75}},
        horizon_reason="next_macro_checkpoint",
        continuation_return=StrategicReturn(
            scheme="engineering_terminal_milestone",
            total=value,
            components=(StrategicReturnComponent("test", value),),
        ),
        elapsed_seconds=1.0,
    )
