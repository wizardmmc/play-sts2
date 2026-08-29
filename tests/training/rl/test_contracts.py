"""验证战斗 rollout 与同入口 GRPO group 数据契约。"""

import importlib
from dataclasses import replace

import pytest

from play_sts2.harness import HarnessAction, HarnessLayer, Observation
from play_sts2.inference import ChatMessage, ModelReply
from play_sts2.runtime import BattleOutcome, BattleResult, DecisionStep
from play_sts2.scenario import BattleScenario, BattleSnapshot, ModelInputSnapshot


def test_build_battle_group_computes_population_relative_advantages() -> None:
    """八条同入口 rollout 形成按总体标准差归一的相对优势。

    Raises:
        AssertionError: group 遗漏 policy、入口或使用了错误的标准差定义。

    Returns:
        None: 此测试验证一个可训练 group 的成功路径。
    """
    rl = importlib.import_module("play_sts2.training.rl")
    snapshot = _snapshot()
    rollouts = tuple(
        _rollout(
            rl,
            arm_index=index,
            snapshot=snapshot,
            action="ACTION: end_turn" if index % 2 == 0 else "ACTION: play_card 0",
            reward=1.0 if index % 2 == 0 else 3.0,
        )
        for index in range(8)
    )

    group = rl.build_battle_rollout_group(
        group_id="battle-demo-001",
        scenario=_scenario(),
        rollouts=rollouts,
        expected_size=8,
    )

    assert group.policy_version == "policy-e3-r1"
    assert group.behavior_logprobs_mode == "processed_logprobs"
    assert group.entry_snapshot == snapshot
    assert group.reward_mean == 2.0
    assert group.reward_std == 1.0
    assert group.advantages == (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)


def test_build_battle_rollout_preserves_policy_facts_and_scores_v0_reward() -> None:
    """Runtime 结果转换为含动作前后状态和行为概率的训练 arm。

    Raises:
        AssertionError: rollout 丢失策略版本、token 对齐、状态或奖励组成。

    Returns:
        None: 此测试验证真实 Runtime 产物到 RL 数据层的投影。
    """
    rl = importlib.import_module("play_sts2.training.rl")
    before_state = {
        "turn": 1,
        "run": {"current_hp": 40, "max_hp": 70},
    }
    after_state = {
        "turn": 2,
        "run": {"current_hp": 35, "max_hp": 70},
    }
    step = DecisionStep(
        observation=Observation(
            layer=HarnessLayer.BATTLE,
            text="当前战斗状态",
            available_actions=("end_turn",),
        ),
        before_state=before_state,
        messages=(ChatMessage(role="user", content="当前战斗状态"),),
        replies=(
            ModelReply(
                text="ACTION: end_turn",
                model="policy-e3-r1",
                token_ids=(741, 25, 1289),
                behavior_logprobs=(-0.1, -0.2, -0.3),
            ),
        ),
        retry_errors=(),
        action=HarnessAction(name="end_turn", parameters={}),
        action_result={"state": after_state, "stable": True},
    )
    result = BattleResult(
        outcome=BattleOutcome.CLEARED,
        steps=(step,),
        final_state={"run": {"current_hp": 30, "max_hp": 70}},
    )

    rollout = rl.build_battle_rollout(
        arm_index=0,
        worker_id="worker-0",
        entry_snapshot=_snapshot(),
        entry_state=before_state,
        result=result,
        behavior_logprobs_mode="processed_logprobs",
    )

    assert rollout.policy_version == "policy-e3-r1"
    assert rollout.behavior_logprobs_mode == "processed_logprobs"
    assert rollout.steps[0].action == "ACTION: end_turn"
    assert rollout.steps[0].token_ids == (741, 25, 1289)
    assert rollout.steps[0].behavior_logprobs == (-0.1, -0.2, -0.3)
    assert rollout.steps[0].before_state == before_state
    assert rollout.steps[0].after_state == after_state
    assert rollout.reward.scheme == "battle-v0"
    assert rollout.reward.total == 1.45
    assert tuple(
        (component.name, component.value) for component in rollout.reward.components
    ) == (
        ("clear", 3.0),
        ("hp_delta", -1.5),
        ("death", 0.0),
        ("turns", -0.05),
    )


def test_build_battle_group_rejects_different_entry_snapshot() -> None:
    """任一 arm 的模型入口不同都拒绝整组。

    Returns:
        None: 此测试验证同状态是 group 的硬准入条件。
    """
    rl = importlib.import_module("play_sts2.training.rl")
    snapshot = _snapshot()
    changed_snapshot = replace(
        snapshot,
        model_input=replace(snapshot.model_input, user="不同战斗状态"),
    )
    rollouts = tuple(
        _rollout(
            rl,
            arm_index=index,
            snapshot=changed_snapshot if index == 7 else snapshot,
            action="ACTION: end_turn" if index % 2 == 0 else "ACTION: play_card 0",
            reward=float(index),
        )
        for index in range(8)
    )

    with pytest.raises(rl.BattleGroupRejected, match="入口快照不一致"):
        rl.build_battle_rollout_group(
            group_id="different-entry",
            scenario=_scenario(),
            rollouts=rollouts,
            expected_size=8,
        )


def test_build_battle_group_rejects_zero_reward_variance() -> None:
    """动作有探索但回报全相同时不产生虚假的零优势更新。

    Returns:
        None: 此测试验证零方差 group 的准入门槛。
    """
    rl = importlib.import_module("play_sts2.training.rl")
    snapshot = _snapshot()
    rollouts = tuple(
        _rollout(
            rl,
            arm_index=index,
            snapshot=snapshot,
            action="ACTION: end_turn" if index % 2 == 0 else "ACTION: play_card 0",
            reward=1.0,
        )
        for index in range(8)
    )

    with pytest.raises(rl.BattleGroupRejected, match="奖励没有方差"):
        rl.build_battle_rollout_group(
            group_id="zero-variance",
            scenario=_scenario(),
            rollouts=rollouts,
            expected_size=8,
        )


def test_build_battle_group_rejects_missing_first_action_exploration() -> None:
    """回报有差异但首动作完全相同时不通过动作探索检查。

    Returns:
        None: 此测试验证 group 确实采到至少两个语义动作。
    """
    rl = importlib.import_module("play_sts2.training.rl")
    snapshot = _snapshot()
    rollouts = tuple(
        _rollout(
            rl,
            arm_index=index,
            snapshot=snapshot,
            action="ACTION: end_turn",
            reward=float(index),
        )
        for index in range(8)
    )

    with pytest.raises(rl.BattleGroupRejected, match="至少两个不同首动作"):
        rl.build_battle_rollout_group(
            group_id="same-action",
            scenario=_scenario(),
            rollouts=rollouts,
            expected_size=8,
        )


def _rollout(
    rl: object,
    *,
    arm_index: int,
    snapshot: BattleSnapshot,
    action: str,
    reward: float,
) -> object:
    """创建一条使用字面量 reward 的完整测试 rollout。

    Args:
        rl (object): 待验证的 RL 公共模块。
        arm_index (int): 当前 arm 在 group 内的编号。
        snapshot (BattleSnapshot): 所有 arm 共享的入口快照。
        action (str): 第一条规范动作。
        reward (float): 手工确定的终局回报。

    Returns:
        object: 可传给 group builder 的 rollout 实例。
    """
    step = rl.BattleRolloutStep(
        index=0,
        before_state={"turn": 1},
        after_state={"turn": 2},
        messages=(),
        reply_text=action,
        action=action,
        token_ids=(101, 102),
        behavior_logprobs=(-0.1, -0.2),
    )
    return rl.BattleRollout(
        arm_index=arm_index,
        worker_id=f"worker-{arm_index % 2}",
        policy_version="policy-e3-r1",
        behavior_logprobs_mode="processed_logprobs",
        entry_snapshot=snapshot,
        steps=(step,),
        outcome="cleared",
        final_state={"run": {"current_hp": 30}},
        reward=rl.BattleReward(
            scheme="test",
            total=reward,
            components=(rl.RewardComponent(name="terminal", value=reward),),
        ),
    )


def _scenario() -> BattleScenario:
    """创建 group 使用的合法战斗场景。

    Returns:
        BattleScenario: 固定入口的最小场景。
    """
    return BattleScenario(
        character_id="DEFECT",
        seed="ABCDEF1234",
        floor=7,
        encounter_id="CULTISTS_NORMAL",
        deck=("ZAP",),
        relics=("CRACKED_CORE",),
        current_hp=40,
        max_hp=70,
    )


def _snapshot() -> BattleSnapshot:
    """创建所有测试 arm 共用的最小入口快照。

    Returns:
        BattleSnapshot: 含模型实际输入的不可变快照。
    """
    return BattleSnapshot(
        turn=1,
        enemies=(),
        hand=(),
        model_input=ModelInputSnapshot(
            system="战斗系统",
            user="战斗状态",
            available_actions=("play_card", "end_turn"),
        ),
    )
