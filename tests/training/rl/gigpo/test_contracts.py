"""验证整局 GiGPO 的分层奖励、策略冻结与精确 anchor census。"""

from dataclasses import replace

import pytest


def test_gigpo_group_builds_episode_and_milestone_advantages() -> None:
    """八条同种子 episode 应分别计算终局与进度相对优势。

    Returns:
        None: 总优势只由本局终局和 episode-level 进度组成。
    """
    from play_sts2.training.rl.gigpo import build_gigpo_group

    episodes = tuple(
        _episode(
            index,
            victory=index == 7,
            bosses_cleared=index // 3,
            final_floor=index + 1,
        )
        for index in range(8)
    )
    audits = tuple(({"rng": {"counter": index}},) for index in range(8))

    group = build_gigpo_group(
        group_id="cycle-000-backbone",
        episodes=episodes,
        anchor_audits=audits,
        lambda_milestone=1.0,
        normalization="one",
    )

    assert len(group.episodes) == 8
    assert sum(group.episode_advantages) == pytest.approx(0.0)
    assert sum(group.milestone_advantages) == pytest.approx(0.0)
    assert group.advantages[7] > group.advantages[0]
    assert group.anchor_census.repeated_steps == 0
    assert group.anchor_census.coverage == 0.0


def test_gigpo_anchor_census_uses_exact_audit_equality() -> None:
    """自然 anchor 只能来自完整原生审计相等的决策状态。

    Returns:
        None: 相同可见动作但 RNG 不同不会被合并。
    """
    from play_sts2.training.rl.gigpo import build_gigpo_group

    episodes = tuple(_episode(index, final_floor=index + 1) for index in range(8))
    shared = {"rng": {"counter": 3}, "room": {"floor": 1}}
    audits = tuple(
        (shared if index < 2 else {"rng": {"counter": index}},) for index in range(8)
    )

    group = build_gigpo_group(
        group_id="cycle-000-anchor",
        episodes=episodes,
        anchor_audits=audits,
        lambda_milestone=1.0,
        normalization="one",
    )

    assert group.anchor_census.groups == 1
    assert group.anchor_census.repeated_steps == 2
    assert group.anchor_census.coverage == pytest.approx(0.25)
    assert group.episodes[0].steps[0].anchor_id == "anchor-000"
    assert group.episodes[1].steps[0].anchor_id == "anchor-000"
    assert group.episodes[2].steps[0].anchor_id is None


def test_gigpo_anchor_requires_same_policy_state() -> None:
    """原生审计相同但模型消息不同的步骤不能组成自然 anchor。

    Returns:
        None: 玩家可见状态或合法动作域变化会拆开 anchor。
    """
    from play_sts2.inference import ChatMessage
    from play_sts2.training.rl.gigpo import build_gigpo_group

    episodes = [_episode(index, final_floor=index + 1) for index in range(8)]
    changed = episodes[1].steps[0]
    episodes[1] = replace(
        episodes[1],
        steps=(
            replace(
                changed,
                messages=(
                    ChatMessage("system", "战略系统"),
                    ChatMessage("user", "不同的选牌页面"),
                ),
                response_choices=("ACTION: choose_card 0", "ACTION: skip"),
                action="ACTION: skip",
                reply_text="ACTION: skip",
            ),
        ),
    )
    shared = {"rng": {"counter": 3}, "room": {"floor": 1}}
    audits = tuple(
        (shared if index < 2 else {"rng": {"counter": index}},) for index in range(8)
    )

    group = build_gigpo_group(
        group_id="cycle-policy-state",
        episodes=episodes,
        anchor_audits=audits,
        lambda_milestone=1.0,
        normalization="one",
    )

    assert group.anchor_census.groups == 0
    assert group.anchor_census.repeated_steps == 0


def test_gigpo_group_rejects_policy_or_seed_drift() -> None:
    """同组八局不能混入不同种子或不同战略 residual。

    Returns:
        None: 两种漂移都在写盘前被拒绝。
    """
    from play_sts2.training.rl.gigpo import GigpoGroupRejected, build_gigpo_group

    episodes = tuple(_episode(index) for index in range(8))
    audits = tuple(({"rng": index},) for index in range(8))

    with pytest.raises(GigpoGroupRejected, match="seed"):
        build_gigpo_group(
            group_id="cycle-seed-drift",
            episodes=(*episodes[:7], replace(episodes[7], seed="ZZZZZZZZZZ")),
            anchor_audits=audits,
            lambda_milestone=1.0,
            normalization="one",
        )
    with pytest.raises(GigpoGroupRejected, match="strategy policy"):
        build_gigpo_group(
            group_id="cycle-policy-drift",
            episodes=(
                *episodes[:7],
                replace(episodes[7], strategy_policy_version="qwen3.5-other"),
            ),
            anchor_audits=audits,
            lambda_milestone=1.0,
            normalization="one",
        )


def _episode(
    index: int,
    *,
    victory: bool = False,
    bosses_cleared: int = 0,
    final_floor: int = 1,
):
    """构造一个含单步战略 token 的最小完整 episode。

    Args:
        index (int): episode 序号。
        victory (bool): 是否整局胜利。
        bosses_cleared (int): 本局击败 Boss 数。
        final_floor (int): 终局楼层。

    Returns:
        GigpoEpisode: 可进入八局组的测试 episode。
    """
    from play_sts2.inference import ChatMessage
    from play_sts2.runtime import DecisionGenerationProfile
    from play_sts2.training.rl.gigpo import GigpoEpisode, GigpoStep

    action = f"ACTION: choose_map_node {index % 2}"
    return GigpoEpisode(
        arm_index=index,
        worker_id=f"worker-{index % 4}",
        seed="ABCDEF1234",
        character_id="DEFECT",
        ascension=0,
        strategy_policy_version="qwen3.5-s0",
        battle_policy_version="qwen3.5-b0",
        generation_profile=DecisionGenerationProfile(
            max_tokens=128,
            temperature=0.8,
            max_retries=0,
            thinking_enabled=False,
        ),
        steps=(
            GigpoStep(
                index=0,
                messages=(
                    ChatMessage("system", "战略系统"),
                    ChatMessage("user", "地图入口"),
                ),
                reply_text=action,
                action=action,
                token_ids=(101 + index, 102),
                behavior_logprobs=(-0.2, -0.1),
                response_choices=(
                    "ACTION: choose_map_node 0",
                    "ACTION: choose_map_node 1",
                ),
                finish_reason="stop",
                anchor_id=None,
            ),
        ),
        victory=victory,
        final_floor=final_floor,
        final_hp=0 if not victory else 20,
        max_hp=75,
        bosses_cleared=bosses_cleared,
        battle_count=max(1, final_floor),
        elapsed_seconds=1.0,
    )
