"""验证第三阶段战斗奖励方案及其已知激励边界。"""

import pytest

from play_sts2.training import rl


def test_core_reward_prioritizes_clear_hp_and_death() -> None:
    """主奖励应按入场 HP 归一，并让对应通关严格优于死亡。

    Returns:
        None: 此测试用手算值固定核心奖励组成。
    """
    cleared = rl.score_battle_reward(
        rl.BattleRewardInput(
            entry_hp=65,
            max_hp=82,
            final_hp=40,
            turns=7,
            cleared=True,
            died=False,
        ),
        scheme="core",
    )
    died = rl.score_battle_reward(
        rl.BattleRewardInput(
            entry_hp=65,
            max_hp=82,
            final_hp=0,
            turns=7,
            cleared=False,
            died=True,
        ),
        scheme="core",
    )

    assert cleared.total == pytest.approx(3 - 6 * 25 / 65 - 0.35)
    assert died.total == pytest.approx(-6 - 3 - 0.35)
    assert cleared.total > died.total


def test_no_turn_reward_is_core_without_speed_tiebreak() -> None:
    """去回合消融只能移除回合项，不得改变其他分量。

    Returns:
        None: 此测试隔离回合 shaping 的单变量差异。
    """
    inputs = rl.BattleRewardInput(
        entry_hp=65,
        max_hp=82,
        final_hp=40,
        turns=7,
        cleared=True,
        died=False,
    )

    core = rl.score_battle_reward(inputs, scheme="core")
    no_turn = rl.score_battle_reward(inputs, scheme="core_no_turn")

    assert no_turn.total - core.total == pytest.approx(0.35)


def test_boss_progress_rewards_damage_on_death_without_turn_tiebreak() -> None:
    """Boss 进度方案在正常死亡时按敌方削减给部分分，其余与去回合一致。

    Returns:
        None: 此测试固定 λ=1 的部分分、缺失数据归零和方案间边界。
    """
    died_more_damage = rl.BattleRewardInput(
        entry_hp=70,
        max_hp=75,
        final_hp=0,
        turns=45,
        cleared=False,
        died=True,
        enemy_entry_hp=321,
        enemy_final_hp=91,
    )
    died_less_damage = rl.BattleRewardInput(
        entry_hp=70,
        max_hp=75,
        final_hp=0,
        turns=45,
        cleared=False,
        died=True,
        enemy_entry_hp=321,
        enemy_final_hp=176,
    )

    more = rl.score_battle_reward(died_more_damage, scheme="core_no_turn_boss_progress")
    less = rl.score_battle_reward(died_less_damage, scheme="core_no_turn_boss_progress")
    flat = rl.score_battle_reward(died_less_damage, scheme="core_no_turn")

    # λ=1 固定：基础 -9 加上 (321-91)/321≈0.716 与 (321-176)/321≈0.452。
    assert more.total == pytest.approx(-9.0 + (321 - 91) / 321)
    assert less.total == pytest.approx(-9.0 + (321 - 176) / 321)
    assert more.total > less.total > flat.total
    components = {c.name: c.value for c in more.components}
    assert components["turns"] == 0.0
    assert components["damage_progress"] == pytest.approx((321 - 91) / 321)

    # 缺少可信敌方终局数据不得记为击杀进度。
    missing = rl.BattleRewardInput(
        entry_hp=70,
        max_hp=75,
        final_hp=0,
        turns=45,
        cleared=False,
        died=True,
        enemy_entry_hp=321,
        enemy_final_hp=None,
    )
    assert rl.score_battle_reward(
        missing, scheme="core_no_turn_boss_progress"
    ).total == pytest.approx(-9.0)

    # 胜利与模型失败不获得部分分，保持原有边界。
    victory = rl.BattleRewardInput(
        entry_hp=70,
        max_hp=75,
        final_hp=30,
        turns=40,
        cleared=True,
        died=False,
        enemy_entry_hp=321,
        enemy_final_hp=0,
    )
    assert (
        rl.score_battle_reward(victory, scheme="core_no_turn_boss_progress").total
        == rl.score_battle_reward(victory, scheme="core_no_turn").total
    )
    model_error = rl.BattleRewardInput(
        entry_hp=70,
        max_hp=75,
        final_hp=0,
        turns=3,
        cleared=False,
        died=False,
        model_error=True,
        enemy_entry_hp=321,
        enemy_final_hp=100,
    )
    assert (
        rl.score_battle_reward(model_error, scheme="core_no_turn_boss_progress").total
        == rl.score_battle_reward(model_error, scheme="core_no_turn").total
    )


def test_boss_progress_rejects_invalid_enemy_hp() -> None:
    """敌方生命事实必须非负，防止把脏数据写进奖励审计。

    Returns:
        None: 此测试固定新字段的输入契约。
    """
    with pytest.raises(ValueError, match="敌方生命值无效"):
        rl.score_battle_reward(
            rl.BattleRewardInput(
                entry_hp=70,
                max_hp=75,
                final_hp=0,
                turns=45,
                cleared=False,
                died=True,
                enemy_entry_hp=-1,
            ),
            scheme="core_no_turn_boss_progress",
        )


def test_potion_cost_penalizes_useless_consumption() -> None:
    """固定消耗成本不能像旧剩余药水项一样奖励无效喝药。

    Returns:
        None: 此测试固定药水成本的方向。
    """
    kept = rl.BattleRewardInput(
        entry_hp=65,
        max_hp=82,
        final_hp=40,
        turns=7,
        cleared=True,
        died=False,
        potions_entry=2,
        potions_used=0,
    )
    used = rl.BattleRewardInput(
        entry_hp=65,
        max_hp=82,
        final_hp=40,
        turns=7,
        cleared=True,
        died=False,
        potions_entry=2,
        potions_used=1,
    )

    kept_reward = rl.score_battle_reward(
        kept,
        scheme="potion_cost",
        potion_cost=0.25,
    )
    used_reward = rl.score_battle_reward(
        used,
        scheme="potion_cost",
        potion_cost=0.25,
    )

    assert kept_reward.total - used_reward.total == pytest.approx(0.25)


def test_legacy_remaining_potion_penalty_exposes_reward_hacking() -> None:
    """旧药水负对照应忠实显示同战果下喝药反而加分的问题。

    Returns:
        None: 此测试把旧方案保留为可检测的负对照。
    """
    common = {
        "entry_hp": 65,
        "max_hp": 82,
        "final_hp": 40,
        "turns": 7,
        "cleared": True,
        "died": False,
        "potions_entry": 2,
    }
    kept = rl.score_battle_reward(
        rl.BattleRewardInput(**common, potions_used=0),
        scheme="legacy_remaining_potion",
    )
    used = rl.score_battle_reward(
        rl.BattleRewardInput(**common, potions_used=2),
        scheme="legacy_remaining_potion",
    )

    assert used.total - kept.total == pytest.approx(3.0)


def test_model_error_is_worse_than_late_real_death() -> None:
    """首回合格式失败不能因回合较少而优于正常打到后期的死亡。

    Returns:
        None: 模型失败使用固定最差 floor，真实死亡保留回合 tie-break。
    """
    model_error = rl.score_battle_reward(
        rl.BattleRewardInput(
            entry_hp=65,
            max_hp=82,
            final_hp=40,
            turns=1,
            cleared=False,
            died=False,
            model_error=True,
        ),
        scheme="core",
    )
    late_death = rl.score_battle_reward(
        rl.BattleRewardInput(
            entry_hp=65,
            max_hp=82,
            final_hp=0,
            turns=20,
            cleared=False,
            died=True,
        ),
        scheme="core",
    )

    assert model_error.total == pytest.approx(-25.0)
    assert model_error.total < late_death.total


def test_reward_accepts_combat_max_hp_and_generated_potion_changes() -> None:
    """Feed 增加最大生命、Alchemize 生成后使用药水都属于合法战斗事实。

    Returns:
        None: 奖励只约束公式所需下界，不拿入场资源上界拒绝合法轨迹。
    """
    reward = rl.score_battle_reward(
        rl.BattleRewardInput(
            entry_hp=70,
            max_hp=70,
            final_hp=75,
            turns=6,
            cleared=True,
            died=False,
            potions_entry=0,
            potions_used=1,
        ),
        scheme="potion_cost",
        potion_cost=0.25,
    )

    assert reward.total == pytest.approx(3 + 6 * 5 / 70 - 0.3 - 0.25)
