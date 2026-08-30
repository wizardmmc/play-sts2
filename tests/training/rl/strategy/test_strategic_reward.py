"""验证 Tree-GRPO 工程 milestone return。"""

import pytest


def test_engineering_return_uses_only_terminal_milestones_and_hp_tiebreak() -> None:
    """工程奖励只能包含真实进度、Boss、终局与极小 HP tie-break。

    Returns:
        None: 手工卡牌强度或错误权重会改变手算总值。
    """
    from play_sts2.training.rl.strategy.reward import (
        StrategicReturnInput,
        score_engineering_milestone_return,
    )

    reward = score_engineering_milestone_return(
        StrategicReturnInput(
            entry_floor=2,
            final_floor=7,
            final_hp=40,
            max_hp=80,
            bosses_cleared=1,
            victory=False,
            died=False,
        )
    )

    assert reward.scheme == "engineering_terminal_milestone"
    assert reward.total == pytest.approx(1.0505)
    assert [(item.name, item.value) for item in reward.components] == pytest.approx(
        [
            ("victory", 0.0),
            ("boss_milestones", 1.0),
            ("floor_progress", 0.05),
            ("hp_tiebreak", 0.0005),
        ]
    )


def test_engineering_return_keeps_victory_dominant() -> None:
    """终局胜利必须比单幕进度和 HP tie-break 更重要。

    Returns:
        None: 胜利分量固定为工程方案中的最高层级。
    """
    from play_sts2.training.rl.strategy.reward import (
        StrategicReturnInput,
        score_engineering_milestone_return,
    )

    reward = score_engineering_milestone_return(
        StrategicReturnInput(
            entry_floor=45,
            final_floor=50,
            final_hp=1,
            max_hp=80,
            bosses_cleared=0,
            victory=True,
            died=False,
        )
    )

    assert reward.total > 4.0
