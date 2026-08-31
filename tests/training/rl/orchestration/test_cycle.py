"""验证阶段七 policy pair 的固定更新与晋升顺序。"""

from pathlib import Path

import pytest


def test_cycle_requires_battle_refresh_before_validation() -> None:
    """战略更新后不能绕过战斗刷新直接验证不完整 policy pair。

    Returns:
        None: 状态机只接受文档冻结的阶段顺序。
    """
    from play_sts2.training.rl.orchestration import CycleJournal, CyclePhase

    journal = CycleJournal.new(
        cycle_id="cycle-000",
        strategy_policy="qwen3.5-s0",
        battle_policy="qwen3.5-b0",
        train_seed="ABCDEF1234",
        last_promoted_strategy_policy="qwen3.5-s0",
        last_promoted_battle_policy="qwen3.5-b0",
    )
    journal = journal.advance(CyclePhase.BACKBONE_COLLECTED)
    journal = journal.advance(CyclePhase.TREE_COLLECTED)
    journal = journal.advance(
        CyclePhase.STRATEGY_UPDATED,
        strategy_policy="qwen3.5-s1",
    )

    with pytest.raises(ValueError, match="battle"):
        journal.advance(CyclePhase.VALIDATED)

    refreshed = journal.advance(
        CyclePhase.BATTLE_UPDATED,
        battle_policy="qwen3.5-b1",
    )
    validated = refreshed.advance(CyclePhase.VALIDATED)

    assert validated.strategy_policy == "qwen3.5-s1"
    assert validated.battle_policy == "qwen3.5-b1"


def test_cycle_persists_provisional_pair_without_promotion(tmp_path: Path) -> None:
    """未运行 3+1 的轮次必须落盘为 provisional 而不是停在 prose。

    Args:
        tmp_path (Path): journal 输出目录。

    Returns:
        None: 新双 policy 可供下一轮工程训练，但没有晋升收据。
    """
    from play_sts2.training.rl.orchestration import (
        CycleJournal,
        CyclePhase,
        load_cycle_journal,
        write_cycle_journal,
    )

    journal = CycleJournal.new(
        cycle_id="cycle-001",
        strategy_policy="qwen3.5-s1",
        battle_policy="qwen3.5-b1",
        train_seed="ABCDEF1234",
        last_promoted_strategy_policy="qwen3.5-s0",
        last_promoted_battle_policy="qwen3.5-b0",
    )
    journal = journal.advance(CyclePhase.BACKBONE_COLLECTED)
    journal = journal.advance(CyclePhase.TREE_COLLECTED)
    journal = journal.advance(
        CyclePhase.STRATEGY_UPDATED,
        strategy_policy="qwen3.5-s2",
    )
    journal = journal.advance(
        CyclePhase.BATTLE_UPDATED,
        battle_policy="qwen3.5-b2",
    ).advance(CyclePhase.PROVISIONAL)
    path = write_cycle_journal(journal, tmp_path / "cycle.json")

    restored = load_cycle_journal(path)

    assert restored.phase is CyclePhase.PROVISIONAL
    assert restored.strategy_policy == "qwen3.5-s2"
    assert restored.battle_policy == "qwen3.5-b2"
    assert restored.promotion_receipt is None


def test_promotion_requires_three_validations_and_all_hard_gates() -> None:
    """原子 pair 不能只凭一份平均楼层报告晋升。

    Returns:
        None: paired、rolling、战斗、Tree、非法动作和药水门槛缺一不可。
    """
    from play_sts2.training.rl.orchestration import build_promotion_receipt

    parent = _evaluation_report("qwen3.5-s0", "qwen3.5-b0", mean_floor=8.0)
    candidates = tuple(
        _evaluation_report("qwen3.5-s1", "qwen3.5-b1", mean_floor=9.0) for _ in range(3)
    )

    receipt = build_promotion_receipt(
        parent_report=parent,
        rolling_candidate_reports=candidates,
        battle_regression_passed=True,
        branch_regret_passed=True,
        illegal_action_passed=True,
        potion_guard_passed=True,
    )
    rejected = build_promotion_receipt(
        parent_report=parent,
        rolling_candidate_reports=candidates,
        battle_regression_passed=False,
        branch_regret_passed=True,
        illegal_action_passed=True,
        potion_guard_passed=True,
    )

    assert receipt["approved"] is True
    assert receipt["parent_strategy_policy"] == "qwen3.5-s0"
    assert receipt["parent_battle_policy"] == "qwen3.5-b0"
    assert rejected["approved"] is False


def test_cycle_rejects_handwritten_incomplete_promotion_receipt() -> None:
    """journal 不能只凭手写的 approved 与双 policy 字段晋升。

    Returns:
        None: receipt 缺少三次验证、paired seeds 或 gates 时被拒绝。
    """
    from play_sts2.training.rl.orchestration import CycleJournal, CyclePhase

    journal = CycleJournal.new(
        cycle_id="cycle-002",
        strategy_policy="qwen3.5-s1",
        battle_policy="qwen3.5-b1",
        train_seed="ABCDEF1234",
        last_promoted_strategy_policy="qwen3.5-s0",
        last_promoted_battle_policy="qwen3.5-b0",
    )
    for phase in (
        CyclePhase.BACKBONE_COLLECTED,
        CyclePhase.TREE_COLLECTED,
    ):
        journal = journal.advance(phase)
    journal = journal.advance(
        CyclePhase.STRATEGY_UPDATED,
        strategy_policy="qwen3.5-s2",
    ).advance(CyclePhase.BATTLE_UPDATED, battle_policy="qwen3.5-b2")
    journal = journal.advance(CyclePhase.VALIDATED)

    with pytest.raises(ValueError, match="promotion"):
        journal.advance(
            CyclePhase.PROMOTED,
            promotion_receipt={
                "approved": True,
                "strategy_policy": "qwen3.5-s2",
                "battle_policy": "qwen3.5-b2",
            },
        )


def _evaluation_report(
    strategy: str,
    battle: str,
    *,
    mean_floor: float,
) -> dict[str, object]:
    """构造 promotion 使用的最小 3+1 报告。

    Args:
        strategy (str): 战略 policy。
        battle (str): 战斗 policy。
        mean_floor (float): 四局平均楼层。

    Returns:
        dict[str, object]: 具有核心指标的阶段七报告。
    """
    return {
        "format": "stage7_policy_evaluation",
        "strategy_policy": strategy,
        "battle_policy": battle,
        "victories": 0,
        "bosses_cleared": 0,
        "mean_floor": mean_floor,
        "frozen": [
            {
                "seed": f"FROZEN{index}",
                "victory": False,
                "bosses_cleared": 0,
                "final_floor": int(mean_floor),
            }
            for index in range(3)
        ],
    }
