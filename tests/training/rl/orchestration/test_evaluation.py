"""验证阶段七固定 3+1 完整游戏评测汇总。"""


def test_policy_evaluation_requires_three_frozen_and_one_fresh_seed() -> None:
    """完整验证必须明确区分三条冻结 seed 与一条当轮 fresh seed。

    Returns:
        None: 报告保留逐 seed 指标并计算四局整体结果。
    """
    from play_sts2.training.rl.orchestration import (
        PolicyRunEvaluation,
        build_policy_evaluation_report,
    )

    records = tuple(
        PolicyRunEvaluation(
            seed=seed,
            split="frozen" if index < 3 else "fresh",
            victory=index == 3,
            final_floor=10 + index,
            bosses_cleared=index // 2,
            battle_count=5 + index,
            final_hp=20 if index == 3 else 0,
            max_hp=75,
            decision_count=30,
            elapsed_seconds=10.0,
        )
        for index, seed in enumerate(
            ("AAAAAAAAAA", "BBBBBBBBBB", "CCCCCCCCCC", "DDDDDDDDDD")
        )
    )

    report = build_policy_evaluation_report(
        records,
        frozen_seeds=("AAAAAAAAAA", "BBBBBBBBBB", "CCCCCCCCCC"),
        fresh_seed="DDDDDDDDDD",
    )

    assert report["games"] == 4
    assert report["victories"] == 1
    assert report["fresh"]["victory"] is True
    assert len(report["frozen"]) == 3


def test_evaluation_tensorboard_payload_keeps_frozen_and_fresh_metrics() -> None:
    """3+1 验证曲线应区分 frozen 聚合与 fresh 单局结果。

    Returns:
        None: TensorBoard payload 不把四局只压成一个 loss。
    """
    from play_sts2.training.rl.orchestration.evaluation import (
        evaluation_tensorboard_payload,
    )

    report = {
        "victories": 1,
        "model_errors": 0,
        "mean_floor": 12.0,
        "bosses_cleared": 2,
        "frozen": [
            {"final_floor": 10, "victory": False, "bosses_cleared": 0},
            {"final_floor": 11, "victory": False, "bosses_cleared": 0},
            {"final_floor": 12, "victory": False, "bosses_cleared": 1},
        ],
        "fresh": {"final_floor": 15, "victory": True, "bosses_cleared": 1},
    }

    payload = evaluation_tensorboard_payload(report)

    assert payload["validation"]["mean_floor"] == 12.0
    assert payload["validation/frozen"]["mean_floor"] == 11.0
    assert payload["validation/fresh"]["floor"] == 15
