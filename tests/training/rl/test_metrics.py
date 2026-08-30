"""验证战斗回归指标的固定计算口径。"""

import pytest

from play_sts2.training import rl


def test_evaluate_battle_regression_metrics_counts_model_failures() -> None:
    """模型失败进入胜率、HP 和非法输出分母，基础设施失败由外层重采。

    Returns:
        None: 此测试用手算结果固定五个回归指标的微平均口径。
    """
    attempts = (
        rl.BattleEvaluationAttempt(
            outcome="cleared",
            final_hp=30,
            max_hp=40,
            turns=3,
            actions=("ACTION: use_potion 0", "ACTION: end_turn"),
            invalid_replies=0,
        ),
        rl.BattleEvaluationAttempt(
            outcome="died",
            final_hp=0,
            max_hp=40,
            turns=4,
            actions=("ACTION: end_turn",),
            invalid_replies=1,
        ),
        rl.BattleEvaluationAttempt(
            outcome="model_error",
            final_hp=30,
            max_hp=40,
            turns=2,
            actions=(),
            invalid_replies=1,
        ),
    )

    metrics = rl.evaluate_battle_regression_metrics(attempts)

    assert metrics.clear_rate == pytest.approx(1 / 3)
    assert metrics.final_hp_ratio == pytest.approx(0.25)
    assert metrics.mean_turns == pytest.approx(3.5)
    assert metrics.potion_use_rate == pytest.approx(1 / 3)
    assert metrics.invalid_output_rate == pytest.approx(0.4)
