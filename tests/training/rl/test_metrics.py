"""验证战斗回归指标的固定计算口径。"""

import json
from pathlib import Path

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


def test_evaluate_battle_rollout_file_projects_group_attempts(tmp_path: Path) -> None:
    """落盘 group 应可直接复算五项固定回归指标。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Returns:
        None: 文件投影保留终局 HP、回合和药水动作。
    """
    path = tmp_path / "group.json"
    profile = {
        "max_tokens": 128,
        "temperature": 0.8,
        "max_retries": 0,
        "thinking_enabled": False,
    }
    snapshot = {"turn": 1, "model_input": "same"}
    path.write_text(
        json.dumps(
            {
                "group_id": "group-test",
                "policy_version": "policy-test",
                "behavior_logprobs_mode": "processed_logprobs",
                "action_constraint_mode": "vllm_structured_choice",
                "generation_profile": profile,
                "entry_snapshot": snapshot,
                "environment": {
                    "game_version": "v0.111.0",
                    "structured_output_backend": "xgrammar",
                    "structured_output_version": "0.1.33",
                },
                "rollouts": [
                    {
                        "arm_index": arm_index,
                        "policy_version": "policy-test",
                        "behavior_logprobs_mode": "processed_logprobs",
                        "action_constraint_mode": "vllm_structured_choice",
                        "generation_profile": profile,
                        "entry_snapshot": snapshot,
                        "outcome": "cleared",
                        "final_state": {"run": {"current_hp": 30, "max_hp": 40}},
                        "steps": [
                            {
                                "before_state": {
                                    "turn": 2,
                                    "run": {"current_hp": 40, "max_hp": 40},
                                },
                                "action": "ACTION: use_potion 0",
                            }
                        ],
                    }
                    for arm_index in range(8)
                ],
            }
        ),
        encoding="utf-8",
    )

    report = rl.evaluate_battle_rollout_file(path)

    assert report["attempts"] == 8
    assert report["metrics"]["clear_rate"] == pytest.approx(1.0)
    assert report["metrics"]["final_hp_ratio"] == pytest.approx(0.75)
    assert report["metrics"]["potion_use_rate"] == pytest.approx(1.0)
