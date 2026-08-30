"""验证仓库示例战斗场景能够直接进入 RL collector。"""

import json
from pathlib import Path

from play_sts2.training.rl import load_battle_scenario


def test_load_battle_scenario_template() -> None:
    """仓库内场景模板应满足确定性战斗场景契约。

    Raises:
        AssertionError: 示例场景字段或数组转换结果错误。

    Returns:
        None: 此测试保证文档命令引用的场景真实可用。
    """
    repository = Path(__file__).resolve().parents[3]
    scenario = load_battle_scenario(
        repository / "configs-template/scenarios/battle.json"
    )

    assert scenario.seed == "EXAMPLE001"
    assert scenario.encounter_id == "CULTISTS_NORMAL"
    assert scenario.deck == (
        "ZAP",
        "DUALCAST",
        "STRIKE_DEFECTx4",
        "DEFEND_DEFECTx4",
    )
    assert scenario.potion_slots == 3


def test_scenario_suite_template_has_disjoint_splits() -> None:
    """示例名册应固定训练、验证和 fresh-seed 场景及回归指标。

    Raises:
        AssertionError: 名册缺少用途、场景无效、seed 重合或指标口径漂移。

    Returns:
        None: 此测试验证可提交的通用评测入口。
    """
    repository = Path(__file__).resolve().parents[3]
    suite_path = repository / "configs-template/scenarios/suite.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))

    assert suite["schema_version"] == 1
    assert suite["policy_model"] == "policy-rl-dev"
    assert suite["group_size"] == 8
    assert suite["primary_metrics"] == ["clear_rate", "final_hp_ratio"]
    assert suite["guardrail_metrics"] == [
        "mean_turns",
        "potion_use_rate",
        "invalid_output_rate",
    ]
    metric_contract = suite["metric_contract"]
    assert metric_contract["name"] == "battle-regression"
    assert metric_contract["scenario_attempts"] == 8
    assert metric_contract["split_aggregation"] == (
        "micro_after_equal_attempts_per_scenario"
    )
    assert metric_contract["infrastructure_failures"] == "resample_and_exclude"
    assert metric_contract["model_errors"] == "include_as_policy_attempt"
    assert set(metric_contract["definitions"]) == {
        "clear_rate",
        "final_hp_ratio",
        "mean_turns",
        "potion_use_rate",
        "invalid_output_rate",
    }

    entries = suite["scenarios"]
    assert {entry["split"] for entry in entries} == {
        "train",
        "validation",
        "fresh_seed",
    }
    assert len({entry["id"] for entry in entries}) == 3

    scenarios = [
        load_battle_scenario(suite_path.parent / entry["path"]) for entry in entries
    ]
    assert len({scenario.seed for scenario in scenarios}) == 3
    assert {scenario.encounter_id for scenario in scenarios} == {"CULTISTS_NORMAL"}
