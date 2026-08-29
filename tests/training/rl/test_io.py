"""验证仓库示例战斗场景能够直接进入 RL collector。"""

from pathlib import Path

from play_sts2.training.rl import load_battle_scenario


def test_load_checked_in_cultists_scenario() -> None:
    """仓库内示例 JSON 应满足确定性战斗场景契约。

    Raises:
        AssertionError: 示例场景字段或数组转换结果错误。

    Returns:
        None: 此测试保证文档命令引用的场景真实可用。
    """
    repository = Path(__file__).resolve().parents[3]
    scenario = load_battle_scenario(repository / "configs/scenarios/cultists.json")

    assert scenario.seed == "ABCDEF1234"
    assert scenario.encounter_id == "CULTISTS_NORMAL"
    assert scenario.deck == (
        "ZAP",
        "DUALCAST",
        "STRIKE_DEFECTx4",
        "DEFEND_DEFECTx4",
    )
    assert scenario.potion_slots == 3
