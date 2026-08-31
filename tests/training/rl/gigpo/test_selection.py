"""验证阶段七战斗场景使用真实结果事实选择。"""

import json
from pathlib import Path


def test_select_battles_prioritizes_death_and_keeps_ordinary_random_slot(
    tmp_path: Path,
) -> None:
    """五场预算应同时覆盖死亡困难场景与一个普通随机场景。

    Args:
        tmp_path (Path): 临时候选与输出目录。

    Returns:
        None: 选择不依赖手写卡牌或遗物强度。
    """
    from play_sts2.training.rl.gigpo import select_battle_candidates

    candidates = []
    for index in range(7):
        candidates.append(
            {
                "arm_index": 0,
                "battle_index": index,
                "floor": index + 1,
                "is_elite": index in {1, 2},
                "is_boss": index == 3,
                "outcome": "died" if index == 0 else "cleared",
                "hp_loss_ratio": 1.0 if index == 0 else index / 10,
                "scenario": {"encounter_id": f"ENCOUNTER_{index}"},
                "entry_snapshot": {"turn": 1, "marker": index},
            }
        )
    source = tmp_path / "candidates.json"
    source.write_text(
        json.dumps(
            {
                "strategy_policy_version": "qwen3.5-s0",
                "battle_policy_version": "qwen3.5-b0",
                "battles": candidates,
            }
        ),
        encoding="utf-8",
    )

    result = select_battle_candidates(
        candidates_path=source,
        output_root=tmp_path / "selected",
        history_path=tmp_path / "history.json",
        max_scenarios=5,
        seed=7,
    )

    assert len(result["selected"]) == 5
    assert any(row["outcome"] == "died" for row in result["selected"])
    assert result["ordinary_selected"] >= 1
    assert result["ordinary_deficit"] == 0


def test_small_battle_budget_carries_ordinary_debt_across_cycles(
    tmp_path: Path,
) -> None:
    """单场预算不能把 20% 普通战斗约束永久四舍五入成零。

    Args:
        tmp_path (Path): 跨轮历史与五个选择目录。

    Returns:
        None: 前四轮累计欠额，第五轮会强制补入普通场景。
    """
    from play_sts2.training.rl.gigpo import select_battle_candidates

    source = tmp_path / "candidates.json"
    candidates = [
        {
            "arm_index": 0,
            "battle_index": 0,
            "floor": 1,
            "is_elite": True,
            "is_boss": False,
            "outcome": "died",
            "hp_loss_ratio": 1.0,
            "scenario": {"encounter_id": "ELITE"},
            "entry_snapshot": {"marker": "elite"},
        },
        {
            "arm_index": 0,
            "battle_index": 1,
            "floor": 2,
            "is_elite": False,
            "is_boss": False,
            "outcome": "cleared",
            "hp_loss_ratio": 0.0,
            "scenario": {"encounter_id": "NORMAL"},
            "entry_snapshot": {"marker": "ordinary"},
        },
    ]
    source.write_text(
        json.dumps(
            {
                "strategy_policy_version": "qwen3.5-s0",
                "battle_policy_version": "qwen3.5-b0",
                "battles": candidates,
            }
        ),
        encoding="utf-8",
    )
    history = tmp_path / "history.json"
    results = [
        select_battle_candidates(
            candidates_path=source,
            output_root=tmp_path / f"selected-{index}",
            history_path=history,
            max_scenarios=1,
            seed=index,
        )
        for index in range(5)
    ]

    assert all(result["ordinary_deficit"] == 1 for result in results[:4])
    assert results[4]["ordinary_selected"] == 1
    assert results[4]["ordinary_deficit"] == 0
