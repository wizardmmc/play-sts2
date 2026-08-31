"""验证阶段七 Tree checkpoint 选择和墙钟预算。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_tree_selection_keeps_one_early_and_one_late_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认两节点预算应覆盖较早节点和最晚节点且优先类型多样性。

    Args:
        tmp_path (Path): 临时候选文件目录。
        monkeypatch (pytest.MonkeyPatch): 用可读入口替代磁盘原生 checkpoint。

    Returns:
        None: 选择只依赖楼层、类型和可见选项数量。
    """
    from play_sts2.training.rl.orchestration import select_tree_checkpoints, selection

    def load_checkpoint(path: Path) -> SimpleNamespace:
        """按测试路径返回不同的精确入口。

        Args:
            path (Path): 候选路径。

        Returns:
            SimpleNamespace: 具备选择器比较字段的 checkpoint。
        """
        entry = SimpleNamespace(
            audit={"marker": str(path)},
            policy_text=str(path),
            legal_actions=("a", "b"),
        )
        return SimpleNamespace(
            game_version="v0.111.0",
            mod_version="mod-test",
            entry=entry,
        )

    monkeypatch.setattr(selection, "load_strategic_checkpoint", load_checkpoint)

    source = tmp_path / "candidates.json"
    source.write_text(
        json.dumps(
            {
                "strategy_policy_version": "qwen3.5-s0",
                "battle_policy_version": "qwen3.5-b0",
                "checkpoints": [
                    {"kind": "map", "floor": 2, "option_ids": ["a", "b"], "path": "a"},
                    {
                        "kind": "event",
                        "floor": 5,
                        "option_ids": ["a", "b"],
                        "path": "b",
                    },
                    {"kind": "map", "floor": 11, "option_ids": ["a", "b"], "path": "c"},
                    {
                        "kind": "rest",
                        "floor": 15,
                        "option_ids": ["a", "b"],
                        "path": "d",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = select_tree_checkpoints(
        candidates_path=source,
        output_path=tmp_path / "tree-selection.json",
        max_checkpoints=2,
        selection_seed=7,
    )

    assert len(result["selected"]) == 2
    assert result["selected"][0]["floor"] <= 5
    assert result["selected"][1]["floor"] >= 11
    assert result["selection_seed"] == 7


def test_cycle_timing_reports_tree_and_validation_budget() -> None:
    """墙钟摘要应按文档口径计算 Tree 40% 与验证 20% 红线。

    Returns:
        None: 正常预算与超预算分别被明确标记。
    """
    from play_sts2.training.rl.orchestration import summarize_cycle_timing

    summary = summarize_cycle_timing(
        backbone_seconds=100,
        tree_seconds=40,
        battle_seconds=50,
        solver_seconds=10,
        strategy_update_seconds=5,
        battle_update_seconds=5,
        validation_seconds=20,
    )

    assert summary["tree_train_game_ratio"] == 0.2
    assert summary["validation_cycle_ratio"] < 0.2
    assert summary["within_budget"] is True

    exceeded = summarize_cycle_timing(
        backbone_seconds=10,
        tree_seconds=50,
        battle_seconds=10,
        solver_seconds=0,
        strategy_update_seconds=0,
        battle_update_seconds=0,
        validation_seconds=40,
    )
    assert exceeded["within_budget"] is False


def test_tree_selection_prefers_materialized_floor_ten_checkpoint_for_late_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """存在 floor>=10 入口时，晚期槽不能被稀有但很早的节点抢占。

    Args:
        tmp_path (Path): 候选和结果目录。
        monkeypatch (pytest.MonkeyPatch): 替换原生 checkpoint 加载。

    Returns:
        None: 第二个全局 Tree 节点来自 floor 10 以后。
    """
    from play_sts2.training.rl.orchestration import select_tree_checkpoints, selection

    def load_checkpoint(path: Path) -> SimpleNamespace:
        """按路径构造不同的 checkpoint identity。

        Args:
            path (Path): 候选路径。

        Returns:
            SimpleNamespace: 最小 checkpoint。
        """
        return SimpleNamespace(
            game_version="v0.111.0",
            mod_version="mod-test",
            entry=SimpleNamespace(
                audit={"path": str(path)},
                policy_text=str(path),
                legal_actions=("a", "b"),
            ),
        )

    monkeypatch.setattr(selection, "load_strategic_checkpoint", load_checkpoint)
    source = tmp_path / "candidates.json"
    source.write_text(
        json.dumps(
            {
                "strategy_policy_version": "qwen3.5-s0",
                "battle_policy_version": "qwen3.5-b0",
                "checkpoints": [
                    {
                        "kind": "event",
                        "floor": 1,
                        "option_ids": ["a", "b"],
                        "path": "a",
                    },
                    {"kind": "map", "floor": 1, "option_ids": ["a", "b"], "path": "b"},
                    {
                        "kind": "card_reward",
                        "floor": 2,
                        "option_ids": ["a", "b"],
                        "path": "c",
                    },
                    {
                        "kind": "card_reward",
                        "floor": 3,
                        "option_ids": ["a", "b"],
                        "path": "d",
                    },
                    {
                        "kind": "card_reward",
                        "floor": 11,
                        "option_ids": ["a", "b"],
                        "path": "e",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = select_tree_checkpoints(
        candidates_path=source,
        output_path=tmp_path / "selected.json",
        max_checkpoints=2,
        selection_seed=1,
    )

    assert result["selected"][1]["floor"] >= 10
