"""验证人类 raw 元数据的 SFT 准入语义。"""

import json
from pathlib import Path

import pytest

from play_sts2.recording.audit import RawRunIntegrityError, audit_human_run


def test_audit_accepts_verified_samples_from_incomplete_recording(
    tmp_path: Path,
) -> None:
    """断流只关闭整局完整性，不否定已经验证的单步 SFT 样本。"""
    run_dir = tmp_path / "PARTIAL-RUN"
    (run_dir / "combat").mkdir(parents=True)
    (run_dir / "strategy").mkdir()
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "PARTIAL-RUN",
                "termination_reason": "stream_interrupted",
                "training_eligible": True,
                "recording_complete": False,
                "integrity": {
                    "samples_verified": True,
                    "ineligibility_reasons": [],
                },
                "battle_count": 0,
                "battle_sample_count": 0,
                "strategic_sample_count": 0,
            }
        ),
        encoding="utf-8",
    )

    audit = audit_human_run(run_dir)

    assert audit.metadata["training_eligible"] is True
    assert audit.metadata["recording_complete"] is False


def test_audit_rejects_mixed_run_with_wrong_action_source_counts(
    tmp_path: Path,
) -> None:
    """人机协作局的来源计数必须与所有物理动作行一致。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 审计器接受了来源数量失真的混合轨迹。

    Returns:
        None: 此测试只验证来源对账边界。
    """
    run_dir = tmp_path / "MIXED-RUN"
    (run_dir / "combat").mkdir(parents=True)
    (run_dir / "strategy").mkdir()
    (run_dir / "strategy/decisions.jsonl").write_text(
        json.dumps({"event_id": 1, "action_source": "human_ui"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "MIXED-RUN",
                "source": "human_combat_solver",
                "termination_reason": "game_over",
                "training_eligible": True,
                "recording_complete": True,
                "integrity": {
                    "samples_verified": True,
                    "ineligibility_reasons": [],
                },
                "battle_count": 0,
                "battle_sample_count": 0,
                "strategic_sample_count": 1,
                "action_source_counts": {"combat_solver": 1},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RawRunIntegrityError, match="action_source_counts"):
        audit_human_run(run_dir)
