"""验证人类 raw 元数据的 SFT 准入语义。"""

import json
from pathlib import Path

from play_sts2.recording.audit import audit_human_run


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
