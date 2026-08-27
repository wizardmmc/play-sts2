"""验证原始轨迹的数据结构与最小落盘格式。"""

import json
from pathlib import Path

from play_sts2.recording import RunMetadata, TrajectoryWriter


def test_writer_creates_metadata_and_append_only_events(tmp_path: Path) -> None:
    """按来源和局 ID 保存元数据，并顺序追加原始事件。

    Args:
        tmp_path (Path): Pytest 提供的临时数据目录。

    Raises:
        AssertionError: 目录、元数据或事件格式不符合约定。

    Returns:
        None: 此测试仅验证轨迹 writer 的可观察结果。
    """
    metadata = RunMetadata(
        run_id="run-001",
        source="human",
        started_at="2026-08-27T08:00:00Z",
        character_id="DEFECT",
        seed="TEST-SEED",
    )
    writer = TrajectoryWriter(tmp_path, metadata)

    first = writer.append(
        "state",
        {"screen": "EVENT", "available_actions": ["choose_event_option"]},
        observed_at="2026-08-27T08:00:01Z",
    )
    second = writer.append(
        "action",
        {"action": "choose_event_option", "option_index": 0},
        observed_at="2026-08-27T08:00:02Z",
    )

    assert writer.run_dir == tmp_path / "human/run-001"
    assert json.loads((writer.run_dir / "meta.json").read_text(encoding="utf-8")) == {
        "run_id": "run-001",
        "source": "human",
        "started_at": "2026-08-27T08:00:00Z",
        "character_id": "DEFECT",
        "seed": "TEST-SEED",
    }
    events = [
        json.loads(line)
        for line in (writer.run_dir / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events == [
        {
            "sequence": 1,
            "observed_at": "2026-08-27T08:00:01Z",
            "type": "state",
            "payload": {
                "screen": "EVENT",
                "available_actions": ["choose_event_option"],
            },
        },
        {
            "sequence": 2,
            "observed_at": "2026-08-27T08:00:02Z",
            "type": "action",
            "payload": {"action": "choose_event_option", "option_index": 0},
        },
    ]
    assert first.sequence == 1
    assert second.sequence == 2


def test_writer_rejects_an_existing_run_directory(tmp_path: Path) -> None:
    """拒绝覆盖已经存在的原始轨迹。

    Args:
        tmp_path (Path): Pytest 提供的临时数据目录。

    Raises:
        AssertionError: writer 没有通过 ``FileExistsError`` 阻止覆盖。

    Returns:
        None: 此测试仅验证原始数据不会被静默覆盖。
    """
    metadata = RunMetadata(
        run_id="same-run",
        source="agent",
        started_at="2026-08-27T08:00:00Z",
    )
    TrajectoryWriter(tmp_path, metadata)

    try:
        TrajectoryWriter(tmp_path, metadata)
    except FileExistsError:
        return
    raise AssertionError("重复的局目录应触发 FileExistsError")
