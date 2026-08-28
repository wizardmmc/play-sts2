"""验证原始轨迹的数据结构与最小落盘格式。"""

import json
from pathlib import Path

from play_sts2.recording import HumanRunWriter, RunMetadata


def test_human_writer_groups_one_jsonl_per_battle_and_updates_metadata(
    tmp_path: Path,
) -> None:
    """按战斗分片精确动作，并持续保存难度与样本计数。

    Args:
        tmp_path (Path): Pytest 提供的临时数据目录。

    Raises:
        AssertionError: 战斗分片、原始字段或元数据统计不符合契约。

    Returns:
        None: 此测试只验证新的人类样本目录格式。
    """
    metadata = RunMetadata(
        run_id="RUN-NEW",
        source="human",
        started_at="2026-08-27T08:00:00Z",
        character_id="DEFECT",
        seed="RUN-NEW",
        ascension=3,
    )
    writer = HumanRunWriter(tmp_path, metadata)
    writer.append_decision(
        _decision(
            event_id=11,
            layer="battle",
            screen="COMBAT",
            floor=2,
            action="end_turn",
        )
    )
    writer.end_battle()
    writer.append_decision(
        _decision(
            event_id=13,
            layer="battle",
            screen="COMBAT",
            floor=3,
            action="end_turn",
        )
    )
    writer.append_decision(
        _decision(
            event_id=12,
            layer="strategic",
            screen="MAP",
            floor=2,
            action="choose_map_node",
            option_index=0,
        )
    )
    writer.finalize("game_over", completed_at="2026-08-27T08:05:00Z")

    battle_files = sorted((writer.run_dir / "combat").glob("*.jsonl"))
    assert [path.name for path in battle_files] == [
        "battle-f002-01.jsonl",
        "battle-f003-02.jsonl",
    ]
    assert [
        len(path.read_text(encoding="utf-8").splitlines()) for path in battle_files
    ] == [
        1,
        1,
    ]
    strategy_path = writer.run_dir / "strategy/decisions.jsonl"
    assert len(strategy_path.read_text(encoding="utf-8").splitlines()) == 1
    assert writer.run_dir == tmp_path / "human/20260827-a3-f3-RUN-NEW"
    assert not (writer.run_dir / "readable").exists()
    assert not (writer.run_dir / "events.jsonl").exists()
    first_battle = json.loads(battle_files[0].read_text(encoding="utf-8"))
    assert set(first_battle) == {
        "event_id",
        "observed_at",
        "before_state",
        "action",
        "parameters",
    }
    saved = json.loads((writer.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert saved["ascension"] == 3
    assert saved["battle_sample_count"] == 2
    assert saved["strategic_sample_count"] == 1
    assert saved["battle_count"] == 2
    assert saved["max_floor_reached"] == 3
    assert saved["schema_version"] == 2
    assert saved["termination_reason"] == "game_over"
    assert saved["completed_at"] == "2026-08-27T08:05:00Z"
    assert saved["training_eligible"] is True
    assert saved["recording_complete"] is True
    assert saved["integrity"] == {
        "samples_verified": True,
        "ineligibility_reasons": [],
    }


def test_human_writer_marks_integrity_failures_as_training_ineligible(
    tmp_path: Path,
) -> None:
    """采集缺口保留在元数据中，并阻止整局进入训练。

    Args:
        tmp_path (Path): Pytest 提供的临时数据目录。

    Raises:
        AssertionError: 完整性失败未被持久化或局仍可训练。

    Returns:
        None: 此测试只检查元数据准入状态。
    """
    writer = HumanRunWriter(
        tmp_path,
        RunMetadata(
            run_id="RUN-GAP",
            source="human",
            started_at="2026-08-27T08:00:00Z",
            character_id="DEFECT",
            seed="RUN-GAP",
            ascension=1,
        ),
    )

    writer.record_integrity_failure("native_ui_capture_gap: play_card: no match")
    writer.record_integrity_failure("native_ui_capture_gap: play_card: no match")
    writer.finalize("game_over", completed_at="2026-08-27T08:05:00Z")

    saved = json.loads((writer.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert saved["training_eligible"] is False
    assert saved["recording_complete"] is False
    assert saved["integrity"] == {
        "samples_verified": False,
        "ineligibility_reasons": ["native_ui_capture_gap: play_card: no match"],
    }


def test_human_writer_preserves_human_correction_identity(tmp_path: Path) -> None:
    """保留人工确认修正的稳定字符串 ID 与最小来源信息。

    Args:
        tmp_path (Path): Pytest 提供的临时数据目录。

    Raises:
        AssertionError: Writer 改写修正身份或丢失来源信息。

    Returns:
        None: 此测试只约束迁移所需的审计字段。
    """
    metadata = RunMetadata(
        run_id="RUN-CORRECTED",
        source="human",
        started_at="2026-08-27T08:00:00Z",
        character_id="DEFECT",
        seed="RUN-CORRECTED",
        ascension=1,
    )
    writer = HumanRunWriter(tmp_path, metadata)
    decision = _decision(
        event_id="human-confirmed:wing-sacrifice-floor19-1",
        layer="strategic",
        screen="MAP",
        floor=19,
        action="choose_map_node",
        option_index=0,
    )
    decision["run_id"] = "RUN-CORRECTED"
    decision["observed_at"] = "2026-08-27T08:01:00Z"
    decision["provenance"] = {
        "kind": "human_confirmed_correction",
        "correction_id": "wing-sacrifice-floor19-1",
    }

    destination = writer.append_decision(decision)

    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["event_id"] == "human-confirmed:wing-sacrifice-floor19-1"
    assert saved["provenance"] == decision["provenance"]


def _decision(
    *,
    event_id: int | str,
    layer: str,
    screen: str,
    floor: int,
    action: str,
    option_index: int | None = None,
) -> dict[str, object]:
    """构造可由当前 Harness 真实渲染的精确决策。

    Args:
        event_id (int | str): Mod SSE 编号或人工修正的稳定 ID。
        layer (str): 录制时的 Harness 层。
        screen (str): 动作前屏幕。
        floor (int): 动作发生楼层。
        action (str): 人类执行的动作。
        option_index (int | None): 可选动作索引。

    Returns:
        dict[str, object]: 精确动作与完整动作前状态。
    """
    state: dict[str, object] = {
        "screen": screen,
        "in_combat": layer == "battle",
        "available_actions": [action],
        "run": {
            "character_name": "故障机器人",
            "ascension": 3,
            "act_id": 0,
            "floor": floor,
            "current_hp": 70,
            "max_hp": 75,
            "gold": 99,
            "relics": [],
            "potions": [],
            "deck": [],
        },
    }
    if screen == "COMBAT":
        state["combat"] = {
            "player": {
                "current_hp": 70,
                "max_hp": 75,
                "block": 0,
                "energy": 3,
                "stars": 0,
                "focus": 0,
                "powers": [],
                "orbs": [],
            },
            "enemies": [],
            "hand": [],
            "draw_count": 0,
            "discard_count": 0,
        }
    else:
        state["map"] = {
            "available_nodes": [
                {"index": 0, "row": floor, "col": 1, "node_type": "Monster"}
            ]
        }
    parameters = {} if option_index is None else {"option_index": option_index}
    return {
        "run_id": "RUN-NEW",
        "source_sequence": event_id,
        "event_id": event_id,
        "observed_at": (
            f"2026-08-27T08:00:{event_id:02d}Z"
            if isinstance(event_id, int)
            else "2026-08-27T08:00:00Z"
        ),
        "recorded_layer": layer,
        "before_state": state,
        "action": action,
        "parameters": parameters,
    }
