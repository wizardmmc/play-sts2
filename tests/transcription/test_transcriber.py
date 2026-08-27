"""验证原始人类轨迹到精确决策记录的转换。"""

import importlib
import json
from pathlib import Path
from typing import Any


def test_transcribe_run_writes_exact_human_ui_decisions(tmp_path: Path) -> None:
    """只转录成功执行的人类 UI 动作，并保留动作前状态。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 转录数量、顺序或规范化字段不符合约定。

    Returns:
        None: 此测试只验证转录器的可观察输出。
    """
    transcription = importlib.import_module("play_sts2.transcription")
    run_dir = tmp_path / "raw/human/RUN-001"
    before_play = {
        "run_id": "RUN-001",
        "screen": "COMBAT",
        "available_actions": ["play_card", "end_turn"],
    }
    before_end_turn = {
        **before_play,
        "state_version": 2,
    }
    _write_run(
        run_dir,
        [
            {
                "sequence": 1,
                "observed_at": "2026-08-27T02:00:00.000Z",
                "type": "state",
                "payload": before_play,
            },
            _action_event(
                sequence=2,
                event_id=7,
                observed_at="2026-08-27T02:00:01.000Z",
                source="human_ui",
                action="play_card",
                before_state=before_play,
                card_index=0,
                target_index=1,
            ),
            _action_event(
                sequence=3,
                event_id=8,
                observed_at="2026-08-27T02:00:02.000Z",
                source="harness",
                action="proceed",
                before_state=before_play,
            ),
            _action_event(
                sequence=4,
                event_id=9,
                observed_at="2026-08-27T02:00:03.000Z",
                source="human_ui",
                action="end_turn",
                before_state=before_end_turn,
                status="completed",
            ),
        ],
    )

    result = transcription.transcribe_run(run_dir, tmp_path / "transcripts")

    assert result.output_path == tmp_path / "transcripts/RUN-001.jsonl"
    assert result.decision_count == 2
    decisions = [
        json.loads(line)
        for line in result.output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert decisions == [
        {
            "run_id": "RUN-001",
            "source_sequence": 2,
            "event_id": 7,
            "observed_at": "2026-08-27T02:00:01.000Z",
            "recorded_layer": "battle",
            "before_state": before_play,
            "action": "play_card",
            "parameters": {"card_index": 0, "target_index": 1},
        },
        {
            "run_id": "RUN-001",
            "source_sequence": 4,
            "event_id": 9,
            "observed_at": "2026-08-27T02:00:03.000Z",
            "recorded_layer": "battle",
            "before_state": before_end_turn,
            "action": "end_turn",
            "parameters": {},
        },
    ]


def _write_run(run_dir: Path, events: list[dict[str, Any]]) -> None:
    """写入一份不含个人数据的最小原始轨迹。

    Args:
        run_dir (Path): 测试轨迹目录。
        events (list[dict[str, Any]]): 要写入 JSONL 的原始事件。

    Returns:
        None: 元数据和事件写入完成后返回。
    """
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": "RUN-001",
                "source": "human",
                "started_at": "2026-08-27T02:00:00.000Z",
                "character_id": "DEFECT",
                "seed": "RUN-001",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def _action_event(
    *,
    sequence: int,
    event_id: int,
    observed_at: str,
    source: str,
    action: str,
    before_state: dict[str, Any],
    status: str = "accepted",
    card_index: int | None = None,
    target_index: int | None = None,
) -> dict[str, Any]:
    """构造与真实 Mod ``action_executed`` 同形状的测试事件。

    Args:
        sequence (int): recorder 分配的原始事件序号。
        event_id (int): Mod 分配的 SSE 事件序号。
        observed_at (str): recorder 观察事件的时间。
        source (str): 动作客户端来源。
        action (str): 已执行动作名称。
        before_state (dict[str, Any]): 动作执行前的完整游戏状态。
        status (str): Mod 报告的动作状态。
        card_index (int | None): 可选的卡牌索引。
        target_index (int | None): 可选的目标索引。

    Returns:
        dict[str, Any]: 可直接写入原始轨迹的事件对象。
    """
    return {
        "sequence": sequence,
        "observed_at": observed_at,
        "type": "mod_event",
        "payload": {
            "event_id": event_id,
            "timestamp_utc": observed_at,
            "type": "action_executed",
            "data": {
                "request": {
                    "action": action,
                    "card_index": card_index,
                    "target_index": target_index,
                    "option_index": None,
                    "command": None,
                    "client_context": {
                        "source": source,
                        "layer": "battle",
                    },
                },
                "before_state": before_state,
                "after_state": before_state,
                "status": status,
                "stable": status == "completed",
            },
        },
    }
