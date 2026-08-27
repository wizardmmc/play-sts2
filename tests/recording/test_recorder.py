"""验证人工游玩录制器的最小只读生命周期。"""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from play_sts2.recording import HumanRunRecorder


class RecordingClient:
    """提供与真实 Mod 相同形状的有限状态和 SSE 事件。

    Args:
        events (Sequence[dict[str, Any]]): 事件线程依次收到的 SSE 对象。
    """

    def __init__(
        self,
        events: Sequence[dict[str, Any]],
    ) -> None:
        """保存测试要播放的生产等价协议数据。

        Args:
            events (Sequence[dict[str, Any]]): 有序 SSE 事件。
        """
        self._events = events

    def state(self) -> dict[str, Any]:
        """拒绝录制器重复请求 Mod 已经为 SSE 构造的完整状态。

        Raises:
            AssertionError: 新录制链错误地恢复了高频 ``/state`` 轮询。
        """
        raise AssertionError("SSE 录制期间不应调用 /state")

    def iter_events(self, stop_event: Event) -> Iterator[dict[str, Any]]:
        """发布事件后等待录制器结束监听。

        Args:
            stop_event (Event): 录制器结束时设置的线程事件。

        Yields:
            dict[str, Any]: 与真实 Mod 相同形状的 SSE 对象。
        """
        yield from self._events
        stop_event.wait()


class InterruptedRecordingClient(RecordingClient):
    """发送有限事件后模拟 SSE 连接意外结束。"""

    def iter_events(self, stop_event: Event) -> Iterator[dict[str, Any]]:
        """发送全部事件后立即结束迭代。

        Args:
            stop_event (Event): 录制器的内部停止信号，本模拟不会等待它。

        Yields:
            dict[str, Any]: 断流前已经抵达的 SSE 事件。
        """
        del stop_event
        yield from self._events


def test_recorder_saves_changed_states_and_exact_mod_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """从开局到终局仅消费 SSE，并保存按层分片的精确动作。

    Args:
        tmp_path (Path): Pytest 提供的原始数据目录。
        monkeypatch (pytest.MonkeyPatch): 用于固定录制时间和最终目录名。

    Raises:
        AssertionError: 元数据、事件内容或终止原因不符合约定。

    Returns:
        None: 此测试仅验证一局完整录制的可观察结果。
    """
    run_state = {
        "state_version": 1,
        "run_id": "SEED-001",
        "screen": "EVENT",
        "session": {"phase": "run"},
        "run": {"character_id": "DEFECT", "floor": 0},
        "available_actions": ["choose_event_option"],
    }
    game_over_state = {
        **run_state,
        "state_version": 2,
        "screen": "GAME_OVER",
        "game_over": {"victory": False},
        "available_actions": ["return_to_menu"],
    }
    exact_action = {
        "event_id": 2,
        "type": "action_executed",
        "data": {
            "request": {
                "action": "choose_event_option",
                "option_index": 0,
                "client_context": {"source": "human_ui"},
            },
            "before_state": run_state,
            "after_state": game_over_state,
            "status": "completed",
            "stable": True,
        },
    }
    client = RecordingClient(
        events=[
            {"event_id": 1, "type": "stream_ready", "data": {}},
            {
                "event_id": 2,
                "type": "run_started",
                "data": {
                    "run_id": "SEED-001",
                    "character_id": "DEFECT",
                    "ascension": 4,
                },
            },
            exact_action,
            {
                "event_id": 3,
                "type": "run_ended",
                "data": {"run_id": "SEED-001", "reason": "game_over"},
            },
        ],
    )

    monkeypatch.setattr(
        "play_sts2.recording.recorder._utc_now",
        lambda: "2026-08-27T16:00:00.000Z",
    )
    result = HumanRunRecorder(
        client,
        tmp_path,
        check_interval=0,
    ).record()

    assert result is not None
    assert result.termination_reason == "game_over"
    assert result.run_dir == tmp_path / "human/20260828-a4-f0-SEED-001"
    metadata = json.loads((result.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == "SEED-001"
    assert metadata["source"] == "human"
    assert metadata["character_id"] == "DEFECT"
    assert metadata["seed"] == "SEED-001"
    assert metadata["ascension"] == 4
    assert metadata["battle_sample_count"] == 0
    assert metadata["strategic_sample_count"] == 1
    assert not (result.run_dir / "events.jsonl").exists()
    rows = [
        json.loads(line)
        for line in (result.run_dir / "strategy/decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["event_id"] == exact_action["event_id"]
    assert rows[0]["action"] == "choose_event_option"
    assert "messages" not in rows[0]
    assert result.event_count == 1


def test_recorder_stops_when_player_returns_to_menu(tmp_path: Path) -> None:
    """中途连接仍保留录制，但明确关闭半局数据的训练准入。

    Args:
        tmp_path (Path): Pytest 提供的原始数据目录。

    Raises:
        AssertionError: 返回主菜单未结束录制，或半局数据仍可训练。

    Returns:
        None: 此测试仅验证一条命令只录制一局。
    """
    client = RecordingClient(
        events=[
            {
                "event_id": 1,
                "type": "stream_ready",
                "data": {
                    "run_id": "SEED-002",
                    "session_phase": "run",
                    "character_id": "DEFECT",
                    "ascension": 0,
                },
            },
            {
                "event_id": 2,
                "type": "run_ended",
                "data": {"run_id": "SEED-002", "reason": "returned_to_menu"},
            },
        ],
    )

    result = HumanRunRecorder(
        client,
        tmp_path,
        check_interval=0,
    ).record()

    assert result is not None
    assert result.termination_reason == "returned_to_menu"
    metadata = json.loads((result.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert metadata["training_eligible"] is False
    assert metadata["integrity"]["ineligibility_reasons"] == [
        "partial_run: recording_started_after_run_start"
    ]


def test_recorder_marks_native_capture_gaps_as_training_ineligible(
    tmp_path: Path,
) -> None:
    """原生 UI 动作缺口保留整局，但关闭其训练准入。

    Args:
        tmp_path (Path): Pytest 提供的原始数据目录。

    Raises:
        AssertionError: 采集缺口未进入完整性元数据或局仍可训练。

    Returns:
        None: 此测试只检查缺口事件的保守处理。
    """
    client = RecordingClient(
        events=[
            {
                "event_id": 1,
                "type": "run_started",
                "data": {
                    "run_id": "GAP-SEED",
                    "character_id": "DEFECT",
                    "ascension": 1,
                },
            },
            {
                "event_id": 2,
                "type": "native_ui_capture_gap",
                "data": {
                    "action": "play_card",
                    "reason": "no matching state transition",
                },
            },
            {
                "event_id": 3,
                "type": "run_ended",
                "data": {"run_id": "GAP-SEED", "reason": "game_over"},
            },
        ],
    )

    result = HumanRunRecorder(client, tmp_path, check_interval=0).record()

    assert result is not None
    metadata = json.loads((result.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert metadata["training_eligible"] is False
    assert metadata["integrity"]["verified"] is False
    assert metadata["integrity"]["ineligibility_reasons"] == [
        "native_ui_capture_gap: play_card: no matching state transition"
    ]


def test_recorder_publishes_received_actions_after_stream_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SSE 异常结束时先保存已收到动作，再发布可识别的最终目录。

    Args:
        tmp_path (Path): Pytest 提供的原始数据目录。
        monkeypatch (pytest.MonkeyPatch): 用于固定目录日期。

    Raises:
        AssertionError: 断流丢动作、遗留隐藏目录或终止原因不明确。

    Returns:
        None: 此测试只验证可恢复的异常结束语义。
    """
    state = {
        "screen": "MAP",
        "in_combat": False,
        "available_actions": ["choose_map_node"],
        "run": {
            "character_name": "故障机器人",
            "ascension": 2,
            "act_id": 0,
            "floor": 7,
            "current_hp": 60,
            "max_hp": 75,
            "gold": 99,
            "relics": [],
            "potions": [],
            "deck": [],
        },
        "map": {
            "available_nodes": [
                {"index": 0, "row": 7, "col": 1, "node_type": "Monster"}
            ]
        },
    }
    client = InterruptedRecordingClient(
        events=[
            {
                "event_id": 1,
                "type": "run_started",
                "data": {
                    "run_id": "INTERRUPTED-SEED",
                    "character_id": "DEFECT",
                    "ascension": 2,
                },
            },
            {
                "event_id": 2,
                "type": "action_executed",
                "data": {
                    "request": {
                        "action": "choose_map_node",
                        "option_index": 0,
                        "client_context": {
                            "source": "human_ui",
                            "layer": "strategic",
                        },
                    },
                    "before_state": state,
                    "status": "accepted",
                },
            },
        ]
    )
    monkeypatch.setattr(
        "play_sts2.recording.recorder._utc_now",
        lambda: "2026-08-27T16:00:00.000Z",
    )

    result = HumanRunRecorder(client, tmp_path, check_interval=0).record()

    assert result is not None
    assert result.termination_reason == "stream_interrupted"
    assert result.event_count == 1
    assert result.run_dir == (tmp_path / "human/20260828-a2-f7-INTERRUPTED-SEED")
    assert not any(
        path.name.startswith(".recording-") for path in result.run_dir.parent.iterdir()
    )
