"""验证人工游玩录制器的最小只读生命周期。"""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from threading import Event
from typing import Any

from play_sts2.recording import HumanRunRecorder


class RecordingClient:
    """提供与真实 Mod 相同形状的有限状态和 SSE 事件。

    Args:
        states (Sequence[dict[str, Any]]): 每次轮询依次返回的完整状态。
        events (Sequence[dict[str, Any]]): 事件线程依次收到的 SSE 对象。
    """

    def __init__(
        self,
        states: Sequence[dict[str, Any]],
        events: Sequence[dict[str, Any]],
    ) -> None:
        """保存测试要播放的生产等价协议数据。

        Args:
            states (Sequence[dict[str, Any]]): 有序状态响应。
            events (Sequence[dict[str, Any]]): 有序 SSE 事件。
        """
        self._states = iter(states)
        self._events = events
        self._last_state = states[-1]

    def state(self) -> dict[str, Any]:
        """返回下一个状态，序列耗尽后保持最后状态。

        Returns:
            dict[str, Any]: 与 ``GET /state`` 相同形状的状态对象。
        """
        self._last_state = next(self._states, self._last_state)
        return self._last_state

    def iter_events(self, stop_event: Event) -> Iterator[dict[str, Any]]:
        """发布事件后等待录制器结束监听。

        Args:
            stop_event (Event): 录制器结束时设置的线程事件。

        Yields:
            dict[str, Any]: 与真实 Mod 相同形状的 SSE 对象。
        """
        yield from self._events
        stop_event.wait()


def test_recorder_saves_changed_states_and_exact_mod_events(tmp_path: Path) -> None:
    """从开局到终局保存去重状态、原始 SSE 和结束原因。

    Args:
        tmp_path (Path): Pytest 提供的原始数据目录。

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
        states=[
            {
                "state_version": 0,
                "run_id": None,
                "screen": "MAIN_MENU",
                "session": {"phase": "menu"},
                "available_actions": ["open_character_select"],
            },
            run_state,
            run_state,
            game_over_state,
        ],
        events=[
            {"event_id": 1, "type": "stream_ready", "data": {}},
            exact_action,
        ],
    )

    result = HumanRunRecorder(
        client,
        tmp_path,
        poll_interval=0,
    ).record()

    assert result is not None
    assert result.termination_reason == "game_over"
    assert result.run_dir == tmp_path / "human/SEED-001"
    metadata = json.loads((result.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == "SEED-001"
    assert metadata["source"] == "human"
    assert metadata["character_id"] == "DEFECT"
    assert metadata["seed"] == "SEED-001"

    events = [
        json.loads(line)
        for line in (result.run_dir / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    state_events = [event for event in events if event["type"] == "state"]
    assert [event["payload"]["screen"] for event in state_events] == [
        "EVENT",
        "GAME_OVER",
    ]
    mod_events = [event for event in events if event["type"] == "mod_event"]
    assert [event["payload"]["type"] for event in mod_events] == [
        "stream_ready",
        "action_executed",
    ]
    assert mod_events[1]["payload"] == exact_action
    assert events[-1]["type"] == "recording_ended"
    assert events[-1]["payload"] == {"reason": "game_over"}
    assert result.event_count == len(events)


def test_recorder_stops_when_player_returns_to_menu(tmp_path: Path) -> None:
    """玩家保存退出到主菜单时以独立原因结束当前轨迹。

    Args:
        tmp_path (Path): Pytest 提供的原始数据目录。

    Raises:
        AssertionError: 返回主菜单没有结束录制或原因不正确。

    Returns:
        None: 此测试仅验证一条命令只录制一局。
    """
    client = RecordingClient(
        states=[
            {
                "run_id": "SEED-002",
                "screen": "MAP",
                "session": {"phase": "run"},
                "run": {"character_id": "DEFECT", "floor": 1},
            },
            {
                "run_id": None,
                "screen": "MAIN_MENU",
                "session": {"phase": "menu"},
                "available_actions": ["continue_run"],
            },
        ],
        events=[{"event_id": 1, "type": "stream_ready", "data": {}}],
    )

    result = HumanRunRecorder(
        client,
        tmp_path,
        poll_interval=0,
    ).record()

    assert result is not None
    assert result.termination_reason == "returned_to_menu"
