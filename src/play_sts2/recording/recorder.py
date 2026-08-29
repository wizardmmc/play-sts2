"""以只读方式从 Mod SSE 流录制一局精确人类决策。"""

import queue
import threading
import time
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .models import RecordedRun, RunMetadata, RunSource
from .writer import HumanRunWriter

_STREAM_START_TIMEOUT_SECONDS = 5.0
_SUCCESS_STATUSES = {"accepted", "completed", "pending"}
_PARTIAL_RUN_REASON = "partial_run: recording_started_after_run_start"
_ACTION_SOURCES_BY_RUN_SOURCE: dict[RunSource, frozenset[str]] = {
    "human": frozenset({"human_ui"}),
    "agent": frozenset(),
    "human_combat_solver": frozenset({"human_ui", "combat_solver"}),
}


class RecordingError(RuntimeError):
    """表示事件流无法提供完整的原始动作记录。"""


class _RecordingStreamError(RecordingError):
    """表示已连接的 SSE 流在一局中途结束。"""


class _RecordingClient(Protocol):
    """描述录制器唯一需要的只读 Mod 事件接口。"""

    def iter_events(self, stop_event: threading.Event) -> Iterator[dict[str, Any]]:
        """持续读取原始 Mod 事件。

        Args:
            stop_event (threading.Event): 结束事件消费的线程事件。

        Yields:
            dict[str, Any]: Mod 的原始 SSE 事件对象。
        """
        ...


class HumanRunRecorder:
    """等待一局开始，并把人类操作涉及的原始事实写入磁盘。

    Args:
        client (_RecordingClient): 只提供事件流的 Mod 客户端。
        output_root (Path): 原始轨迹根目录，通常为 ``data/raw``。
        source (RunSource): 整局数据来源；人机协作录制使用
            ``human_combat_solver``。
        recording_context (Mapping[str, Any] | None): 游戏、Mod 与教师设置等
            可审计环境信息。
        stop_event (threading.Event | None): 外部中止录制的线程事件。
        check_interval (float): 检查停止信号与事件线程异常的间隔秒数。
    """

    def __init__(
        self,
        client: _RecordingClient,
        output_root: Path,
        *,
        source: RunSource = "human",
        recording_context: Mapping[str, Any] | None = None,
        stop_event: threading.Event | None = None,
        check_interval: float = 0.1,
    ) -> None:
        """保存录制所需依赖，不主动读取或改变游戏。

        Args:
            client (_RecordingClient): 只读 Mod 客户端。
            output_root (Path): 原始轨迹根目录。
            source (RunSource): 整局数据来源。
            recording_context (Mapping[str, Any] | None): 可审计环境信息。
            stop_event (threading.Event | None): 可选的外部停止信号。
            check_interval (float): 停止信号与线程异常检查间隔秒数。
        """
        self._client = client
        self._output_root = Path(output_root)
        if source not in _ACTION_SOURCES_BY_RUN_SOURCE:
            raise ValueError(f"不支持的录制来源: {source}")
        self._source = source
        self._recording_context = dict(recording_context or {})
        if source == "human_combat_solver":
            self._recording_context.update(
                {
                    "student_observation_policy": "visible_only",
                    "teacher_uses_hidden_rng": True,
                }
            )
        self._stop_event = stop_event or threading.Event()
        self._check_interval = check_interval

    def record(self) -> RecordedRun | None:
        """录制从当前时刻开始遇到的第一局游戏。

        录制器只消费事件流；Mod 已经为事件检测构造完整状态，因此这里不会
        额外轮询 ``/state``。角色选择和出发等环境动作不进入局内策略样本。

        Raises:
            RecordingError: 事件流未就绪或在录制期间中断。
            httpx.HTTPError: SSE 端点连接失败。
            ProtocolError: Mod 返回了不符合客户端协议的数据。
            OSError: 无法创建或写入轨迹目录。

        Returns:
            RecordedRun | None: 完成的轨迹；在开局前被中止时为 ``None``。
        """
        if self._stop_event.is_set():
            return None

        event_queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        error_queue: queue.Queue[Exception] = queue.Queue()
        listener_stop = threading.Event()
        stream_ready = threading.Event()
        listener = self._start_listener(
            event_queue,
            error_queue,
            listener_stop,
            stream_ready,
        )
        writer: HumanRunWriter | None = None
        active_run_id: str | None = None
        event_count = 0
        termination_reason: str | None = None
        victory: bool | None = None

        try:
            self._wait_for_stream(stream_ready, error_queue, listener_stop)
            while not self._stop_event.is_set():
                try:
                    observed_at, envelope = event_queue.get(
                        timeout=max(self._check_interval, 0.01)
                    )
                except queue.Empty:
                    self._raise_stream_error(error_queue)
                    continue
                if writer is None:
                    metadata = self._metadata_from_event(envelope)
                    if metadata is not None:
                        writer = HumanRunWriter(self._output_root, metadata)
                        active_run_id = metadata.run_id
                        if envelope.get("type") == "stream_ready":
                            writer.record_integrity_failure(_PARTIAL_RUN_REASON)
                if writer is None:
                    continue
                assert active_run_id is not None
                if envelope.get("type") == "combat_ended":
                    writer.end_battle()
                recording_gap = self._recording_gap_from_event(envelope)
                if recording_gap is not None:
                    writer.record_recording_gap(recording_gap)
                decision = self._decision_from_event(
                    active_run_id,
                    observed_at,
                    envelope,
                )
                if decision is not None:
                    writer.append_decision(decision)
                    event_count += 1
                termination = self._termination_from_event(
                    envelope,
                    active_run_id,
                )
                if termination is not None:
                    termination_reason, victory = termination
                    break
        except _RecordingStreamError:
            if writer is None:
                raise
            termination_reason = "stream_interrupted"
        except KeyboardInterrupt:
            if writer is None:
                raise
            termination_reason = "interrupted"
        finally:
            listener_stop.set()
            listener.join(timeout=max(self._check_interval, 0.01))

        if writer is None:
            return None
        termination_reason = termination_reason or "interrupted"
        writer.finalize(
            termination_reason,
            completed_at=_utc_now(),
            victory=victory,
        )
        return RecordedRun(
            run_dir=writer.run_dir,
            termination_reason=termination_reason,
            event_count=event_count,
        )

    def _start_listener(
        self,
        destination: queue.Queue[tuple[str, dict[str, Any]]],
        errors: queue.Queue[Exception],
        listener_stop: threading.Event,
        stream_ready: threading.Event,
    ) -> threading.Thread:
        """在独立线程中持续接收 SSE，允许主线程处理停止信号。

        Args:
            destination (queue.Queue[tuple[str, dict[str, Any]]]): 事件目标队列。
            errors (queue.Queue[Exception]): 事件流异常队列。
            listener_stop (threading.Event): 内部事件流停止信号。
            stream_ready (threading.Event): 收到首个事件后的就绪信号。

        Returns:
            threading.Thread: 已经启动的守护线程。
        """

        def listen() -> None:
            """把客户端事件及其接收时间送入主录制循环。"""
            try:
                for envelope in self._client.iter_events(listener_stop):
                    destination.put((_utc_now(), envelope))
                    stream_ready.set()
                if not listener_stop.is_set():
                    errors.put(_RecordingStreamError("Mod 事件流意外结束"))
            # 异常必须跨线程交还给调用者，不能让守护线程静默死亡。
            except Exception as exc:  # noqa: BLE001
                if not listener_stop.is_set():
                    errors.put(_RecordingStreamError(f"Mod 事件流中断: {exc}"))

        listener = threading.Thread(
            target=listen,
            name="human-run-recorder-events",
            daemon=True,
        )
        listener.start()
        return listener

    def _wait_for_stream(
        self,
        stream_ready: threading.Event,
        errors: queue.Queue[Exception],
        listener_stop: threading.Event,
    ) -> None:
        """在创建任何局目录前确认精确动作事件流已经连接。

        Args:
            stream_ready (threading.Event): 首个 SSE 事件就绪信号。
            errors (queue.Queue[Exception]): 事件线程报告的异常。
            listener_stop (threading.Event): 内部事件流停止信号。

        Raises:
            RecordingError: 事件流报错或未在限定时间内就绪。

        Returns:
            None: 收到首个事件后返回。
        """
        deadline = time.monotonic() + _STREAM_START_TIMEOUT_SECONDS
        while not stream_ready.wait(timeout=0.01):
            self._raise_stream_error(errors)
            if self._stop_event.is_set():
                listener_stop.set()
                return
            if time.monotonic() >= deadline:
                raise RecordingError("Mod 事件流未在 5 秒内就绪")

    @staticmethod
    def _raise_stream_error(errors: queue.Queue[Exception]) -> None:
        """把事件线程的异常传播到调用录制器的线程。

        Args:
            errors (queue.Queue[Exception]): 事件线程异常队列。

        Raises:
            RecordingError: 事件流已经中断。

        Returns:
            None: 当前没有事件流异常时返回。
        """
        try:
            error = errors.get_nowait()
        except queue.Empty:
            return
        raise error

    def _metadata_from_event(
        self,
        envelope: Mapping[str, Any],
    ) -> RunMetadata | None:
        """从 ``run_started`` 或运行态 ``stream_ready`` 构造局身份。

        Args:
            envelope (Mapping[str, Any]): Mod 的原始 SSE 事件对象。

        Returns:
            RunMetadata | None: 已进入一局时的元数据，否则为 ``None``。
        """
        event_type = envelope.get("type")
        data = envelope.get("data")
        if (
            event_type not in {"run_started", "stream_ready"}
            or not isinstance(data, Mapping)
            or (event_type == "stream_ready" and data.get("session_phase") != "run")
        ):
            return None
        run_id = data.get("run_id")
        if (
            isinstance(run_id, bool)
            or not isinstance(run_id, (str, int))
            or not str(run_id).strip()
            or str(run_id) == "run_unknown"
        ):
            return None
        character_id = data.get("character_id")
        ascension = data.get("ascension")
        return RunMetadata(
            run_id=str(run_id),
            source=self._source,
            started_at=_utc_now(),
            character_id=character_id if isinstance(character_id, str) else None,
            seed=str(run_id),
            ascension=(
                ascension
                if isinstance(ascension, int) and not isinstance(ascension, bool)
                else None
            ),
            recording_context=(
                dict(self._recording_context) if self._recording_context else None
            ),
        )

    @staticmethod
    def _termination_from_event(
        envelope: Mapping[str, Any],
        run_id: str,
    ) -> tuple[str, bool | None] | None:
        """读取 Mod 明确发布的一局终止原因。

        Args:
            envelope (Mapping[str, Any]): Mod 的原始 SSE 事件对象。
            run_id (str): 当前正在录制的局 ID。

        Returns:
            tuple[str, bool | None] | None: 终止原因和胜负；仍在局中时为
                ``None``。
        """
        data = envelope.get("data")
        if envelope.get("type") != "run_ended" or not isinstance(data, Mapping):
            return None
        if str(data.get("run_id")) != run_id:
            return None
        reason = data.get("reason")
        victory = data.get("victory")
        return (
            reason if isinstance(reason, str) and reason else "game_over",
            victory if isinstance(victory, bool) else None,
        )

    @staticmethod
    def _recording_gap_from_event(envelope: Mapping[str, Any]) -> str | None:
        """把 Mod 报告的动作缺口转成不否定已有样本的审计原因。

        Args:
            envelope (Mapping[str, Any]): Mod 的原始 SSE 事件对象。

        Returns:
            str | None: 可写入局元数据的原因；其他事件返回 ``None``。
        """
        data = envelope.get("data")
        if envelope.get("type") != "native_ui_capture_gap" or not isinstance(
            data, Mapping
        ):
            return None
        action = data.get("action")
        reason = data.get("reason")
        action_text = (
            action.strip() if isinstance(action, str) and action.strip() else "?"
        )
        reason_text = (
            reason.strip() if isinstance(reason, str) and reason.strip() else "?"
        )
        source = data.get("source")
        if source in {"human_ui", "combat_solver"}:
            return f"action_capture_gap: {source}: {action_text}: {reason_text}"
        return f"native_ui_capture_gap: {action_text}: {reason_text}"

    def _decision_from_event(
        self,
        run_id: str,
        observed_at: str,
        envelope: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """从一条成功的人类 UI 动作事件提取精确决策。

        Args:
            run_id (str): 当前正在录制的局 ID。
            observed_at (str): Python 收到事件时的 UTC 时间。
            envelope (Mapping[str, Any]): Mod 的原始 SSE 事件对象。

        Returns:
            dict[str, Any] | None: 可写入人类数据集的决策；无关事件返回
                ``None``。
        """
        if envelope.get("type") != "action_executed":
            return None
        data = envelope.get("data")
        if not isinstance(data, Mapping) or data.get("status") not in _SUCCESS_STATUSES:
            return None
        request = data.get("request")
        if not isinstance(request, Mapping):
            return None
        context = request.get("client_context")
        if not isinstance(context, Mapping):
            return None
        action_source = context.get("source")
        if action_source not in _ACTION_SOURCES_BY_RUN_SOURCE[self._source]:
            return None
        event_id = envelope.get("event_id")
        state = data.get("before_state")
        action = request.get("action")
        if (
            isinstance(event_id, bool)
            or not isinstance(event_id, int)
            or not isinstance(state, Mapping)
            or not isinstance(action, str)
            or not action
        ):
            raise RecordingError("action_executed 缺少精确人类决策字段")
        layer = context.get("layer")
        return {
            "run_id": run_id,
            "source_sequence": event_id,
            "event_id": event_id,
            "observed_at": observed_at,
            "recorded_layer": layer if isinstance(layer, str) else None,
            "action_source": action_source,
            "before_state": dict(state),
            "action": action,
            "parameters": {
                key: value
                for key, value in request.items()
                if key not in {"action", "client_context"} and value is not None
            },
        }


def _utc_now() -> str:
    """生成适合写入 JSON 的毫秒级 UTC 时间。

    Returns:
        str: 以 ``Z`` 结尾的 ISO 8601 时间。
    """
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
