"""以只读方式录制一局人类游玩产生的状态和 Mod 事件。"""

import copy
import queue
import threading
import time
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .models import RecordedRun, RunMetadata
from .writer import TrajectoryWriter

_STREAM_START_TIMEOUT_SECONDS = 5.0


class RecordingError(RuntimeError):
    """表示事件流无法提供完整的原始动作记录。"""


class _RecordingClient(Protocol):
    """描述录制器需要的两个只读 Mod 接口。"""

    def state(self) -> dict[str, Any]:
        """读取当前完整状态。

        Returns:
            dict[str, Any]: Mod 的 ``GET /state`` 数据对象。
        """
        ...

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
        client (_RecordingClient): 只提供状态和事件流的 Mod 客户端。
        output_root (Path): 原始轨迹根目录，通常为 ``data/raw``。
        stop_event (threading.Event | None): 外部中止录制的线程事件。
        poll_interval (float): 两次状态读取之间的秒数。
    """

    def __init__(
        self,
        client: _RecordingClient,
        output_root: Path,
        *,
        stop_event: threading.Event | None = None,
        poll_interval: float = 0.1,
    ) -> None:
        """保存录制所需依赖，不主动读取或改变游戏。

        Args:
            client (_RecordingClient): 只读 Mod 客户端。
            output_root (Path): 原始轨迹根目录。
            stop_event (threading.Event | None): 可选的外部停止信号。
            poll_interval (float): 状态轮询间隔秒数。
        """
        self._client = client
        self._output_root = Path(output_root)
        self._stop_event = stop_event or threading.Event()
        self._poll_interval = poll_interval

    def record(self) -> RecordedRun | None:
        """录制从当前时刻开始遇到的第一局游戏。

        录制器只调用状态与事件流端点。角色选择和出发等开局事件可以保留在
        原始流中，但是否作为训练决策由后续转录器决定。

        Raises:
            RecordingError: 事件流未就绪或在录制期间中断。
            httpx.HTTPError: 状态端点连接失败。
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
        writer: TrajectoryWriter | None = None
        previous_state: dict[str, Any] | None = None
        event_count = 0
        termination_reason: str | None = None

        try:
            self._wait_for_stream(stream_ready, error_queue, listener_stop)
            while not self._stop_event.is_set():
                self._raise_stream_error(error_queue)
                state = self._client.state()
                if writer is None:
                    metadata = self._metadata_from(state)
                    if metadata is None:
                        time.sleep(self._poll_interval)
                        continue
                    writer = TrajectoryWriter(self._output_root, metadata)
                    event_count += self._drain_events(writer, event_queue)

                if state != previous_state:
                    writer.append("state", state, observed_at=_utc_now())
                    event_count += 1
                    previous_state = copy.deepcopy(state)
                event_count += self._drain_events(writer, event_queue)

                termination_reason = self._termination_reason(state)
                if termination_reason is not None:
                    time.sleep(self._poll_interval)
                    event_count += self._drain_events(writer, event_queue)
                    break
                time.sleep(self._poll_interval)
        finally:
            listener_stop.set()
            listener.join(timeout=max(self._poll_interval, 0.01))

        if writer is None:
            return None
        termination_reason = termination_reason or "interrupted"
        writer.append(
            "recording_ended",
            {"reason": termination_reason},
            observed_at=_utc_now(),
        )
        event_count += 1
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
        """在独立线程中持续接收 SSE，避免阻塞状态轮询。

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
                    errors.put(RecordingError("Mod 事件流意外结束"))
            # 异常必须跨线程交还给调用者，不能让守护线程静默死亡。
            except Exception as exc:  # noqa: BLE001
                if not listener_stop.is_set():
                    errors.put(RecordingError(f"Mod 事件流中断: {exc}"))

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
        """在读取状态前确认精确动作事件流已经连接。

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

    @staticmethod
    def _drain_events(
        writer: TrajectoryWriter,
        source: queue.Queue[tuple[str, dict[str, Any]]],
    ) -> int:
        """把当前已到达的原始 Mod 事件写入同一局轨迹。

        Args:
            writer (TrajectoryWriter): 当前局的事件 writer。
            source (queue.Queue[tuple[str, dict[str, Any]]]): 待写入事件队列。

        Returns:
            int: 本次追加的 Mod 事件数量。
        """
        count = 0
        while True:
            try:
                observed_at, envelope = source.get_nowait()
            except queue.Empty:
                return count
            writer.append("mod_event", envelope, observed_at=observed_at)
            count += 1

    @staticmethod
    def _metadata_from(state: Mapping[str, Any]) -> RunMetadata | None:
        """从第一个运行态状态构造轨迹身份。

        Args:
            state (Mapping[str, Any]): Mod 的完整当前状态。

        Returns:
            RunMetadata | None: 已进入一局时的元数据，否则为 ``None``。
        """
        session = state.get("session")
        run_id = state.get("run_id")
        if (
            not isinstance(session, Mapping)
            or session.get("phase") != "run"
            or isinstance(run_id, bool)
            or not isinstance(run_id, (str, int))
            or not str(run_id).strip()
        ):
            return None

        run = state.get("run")
        character_id = run.get("character_id") if isinstance(run, Mapping) else None
        return RunMetadata(
            run_id=str(run_id),
            source="human",
            started_at=_utc_now(),
            character_id=character_id if isinstance(character_id, str) else None,
            seed=str(run_id),
        )

    @staticmethod
    def _termination_reason(state: Mapping[str, Any]) -> str | None:
        """判断当前状态是否已经离开可继续录制的一局。

        Args:
            state (Mapping[str, Any]): Mod 的完整当前状态。

        Returns:
            str | None: 终局或返回菜单的原因；仍在局中时为 ``None``。
        """
        if state.get("screen") == "GAME_OVER" or bool(state.get("game_over")):
            return "game_over"
        if state.get("screen") == "MAIN_MENU":
            return "returned_to_menu"
        return None


def _utc_now() -> str:
    """生成适合写入 JSON 的毫秒级 UTC 时间。

    Returns:
        str: 以 ``Z`` 结尾的 ISO 8601 时间。
    """
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
