"""把一局原始轨迹保存为简单的 JSON 和 JSONL 文件。"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import RecordedEvent, RunMetadata


class TrajectoryWriter:
    """创建一局轨迹目录并按观察顺序追加事件。

    Args:
        output_root (Path): 原始轨迹根目录，例如 ``data/raw``。
        metadata (RunMetadata): 当前局的来源与身份信息。

    Raises:
        FileExistsError: 同一来源下已经存在相同的局目录。
        OSError: 无法创建目录或写入元数据。
    """

    def __init__(self, output_root: Path, metadata: RunMetadata) -> None:
        """创建不会覆盖既有数据的局目录。

        Args:
            output_root (Path): 原始轨迹根目录。
            metadata (RunMetadata): 要写入当前局的元数据。

        Raises:
            FileExistsError: 目标局目录已经存在。
            OSError: 无法创建目录或写入元数据。
        """
        self._run_dir = Path(output_root) / metadata.source / metadata.run_id
        self._run_dir.mkdir(parents=True)
        self._events_path = self._run_dir / "events.jsonl"
        self._sequence = 0
        self._write_json(self._run_dir / "meta.json", metadata.to_dict())

    @property
    def run_dir(self) -> Path:
        """返回当前局的实际存放目录。

        Returns:
            Path: 包含 ``meta.json`` 和 ``events.jsonl`` 的目录。
        """
        return self._run_dir

    def append(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        observed_at: str,
    ) -> RecordedEvent:
        """向当前局追加一条原始事件。

        Args:
            event_type (str): 事件类型，例如 ``state`` 或 ``action``。
            payload (Mapping[str, Any]): 要保留的原始事件内容。
            observed_at (str): 观察到事件时的 UTC 时间。

        Raises:
            OSError: 无法追加事件文件。
            ValueError: 事件内容不能表示成标准 JSON。

        Returns:
            RecordedEvent: 已写入且带有连续序号的事件。
        """
        self._sequence += 1
        event = RecordedEvent(
            sequence=self._sequence,
            observed_at=observed_at,
            type=event_type,
            payload=dict(payload),
        )
        line = json.dumps(event.to_dict(), ensure_ascii=False, allow_nan=False)
        with self._events_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{line}\n")
        return event

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        """写入便于人工阅读的 JSON 文件。

        Args:
            path (Path): 目标 JSON 文件。
            payload (Mapping[str, Any]): 要序列化的对象。

        Raises:
            OSError: 无法写入目标文件。
            ValueError: 对象不能表示成标准 JSON。

        Returns:
            None: 文件写入完成后返回。
        """
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
