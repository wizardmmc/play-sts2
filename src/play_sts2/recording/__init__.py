"""提供原始游戏轨迹的数据结构与落盘工具。"""

from .models import RecordedEvent, RecordedRun, RunMetadata
from .recorder import HumanRunRecorder, RecordingError
from .writer import TrajectoryWriter

__all__ = [
    "HumanRunRecorder",
    "RecordedEvent",
    "RecordedRun",
    "RecordingError",
    "RunMetadata",
    "TrajectoryWriter",
]
