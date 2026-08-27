"""提供原始游戏轨迹的数据结构与落盘工具。"""

from .models import RecordedEvent, RunMetadata
from .writer import TrajectoryWriter

__all__ = ["RecordedEvent", "RunMetadata", "TrajectoryWriter"]
