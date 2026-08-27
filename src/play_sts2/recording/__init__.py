"""提供原始游戏轨迹的数据结构、审计与落盘工具。"""

from .audit import HumanRunAudit, RawRunIntegrityError, audit_human_run
from .models import RecordedRun, RunMetadata
from .recorder import HumanRunRecorder, RecordingError
from .writer import HumanRunWriter

__all__ = [
    "HumanRunAudit",
    "HumanRunRecorder",
    "HumanRunWriter",
    "RawRunIntegrityError",
    "RecordedRun",
    "RecordingError",
    "RunMetadata",
    "audit_human_run",
]
