"""定义录制阶段使用的最小轨迹数据结构。"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """描述一局原始轨迹的基本身份。

    Args:
        run_id (str): 当前轨迹在数据目录中的唯一名称。
        source (Literal["human", "agent"]): 轨迹来自人类还是 Agent。
        started_at (str): 录制开始时的 UTC 时间。
        character_id (str | None): 当前角色 ID，尚未开局时为 ``None``。
        seed (str | None): 游戏种子，尚不可用时为 ``None``。
        ascension (int | None): 当前局进阶难度，尚不可用时为 ``None``。
    """

    run_id: str
    source: Literal["human", "agent"]
    started_at: str
    character_id: str | None = None
    seed: str | None = None
    ascension: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换成可直接写入 JSON 的对象。

        Returns:
            dict[str, Any]: 包含全部元数据字段的普通字典。
        """
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecordedRun:
    """描述已经结束的一次人工轨迹录制。

    Args:
        run_dir (Path): 当前局的原始轨迹目录。
        termination_reason (str): 录制结束原因。
        event_count (int): 本次写入的精确决策事件总数。
    """

    run_dir: Path
    termination_reason: str
    event_count: int
