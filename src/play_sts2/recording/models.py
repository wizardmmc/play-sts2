"""定义录制阶段使用的最小轨迹数据结构。"""

from dataclasses import asdict, dataclass
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
    """

    run_id: str
    source: Literal["human", "agent"]
    started_at: str
    character_id: str | None = None
    seed: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换成可直接写入 JSON 的对象。

        Returns:
            dict[str, Any]: 包含全部元数据字段的普通字典。
        """
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecordedEvent:
    """表示原始轨迹中的一条有序事件。

    Args:
        sequence (int): 从 1 开始递增的局内事件序号。
        observed_at (str): 观察到事件时的 UTC 时间。
        type (str): 事件类型，例如 ``state`` 或 ``action``。
        payload (dict[str, Any]): 未转录的 Mod 状态或动作内容。
    """

    sequence: int
    observed_at: str
    type: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换成可写入 JSONL 的对象。

        Returns:
            dict[str, Any]: 保留字段顺序的事件字典。
        """
        return asdict(self)
