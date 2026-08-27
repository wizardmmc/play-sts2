"""定义精确决策转录阶段的数据结构。"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TranscribedDecision:
    """表示从一条原生 UI 事件提取的精确决策。

    Args:
        run_id (str): 决策所属的游戏局 ID。
        source_sequence (int): recorder 中原始事件的连续序号。
        event_id (int): Agent Mod 为 SSE 事件分配的序号。
        observed_at (str): recorder 观察到动作的 UTC 时间。
        recorded_layer (str | None): Mod 记录的 Harness 层提示。
        before_state (dict[str, Any]): 动作执行前的完整游戏状态。
        action (str): 已执行动作的稳定名称。
        parameters (dict[str, Any]): 动作携带的非空参数。
    """

    run_id: str
    source_sequence: int
    event_id: int
    observed_at: str
    recorded_layer: str | None
    before_state: dict[str, Any]
    action: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换成可直接写入 JSONL 的普通字典。

        Returns:
            dict[str, Any]: 包含完整决策字段的字典。
        """
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TranscribedRun:
    """描述一次单局转录的输出结果。

    Args:
        output_path (Path): 当前局精确决策 JSONL 的路径。
        decision_count (int): 写入文件的人类决策数量。
    """

    output_path: Path
    decision_count: int
