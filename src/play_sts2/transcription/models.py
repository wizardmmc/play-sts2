"""定义 transcript 派生阶段的结果对象。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    """描述一局 raw 已生成的可读 transcript。

    Args:
        output_dir (Path): 来源目录下与 raw 局同名的 transcript 目录。
        battle_decision_count (int): 已渲染的战斗动作数。
        strategic_decision_count (int): 已渲染的战略动作数。
    """

    output_dir: Path
    battle_decision_count: int
    strategic_decision_count: int
