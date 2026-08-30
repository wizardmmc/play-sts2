"""读取战斗场景配置并持久化可审计 rollout group。"""

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ...scenario import BattleScenario
from .contracts import BattleRolloutGroup


def load_battle_scenario(path: Path) -> BattleScenario:
    """从一个 JSON 对象加载确定性战斗场景。

    Args:
        path (Path): 含 `BattleScenario` 字段的 UTF-8 JSON 文件。

    Raises:
        TypeError: JSON 顶层或数组字段的类型错误。
        ValueError: 字段无法构成合法场景。
        OSError: 场景文件无法读取。
        json.JSONDecodeError: 文件不是合法 JSON。

    Returns:
        BattleScenario: 已执行原有字段校验的场景对象。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("战斗场景 JSON 顶层必须是对象")
    values: dict[str, Any] = dict(payload)
    for field in ("deck", "relics", "potions"):
        if field in values:
            raw = values[field]
            if not isinstance(raw, list):
                raise TypeError(f"战斗场景字段必须是数组: {field}")
            values[field] = tuple(raw)
    try:
        return BattleScenario(**values)
    except TypeError as exc:
        raise ValueError(f"战斗场景字段无效: {path}") from exc


def write_battle_rollout_group(
    path: Path,
    group: BattleRolloutGroup,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """把完整 group 以可读 JSON 写入指定路径。

    Args:
        path (Path): 输出 JSON 路径。
        group (BattleRolloutGroup): 已通过准入的完整 rollout group。
        environment (Mapping[str, str] | None): 可选的游戏、Mod 与协议版本收据。

    Returns:
        Path: 实际写入的输出路径。
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(group)
    if environment is not None:
        payload["environment"] = dict(environment)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output
