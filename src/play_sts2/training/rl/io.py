"""读取战斗场景配置并持久化可审计 rollout group。"""

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ...scenario import (
    BattleScenario,
    BattleSnapshot,
    CardSnapshot,
    EnemySnapshot,
    IntentSnapshot,
    ModelInputSnapshot,
)
from .contracts import BattleRollout, BattleRolloutGroup


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


def write_battle_rollout(
    path: Path,
    rollout: BattleRollout,
    *,
    group_id: str,
    scenario: BattleScenario,
    environment: Mapping[str, str],
) -> Path:
    """立即保存已完成的一臂，明确它不是完整训练组。

    Args:
        path (Path): 独立臂文件的输出路径。
        rollout (BattleRollout): 已完成并通过worker校验的战斗轨迹。
        group_id (str): 所属采集组标识。
        scenario (BattleScenario): 统一的场景配置。
        environment (Mapping[str, str]): 当前游戏与采样后端版本收据。

    Returns:
        Path: 完整写入后的文件路径。

    Raises:
        FileExistsError: 输出已存在，不覆盖先前证据。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    payload = {
        "format": "battle_rollout",
        "group_id": group_id,
        "scenario": asdict(scenario),
        "environment": dict(environment),
        "rollout": asdict(rollout),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")
    temporary.replace(path)
    return path


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


def load_battle_snapshot(path: Path) -> BattleSnapshot:
    """从 selection 写出的 JSON 恢复原始完整游戏战斗入口快照。

    Args:
        path (Path): 单个 ``BattleSnapshot`` JSON 文件。

    Raises:
        ValueError: 嵌套字段不能构成完整快照。

    Returns:
        BattleSnapshot: 可作为第一条 scenario reset 的严格入口基准。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("战斗入口 snapshot 必须是 JSON 对象")
    try:
        model = payload["model_input"]
        hand = payload["hand"]
        enemies = payload["enemies"]
        if (
            not isinstance(model, Mapping)
            or not isinstance(hand, list)
            or not isinstance(enemies, list)
        ):
            raise TypeError
        snapshot = BattleSnapshot(
            turn=int(payload["turn"]),
            hand=tuple(CardSnapshot(**dict(card)) for card in hand),
            enemies=tuple(
                EnemySnapshot(
                    index=int(enemy["index"]),
                    enemy_id=str(enemy["enemy_id"]),
                    current_hp=int(enemy["current_hp"]),
                    max_hp=int(enemy["max_hp"]),
                    move_id=enemy.get("move_id"),
                    intents=tuple(
                        IntentSnapshot(**dict(intent)) for intent in enemy["intents"]
                    ),
                )
                for enemy in enemies
            ),
            model_input=ModelInputSnapshot(
                system=str(model["system"]),
                user=str(model["user"]),
                available_actions=tuple(model["available_actions"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("战斗入口 snapshot 字段无效") from exc
    return snapshot
