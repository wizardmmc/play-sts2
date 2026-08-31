"""按节点类型与楼层选择少量 terminal Tree checkpoint。"""

import json
import random
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ....checkpoint import StrategicCheckpoint, load_strategic_checkpoint


def select_tree_checkpoints(
    *,
    candidates_path: Path,
    output_path: Path,
    max_checkpoints: int = 2,
    selection_seed: int = 0,
) -> dict[str, Any]:
    """选择一个早/中期和一个较晚 checkpoint。

    选择只使用本轮已观测的节点类型、楼层、根选项数和路径，不引入卡牌、遗物、
    事件或路线强度表。第一轮没有历史 regret/方差时，类型欠覆盖用于早/中期
    tie-break，最晚节点用于长 horizon 覆盖。

    Args:
        candidates_path (Path): backbone 写出的本地候选清单。
        output_path (Path): 选择摘要 JSON。
        max_checkpoints (int): 墙钟允许的一至两个节点。
        selection_seed (int): 类型分层后随机抽样使用的固定种子。

    Raises:
        ValueError: 候选或预算无效。

    Returns:
        dict[str, Any]: 被选 checkpoint 的可读字段和路径。
    """
    if not 1 <= max_checkpoints <= 2:
        raise ValueError("第一版 terminal Tree 每轮只允许一至两个 checkpoint")
    payload = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    raw = payload.get("checkpoints") if isinstance(payload, Mapping) else None
    strategy_policy = (
        payload.get("strategy_policy_version") if isinstance(payload, Mapping) else None
    )
    battle_policy = (
        payload.get("battle_policy_version") if isinstance(payload, Mapping) else None
    )
    if not isinstance(raw, list) or not raw:
        raise ValueError("阶段七候选清单没有宏 checkpoint")
    if (
        not isinstance(strategy_policy, str)
        or not strategy_policy
        or not isinstance(battle_policy, str)
        or not battle_policy
    ):
        raise ValueError("阶段七 Tree 候选缺少来源双 policy")
    candidates = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise TypeError("宏 checkpoint 候选必须是对象")
        kind = item.get("kind")
        floor = item.get("floor")
        path = item.get("path")
        options = item.get("option_ids")
        if path is None:
            continue
        if (
            not isinstance(kind, str)
            or not kind
            or isinstance(floor, bool)
            or not isinstance(floor, int)
            or floor < 0
            or not isinstance(path, str)
            or not path
            or not isinstance(options, list)
            or len(options) < 2
        ):
            raise ValueError("宏 checkpoint 候选字段无效")
        checkpoint = load_strategic_checkpoint(Path(path))
        if any(
            _same_checkpoint_state(existing["checkpoint"], checkpoint)
            for existing in candidates
        ):
            continue
        candidates.append({**dict(item), "checkpoint": checkpoint})
    if not candidates:
        raise ValueError("可分支宏 checkpoint 去重后为空")
    counts = Counter(str(item["kind"]) for item in candidates)
    rng = random.Random(selection_seed)
    ordered_by_floor = sorted(
        candidates, key=lambda item: (item["floor"], item["path"])
    )
    median_floor = ordered_by_floor[(len(ordered_by_floor) - 1) // 2]["floor"]
    early_pool = [item for item in candidates if item["floor"] <= median_floor]
    early_count = min(counts[str(item["kind"])] for item in early_pool)
    early_candidates = sorted(
        (item for item in early_pool if counts[str(item["kind"])] == early_count),
        key=lambda item: (item["floor"], item["path"]),
    )
    early = rng.choice(early_candidates)
    selected = [early]
    if max_checkpoints == 2:
        later = [
            item
            for item in candidates
            if item["path"] != early["path"] and item["floor"] > median_floor
        ]
        diverse = [item for item in later if item["kind"] != early["kind"]]
        if diverse:
            later = diverse
        if later:
            late_count = min(counts[str(item["kind"])] for item in later)
            late_candidates = sorted(
                (item for item in later if counts[str(item["kind"])] == late_count),
                key=lambda item: (item["floor"], item["path"]),
            )
            selected.append(rng.choice(late_candidates))
    result = {
        "format": "stage7_tree_selection",
        "source": str(candidates_path),
        "strategy_policy_version": strategy_policy,
        "battle_policy_version": battle_policy,
        "selection_seed": selection_seed,
        "selected": [
            {key: value for key, value in item.items() if key != "checkpoint"}
            for item in selected
        ],
        "available": len(candidates),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _same_checkpoint_state(
    left: StrategicCheckpoint,
    right: StrategicCheckpoint,
) -> bool:
    """比较两个物化 checkpoint 的完整环境与 policy 入口。

    Args:
        left (StrategicCheckpoint): 已保留候选。
        right (StrategicCheckpoint): 新候选。

    Returns:
        bool: 原生审计、消息文本和合法动作均相同时返回真。
    """
    return (
        left.game_version == right.game_version
        and left.mod_version == right.mod_version
        and left.entry.audit == right.entry.audit
        and left.entry.policy_text == right.entry.policy_text
        and left.entry.legal_actions == right.entry.legal_actions
    )
