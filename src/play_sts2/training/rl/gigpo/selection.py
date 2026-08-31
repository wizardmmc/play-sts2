"""按可解释结果事实选择阶段七战斗刷新场景。"""

import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def select_battle_candidates(
    *,
    candidates_path: Path,
    additional_candidates_paths: Sequence[Path] = (),
    output_root: Path,
    history_path: Path | None = None,
    max_scenarios: int,
    seed: int,
) -> dict[str, Any]:
    """从 backbone/Tree 暴露入口选择少量战斗刷新场景。

    第一版只使用已观测结果：死亡、可见 HP 损失、Boss/精英覆盖和普通战斗随机
    保底。它不训练判别器，也不引入卡牌或遗物强度表。``max_scenarios < 5`` 时
    不强行取整 20%，只在摘要中记录普通场景欠额。

    Args:
        candidates_path (Path): ``collect-rl-gigpo`` 写出的本地候选清单。
        additional_candidates_paths (Sequence[Path]): terminal Tree group 文件。
        output_root (Path): scenario、snapshot 与 selection 摘要目录。
        history_path (Path | None): 跨轮保留最近二十个选择的历史文件。
        max_scenarios (int): 本轮墙钟预算允许的最大场景数。
        seed (int): 普通场景随机保底的确定种子。

    Raises:
        ValueError: 候选文件、预算或战斗字段无效。

    Returns:
        dict[str, Any]: 已选数量、普通场景欠额与输出路径。
    """
    if max_scenarios <= 0:
        raise ValueError("战斗场景选择预算必须为正")
    sources = (
        Path(candidates_path),
        *(Path(path) for path in additional_candidates_paths),
    )
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sources]
    if any(not isinstance(payload, Mapping) for payload in payloads):
        raise TypeError("阶段七战斗候选文件必须是对象")
    strategy_policy = payloads[0].get("strategy_policy_version")
    battle_policy = payloads[0].get("battle_policy_version")
    if (
        not isinstance(strategy_policy, str)
        or not strategy_policy
        or not isinstance(battle_policy, str)
        or not battle_policy
        or any(
            payload.get("strategy_policy_version") != strategy_policy
            or payload.get("battle_policy_version") != battle_policy
            for payload in payloads[1:]
        )
    ):
        raise ValueError("战斗场景源混入不同 S_n/B_n")
    raw = [item for payload in payloads for item in payload.get("battles", [])]
    if not raw:
        raise ValueError("阶段七 backbone/Tree 候选没有战斗入口")
    candidates = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise TypeError("阶段七战斗候选必须是对象")
        scenario = item.get("scenario")
        snapshot = item.get("entry_snapshot")
        if not isinstance(scenario, Mapping) or not isinstance(snapshot, Mapping):
            raise TypeError("阶段七战斗候选缺少 scenario 或入口 snapshot")
        if any(
            existing.get("scenario") == scenario
            and existing.get("entry_snapshot") == snapshot
            for existing in candidates
        ):
            continue
        candidates.append(dict(item))
    if not candidates:
        raise ValueError("阶段七战斗候选去重后为空")

    ordinary = [
        item
        for item in candidates
        if item.get("is_boss") is not True and item.get("is_elite") is not True
    ]
    rng = random.Random(seed)
    rng.shuffle(ordinary)
    history_file = Path(history_path) if history_path is not None else None
    history = _load_selection_history(history_file)
    ordinary_debt = float(history.get("ordinary_debt", 0.0))
    required_now = math.floor(ordinary_debt + 0.2 * max_scenarios + 1e-9)
    reserve_ordinary = ordinary[:required_now]
    reserved_ids = {id(item) for item in reserve_ordinary}
    ranked = sorted(
        (item for item in candidates if id(item) not in reserved_ids),
        key=_priority,
    )
    selected = [*ranked[: max_scenarios - len(reserve_ordinary)], *reserve_ordinary]
    selected = selected[:max_scenarios]
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"战斗场景选择输出已存在: {output}")
    output.mkdir(parents=True)
    rows = []
    for index, item in enumerate(selected):
        scenario_path = output / f"battle-{index:03d}.json"
        snapshot_path = output / f"battle-{index:03d}.snapshot.json"
        scenario_path.write_text(
            json.dumps(item["scenario"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        snapshot_path.write_text(
            json.dumps(item["entry_snapshot"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "scenario": str(scenario_path),
                "entry_snapshot": str(snapshot_path),
                "outcome": item.get("outcome"),
                "hp_loss_ratio": item.get("hp_loss_ratio"),
                "is_elite": item.get("is_elite") is True,
                "is_boss": item.get("is_boss") is True,
                "strategy_policy_version": strategy_policy,
                "battle_policy_version": battle_policy,
            }
        )
    ordinary_selected = sum(not row["is_elite"] and not row["is_boss"] for row in rows)
    new_debt = max(0.0, ordinary_debt + 0.2 * len(rows) - ordinary_selected)
    history_rows = [
        *history.get("selected", []),
        *(
            {
                "ordinary": not row["is_elite"] and not row["is_boss"],
                "strategy_policy_version": strategy_policy,
                "battle_policy_version": battle_policy,
            }
            for row in rows
        ),
    ][-20:]
    summary = {
        "format": "stage7_battle_selection",
        "sources": [str(path) for path in sources],
        "strategy_policy_version": strategy_policy,
        "battle_policy_version": battle_policy,
        "selection_seed": seed,
        "available": len(candidates),
        "selected": rows,
        "ordinary_selected": ordinary_selected,
        "ordinary_deficit": int(new_debt > 1e-9),
        "ordinary_debt": new_debt,
    }
    (output / "selection.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if history_file is not None:
        history_file.parent.mkdir(parents=True, exist_ok=True)
        history_file.write_text(
            json.dumps(
                {
                    "format": "stage7_battle_selection_history",
                    "ordinary_debt": new_debt,
                    "selected": history_rows,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return {**summary, "output": str(output)}


def _load_selection_history(path: Path | None) -> dict[str, Any]:
    """读取可选的跨轮二十场选择历史。

    Args:
        path (Path | None): 历史 JSON；为空或尚不存在时从零开始。

    Raises:
        ValueError: 已存在文件的格式或普通场景欠额无效。

    Returns:
        dict[str, Any]: 已裁剪到最近二十场的选择历史。
    """
    if path is None or not path.exists():
        return {"ordinary_debt": 0.0, "selected": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("format") != "stage7_battle_selection_history"
        or not isinstance(payload.get("selected"), list)
    ):
        raise ValueError("阶段七战斗选择历史格式无效")
    try:
        debt = float(payload.get("ordinary_debt", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("阶段七普通战斗欠额无效") from exc
    if not math.isfinite(debt) or debt < 0:
        raise ValueError("阶段七普通战斗欠额无效")
    return {"ordinary_debt": debt, "selected": payload["selected"][-20:]}


def _priority(item: Mapping[str, Any]) -> tuple[float, ...]:
    """按死亡、HP 损失和 Boss/精英覆盖生成稳定排序键。

    Args:
        item (Mapping[str, Any]): 单个真实战斗候选。

    Returns:
        tuple[float, ...]: 供升序排序使用的负优先级。
    """
    outcome = str(item.get("outcome") or "")
    try:
        hp_loss = float(item.get("hp_loss_ratio", 0.0))
    except (TypeError, ValueError):
        hp_loss = 0.0
    return (
        -float(outcome in {"died", "model_error"}),
        -max(0.0, hp_loss),
        -float(item.get("is_boss") is True),
        -float(item.get("is_elite") is True),
        float(item.get("floor", 0)),
        float(item.get("arm_index", 0)),
        float(item.get("battle_index", 0)),
    )
