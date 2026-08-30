"""保留全部知识，只对训练集高频人类动作应用显式上限。"""

import random
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import DatasetBuildError


@dataclass(frozen=True, slots=True)
class SftMixConfig:
    """保存不会破坏知识覆盖的 SFT 数据混合配方。

    Args:
        seed (int): 确定性人类动作抽样种子。
        human_game_version (str | None): 人类行为录制时的游戏版本声明。
        human_train_action_limits (dict[str, int]): 训练高频动作上限。
        arithmetic_train_per_kind (dict[str, int]): 每类算术训练样本数。
    """

    seed: int
    human_game_version: str | None
    human_train_action_limits: dict[str, int]
    arithmetic_train_per_kind: dict[str, int]


def load_sft_mix(path: Path) -> SftMixConfig:
    """读取只允许限制训练人类动作的 TOML 配方。

    Args:
        path (Path): 配方文件路径。

    Raises:
        DatasetBuildError: 配方结构、数值无效或仍声明知识截断。
        OSError: 文件无法读取。

    Returns:
        SftMixConfig: 可直接交给混合器的配置。
    """
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if "knowledge" in data:
        raise DatasetBuildError("SFT 混合配方不能再设置知识类别上限")
    try:
        seed = int(data["seed"])
        human = data["human"]
        raw_game_version = human.get("game_version")
        if raw_game_version is not None and (
            not isinstance(raw_game_version, str) or not raw_game_version.strip()
        ):
            raise TypeError("human.game_version 必须是非空字符串")
        actions = _integer_mapping(
            human["train_max_per_action"],
            "human.train_max_per_action",
        )
        arithmetic = data.get("arithmetic", {})
        arithmetic_per_kind = _integer_mapping(
            arithmetic.get("train_per_kind", {}),
            "arithmetic.train_per_kind",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetBuildError(f"无效 SFT 混合配方: {path}: {exc}") from exc
    config = SftMixConfig(
        seed=seed,
        human_game_version=(
            raw_game_version.strip() if isinstance(raw_game_version, str) else None
        ),
        human_train_action_limits=actions,
        arithmetic_train_per_kind=arithmetic_per_kind,
    )
    apply_sft_mix(
        {"train": [], "dev": [], "test": []},
        seed=config.seed,
        human_train_action_limits=config.human_train_action_limits,
    )
    return config


def _integer_mapping(value: object, name: str) -> dict[str, int]:
    """把 TOML 表转换成字符串到整数的映射。

    Args:
        value (object): TOML 解析后的候选表。
        name (str): 报错时使用的字段名。

    Raises:
        TypeError: 值不是字符串键和整数值组成的表。

    Returns:
        dict[str, int]: 保留 TOML 字段顺序的整数映射。
    """
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str)
        or not isinstance(item, int)
        or isinstance(item, bool)
        or item < 0
        for key, item in value.items()
    ):
        raise TypeError(f"{name} 必须是整数表")
    return {str(key): int(item) for key, item in value.items()}


def apply_sft_mix(
    splits: Mapping[str, Sequence[dict[str, Any]]],
    *,
    seed: int,
    human_train_action_limits: Mapping[str, int],
    arithmetic_train_per_kind: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """保留全部知识和留出行为，只裁剪训练集指定高频动作。

    Args:
        splits (Mapping[str, Sequence[dict[str, Any]]]): 已完成知识与整局分卷的样本。
        seed (int): 确定性抽样种子。
        human_train_action_limits (Mapping[str, int]): 训练行为动作上限。
        arithmetic_train_per_kind (Mapping[str, int] | None): 可选的每类算术数量。

    Raises:
        DatasetBuildError: 分卷缺失或行为上限不是非负整数。

    Returns:
        dict[str, list[dict[str, Any]]]: 保持原始相对顺序的混合后分卷。
    """
    missing = {"train", "dev", "test"} - set(splits)
    if missing:
        raise DatasetBuildError(f"SFT 混合缺少分卷: {sorted(missing)}")
    invalid_actions = [
        action
        for action, limit in human_train_action_limits.items()
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0
    ]
    if invalid_actions:
        raise DatasetBuildError(f"SFT 人类动作上限无效: {invalid_actions}")
    arithmetic_limits = dict(arithmetic_train_per_kind or {})
    invalid_arithmetic = [
        kind
        for kind, limit in arithmetic_limits.items()
        if not isinstance(kind, str)
        or not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit < 0
    ]
    if invalid_arithmetic:
        raise DatasetBuildError(f"SFT 算术数量无效: {invalid_arithmetic}")

    rows = list(splits["train"])
    selected = {
        index
        for index, row in enumerate(rows)
        if not (
            row.get("source") == "human_play"
            and row.get("behavior_origin", "human") == "human"
            and str(row.get("action", "")) in human_train_action_limits
        )
        and not (arithmetic_limits and row.get("source") == "synthetic_arithmetic")
    }
    rng = random.Random(seed)
    for action in sorted(human_train_action_limits):
        indices = [
            index
            for index, row in enumerate(rows)
            if row.get("source") == "human_play"
            and row.get("behavior_origin", "human") == "human"
            and row.get("action") == action
        ]
        selected.update(_take_indices(indices, human_train_action_limits[action], rng))
    if arithmetic_limits:
        observed_kinds = {
            str(row.get("object_id"))
            for row in rows
            if row.get("source") == "synthetic_arithmetic"
        }
        if observed_kinds != set(arithmetic_limits):
            raise DatasetBuildError(
                "SFT 算术配方与实际题型不一致: "
                f"expected={sorted(arithmetic_limits)}, actual={sorted(observed_kinds)}"
            )
        for kind in sorted(arithmetic_limits):
            indices = [
                index
                for index, row in enumerate(rows)
                if row.get("source") == "synthetic_arithmetic"
                and row.get("object_id") == kind
            ]
            limit = arithmetic_limits[kind]
            if len(indices) < limit:
                raise DatasetBuildError(
                    f"SFT 算术题型 {kind} 只有 {len(indices)} 条，少于要求 {limit}"
                )
            selected.update(_take_indices(indices, limit, rng))
    return {
        "train": [row for index, row in enumerate(rows) if index in selected],
        "dev": list(splits["dev"]),
        "test": list(splits["test"]),
    }


def _take_indices(
    indices: list[int],
    limit: int,
    rng: random.Random,
) -> list[int]:
    """从一组索引中确定性抽取不超过上限的项目。

    Args:
        indices (list[int]): 同一动作的原始索引。
        limit (int): 最大保留数量。
        rng (random.Random): 确定性随机源。

    Returns:
        list[int]: 被选中的原始索引。
    """
    if len(indices) <= limit:
        return indices
    candidates = list(indices)
    rng.shuffle(candidates)
    return candidates[:limit]
