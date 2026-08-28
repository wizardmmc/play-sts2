"""按显式上限构建小规模、定向的 SFT 数据混合。"""

import random
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import DatasetBuildError


@dataclass(frozen=True, slots=True)
class SftMixConfig:
    """保存一份简洁的 SFT 数据混合配方。

    Args:
        seed (int): 确定性抽样种子。
        knowledge_limits (dict[str, dict[str, int]]): 训练和验证知识类别上限。
        human_train_action_limits (dict[str, int]): 训练高频动作上限。
    """

    seed: int
    knowledge_limits: dict[str, dict[str, int]]
    human_train_action_limits: dict[str, int]


def load_sft_mix(path: Path) -> SftMixConfig:
    """读取一份 TOML 数据混合配方。

    Args:
        path (Path): 配方文件路径。

    Raises:
        DatasetBuildError: 配方结构或数值无效。
        OSError: 文件无法读取。

    Returns:
        SftMixConfig: 可直接交给混合器的配置。
    """
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    try:
        seed = int(data["seed"])
        knowledge = data["knowledge"]
        human = data["human"]
        train = _integer_mapping(knowledge["train"], "knowledge.train")
        dev = _integer_mapping(knowledge["dev"], "knowledge.dev")
        actions = _integer_mapping(
            human["train_max_per_action"],
            "human.train_max_per_action",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetBuildError(f"无效 SFT 混合配方: {path}: {exc}") from exc
    config = SftMixConfig(
        seed=seed,
        knowledge_limits={"train": train, "dev": dev},
        human_train_action_limits=actions,
    )
    apply_sft_mix(
        {"train": [], "dev": [], "test": []},
        seed=config.seed,
        knowledge_limits=config.knowledge_limits,
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
        not isinstance(key, str) or not isinstance(item, int) or isinstance(item, bool)
        for key, item in value.items()
    ):
        raise TypeError(f"{name} 必须是整数表")
    return {str(key): int(item) for key, item in value.items()}


def apply_sft_mix(
    splits: Mapping[str, Sequence[dict[str, Any]]],
    *,
    seed: int,
    knowledge_limits: Mapping[str, Mapping[str, int]],
    human_train_action_limits: Mapping[str, int],
) -> dict[str, list[dict[str, Any]]]:
    """按知识类别和高频动作上限筛选三个既有分卷。

    未在知识上限中声明的类别不会进入对应分卷。人类行为只裁剪训练集里显式
    声明的高频动作；未声明动作以及验证、测试行为保持不变。

    Args:
        splits (Mapping[str, Sequence[dict[str, Any]]]): 已完成知识与整局分卷的样本。
        seed (int): 各分组确定性抽样使用的随机种子。
        knowledge_limits (Mapping[str, Mapping[str, int]]): 训练和验证知识类别上限；
            ``-1`` 表示全部保留。
        human_train_action_limits (Mapping[str, int]): 训练行为动作上限。

    Raises:
        DatasetBuildError: 分卷缺失、知识上限小于 ``-1`` 或行为上限为负数。

    Returns:
        dict[str, list[dict[str, Any]]]: 保持原始相对顺序的混合后分卷。
    """
    missing = {"train", "dev", "test"} - set(splits)
    if missing:
        raise DatasetBuildError(f"SFT 混合缺少分卷: {sorted(missing)}")
    invalid_knowledge = [
        f"{split}.{category}"
        for split, limits in knowledge_limits.items()
        for category, limit in limits.items()
        if not isinstance(limit, int) or limit < -1
    ]
    invalid_actions = [
        action
        for action, limit in human_train_action_limits.items()
        if not isinstance(limit, int) or limit < 0
    ]
    if invalid_knowledge or invalid_actions:
        raise DatasetBuildError(
            f"SFT 混合上限无效: 知识={invalid_knowledge}, 行为={invalid_actions}"
        )

    rng = random.Random(seed)
    train = _select_split(
        list(splits["train"]),
        knowledge_limits.get("train", {}),
        human_train_action_limits,
        rng,
    )
    train_facts = {
        _knowledge_fact_key(row)
        for row in train
        if row.get("source") != "human_play" and row.get("category") != "arithmetic"
    }
    dev_candidates = [
        row
        for row in splits["dev"]
        if row.get("source") == "human_play"
        or row.get("category") == "arithmetic"
        or row.get("dataset_split") == "dev"
        or _knowledge_fact_key(row) in train_facts
    ]
    output = {
        "train": train,
        "dev": _select_split(
            dev_candidates,
            knowledge_limits.get("dev", {}),
            {},
            rng,
        ),
        "test": list(splits["test"]),
    }
    return output


def _knowledge_fact_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """取得同一实体同一答案共享的事实身份。

    Args:
        row (Mapping[str, Any]): 一条知识监督样本。

    Returns:
        tuple[str, str, str]: 类别、实体和规范化答案组成的身份。
    """
    messages = row.get("messages", [])
    answer = next(
        (
            message.get("content", "")
            for message in messages
            if isinstance(message, Mapping) and message.get("role") == "assistant"
        ),
        "",
    )
    return (
        str(row.get("category", "")),
        str(row.get("object_id", "")),
        " ".join(str(answer).split()),
    )


def _select_split(
    rows: list[dict[str, Any]],
    category_limits: Mapping[str, int],
    action_limits: Mapping[str, int],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """选择一个分卷中的知识和可选高频行为。

    Args:
        rows (list[dict[str, Any]]): 当前分卷的原始样本。
        category_limits (Mapping[str, int]): 知识类别上限。
        action_limits (Mapping[str, int]): 人类动作上限。
        rng (random.Random): 本次混合共用的确定性随机源。

    Returns:
        list[dict[str, Any]]: 保持原始相对顺序的选中样本。
    """
    selected: set[int] = set()
    for category in sorted(category_limits):
        indices = [
            index
            for index, row in enumerate(rows)
            if row.get("source") != "human_play" and row.get("category") == category
        ]
        selected.update(_take_indices(indices, category_limits[category], rng))

    for index, row in enumerate(rows):
        if row.get("source") != "human_play":
            continue
        action = str(row.get("action", ""))
        if action not in action_limits:
            selected.add(index)
    for action in sorted(action_limits):
        indices = [
            index
            for index, row in enumerate(rows)
            if row.get("source") == "human_play" and row.get("action") == action
        ]
        selected.update(_take_indices(indices, action_limits[action], rng))
    return [row for index, row in enumerate(rows) if index in selected]


def _take_indices(
    indices: list[int],
    limit: int,
    rng: random.Random,
) -> list[int]:
    """从一组索引中确定性抽取不超过上限的项目。

    Args:
        indices (list[int]): 同一类别或动作的原始索引。
        limit (int): 最大保留数量；``-1`` 表示全部。
        rng (random.Random): 确定性随机源。

    Returns:
        list[int]: 被选中的原始索引。
    """
    if limit < 0 or len(indices) <= limit:
        return indices
    candidates = list(indices)
    rng.shuffle(candidates)
    return candidates[:limit]
