"""把 Markdown 游戏知识与精确决策构建为可读 SFT messages。"""

import json
import re
from collections import Counter
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..game_knowledge import KnowledgeEntry, KnowledgeFormatError, parse_knowledge_entry
from ..harness import HarnessAction, build_observation, format_action, system_prompt

_CATEGORY_NAMES = {
    "cards": "卡牌",
    "characters": "角色",
    "enchantments": "附魔",
    "encounters": "遭遇",
    "events": "事件",
    "keywords": "关键词",
    "monsters": "敌人",
    "potions": "药水",
    "powers": "能力",
    "relics": "遗物",
}
_FIELD_NAMES = {
    "act": "幕",
    "card_type": "卡牌类型",
    "character": "角色",
    "cost": "费用",
    "energy": "初始能量",
    "event_type": "事件类型",
    "gold": "初始金币",
    "hp": "初始生命",
    "hp_ascension": "进阶生命",
    "innate": "固有能力",
    "is_weak": "弱遭遇",
    "max_hp": "最大生命",
    "min_hp": "最小生命",
    "orb_slots": "充能球槽",
    "pool": "池",
    "power_type": "能力类型",
    "rarity": "稀有度",
    "room": "房间类型",
    "stack_type": "叠加方式",
    "target": "目标",
    "usage": "使用方式",
}
_PROVENANCE_FIELDS = {"id", "name", "type", "source", "source_detail", "game_version"}
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_ENVIRONMENT_ACTIONS = {
    "continue_run",
    "decrease_ascension",
    "embark",
    "increase_ascension",
    "open_character_select",
    "select_character",
    "set_seed",
}


class DatasetBuildError(RuntimeError):
    """表示知识或精确决策无法可靠转换成训练样本。"""


@dataclass(frozen=True, slots=True)
class SftDatasetResult:
    """描述一次可读 SFT 数据集构建结果。

    Args:
        output_root (Path): 本次数据集目录。
        train_path (Path): 训练集 JSONL 路径。
        dev_path (Path): 开发集 JSONL 路径。
        test_path (Path): 测试集 JSONL 路径。
        manifest_path (Path): 数据来源与计数清单路径。
        train_count (int): 训练样本数。
        dev_count (int): 开发样本数。
        test_count (int): 测试样本数。
    """

    output_root: Path
    train_path: Path
    dev_path: Path
    test_path: Path
    manifest_path: Path
    train_count: int
    dev_count: int
    test_count: int


def build_sft_dataset(
    *,
    knowledge_root: Path,
    transcripts_root: Path,
    output_root: Path,
    dev_run_ids: Collection[str] = (),
    test_run_ids: Collection[str] = (),
) -> SftDatasetResult:
    """构建知识与人类行为混合的可读 SFT 数据集。

    知识样本默认进入训练集；行为样本只按整局 ID 分卷，避免同一局的相邻
    状态跨越 train/dev/test。环境初始化动作不会成为策略监督标签。

    Args:
        knowledge_root (Path): 单一知识来源的 Markdown 根目录。
        transcripts_root (Path): 精确人类决策 JSONL 目录。
        output_root (Path): 数据集输出目录。
        dev_run_ids (Collection[str]): 整局进入开发集的 run ID。
        test_run_ids (Collection[str]): 整局进入测试集的 run ID。

    Raises:
        DatasetBuildError: 分卷重叠、输入 JSON 无效或 Harness 契约不匹配。
        KnowledgeFormatError: Markdown 知识条目无效。
        OSError: 无法读取输入或写入数据集。

    Returns:
        SftDatasetResult: 三个分卷路径、清单路径与样本计数。
    """
    dev_runs = set(dev_run_ids)
    test_runs = set(test_run_ids)
    overlap = dev_runs & test_runs
    if overlap:
        raise DatasetBuildError(f"dev/test run ID 重叠: {sorted(overlap)}")

    knowledge_rows = list(_knowledge_rows(Path(knowledge_root)))
    splits: dict[str, list[dict[str, Any]]] = {
        "train": knowledge_rows,
        "dev": [],
        "test": [],
    }
    skipped_actions: Counter[str] = Counter()
    for decision in _read_transcripts(Path(transcripts_root)):
        action = decision.get("action")
        if isinstance(action, str) and action in _ENVIRONMENT_ACTIONS:
            skipped_actions[action] += 1
            continue
        row = _behavior_row(decision)
        run_id = row["run_id"]
        split = (
            "dev" if run_id in dev_runs else "test" if run_id in test_runs else "train"
        )
        splits[split].append(row)

    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    split_paths: dict[str, Path] = {}
    for name, rows in splits.items():
        split_paths[name] = destination / f"{name}.jsonl"
        _write_jsonl(split_paths[name], rows)

    sources = Counter(str(row["source"]) for rows in splits.values() for row in rows)
    manifest = {
        "format": "chat_messages",
        "splits": {name: len(rows) for name, rows in splits.items()},
        "sources": dict(sorted(sources.items())),
        "knowledge": {
            "root": str(knowledge_root),
            "game_versions": sorted(
                {
                    str(row["game_version"])
                    for row in knowledge_rows
                    if row.get("game_version")
                }
            ),
        },
        "dev_runs": sorted(dev_runs),
        "test_runs": sorted(test_runs),
        "skipped_environment_actions": dict(sorted(skipped_actions.items())),
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return SftDatasetResult(
        output_root=destination,
        train_path=split_paths["train"],
        dev_path=split_paths["dev"],
        test_path=split_paths["test"],
        manifest_path=manifest_path,
        train_count=len(splits["train"]),
        dev_count=len(splits["dev"]),
        test_count=len(splits["test"]),
    )


def _knowledge_rows(root: Path) -> Iterator[dict[str, Any]]:
    """把单实体 Markdown 逐个转换成单轮知识监督样本。

    Args:
        root (Path): 当前知识来源根目录。

    Yields:
        dict[str, Any]: 带来源、实体 ID 和 messages 的可读 SFT 行。
    """
    for knowledge_file in sorted(root.rglob("*.md")):
        if knowledge_file.name.casefold() == "readme.md":
            continue
        try:
            entry = parse_knowledge_entry(knowledge_file.read_text(encoding="utf-8"))
        except KnowledgeFormatError as exc:
            raise KnowledgeFormatError(f"{knowledge_file}: {exc}") from exc
        category = knowledge_file.parent.name
        category_name = _CATEGORY_NAMES.get(category, entry.metadata["type"])
        game_version = entry.metadata.get("game_version")
        sample_prefix = (
            f"{entry.source}/{game_version}" if game_version else entry.source
        )
        row = {
            "sample_id": f"{sample_prefix}/{category}/{entry.object_id}",
            "source": entry.source,
            "category": category,
            "object_id": entry.object_id,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"请说明《杀戮尖塔 2》中的{category_name}"
                        f"“{entry.name}”（{entry.object_id}）。"
                    ),
                },
                {"role": "assistant", "content": _knowledge_answer(entry)},
            ],
        }
        source_detail = entry.metadata.get("source_detail")
        if source_detail:
            row["source_detail"] = source_detail
        if game_version:
            row["game_version"] = game_version
        yield row


def _knowledge_answer(entry: KnowledgeEntry) -> str:
    """把条目属性和正文组合成模型可直接学习的 Markdown 答案。

    Args:
        entry (KnowledgeEntry): 已解析的单实体知识条目。

    Returns:
        str: 不包含数据来源元信息的事实答案。
    """
    lines = [f"# {entry.name}（{entry.object_id}）"]
    facts = [
        f"- {_FIELD_NAMES.get(key, key)}：{value}"
        for key, value in entry.metadata.items()
        if key not in _PROVENANCE_FIELDS and value not in {"", "None", "null"}
    ]
    if facts:
        lines.extend(("", "## 属性", *facts))
    body = _MARKDOWN_LINK.sub(r"\1", entry.body).strip()
    if body:
        lines.extend(("", body))
    return "\n".join(lines)


def _behavior_row(decision: Mapping[str, Any]) -> dict[str, Any]:
    """用当前 Harness 把一条精确决策渲染成行为监督样本。

    Args:
        decision (Mapping[str, Any]): 转录器输出的一条人类决策。

    Raises:
        DatasetBuildError: 状态、层级、动作或参数不再符合当前 Harness。

    Returns:
        dict[str, Any]: 单步观测到规范 ``ACTION:`` 的 SFT 行。
    """
    run_id = decision.get("run_id")
    sequence = decision.get("source_sequence")
    state = decision.get("before_state")
    action_name = decision.get("action")
    parameters = decision.get("parameters")
    if (
        not isinstance(run_id, str)
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or not isinstance(state, Mapping)
        or not isinstance(action_name, str)
        or not isinstance(parameters, Mapping)
    ):
        raise DatasetBuildError("精确决策缺少 SFT 构建字段")
    try:
        observation = build_observation(state)
    except ValueError as exc:
        raise DatasetBuildError(f"{run_id}:{sequence} 无法构建 Harness 观测") from exc
    recorded_layer = decision.get("recorded_layer")
    if isinstance(recorded_layer, str) and recorded_layer != observation.layer.value:
        raise DatasetBuildError(
            f"{run_id}:{sequence} 层级不一致: {recorded_layer} != "
            f"{observation.layer.value}"
        )
    if action_name not in observation.available_actions:
        raise DatasetBuildError(f"{run_id}:{sequence} 动作未向模型开放: {action_name}")
    if any(
        not isinstance(key, str)
        or isinstance(value, bool)
        or not isinstance(value, int)
        for key, value in parameters.items()
    ):
        raise DatasetBuildError(f"{run_id}:{sequence} 动作参数不是整数索引")
    try:
        action_line = format_action(
            HarnessAction(name=action_name, parameters=dict(parameters))
        )
    except ValueError as exc:
        raise DatasetBuildError(f"{run_id}:{sequence} 动作不符合 Harness") from exc
    return {
        "sample_id": f"human_play/{run_id}/{sequence}",
        "source": "human_play",
        "run_id": run_id,
        "layer": observation.layer.value,
        "messages": [
            {
                "role": "system",
                "content": system_prompt(observation.layer),
            },
            {"role": "user", "content": observation.text},
            {"role": "assistant", "content": action_line},
        ],
    }


def _read_transcripts(root: Path) -> Iterator[dict[str, Any]]:
    """按文件名和行号顺序读取精确决策 JSONL。

    Args:
        root (Path): 转录结果目录。

    Raises:
        DatasetBuildError: 某行不是有效 JSON 对象。
        OSError: 无法读取转录文件。

    Yields:
        dict[str, Any]: 一条精确人类决策。
    """
    for transcript in sorted(root.glob("*.jsonl")):
        with transcript.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetBuildError(
                        f"无效转录 JSONL: {transcript}:{line_number}"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise DatasetBuildError(
                        f"转录行不是对象: {transcript}:{line_number}"
                    )
                yield dict(value)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """以 UTF-8 JSONL 写入可读训练样本。

    Args:
        path (Path): 目标分卷文件。
        rows (list[dict[str, Any]]): 待写入的 SFT 样本。

    Returns:
        None: 所有行写入完成后返回。
    """
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
