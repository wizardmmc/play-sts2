"""把 Markdown 游戏知识与精确决策构建为可读 SFT messages。"""

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...game_knowledge import (
    KnowledgeEntry,
    KnowledgeFormatError,
    parse_knowledge_entry,
)
from ...harness import HarnessAction, build_observation, format_action, system_prompt
from ...recording.audit import RawRunIntegrityError, audit_human_run

_CATEGORY_NAMES = {
    "cards": "卡牌",
    "catalog": "目录",
    "characters": "角色",
    "ancients": "远古者",
    "arithmetic": "战斗算术",
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
    human_root: Path,
    output_root: Path,
    dev_run_ids: Collection[str] = (),
    test_run_ids: Collection[str] = (),
) -> SftDatasetResult:
    """构建知识与人类行为混合的可读 SFT 数据集。

    知识样本默认进入训练集；行为样本只按整局 ID 分卷，避免同一局的相邻
    状态跨越 train/dev/test。环境初始化动作不会成为策略监督标签。

    Args:
        knowledge_root (Path): 单一知识来源的 Markdown 根目录。
        human_root (Path): 含按局、战斗和战略分片的人类精确决策目录。
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
    declared_splits = _read_run_splits(Path(human_root) / "splits.json")
    dev_runs = set(dev_run_ids) or set(declared_splits["dev"])
    test_runs = set(test_run_ids) or set(declared_splits["test"])
    overlap = dev_runs & test_runs
    if overlap:
        raise DatasetBuildError(f"dev/test run ID 重叠: {sorted(overlap)}")

    knowledge_rows = list(_knowledge_rows(Path(knowledge_root)))
    splits: dict[str, list[dict[str, Any]]] = {
        "train": knowledge_rows,
        "dev": [],
        "test": [],
    }
    for row in _human_rows(Path(human_root)):
        run_id = str(row["run_id"])
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
        "files": {
            path.name: {
                "rows": len(splits[name]),
                "sha256": _sha256(path),
            }
            for name, path in split_paths.items()
        },
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
        "human": {
            "root": str(human_root),
            "dev_runs": sorted(dev_runs),
            "test_runs": sorted(test_runs),
        },
    }
    manifest_path = destination / "manifest.json"
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
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


def validate_sft_dataset(root: Path) -> dict[str, Any]:
    """用已发布 manifest 的 SHA-256 校验三个固定分卷。

    Args:
        root (Path): 含 ``manifest.json`` 与三个 JSONL 分卷的数据集目录。

    Raises:
        DatasetBuildError: manifest 结构或任一分卷内容不一致。
        OSError: 文件无法读取。

    Returns:
        dict[str, Any]: 已验证的 manifest。
    """
    root = Path(root)
    manifest = _read_json_object(root / "manifest.json")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise DatasetBuildError(f"SFT manifest 缺少 files: {root}")
    for split in ("train", "dev", "test"):
        name = f"{split}.jsonl"
        record = files.get(name)
        expected = record.get("sha256") if isinstance(record, Mapping) else None
        path = root / name
        if not isinstance(expected, str) or not path.is_file():
            raise DatasetBuildError(f"SFT manifest 缺少分卷记录: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise DatasetBuildError(
                f"{name} 的 SHA-256 与 manifest 不一致: {actual} != {expected}"
            )
    return manifest


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
    yield from _curated_knowledge_rows(root)


def _curated_knowledge_rows(root: Path) -> Iterator[dict[str, Any]]:
    """读取已经核验的知识问法与回答变体。

    Args:
        root (Path): 含按类别组织的 ``prompt``/``completion`` JSONL 根目录。

    Raises:
        DatasetBuildError: 某行缺少知识身份、问题或回答。
        OSError: JSONL 无法读取。

    Yields:
        dict[str, Any]: 保留原问法、答案与来源追溯的 chat messages 行。
    """
    for path in sorted(root.rglob("*.jsonl")):
        relative_stem = path.relative_to(root).with_suffix("").as_posix()
        for line_number, row in enumerate(_read_jsonl(path), start=1):
            prompt = row.get("prompt")
            completion = row.get("completion")
            category = row.get("category", path.parent.name)
            object_id = row.get("object_id", path.stem)
            source = row.get("source", "curated_knowledge")
            if not all(
                isinstance(value, str) and value.strip()
                for value in (prompt, completion, category, object_id, source)
            ):
                raise DatasetBuildError(f"知识问答字段无效: {path}:{line_number}")
            question = re.sub(r"\s*A:\s*$", "", prompt).strip()
            yield {
                "sample_id": f"curated/{relative_stem}/{line_number:04d}",
                "source": "curated_knowledge",
                "source_detail": source,
                "category": category,
                "object_id": object_id,
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": completion.strip()},
                ],
            }


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


def _behavior_row(
    decision: Mapping[str, Any],
    *,
    run_id: str | None = None,
    battle_key: str | None = None,
) -> dict[str, Any]:
    """用当前 Harness 把一条精确决策渲染成行为监督样本。

    Args:
        decision (Mapping[str, Any]): Raw 中的一条精确人类决策。
        run_id (str | None): 从当前局 meta 取得的稳定局 ID。
        battle_key (str | None): 战斗分片名；战略动作省略。

    Raises:
        DatasetBuildError: 状态、层级、动作或参数不再符合当前 Harness。

    Returns:
        dict[str, Any]: 单步观测到规范 ``ACTION:`` 的 SFT 行。
    """
    resolved_run_id = run_id or decision.get("run_id")
    sequence = decision.get("event_id", decision.get("source_sequence"))
    state = decision.get("before_state")
    action_name = decision.get("action")
    parameters = decision.get("parameters")
    if (
        not isinstance(resolved_run_id, str)
        or not _is_event_id(sequence)
        or not isinstance(state, Mapping)
        or not isinstance(action_name, str)
        or not isinstance(parameters, Mapping)
    ):
        raise DatasetBuildError("精确决策缺少 SFT 构建字段")
    try:
        observation = build_observation(state)
    except ValueError as exc:
        raise DatasetBuildError(
            f"{resolved_run_id}:{sequence} 无法构建 Harness 观测"
        ) from exc
    recorded_layer = decision.get("recorded_layer", decision.get("layer"))
    if isinstance(recorded_layer, str) and recorded_layer != observation.layer.value:
        raise DatasetBuildError(
            f"{resolved_run_id}:{sequence} 层级不一致: {recorded_layer} != "
            f"{observation.layer.value}"
        )
    if action_name not in observation.available_actions:
        raise DatasetBuildError(
            f"{resolved_run_id}:{sequence} 动作未向模型开放: {action_name}"
        )
    if any(
        not isinstance(key, str)
        or isinstance(value, bool)
        or not isinstance(value, int)
        for key, value in parameters.items()
    ):
        raise DatasetBuildError(f"{resolved_run_id}:{sequence} 动作参数不是整数索引")
    try:
        action_line = format_action(
            HarnessAction(name=action_name, parameters=dict(parameters))
        )
    except ValueError as exc:
        raise DatasetBuildError(
            f"{resolved_run_id}:{sequence} 动作不符合 Harness"
        ) from exc
    row = {
        "sample_id": f"human_play/{resolved_run_id}/{sequence}",
        "source": "human_play",
        "run_id": resolved_run_id,
        "layer": observation.layer.value,
        "screen": str(state.get("screen", "UNKNOWN")),
        "action": action_name,
        "tags": _behavior_tags(state, action_name, parameters),
        "messages": [
            {
                "role": "system",
                "content": system_prompt(observation.layer),
            },
            {"role": "user", "content": observation.text},
            {"role": "assistant", "content": action_line},
        ],
    }
    if battle_key is not None:
        row["battle_key"] = battle_key
    return row


def _behavior_tags(
    state: Mapping[str, Any],
    action: str,
    parameters: Mapping[str, Any],
) -> list[str]:
    """为后续定向采样派生少量稳定行为标签。

    Args:
        state (Mapping[str, Any]): 动作前完整状态。
        action (str): 已执行动作名。
        parameters (Mapping[str, Any]): 动作参数。

    Returns:
        list[str]: 按名称排序且不重复的行为标签。
    """
    tags = {action}
    if action == "use_potion":
        tags.add("potion_use")
    if action == "choose_rest_option":
        rest = state.get("rest")
        options = rest.get("options") if isinstance(rest, Mapping) else None
        option_index = parameters.get("option_index")
        if isinstance(options, list) and isinstance(option_index, int):
            selected = next(
                (
                    option
                    for option in options
                    if isinstance(option, Mapping)
                    and option.get("index") == option_index
                ),
                None,
            )
            option_id = selected.get("option_id") if selected is not None else None
            if isinstance(option_id, str) and option_id:
                tags.add(f"rest:{option_id.casefold()}")
    return sorted(tags)


def _is_event_id(value: object) -> bool:
    """判断事件身份是否为 Mod 编号或人工确认的稳定字符串。

    Args:
        value (object): Raw 行中的事件身份。

    Returns:
        bool: 整数（不含布尔值）或非空字符串时为真。
    """
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        or isinstance(value, str)
        and bool(value.strip())
    )


def _human_rows(root: Path) -> Iterator[dict[str, Any]]:
    """读取按局分片的人类精确动作并恢复训练消息。

    Args:
        root (Path): ``data/raw/human`` 风格的人类数据根目录。

    Raises:
        DatasetBuildError: 元数据、分片或消息不符合当前数据契约。
        OSError: 无法读取人类数据。

    Yields:
        dict[str, Any]: 战斗或战略的独立单步行为监督样本。
    """
    if not root.is_dir():
        raise DatasetBuildError(f"人类数据目录不存在: {root}")
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if run_dir.name.startswith("."):
            continue
        meta_path = run_dir / "meta.json"
        if not meta_path.is_file():
            continue
        metadata = _read_json_object(meta_path)
        termination_reason = metadata.get("termination_reason")
        if not isinstance(termination_reason, str) or not termination_reason.strip():
            continue
        if metadata.get("training_eligible") is False:
            continue
        try:
            audit = audit_human_run(run_dir)
        except RawRunIntegrityError as exc:
            raise DatasetBuildError(str(exc)) from exc
        metadata = audit.metadata
        run_id = metadata.get("run_id", run_dir.name)
        if not isinstance(run_id, str) or not run_id:
            raise DatasetBuildError(f"人类局缺少 run_id: {meta_path}")
        for battle_path in audit.battle_paths:
            for row in _read_jsonl(battle_path):
                yield _behavior_row(
                    row,
                    run_id=run_id,
                    battle_key=battle_path.stem,
                )
        if audit.strategy_path is not None:
            for row in _read_jsonl(audit.strategy_path):
                yield _behavior_row(row, run_id=run_id)


def _read_run_splits(path: Path) -> dict[str, list[str]]:
    """读取可选的整局 train/dev/test 分卷。

    Args:
        path (Path): 人类数据根目录下的 ``splits.json``。

    Raises:
        DatasetBuildError: 分卷字段无效或整局跨卷泄漏。

    Returns:
        dict[str, list[str]]: 总是包含三个分卷键的字符串数组。
    """
    if not path.is_file():
        return {"train": [], "dev": [], "test": []}
    value = _read_json_object(path)
    splits: dict[str, list[str]] = {}
    all_runs: list[str] = []
    for name in ("train", "dev", "test"):
        runs = value.get(name)
        if not isinstance(runs, list) or any(not isinstance(run, str) for run in runs):
            raise DatasetBuildError(f"无效人类分卷: {path}:{name}")
        splits[name] = runs
        all_runs.extend(runs)
    if len(all_runs) != len(set(all_runs)):
        raise DatasetBuildError("同一人类局出现在多个分卷")
    return splits


def _read_json_object(path: Path) -> dict[str, Any]:
    """读取顶层必须为对象的 JSON 文件。

    Args:
        path (Path): 目标 JSON 文件。

    Raises:
        DatasetBuildError: JSON 无效或顶层不是对象。
        OSError: 文件无法读取。

    Returns:
        dict[str, Any]: 解码后的普通字典。
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetBuildError(f"无效 JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise DatasetBuildError(f"JSON 顶层不是对象: {path}")
    return dict(value)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """逐行读取只包含 JSON 对象的文件。

    Args:
        path (Path): 待读取的 JSONL。

    Raises:
        DatasetBuildError: 某行 JSON 无效或不是对象。
        OSError: 文件无法读取。

    Yields:
        dict[str, Any]: 保持原始行顺序的对象。
    """
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetBuildError(f"无效 JSONL: {path}:{line_number}") from exc
            if not isinstance(value, Mapping):
                raise DatasetBuildError(f"JSONL 行不是对象: {path}:{line_number}")
            yield dict(value)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """以 UTF-8 JSONL 原子替换一个可读训练分卷。

    Args:
        path (Path): 目标分卷文件。
        rows (list[dict[str, Any]]): 待写入的 SFT 样本。

    Returns:
        None: 所有行写入完成后返回。
    """
    _atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
        ),
    )


def _atomic_write_text(path: Path, content: str) -> None:
    """同文件系统写完临时文件后原子替换目标文本。

    Args:
        path (Path): 最终文件路径。
        content (str): 完整 UTF-8 文本。

    Raises:
        OSError: 暂存写入或原子替换失败。

    Returns:
        None: 目标文件完整替换后返回。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    """计算文件内容的十六进制 SHA-256。

    Args:
        path (Path): 待读取的文件。

    Returns:
        str: 64 位小写十六进制摘要。
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
