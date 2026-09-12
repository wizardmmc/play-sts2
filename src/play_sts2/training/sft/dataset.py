"""把 Markdown 游戏知识与精确决策构建为可读 SFT messages。"""

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...game_knowledge import (
    KnowledgeEntry,
    KnowledgeFormatError,
    parse_knowledge_entry,
)
from ...harness import (
    HarnessAction,
    build_observation,
    format_action,
    legal_action_lines,
    model_actions,
    shop_purchase_available,
    system_prompt,
)
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
SFT_SPLIT_PATHS = {
    "train": Path("train"),
    "dev": Path("validation"),
    "test": Path("eval"),
}
_PUBLIC_SPLIT_NAMES = {"train": "train", "dev": "validation", "test": "eval"}
_SPLIT_ALIASES = {
    "train": "train",
    "validation": "dev",
    "dev": "dev",
    "eval": "test",
    "test": "test",
}
_REQUIRED_KNOWLEDGE_CATEGORIES = {
    "cards",
    "characters",
    "enchantments",
    "encounters",
    "events",
    "keywords",
    "monsters",
    "potions",
    "powers",
    "relics",
}


class DatasetBuildError(RuntimeError):
    """表示知识或精确决策无法可靠转换成训练样本。"""


@dataclass(frozen=True, slots=True)
class SftDatasetResult:
    """描述一次可读 SFT 数据集构建结果。

    Args:
        output_root (Path): 本次数据集目录。
        train_path (Path): 训练集目录。
        dev_path (Path): 验证集目录。
        test_path (Path): 最终评测集目录。
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
    additional_human_roots: Sequence[Path] = (),
    output_root: Path,
    train_run_ids: Collection[str] = (),
    dev_run_ids: Collection[str] = (),
    test_run_ids: Collection[str] = (),
    run_splits_path: Path | None = None,
    dagger_root: Path | None = None,
    dagger_policy_version: str | None = None,
    mix_config_path: Path | None = None,
) -> SftDatasetResult:
    """构建三棵同构且可逐实体审查的 SFT 数据目录。

    生成知识必须通过 ``question_role`` 明确声明 train、validation 或 eval 用途；
    行为样本只按整局 ID 分卷。带正式知识 manifest 的输入还会强制检查完整事实、
    类别、六类算术、战斗和战略覆盖。

    Args:
        knowledge_root (Path): 含规范事实或生成问答候选的知识根目录。
        human_root (Path): 含按局、战斗和战略分片的人类精确决策目录。
        additional_human_roots (Sequence[Path]): 需要按各自 splits 合并的附加 raw 根。
        output_root (Path): 数据集输出目录。
        train_run_ids (Collection[str]): 整局进入训练集的 run ID。
        dev_run_ids (Collection[str]): 整局进入 validation 的 run ID。
        test_run_ids (Collection[str]): 整局进入 eval 的 run ID。
        run_splits_path (Path | None): 可选的跨 raw 根独立整局分卷名册。
        dagger_root (Path | None): 可选的学生状态 CombatSolver 标签目录。
        dagger_policy_version (str | None): DAgger 标签唯一允许的学生父 policy。
        mix_config_path (Path | None): 可选的训练高频行为上限配方。

    Raises:
        DatasetBuildError: 分卷重叠、输入 JSON 无效或 Harness 契约不匹配。
        KnowledgeFormatError: Markdown 知识条目无效。
        OSError: 无法读取输入或写入数据集。

    Returns:
        SftDatasetResult: 三个分卷目录、清单路径与样本计数。
    """
    human_roots = (Path(human_root), *(Path(root) for root in additional_human_roots))
    if (dagger_root is None) != (dagger_policy_version is None):
        raise DatasetBuildError("dagger_root 与 dagger_policy_version 必须同时提供")
    explicit_ids = any((train_run_ids, dev_run_ids, test_run_ids))
    if run_splits_path is not None and explicit_ids:
        raise DatasetBuildError(
            "不能同时使用 run_splits_path 与逐局 train/dev/test 参数"
        )
    declared_splits = {"train": [], "dev": [], "test": []}
    root_declared_splits: list[dict[str, list[str]]] = []
    for root in human_roots:
        root_splits = _read_run_splits(root / "splits.json")
        root_declared_splits.append(root_splits)
        for name, runs in declared_splits.items():
            runs.extend(root_splits[name])
    explicit = run_splits_path is not None or explicit_ids
    if run_splits_path is not None:
        split_path = Path(run_splits_path)
        if not split_path.is_file():
            raise DatasetBuildError(f"显式人类分卷不存在: {split_path}")
        selected_splits = _read_run_splits(split_path)
        if not set().union(*selected_splits.values()):
            raise DatasetBuildError(f"显式人类分卷为空: {split_path}")
    elif explicit_ids:
        selected_splits = {
            "train": list(train_run_ids),
            "dev": list(dev_run_ids),
            "test": list(test_run_ids),
        }
    else:
        selected_splits = declared_splits
    run_splits = {name: set(selected_splits[name]) for name in ("train", "dev", "test")}
    _validate_run_split_sets(run_splits)
    assigned_runs = set().union(*run_splits.values())

    knowledge_rows = _deduplicate_knowledge_rows(
        list(_knowledge_rows(Path(knowledge_root)))
    )
    splits: dict[str, list[dict[str, Any]]] = {
        "train": [row for row in knowledge_rows if row.get("dataset_split") == "train"],
        "dev": [row for row in knowledge_rows if row.get("dataset_split") == "dev"],
        "test": [row for row in knowledge_rows if row.get("dataset_split") == "test"],
    }
    if explicit:
        human_rows = [
            row
            for root in human_roots
            for row in _human_rows(root, included_run_ids=assigned_runs)
        ]
    else:
        human_rows = list(_human_rows(human_roots[0]))
        for root, root_splits in zip(
            human_roots[1:],
            root_declared_splits[1:],
            strict=True,
        ):
            selected_runs = set().union(*root_splits.values())
            human_rows.extend(_human_rows(root, included_run_ids=selected_runs))
    observed_runs = {str(row["run_id"]) for row in human_rows}
    missing_runs = assigned_runs - observed_runs
    if missing_runs:
        raise DatasetBuildError(
            f"名册中的人类局 {sorted(missing_runs)} 不存在或不可训练"
        )
    unassigned = observed_runs - assigned_runs
    if unassigned:
        raise DatasetBuildError(f"可训练人类局未分配到名册: {sorted(unassigned)}")
    for row in human_rows:
        run_id = str(row["run_id"])
        split = next(name for name, runs in run_splits.items() if run_id in runs)
        splits[split].append(row)
    dagger_rows: list[dict[str, Any]] = []
    if dagger_root is not None:
        from ..rl.dagger import load_dagger_sft_rows

        dagger_rows = load_dagger_sft_rows(
            dagger_root,
            expected_policy_version=dagger_policy_version or "",
        )
        splits["train"].extend(dagger_rows)
    mix_manifest = None
    if mix_config_path is not None:
        from .mix import apply_sft_mix, load_sft_mix

        mix_config = load_sft_mix(mix_config_path)
        splits = apply_sft_mix(
            splits,
            seed=mix_config.seed,
            human_train_action_limits=mix_config.human_train_action_limits,
            human_oversample_per_action=mix_config.human_oversample_per_action,
            arithmetic_train_per_kind=mix_config.arithmetic_train_per_kind,
        )
        mix_manifest = {
            "seed": mix_config.seed,
            "human_train_action_limits": mix_config.human_train_action_limits,
            "human_oversample_per_action": mix_config.human_oversample_per_action,
            "arithmetic_train_per_kind": mix_config.arithmetic_train_per_kind,
        }
        if mix_config.human_game_version is not None:
            mix_manifest["human_game_version"] = mix_config.human_game_version
    _validate_dataset_identity(splits)
    formal_manifest = Path(knowledge_root) / "_knowledge_manifest.json"
    is_formal = formal_manifest.is_file()
    if is_formal:
        _validate_formal_coverage(knowledge_rows, splits)

    sources = Counter(str(row["source"]) for rows in splits.values() for row in rows)
    sources_by_split = {
        _PUBLIC_SPLIT_NAMES[name]: dict(
            sorted(Counter(str(row["source"]) for row in rows).items())
        )
        for name, rows in splits.items()
    }
    categories_by_split = {
        _PUBLIC_SPLIT_NAMES[name]: dict(
            sorted(
                Counter(
                    str(row["category"])
                    for row in rows
                    if row.get("source") != "human_play"
                ).items()
            )
        )
        for name, rows in splits.items()
    }
    destination = Path(output_root)
    split_paths = {
        name: destination / relative for name, relative in SFT_SPLIT_PATHS.items()
    }
    manifest = {
        "format": "chat_messages_directory_tree",
        "splits": {
            _PUBLIC_SPLIT_NAMES[name]: len(rows) for name, rows in splits.items()
        },
        "sources": dict(sorted(sources.items())),
        "sources_by_split": sources_by_split,
        "knowledge_categories_by_split": categories_by_split,
        "coverage": _coverage_manifest(knowledge_rows, splits),
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
            "additional_roots": [str(root) for root in human_roots[1:]],
            **(
                {"split_manifest": str(run_splits_path)}
                if run_splits_path is not None
                else {}
            ),
            "train_runs": sorted(run_splits["train"]),
            "validation_runs": sorted(run_splits["dev"]),
            "eval_runs": sorted(run_splits["test"]),
            **_human_provenance_manifest(human_roots, human_rows, run_splits),
        },
        "behavior_audit": _behavior_audit(splits),
    }
    if mix_manifest is not None:
        manifest["mix"] = mix_manifest
        if mix_config.human_game_version is not None:
            manifest["human"]["game_version"] = mix_config.human_game_version
    if dagger_root is not None:
        manifest["dagger"] = {
            "root": str(dagger_root),
            "samples": len(dagger_rows),
            "student_policy_version": dagger_policy_version,
            "game_versions": dict(
                sorted(Counter(str(row["game_version"]) for row in dagger_rows).items())
            ),
            "solver_versions": dict(
                sorted(
                    Counter(str(row["solver_version"]) for row in dagger_rows).items()
                )
            ),
        }
    arithmetic_provenance = _arithmetic_provenance(Path(knowledge_root))
    if arithmetic_provenance is not None:
        manifest["arithmetic"] = arithmetic_provenance
    manifest_path = _write_dataset_tree(destination, splits, manifest)
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


def _arithmetic_provenance(knowledge_root: Path) -> dict[str, Any] | None:
    """提取合成算术候选的输入与随机种子血缘。

    Args:
        knowledge_root (Path): 同时包含知识和算术候选的根目录。

    Raises:
        DatasetBuildError: 算术 manifest 存在但缺少必需来源字段。

    Returns:
        dict[str, Any] | None: 可嵌入 SFT manifest 的精简来源；不存在时为空。
    """
    path = Path(knowledge_root) / "arithmetic/_manifest.json"
    if not path.is_file():
        return None
    manifest = _read_json_object(path)
    required = ("source", "human_root", "training_run_ids", "seed")
    if any(field not in manifest for field in required):
        raise DatasetBuildError(f"算术候选 manifest 缺少来源字段: {path}")
    return {
        "candidate_manifest": path.relative_to(knowledge_root).as_posix(),
        "candidate_manifest_sha256": _sha256(path),
        "source": manifest["source"],
        "human_root": manifest["human_root"],
        "training_run_ids": manifest["training_run_ids"],
        "seed": manifest["seed"],
    }


def validate_sft_dataset(root: Path) -> dict[str, Any]:
    """用已发布 manifest 校验递归 JSONL 文件集合、行数与摘要。

    Args:
        root (Path): 含 ``manifest.json`` 与三棵分卷目录的数据集根目录。

    Raises:
        DatasetBuildError: manifest 结构或任一分卷内容不一致。
        OSError: 文件无法读取。

    Returns:
        dict[str, Any]: 已验证的 manifest。
    """
    root = Path(root)
    manifest = _read_json_object(root / "manifest.json")
    behavior_audit = _read_json_object(root / "behavior-audit.json")
    if behavior_audit != manifest.get("behavior_audit"):
        raise DatasetBuildError("SFT 行为审计与 manifest 不一致")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise DatasetBuildError(f"SFT manifest 缺少 files: {root}")
    missing_directories = [
        relative.as_posix()
        for relative in SFT_SPLIT_PATHS.values()
        if not (root / relative).is_dir()
    ]
    if missing_directories:
        raise DatasetBuildError(f"SFT 分卷目录缺失: {missing_directories}")
    legacy_paths = (
        root / "train.jsonl",
        root / "validation/dev.jsonl",
        root / "eval/test.jsonl",
        root / "eval/knowledge",
    )
    existing_legacy = [
        path.relative_to(root).as_posix() for path in legacy_paths if path.exists()
    ]
    if existing_legacy:
        raise DatasetBuildError(f"SFT 仍含旧聚合或 probe 产物: {existing_legacy}")
    declared_names = {str(name) for name in files}
    actual_names = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.jsonl")
        if path.is_file()
    }
    if declared_names != actual_names:
        raise DatasetBuildError(
            "SFT manifest 文件集合不一致: "
            f"缺失={sorted(declared_names - actual_names)}, "
            f"未声明={sorted(actual_names - declared_names)}"
        )
    for name in sorted(declared_names):
        record = files[name]
        expected = record.get("sha256") if isinstance(record, Mapping) else None
        expected_rows = record.get("rows") if isinstance(record, Mapping) else None
        path = root / name
        if not isinstance(expected, str) or not isinstance(expected_rows, int):
            raise DatasetBuildError(f"SFT manifest 文件记录无效: {name}")
        actual = _sha256(path)
        if actual != expected:
            raise DatasetBuildError(
                f"{name} 的 SHA-256 与 manifest 不一致: {actual} != {expected}"
            )
        actual_rows = sum(1 for _ in _read_jsonl(path))
        if actual_rows != expected_rows:
            raise DatasetBuildError(
                f"{name} 的行数与 manifest 不一致: {actual_rows} != {expected_rows}"
            )
    return manifest


def dataset_split_path(root: Path, split: str) -> Path:
    """返回固定目录布局中的 SFT 分卷路径。

    Args:
        root (Path): 含 ``manifest.json`` 的 SFT 数据集根目录。
        split (str): ``train``、``validation``/``dev`` 或 ``eval``/``test``。

    Raises:
        DatasetBuildError: 分卷名称不受支持。

    Returns:
        Path: 训练、验证或最终测试目录的完整路径。
    """
    try:
        relative = SFT_SPLIT_PATHS[_SPLIT_ALIASES[split]]
    except KeyError as exc:
        raise DatasetBuildError(f"未知 SFT 分卷: {split}") from exc
    return Path(root) / relative


def dataset_split_files(root: Path, split: str) -> list[Path]:
    """返回某个 SFT 分卷下按路径稳定排序的全部 JSONL。

    Args:
        root (Path): SFT 数据集根目录。
        split (str): 支持的公开或兼容分卷名称。

    Returns:
        list[Path]: 递归发现的非空或空 JSONL 文件路径。
    """
    return sorted(dataset_split_path(root, split).rglob("*.jsonl"))


def _validate_formal_coverage(
    knowledge_rows: list[dict[str, Any]],
    splits: Mapping[str, list[dict[str, Any]]],
) -> None:
    """验证正式候选的事实、题型和三分卷行为覆盖。

    Args:
        knowledge_rows (list[dict[str, Any]]): 尚未混入人类行为的全部知识行。
        splits (Mapping[str, list[dict[str, Any]]]): 已分配并应用行为上限的样本。

    Raises:
        DatasetBuildError: 事实角色、答案、类别、算术或行为覆盖不完整。

    Returns:
        None: 正式数据满足发布契约时返回。
    """
    fact_rows: dict[str, list[dict[str, Any]]] = {}
    for row in knowledge_rows:
        category = str(row.get("category", ""))
        if category == "ancients":
            raise DatasetBuildError("正式 SFT 不允许包含远古者知识")
        if category == "arithmetic":
            continue
        fact_id = row.get("fact_id")
        if not isinstance(fact_id, str) or not fact_id.strip():
            raise DatasetBuildError(f"正式知识缺少 fact_id: {row.get('sample_id')}")
        fact_rows.setdefault(fact_id, []).append(row)
    if not fact_rows:
        raise DatasetBuildError("正式知识候选没有可训练事实")
    for fact_id, rows in fact_rows.items():
        roles = Counter(str(row.get("question_role", "")) for row in rows)
        if roles["train"] != 5 or roles["validation"] != 1 or roles["eval"] != 1:
            raise DatasetBuildError(
                f"事实 {fact_id} 必须有 train=5、validation=1、eval=1: {dict(roles)}"
            )
        training_epochs = sorted(
            row.get("training_epoch")
            for row in rows
            if row.get("question_role") == "train"
        )
        if training_epochs != [1, 2, 3, 4, 5]:
            raise DatasetBuildError(
                f"事实 {fact_id} 的训练轮次必须为 [1, 2, 3, 4, 5]: {training_epochs}"
            )
        if any(
            "training_epoch" in row
            for row in rows
            if row.get("question_role") != "train"
        ):
            raise DatasetBuildError(f"事实 {fact_id} 的留出问法不能声明训练轮次")
        answers = {_knowledge_answer_text(row) for row in rows}
        if len(answers) != 1:
            raise DatasetBuildError(f"事实 {fact_id} 的三套问法答案不一致")

    candidate_categories = {
        str(row["category"])
        for row in knowledge_rows
        if row.get("category") not in {"arithmetic", "ancients"}
    }
    if candidate_categories != _REQUIRED_KNOWLEDGE_CATEGORIES:
        raise DatasetBuildError(
            "正式知识候选类别不完整: "
            f"缺失={sorted(_REQUIRED_KNOWLEDGE_CATEGORIES - candidate_categories)}, "
            f"多余={sorted(candidate_categories - _REQUIRED_KNOWLEDGE_CATEGORIES)}"
        )
    candidate_fact_ids = {
        category: {
            str(row["fact_id"])
            for row in knowledge_rows
            if row.get("category") == category
        }
        for category in _REQUIRED_KNOWLEDGE_CATEGORIES
    }
    candidate_object_ids = {
        category: {
            str(row["object_id"])
            for row in knowledge_rows
            if row.get("category") == category
        }
        for category in _REQUIRED_KNOWLEDGE_CATEGORIES
    }
    arithmetic_buckets = {
        "block_math",
        "energy_math",
        "lethal",
        "multihit",
        "orb_focus",
        "status_math",
    }
    arithmetic_case_owners: dict[str, str] = {}
    for split, rows in splits.items():
        public_name = _PUBLIC_SPLIT_NAMES[split]
        categories = {
            str(row["category"])
            for row in rows
            if row.get("source") != "human_play" and row.get("category") != "arithmetic"
        }
        if categories != _REQUIRED_KNOWLEDGE_CATEGORIES:
            raise DatasetBuildError(
                f"{public_name} 知识类别覆盖不完整: "
                f"缺失={sorted(_REQUIRED_KNOWLEDGE_CATEGORIES - categories)}"
            )
        split_knowledge = [
            row for row in rows if row.get("category") in _REQUIRED_KNOWLEDGE_CATEGORIES
        ]
        invalid_roles = sorted(
            {
                str(row.get("question_role", ""))
                for row in split_knowledge
                if row.get("question_role") != public_name
            }
        )
        if invalid_roles:
            raise DatasetBuildError(
                f"{public_name} 知识问法用途不一致: {invalid_roles}"
            )
        for category in sorted(_REQUIRED_KNOWLEDGE_CATEGORIES):
            observed_fact_ids = {
                str(row["fact_id"])
                for row in split_knowledge
                if row.get("category") == category
            }
            missing_facts = candidate_fact_ids[category] - observed_fact_ids
            extra_facts = observed_fact_ids - candidate_fact_ids[category]
            if missing_facts or extra_facts:
                raise DatasetBuildError(
                    f"{public_name} 知识事实覆盖不完整({category}): "
                    f"缺失={sorted(missing_facts)}, 多余={sorted(extra_facts)}"
                )
            observed_object_ids = {
                str(row["object_id"])
                for row in split_knowledge
                if row.get("category") == category
            }
            missing_objects = candidate_object_ids[category] - observed_object_ids
            extra_objects = observed_object_ids - candidate_object_ids[category]
            if missing_objects or extra_objects:
                raise DatasetBuildError(
                    f"{public_name} 知识实体覆盖不完整({category}): "
                    f"缺失={sorted(missing_objects)}, 多余={sorted(extra_objects)}"
                )
        observed_buckets = {
            str(row["object_id"]) for row in rows if row.get("category") == "arithmetic"
        }
        if observed_buckets != arithmetic_buckets:
            raise DatasetBuildError(
                f"{public_name} 算术题型覆盖不完整: "
                f"缺失={sorted(arithmetic_buckets - observed_buckets)}"
            )
        arithmetic_rows = [row for row in rows if row.get("category") == "arithmetic"]
        for row in arithmetic_rows:
            if row.get("question_role") != public_name:
                raise DatasetBuildError(
                    f"{public_name} 算术问法用途不一致: {row.get('question_role')}"
                )
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or not case_id.strip():
                raise DatasetBuildError(
                    f"{public_name} 算术候选缺少 case_id: {row.get('sample_id')}"
                )
            previous_owner = arithmetic_case_owners.get(case_id)
            if previous_owner is not None:
                raise DatasetBuildError(
                    f"算术案例跨分卷重复: {case_id} ({previous_owner}/{public_name})"
                )
            arithmetic_case_owners[case_id] = public_name
        layers = {
            str(row.get("layer", ""))
            for row in rows
            if row.get("source") == "human_play"
        }
        missing_layers = {"battle", "strategic"} - layers
        if missing_layers:
            raise DatasetBuildError(
                f"{public_name} 人类行为覆盖不完整: {sorted(missing_layers)}"
            )


def _coverage_manifest(
    knowledge_rows: list[dict[str, Any]],
    splits: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """汇总每个知识类别的候选及三种用途覆盖。

    Args:
        knowledge_rows (list[dict[str, Any]]): 全部知识候选。
        splits (Mapping[str, list[dict[str, Any]]]): 最终三分卷样本。

    Returns:
        dict[str, dict[str, Any]]: 按类别组织的实体、事实与问法计数。
    """
    categories = sorted(
        {
            str(row["category"])
            for row in knowledge_rows
            if row.get("category") != "ancients"
        }
    )
    output: dict[str, dict[str, Any]] = {}
    for category in categories:
        candidates = [row for row in knowledge_rows if row.get("category") == category]
        record: dict[str, Any] = {
            "candidate_entities": len({str(row["object_id"]) for row in candidates}),
            "candidate_facts": len({str(row.get("fact_id", "")) for row in candidates}),
            "candidate_questions": len(candidates),
        }
        for split, rows in splits.items():
            selected = [row for row in rows if row.get("category") == category]
            record[_PUBLIC_SPLIT_NAMES[split]] = {
                "entities": len({str(row["object_id"]) for row in selected}),
                "facts": len({str(row.get("fact_id", "")) for row in selected}),
                "questions": len(selected),
            }
        output[category] = record
    return output


def _write_dataset_tree(
    destination: Path,
    splits: Mapping[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
) -> Path:
    """在同级暂存目录构建完整树后替换旧数据产物。

    Args:
        destination (Path): 正式 SFT 数据根目录。
        splits (Mapping[str, list[dict[str, Any]]]): 三个待发布分卷。
        manifest (dict[str, Any]): 尚未附加文件摘要的构建清单。

    Raises:
        OSError: 暂存写入或目录替换失败。

    Returns:
        Path: 最终 ``manifest.json`` 路径。
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-build-", dir=destination.parent)
    )
    categories = sorted(
        {
            str(row["category"])
            for rows in splits.values()
            for row in rows
            if row.get("source") != "human_play"
        }
    )
    try:
        file_rows: dict[Path, list[dict[str, Any]]] = {}
        for split, rows in splits.items():
            split_root = staging / SFT_SPLIT_PATHS[split]
            for category in (*categories, "combat", "strategy"):
                (split_root / category).mkdir(parents=True, exist_ok=True)
            for row in rows:
                relative = _row_output_path(row)
                file_rows.setdefault(split_root / relative, []).append(row)
        for path, rows in sorted(file_rows.items(), key=lambda item: str(item[0])):
            _write_jsonl(path, [_published_row(row) for row in rows])
        manifest["files"] = {
            path.relative_to(staging).as_posix(): {
                "rows": len(rows),
                "sha256": _sha256(path),
            }
            for path, rows in sorted(file_rows.items(), key=lambda item: str(item[0]))
        }
        _atomic_write_text(
            staging / "behavior-audit.json",
            json.dumps(manifest["behavior_audit"], ensure_ascii=False, indent=2) + "\n",
        )
        _atomic_write_text(
            staging / "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        backup = destination.parent / f".{destination.name}-previous-{os.getpid()}"
        if backup.exists():
            raise OSError(f"SFT 备份目录已存在: {backup}")
        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except BaseException:
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination / "manifest.json"


def _published_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """移除仅供构建器路由使用的内部字段。

    Args:
        row (Mapping[str, Any]): 已完成分卷的内存样本。

    Returns:
        dict[str, Any]: 只保留可供人工审查和训练消费的公开字段。
    """
    internal = {"dataset_split", "_human_root"}
    return {key: value for key, value in row.items() if key not in internal}


def _row_output_path(row: Mapping[str, Any]) -> Path:
    """确定一条 SFT 行在分卷目录内的可读文件路径。

    Args:
        row (Mapping[str, Any]): 知识、算术或人类行为样本。

    Raises:
        DatasetBuildError: 行缺少构成安全相对路径的身份字段。

    Returns:
        Path: 不含分卷根目录的 JSONL 相对路径。
    """
    if row.get("source") == "human_play":
        run_id = str(row.get("run_id", ""))
        if not run_id or "/" in run_id or ".." in run_id:
            raise DatasetBuildError(f"人类行为 run_id 无效: {run_id}")
        if row.get("layer") == "battle":
            battle_key = str(row.get("battle_key", ""))
            if not battle_key or "/" in battle_key or ".." in battle_key:
                raise DatasetBuildError(f"战斗分片身份无效: {battle_key}")
            return Path("combat") / run_id / f"{battle_key}.jsonl"
        return Path("strategy") / f"{run_id}.jsonl"
    category = str(row.get("category", ""))
    object_id = str(row.get("object_id", ""))
    if (
        not category
        or not object_id
        or any(
            token in value for value in (category, object_id) for token in ("/", "..")
        )
    ):
        raise DatasetBuildError(f"知识输出身份无效: {category}/{object_id}")
    return Path(category) / f"{object_id}.jsonl"


def _validate_run_split_sets(run_splits: Mapping[str, set[str]]) -> None:
    """确认整局归属名册中的三个集合互不重叠。

    Args:
        run_splits (Mapping[str, set[str]]): 分卷名到 run ID 集合的映射。

    Raises:
        DatasetBuildError: 同一整局出现在多个分卷。

    Returns:
        None: 名册互斥时返回。
    """
    owners: dict[str, str] = {}
    for split, runs in run_splits.items():
        for run_id in runs:
            previous = owners.get(run_id)
            if previous is not None:
                raise DatasetBuildError(
                    f"同一人类局出现在多个分卷: {run_id} ({previous}/{split})"
                )
            owners[run_id] = split


def _deduplicate_knowledge_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按规范化问题去除同答案重复项并拒绝答案冲突。

    Args:
        rows (list[dict[str, Any]]): 尚未分卷的知识 SFT 行。

    Raises:
        DatasetBuildError: 相同问题对应不同答案。

    Returns:
        list[dict[str, Any]]: 保留首次出现顺序的唯一知识行。
    """
    seen: dict[str, tuple[str, str]] = {}
    unique: list[dict[str, Any]] = []
    for row in rows:
        prompt = _knowledge_prompt(row)
        answer = _knowledge_answer_text(row)
        role = str(row.get("question_role", ""))
        previous = seen.get(prompt)
        if previous is None:
            seen[prompt] = (answer, role)
            unique.append(row)
        elif previous[0] != answer:
            raise DatasetBuildError(f"相同知识问题存在不同答案: {prompt}")
        elif previous[1] != role:
            unique.append(row)
    return unique


def _validate_dataset_identity(splits: Mapping[str, list[dict[str, Any]]]) -> None:
    """拒绝重复样本身份和跨分卷重复的知识问题。

    Args:
        splits (Mapping[str, list[dict[str, Any]]]): 三个待发布分卷。

    Raises:
        DatasetBuildError: 样本 ID 重复或同一知识问题跨分卷出现。

    Returns:
        None: 数据集身份与知识问题互斥时返回。
    """
    sample_owners: dict[str, str] = {}
    prompt_owners: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            sample_id = str(row.get("sample_id", ""))
            previous_sample = sample_owners.get(sample_id)
            if previous_sample is not None:
                raise DatasetBuildError(
                    f"样本 ID 重复: {sample_id} ({previous_sample}/{split})"
                )
            sample_owners[sample_id] = split
            if row.get("source") == "human_play":
                continue
            prompt = _knowledge_prompt(row)
            previous_prompt = prompt_owners.get(prompt)
            if previous_prompt is not None:
                raise DatasetBuildError(
                    f"知识问题跨分卷重复: {prompt} ({previous_prompt}/{split})"
                )
            prompt_owners[prompt] = split


def _knowledge_prompt(row: Mapping[str, Any]) -> str:
    """取得知识行唯一 user 问题的规范化文本。

    Args:
        row (Mapping[str, Any]): 可读 SFT 知识行。

    Raises:
        DatasetBuildError: messages 不含单一有效 user 问题。

    Returns:
        str: 去除 ``Q:``/``A:`` 包装并压缩空白后的问题。
    """
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise DatasetBuildError("知识行缺少 messages")
    users = [
        message.get("content")
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "user"
    ]
    if len(users) != 1 or not isinstance(users[0], str):
        raise DatasetBuildError("知识行必须包含一个 user 问题")
    return _normalize_question(users[0])


def _knowledge_answer_text(row: Mapping[str, Any]) -> str:
    """取得知识行唯一 assistant 答案的规范化文本。

    Args:
        row (Mapping[str, Any]): 可读 SFT 知识行。

    Raises:
        DatasetBuildError: messages 不含单一有效 assistant 答案。

    Returns:
        str: 压缩空白后的答案文本。
    """
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise DatasetBuildError("知识行缺少 messages")
    answers = [
        message.get("content")
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "assistant"
    ]
    if len(answers) != 1 or not isinstance(answers[0], str):
        raise DatasetBuildError("知识行必须包含一个 assistant 答案")
    return re.sub(r"\s+", " ", answers[0]).strip()


def _normalize_question(value: str) -> str:
    """移除问答包装并压缩问题中的空白。

    Args:
        value (str): SFT 候选或已发布数据中的原始问题。

    Returns:
        str: 可用于精确泄漏检查的稳定问题文本。
    """
    value = re.sub(r"^\s*Q:\s*", "", value)
    value = re.sub(r"\s*A:\s*$", "", value)
    return re.sub(r"\s+", " ", value).strip()


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
            "fact_id": f"{category}/{entry.object_id}/description",
            "question_role": "train",
            "dataset_split": "train",
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
            role = row.get("question_role")
            legacy_split = row.get("split")
            if not all(
                isinstance(value, str) and value.strip()
                for value in (prompt, completion, category, object_id, source)
            ):
                raise DatasetBuildError(f"知识问答字段无效: {path}:{line_number}")
            if role is None and legacy_split is None:
                role = "train"
            elif role is None:
                role = {"train": "train", "dev": "validation"}.get(legacy_split)
            if role not in {"train", "validation", "eval"}:
                raise DatasetBuildError(
                    f"知识问答 question_role 无效: {path}:{line_number}"
                )
            fact_id = row.get("fact_id")
            if not isinstance(fact_id, str) or not fact_id.strip():
                fact_id = f"{category}/{object_id}/legacy-{line_number:04d}"
            question = re.sub(r"\s*A:\s*$", "", prompt).strip()
            output = {
                "sample_id": (f"curated/{relative_stem}/{role}/{line_number:04d}"),
                "source": source,
                "source_detail": f"{path.relative_to(root).as_posix()}:{line_number}",
                "category": category,
                "object_id": object_id,
                "fact_id": fact_id,
                "question_role": role,
                "dataset_split": {
                    "train": "train",
                    "validation": "dev",
                    "eval": "test",
                }[role],
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": completion.strip()},
                ],
            }
            training_epoch = row.get("training_epoch")
            if training_epoch is not None:
                if (
                    role != "train"
                    or not isinstance(training_epoch, int)
                    or isinstance(training_epoch, bool)
                    or training_epoch not in {1, 2, 3, 4, 5}
                ):
                    raise DatasetBuildError(
                        f"知识问答 training_epoch 无效: {path}:{line_number}"
                    )
                output["training_epoch"] = training_epoch
            detail = row.get("source_detail")
            if isinstance(detail, str) and detail.strip():
                output["source_detail"] = detail
            supplement_source = row.get("supplement_source")
            if isinstance(supplement_source, str) and supplement_source.strip():
                output["supplement_source"] = supplement_source
            game_version = row.get("game_version")
            if isinstance(game_version, str) and game_version:
                output["game_version"] = game_version
            case_id = row.get("case_id")
            if isinstance(case_id, str) and case_id:
                output["case_id"] = case_id
            answer_numbers = row.get("answer_numbers")
            if isinstance(answer_numbers, list) and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in answer_numbers
            ):
                output["answer_numbers"] = answer_numbers
            answer_choice = row.get("answer_choice")
            if isinstance(answer_choice, str) and answer_choice:
                output["answer_choice"] = answer_choice
            yield output


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
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """用当前 Harness 把一条精确决策渲染成行为监督样本。

    Args:
        decision (Mapping[str, Any]): Raw 中的一条精确人类决策。
        run_id (str | None): 从当前局 meta 取得的稳定局 ID。
        battle_key (str | None): 战斗分片名；战略动作省略。
        run_metadata (Mapping[str, Any] | None): 当前局来源、胜负与完整性元数据。

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
        legal_lines = legal_action_lines(state)
    except ValueError as exc:
        raise DatasetBuildError(
            f"{resolved_run_id}:{sequence} 动作不符合 Harness"
        ) from exc
    if action_line not in legal_lines:
        raise DatasetBuildError(
            f"{resolved_run_id}:{sequence} 动作参数不在当前合法边界: {action_line}"
        )
    row = {
        "sample_id": f"human_play/{resolved_run_id}/{sequence}",
        "source": "human_play",
        "run_id": resolved_run_id,
        "layer": observation.layer.value,
        "screen": str(state.get("screen", "UNKNOWN")),
        "action": action_name,
        "available_actions": list(observation.available_actions),
        "legal_actions": list(legal_lines),
        "tags": _behavior_tags(state, action_name, parameters),
        "messages": [
            {
                "role": "system",
                "content": system_prompt(observation.layer, state),
            },
            {"role": "user", "content": observation.text},
            {"role": "assistant", "content": action_line},
        ],
    }
    if battle_key is not None:
        row["battle_key"] = battle_key
    if run_metadata is not None:
        origin = str(run_metadata.get("source") or "human")
        action_source = decision.get("action_source")
        if action_source is None and origin == "human":
            action_source = "human_ui"
        if action_source not in {"human_ui", "combat_solver"}:
            raise DatasetBuildError(f"{resolved_run_id}:{sequence} 缺少可审计动作来源")
        integrity = run_metadata.get("integrity")
        gaps = (
            integrity.get("recording_gaps", [])
            if isinstance(integrity, Mapping)
            else []
        )
        if not isinstance(gaps, list) or any(not isinstance(gap, str) for gap in gaps):
            raise DatasetBuildError(f"{resolved_run_id} recording_gaps 结构无效")
        victory = run_metadata.get("victory")
        row.update(
            {
                "behavior_origin": origin,
                "action_source": action_source,
                "run_victory": victory if isinstance(victory, bool) else None,
                "recording_complete": run_metadata.get("recording_complete") is True,
                "recording_gaps": list(gaps),
            }
        )
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
    available = set(model_actions(state))
    if "use_potion" in available:
        tags.add("potion:use" if action == "use_potion" else "potion:hold")
    if action == "discard_potion":
        tags.add("potion:discard")
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
    screen = state.get("screen")
    if screen == "SHOP":
        if action in {"buy_card", "buy_potion", "buy_relic"}:
            tags.update({"shop:purchase", f"shop:{action}"})
        elif action == "proceed":
            tags.add("shop:leave")
        elif action == "open_shop_inventory":
            tags.add("shop:open")
        elif action == "close_shop_inventory":
            tags.add("shop:close")
        elif action == "remove_card_at_shop":
            tags.add("shop:remove_card")
        shop = state.get("shop")
        if isinstance(shop, Mapping):
            purchase_available = shop_purchase_available(shop)
            if purchase_available is not None:
                tags.add(
                    "shop:affordable" if purchase_available else "shop:no_affordable"
                )
    if screen == "CARD_SELECTION":
        if action == "choose_reward_card":
            tags.add("card_reward:choose")
        elif action == "skip_reward_cards":
            tags.add("card_reward:skip")
        elif action == "choose_reward_alternative":
            tags.add("card_reward:alternative")
    return sorted(tags)


def _human_provenance_manifest(
    human_roots: Sequence[Path],
    human_rows: Sequence[Mapping[str, Any]],
    run_splits: Mapping[str, set[str]],
) -> dict[str, Any]:
    """生成逐根分卷和逐局完整性血缘。

    Args:
        human_roots (Sequence[Path]): 按构建参数顺序排列的 raw 根。
        human_rows (Sequence[Mapping[str, Any]]): 混合裁剪前的全部行为样本。
        run_splits (Mapping[str, set[str]]): 当前构建实际采用的整局分卷。

    Raises:
        DatasetBuildError: 同一 run 跨 root，或逐样本元数据互相矛盾。

    Returns:
        dict[str, Any]: 可合并到 manifest ``human`` 字段的 roots 与 runs。
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in human_rows:
        grouped.setdefault(str(row["run_id"]), []).append(row)

    runs: dict[str, dict[str, Any]] = {}
    for run_id, rows in sorted(grouped.items()):
        roots = {str(row.get("_human_root", "")) for row in rows}
        origins = {str(row.get("behavior_origin", "human")) for row in rows}
        victories = {row.get("run_victory") for row in rows}
        completion = {row.get("recording_complete") is True for row in rows}
        gaps = {tuple(row.get("recording_gaps", ())) for row in rows}
        if not all(
            len(values) == 1 for values in (roots, origins, victories, completion, gaps)
        ):
            raise DatasetBuildError(f"人类局血缘不一致: {run_id}")
        root = next(iter(roots))
        if not root:
            raise DatasetBuildError(f"人类局缺少来源根: {run_id}")
        split = next(name for name, values in run_splits.items() if run_id in values)
        runs[run_id] = {
            "root": root,
            "split": _PUBLIC_SPLIT_NAMES[split],
            "origin": next(iter(origins)),
            "victory": next(iter(victories)),
            "recording_complete": next(iter(completion)),
            "recording_gaps": list(next(iter(gaps))),
            "samples": len(rows),
            "action_sources": dict(
                sorted(
                    Counter(
                        str(row.get("action_source", "human_ui")) for row in rows
                    ).items()
                )
            ),
        }

    root_records = []
    for root in human_roots:
        root_text = str(root)
        owned = {
            run_id for run_id, record in runs.items() if record["root"] == root_text
        }
        root_records.append(
            {
                "root": root_text,
                "splits": {
                    _PUBLIC_SPLIT_NAMES[name]: sorted(values & owned)
                    for name, values in run_splits.items()
                },
            }
        )
    return {"roots": root_records, "runs": runs}


def _behavior_audit(
    splits: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """汇总关键行为及训练分卷的成对决策覆盖。

    Args:
        splits (Mapping[str, list[dict[str, Any]]]): 最终三个 SFT 分卷。

    Returns:
        dict[str, Any]: 可单独审查的动作、页面、标签和训练对照计数。
    """
    split_reports: dict[str, dict[str, Any]] = {}
    for split, rows in splits.items():
        behavior_rows = [row for row in rows if row.get("source") == "human_play"]
        actions = Counter(str(row.get("action", "")) for row in behavior_rows)
        screens = Counter(str(row.get("screen", "")) for row in behavior_rows)
        origins = Counter(
            str(row.get("behavior_origin", "human")) for row in behavior_rows
        )
        action_sources = Counter(
            str(row.get("action_source", "human_ui")) for row in behavior_rows
        )
        tags = Counter(
            str(tag)
            for row in behavior_rows
            for tag in row.get("tags", [])
            if isinstance(tag, str)
        )
        split_reports[_PUBLIC_SPLIT_NAMES[split]] = {
            "samples": len(behavior_rows),
            "actions": dict(sorted(actions.items())),
            "screens": dict(sorted(screens.items())),
            "origins": dict(sorted(origins.items())),
            "action_sources": dict(sorted(action_sources.items())),
            "tags": dict(sorted(tags.items())),
        }

    train_tags = split_reports["train"]["tags"]
    contrast_specs = {
        "card_reward_choose_vs_skip": {
            "choose": "card_reward:choose",
            "skip": "card_reward:skip",
        },
        "potion_use_vs_hold": {"hold": "potion:hold", "use": "potion:use"},
        "rest_heal_vs_smith": {"heal": "rest:heal", "smith": "rest:smith"},
        "shop_purchase_vs_leave": {
            "leave": "shop:leave",
            "purchase": "shop:purchase",
        },
    }
    contrasts = {
        name: _behavior_contrast(train_tags, tags)
        for name, tags in sorted(contrast_specs.items())
    }
    return {
        "format": "sft_behavior_audit_v1",
        "splits": split_reports,
        "training_contrasts": contrasts,
        "missing_training_contrasts": [
            name for name, record in contrasts.items() if not record["available"]
        ],
    }


def _behavior_contrast(
    tag_counts: Mapping[str, int],
    variants: Mapping[str, str],
) -> dict[str, Any]:
    """把一组行为标签转换为成对覆盖记录。

    Args:
        tag_counts (Mapping[str, int]): 训练分卷的标签计数。
        variants (Mapping[str, str]): 展示名称到稳定标签的映射。

    Returns:
        dict[str, Any]: 各分支计数及是否全部出现。
    """
    counts = {
        name: int(tag_counts.get(tag, 0)) for name, tag in sorted(variants.items())
    }
    return {"available": all(count > 0 for count in counts.values()), "counts": counts}


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


def _human_rows(
    root: Path,
    *,
    included_run_ids: Collection[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """读取按局分片的人类精确动作并恢复训练消息。

    Args:
        root (Path): ``data/raw/human`` 风格的人类数据根目录。
        included_run_ids (Collection[str] | None): 可选的显式入选局 ID。

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
        run_id = metadata.get("run_id", run_dir.name)
        if included_run_ids is not None and run_id not in included_run_ids:
            continue
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
                behavior = _behavior_row(
                    row,
                    run_id=run_id,
                    battle_key=battle_path.stem,
                    run_metadata=metadata,
                )
                behavior["_human_root"] = str(root)
                yield behavior
        if audit.strategy_path is not None:
            for row in _read_jsonl(audit.strategy_path):
                behavior = _behavior_row(row, run_id=run_id, run_metadata=metadata)
                behavior["_human_root"] = str(root)
                yield behavior


def _read_run_splits(path: Path) -> dict[str, list[str]]:
    """读取可选的整局训练、验证和测试归属名册。

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
