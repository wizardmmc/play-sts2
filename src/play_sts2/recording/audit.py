"""校验已发布人类 raw 的元数据与物理分片是否一致。"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 2


class RawRunIntegrityError(RuntimeError):
    """表示已发布 raw 的完整性声明与磁盘事实不一致。"""


@dataclass(frozen=True, slots=True)
class HumanRunAudit:
    """保存一局通过审计后的元数据和分片位置。

    Args:
        metadata (dict[str, Any]): 已校验的 ``meta.json``。
        battle_paths (tuple[Path, ...]): 按名称排序的战斗分片。
        strategy_path (Path | None): 存在时的战略分片。
    """

    metadata: dict[str, Any]
    battle_paths: tuple[Path, ...]
    strategy_path: Path | None


def audit_human_run(run_dir: Path) -> HumanRunAudit:
    """对照 meta 计数审计一局已发布的人类 raw。

    Args:
        run_dir (Path): 含 ``meta.json``、``combat`` 和 ``strategy`` 的局目录。

    Raises:
        RawRunIntegrityError: 元数据结构、发布状态或任一分片计数不一致。
        OSError: 文件无法读取。

    Returns:
        HumanRunAudit: 后续消费者可安全读取的元数据和分片路径。
    """
    root = Path(run_dir)
    metadata = _read_metadata(root / "meta.json")
    _validate_metadata(metadata, root)
    battle_paths = tuple(sorted((root / "combat").glob("*.jsonl")))
    strategy_candidate = root / "strategy/decisions.jsonl"
    strategy_path = strategy_candidate if strategy_candidate.is_file() else None
    actual_counts = {
        "battle_count": len(battle_paths),
        "battle_sample_count": sum(_count_rows(path) for path in battle_paths),
        "strategic_sample_count": (
            _count_rows(strategy_path) if strategy_path is not None else 0
        ),
    }
    for field, actual in actual_counts.items():
        declared = metadata.get(field)
        if declared != actual:
            raise RawRunIntegrityError(
                f"{root} 的 {field} 不一致: meta={declared}, actual={actual}"
            )
    return HumanRunAudit(metadata, battle_paths, strategy_path)


def _read_metadata(path: Path) -> dict[str, Any]:
    """读取顶层必须是对象的 raw 元数据。

    Args:
        path (Path): ``meta.json`` 路径。

    Raises:
        RawRunIntegrityError: 文件缺失、JSON 无效或顶层不是对象。
        OSError: 文件无法读取。

    Returns:
        dict[str, Any]: 解码后的元数据。
    """
    if not path.is_file():
        raise RawRunIntegrityError(f"人类局缺少 meta.json: {path.parent}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RawRunIntegrityError(f"无效 JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise RawRunIntegrityError(f"meta.json 顶层不是对象: {path}")
    return dict(value)


def _validate_metadata(metadata: Mapping[str, Any], run_dir: Path) -> None:
    """校验当前 raw schema 的发布、单步训练准入与整局完整性字段。

    Args:
        metadata (Mapping[str, Any]): 已解码的 ``meta.json``。
        run_dir (Path): 用于错误定位的局目录。

    Raises:
        RawRunIntegrityError: schema、终止状态、完整性或计数字段无效。

    Returns:
        None: 所有必需字段合法时返回。
    """
    if metadata.get("schema_version") != _SCHEMA_VERSION:
        raise RawRunIntegrityError(f"{run_dir} 的 schema_version 不是 2")
    reason = metadata.get("termination_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise RawRunIntegrityError(f"{run_dir} 尚未发布完成")
    eligible = metadata.get("training_eligible")
    recording_complete = metadata.get("recording_complete")
    integrity = metadata.get("integrity")
    samples_verified = (
        integrity.get("samples_verified") if isinstance(integrity, Mapping) else None
    )
    if (
        not isinstance(eligible, bool)
        or not isinstance(recording_complete, bool)
        or not isinstance(samples_verified, bool)
    ):
        raise RawRunIntegrityError(f"{run_dir} 缺少明确的训练准入或完整性结论")
    if eligible != samples_verified:
        raise RawRunIntegrityError(f"{run_dir} 的训练准入与样本校验结论冲突")
    if recording_complete and not samples_verified:
        raise RawRunIntegrityError(f"{run_dir} 声称录制完整但样本未通过校验")
    for field in (
        "battle_count",
        "battle_sample_count",
        "strategic_sample_count",
    ):
        value = metadata.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RawRunIntegrityError(f"{run_dir} 的 {field} 无效")


def _count_rows(path: Path) -> int:
    """统计 JSONL 中非空物理行，供 meta 对账使用。

    Args:
        path (Path): 待统计的 JSONL。

    Returns:
        int: 非空行数。
    """
    return sum(
        bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
    )
