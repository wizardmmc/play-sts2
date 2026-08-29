"""把精确人类决策保存为可恢复、可重建的原始分片。"""

import json
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ..harness import HarnessAction, build_observation, format_action
from .models import RunMetadata

_LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")
_RAW_SCHEMA_VERSION = 2
_SAFE_NAME = re.compile(r"[A-Za-z0-9._-]+")


class HumanRunWriter:
    """把一局人类动作直接分片为战斗与战略原始事实。

    Writer 在录制期间使用隐藏临时目录，并持续刷新 ``meta.json``。完成或中断时，
    它根据本地游玩日期、进阶、最高层和 seed 原子改为最终目录名。训练资格描述
    已落盘单步样本，录制完整性则只在完整到达无已知缺口的 ``game_over`` 时成立。

    Args:
        output_root (Path): 原始数据根目录，例如 ``data/raw``。
        metadata (RunMetadata): 当前局的身份、角色和进阶信息。

    Raises:
        FileExistsError: 临时目录或最终目录已经存在。
        ValueError: 元数据不能形成安全、稳定的目录名。
        OSError: 无法创建目录或写入元数据。
    """

    def __init__(self, output_root: Path, metadata: RunMetadata) -> None:
        """创建一局不会覆盖既有数据的临时目录。

        Args:
            output_root (Path): 原始数据根目录。
            metadata (RunMetadata): 当前局元数据。
        """
        identity = metadata.seed or metadata.run_id
        if not identity or _SAFE_NAME.fullmatch(identity) is None:
            raise ValueError(f"无法用作目录名的 seed: {identity!r}")
        self._metadata = metadata
        self._identity = identity
        self._local_start = _parse_timestamp(metadata.started_at).astimezone(
            _LOCAL_TIMEZONE
        )
        self._ascension = metadata.ascension
        self._max_floor_reached = 0
        self._battle_sample_count = 0
        self._strategic_sample_count = 0
        self._battle_count = 0
        self._action_source_counts: dict[str, int] = {}
        self._current_battle_key: str | None = None
        self._event_ids: set[int | str] = set()
        self._integrity_failures: list[str] = []
        self._recording_gaps: list[str] = []

        source_root = Path(output_root) / metadata.source
        source_root.mkdir(parents=True, exist_ok=True)
        started = self._local_start.strftime("%Y%m%dT%H%M%S%f")
        ascension = f"a{self._ascension}" if self._ascension is not None else "ax"
        self._run_dir = source_root / (
            f".recording-{started}-{ascension}-{self._identity}"
        )
        self._run_dir.mkdir()
        (self._run_dir / "combat").mkdir()
        (self._run_dir / "strategy").mkdir()
        self._write_metadata()

    @property
    def run_dir(self) -> Path:
        """返回当前局的实际目录，完成后即为最终规范目录。

        Returns:
            Path: 当前临时目录或已经发布的最终目录。
        """
        return self._run_dir

    @property
    def sample_count(self) -> int:
        """返回已经成功保存的人类决策总数。

        Returns:
            int: 战斗与战略动作数之和。
        """
        return self._battle_sample_count + self._strategic_sample_count

    def append_decision(self, decision: Mapping[str, Any]) -> Path:
        """校验并保存一条精确人类决策。

        Args:
            decision (Mapping[str, Any]): 含动作前状态、动作和参数的原始决策。

        Raises:
            ValueError: 决策字段、事件顺序或 Harness 契约不合法。
            OSError: 无法追加分片或更新元数据。

        Returns:
            Path: 本条决策所属的战斗或战略 JSONL。
        """
        row, layer, floor, action_source = self._normalize_decision(decision)
        event_id = row["event_id"]
        if event_id in self._event_ids:
            raise ValueError(f"重复的人类动作事件: {event_id}")
        self._event_ids.add(event_id)
        self._max_floor_reached = max(self._max_floor_reached, floor)
        if action_source is not None:
            self._action_source_counts[action_source] = (
                self._action_source_counts.get(action_source, 0) + 1
            )

        if layer == "battle":
            destination = self._battle_path(floor)
            self._battle_sample_count += 1
        else:
            self._current_battle_key = None
            destination = self._run_dir / "strategy/decisions.jsonl"
            self._strategic_sample_count += 1
        self._append_jsonl(destination, row)
        self._write_metadata()
        return destination

    def finalize(
        self,
        reason: str,
        *,
        completed_at: str,
        victory: bool | None = None,
    ) -> None:
        """写入终止状态，并把临时目录原子发布为最终名称。

        Args:
            reason (str): ``game_over``、``returned_to_menu`` 或 ``interrupted``。
            completed_at (str): 录制完成时的 UTC ISO 8601 时间。
            victory (bool | None): 完整终局的胜负；非终局或未知时为 ``None``。

        Raises:
            FileExistsError: 同名最终局已经存在，拒绝覆盖。
            OSError: 无法写入元数据或重命名目录。

        Returns:
            None: 元数据和目录发布完成后返回。
        """
        self._write_metadata(
            termination_reason=reason,
            completed_at=completed_at,
            victory=victory,
        )
        destination = self._run_dir.parent / self._final_name()
        if destination.exists():
            raise FileExistsError(destination)
        self._run_dir.rename(destination)
        self._run_dir = destination

    def end_battle(self) -> None:
        """关闭当前战斗分片，让下一场战斗创建新文件。

        Returns:
            None: 当前没有战斗时也保持幂等。
        """
        self._current_battle_key = None

    def record_integrity_failure(self, reason: str) -> None:
        """记录会关闭整局训练准入的采集完整性问题。

        Args:
            reason (str): 可供人工审计的稳定失败原因。

        Raises:
            ValueError: 原因不是非空字符串。
            OSError: 无法刷新元数据。

        Returns:
            None: 原因去重并持久化后返回。
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("完整性失败原因不能为空")
        normalized = reason.strip()
        if normalized not in self._integrity_failures:
            self._integrity_failures.append(normalized)
            self._write_metadata()

    def record_recording_gap(self, reason: str) -> None:
        """记录缺失动作，但保留已验证独立样本的训练资格。

        Args:
            reason (str): 可供逐局审计的稳定缺口原因。

        Raises:
            ValueError: 原因不是非空字符串。
            OSError: 无法刷新元数据。

        Returns:
            None: 原因去重并持久化后返回。
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("录制缺口原因不能为空")
        normalized = reason.strip()
        if normalized not in self._recording_gaps:
            self._recording_gaps.append(normalized)
            self._write_metadata()

    def _normalize_decision(
        self,
        decision: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str, int, str | None]:
        """验证动作并剥离能由路径、meta 或 Harness 重建的字段。

        Args:
            decision (Mapping[str, Any]): Recorder 提取的精确动作事件。

        Raises:
            ValueError: 字段缺失、层级漂移或动作未向 Harness 开放。

        Returns:
            tuple[dict[str, Any], str, int]: 精简 raw 行、层级和当前楼层。
        """
        run_id = decision.get("run_id")
        event_id = decision.get("event_id")
        observed_at = decision.get("observed_at")
        state = decision.get("before_state")
        action = decision.get("action")
        parameters = decision.get("parameters")
        provenance = decision.get("provenance")
        action_source = decision.get("action_source")
        if (
            run_id != self._metadata.run_id
            or not _is_event_id(event_id)
            or not isinstance(observed_at, str)
            or not isinstance(state, Mapping)
            or not isinstance(action, str)
            or not isinstance(parameters, Mapping)
            or (provenance is not None and not isinstance(provenance, Mapping))
            or (
                action_source is not None
                and action_source not in {"human_ui", "combat_solver"}
            )
        ):
            raise ValueError("精确人类决策字段无效")
        if self._metadata.source == "human_combat_solver" and action_source not in {
            "human_ui",
            "combat_solver",
        }:
            raise ValueError("人机协作决策缺少明确的 action_source")

        observation = build_observation(state)
        recorded_layer = decision.get("recorded_layer")
        if (
            isinstance(recorded_layer, str)
            and recorded_layer != observation.layer.value
        ):
            raise ValueError(
                f"录制层与当前 Harness 不一致: {recorded_layer} != "
                f"{observation.layer.value}"
            )
        if action not in observation.available_actions:
            raise ValueError(f"动作未向当前 Harness 开放: {action}")
        format_action(HarnessAction(action, dict(parameters)))

        run = state.get("run")
        floor = run.get("floor") if isinstance(run, Mapping) else 0
        if isinstance(floor, bool) or not isinstance(floor, int) or floor < 0:
            floor = 0
        ascension = run.get("ascension") if isinstance(run, Mapping) else None
        if (
            self._ascension is None
            and isinstance(ascension, int)
            and not isinstance(ascension, bool)
        ):
            self._ascension = ascension
        row = {
            "event_id": event_id,
            "observed_at": observed_at,
            "before_state": dict(state),
            "action": action,
            "parameters": dict(parameters),
        }
        if provenance is not None:
            row["provenance"] = dict(provenance)
        if isinstance(action_source, str):
            row["action_source"] = action_source
        return (
            row,
            observation.layer.value,
            floor,
            (action_source if isinstance(action_source, str) else None),
        )

    def _battle_path(self, floor: int) -> Path:
        """返回当前战斗稳定的 JSONL 路径。

        Args:
            floor (int): 当前动作所在楼层。

        Returns:
            Path: 当前场战斗的结构化分片路径。
        """
        if self._current_battle_key is None:
            self._battle_count += 1
            self._current_battle_key = f"battle-f{floor:03d}-{self._battle_count:02d}"
        return self._run_dir / f"combat/{self._current_battle_key}.jsonl"

    def _final_name(self) -> str:
        """根据最终局摘要生成规范目录名。

        Raises:
            ValueError: 录制结束时仍无法确定进阶难度。

        Returns:
            str: ``YYYYMMDD-aN-fN-SEED`` 格式的目录名。
        """
        if self._ascension is None:
            raise ValueError("录制结束时仍缺少进阶难度")
        played_on = self._local_start.strftime("%Y%m%d")
        return (
            f"{played_on}-a{self._ascension}-f{self._max_floor_reached}-"
            f"{self._identity}"
        )

    def _write_metadata(
        self,
        *,
        termination_reason: str | None = None,
        completed_at: str | None = None,
        victory: bool | None = None,
    ) -> None:
        """以文件替换方式保存可用于异常恢复的当前统计。

        Args:
            termination_reason (str | None): 可选的最终终止原因。
            completed_at (str | None): 可选的最终完成时间。
            victory (bool | None): 完整终局的胜负；未知时为 ``None``。

        Returns:
            None: 元数据原子替换完成后返回。
        """
        samples_verified = not self._integrity_failures
        recording_complete = (
            termination_reason == "game_over"
            and isinstance(victory, bool)
            and samples_verified
            and not self._recording_gaps
        )
        metadata = self._metadata.to_dict()
        if metadata.get("recording_context") is None:
            metadata.pop("recording_context", None)
        payload = {
            **metadata,
            "schema_version": _RAW_SCHEMA_VERSION,
            "played_on": self._local_start.date().isoformat(),
            "ascension": self._ascension,
            "max_floor_reached": self._max_floor_reached,
            "battle_count": self._battle_count,
            "battle_sample_count": self._battle_sample_count,
            "strategic_sample_count": self._strategic_sample_count,
            "action_source_counts": dict(sorted(self._action_source_counts.items())),
            "termination_reason": termination_reason,
            "victory": victory,
            "completed_at": completed_at,
            "training_eligible": samples_verified,
            "recording_complete": recording_complete,
            "integrity": {
                "samples_verified": samples_verified,
                "ineligibility_reasons": list(self._integrity_failures),
                "recording_gaps": list(self._recording_gaps),
            },
        }
        temporary = self._run_dir / ".meta.json.tmp"
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._run_dir / "meta.json")

    @staticmethod
    def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
        """向一个动作分片追加标准 JSON 行。

        Args:
            path (Path): 目标 JSONL。
            row (Mapping[str, Any]): 已校验的精确动作事实。

        Returns:
            None: 单行追加完成后返回。
        """
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _is_event_id(value: object) -> bool:
    """判断事件身份是否为 Mod 编号或人工确认的稳定字符串。

    Args:
        value (object): Recorder 或迁移器提供的事件身份。

    Returns:
        bool: 整数（不含布尔值）或非空字符串时为真。
    """
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        or isinstance(value, str)
        and bool(value.strip())
    )


def _parse_timestamp(value: str) -> datetime:
    """解析要求带时区的 ISO 8601 时间。

    Args:
        value (str): UTC 或带偏移量的 ISO 8601 时间。

    Raises:
        ValueError: 时间无效或缺少时区。

    Returns:
        datetime: 带时区的时间对象。
    """
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("started_at 必须包含时区")
    return parsed
