"""从 recorder 原始事件中提取无损的人类 UI 决策。"""

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .models import TranscribedDecision, TranscribedRun

_SUCCESS_STATUSES = {"accepted", "completed", "pending"}


class TranscriptionError(RuntimeError):
    """表示原始轨迹无法转换为可信的精确决策。"""


def transcribe_run(run_dir: Path, output_root: Path) -> TranscribedRun:
    """把一局原始人类轨迹转录为按动作排序的精确决策。

    Args:
        run_dir (Path): 包含 ``meta.json`` 和 ``events.jsonl`` 的原始局目录。
        output_root (Path): 精确决策文件的输出目录。

    Raises:
        TranscriptionError: 输入文件不可读、JSON 无效或人类动作字段不完整。
        OSError: 无法创建输出目录或写入转录结果。
        ValueError: 决策包含不能表示成标准 JSON 的值。

    Returns:
        TranscribedRun: 输出路径和转录到的人类决策数量。
    """
    run_dir = Path(run_dir)
    metadata = _read_object(run_dir / "meta.json")
    run_id = metadata.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise TranscriptionError("meta.json 缺少有效的 run_id")
    if metadata.get("source") != "human":
        raise TranscriptionError("精确人类转录只接受 source=human 的轨迹")

    decisions: list[TranscribedDecision] = []
    for event in _read_jsonl(run_dir / "events.jsonl"):
        decision = _decision_from_event(run_id, event)
        if decision is not None:
            decisions.append(decision)

    output_path = Path(output_root) / f"{run_id}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(
                decision.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for decision in decisions
        ),
        encoding="utf-8",
    )
    return TranscribedRun(
        output_path=output_path,
        decision_count=len(decisions),
    )


def _decision_from_event(
    run_id: str,
    event: Mapping[str, Any],
) -> TranscribedDecision | None:
    """从一条 recorder 事件中提取人类 UI 决策。

    Args:
        run_id (str): 当前原始轨迹的局 ID。
        event (Mapping[str, Any]): recorder 保存的一条事件。

    Raises:
        TranscriptionError: 匹配到的人类动作缺少精确转录所需字段。

    Returns:
        TranscribedDecision | None: 精确人类决策；无关事件返回 ``None``。
    """
    if event.get("type") != "mod_event":
        return None
    payload = event.get("payload")
    if not isinstance(payload, Mapping) or payload.get("type") != "action_executed":
        return None
    data = payload.get("data")
    if not isinstance(data, Mapping):
        return None
    request = data.get("request")
    if not isinstance(request, Mapping):
        return None
    context = request.get("client_context")
    if not isinstance(context, Mapping) or context.get("source") != "human_ui":
        return None
    if data.get("status") not in _SUCCESS_STATUSES:
        return None

    source_sequence = event.get("sequence")
    event_id = payload.get("event_id")
    observed_at = event.get("observed_at")
    before_state = data.get("before_state")
    action = request.get("action")
    if (
        isinstance(source_sequence, bool)
        or not isinstance(source_sequence, int)
        or isinstance(event_id, bool)
        or not isinstance(event_id, int)
        or not isinstance(observed_at, str)
        or not isinstance(before_state, Mapping)
        or not isinstance(action, str)
        or not action
    ):
        raise TranscriptionError("action_executed 缺少精确决策字段")

    recorded_layer = context.get("layer")
    if not isinstance(recorded_layer, str):
        recorded_layer = None
    parameters = {
        key: value
        for key, value in request.items()
        if key not in {"action", "client_context"} and value is not None
    }
    return TranscribedDecision(
        run_id=run_id,
        source_sequence=source_sequence,
        event_id=event_id,
        observed_at=observed_at,
        recorded_layer=recorded_layer,
        before_state=dict(before_state),
        action=action,
        parameters=parameters,
    )


def _read_object(path: Path) -> dict[str, Any]:
    """读取一个必须为 JSON 对象的文件。

    Args:
        path (Path): 待读取的 JSON 文件。

    Raises:
        TranscriptionError: 文件不可读、JSON 无效或顶层不是对象。

    Returns:
        dict[str, Any]: 解码后的普通字典。
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranscriptionError(f"无法读取转录输入: {path}") from exc
    if not isinstance(value, Mapping):
        raise TranscriptionError(f"转录输入不是 JSON 对象: {path}")
    return dict(value)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """逐行读取只包含 JSON 对象的事件文件。

    Args:
        path (Path): 待读取的 JSONL 文件。

    Raises:
        TranscriptionError: 文件不可读、某行 JSON 无效或某行不是对象。

    Yields:
        dict[str, Any]: 按文件顺序读取的事件字典。
    """
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TranscriptionError(
                        f"无效 JSONL: {path}:{line_number}"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise TranscriptionError(f"JSONL 行不是对象: {path}:{line_number}")
                yield dict(value)
    except OSError as exc:
        raise TranscriptionError(f"无法读取转录输入: {path}") from exc
