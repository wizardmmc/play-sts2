"""把当前人类 raw 无损投影为便于审阅的文本。"""

import json
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..harness import (
    HarnessAction,
    Observation,
    build_observation,
    format_action,
    system_prompt,
)
from ..recording.audit import RawRunIntegrityError, audit_human_run
from .models import TranscriptResult

_SCREEN_NAMES = {
    "BUNDLE_SELECTION": "卡牌包",
    "CARD_SELECTION": "选牌",
    "CARDS_VIEW": "卡牌浏览",
    "CHEST": "宝箱",
    "COMBAT": "战斗",
    "CRYSTAL_SPHERE": "水晶球",
    "EVENT": "事件",
    "MAP": "地图",
    "MODAL": "弹窗",
    "REST": "休息处",
    "REWARD": "奖励",
    "SHOP": "商店",
    "TIMELINE": "时间线",
    "UNKNOWN": "房间结算",
}


class TranscriptError(RuntimeError):
    """表示当前 raw 不能可靠生成 transcript。"""


def render_run(run_dir: Path, output_root: Path) -> TranscriptResult:
    """重新生成一局完整的人类可读 transcript 目录。

    Args:
        run_dir (Path): 含 ``meta.json``、``combat`` 和 ``strategy`` 的 raw 局。
        output_root (Path): Transcript 根目录，例如 ``data/transcripts``。

    Raises:
        TranscriptError: Raw 目录、JSONL 或动作字段不符合当前契约。
        OSError: 无法读取 raw 或原子发布 transcript。

    Returns:
        TranscriptResult: 输出目录和两类决策计数。
    """
    source = Path(run_dir).resolve()
    destination_root = Path(output_root).resolve()
    destination = destination_root / source.name
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise TranscriptError(
            f"Transcript 输出不能与 raw 重叠: {source} -> {destination}"
        )
    try:
        audit = audit_human_run(source)
    except RawRunIntegrityError as exc:
        raise TranscriptError(str(exc)) from exc
    destination_root.mkdir(parents=True, exist_ok=True)
    previous = destination_root / f".{source.name}-previous"
    if previous.exists() and not destination.exists():
        previous.rename(destination)
    staging = Path(tempfile.mkdtemp(prefix=f".{source.name}-", dir=destination_root))
    battle_count = 0
    strategic_count = 0
    try:
        battle_output = staging / "combat"
        battle_output.mkdir()
        for battle_path in audit.battle_paths:
            rows = list(_read_jsonl(battle_path))
            if not rows:
                continue
            text = _render_file(rows, title=_battle_title(rows))
            (battle_output / f"{battle_path.stem}.txt").write_text(
                text,
                encoding="utf-8",
            )
            battle_count += len(rows)

        if audit.strategy_path is not None:
            rows = list(_read_jsonl(audit.strategy_path))
            if rows:
                strategy_output = staging / "strategy"
                strategy_output.mkdir()
                (strategy_output / "decisions.txt").write_text(
                    _render_file(rows, title="# 战略决策"),
                    encoding="utf-8",
                )
                strategic_count = len(rows)

        if previous.exists():
            shutil.rmtree(previous)
        if destination.exists():
            destination.rename(previous)
        staging.rename(destination)
        if previous.exists():
            shutil.rmtree(previous)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if previous.exists() and not destination.exists():
            previous.rename(destination)
        raise
    return TranscriptResult(destination, battle_count, strategic_count)


def _render_file(rows: list[dict[str, Any]], *, title: str) -> str:
    """把同层动作渲染为与独立训练消息一致的可读文件。

    Args:
        rows (list[dict[str, Any]]): 按时间排序的 raw 动作。
        title (str): 文件首行的人类可读标题。

    Raises:
        TranscriptError: 动作无法由当前 Harness 重建。

    Returns:
        str: 以换行结尾的 transcript 文本。
    """
    rendered = [_render_decision(row) for row in rows]
    layers = {item[0].layer for item in rendered}
    if len(layers) != 1:
        raise TranscriptError("同一 transcript 文件包含多个 Harness 层")
    prompt = system_prompt(rendered[0][0].layer)
    sections = [title]
    for index, (observation, action_line) in enumerate(rendered, start=1):
        heading = f"## 决策 {index}"
        if observation.layer.value == "strategic":
            heading += _decision_context(rows[index - 1])
        sections.append(
            f"{heading}\n\n"
            f"──── system ────\n{prompt}\n\n"
            f"──── user ────\n{observation.text}\n\n"
            f"──── assistant ────\n{action_line}"
        )
    return "\n\n".join(sections) + "\n"


def _render_decision(row: Mapping[str, Any]) -> tuple[Observation, str]:
    """用当前 Harness 重建一条动作的观测与规范动作行。

    Args:
        row (Mapping[str, Any]): 当前 raw schema 的动作行。

    Raises:
        TranscriptError: 状态、动作或参数缺失或不合法。

    Returns:
        tuple[Observation, str]: Harness 观测与规范 ``ACTION:`` 文本。
    """
    state = row.get("before_state")
    action = row.get("action")
    parameters = row.get("parameters")
    if (
        not isinstance(state, Mapping)
        or not isinstance(action, str)
        or not isinstance(parameters, Mapping)
    ):
        raise TranscriptError("Raw 动作缺少 before_state/action/parameters")
    try:
        observation = build_observation(state)
        if action not in observation.available_actions:
            raise ValueError(f"动作未向 Harness 开放: {action}")
        action_line = format_action(HarnessAction(action, dict(parameters)))
    except ValueError as exc:
        raise TranscriptError(f"Raw 动作无法重建: {exc}") from exc
    return observation, action_line


def _battle_title(rows: list[dict[str, Any]]) -> str:
    """从首个战斗状态生成文件标题。

    Args:
        rows (list[dict[str, Any]]): 当前战斗的 raw 动作。

    Returns:
        str: 含楼层的战斗标题。
    """
    state = rows[0].get("before_state")
    run = state.get("run") if isinstance(state, Mapping) else None
    floor = run.get("floor") if isinstance(run, Mapping) else None
    return f"# 第 {floor} 层 · 战斗" if isinstance(floor, int) else "# 战斗"


def _decision_context(row: Mapping[str, Any]) -> str:
    """为战略决策生成楼层和页面后缀。

    Args:
        row (Mapping[str, Any]): 当前战略 raw 动作。

    Returns:
        str: 可直接追加到决策标题的上下文。
    """
    state = row.get("before_state")
    if not isinstance(state, Mapping):
        return ""
    run = state.get("run")
    floor = run.get("floor") if isinstance(run, Mapping) else None
    screen = state.get("screen")
    screen_name = _SCREEN_NAMES.get(str(screen), str(screen))
    details = [f"第 {floor} 层" if isinstance(floor, int) else "", screen_name]
    suffix = " · ".join(item for item in details if item)
    return f" · {suffix}" if suffix else ""


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """逐行读取并校验 JSON object。

    Args:
        path (Path): Raw JSONL 路径。

    Raises:
        TranscriptError: 某行不是合法 JSON object。
        OSError: 文件无法读取。

    Yields:
        dict[str, Any]: 保持文件顺序的动作对象。
    """
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TranscriptError(f"无效 JSON: {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise TranscriptError(f"JSONL 行必须是 object: {path}:{line_number}")
        yield value
