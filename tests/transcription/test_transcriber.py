"""验证原始人类动作到可读 transcript 的可重复投影。"""

import importlib
import json
from pathlib import Path

import pytest


def test_render_run_writes_readable_files_without_internal_ids(tmp_path: Path) -> None:
    """每个战斗只写一个可读文件，并隐藏机器内部定位字段。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: Transcript 目录、内容或计数不符合契约。

    Returns:
        None: 此测试只检查可观察的派生文件。
    """
    run_dir = tmp_path / "raw/human/20260827-a1-f2-TEST-SEED"
    (run_dir / "combat").mkdir(parents=True)
    (run_dir / "strategy").mkdir()
    _write_meta(run_dir, battle_count=1, battle_samples=2, strategic_samples=1)
    battle_rows = [
        _decision(event_id=101, screen="COMBAT", action="end_turn"),
        _decision(event_id=102, screen="COMBAT", action="end_turn"),
    ]
    _write_jsonl(run_dir / "combat/battle-f002-01.jsonl", battle_rows)
    _write_jsonl(
        run_dir / "strategy/decisions.jsonl",
        [
            _decision(
                event_id=103,
                screen="MAP",
                action="choose_map_node",
                option_index=0,
            )
        ],
    )

    transcription = importlib.import_module("play_sts2.transcription")
    result = transcription.render_run(run_dir, tmp_path / "transcripts")

    assert result.output_dir == (tmp_path / "transcripts/20260827-a1-f2-TEST-SEED")
    assert result.battle_decision_count == 2
    assert result.strategic_decision_count == 1
    battle_text = (result.output_dir / "combat/battle-f002-01.txt").read_text(
        encoding="utf-8"
    )
    strategy_text = (result.output_dir / "strategy/decisions.txt").read_text(
        encoding="utf-8"
    )
    assert "## 规则" not in battle_text
    assert battle_text.count("## 决策") == 2
    assert battle_text.count("──── system ────") == 2
    assert battle_text.count("──── user（回合 1） ────") == 2
    assert battle_text.count("──── assistant ────") == 2
    assert "ACTION: end_turn" in battle_text
    assert battle_text.count("【遗物】") == 2
    assert battle_text.count("- [0] 破损核心: 战斗开始时生成1个闪电充能球。") == 2
    assert "【当前回合】" not in battle_text
    assert "角色:" not in battle_text
    assert "牌组 " not in battle_text
    assert "遗物效果:" not in battle_text
    assert "human_play/" not in battle_text
    assert "event 101" not in battle_text
    assert "sample_id" not in strategy_text
    assert strategy_text.count("──── system ────") == 1
    assert strategy_text.count("──── user ────") == 1
    assert strategy_text.count("──── assistant ────") == 1
    assert "ACTION: choose_map_node 0" in strategy_text


def test_render_run_preserves_all_dynamic_rest_options(tmp_path: Path) -> None:
    """Transcript 保留微型帐篷与铲子提供的完整休息处选项。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 共享 Harness 或 transcript 丢失动态休息选项。

    Returns:
        None: 此测试只验证 raw 到可读战略记录的投影。
    """
    run_dir = tmp_path / "raw/human/20260827-a1-f2-REST-SEED"
    (run_dir / "combat").mkdir(parents=True)
    (run_dir / "strategy").mkdir()
    _write_meta(run_dir, battle_count=0, battle_samples=0, strategic_samples=1)
    decision = _decision(
        event_id=104,
        screen="MAP",
        action="choose_map_node",
        option_index=0,
    )
    state = decision["before_state"]
    assert isinstance(state, dict)
    state["screen"] = "REST"
    state["available_actions"] = ["choose_rest_option"]
    state.pop("map")
    state["rest"] = {
        "options": [
            {"index": 0, "option_id": "HEAL", "title": "休息"},
            {"index": 1, "option_id": "SMITH", "title": "锻造"},
            {"index": 2, "option_id": "DIG", "title": "挖掘"},
            {"index": 3, "option_id": "LEAVE", "title": "离开"},
        ]
    }
    decision["action"] = "choose_rest_option"
    decision["parameters"] = {"option_index": 2}
    _write_jsonl(run_dir / "strategy/decisions.jsonl", [decision])

    transcription = importlib.import_module("play_sts2.transcription")
    result = transcription.render_run(run_dir, tmp_path / "transcripts")
    strategy_text = (result.output_dir / "strategy/decisions.txt").read_text(
        encoding="utf-8"
    )

    assert "[0] 休息" in strategy_text
    assert "[1] 锻造" in strategy_text
    assert "[2] 挖掘" in strategy_text
    assert "[3] 离开" in strategy_text
    assert "ACTION: choose_rest_option 2" in strategy_text


def test_render_run_replaces_stale_transcript_tree(tmp_path: Path) -> None:
    """重复渲染时原子替换旧派生目录，不留下已删除战斗。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 重新生成后仍保留旧 transcript 文件。

    Returns:
        None: 此测试只检查派生视图的覆盖语义。
    """
    run_dir = tmp_path / "raw/human/20260827-a1-f2-TEST-SEED"
    (run_dir / "combat").mkdir(parents=True)
    (run_dir / "strategy").mkdir()
    _write_meta(run_dir, battle_count=1, battle_samples=1, strategic_samples=0)
    _write_jsonl(
        run_dir / "combat/battle-f002-01.jsonl",
        [_decision(event_id=1, screen="COMBAT", action="end_turn")],
    )
    output_root = tmp_path / "transcripts"
    transcription = importlib.import_module("play_sts2.transcription")
    transcription.render_run(run_dir, output_root)
    stale = output_root / run_dir.name / "combat/stale.txt"
    stale.write_text("旧文件", encoding="utf-8")

    transcription.render_run(run_dir, output_root)

    assert not stale.exists()


def test_render_run_restores_previous_tree_when_publish_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """上次发布中断后，新发布再次失败时仍保留最近完整 transcript。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。
        monkeypatch (pytest.MonkeyPatch): 用于模拟暂存目录发布失败。

    Raises:
        AssertionError: 两次发布中断导致最近完整 transcript 丢失。

    Returns:
        None: 此测试只检查跨进程恢复边界。
    """
    run_dir = tmp_path / "raw/human/20260827-a1-f2-TEST-SEED"
    (run_dir / "combat").mkdir(parents=True)
    (run_dir / "strategy").mkdir()
    _write_meta(run_dir, battle_count=1, battle_samples=1, strategic_samples=0)
    _write_jsonl(
        run_dir / "combat/battle-f002-01.jsonl",
        [_decision(event_id=1, screen="COMBAT", action="end_turn")],
    )
    output_root = tmp_path / "transcripts"
    transcription = importlib.import_module("play_sts2.transcription")
    result = transcription.render_run(run_dir, output_root)
    previous = output_root / f".{run_dir.name}-previous"
    result.output_dir.rename(previous)
    original_rename = Path.rename

    def fail_staging_publish(path: Path, target: Path) -> Path:
        """只拒绝新暂存树到最终目录的发布。

        Args:
            path (Path): 正在重命名的源路径。
            target (Path): 目标路径。

        Raises:
            OSError: 源是本次暂存目录时固定失败。

        Returns:
            Path: 其他重命名委托给 pathlib。
        """
        if (
            path.parent == output_root
            and path.name.startswith(f".{run_dir.name}-")
            and path != previous
            and Path(target) == result.output_dir
        ):
            raise OSError("simulated publish failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staging_publish)

    with pytest.raises(OSError, match="simulated publish failure"):
        transcription.render_run(run_dir, output_root)

    assert result.output_dir.is_dir()
    assert (result.output_dir / "combat/battle-f002-01.txt").is_file()


def test_render_run_rejects_output_that_overlaps_raw(tmp_path: Path) -> None:
    """拒绝把 transcript 发布到会覆盖 raw 事实源的位置。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 危险输出路径未被拒绝或 raw 被改写。

    Returns:
        None: 此测试只检查数据保护边界。
    """
    run_dir = tmp_path / "raw/human/20260827-a1-f2-TEST-SEED"
    (run_dir / "combat").mkdir(parents=True)
    (run_dir / "strategy").mkdir()
    meta_path = run_dir / "meta.json"
    meta_path.write_text("{}\n", encoding="utf-8")
    transcription = importlib.import_module("play_sts2.transcription")

    with pytest.raises(transcription.TranscriptError, match="不能与 raw 重叠"):
        transcription.render_run(run_dir, run_dir.parent)

    assert meta_path.is_file()


def test_render_run_rejects_truncated_published_raw(tmp_path: Path) -> None:
    """meta 宣称的样本数与分片不一致时不发布误导性 transcript。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 截断 raw 仍被渲染成看似完整的 transcript。

    Returns:
        None: 此测试只检查 raw 完整性边界。
    """
    run_dir = tmp_path / "raw/human/20260827-a1-f2-TRUNCATED"
    (run_dir / "combat").mkdir(parents=True)
    (run_dir / "strategy").mkdir()
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "TRUNCATED",
                "termination_reason": "game_over",
                "training_eligible": True,
                "integrity": {"verified": True, "ineligibility_reasons": []},
                "battle_count": 0,
                "battle_sample_count": 0,
                "strategic_sample_count": 1,
            }
        ),
        encoding="utf-8",
    )

    transcription = importlib.import_module("play_sts2.transcription")
    with pytest.raises(transcription.TranscriptError, match="strategic_sample_count"):
        transcription.render_run(run_dir, tmp_path / "transcripts")


def _decision(
    *,
    event_id: int,
    screen: str,
    action: str,
    option_index: int | None = None,
) -> dict[str, object]:
    """构造一条能由当前 Harness 渲染的原始动作。

    Args:
        event_id (int): Mod 事件编号。
        screen (str): 动作前屏幕。
        action (str): 已执行动作名称。
        option_index (int | None): 可选动作索引。

    Returns:
        dict[str, object]: 当前 raw schema 的动作行。
    """
    battle = screen == "COMBAT"
    state: dict[str, object] = {
        "screen": screen,
        "in_combat": battle,
        "available_actions": [action],
        "run": {
            "character_name": "故障机器人",
            "ascension": 1,
            "act_id": 0,
            "floor": 2,
            "current_hp": 70,
            "max_hp": 75,
            "gold": 99,
            "relics": (
                [
                    {
                        "index": 0,
                        "name": "破损核心",
                        "description": "战斗开始时生成1个闪电充能球。",
                    }
                ]
                if battle
                else []
            ),
            "potions": [],
            "deck": [],
        },
    }
    if battle:
        state["turn"] = 1
        state["combat"] = {
            "player": {
                "current_hp": 70,
                "max_hp": 75,
                "block": 0,
                "energy": 3,
                "stars": 0,
                "focus": 0,
                "powers": [],
                "orbs": [],
            },
            "enemies": [],
            "hand": [],
            "draw_count": 0,
            "discard_count": 0,
        }
    else:
        state["map"] = {
            "available_nodes": [
                {"index": 0, "row": 2, "col": 1, "node_type": "Monster"}
            ]
        }
    parameters = {} if option_index is None else {"option_index": option_index}
    return {
        "event_id": event_id,
        "observed_at": f"2026-08-27T08:00:{event_id:02d}Z",
        "before_state": state,
        "action": action,
        "parameters": parameters,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """把测试动作按 JSONL 格式写入目标文件。

    Args:
        path (Path): 目标文件。
        rows (list[dict[str, object]]): 待写入动作行。

    Returns:
        None: 文件写入完成后返回。
    """
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_meta(
    run_dir: Path,
    *,
    battle_count: int,
    battle_samples: int,
    strategic_samples: int,
) -> None:
    """写入与测试分片严格对账的当前 raw 元数据。

    Args:
        run_dir (Path): 测试局目录。
        battle_count (int): 战斗文件数。
        battle_samples (int): 战斗动作行数。
        strategic_samples (int): 战略动作行数。

    Returns:
        None: 元数据写入完成后返回。
    """
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "TEST-SEED",
                "seed": "TEST-SEED",
                "termination_reason": "game_over",
                "training_eligible": True,
                "integrity": {"verified": True, "ineligibility_reasons": []},
                "battle_count": battle_count,
                "battle_sample_count": battle_samples,
                "strategic_sample_count": strategic_samples,
            }
        ),
        encoding="utf-8",
    )
