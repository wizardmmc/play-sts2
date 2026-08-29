"""验证 transcript 渲染命令的用户可观察行为。"""

import importlib
import json
from pathlib import Path

import pytest


def test_main_renders_run_to_default_transcript_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """默认把当前 raw 局渲染到同名 transcript 目录。

    Args:
        tmp_path (Path): Pytest 提供的隔离项目目录。
        monkeypatch (pytest.MonkeyPatch): 用于切换隔离工作目录。
        capsys (pytest.CaptureFixture[str]): 用于读取命令标准输出。

    Raises:
        AssertionError: 默认输出位置、内容或完成提示不符合约定。

    Returns:
        None: 此测试只验证命令的用户可观察结果。
    """
    cli = importlib.import_module("play_sts2.transcription.cli")
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "data/raw/human/20260827-a0-f1-CLI-SEED"
    (run_dir / "strategy").mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "CLI-SEED",
                "source": "human",
                "termination_reason": "game_over",
                "training_eligible": True,
                "recording_complete": True,
                "integrity": {
                    "samples_verified": True,
                    "ineligibility_reasons": [],
                },
                "battle_count": 0,
                "battle_sample_count": 0,
                "strategic_sample_count": 1,
            }
        ),
        encoding="utf-8",
    )
    state = {
        "screen": "MAP",
        "available_actions": ["choose_map_node"],
        "run": {
            "character_name": "故障机器人",
            "ascension": 0,
            "act_id": 0,
            "floor": 1,
            "current_hp": 75,
            "max_hp": 75,
            "gold": 99,
            "relics": [],
            "potions": [],
            "deck": [],
        },
        "map": {
            "available_nodes": [
                {"index": 2, "row": 1, "col": 1, "node_type": "Monster"}
            ]
        },
    }
    row = {
        "event_id": 3,
        "observed_at": "2026-08-27T03:00:01.000Z",
        "before_state": state,
        "action": "choose_map_node",
        "parameters": {"option_index": 2},
    }
    (run_dir / "strategy/decisions.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )

    exit_code = cli.main([str(run_dir)])

    output_dir = tmp_path / "data/transcripts/human/20260827-a0-f1-CLI-SEED"
    output_path = output_dir / "strategy/decisions.txt"
    assert exit_code == 0
    assert "ACTION: choose_map_node 2" in output_path.read_text(encoding="utf-8")
    assert capsys.readouterr().out == (
        f"Transcript 完成: {output_dir}\n战斗决策: 0\n战略决策: 1\n"
    )
