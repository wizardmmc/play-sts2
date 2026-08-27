"""验证精确决策转录命令的用户可观察行为。"""

import importlib
import json
from pathlib import Path

import pytest


def test_main_transcribes_run_to_default_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """默认把指定原始局转录到 ``data/transcripts``。

    Args:
        tmp_path (Path): Pytest 提供的隔离项目目录。
        monkeypatch (pytest.MonkeyPatch): 用于切换隔离工作目录。
        capsys (pytest.CaptureFixture[str]): 用于读取命令的标准输出。

    Raises:
        AssertionError: 默认输出位置、决策内容或完成提示不符合约定。

    Returns:
        None: 此测试只验证命令的用户可观察结果。
    """
    cli = importlib.import_module("play_sts2.transcription.cli")
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "data/raw/human/CLI-RUN"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": "CLI-RUN",
                "source": "human",
                "started_at": "2026-08-27T03:00:00.000Z",
                "character_id": "DEFECT",
                "seed": "CLI-RUN",
            }
        ),
        encoding="utf-8",
    )
    before_state = {
        "run_id": "CLI-RUN",
        "screen": "MAP",
        "available_actions": ["choose_map_node"],
    }
    event = {
        "sequence": 5,
        "observed_at": "2026-08-27T03:00:01.000Z",
        "type": "mod_event",
        "payload": {
            "event_id": 3,
            "timestamp_utc": "2026-08-27T03:00:01.000Z",
            "type": "action_executed",
            "data": {
                "request": {
                    "action": "choose_map_node",
                    "card_index": None,
                    "target_index": None,
                    "option_index": 2,
                    "command": None,
                    "client_context": {
                        "source": "human_ui",
                        "layer": "strategic",
                    },
                },
                "before_state": before_state,
                "after_state": before_state,
                "status": "accepted",
                "stable": False,
            },
        },
    }
    (run_dir / "events.jsonl").write_text(
        json.dumps(event) + "\n",
        encoding="utf-8",
    )

    exit_code = cli.main([str(run_dir)])

    output_path = tmp_path / "data/transcripts/CLI-RUN.jsonl"
    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["parameters"] == {
        "option_index": 2
    }
    assert capsys.readouterr().out == (f"转录完成: {output_path}\n人类决策: 1\n")
