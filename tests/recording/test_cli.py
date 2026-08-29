"""验证人类轨迹录制命令的用户可观察行为。"""

import importlib
import json
from collections.abc import Iterator
from pathlib import Path
from threading import Event
from types import TracebackType
from typing import Any, Self

import pytest


class CliGameClient:
    """提供足够完成一次命令行录制的生产等价 Mod 协议。

    Args:
        base_url (str): CLI 使用的 Mod 服务地址。
    """

    def __init__(self, base_url: str) -> None:
        """校验默认地址并准备一局即将结束的 SSE 事件。

        Args:
            base_url (str): CLI 使用的 Mod 服务地址。

        Raises:
            AssertionError: CLI 没有使用约定的默认 Mod 地址。
        """
        assert base_url == "http://127.0.0.1:8080"

    def __enter__(self) -> Self:
        """进入测试客户端上下文。

        Returns:
            Self: 当前测试客户端。
        """
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """离开测试客户端上下文。

        Args:
            _exc_type (type[BaseException] | None): 上下文异常类型。
            _exc (BaseException | None): 上下文异常实例。
            _traceback (TracebackType | None): 上下文异常调用栈。

        Returns:
            None: 测试客户端没有需要释放的外部资源。
        """

    def state(self) -> dict[str, Any]:
        """拒绝录制 CLI 重新引入重复 ``/state`` 请求。

        Raises:
            AssertionError: CLI 使用了事件流之外的状态轮询。
        """
        raise AssertionError("录制器不应调用 /state")

    def iter_events(self, stop_event: Event) -> Iterator[dict[str, Any]]:
        """发送就绪事件并等待录制器停止监听。

        Args:
            stop_event (Event): 录制器关闭事件流时设置的信号。

        Yields:
            dict[str, Any]: 与 Mod SSE 同形状的就绪事件。
        """
        yield {
            "event_id": 1,
            "type": "run_started",
            "data": {
                "run_id": "CLI-SEED",
                "character_id": "DEFECT",
                "ascension": 0,
            },
        }
        yield {
            "event_id": 2,
            "type": "run_ended",
            "data": {"run_id": "CLI-SEED", "reason": "game_over"},
        }
        stop_event.wait()


def test_main_records_one_run_to_default_human_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """无额外目录参数时把一局录制到 ``data/raw/human``。

    Args:
        tmp_path (Path): Pytest 提供的隔离项目目录。
        monkeypatch (pytest.MonkeyPatch): 用于隔离工作目录和本地网络边界。
        capsys (pytest.CaptureFixture[str]): 用于读取命令的标准输出。

    Raises:
        AssertionError: CLI 默认参数、落盘结果或完成提示不符合约定。

    Returns:
        None: 此测试只验证命令的用户可观察结果。
    """
    cli = importlib.import_module("play_sts2.recording.cli")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "GameClient", CliGameClient)

    exit_code = cli.main(["--check-interval", "0"])

    human_root = tmp_path / "data/raw/human"
    run_dir = next(path for path in human_root.iterdir() if path.is_dir())
    assert exit_code == 0
    assert run_dir.name.endswith("-a0-f0-CLI-SEED")
    assert (
        json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))["source"]
        == "human"
    )
    metadata = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert metadata["termination_reason"] == "game_over"
    assert metadata["battle_sample_count"] == 0
    assert metadata["strategic_sample_count"] == 0
    assert not (run_dir / "events.jsonl").exists()
    assert capsys.readouterr().out == f"录制完成: {run_dir}\n"


def test_main_records_mixed_teacher_context_from_runtime_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """人机协作入口必须把固定运行时和 Solver 预算写入元数据。

    Args:
        tmp_path (Path): Pytest 提供的隔离项目目录。
        monkeypatch (pytest.MonkeyPatch): 用于隔离工作目录和本地网络边界。

    Raises:
        AssertionError: CLI 未接受混合来源或丢失教师环境元数据。

    Returns:
        None: 此测试只验证命令行参数到 ``meta.json`` 的数据流。
    """
    cli = importlib.import_module("play_sts2.recording.cli")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "GameClient", CliGameClient)
    receipt = tmp_path / "runtime-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "game": {"version": "v0.111.0", "branch": "public-beta"},
                "mods": {
                    "STS2AIAgent": {"version": "0.8.0-rlsts2.46"},
                    "STS2-RitsuLib": {"version": "0.5.18"},
                    "CombatSolver": {"version": "0.17.0"},
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "--source",
            "human_combat_solver",
            "--runtime-receipt",
            str(receipt),
            "--solver-preset",
            "medium",
            "--check-interval",
            "0",
        ]
    )

    run_root = tmp_path / "data/raw/human_combat_solver"
    run_dir = next(path for path in run_root.iterdir() if path.is_dir())
    metadata = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert metadata["source"] == "human_combat_solver"
    assert metadata["recording_context"] == {
        "game": {"version": "v0.111.0", "branch": "public-beta"},
        "mods": {
            "STS2AIAgent": {"version": "0.8.0-rlsts2.46"},
            "STS2-RitsuLib": {"version": "0.5.18"},
            "CombatSolver": {"version": "0.17.0"},
        },
        "solver": {
            "preset": "medium",
            "settings_source": "recorder_argument",
            "short_time_budget_ms": 5_000,
            "deep_time_budget_ms": 60_000,
            "short_node_budget": 1_200,
            "deep_node_budget": 6_000,
            "memory_budget_bytes": 6_000_000_000,
        },
        "student_observation_policy": "visible_only",
        "teacher_uses_hidden_rng": True,
    }
