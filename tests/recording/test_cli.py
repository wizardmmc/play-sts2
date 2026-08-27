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
        """校验默认地址并准备一局即将结束的状态。

        Args:
            base_url (str): CLI 使用的 Mod 服务地址。

        Raises:
            AssertionError: CLI 没有使用约定的默认 Mod 地址。
        """
        assert base_url == "http://127.0.0.1:8080"
        run_state = {
            "state_version": 1,
            "run_id": "CLI-SEED",
            "screen": "MAP",
            "session": {"phase": "run"},
            "run": {"character_id": "DEFECT", "floor": 1},
        }
        self._states: Iterator[dict[str, Any]] = iter(
            [
                run_state,
                {
                    **run_state,
                    "state_version": 2,
                    "screen": "GAME_OVER",
                    "game_over": {"victory": True},
                },
            ]
        )
        self._last_state = run_state

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
        """返回下一份状态，耗尽后保持最终状态。

        Returns:
            dict[str, Any]: 与 Mod ``GET /state`` 同形状的状态。
        """
        self._last_state = next(self._states, self._last_state)
        return self._last_state

    def iter_events(self, stop_event: Event) -> Iterator[dict[str, Any]]:
        """发送就绪事件并等待录制器停止监听。

        Args:
            stop_event (Event): 录制器关闭事件流时设置的信号。

        Yields:
            dict[str, Any]: 与 Mod SSE 同形状的就绪事件。
        """
        yield {"event_id": 1, "type": "stream_ready", "data": {}}
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

    exit_code = cli.main(["--poll-interval", "0"])

    run_dir = tmp_path / "data/raw/human/CLI-SEED"
    assert exit_code == 0
    assert (
        json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))["source"]
        == "human"
    )
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["payload"] == {"reason": "game_over"}
    assert capsys.readouterr().out == f"录制完成: {run_dir}\n"
