"""验证完成监听使用真实进程退出事件，并诚实记录通知失败。"""

import subprocess
import sys

import pytest

from play_sts2.training.rl.orchestration.completion_watch import (
    queue_completion,
    wait_for_exit,
)


def test_wait_observes_child_exit_without_polling() -> None:
    """进程仍活着时等待，退出后才返回。"""
    with subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.2)"]
    ) as child:
        assert wait_for_exit(child.pid) == "exited"
        assert child.wait(timeout=1) == 0


def test_already_exited_process_is_reported() -> None:
    """监听注册之前已结束的任务也应进入结果核实。"""
    with subprocess.Popen([sys.executable, "-c", "pass"]) as child:
        child.wait()
        assert wait_for_exit(child.pid) == "already_exited"


def test_notification_failure_is_not_claimed_as_queued(monkeypatch) -> None:
    """CLI拒绝入队必须显式失败，供定期兜底发现。"""

    def fail(*args, **kwargs):
        """模拟通知CLI未能联系桌面服务。"""
        raise subprocess.CalledProcessError(1, args[0], stderr="daemon unavailable")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        queue_completion("test-thread", "report.json", "already_exited")
