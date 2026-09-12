"""阻塞等待训练或评测进程退出，再向原Codex任务提交一次完成通知。"""

import argparse
import errno
import json
import os
import select
import subprocess
import sys
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path


def wait_for_exit(pid: int) -> str:
    """通过内核事件等待进程结束，不轮询日志或调用模型。

    Args:
        pid (int): 本机训练、评测或前台SSH作业进程。

    Returns:
        str: exited或already_exited；不据此断言训练成功。
    """
    if pid <= 0:
        raise ValueError("PID必须为正数")
    try:
        if sys.platform == "darwin":
            with closing(select.kqueue()) as queue:
                event = select.kevent(
                    pid,
                    filter=select.KQ_FILTER_PROC,
                    flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                    fflags=select.KQ_NOTE_EXIT,
                )
                completed = queue.control([event], 1, None)[0]
                if completed.flags & select.KQ_EV_ERROR:
                    if completed.data == errno.ESRCH:
                        return "already_exited"
                    raise OSError(completed.data, os.strerror(completed.data))
        elif sys.platform == "linux":
            descriptor = os.pidfd_open(pid)
            try:
                select.select([descriptor], [], [])
            finally:
                os.close(descriptor)
        else:
            raise RuntimeError("完成事件监听仅支持macOS和Linux")
    except ProcessLookupError:
        return "already_exited"
    return "exited"


def queue_completion(thread: str, report: str, event: str) -> str:
    """向同一桌面任务入队一条待核实的退出通知。

    Returns:
        str: CLI成功入队的收据；失败直接抛出，不伪装成通知成功。
    """
    message = (
        f"实验进程退出事件（自动回调，{event}）。请读取 {report} 与同项目当前status.json，"
        "核实作业是否真正成功、模型身份及结果，再按已有实验单继续工作。"
        "进程退出不等于训练成功；若用户已停止工作则只归档，不重启。"
    )
    result = subprocess.run(
        ["codex", "queue", "--thread", thread, "--message", message],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def main() -> None:
    """为现有进程注册一次性完成通知；远端作业应监听其前台SSH进程。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "pid": args.pid,
        "thread": args.thread,
        "report": str(args.report.resolve()),
        "status": "waiting",
    }
    with args.receipt.open("x") as output:
        json.dump(receipt, output, ensure_ascii=False, indent=2)
    try:
        receipt["event"] = wait_for_exit(args.pid)
        receipt["queue_receipt"] = queue_completion(
            args.thread, str(args.report.resolve()), receipt["event"]
        )
        receipt["status"] = "queued"
    except Exception as exc:
        receipt.update(status="notification_failed", error=str(exc))
        raise
    finally:
        receipt["updated_at"] = datetime.now(UTC).isoformat()
        args.receipt.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
