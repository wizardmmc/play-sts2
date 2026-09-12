"""验证完整游戏采样可由空闲 worker 接手待跑局，且不并发占用同一实例。"""

from threading import Event, Lock
from types import SimpleNamespace

from play_sts2.training.rl.gigpo.collector import BackboneGroupCollector

from .test_contracts import _episode


def test_idle_worker_takes_pending_episodes_while_other_is_busy() -> None:
    """一局阻塞时，其余七局应仍能完成；静态轮流分配会在此等待超时。"""
    others_done = Event()
    completed = set()
    completed_lock = Lock()

    class Worker:
        """模拟独占一个游戏端口的完整局 worker。"""

        def __init__(self, worker_id: str) -> None:
            """建立用于检查同实例并发的锁。"""
            self.worker_id = worker_id
            self.busy = Lock()

        def collect_arm(self, *, arm_index: int) -> SimpleNamespace:
            """第一局等待其他局，返回真实契约可接受的 episode。"""
            assert self.busy.acquire(blocking=False), "同一个游戏实例被并发使用"
            try:
                if arm_index == 0:
                    assert others_done.wait(2), "空闲 worker 没有接手待跑局"
                else:
                    with completed_lock:
                        completed.add(arm_index)
                        if len(completed) == 7:
                            others_done.set()
                return SimpleNamespace(
                    episode=_episode(arm_index, final_floor=arm_index + 1),
                    anchor_audits=({"rng": arm_index},),
                )
            finally:
                self.busy.release()

    result = BackboneGroupCollector([Worker("worker-0"), Worker("worker-1")]).collect(
        group_id="dynamic-eight"
    )
    assert [draft.episode.arm_index for draft in result.drafts] == list(range(8))
    assert len(result.group.episodes) == 8
