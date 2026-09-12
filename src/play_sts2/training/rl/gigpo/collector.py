"""用一至四个本地游戏 worker 并行收集同种子八局 backbone。"""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event
from typing import Protocol

from .contracts import GigpoGroup, build_gigpo_group
from .rollout import BackboneEpisodeDraft


class BackboneRolloutWorker(Protocol):
    """声明一个本地完整游戏 worker 的最小接口。"""

    worker_id: str

    def collect_arm(self, *, arm_index: int) -> BackboneEpisodeDraft:
        """完成一条完整游戏。

        Args:
            arm_index (int): 同种子组内序号。

        Returns:
            BackboneEpisodeDraft: episode、候选和本地审计。
        """
        ...


@dataclass(frozen=True, slots=True)
class CollectedBackboneGroup:
    """保存训练 group 与本地 Tree/战斗候选。

    Args:
        group (GigpoGroup): 不含隐藏审计的八局训练组。
        drafts (tuple[BackboneEpisodeDraft, ...]): 按 arm 排列的本地完整产物。
    """

    group: GigpoGroup
    drafts: tuple[BackboneEpisodeDraft, ...]


class BackboneGroupCollector:
    """在最多四个本地进程上收集固定 K=8 完整游戏。"""

    def __init__(
        self,
        workers: Sequence[BackboneRolloutWorker],
        *,
        group_size: int = 8,
    ) -> None:
        """保存 worker 池并固定八局预算。

        Args:
            workers (Sequence[BackboneRolloutWorker]): 隔离游戏 workers。
            group_size (int): 必须为八条完整游戏。

        Raises:
            ValueError: worker 数或 group size 无效。
        """
        self._workers = tuple(workers)
        if not 1 <= len(self._workers) <= 4:
            raise ValueError("GiGPO collector 需要一至四个本地游戏 worker")
        if group_size != 8:
            raise ValueError("GiGPO collector 固定使用 K=8 完整游戏")
        self._group_size = group_size

    def collect(
        self,
        *,
        group_id: str,
        lambda_milestone: float = 1.0,
        normalization: str = "one",
    ) -> CollectedBackboneGroup:
        """并行收集八局并执行 GiGPO 准入。

        Args:
            group_id (str): 当前 backbone group 名称。
            lambda_milestone (float): episode-level 进度课程权重。
            normalization (str): ``one`` 或 ``std``。

        Returns:
            CollectedBackboneGroup: 训练组与 Mac 本地候选。
        """
        pending: Queue[int] = Queue()
        for index in range(self._group_size):
            pending.put(index)
        stop = Event()
        batches = []
        failure: BaseException | None = None
        with ThreadPoolExecutor(max_workers=len(self._workers)) as executor:
            futures = [
                executor.submit(_collect_batch, worker, pending, stop)
                for worker in self._workers
            ]
            for future in as_completed(futures):
                exception = future.exception()
                if exception is not None:
                    stop.set()
                    failure = exception
                else:
                    batches.append(future.result())
        if failure is not None:
            raise failure
        drafts = tuple(
            sorted(
                (draft for batch in batches for draft in batch),
                key=lambda draft: draft.episode.arm_index,
            )
        )
        group = build_gigpo_group(
            group_id=group_id,
            episodes=tuple(draft.episode for draft in drafts),
            anchor_audits=tuple(draft.anchor_audits for draft in drafts),
            lambda_milestone=lambda_milestone,
            normalization=normalization,  # type: ignore[arg-type]
        )
        return CollectedBackboneGroup(group=group, drafts=drafts)


def _collect_batch(
    worker: BackboneRolloutWorker,
    arm_indices: Queue[int],
    stop: Event,
) -> tuple[BackboneEpisodeDraft, ...]:
    """让独占实例的 worker 完成一局后再领取下一局。

    Args:
        worker (BackboneRolloutWorker): 独占端口与 HOME 的 worker。
        arm_indices (Queue[int]): 全组尚未开始的 episode 序号队列。
        stop (Event): 另一 worker 失败后阻止启动下一条 episode。

    Returns:
        tuple[BackboneEpisodeDraft, ...]: 按分配顺序完成的游戏。
    """
    drafts = []
    while not stop.is_set():
        try:
            index = arm_indices.get_nowait()
        except Empty:
            break
        try:
            drafts.append(worker.collect_arm(arm_index=index))
        except BaseException:
            stop.set()
            raise
    return tuple(drafts)
