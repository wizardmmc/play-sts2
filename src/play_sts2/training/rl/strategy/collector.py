"""用两个隔离游戏 worker 编排总预算 K=8 的 Tree suffix。"""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from .contracts import TreeRolloutArm, TreeRolloutGroup, build_tree_rollout_group


class TreeRolloutWorker(Protocol):
    """声明 Tree collector 对单个本地游戏 worker 的要求。"""

    worker_id: str

    def collect_arm(self, *, arm_index: int) -> TreeRolloutArm:
        """从同一 checkpoint 恢复并完成一条 suffix。

        Args:
            arm_index (int): 当前 group 内的 arm 序号。

        Returns:
            TreeRolloutArm: 带真实行为概率和 continuation return 的 suffix。
        """
        ...


class TreeGroupCollector:
    """把一个 checkpoint 的固定 K=8 预算分配给本地 workers。"""

    def __init__(
        self,
        workers: Sequence[TreeRolloutWorker],
        *,
        group_size: int = 8,
    ) -> None:
        """保存 worker 池并固定 Tree group 总预算。

        Args:
            workers (Sequence[TreeRolloutWorker]): 彼此隔离的游戏 workers。
            group_size (int): 当前实现必须为八条的总 suffix 数。

        Raises:
            ValueError: worker 为空、超过四个或 group_size 不是八。

        Returns:
            None: 此方法只初始化编排器。
        """
        self._workers = tuple(workers)
        if not 1 <= len(self._workers) <= 4:
            raise ValueError("Tree collector 需要一到四个本地游戏 worker")
        if group_size != 8:
            raise ValueError("Tree collector 总预算固定为 K=8")
        self._group_size = group_size

    def collect(
        self,
        *,
        group_id: str,
        checkpoint_kind: str,
        checkpoint_option_ids: Sequence[str],
        checkpoint_policy_text: str,
    ) -> TreeRolloutGroup:
        """并行收集八条 suffix 并执行同 checkpoint group 准入。

        Args:
            group_id (str): 当前组标识。
            checkpoint_kind (str): 宏节点类型。
            checkpoint_option_ids (Sequence[str]): 入口语义候选。
            checkpoint_policy_text (str): 入口玩家可见 observation。

        Returns:
            TreeRolloutGroup: 已计算 branch advantage 的八臂组。
        """
        allocations = [
            list(range(index, self._group_size, len(self._workers)))
            for index in range(len(self._workers))
        ]
        with ThreadPoolExecutor(max_workers=len(self._workers)) as executor:
            batches = tuple(
                executor.map(
                    _collect_batch,
                    self._workers,
                    allocations,
                )
            )
        arms = tuple(arm for batch in batches for arm in batch)
        return build_tree_rollout_group(
            group_id=group_id,
            checkpoint_kind=checkpoint_kind,
            checkpoint_option_ids=checkpoint_option_ids,
            checkpoint_policy_text=checkpoint_policy_text,
            arms=arms,
            expected_size=self._group_size,
        )


def _collect_batch(
    worker: TreeRolloutWorker,
    arm_indices: Sequence[int],
) -> tuple[TreeRolloutArm, ...]:
    """让一个 worker 顺序完成分配给它的 suffix。

    Args:
        worker (TreeRolloutWorker): 当前独占游戏 worker。
        arm_indices (Sequence[int]): 按顺序分配的 arm 序号。

    Returns:
        tuple[TreeRolloutArm, ...]: 当前 worker 完成的 suffix。
    """
    return tuple(worker.collect_arm(arm_index=index) for index in arm_indices)
