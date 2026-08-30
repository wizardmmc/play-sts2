"""验证两个 worker 对 Tree K=8 总预算的编排。"""

from typing import Any


class Worker:
    """按 arm 序号返回两个重复计划的确定性 worker。"""

    def __init__(self, worker_id: str) -> None:
        """保存 worker ID 与调用记录。

        Args:
            worker_id (str): 当前本地游戏 worker 名称。

        Returns:
            None: 此方法只初始化记录。
        """
        self.worker_id = worker_id
        self.calls: list[int] = []

    def collect_arm(self, *, arm_index: int) -> Any:
        """返回两个计划各四条的最小合法 arm。

        Args:
            arm_index (int): 当前 K=8 arm 序号。

        Returns:
            TreeRolloutArm: 绑定当前 worker 的 suffix。
        """
        from dataclasses import replace

        from test_tree_contracts import _arm

        self.calls.append(arm_index)
        arm = _arm(
            arm_index,
            "ACTION: choose_map_node 0"
            if arm_index < 4
            else "ACTION: choose_map_node 1",
            1.0 if arm_index < 4 else 2.0,
        )
        return replace(arm, worker_id=self.worker_id)


def test_tree_collector_uses_total_k8_across_two_workers() -> None:
    """两个本地实例应共同完成总计八条 suffix，而不是每个各八条。

    Returns:
        None: arm 按轮询分配且最终组通过 branch advantage 准入。
    """
    from play_sts2.training.rl.strategy.collector import TreeGroupCollector

    workers = (Worker("worker-0"), Worker("worker-1"))
    collector = TreeGroupCollector(workers, group_size=8)

    group = collector.collect(
        group_id="tree-collect",
        checkpoint_kind="map",
        checkpoint_option_ids=("map:2:1", "map:2:3"),
        checkpoint_policy_text="地图入口",
    )

    assert workers[0].calls == [0, 2, 4, 6]
    assert workers[1].calls == [1, 3, 5, 7]
    assert len(group.arms) == 8
    assert group.plan_counts == (
        ("ACTION: choose_map_node 0", 4),
        ("ACTION: choose_map_node 1", 4),
    )
