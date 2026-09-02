"""验证服务器端 GPU 空闲哨兵。"""

import json
import subprocess
from pathlib import Path


def test_free_gpu_parser_rejects_gpu_with_compute_process() -> None:
    """低显存但仍有 compute process 的卡不能被视为空闲。

    Returns:
        None: 只有无 compute process 且显存低于门槛的 GPU 被返回。
    """
    from play_sts2.training.rl.orchestration.gpu_wait import parse_free_gpu_indices

    gpu_output = "0, GPU-a, 4\n1, GPU-b, 800\n2, GPU-c, 1200\n"
    compute_output = "GPU-b, 100\n"

    assert parse_free_gpu_indices(
        gpu_output,
        compute_output,
        memory_limit_mib=1024,
    ) == (0,)


def test_gpu_watcher_requires_two_stable_checks_before_receipt(
    tmp_path: Path,
) -> None:
    """同一张卡必须连续两次空闲才写原子收据。

    Args:
        tmp_path (Path): 收据输出目录。

    Returns:
        None: 中途失去空闲会重置连续检查计数。
    """
    from play_sts2.training.rl.orchestration.gpu_wait import watch_free_gpu

    snapshots = iter(((1,), (), (1,), (1,)))
    sleeps = []
    receipt_path = tmp_path / "gpu-ready.json"

    receipt = watch_free_gpu(
        receipt_path,
        poll_seconds=0.01,
        stable_checks=2,
        snapshot_provider=lambda: next(snapshots),
        sleep_fn=sleeps.append,
        observed_at_fn=lambda: "2026-08-31T10:00:00Z",
    )

    assert receipt == {
        "format": "formal_gpu_ready",
        "gpu_index": 1,
        "stable_checks": 2,
        "observed_at": "2026-08-31T10:00:00Z",
    }
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    assert sleeps == [0.01, 0.01, 0.01]
    assert not receipt_path.with_suffix(".json.tmp").exists()


def test_gpu_watcher_resets_streak_after_transient_query_failure(
    tmp_path: Path,
) -> None:
    """瞬时 nvidia-smi 失败应清零连续计数并继续等待。

    Args:
        tmp_path (Path): 收据输出目录。

    Returns:
        None: 失败后的同一卡仍需重新连续通过两次检查。
    """
    snapshots = iter(
        (
            (1,),
            subprocess.CalledProcessError(1, ["nvidia-smi"]),
            (1,),
            (1,),
        )
    )
    sleeps = []

    def snapshot_provider() -> tuple[int, ...]:
        """返回空闲卡或抛出一次瞬时查询失败。

        Raises:
            subprocess.CalledProcessError: 序列中的模拟查询失败。

        Returns:
            tuple[int, ...]: 当前模拟空闲卡。
        """
        value = next(snapshots)
        if isinstance(value, Exception):
            raise value
        return value

    from play_sts2.training.rl.orchestration.gpu_wait import watch_free_gpu

    receipt = watch_free_gpu(
        tmp_path / "gpu-ready.json",
        poll_seconds=0.01,
        stable_checks=2,
        snapshot_provider=snapshot_provider,
        sleep_fn=sleeps.append,
        observed_at_fn=lambda: "2026-08-31T10:00:00Z",
    )

    assert receipt["gpu_index"] == 1
    assert sleeps == [0.01, 0.01, 0.01]
