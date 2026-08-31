"""验证 RL TensorBoard 只写小型数值标量。"""

from pathlib import Path


def test_flatten_tensorboard_metrics_keeps_numeric_leaves() -> None:
    """嵌套训练指标应形成稳定 tag，并忽略文本与空值。

    Returns:
        None: loss、胜率和布尔护栏都可写成标量。
    """
    from play_sts2.training.rl.orchestration.telemetry import (
        flatten_tensorboard_metrics,
    )

    metrics = flatten_tensorboard_metrics(
        {
            "train": {"loss": 0.25, "group_id": "group-1"},
            "rollout": {"wins": 2, "stable": True, "note": None},
        }
    )

    assert metrics == {
        "rollout/stable": 1.0,
        "rollout/wins": 2.0,
        "train/loss": 0.25,
    }


def test_tensorboard_writer_creates_readable_scalar_events(tmp_path: Path) -> None:
    """实际 SummaryWriter 产物应可由 TensorBoard event accumulator 读取。

    Args:
        tmp_path (Path): event 输出目录。

    Returns:
        None: 标量 tag 与 step 能从真实 event 文件恢复。
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from play_sts2.training.rl.orchestration import TensorboardMetricsWriter

    with TensorboardMetricsWriter(tmp_path) as writer:
        writer.write({"train": {"loss": 0.5}}, step=7)

    events = EventAccumulator(str(tmp_path))
    events.Reload()

    scalar = events.Scalars("train/loss")
    assert [(item.step, item.value) for item in scalar] == [(7, 0.5)]
