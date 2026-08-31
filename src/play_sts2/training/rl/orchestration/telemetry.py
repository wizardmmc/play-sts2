"""把 RL 的小型数值指标写入 TensorBoard event。"""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self


def flatten_tensorboard_metrics(
    metrics: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, float]:
    """递归展开嵌套指标，并忽略文本、空值和非有限数值。

    Args:
        metrics (Mapping[str, Any]): JSON 风格的训练、rollout 或验证指标。
        prefix (str): 可选的 tag 前缀。

    Returns:
        dict[str, float]: 按 tag 排序的有限标量。
    """
    flattened: dict[str, float] = {}
    for key, value in metrics.items():
        tag = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_tensorboard_metrics(value, prefix=tag))
            continue
        if isinstance(value, bool):
            flattened[tag] = float(value)
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            flattened[tag] = float(value)
    return dict(sorted(flattened.items()))


class TensorboardMetricsWriter:
    """在一个长期 run 目录中追加数值标量。"""

    def __init__(
        self,
        log_dir: Path,
        *,
        purge_step: int | None = None,
    ) -> None:
        """创建 PyTorch TensorBoard writer。

        Args:
            log_dir (Path): event 文件目录。
            purge_step (int | None): 精确恢复后隐藏该 step 起的旧 event。

        Raises:
            ImportError: training 依赖中没有安装 TensorBoard。
        """
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(
            log_dir=str(log_dir),
            purge_step=purge_step,
            max_queue=20,
            flush_secs=120,
        )

    def write(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int,
    ) -> None:
        """为一个 optimizer 或 curriculum step 追加全部有限标量。

        Args:
            metrics (Mapping[str, Any]): 可嵌套的指标对象。
            step (int): 非负全局步数。

        Raises:
            ValueError: step 为负。

        Returns:
            None: 标量已进入 TensorBoard 队列。
        """
        if step < 0:
            raise ValueError("TensorBoard step 不能为负")
        for tag, value in flatten_tensorboard_metrics(metrics).items():
            self._writer.add_scalar(tag, value, global_step=step)

    def close(self) -> None:
        """刷新并关闭 event writer。

        Returns:
            None: 队列中的标量已写盘。
        """
        self._writer.flush()
        self._writer.close()

    def __enter__(self) -> Self:
        """进入 writer 生命周期。

        Returns:
            TensorboardMetricsWriter: 当前 writer。
        """
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        """离开上下文时关闭 writer。

        Args:
            _exc_type (object): 可选异常类型。
            _exc_value (object): 可选异常值。
            _traceback (object): 可选 traceback。

        Returns:
            None: writer 总会被关闭。
        """
        self.close()
