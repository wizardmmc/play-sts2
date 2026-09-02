"""在共享服务器上等待一张稳定空闲的 GPU。"""

import json
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path


def parse_free_gpu_indices(
    gpu_output: str,
    compute_output: str,
    *,
    memory_limit_mib: int,
) -> tuple[int, ...]:
    """从 nvidia-smi CSV 中找出无计算进程的低显存 GPU。

    Args:
        gpu_output (str): index、uuid、memory.used 三列 GPU CSV。
        compute_output (str): gpu_uuid、used_memory 两列计算进程 CSV。
        memory_limit_mib (int): 空闲卡允许的最大已用显存，不含该值。

    Raises:
        ValueError: 显存门槛或任一 CSV 行无效。

    Returns:
        tuple[int, ...]: 按物理下标升序排列的空闲 GPU。
    """
    if memory_limit_mib <= 0:
        raise ValueError("GPU 空闲显存门槛必须为正")
    active_uuids = set()
    for line in compute_output.splitlines():
        if not line.strip():
            continue
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 2 or not fields[0]:
            raise ValueError("nvidia-smi 计算进程 CSV 无效")
        _parse_nonnegative_integer(fields[1], "计算进程显存")
        active_uuids.add(fields[0])

    rows = []
    seen_indices = set()
    seen_uuids = set()
    for line in gpu_output.splitlines():
        if not line.strip():
            continue
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 3 or not fields[1]:
            raise ValueError("nvidia-smi GPU CSV 无效")
        index = _parse_nonnegative_integer(fields[0], "GPU 下标")
        memory_used = _parse_nonnegative_integer(fields[2], "GPU 已用显存")
        uuid = fields[1]
        if index in seen_indices or uuid in seen_uuids:
            raise ValueError("nvidia-smi GPU CSV 含重复身份")
        seen_indices.add(index)
        seen_uuids.add(uuid)
        rows.append((index, uuid, memory_used))
    if not rows:
        raise ValueError("nvidia-smi 没有返回 GPU")
    return tuple(
        index
        for index, uuid, memory_used in sorted(rows)
        if uuid not in active_uuids and memory_used < memory_limit_mib
    )


def query_free_gpu_indices(*, memory_limit_mib: int = 1024) -> tuple[int, ...]:
    """调用 nvidia-smi 读取当前空闲 GPU。

    Args:
        memory_limit_mib (int): 空闲卡允许的最大已用显存，不含该值。

    Raises:
        subprocess.CalledProcessError: nvidia-smi 查询失败。
        ValueError: 查询结果不是预期 CSV。

    Returns:
        tuple[int, ...]: 当前无计算进程且低显存的 GPU 下标。
    """
    gpu_output = _nvidia_smi(
        "--query-gpu=index,uuid,memory.used",
    )
    compute_output = _nvidia_smi(
        "--query-compute-apps=gpu_uuid,used_memory",
    )
    return parse_free_gpu_indices(
        gpu_output,
        compute_output,
        memory_limit_mib=memory_limit_mib,
    )


def watch_free_gpu(
    receipt_path: Path,
    *,
    poll_seconds: float = 30.0,
    stable_checks: int = 2,
    memory_limit_mib: int = 1024,
    snapshot_provider: Callable[[], tuple[int, ...]] | None = None,
    sleep_fn: Callable[[float], object] = time.sleep,
    observed_at_fn: Callable[[], str] | None = None,
) -> dict[str, object]:
    """等待同一张卡连续多次空闲并原子写提示收据。

    收据只是启动提示；调用方真正占卡前仍必须再次检查，避免检查与启动之间的竞态。

    Args:
        receipt_path (Path): 固定 JSON 收据路径。
        poll_seconds (float): 两次服务器本地检查的间隔秒数。
        stable_checks (int): 同一卡连续空闲的最少检查次数。
        memory_limit_mib (int): 单次检查的空闲显存门槛。
        snapshot_provider (Callable[[], tuple[int, ...]] | None): 可替换的查询函数。
        sleep_fn (Callable[[float], object]): 可替换的等待函数。
        observed_at_fn (Callable[[], str] | None): 可替换的 UTC 时间函数。

    Raises:
        ValueError: 等待参数或查询结果无效。
        OSError: 收据无法写入。
        subprocess.CalledProcessError: nvidia-smi 查询失败。

    Returns:
        dict[str, object]: 已写入磁盘的 GPU 空闲提示。
    """
    if poll_seconds <= 0:
        raise ValueError("GPU 检查间隔必须为正")
    if stable_checks < 2:
        raise ValueError("GPU 空闲提示至少需要连续两次检查")
    provider = snapshot_provider or (
        lambda: query_free_gpu_indices(memory_limit_mib=memory_limit_mib)
    )
    now = observed_at_fn or _observed_at
    destination = Path(receipt_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    streaks: dict[int, int] = {}
    while True:
        try:
            candidates = provider()
        except (subprocess.CalledProcessError, ValueError) as exc:
            streaks = {}
            print(f"GPU 空闲查询暂时失败: {exc}", file=sys.stderr, flush=True)
            sleep_fn(poll_seconds)
            continue
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in candidates
        ):
            raise ValueError("GPU 空闲查询返回了无效下标")
        current = set(candidates)
        streaks = {index: streaks.get(index, 0) + 1 for index in current}
        ready = sorted(
            index for index, count in streaks.items() if count >= stable_checks
        )
        if ready:
            receipt = {
                "format": "formal_gpu_ready",
                "gpu_index": ready[0],
                "stable_checks": stable_checks,
                "observed_at": now(),
            }
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(destination)
            return receipt
        sleep_fn(poll_seconds)


def _nvidia_smi(query: str) -> str:
    """执行一次无表头、无单位的 nvidia-smi CSV 查询。

    Args:
        query (str): 单个 nvidia-smi 查询参数。

    Returns:
        str: 标准输出 CSV。
    """
    return subprocess.run(
        ["nvidia-smi", query, "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _parse_nonnegative_integer(value: str, label: str) -> int:
    """解析 nvidia-smi 非负整数字段。

    Args:
        value (str): 待解析文本。
        label (str): 错误消息中的字段名。

    Raises:
        ValueError: 文本不是非负整数。

    Returns:
        int: 解析后的数值。
    """
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{label}不是整数") from exc
    if parsed < 0:
        raise ValueError(f"{label}不能为负")
    return parsed


def _observed_at() -> str:
    """返回秒精度 UTC 时间。

    Returns:
        str: 末尾为 Z 的 ISO-8601 时间。
    """
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
