"""区分完整游戏中的策略失败与可有限重采的基础设施故障。"""

import httpx

from ...checkpoint import CheckpointError
from ...runtime import BattleRunError, RunError


def retryable_full_run_error(error: BaseException) -> bool:
    """判断一次完整 run 是否可从干净 HOME 有限重采。

    Args:
        error (BaseException): 游戏启动、Mod、checkpoint 或远程推理异常。

    Returns:
        bool: 只对网络、可重试 HTTP、等待超时、启动和 checkpoint 故障返回真。
    """
    if isinstance(error, (httpx.TransportError, TimeoutError, CheckpointError)):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in {408, 425, 429} or (
            error.response.status_code >= 500
        )
    if isinstance(error, (RunError, BattleRunError)):
        return "超时" in str(error) or "等待" in str(error)
    if isinstance(error, RuntimeError):
        message = str(error)
        return "游戏进程" in message or "启动" in message
    return False
