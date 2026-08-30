"""执行彼此独立的战略页面模型决策。"""

from collections.abc import Mapping
from typing import Any

from ..client import GameClient
from ..inference import DecisionProvider
from .decision import DecisionEngine, DecisionStep
from .router import RunRoute, classify_run_state

_DEFAULT_MAX_RETRIES = 3


class StrategicRunError(RuntimeError):
    """表示当前状态不能由战略 Runner 可靠处理。"""


class StrategicRunner:
    """使用无跨页面历史的上下文执行一次战略动作。"""

    def __init__(
        self,
        game: GameClient,
        provider: DecisionProvider,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        constrain_actions: bool = False,
    ) -> None:
        """初始化不拥有游戏与模型连接生命周期的战略 Runner。

        Args:
            game (GameClient): 已连接到当前游戏实例的客户端。
            provider (DecisionProvider): 已连接到推理服务的模型提供者。
            max_tokens (int): 每次模型回复允许生成的最大 token 数。
            temperature (float): 每次模型回复使用的采样温度。
            max_retries (int): 首次输出失败后允许的重试次数。
            constrain_actions (bool): 是否把当前完整合法动作行交给推理服务约束。

        Returns:
            None: 此方法只组合单步战略闭环需要的依赖。
        """
        self._engine = DecisionEngine(
            game,
            provider,
            max_tokens=max_tokens,
            temperature=temperature,
            constrain_actions=constrain_actions,
        )
        self._max_retries = max_retries

    def step(
        self,
        state: Mapping[str, Any],
        *,
        notice: str | None = None,
    ) -> DecisionStep:
        """根据当前战略页面生成、校验并执行一个模型动作。

        Args:
            state (Mapping[str, Any]): Mod 返回的稳定战略决策状态。
            notice (str | None): 仅针对当前状态附加的运行时纠偏提示。

        Raises:
            StrategicRunError: 当前状态不属于战略决策。
            DecisionRetriesExhausted: 模型输出耗尽重试仍然非法。
            httpx.HTTPStatusError: 推理服务或 Mod 拒绝 HTTP 请求。

        Returns:
            DecisionStep: 本页独立产生的模型消息、动作和执行结果。
        """
        if classify_run_state(state) is not RunRoute.STRATEGIC:
            raise StrategicRunError("当前状态不属于战略决策")
        return self._engine.step(
            state,
            notice=notice,
            max_retries=self._max_retries,
        )
