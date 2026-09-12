"""并行编排多个隔离游戏 worker 收集同场景战斗 arms。"""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import httpx

from ... import RunStartError
from ...inference import InferenceGenerationTruncated
from ...runtime import (
    BattlePolicyFailure,
    BattleRunError,
    BattleRunner,
    BattleStepLimitExceeded,
    DecisionRetriesExhausted,
)
from ...scenario import (
    BattleResetError,
    BattleResetter,
    BattleScenario,
    BattleSnapshot,
    ScenarioVerificationError,
)
from .contracts import (
    BEHAVIOR_LOGPROBS_MODE,
    BattleGroupRejected,
    BattleRollout,
    BattleRolloutGroup,
    RolloutContractError,
    build_battle_rollout_group,
)
from .rollout import build_battle_failure_rollout, build_battle_rollout

_RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class RolloutInfrastructureError(RuntimeError):
    """表示可通过重新采样恢复的游戏、网络或服务故障。"""


class RolloutModelError(RuntimeError):
    """表示模型本身未能完成合法战斗动作，不能按基础设施故障重采。"""


class BattleRolloutWorker(Protocol):
    """声明 collector 对单个隔离游戏 worker 的最小要求。"""

    worker_id: str

    def collect_arm(
        self,
        scenario: BattleScenario,
        *,
        arm_index: int,
        expected_snapshot: BattleSnapshot | None,
    ) -> BattleRollout:
        """重置场景并完成一条战斗 arm。

        Args:
            scenario (BattleScenario): 所有 arms 共享的场景配置。
            arm_index (int): 当前 arm 在 group 内的序号。
            expected_snapshot (BattleSnapshot | None): 首条建立的入口基准。

        Returns:
            BattleRollout: 正常离场并带完整行为概率的战斗采样。
        """
        ...


class GameBattleRolloutWorker:
    """组合一个场景重置器与一个战斗 Runner 完成真实采样。"""

    def __init__(
        self,
        *,
        worker_id: str,
        resetter: BattleResetter,
        runner: BattleRunner,
        behavior_logprobs_mode: str,
    ) -> None:
        """保存当前隔离游戏实例独占的 resetter 与 runner。

        Args:
            worker_id (str): 当前本地游戏 worker 的稳定标识。
            resetter (BattleResetter): 连接该游戏实例的场景重置器。
            runner (BattleRunner): 使用冻结远程 policy 的战斗 Runner。
            behavior_logprobs_mode (str): 服务端声明的行为概率计算模式。

        Raises:
            ValueError: 行为概率不是采样处理后的真实分布。

        Returns:
            None: 此方法只组合单个 worker 的依赖。
        """
        self.worker_id = worker_id
        self._resetter = resetter
        self._runner = runner
        if behavior_logprobs_mode != BEHAVIOR_LOGPROBS_MODE:
            raise ValueError("战斗 worker 必须使用 processed_logprobs 行为概率")
        self._behavior_logprobs_mode = behavior_logprobs_mode

    def collect_arm(
        self,
        scenario: BattleScenario,
        *,
        arm_index: int,
        expected_snapshot: BattleSnapshot | None,
    ) -> BattleRollout:
        """重置统一场景、完成战斗并投影为训练 arm。

        Args:
            scenario (BattleScenario): 当前 group 共享的场景配置。
            arm_index (int): 当前 arm 在 group 内的序号。
            expected_snapshot (BattleSnapshot | None): 首条建立的入口基准。

        Raises:
            RolloutModelError: 模型在有限尝试内没有生成合法动作。
            RolloutInfrastructureError: 游戏、网络或推理服务无法完成本条 arm。
            BattleGroupRejected: 场景装载结果或入口快照不满足组契约。

        Returns:
            BattleRollout: 正常离场并带 token 行为概率的战斗 arm。
        """
        try:
            reset = self._resetter.reset(
                scenario,
                expected_snapshot=expected_snapshot,
            )
            result = self._runner.run(reset.state)
        except BattlePolicyFailure as exc:
            try:
                return build_battle_failure_rollout(
                    arm_index=arm_index,
                    worker_id=self.worker_id,
                    entry_snapshot=reset.snapshot,
                    entry_state=reset.state,
                    failure=exc,
                    behavior_logprobs_mode=self._behavior_logprobs_mode,
                )
            except RolloutContractError as build_error:
                raise RolloutModelError(
                    f"{self.worker_id} 的模型失败没有可训练 token"
                ) from build_error
        except (
            DecisionRetriesExhausted,
            InferenceGenerationTruncated,
            BattleStepLimitExceeded,
        ) as exc:
            raise RolloutModelError(
                f"{self.worker_id} 的模型未能完成有效战斗轨迹"
            ) from exc
        except ScenarioVerificationError as exc:
            raise BattleGroupRejected(f"入口场景验证失败: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _RETRYABLE_HTTP_STATUSES:
                raise
            raise RolloutInfrastructureError(
                f"{self.worker_id} 的战斗采样服务暂时不可用"
            ) from exc
        except (
            httpx.TransportError,
            TimeoutError,
            BattleResetError,
            BattleRunError,
            RunStartError,
        ) as exc:
            raise RolloutInfrastructureError(
                f"{self.worker_id} 的战斗采样基础设施故障"
            ) from exc
        return build_battle_rollout(
            arm_index=arm_index,
            worker_id=self.worker_id,
            entry_snapshot=reset.snapshot,
            entry_state=reset.state,
            result=result,
            behavior_logprobs_mode=self._behavior_logprobs_mode,
        )


class BattleGroupCollector:
    """建立入口基准，并在多个 worker 上收集固定大小的战斗组。"""

    def __init__(
        self,
        workers: Sequence[BattleRolloutWorker],
        *,
        group_size: int = 8,
        infrastructure_attempts: int = 3,
        allow_zero_variance: bool = False,
        on_rollout: Callable[[BattleRollout], None] | None = None,
    ) -> None:
        """保存 worker 池与基础设施重采边界。

        Args:
            workers (Sequence[BattleRolloutWorker]): 彼此隔离的本地游戏 workers。
            group_size (int): 每个同状态 group 的精确 arm 数。
            infrastructure_attempts (int): 单条 arm 遭遇基础设施故障时的总尝试数。
            allow_zero_variance (bool): 是否为评估或独立 DAgger 保留无探索/零优势完整组。
            on_rollout (Callable[[BattleRollout], None] | None): 每臂完成后立即调用；
                可逐臂持久化，回调失败直接传播，不作为游戏故障重采。

        Raises:
            ValueError: worker 为空、group 小于 2 或尝试数小于 1。

        Returns:
            None: 此方法只初始化采样编排器。
        """
        self._workers = tuple(workers)
        if not self._workers:
            raise ValueError("战斗 collector 至少需要一个 worker")
        if group_size < 2:
            raise ValueError("战斗 group_size 不能小于 2")
        if infrastructure_attempts < 1:
            raise ValueError("基础设施总尝试数不能小于 1")
        self._group_size = group_size
        self._infrastructure_attempts = infrastructure_attempts
        self._allow_zero_variance = allow_zero_variance
        self._on_rollout = on_rollout

    def collect(
        self,
        scenario: BattleScenario,
        *,
        group_id: str,
        expected_snapshot: BattleSnapshot | None = None,
    ) -> BattleRolloutGroup:
        """先建立入口基准，再并行收集其余 arms 并执行 group 准入。

        Args:
            scenario (BattleScenario): 待重复构造的确定性战斗。
            group_id (str): 当前采样组的稳定标识。
            expected_snapshot (BattleSnapshot | None): 可选的完整游戏原始战斗入口。

        Raises:
            RolloutInfrastructureError: 某条 arm 重采后仍无法完成。
            RolloutModelError: 模型无法完成合法动作。
            BattleGroupRejected: 完成的 arms 不满足同状态相对学习准入。

        Returns:
            BattleRolloutGroup: 按 arm 序号排列并计算相对优势的完整组。
        """
        first = self._collect_arm(
            self._workers[0],
            scenario,
            arm_index=0,
            expected_snapshot=expected_snapshot,
        )
        allocations: list[list[int]] = [[] for _worker in self._workers]
        for arm_index in range(1, self._group_size):
            allocations[arm_index % len(self._workers)].append(arm_index)

        rollouts = [first]
        with ThreadPoolExecutor(max_workers=len(self._workers)) as executor:
            futures = [
                executor.submit(
                    self._collect_batch,
                    worker,
                    scenario,
                    arm_indices,
                    first.entry_snapshot,
                )
                for worker, arm_indices in zip(
                    self._workers,
                    allocations,
                    strict=True,
                )
                if arm_indices
            ]
            for future in futures:
                rollouts.extend(future.result())

        return build_battle_rollout_group(
            group_id=group_id,
            scenario=scenario,
            rollouts=rollouts,
            expected_size=self._group_size,
            allow_zero_variance=self._allow_zero_variance,
        )

    def _collect_batch(
        self,
        worker: BattleRolloutWorker,
        scenario: BattleScenario,
        arm_indices: Sequence[int],
        expected_snapshot: BattleSnapshot,
    ) -> tuple[BattleRollout, ...]:
        """让一个游戏 worker 顺序完成分配给它的 arms。

        Args:
            worker (BattleRolloutWorker): 独占一个游戏实例的采样 worker。
            scenario (BattleScenario): 当前统一战斗场景。
            arm_indices (Sequence[int]): 分配给该 worker 的 arm 序号。
            expected_snapshot (BattleSnapshot): 首条 arm 建立的入口基准。

        Returns:
            tuple[BattleRollout, ...]: 该 worker 正常完成的全部 arms。
        """
        return tuple(
            self._collect_arm(
                worker,
                scenario,
                arm_index=arm_index,
                expected_snapshot=expected_snapshot,
            )
            for arm_index in arm_indices
        )

    def _collect_arm(
        self,
        worker: BattleRolloutWorker,
        scenario: BattleScenario,
        *,
        arm_index: int,
        expected_snapshot: BattleSnapshot | None,
    ) -> BattleRollout:
        """只对明确分类的基础设施故障重采同一条 arm。

        Args:
            worker (BattleRolloutWorker): 执行当前采样的游戏 worker。
            scenario (BattleScenario): 当前统一战斗场景。
            arm_index (int): 当前 arm 序号。
            expected_snapshot (BattleSnapshot | None): 可选入口基准。

        Raises:
            RolloutInfrastructureError: 达到总尝试数后故障仍存在。

        Returns:
            BattleRollout: 首次正常完成的采样结果。
        """
        for attempt in range(self._infrastructure_attempts):
            try:
                rollout = worker.collect_arm(
                    scenario,
                    arm_index=arm_index,
                    expected_snapshot=expected_snapshot,
                )
            except RolloutInfrastructureError:
                if attempt + 1 == self._infrastructure_attempts:
                    raise
            else:
                if self._on_rollout is not None:
                    self._on_rollout(rollout)
                return rollout
        raise RuntimeError("基础设施重采循环意外结束")
