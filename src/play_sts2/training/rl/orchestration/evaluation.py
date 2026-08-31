"""使用双 residual 对 3 frozen + 1 fresh seed 运行低成本完整验证。"""

import json
import statistics
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ....client import GameClient, Health
from ....game_launcher import launch_game
from ....inference import OpenAICompatibleProvider
from ....run_start import start_run
from ....runtime import RunOutcome, RunRunner
from ..entrypoint import validate_rl_game_health
from ..full_run import retryable_full_run_error


@dataclass(frozen=True, slots=True)
class PolicyRunEvaluation:
    """保存一条完整验证游戏的结果。

    Args:
        seed (str): 当前冻结或 fresh 游戏种子。
        split (Literal["frozen", "fresh"]): 种子用途。
        victory (bool): 是否整局通关。
        final_floor (int): 终局楼层。
        bosses_cleared (int): 实际完成 Boss 战数。
        battle_count (int): 完成战斗数。
        final_hp (int): 终局当前生命，死亡为零。
        max_hp (int): 终局最大生命。
        decision_count (int): 战略与战斗模型动作总数。
        elapsed_seconds (float): 当前完整游戏墙钟。
        model_error (bool): 是否因明确的 policy 动作上限失败。
        failure_reason (str | None): 模型失败原因。
    """

    seed: str
    split: Literal["frozen", "fresh"]
    victory: bool
    final_floor: int
    bosses_cleared: int
    battle_count: int
    final_hp: int
    max_hp: int
    decision_count: int
    elapsed_seconds: float
    model_error: bool = False
    failure_reason: str | None = None


def build_policy_evaluation_report(
    records: Sequence[PolicyRunEvaluation],
    *,
    frozen_seeds: Sequence[str],
    fresh_seed: str,
) -> dict[str, Any]:
    """核对并汇总固定三条冻结 seed 与一条 fresh seed。

    Args:
        records (Sequence[PolicyRunEvaluation]): 四条完整游戏结果。
        frozen_seeds (Sequence[str]): 永不进入训练的三条固定 seed。
        fresh_seed (str): 本轮唯一未见 seed。

    Raises:
        ValueError: 数量、用途、seed 或结果字段不符合 3+1 合同。

    Returns:
        dict[str, Any]: 逐 seed 结果与四局聚合指标。
    """
    frozen = tuple(frozen_seeds)
    if len(frozen) != 3 or len(set(frozen)) != 3 or fresh_seed in frozen:
        raise ValueError("完整验证必须使用三个不同 frozen seed 加一个 fresh seed")
    ordered = tuple(records)
    expected = (*frozen, fresh_seed)
    if len(ordered) != 4 or tuple(record.seed for record in ordered) != expected:
        raise ValueError("完整验证结果没有按 3+1 seed 顺序对齐")
    if any(
        record.split != ("frozen" if index < 3 else "fresh")
        or record.final_floor < 0
        or record.bosses_cleared < 0
        or record.battle_count < 0
        or record.max_hp <= 0
        or not 0 <= record.final_hp <= record.max_hp
        or record.decision_count < 0
        or record.elapsed_seconds < 0
        for index, record in enumerate(ordered)
    ):
        raise ValueError("完整验证结果字段无效")
    return {
        "format": "stage7_policy_evaluation",
        "games": 4,
        "victories": sum(record.victory for record in ordered),
        "model_errors": sum(record.model_error for record in ordered),
        "mean_floor": sum(record.final_floor for record in ordered) / 4,
        "bosses_cleared": sum(record.bosses_cleared for record in ordered),
        "elapsed_seconds": sum(record.elapsed_seconds for record in ordered),
        "frozen": [asdict(record) for record in ordered[:3]],
        "fresh": asdict(ordered[3]),
    }


def evaluation_tensorboard_payload(report: dict[str, Any]) -> dict[str, Any]:
    """把 3+1 报告投影为总览、frozen 聚合与 fresh 单局曲线。

    Args:
        report (dict[str, Any]): ``build_policy_evaluation_report`` 结果。

    Raises:
        ValueError: 报告缺少三条 frozen 或一条 fresh 结果。

    Returns:
        dict[str, Any]: 可由 TensorBoard writer 展开的嵌套数值指标。
    """
    frozen = report.get("frozen")
    fresh = report.get("fresh")
    if (
        not isinstance(frozen, list)
        or len(frozen) != 3
        or any(not isinstance(item, dict) for item in frozen)
        or not isinstance(fresh, dict)
    ):
        raise ValueError("TensorBoard 需要完整 3+1 验证结果")
    return {
        "validation": {
            "victories": report.get("victories"),
            "model_errors": report.get("model_errors"),
            "mean_floor": report.get("mean_floor"),
            "bosses_cleared": report.get("bosses_cleared"),
        },
        "validation/frozen": {
            "mean_floor": statistics.fmean(item["final_floor"] for item in frozen),
            "victories": sum(bool(item["victory"]) for item in frozen),
            "bosses_cleared": sum(int(item["bosses_cleared"]) for item in frozen),
        },
        "validation/fresh": {
            "floor": fresh.get("final_floor"),
            "victory": fresh.get("victory"),
            "bosses_cleared": fresh.get("bosses_cleared"),
        },
    }


class _EvaluationWorker:
    """在一个本地端口上顺序完成被分配的验证 seed。"""

    def __init__(
        self,
        *,
        worker_id: str,
        executable: Path,
        profile: Path,
        home_root: Path,
        port: int,
        model_url: str,
        strategy_policy: str,
        battle_policy: str,
        character_id: str,
        ascension: int,
        max_tokens: int,
        temperature: float,
    ) -> None:
        """保存验证进程与冻结双 residual 参数。

        Args:
            worker_id (str): 本地 worker 名称。
            executable (Path): v0.111.0 游戏可执行文件。
            profile (Path): 隔离 profile。
            home_root (Path): 当前 worker 验证 HOME 父目录。
            port (int): 独占 Mod 端口。
            model_url (str): A100 vLLM 地址。
            strategy_policy (str): 候选战略 residual 名。
            battle_policy (str): 候选战斗 residual 名。
            character_id (str): 角色稳定 ID。
            ascension (int): 进阶等级。
            max_tokens (int): 单次回复 token 预算。
            temperature (float): 固定低温或确定性采样温度。
        """
        self.worker_id = worker_id
        self._executable = executable
        self._profile = profile
        self._home_root = home_root
        self._port = port
        self._model_url = model_url
        self._strategy_policy = strategy_policy
        self._battle_policy = battle_policy
        self._character_id = character_id
        self._ascension = ascension
        self._max_tokens = max_tokens
        self._temperature = temperature
        self.healths: list[Health] = []

    def run(
        self,
        *,
        seed: str,
        split: Literal["frozen", "fresh"],
        index: int,
    ) -> PolicyRunEvaluation:
        """完成一条无 checkpoint、无训练元数据的验证游戏。

        Args:
            seed (str): 当前游戏种子。
            split (Literal["frozen", "fresh"]): 种子用途。
            index (int): 四局验证中的序号。

        Returns:
            PolicyRunEvaluation: 当前整局结果。
        """
        for attempt in range(2):
            try:
                return self._run_once(
                    seed=seed,
                    split=split,
                    index=index,
                    attempt=attempt,
                )
            except Exception as exc:
                if not retryable_full_run_error(exc) or attempt == 1:
                    raise
        raise RuntimeError("完整验证基础设施重采循环意外结束")

    def _run_once(
        self,
        *,
        seed: str,
        split: Literal["frozen", "fresh"],
        index: int,
        attempt: int,
    ) -> PolicyRunEvaluation:
        """从干净 HOME 执行一次验证尝试。

        Args:
            seed (str): 当前验证 seed。
            split (Literal["frozen", "fresh"]): 验证用途。
            index (int): 四局验证中的序号。
            attempt (int): 基础设施重采序号。

        Returns:
            PolicyRunEvaluation: 正常终局或带部分计数的策略失败。
        """
        home = self._home_root / f"game-{index:02d}-attempt-{attempt}"
        started = time.monotonic()
        with ExitStack() as stack:
            running = stack.enter_context(
                launch_game(
                    self._executable,
                    port=self._port,
                    home=home,
                    profile=self._profile,
                    mode="headless",
                )
            )
            game = stack.enter_context(GameClient(running.base_url))
            strategy = stack.enter_context(
                OpenAICompatibleProvider(
                    self._model_url,
                    model=self._strategy_policy,
                    enable_thinking=False,
                )
            )
            battle = stack.enter_context(
                OpenAICompatibleProvider(
                    self._model_url,
                    model=self._battle_policy,
                    enable_thinking=False,
                )
            )
            self.healths.append(game.health())
            initial = start_run(
                game,
                self._character_id,
                seed=seed,
                ascension=self._ascension,
            )
            result = RunRunner(
                game,
                strategy_provider=strategy,
                battle_provider=battle,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                max_retries=0,
                max_conflict_retries=3,
                constrain_actions=True,
                capture_policy_failures=True,
            )
            completed = result.run(initial)
        final_run = completed.final_state.get("run")
        if not isinstance(final_run, dict):
            raise TypeError("完整验证终局缺少 run 状态")
        victory = completed.outcome is RunOutcome.VICTORY
        return PolicyRunEvaluation(
            seed=seed,
            split=split,
            victory=victory,
            final_floor=int(final_run["floor"]),
            bosses_cleared=completed.bosses_cleared,
            battle_count=completed.battle_count,
            final_hp=int(final_run["current_hp"]) if victory else 0,
            max_hp=int(final_run["max_hp"]),
            decision_count=len(completed.decisions),
            elapsed_seconds=time.monotonic() - started,
            model_error=completed.model_error,
            failure_reason=completed.failure_reason,
        )


def evaluate_policy_pair(
    *,
    executable: Path,
    profile: Path,
    home_root: Path,
    ports: tuple[int, ...],
    model_url: str,
    strategy_policy: str,
    battle_policy: str,
    frozen_seeds: tuple[str, str, str],
    fresh_seed: str,
    output_path: Path,
    character_id: str = "DEFECT",
    ascension: int = 0,
    max_tokens: int = 128,
    temperature: float = 0.0,
    tensorboard_dir: Path | None = None,
    cycle_index: int | None = None,
) -> dict[str, Any]:
    """并发运行固定 3+1 完整验证并写出 policy pair 报告。

    Args:
        executable (Path): v0.111.0 游戏可执行文件。
        profile (Path): 隔离 profile。
        home_root (Path): 四局验证 HOME 父目录。
        ports (tuple[int, ...]): 阶段七允许的一至两个不同本地端口。
        model_url (str): A100 vLLM 地址。
        strategy_policy (str): 候选战略 residual 名。
        battle_policy (str): 候选战斗 residual 名。
        frozen_seeds (tuple[str, str, str]): 三条永不训练 seed。
        fresh_seed (str): 当轮未见 seed。
        output_path (Path): 验证报告 JSON。
        character_id (str): 角色稳定 ID。
        ascension (int): 进阶等级。
        max_tokens (int): 单次回复 token 预算。
        temperature (float): 固定低温或确定性采样温度。
        tensorboard_dir (Path | None): 可选的长期 RL event 目录。
        cycle_index (int | None): TensorBoard 使用的全局训练轮次。

    Raises:
        ValueError: worker 或 3+1 seed 配置无效。

    Returns:
        dict[str, Any]: policy、运行时与四局聚合指标。
    """
    if (tensorboard_dir is None) != (cycle_index is None):
        raise ValueError("TensorBoard 目录与 cycle index 必须同时提供")
    if cycle_index is not None and cycle_index < 0:
        raise ValueError("TensorBoard cycle index 不能为负")
    if not 1 <= len(ports) <= 2 or len(set(ports)) != len(ports):
        raise ValueError("阶段七完整验证只允许一至两个本地端口")
    seeds = (*frozen_seeds, fresh_seed)
    if len(set(seeds)) != 4:
        raise ValueError("完整验证 3+1 seed 必须彼此不同")
    workers = tuple(
        _EvaluationWorker(
            worker_id=f"worker-{index}",
            executable=executable,
            profile=profile,
            home_root=home_root / f"worker-{index}",
            port=port,
            model_url=model_url,
            strategy_policy=strategy_policy,
            battle_policy=battle_policy,
            character_id=character_id,
            ascension=ascension,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for index, port in enumerate(ports)
    )
    allocations = [list(range(index, 4, len(workers))) for index in range(len(workers))]
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = [
            executor.submit(_run_eval_batch, worker, indices, seeds)
            for worker, indices in zip(workers, allocations, strict=True)
            if indices
        ]
        records = tuple(
            sorted(
                (item for future in futures for item in future.result()),
                key=lambda item: seeds.index(item.seed),
            )
        )
    report = build_policy_evaluation_report(
        records,
        frozen_seeds=frozen_seeds,
        fresh_seed=fresh_seed,
    )
    healths = [health for worker in workers for health in worker.healths]
    report.update(
        {
            "strategy_policy": strategy_policy,
            "battle_policy": battle_policy,
            "environment": validate_rl_game_health(healths),
        }
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if tensorboard_dir is not None and cycle_index is not None:
        from .telemetry import TensorboardMetricsWriter

        with TensorboardMetricsWriter(tensorboard_dir) as writer:
            writer.write(
                evaluation_tensorboard_payload(report),
                step=cycle_index,
            )
    return report


def _run_eval_batch(
    worker: _EvaluationWorker,
    indices: Sequence[int],
    seeds: Sequence[str],
) -> tuple[PolicyRunEvaluation, ...]:
    """让一个 worker 顺序完成分配的验证游戏。

    Args:
        worker (_EvaluationWorker): 当前独占端口 worker。
        indices (Sequence[int]): 四局中的序号。
        seeds (Sequence[str]): 固定 3+1 seed 顺序。

    Returns:
        tuple[PolicyRunEvaluation, ...]: 当前 worker 的结果。
    """
    return tuple(
        worker.run(
            seed=seeds[index],
            split="frozen" if index < 3 else "fresh",
            index=index,
        )
        for index in indices
    )
