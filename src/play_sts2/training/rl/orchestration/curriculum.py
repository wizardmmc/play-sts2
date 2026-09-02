"""持久化长期 RL 的冷启动、迁移、seed 池与逐级进阶课程。"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Literal

SeedSource = Literal["target", "cleared", "hard", "fresh"]
SeedBucket = Literal["cleared", "hard", "fresh"]


class CurriculumPhase(str, Enum):
    """表示当前难度内的长期训练阶段。"""

    COLD_START = "cold_start"
    TRANSFER = "transfer"
    FORMAL = "formal"


@dataclass(frozen=True, slots=True)
class TrainingAssignment:
    """保存下一轮 K=8 backbone 的调度决定。

    Args:
        seed (str): 当前训练 seed。
        source (SeedSource): target、已通关池、困难池或 fresh。
        ascension (int): 当前整轮统一使用的进阶。
        cycle_index (int): 全局零基训练轮次。
        full_validation_due (bool): 本轮是否应执行完整 3+1。
    """

    seed: str
    source: SeedSource
    ascension: int
    cycle_index: int
    full_validation_due: bool


@dataclass(frozen=True, slots=True)
class TrainingCycleResult:
    """保存一轮严格 K=8 完整 backbone 的课程指标。

    Args:
        seed (str): 当前训练 seed。
        ascension (int): 本轮进阶。
        wins (int): 八局完整胜利数。
        attempts (int): 固定为八条完整 rollout。
        mean_floor (float): 八局平均终局楼层。
        bosses_cleared (int): 八局累计击败 Boss 数。
    """

    seed: str
    ascension: int
    wins: int
    attempts: int
    mean_floor: float
    bosses_cleared: int


@dataclass(frozen=True, slots=True)
class LongRunCurriculumState:
    """保存无需隐藏状态即可恢复的长期课程进度。

    Args:
        phase (CurriculumPhase): 当前 cold-start、迁移或正式阶段。
        ascension (int): 当前所有训练 rollout 的统一进阶。
        target_seed (str): 当前难度允许受控背图的固定 seed。
        total_cycles (int): 已完成完整训练轮数。
        phase_cycles (int): 当前 cold-start/迁移阶段已完成轮数。
        transfer_fast_clear_seeds (tuple[str, ...]): 单轮至少一胜的新 seed。
        seen_fresh_seeds (tuple[str, ...]): 当前难度已经使用过的全部 fresh seed。
        cleared_seeds (tuple[str, ...]): 当前难度已出现完整胜利的 seed 池。
        hard_seeds (tuple[str, ...]): 当前难度未通关或低于验证基线的 seed 池。
        active_formal_seed (str | None): 正式阶段正在连续训练的 seed。
        active_formal_source (SeedSource | None): 当前 seed 的来源池。
        active_formal_cycles (int): 当前 seed 已连续训练轮数。
        formal_seed_index (int): 已结束的正式 seed block 数。
        recent_fresh_results (tuple[TrainingCycleResult, ...]): 最近三条完整 fresh
            seed block 结果，用于升进阶。
        last_completed_seed (str | None): 最近结束 block 的 seed，避免立即重抽。
        validation_interval (int): 正式阶段完整 3+1 的轮次间隔，一或二。
    """

    phase: CurriculumPhase
    ascension: int
    target_seed: str
    total_cycles: int = 0
    phase_cycles: int = 0
    transfer_fast_clear_seeds: tuple[str, ...] = ()
    seen_fresh_seeds: tuple[str, ...] = ()
    cleared_seeds: tuple[str, ...] = ()
    hard_seeds: tuple[str, ...] = ()
    active_formal_seed: str | None = None
    active_formal_source: SeedSource | None = None
    active_formal_cycles: int = 0
    formal_seed_index: int = 0
    recent_fresh_results: tuple[TrainingCycleResult, ...] = ()
    last_completed_seed: str | None = None
    validation_interval: int = 1


def new_long_run_curriculum(
    *,
    target_seed: str,
    ascension: int = 0,
) -> LongRunCurriculumState:
    """创建当前难度尚未通关的长期课程。

    Args:
        target_seed (str): 第一条固定训练 seed。
        ascension (int): 初始进阶，默认 A0。

    Raises:
        ValueError: seed 为空或进阶为负。

    Returns:
        LongRunCurriculumState: 全部池为空的 cold-start 状态。
    """
    if not target_seed or ascension < 0:
        raise ValueError("长期 RL target seed 与进阶必须有效")
    return LongRunCurriculumState(
        phase=CurriculumPhase.COLD_START,
        ascension=ascension,
        target_seed=target_seed,
    )


def formal_seed_bucket(seed_index: int) -> SeedBucket:
    """按五个 seed block 返回 20% cleared、40% hard、40% fresh。

    Args:
        seed_index (int): 已完成的正式 seed block 数。

    Raises:
        ValueError: seed block 下标为负。

    Returns:
        SeedBucket: 下一 block 的目标池。
    """
    if seed_index < 0:
        raise ValueError("formal seed index 不能为负")
    return ("cleared", "hard", "hard", "fresh", "fresh")[seed_index % 5]


def next_training_assignment(
    state: LongRunCurriculumState,
    *,
    fresh_seed: str,
) -> TrainingAssignment:
    """根据当前课程状态选择下一轮 seed 与进阶。

    Args:
        state (LongRunCurriculumState): 已完成上一轮后的持久化状态。
        fresh_seed (str): 调用方提供的从未训练的新 seed。

    Raises:
        ValueError: 需要 fresh 时 seed 为空、已见或与当前 target 相同。

    Returns:
        TrainingAssignment: 下一轮 K=8 训练分配。
    """
    if state.phase in {CurriculumPhase.COLD_START, CurriculumPhase.TRANSFER}:
        validation_due = full_validation_due(state)
        if not validation_due:
            return TrainingAssignment(
                seed=state.target_seed,
                source="target",
                ascension=state.ascension,
                cycle_index=state.total_cycles,
                full_validation_due=False,
            )
        _validate_fresh_seed(state, fresh_seed)
        return TrainingAssignment(
            seed=fresh_seed,
            source="fresh",
            ascension=state.ascension,
            cycle_index=state.total_cycles,
            full_validation_due=True,
        )
    if state.active_formal_seed is not None:
        if state.active_formal_source is None:
            raise ValueError("正式课程 active seed 缺少来源")
        return TrainingAssignment(
            seed=state.active_formal_seed,
            source=state.active_formal_source,
            ascension=state.ascension,
            cycle_index=state.total_cycles,
            full_validation_due=full_validation_due(state),
        )
    bucket = formal_seed_bucket(state.formal_seed_index)
    pool = state.cleared_seeds if bucket == "cleared" else state.hard_seeds
    available = tuple(seed for seed in pool if seed != state.last_completed_seed)
    if bucket != "fresh" and available:
        seed = available[state.formal_seed_index % len(available)]
        return TrainingAssignment(
            seed=seed,
            source=bucket,
            ascension=state.ascension,
            cycle_index=state.total_cycles,
            full_validation_due=full_validation_due(state),
        )
    _validate_fresh_seed(state, fresh_seed)
    return TrainingAssignment(
        seed=fresh_seed,
        source="fresh",
        ascension=state.ascension,
        cycle_index=state.total_cycles,
        full_validation_due=full_validation_due(state),
    )


def record_training_result(
    state: LongRunCurriculumState,
    assignment: TrainingAssignment,
    result: TrainingCycleResult,
    *,
    rolling_validation_mean_floor: float,
    validation_stable: bool,
    battle_regression_passed: bool,
    tree_regression_passed: bool,
    validation_over_budget: bool = False,
) -> LongRunCurriculumState:
    """记录一轮结果并推进冷启动、迁移、seed 池或进阶状态。

    Args:
        state (LongRunCurriculumState): 本轮开始前状态。
        assignment (TrainingAssignment): 本轮实际采用的调度。
        result (TrainingCycleResult): 严格 K=8 完整 backbone 结果。
        rolling_validation_mean_floor (float): 最近三次 3+1 的平均楼层基线。
        validation_stable (bool): frozen validation 是否无硬退化。
        battle_regression_passed (bool): 固定战斗回归是否通过。
        tree_regression_passed (bool): held-out Tree 回归是否通过。
        validation_over_budget (bool): 本次完整验证墙钟是否超过整轮 20%。

    Raises:
        ValueError: assignment、K、结果或验证指标不符合课程契约。

    Returns:
        LongRunCurriculumState: 下一轮可直接读取的新状态。
    """
    _validate_training_result(state, assignment, result)
    if (
        not math.isfinite(rolling_validation_mean_floor)
        or rolling_validation_mean_floor < 0
    ):
        raise ValueError("长期 RL rolling validation 平均楼层无效")
    if assignment.source == "fresh" and result.seed not in state.seen_fresh_seeds:
        state = replace(
            state,
            seen_fresh_seeds=(*state.seen_fresh_seeds, result.seed),
        )
    if validation_over_budget and state.phase is CurriculumPhase.FORMAL:
        state = replace(state, validation_interval=2)
    if state.phase is CurriculumPhase.COLD_START:
        return _record_cold_start(state, assignment, result)
    if state.phase is CurriculumPhase.TRANSFER:
        return _record_transfer(
            state,
            assignment,
            result,
            validation_stable=validation_stable,
        )
    return _record_formal(
        state,
        assignment,
        result,
        rolling_validation_mean_floor=rolling_validation_mean_floor,
        validation_stable=validation_stable,
        battle_regression_passed=battle_regression_passed,
        tree_regression_passed=tree_regression_passed,
    )


def ascension_ready(
    results: Sequence[TrainingCycleResult],
    *,
    validation_stable: bool,
    battle_regression_passed: bool,
    tree_regression_passed: bool,
) -> bool:
    """判断三个连续 fresh seed 是否各自在 K=8 中至少四胜。

    Args:
        results (Sequence[TrainingCycleResult]): 最近三个 fresh seed block 结果。
        validation_stable (bool): frozen validation 是否无硬退化。
        battle_regression_passed (bool): 固定战斗回归是否通过。
        tree_regression_passed (bool): held-out Tree 回归是否通过。

    Returns:
        bool: 三个 seed 不同、同进阶、各四胜且全部回归通过时为真。
    """
    recent = tuple(results)[-3:]
    return (
        len(recent) == 3
        and len({result.seed for result in recent}) == 3
        and len({result.ascension for result in recent}) == 1
        and all(result.attempts == 8 and result.wins >= 4 for result in recent)
        and validation_stable
        and battle_regression_passed
        and tree_regression_passed
    )


def write_long_run_curriculum(
    state: LongRunCurriculumState,
    path: Path,
) -> Path:
    """原子覆盖一个可读的长期课程 JSON。

    Args:
        state (LongRunCurriculumState): 当前完整课程状态。
        path (Path): 固定状态文件。

    Returns:
        Path: 实际写入路径。
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {"format": "rl_long_run_curriculum", **asdict(state)}
    payload["phase"] = state.phase.value
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_long_run_curriculum(path: Path) -> LongRunCurriculumState:
    """读取一个长期课程状态文件。

    Args:
        path (Path): ``write_long_run_curriculum`` 写出的 JSON。

    Raises:
        ValueError: 格式或必要字段无效。

    Returns:
        LongRunCurriculumState: 可继续选择下一轮的状态。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("format") != "rl_long_run_curriculum"
    ):
        raise ValueError("长期 RL curriculum 文件格式无效")
    try:
        results = tuple(
            TrainingCycleResult(**dict(item))
            for item in payload.get("recent_fresh_results", [])
        )
        state = LongRunCurriculumState(
            phase=CurriculumPhase(str(payload["phase"])),
            ascension=int(payload["ascension"]),
            target_seed=str(payload["target_seed"]),
            total_cycles=int(payload.get("total_cycles", 0)),
            phase_cycles=int(payload.get("phase_cycles", 0)),
            transfer_fast_clear_seeds=tuple(
                payload.get("transfer_fast_clear_seeds", [])
            ),
            seen_fresh_seeds=tuple(payload.get("seen_fresh_seeds", [])),
            cleared_seeds=tuple(payload.get("cleared_seeds", [])),
            hard_seeds=tuple(payload.get("hard_seeds", [])),
            active_formal_seed=payload.get("active_formal_seed"),
            active_formal_source=payload.get("active_formal_source"),
            active_formal_cycles=int(payload.get("active_formal_cycles", 0)),
            formal_seed_index=int(payload.get("formal_seed_index", 0)),
            recent_fresh_results=results,
            last_completed_seed=payload.get("last_completed_seed"),
            validation_interval=int(payload.get("validation_interval", 1)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("长期 RL curriculum 字段无效") from exc
    _validate_state(state)
    return state


def _record_cold_start(
    state: LongRunCurriculumState,
    assignment: TrainingAssignment,
    result: TrainingCycleResult,
) -> LongRunCurriculumState:
    """推进当前难度首次固定图通关阶段。

    Args:
        state (LongRunCurriculumState): cold-start 状态。
        assignment (TrainingAssignment): 本轮 target 或 fresh 分配。
        result (TrainingCycleResult): 本轮 K=8 结果。

    Returns:
        LongRunCurriculumState: 继续 3+1 或进入迁移阶段的状态。
    """
    if assignment.source == "target" and result.wins > 0:
        return replace(
            state,
            phase=CurriculumPhase.TRANSFER,
            total_cycles=state.total_cycles + 1,
            phase_cycles=0,
        )
    return replace(
        state,
        total_cycles=state.total_cycles + 1,
        phase_cycles=state.phase_cycles + 1,
    )


def _record_transfer(
    state: LongRunCurriculumState,
    assignment: TrainingAssignment,
    result: TrainingCycleResult,
    *,
    validation_stable: bool,
) -> LongRunCurriculumState:
    """推进首次通关后的跨 seed 迁移阶段。

    Args:
        state (LongRunCurriculumState): transfer 状态。
        assignment (TrainingAssignment): 本轮 target 或 fresh 分配。
        result (TrainingCycleResult): 本轮 K=8 结果。
        validation_stable (bool): frozen validation 是否无硬退化。

    Returns:
        LongRunCurriculumState: 继续 3+1 或进入正式 seed 池的状态。
    """
    fast = state.transfer_fast_clear_seeds
    if (
        assignment.source == "fresh"
        and result.wins > 0
        and validation_stable
        and result.seed not in fast
    ):
        fast = (*fast, result.seed)
    if len(fast) >= 2:
        return replace(
            state,
            phase=CurriculumPhase.FORMAL,
            total_cycles=state.total_cycles + 1,
            phase_cycles=0,
            transfer_fast_clear_seeds=fast,
            cleared_seeds=fast,
        )
    return replace(
        state,
        total_cycles=state.total_cycles + 1,
        phase_cycles=state.phase_cycles + 1,
        transfer_fast_clear_seeds=fast,
    )


def _record_formal(
    state: LongRunCurriculumState,
    assignment: TrainingAssignment,
    result: TrainingCycleResult,
    *,
    rolling_validation_mean_floor: float,
    validation_stable: bool,
    battle_regression_passed: bool,
    tree_regression_passed: bool,
) -> LongRunCurriculumState:
    """推进正式 seed block、池分类和逐级进阶。

    Args:
        state (LongRunCurriculumState): formal 状态。
        assignment (TrainingAssignment): 当前池选择。
        result (TrainingCycleResult): 本轮 K=8 结果。
        rolling_validation_mean_floor (float): 最近验证楼层基线。
        validation_stable (bool): frozen validation 是否稳定。
        battle_regression_passed (bool): 战斗回归是否通过。
        tree_regression_passed (bool): Tree 回归是否通过。

    Returns:
        LongRunCurriculumState: 保持当前 seed、切换 block 或升进阶的状态。
    """
    previous_cycles = (
        state.active_formal_cycles if state.active_formal_seed == assignment.seed else 0
    )
    cycles = previous_cycles + 1
    if result.wins == 0 and cycles < 3:
        return replace(
            state,
            total_cycles=state.total_cycles + 1,
            active_formal_seed=assignment.seed,
            active_formal_source=assignment.source,
            active_formal_cycles=cycles,
        )
    cleared = tuple(seed for seed in state.cleared_seeds if seed != result.seed)
    hard = tuple(seed for seed in state.hard_seeds if seed != result.seed)
    if result.wins > 0:
        cleared = (*cleared, result.seed)
    elif result.mean_floor < rolling_validation_mean_floor or cycles >= 3:
        hard = (*hard, result.seed)
    recent = state.recent_fresh_results
    if assignment.source == "fresh":
        recent = (*recent, result)[-3:]
    common = replace(
        state,
        total_cycles=state.total_cycles + 1,
        cleared_seeds=cleared,
        hard_seeds=hard,
        active_formal_seed=None,
        active_formal_source=None,
        active_formal_cycles=0,
        formal_seed_index=state.formal_seed_index + 1,
        recent_fresh_results=recent,
        last_completed_seed=result.seed,
    )
    if not ascension_ready(
        recent,
        validation_stable=validation_stable,
        battle_regression_passed=battle_regression_passed,
        tree_regression_passed=tree_regression_passed,
    ):
        return common
    return LongRunCurriculumState(
        phase=CurriculumPhase.COLD_START,
        ascension=state.ascension + 1,
        target_seed=result.seed,
        total_cycles=state.total_cycles + 1,
        validation_interval=state.validation_interval,
    )


def _validate_training_result(
    state: LongRunCurriculumState,
    assignment: TrainingAssignment,
    result: TrainingCycleResult,
) -> None:
    """核对 assignment 与严格 K=8 结果一致。

    Args:
        state (LongRunCurriculumState): 本轮开始状态。
        assignment (TrainingAssignment): 调度决定。
        result (TrainingCycleResult): 实际完整游戏结果。

    Raises:
        ValueError: seed、进阶、轮次、K 或指标无效。

    Returns:
        None: 结果可推进课程时返回。
    """
    if (
        assignment.cycle_index != state.total_cycles
        or assignment.ascension != state.ascension
        or result.seed != assignment.seed
        or result.ascension != assignment.ascension
        or result.attempts != 8
        or result.wins < 0
        or result.wins > result.attempts
        or not math.isfinite(result.mean_floor)
        or result.mean_floor < 0
        or result.bosses_cleared < 0
    ):
        raise ValueError("长期 RL assignment 与 K=8 结果不一致")


def _validate_fresh_seed(state: LongRunCurriculumState, seed: str) -> None:
    """要求调用方提供当前难度从未训练的新 seed。

    Args:
        state (LongRunCurriculumState): 当前课程状态。
        seed (str): fresh 候选。

    Raises:
        ValueError: seed 为空或已经进入任一当前难度池。

    Returns:
        None: seed 可作为 fresh 使用时返回。
    """
    seen = {
        state.target_seed,
        *state.transfer_fast_clear_seeds,
        *state.seen_fresh_seeds,
        *state.cleared_seeds,
        *state.hard_seeds,
    }
    if not seed or seed in seen:
        raise ValueError("长期 RL fresh seed 必须从未在当前难度训练")


def _validate_state(state: LongRunCurriculumState) -> None:
    """核对落盘课程状态的必要计数与池字段。

    Args:
        state (LongRunCurriculumState): 已解析状态。

    Raises:
        ValueError: 进阶、游标、target 或 active 字段无效。

    Returns:
        None: 状态可继续调度时返回。
    """
    if (
        state.ascension < 0
        or not state.target_seed
        or state.total_cycles < 0
        or state.phase_cycles < 0
        or state.active_formal_cycles < 0
        or state.formal_seed_index < 0
        or state.validation_interval not in {1, 2}
        or (state.active_formal_seed is None) != (state.active_formal_source is None)
    ):
        raise ValueError("长期 RL curriculum 状态无效")


def full_validation_due(state: LongRunCurriculumState) -> bool:
    """判断下一轮是否需要完整 3+1 validation。

    Args:
        state (LongRunCurriculumState): 当前课程状态。

    Returns:
        bool: 冷启动/迁移仅 training-fresh 轮执行；正式阶段按间隔执行。
    """
    if state.phase in {CurriculumPhase.COLD_START, CurriculumPhase.TRANSFER}:
        return state.phase_cycles % 4 == 3
    return state.total_cycles % state.validation_interval == 0
