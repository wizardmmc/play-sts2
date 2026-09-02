"""验证长期 RL 的冷启动、迁移、seed 池和升进阶课程。"""

from pathlib import Path

import pytest


def test_cold_start_inserts_one_fresh_cycle_after_three_target_cycles() -> None:
    """首次通关前应按三个 target 加一个 fresh 的节奏训练。

    Returns:
        None: 前八轮重复两个完整四轮周期。
    """
    from play_sts2.training.rl.orchestration import (
        new_long_run_curriculum,
        next_training_assignment,
        record_training_result,
    )

    state = new_long_run_curriculum(target_seed="AAAAAAAAAA", ascension=0)
    sources = []
    seeds = []
    for index in range(8):
        assignment = next_training_assignment(
            state,
            fresh_seed=f"FRESH{index:05d}",
        )
        sources.append(assignment.source)
        seeds.append(assignment.seed)
        state = record_training_result(
            state,
            assignment,
            _result(assignment.seed, ascension=0),
            rolling_validation_mean_floor=5.0,
            validation_stable=True,
            battle_regression_passed=True,
            tree_regression_passed=True,
        )

    assert tuple(sources) == (
        "target",
        "target",
        "target",
        "fresh",
        "target",
        "target",
        "target",
        "fresh",
    )
    assert seeds[0:3] == ["AAAAAAAAAA"] * 3
    assert seeds[4:7] == ["AAAAAAAAAA"] * 3


def test_cold_start_runs_full_validation_only_after_training_fresh_cycle() -> None:
    """冷启动完整验证应绑定到每个三 target 加一 fresh 训练块末尾。

    Returns:
        None: 前三轮跳过完整验证，training-fresh 轮才执行一次 3+1。
    """
    from play_sts2.training.rl.orchestration import (
        new_long_run_curriculum,
        next_training_assignment,
        record_training_result,
    )

    state = new_long_run_curriculum(target_seed="AAAAAAAAAA", ascension=0)
    due = []
    for index in range(4):
        assignment = next_training_assignment(
            state,
            fresh_seed=f"FRESH{index:05d}",
        )
        due.append(assignment.full_validation_due)
        state = record_training_result(
            state,
            assignment,
            _result(assignment.seed, ascension=0),
            rolling_validation_mean_floor=5.0,
            validation_stable=True,
            battle_regression_passed=True,
            tree_regression_passed=True,
        )

    assert due == [False, False, False, True]


def test_transfer_requires_two_distinct_fresh_one_cycle_clears() -> None:
    """固定图首次通关后不能靠一个幸运 fresh seed 直接进入正式循环。

    Returns:
        None: 两个不同 fresh seed 均在单轮 K=8 中至少一胜才进入 formal。
    """
    from play_sts2.training.rl.orchestration import (
        CurriculumPhase,
        new_long_run_curriculum,
        next_training_assignment,
        record_training_result,
    )

    state = new_long_run_curriculum(target_seed="AAAAAAAAAA", ascension=0)
    target = next_training_assignment(state, fresh_seed="IGNORED000")
    state = record_training_result(
        state,
        target,
        _result(target.seed, ascension=0, wins=1),
        rolling_validation_mean_floor=5.0,
        validation_stable=True,
        battle_regression_passed=True,
        tree_regression_passed=True,
    )
    assert state.phase is CurriculumPhase.TRANSFER

    state = _advance_to_fresh(state, "FRESH00001", wins=1)
    assert state.phase is CurriculumPhase.TRANSFER
    assert state.transfer_fast_clear_seeds == ("FRESH00001",)

    state = _advance_to_fresh(state, "FRESH00002", wins=1)
    assert state.phase is CurriculumPhase.FORMAL
    assert state.transfer_fast_clear_seeds == ("FRESH00001", "FRESH00002")
    assert state.cleared_seeds == ("FRESH00001", "FRESH00002")


def test_transfer_runs_full_validation_only_after_training_fresh_cycle() -> None:
    """迁移阶段也应只在 training-fresh 轮末执行完整验证。

    Returns:
        None: 三个 target assignment 跳过验证，第四个 fresh assignment 执行。
    """
    from play_sts2.training.rl.orchestration import (
        new_long_run_curriculum,
        next_training_assignment,
        record_training_result,
    )

    state = new_long_run_curriculum(target_seed="AAAAAAAAAA", ascension=0)
    first = next_training_assignment(state, fresh_seed="FRESH00001")
    state = record_training_result(
        state,
        first,
        _result(first.seed, ascension=0, wins=1),
        rolling_validation_mean_floor=5.0,
        validation_stable=True,
        battle_regression_passed=True,
        tree_regression_passed=True,
    )
    due = []
    for index in range(4):
        assignment = next_training_assignment(
            state,
            fresh_seed=f"FRESH{index + 10:05d}",
        )
        due.append(assignment.full_validation_due)
        state = record_training_result(
            state,
            assignment,
            _result(assignment.seed, ascension=0),
            rolling_validation_mean_floor=5.0,
            validation_stable=True,
            battle_regression_passed=True,
            tree_regression_passed=True,
        )

    assert due == [False, False, False, True]


def test_non_formal_validation_budget_does_not_change_formal_interval() -> None:
    """冷启动块末验证超预算不能提前改变正式阶段默认间隔。

    Returns:
        None: 非正式阶段保留 validation_interval=1。
    """
    from play_sts2.training.rl.orchestration import (
        new_long_run_curriculum,
        next_training_assignment,
        record_training_result,
    )

    state = new_long_run_curriculum(target_seed="AAAAAAAAAA", ascension=0)
    assignment = next_training_assignment(state, fresh_seed="FRESH00001")
    state = record_training_result(
        state,
        assignment,
        _result(assignment.seed, ascension=0),
        rolling_validation_mean_floor=5.0,
        validation_stable=True,
        battle_regression_passed=True,
        tree_regression_passed=True,
        validation_over_budget=True,
    )

    assert state.validation_interval == 1


def test_failed_fresh_seed_cannot_be_reused_as_new() -> None:
    """没有通关的 fresh seed 也已经见过，后续不能再次冒充新图。

    Returns:
        None: 下一个 3+1 窗口重用同 seed 时明确拒绝。
    """
    from play_sts2.training.rl.orchestration import (
        new_long_run_curriculum,
        next_training_assignment,
        record_training_result,
    )

    state = new_long_run_curriculum(target_seed="AAAAAAAAAA", ascension=0)
    for _ in range(4):
        assignment = next_training_assignment(state, fresh_seed="FRESH00009")
        state = record_training_result(
            state,
            assignment,
            _result(assignment.seed, ascension=0),
            rolling_validation_mean_floor=5.0,
            validation_stable=True,
            battle_regression_passed=True,
            tree_regression_passed=True,
        )
    for _ in range(3):
        assignment = next_training_assignment(state, fresh_seed="IGNORED000")
        state = record_training_result(
            state,
            assignment,
            _result(assignment.seed, ascension=0),
            rolling_validation_mean_floor=5.0,
            validation_stable=True,
            battle_regression_passed=True,
            tree_regression_passed=True,
        )

    with pytest.raises(ValueError, match="fresh seed"):
        next_training_assignment(state, fresh_seed="FRESH00009")


def test_formal_seed_repeats_at_most_three_cycles_then_enters_hard_pool() -> None:
    """正式循环中的同一 seed 最多连续训练三轮。

    Returns:
        None: 三轮仍未通关的 fresh seed 进入 hard 池并切换下一个 seed。
    """
    from play_sts2.training.rl.orchestration import (
        next_training_assignment,
        record_training_result,
    )

    state = _formal_state()
    assignments = []
    for _ in range(3):
        assignment = next_training_assignment(state, fresh_seed="FRESH00003")
        assignments.append(assignment)
        state = record_training_result(
            state,
            assignment,
            _result(assignment.seed, ascension=0, mean_floor=4.0),
            rolling_validation_mean_floor=8.0,
            validation_stable=True,
            battle_regression_passed=True,
            tree_regression_passed=True,
        )

    trained_seed = assignments[0].seed
    assert {assignment.seed for assignment in assignments} == {trained_seed}
    assert trained_seed in {"TRANSFER01", "TRANSFER02"}
    assert trained_seed in state.hard_seeds
    assert trained_seed not in state.cleared_seeds
    assert state.active_formal_seed is None
    assert next_training_assignment(state, fresh_seed="FRESH00004").seed != trained_seed


def test_three_distinct_fresh_four_of_eight_clears_raise_ascension() -> None:
    """连续三个新 seed 各至少四胜并通过回归后才提高当前进阶。

    Returns:
        None: 晋级后清空当前难度的池并回到新难度 cold-start。
    """
    from play_sts2.training.rl.orchestration import (
        CurriculumPhase,
        next_training_assignment,
        record_training_result,
    )

    state = _formal_state()
    replay = next_training_assignment(state, fresh_seed="UNUSED0001")
    assert replay.source == "cleared"
    state = record_training_result(
        state,
        replay,
        _result(replay.seed, ascension=0, wins=1, mean_floor=51.0, bosses=3),
        rolling_validation_mean_floor=20.0,
        validation_stable=True,
        battle_regression_passed=True,
        tree_regression_passed=True,
    )
    for seed in ("FRESH10001", "FRESH10002", "FRESH10003"):
        assignment = next_training_assignment(state, fresh_seed=seed)
        assert assignment.source == "fresh"
        state = record_training_result(
            state,
            assignment,
            _result(seed, ascension=0, wins=4, mean_floor=51.0, bosses=3),
            rolling_validation_mean_floor=20.0,
            validation_stable=True,
            battle_regression_passed=True,
            tree_regression_passed=True,
        )

    assert state.ascension == 1
    assert state.phase is CurriculumPhase.COLD_START
    assert state.target_seed == "FRESH10003"
    assert state.cleared_seeds == ()
    assert state.hard_seeds == ()


def test_curriculum_state_round_trips_without_hidden_scheduler_state(
    tmp_path: Path,
) -> None:
    """长期课程状态应通过一个可读 JSON 原子恢复。

    Args:
        tmp_path (Path): journal 输出目录。

    Returns:
        None: 课程阶段、难度、池和游标无损恢复。
    """
    from play_sts2.training.rl.orchestration import (
        load_long_run_curriculum,
        new_long_run_curriculum,
        write_long_run_curriculum,
    )

    state = new_long_run_curriculum(target_seed="AAAAAAAAAA", ascension=0)
    path = write_long_run_curriculum(state, tmp_path / "curriculum.json")

    assert load_long_run_curriculum(path) == state


def test_formal_bucket_schedule_uses_twenty_forty_forty_ratio() -> None:
    """正式循环五个 seed block 应包含一 cleared、两 hard 和两 fresh。

    Returns:
        None: 调度顺序固定，池内具体 seed 才做轮转。
    """
    from play_sts2.training.rl.orchestration import formal_seed_bucket

    buckets = tuple(formal_seed_bucket(index) for index in range(5))

    assert buckets == ("cleared", "hard", "hard", "fresh", "fresh")


def test_validation_over_budget_switches_full_three_plus_one_to_every_two_cycles() -> (
    None
):
    """正式阶段验证超过 20% 墙钟后应隔轮执行。

    Returns:
        None: 第一次超预算后下一轮跳过，再下一轮恢复完整验证。
    """
    from dataclasses import replace

    from play_sts2.training.rl.orchestration import (
        next_training_assignment,
        record_training_result,
    )

    state = replace(_formal_state(), total_cycles=0)
    first = next_training_assignment(state, fresh_seed="FRESH20001")
    assert first.full_validation_due is True
    state = record_training_result(
        state,
        first,
        _result(first.seed, ascension=0),
        rolling_validation_mean_floor=5.0,
        validation_stable=True,
        battle_regression_passed=True,
        tree_regression_passed=True,
        validation_over_budget=True,
    )
    second = next_training_assignment(state, fresh_seed="FRESH20002")
    assert second.full_validation_due is False
    state = record_training_result(
        state,
        second,
        _result(second.seed, ascension=0),
        rolling_validation_mean_floor=5.0,
        validation_stable=True,
        battle_regression_passed=True,
        tree_regression_passed=True,
    )

    assert (
        next_training_assignment(
            state,
            fresh_seed="FRESH20003",
        ).full_validation_due
        is True
    )


def _advance_to_fresh(state: object, fresh_seed: str, *, wins: int):
    """把 3+1 迁移阶段推进到下一次 fresh 并记录结果。

    Args:
        state (object): 当前长期课程状态。
        fresh_seed (str): 下一条新 seed。
        wins (int): fresh K=8 胜局数。

    Returns:
        LongRunCurriculumState: 记录 fresh 结果后的状态。
    """
    from play_sts2.training.rl.orchestration import (
        next_training_assignment,
        record_training_result,
    )

    while True:
        assignment = next_training_assignment(state, fresh_seed=fresh_seed)
        state = record_training_result(
            state,
            assignment,
            _result(
                assignment.seed,
                ascension=assignment.ascension,
                wins=wins if assignment.source == "fresh" else 0,
            ),
            rolling_validation_mean_floor=5.0,
            validation_stable=True,
            battle_regression_passed=True,
            tree_regression_passed=True,
        )
        if assignment.source == "fresh":
            return state


def _formal_state():
    """通过真实 cold-start 与 transfer 规则构造 formal 状态。

    Returns:
        LongRunCurriculumState: 已由两个不同 fresh seed 验证迁移的状态。
    """
    from play_sts2.training.rl.orchestration import (
        new_long_run_curriculum,
        next_training_assignment,
        record_training_result,
    )

    state = new_long_run_curriculum(target_seed="AAAAAAAAAA", ascension=0)
    target = next_training_assignment(state, fresh_seed="IGNORED000")
    state = record_training_result(
        state,
        target,
        _result(target.seed, ascension=0, wins=1),
        rolling_validation_mean_floor=5.0,
        validation_stable=True,
        battle_regression_passed=True,
        tree_regression_passed=True,
    )
    state = _advance_to_fresh(state, "TRANSFER01", wins=1)
    return _advance_to_fresh(state, "TRANSFER02", wins=1)


def _result(
    seed: str,
    *,
    ascension: int,
    wins: int = 0,
    mean_floor: float = 3.0,
    bosses: int = 0,
):
    """构造一个严格 K=8 的完整训练 cycle 结果。

    Args:
        seed (str): 当前训练 seed。
        ascension (int): 当前进阶。
        wins (int): 八条 backbone 胜局数。
        mean_floor (float): 八局平均终局楼层。
        bosses (int): 八局总 Boss 击败数。

    Returns:
        TrainingCycleResult: 可交给课程 controller 的结果。
    """
    from play_sts2.training.rl.orchestration import TrainingCycleResult

    return TrainingCycleResult(
        seed=seed,
        ascension=ascension,
        wins=wins,
        attempts=8,
        mean_floor=mean_floor,
        bosses_cleared=bosses,
    )
