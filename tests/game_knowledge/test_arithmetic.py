"""验证可重建的战斗算术候选生成。"""

import hashlib
import json
from pathlib import Path

import pytest

from play_sts2.game_knowledge.arithmetic import (
    build_arithmetic_samples,
    generate_arithmetic_candidates,
    load_observed_attack_values,
)


def _write_human_run(
    human_root: Path,
    run_id: str,
    damage: int,
    *,
    training_eligible: bool = True,
) -> Path:
    """写入一局可由 raw 审计器验证的最小战斗记录。

    Args:
        human_root (Path): ``data/raw/human`` 风格根目录。
        run_id (str): 写入 meta 与分卷名册的局 ID。
        damage (int): 唯一单段攻击意图数值。
        training_eligible (bool): 该局是否允许进入训练。

    Returns:
        Path: 写入的战斗 JSONL 路径。
    """
    run_dir = human_root / f"20260828-a0-f1-{run_id}"
    combat = run_dir / "combat"
    combat.mkdir(parents=True)
    state = {
        "combat": {
            "enemies": [
                {
                    "name": "测试敌人",
                    "intents": [
                        {
                            "intent_type": "Attack",
                            "damage": damage,
                            "hits": 1,
                        }
                    ],
                }
            ]
        }
    }
    battle_path = combat / "battle.jsonl"
    battle_path.write_text(
        json.dumps({"before_state": state}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": run_id,
                "termination_reason": "stream_interrupted",
                "training_eligible": training_eligible,
                "recording_complete": False,
                "integrity": {"samples_verified": training_eligible},
                "battle_count": 1,
                "battle_sample_count": 1,
                "strategic_sample_count": 0,
            }
        ),
        encoding="utf-8",
    )
    return battle_path


def test_arithmetic_samples_are_unique_and_require_the_answer_style() -> None:
    """CoT 与直答候选的问题应互斥，并明确要求回答风格。

    Raises:
        AssertionError: 候选问题重复，或没有区分计算过程与直答。

    Returns:
        None: 此测试只验证确定性生成器的样本契约。
    """
    worked, direct = build_arithmetic_samples(
        attack_values=[7, 11],
        quotas={
            "block_math": 3,
            "lethal": 3,
            "status_math": 3,
            "multihit": 3,
            "energy_math": 3,
            "orb_focus": 3,
        },
        direct_count=6,
    )

    prompts = [str(row["prompt"]) for row in worked + direct]
    assert len(prompts) == len(set(prompts))
    assert all("请写出计算过程" in str(row["prompt"]) for row in worked)
    assert all("只给出结论" in str(row["prompt"]) for row in direct)


def test_orb_focus_has_room_for_training_validation_and_final_probes() -> None:
    """充能球数值空间应同时容纳训练、验证和最终评测配额。

    Raises:
        AssertionError: 三套题面发生重复或默认配额无法生成。

    Returns:
        None: 此测试防止退回 human-rl 只有 120 种题面的旧上限。
    """
    training, _ = build_arithmetic_samples(
        attack_values=[],
        quotas={"orb_focus": 100},
        direct_count=0,
    )
    probes, _ = build_arithmetic_samples(
        attack_values=[],
        quotas={"orb_focus": 12},
        direct_count=0,
        seed=20260924,
        exclude_prompts={str(row["prompt"]) for row in training},
        exclude_case_ids={str(row["case_id"]) for row in training},
    )
    validation, _ = build_arithmetic_samples(
        attack_values=[],
        quotas={"orb_focus": 10},
        direct_count=0,
        seed=20261024,
        exclude_prompts={str(row["prompt"]) for row in training + probes},
        exclude_case_ids={str(row["case_id"]) for row in training + probes},
    )

    prompts = [str(row["prompt"]) for row in training + probes + validation]
    case_ids = [str(row["case_id"]) for row in training + probes + validation]
    assert len(prompts) == 122
    assert len(prompts) == len(set(prompts))
    assert len(case_ids) == len(set(case_ids))


def test_generate_arithmetic_candidates_builds_three_independent_splits(
    tmp_path: Path,
) -> None:
    """训练、验证和最终 eval 应使用互不重复的算术问题。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Raises:
        AssertionError: 战斗帧未被读取、三套数字或分卷用途发生重复。

    Returns:
        None: 此测试只验证算术候选的发布边界。
    """
    human_root = tmp_path / "raw/human"
    battle_path = _write_human_run(human_root, "RUN", 11)
    (human_root / "splits.json").write_text(
        json.dumps({"train": ["RUN"], "dev": [], "test": []}),
        encoding="utf-8",
    )
    output = tmp_path / "generated-v0.107.1"

    result = generate_arithmetic_candidates(
        human_root=human_root,
        output_root=output,
        training_quotas={"block_math": 8, "orb_focus": 4},
        training_direct_count=4,
        validation_quotas={"block_math": 3, "orb_focus": 2},
        validation_direct_count=2,
        evaluation_quotas={"block_math": 2, "orb_focus": 1},
        evaluation_direct_count=1,
    )

    training = [
        json.loads(line)
        for path in sorted((output / "arithmetic/train").glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    validation = [
        json.loads(line)
        for path in sorted((output / "arithmetic/validation").glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    evaluation = [
        json.loads(line)
        for path in sorted((output / "arithmetic/eval").glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    train_prompts = {row["prompt"] for row in training}
    validation_prompts = {row["prompt"] for row in validation}
    evaluation_prompts = {row["prompt"] for row in evaluation}
    train_case_ids = {row["case_id"] for row in training}
    validation_case_ids = {row["case_id"] for row in validation}
    evaluation_case_ids = {row["case_id"] for row in evaluation}
    assert result.train_count == len(training) == 16
    assert result.validation_count == len(validation) == 7
    assert result.evaluation_count == len(evaluation) == 4
    assert all(row["question_role"] == "train" for row in training)
    assert all(row["question_role"] == "validation" for row in validation)
    assert all(row["question_role"] == "eval" for row in evaluation)
    assert all(isinstance(row["answer_numbers"], list) for row in training)
    assert all(isinstance(row["answer_numbers"], list) for row in validation)
    assert all(isinstance(row["answer_numbers"], list) for row in evaluation)
    assert train_prompts.isdisjoint(validation_prompts)
    assert train_prompts.isdisjoint(evaluation_prompts)
    assert validation_prompts.isdisjoint(evaluation_prompts)
    assert train_case_ids.isdisjoint(validation_case_ids)
    assert train_case_ids.isdisjoint(evaluation_case_ids)
    assert validation_case_ids.isdisjoint(evaluation_case_ids)
    manifest = json.loads(
        (output / "arithmetic/_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source"] == "data/raw/human/*/combat/*.jsonl"
    assert manifest["training_run_ids"] == ["RUN"]
    assert manifest["combat_sha256"] == {
        "RUN/combat/battle.jsonl": hashlib.sha256(battle_path.read_bytes()).hexdigest()
    }
    assert manifest["attack_values"] == [11]
    assert manifest["evaluation"]["worked"] == 3
    assert manifest["evaluation"]["direct"] == 1


def test_attack_values_only_use_audited_training_roster(tmp_path: Path) -> None:
    """验证与测试局及不合格局不得改变算术攻击值集合。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Raises:
        AssertionError: 非训练局的独有攻击值进入生成输入。

    Returns:
        None: 此测试只检查攻击值来源隔离。
    """
    human_root = tmp_path / "raw/human"
    _write_human_run(human_root, "TRAIN", 11)
    _write_human_run(human_root, "DEV", 31)
    _write_human_run(human_root, "TEST", 35)
    _write_human_run(human_root, "INELIGIBLE", 43, training_eligible=False)
    (human_root / "splits.json").write_text(
        json.dumps({"train": ["TRAIN"], "dev": ["DEV"], "test": ["TEST"]}),
        encoding="utf-8",
    )

    assert load_observed_attack_values(human_root) == [11]


def test_attack_values_reject_unassigned_eligible_run(tmp_path: Path) -> None:
    """未列入任何分卷的合格局必须令算术生成失败。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Raises:
        AssertionError: 未分配合格局被静默忽略或参与生成。

    Returns:
        None: 此测试只检查名册的 fail-closed 边界。
    """
    human_root = tmp_path / "raw/human"
    _write_human_run(human_root, "TRAIN", 11)
    _write_human_run(human_root, "UNASSIGNED", 45)
    (human_root / "splits.json").write_text(
        json.dumps({"train": ["TRAIN"], "dev": [], "test": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="未分配到名册"):
        load_observed_attack_values(human_root)
