"""从当前项目的实战意图确定性生成训练与验证算术候选。"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..recording.audit import HumanRunAudit, audit_human_run

ARITHMETIC_SEED = 20260824
TRAINING_QUOTAS = {
    "block_math": 550,
    "lethal": 400,
    "status_math": 500,
    "multihit": 250,
    "energy_math": 200,
    "orb_focus": 100,
}
VALIDATION_QUOTAS = {
    "block_math": 55,
    "lethal": 40,
    "status_math": 50,
    "multihit": 25,
    "energy_math": 20,
    "orb_focus": 10,
}
EVALUATION_QUOTAS = {
    "block_math": 55,
    "lethal": 40,
    "status_math": 50,
    "multihit": 25,
    "energy_math": 20,
    "orb_focus": 10,
}
_VALIDATION_SEED_OFFSET = 200
_EVALUATION_SEED_OFFSET = 400

ArithmeticRow = dict[str, Any]
ArithmeticCase = tuple[str, str]
ArithmeticGenerator = Callable[
    [random.Random, Sequence[int]],
    ArithmeticCase | None,
]


@dataclass(frozen=True, slots=True)
class ArithmeticBuildResult:
    """描述一次算术候选生成结果。

    Args:
        output_root (Path): 包含 ``arithmetic`` 子目录的知识候选根目录。
        train_count (int): 训练用途候选数量。
        validation_count (int): 验证用途候选数量。
        evaluation_count (int): 最终评测用途候选数量。
        buckets (dict[str, int]): 各算术类别的总候选数量。
    """

    output_root: Path
    train_count: int
    validation_count: int
    evaluation_count: int
    buckets: dict[str, int]


@dataclass(frozen=True, slots=True)
class ArithmeticAttackInputs:
    """保存经过名册与 raw 审计的算术来源。

    Args:
        values (tuple[int, ...]): 训练局中出现的单段攻击值。
        run_ids (tuple[str, ...]): 实际贡献输入的训练局 ID。
        combat_files (dict[str, Path]): 稳定相对名到战斗分片路径的映射。
    """

    values: tuple[int, ...]
    run_ids: tuple[str, ...]
    combat_files: dict[str, Path]


def load_observed_attack_values(human_root: Path) -> list[int]:
    """从训练名册内已审计的人类战斗状态读取单段攻击意图数值。

    Args:
        human_root (Path): ``data/raw/human`` 风格的按局目录。

    Raises:
        TypeError: JSONL 行不是对象。
        ValueError: 名册、JSONL 无效，存在未分配合格局或没有可用攻击意图。
        RawRunIntegrityError: 训练局 raw 未完整发布或物理计数不一致。
        OSError: 战斗分片无法读取。

    Returns:
        list[int]: 去重并升序排列的正整数攻击值。
    """
    return list(_load_observed_attack_inputs(human_root).values)


def _load_observed_attack_inputs(human_root: Path) -> ArithmeticAttackInputs:
    """取得训练名册内的攻击值、局 ID 与输入文件。

    Args:
        human_root (Path): ``data/raw/human`` 风格的按局目录。

    Raises:
        TypeError: JSONL 行不是对象。
        ValueError: 名册、JSONL 或训练准入无效，或没有可用攻击意图。
        RawRunIntegrityError: 任一已发布 raw 未通过物理完整性审计。
        OSError: 输入文件无法读取。

    Returns:
        ArithmeticAttackInputs: 只含训练分卷来源的稳定输入集合。
    """
    root = Path(human_root)
    splits = _read_human_splits(root / "splits.json")
    declared = set().union(*splits.values())
    audits: dict[str, tuple[Path, HumanRunAudit]] = {}
    for metadata_path in sorted(root.glob("*/meta.json")):
        run_dir = metadata_path.parent
        audit = audit_human_run(run_dir)
        run_id = audit.metadata.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"人类局缺少 run_id: {run_dir}")
        if run_id in audits:
            raise ValueError(f"人类局 run_id 重复: {run_id}")
        audits[run_id] = (run_dir, audit)
        if audit.metadata["training_eligible"] and run_id not in declared:
            raise ValueError(f"可训练人类局未分配到名册: {run_id}")

    combat_files: dict[str, Path] = {}
    for run_id in sorted(splits["train"]):
        if run_id not in audits:
            raise ValueError(f"训练名册中的人类局不存在: {run_id}")
        run_dir, audit = audits[run_id]
        if not audit.metadata["training_eligible"]:
            raise ValueError(f"训练名册包含不合格人类局: {run_id}")
        for path in audit.battle_paths:
            relative = path.relative_to(run_dir).as_posix()
            combat_files[f"{run_id}/{relative}"] = path

    values: set[int] = set()
    for path in combat_files.values():
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"无效战斗 JSONL: {path}:{line_number}") from exc
                if not isinstance(row, Mapping):
                    raise TypeError(f"战斗 JSONL 行不是对象: {path}:{line_number}")
                state = row.get("before_state")
                combat = state.get("combat") if isinstance(state, Mapping) else None
                enemies = combat.get("enemies") if isinstance(combat, Mapping) else None
                if not isinstance(enemies, list):
                    continue
                for enemy in enemies:
                    intents = (
                        enemy.get("intents") if isinstance(enemy, Mapping) else None
                    )
                    if not isinstance(intents, list):
                        continue
                    for intent in intents:
                        if not isinstance(intent, Mapping):
                            continue
                        damage = intent.get("damage")
                        hits = intent.get("hits")
                        if (
                            intent.get("intent_type") == "Attack"
                            and hits in (None, 1)
                            and isinstance(damage, int)
                            and not isinstance(damage, bool)
                            and damage > 0
                        ):
                            values.add(damage)
    if not values:
        raise ValueError(f"人类战斗帧没有可用的单段攻击意图: {human_root}")
    return ArithmeticAttackInputs(
        values=tuple(sorted(values)),
        run_ids=tuple(sorted(splits["train"])),
        combat_files=combat_files,
    )


def _read_human_splits(path: Path) -> dict[str, set[str]]:
    """读取并校验人类整局 train/dev/test 名册。

    Args:
        path (Path): ``data/raw/human/splits.json`` 路径。

    Raises:
        ValueError: 文件缺失、JSON 无效、字段类型错误或分卷重叠。
        OSError: 文件无法读取。

    Returns:
        dict[str, set[str]]: 三个互斥分卷的局 ID 集合。
    """
    if not Path(path).is_file():
        raise ValueError(f"人类整局名册不存在: {path}")
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"无效人类整局名册: {path}") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"人类整局名册顶层不是对象: {path}")
    result: dict[str, set[str]] = {}
    owners: dict[str, str] = {}
    for split in ("train", "dev", "test"):
        rows = value.get(split)
        if not isinstance(rows, list) or not all(
            isinstance(run_id, str) and run_id.strip() for run_id in rows
        ):
            raise ValueError(f"人类整局名册 {split} 必须是非空字符串数组: {path}")
        result[split] = set(rows)
        for run_id in result[split]:
            previous = owners.get(run_id)
            if previous is not None:
                raise ValueError(f"同一人类局出现在多个分卷: {run_id}")
            owners[run_id] = split
    return result


def build_arithmetic_samples(
    *,
    attack_values: Sequence[int],
    quotas: Mapping[str, int] | None = None,
    direct_count: int = 800,
    seed: int = ARITHMETIC_SEED,
    exclude_prompts: Collection[str] = (),
    exclude_case_ids: Collection[str] = (),
) -> tuple[list[ArithmeticRow], list[ArithmeticRow]]:
    """生成题面和数值案例都互不重复的计算过程与直答候选。

    Args:
        attack_values (Sequence[int]): 从真实战斗意图取得的单段攻击值。
        quotas (Mapping[str, int] | None): 各类别计算过程样本数量。
        direct_count (int): 另行生成的直答样本总数。
        seed (int): 确定性随机种子。
        exclude_prompts (Collection[str]): 不得复用的训练、验证或考试题面。
        exclude_case_ids (Collection[str]): 不得复用的题型与运算案例标识。

    Raises:
        ValueError: 配额、攻击值或类别无效。
        RuntimeError: 在尝试上限内无法生成足够的唯一问题。

    Returns:
        tuple[list[ArithmeticRow], list[ArithmeticRow]]: 计算过程与直答候选。
    """
    resolved_quotas = dict(quotas or TRAINING_QUOTAS)
    unknown = set(resolved_quotas) - set(_GENERATORS)
    if unknown:
        raise ValueError(f"未知算术类别: {sorted(unknown)}")
    if any(
        not isinstance(count, int) or count < 0 for count in resolved_quotas.values()
    ):
        raise ValueError("算术配额必须是非负整数")
    if not isinstance(direct_count, int) or direct_count < 0:
        raise ValueError("直答样本数必须是非负整数")
    total_worked = sum(resolved_quotas.values())
    if total_worked == 0:
        if direct_count:
            raise ValueError("计算过程配额为零时不能分配直答样本")
        return [], []
    needs_attack_values = any(
        resolved_quotas.get(bucket, 0) > 0
        for bucket in ("block_math", "lethal", "multihit")
    )
    normalized_values = sorted(
        {
            value
            for value in attack_values
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        }
    )
    if needs_attack_values and not normalized_values:
        raise ValueError("格挡、致死和多段算术需要至少一个真实攻击值")

    seen_prompts = {_prompt_key(prompt) for prompt in exclude_prompts}
    seen_case_ids = {str(case_id) for case_id in exclude_case_ids}
    worked_rng = random.Random(seed)
    worked: list[ArithmeticRow] = []
    for bucket, quota in resolved_quotas.items():
        worked.extend(
            _fill_samples(
                bucket,
                quota,
                _GENERATORS[bucket],
                worked_rng,
                normalized_values,
                suffix="请写出计算过程。",
                seen_prompts=seen_prompts,
                seen_case_ids=seen_case_ids,
            )
        )

    direct_quotas = _scale_quotas(resolved_quotas, direct_count)
    direct_rng = random.Random(seed + 1)
    direct: list[ArithmeticRow] = []
    for bucket, quota in direct_quotas.items():
        generated = _fill_samples(
            bucket,
            quota,
            _GENERATORS[bucket],
            direct_rng,
            normalized_values,
            suffix="只给出结论。",
            seen_prompts=seen_prompts,
            seen_case_ids=seen_case_ids,
        )
        direct.extend(
            {**row, "completion": " " + _last_sentence(str(row["completion"]))}
            for row in generated
        )
    return worked, direct


def generate_arithmetic_candidates(
    *,
    human_root: Path,
    output_root: Path,
    training_quotas: Mapping[str, int] | None = None,
    training_direct_count: int = 800,
    validation_quotas: Mapping[str, int] | None = None,
    validation_direct_count: int = 80,
    evaluation_quotas: Mapping[str, int] | None = None,
    evaluation_direct_count: int = 80,
    seed: int = ARITHMETIC_SEED,
) -> ArithmeticBuildResult:
    """发布题面和运算案例都互斥的训练、验证与最终评测候选。

    三种用途使用不同随机种子。默认数量沿用 2,000/800 训练配方，并额外各生成
    200/80 条独立验证与最终评测候选。

    Args:
        human_root (Path): 当前项目的人类精确战斗目录。
        output_root (Path): ``generated-v0.107.1`` 风格的候选根目录。
        training_quotas (Mapping[str, int] | None): 训练计算过程配额。
        training_direct_count (int): 训练直答候选数量。
        validation_quotas (Mapping[str, int] | None): 验证计算过程配额。
        validation_direct_count (int): 验证直答候选数量。
        evaluation_quotas (Mapping[str, int] | None): 最终评测计算过程配额。
        evaluation_direct_count (int): 最终评测直答候选数量。
        seed (int): 训练候选随机种子。

    Raises:
        ValueError: 战斗帧或配额无效。
        OSError: 输入无法读取或输出无法写入。

    Returns:
        ArithmeticBuildResult: 输出目录、三种用途数量和类别计数。
    """
    attack_inputs = _load_observed_attack_inputs(human_root)
    attack_values = list(attack_inputs.values)
    resolved_training_quotas = dict(training_quotas or TRAINING_QUOTAS)
    resolved_validation_quotas = dict(validation_quotas or VALIDATION_QUOTAS)
    resolved_evaluation_quotas = dict(evaluation_quotas or EVALUATION_QUOTAS)
    train_worked, train_direct = build_arithmetic_samples(
        attack_values=attack_values,
        quotas=resolved_training_quotas,
        direct_count=training_direct_count,
        seed=seed,
    )
    training = train_worked + train_direct
    validation_exclusions = {str(row["prompt"]) for row in training}
    validation_case_exclusions = {str(row["case_id"]) for row in training}
    validation_worked, validation_direct = build_arithmetic_samples(
        attack_values=attack_values,
        quotas=resolved_validation_quotas,
        direct_count=validation_direct_count,
        seed=seed + _VALIDATION_SEED_OFFSET,
        exclude_prompts=validation_exclusions,
        exclude_case_ids=validation_case_exclusions,
    )
    validation = validation_worked + validation_direct
    evaluation_exclusions = validation_exclusions | {
        str(row["prompt"]) for row in validation
    }
    evaluation_case_exclusions = validation_case_exclusions | {
        str(row["case_id"]) for row in validation
    }
    evaluation_worked, evaluation_direct = build_arithmetic_samples(
        attack_values=attack_values,
        quotas=resolved_evaluation_quotas,
        direct_count=evaluation_direct_count,
        seed=seed + _EVALUATION_SEED_OFFSET,
        exclude_prompts=evaluation_exclusions,
        exclude_case_ids=evaluation_case_exclusions,
    )
    evaluation = evaluation_worked + evaluation_direct

    destination = Path(output_root)
    arithmetic_root = destination / "arithmetic"
    _write_rows(arithmetic_root / "train/cot.jsonl", train_worked, role="train")
    _write_rows(
        arithmetic_root / "train/direct.jsonl",
        train_direct,
        role="train",
    )
    _write_rows(
        arithmetic_root / "validation/cot.jsonl",
        validation_worked,
        role="validation",
    )
    _write_rows(
        arithmetic_root / "validation/direct.jsonl",
        validation_direct,
        role="validation",
    )
    _write_rows(
        arithmetic_root / "eval/cot.jsonl",
        evaluation_worked,
        role="eval",
    )
    _write_rows(
        arithmetic_root / "eval/direct.jsonl",
        evaluation_direct,
        role="eval",
    )
    buckets = Counter(
        str(row["object_id"]) for row in training + validation + evaluation
    )
    manifest = {
        "format": "prompt_completion_candidates",
        "source": "data/raw/human/*/combat/*.jsonl",
        "human_root": str(human_root),
        "training_run_ids": list(attack_inputs.run_ids),
        "combat_sha256": {
            name: _sha256(path)
            for name, path in sorted(attack_inputs.combat_files.items())
        },
        "seed": seed,
        "validation_seed": seed + _VALIDATION_SEED_OFFSET,
        "evaluation_seed": seed + _EVALUATION_SEED_OFFSET,
        "attack_values": attack_values,
        "training": {
            "worked": len(train_worked),
            "direct": len(train_direct),
            "quotas": resolved_training_quotas,
        },
        "validation": {
            "worked": len(validation_worked),
            "direct": len(validation_direct),
            "quotas": resolved_validation_quotas,
        },
        "evaluation": {
            "worked": len(evaluation_worked),
            "direct": len(evaluation_direct),
            "quotas": resolved_evaluation_quotas,
        },
        "buckets": dict(sorted(buckets.items())),
    }
    arithmetic_root.mkdir(parents=True, exist_ok=True)
    (arithmetic_root / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ArithmeticBuildResult(
        output_root=destination,
        train_count=len(training),
        validation_count=len(validation),
        evaluation_count=len(evaluation),
        buckets=dict(sorted(buckets.items())),
    )


def _sha256(path: Path) -> str:
    """流式计算算术来源文件的 SHA-256。

    Args:
        path (Path): 待摘要的战斗 JSONL。

    Returns:
        str: 小写十六进制摘要。
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fill_samples(
    bucket: str,
    quota: int,
    generator: ArithmeticGenerator,
    rng: random.Random,
    attack_values: Sequence[int],
    *,
    suffix: str,
    seen_prompts: set[str],
    seen_case_ids: set[str],
) -> list[ArithmeticRow]:
    """持续生成某类别，直到达到唯一题面和运算案例配额。

    Args:
        bucket (str): 算术类别。
        quota (int): 目标样本数。
        generator (ArithmeticGenerator): 单题生成函数。
        rng (random.Random): 当前分卷的确定性随机源。
        attack_values (Sequence[int]): 真实单段攻击值。
        suffix (str): 回答风格要求。
        seen_prompts (set[str]): 全局已占用或禁止的问题键。
        seen_case_ids (set[str]): 全局已占用或禁止的运算案例标识。

    Raises:
        RuntimeError: 在尝试上限内无法产生足够唯一问题。

    Returns:
        list[ArithmeticRow]: 达到目标数量的唯一候选。
    """
    output: list[ArithmeticRow] = []
    attempts = 0
    while len(output) < quota:
        attempts += 1
        if attempts > max(1000, quota * 1000):
            raise RuntimeError(f"{bucket} 无法生成 {quota} 条唯一问题")
        generated = generator(rng, attack_values)
        if generated is None:
            continue
        prompt, completion = generated
        prompt = prompt.rstrip("。？") + f"。{suffix}"
        full_prompt = _prompt_key(prompt)
        case_id = _case_id(bucket, completion)
        if full_prompt in seen_prompts or case_id in seen_case_ids:
            continue
        seen_prompts.add(full_prompt)
        seen_case_ids.add(case_id)
        output.append(_make_row(bucket, prompt, completion, case_id=case_id))
    return output


def _make_row(
    bucket: str,
    prompt: str,
    completion: str,
    *,
    case_id: str,
) -> ArithmeticRow:
    """把一道算术题包装成知识候选行。

    Args:
        bucket (str): 算术类别。
        prompt (str): 不含问答包装的问题。
        completion (str): 标准答案。
        case_id (str): 与自然语言问法无关的运算案例标识。

    Returns:
        ArithmeticRow: 与其他知识 JSONL 相同的字段结构。
    """
    conclusion = _last_sentence(completion)
    row: ArithmeticRow = {
        "category": "arithmetic",
        "object_id": bucket,
        "source": "synthetic_arithmetic",
        "case_id": case_id,
        "prompt": f"Q: {prompt}\nA:",
        "completion": " " + completion.strip(),
        "answer_numbers": [int(value) for value in re.findall(r"\d+", conclusion)],
    }
    if "不会死" in conclusion:
        row["answer_choice"] = "不会死"
    elif "会死" in conclusion:
        row["answer_choice"] = "会死"
    return row


def _case_id(bucket: str, completion: str) -> str:
    """由题型和完整计算语义构造与问法无关的案例标识。

    Args:
        bucket (str): 算术类别。
        completion (str): 尚未裁成直答的完整计算过程。

    Returns:
        str: 可读、稳定且保留全部操作数和运算顺序的案例标识。
    """
    normalized = re.sub(r"\s+", "", completion).strip("。")
    return f"{bucket}/{normalized}"


def _scale_quotas(quotas: Mapping[str, int], total: int) -> dict[str, int]:
    """按计算过程类别比例分配直答样本数量。

    Args:
        quotas (Mapping[str, int]): 计算过程类别配额。
        total (int): 直答样本总数。

    Returns:
        dict[str, int]: 顺序稳定且总和精确等于 ``total`` 的类别配额。
    """
    worked_total = sum(quotas.values())
    if total == 0:
        return {bucket: 0 for bucket in quotas}
    scaled = {
        bucket: round(quota * total / worked_total) for bucket, quota in quotas.items()
    }
    difference = total - sum(scaled.values())
    if difference:
        first = next(iter(scaled))
        scaled[first] += difference
    return scaled


def _prompt_key(prompt: str) -> str:
    """把三种用途的问题统一成带问答包装的精确比较键。

    Args:
        prompt (str): 可带 ``Q:``/``A:`` 包装的问题。

    Returns:
        str: 空白压缩后的稳定问题键。
    """
    value = re.sub(r"^\s*Q:\s*", "", str(prompt))
    value = re.sub(r"\s*A:\s*$", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return f"Q: {value}\nA:"


def _write_rows(path: Path, rows: Sequence[ArithmeticRow], *, role: str) -> None:
    """写入带明确训练、验证或评测用途的算术候选。

    Args:
        path (Path): JSONL 输出路径。
        rows (Sequence[ArithmeticRow]): 尚未附加分卷用途的候选。
        role (str): ``train``、``validation`` 或 ``eval``。

    Returns:
        None: 文件完整写入后返回。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    **row,
                    "fact_id": (
                        f"arithmetic/{row['object_id']}/{role}-{path.stem}-{index:04d}"
                    ),
                    "question_role": role,
                },
                ensure_ascii=False,
            )
            + "\n"
            for index, row in enumerate(rows, start=1)
        ),
        encoding="utf-8",
    )


def _last_sentence(completion: str) -> str:
    """从计算过程答案中取包含最终结论的最后一句。

    Args:
        completion (str): 完整计算过程答案。

    Returns:
        str: 以中文句号结尾的单句结论。
    """
    text = completion.strip().rstrip("。")
    parts = [part.strip() for part in re.split(r"[。；]", text) if part.strip()]
    return parts[-1] + "。"


def _block_math(
    rng: random.Random,
    attack_values: Sequence[int],
) -> ArithmeticCase:
    """生成格挡优先抵消伤害后的生命计算题。

    Args:
        rng (random.Random): 确定性随机源。
        attack_values (Sequence[int]): 真实单段攻击值。

    Returns:
        ArithmeticCase: 问题与完整计算过程。
    """
    attack = rng.choice(attack_values)
    block = rng.randint(max(0, attack - 8), attack + 8)
    hp = rng.randint(15, 80)
    loss = max(attack - block, 0)
    ask = rng.choice(
        [
            f"敌人意图'攻击'(攻击{attack})。你格挡{block}、HP{hp}。结束回合后HP是多少？",
            f"敌人这一击造成{attack}点伤害，你有{block}点格挡、当前HP{hp}。挨完这一下HP剩多少？",
            f"意图攻击{attack}，我的格挡{block}，HP{hp}，结束回合我会剩多少HP？",
        ]
    )
    if loss:
        steps = (
            f"格挡结算：{attack}−{block}，透过伤害 max(0, {attack}−{block})={loss}。"
            f"HP {hp}−{loss}={hp - loss}。所以HP是{hp - loss}。"
        )
    else:
        steps = (
            f"格挡结算：攻击{attack}≤格挡{block}，伤害全部被挡下"
            f"（max(0, {attack}−{block})=0，多出的格挡不转化）。HP不变，仍是{hp}。"
        )
    return ask, steps


def _lethal(
    rng: random.Random,
    attack_values: Sequence[int],
) -> ArithmeticCase:
    """生成总伤、格挡与当前生命的致死判断题。

    Args:
        rng (random.Random): 确定性随机源。
        attack_values (Sequence[int]): 真实单段攻击值。

    Returns:
        ArithmeticCase: 问题与完整计算过程。
    """
    attack = rng.choice(attack_values)
    hits = rng.choice([1, 1, 1, 2, 3])
    total = attack * hits
    block = rng.randint(0, max(1, total // 2))
    hp = rng.choice(
        [
            max(1, total - block - rng.randint(0, 6)),
            total - block + rng.randint(1, 10),
        ]
    )
    loss = max(total - block, 0)
    description = f"(攻击{attack}×{hits}次)" if hits > 1 else f"(攻击{attack})"
    ask = rng.choice(
        [
            f"你HP{hp}、格挡{block}，敌人意图'当前攻击意图'{description}。结束回合会死吗？",
            f"HP{hp}、格挡{block}，对方这一轮总共要打{total}点。结束回合我能活吗？",
        ]
    )
    step = (
        f"总伤 {attack}"
        + (f"×{hits}" if hits > 1 else "")
        + f"={total}；透过 max(0, {total}−{block})={loss}。"
    )
    if loss >= hp:
        return ask, f"{step}透过{loss}≥HP{hp}：会死。"
    return ask, f"{step}透过{loss}<HP{hp}：不会死，剩{hp - loss}。"


def _status_math(
    rng: random.Random,
    attack_values: Sequence[int],
) -> ArithmeticCase:
    """生成虚弱、易伤、脆弱与力量的整除算术题。

    Args:
        rng (random.Random): 确定性随机源。
        attack_values (Sequence[int]): 为统一生成函数签名保留的实战攻击值。

    Returns:
        ArithmeticCase: 问题与完整计算过程。
    """
    del attack_values
    mode = rng.choice(
        [
            "weak",
            "vulnerable_attack",
            "vulnerable_defend",
            "frail",
            "strength",
            "weak_vulnerable",
            "strength_multi",
        ]
    )
    if mode == "weak":
        base = 4 * rng.randint(1, 15)
        actual = base * 3 // 4
        return (
            f"你带着虚弱(造成的攻击伤害-25%)，打出一张造成{base}点伤害的攻击牌。实际造成多少伤害？",
            f"虚弱结算：{base}×(1−25%)={base}×3÷4={actual}。实际造成{actual}点。",
        )
    if mode == "vulnerable_attack":
        base = 2 * rng.randint(2, 25)
        actual = base * 3 // 2
        return (
            f"敌人处于易伤(受到的攻击伤害+50%)，你的攻击牌造成{base}点伤害。它会受到多少伤害？",
            f"易伤结算：{base}×(1+50%)={base}×3÷2={actual}。敌人受到{actual}点。",
        )
    if mode == "vulnerable_defend":
        base = 2 * rng.randint(2, 30)
        vulnerable = base * 3 // 2
        block = rng.randint(0, vulnerable - 1)
        loss = vulnerable - block
        return (
            f"你易伤(受到攻击伤害+50%)且格挡{block}。敌人攻击{base}点，结束回合你掉多少HP？",
            f"易伤先算：{base}×3÷2={vulnerable}。再过格挡：max(0, {vulnerable}−{block})={loss}。掉{loss}点HP。",
        )
    if mode == "frail":
        base = 4 * rng.randint(1, 15)
        actual = base * 3 // 4
        return (
            f"你带着脆弱(从卡牌获得的格挡-25%)，打出一张获得{base}点格挡的牌。实际获得多少格挡？",
            f"脆弱结算：{base}×(1−25%)={base}×3÷4={actual}。实际获得{actual}点格挡。",
        )
    if mode == "strength":
        strength = rng.randint(1, 10)
        base = rng.randint(3, 30)
        return (
            f"你有{strength}层力量(攻击牌伤害每次+{strength})，打出造成{base}点伤害的攻击牌。这一下打多少？",
            f"力量加成：{base}+{strength}={base + strength}。这一下造成{base + strength}点。",
        )
    if mode == "weak_vulnerable":
        base = 8 * rng.randint(1, 10)
        weakened = base * 3 // 4
        actual = weakened * 3 // 2
        return (
            f"你虚弱(攻击伤害-25%)且敌人易伤(受到攻击伤害+50%)。你打出造成{base}点伤害的攻击牌，敌人最终受多少伤害？",
            f"先虚己：{base}×3÷4={weakened}；再易敌：{weakened}×3÷2={actual}。敌人最终受{actual}点。",
        )
    strength = rng.randint(1, 8)
    base = rng.randint(2, 20)
    hits = rng.randint(2, 5)
    per_hit = base + strength
    return (
        f"你有{strength}层力量，打出'{base}点伤害{hits}次'的攻击牌。总共造成多少伤害？",
        f"每段 {base}+{strength}={per_hit}（力量逐段加成）；总伤 {per_hit}×{hits}={per_hit * hits}。",
    )


def _multihit(
    rng: random.Random,
    attack_values: Sequence[int],
) -> ArithmeticCase:
    """生成多段伤害合计后再过格挡的生命计算题。

    Args:
        rng (random.Random): 确定性随机源。
        attack_values (Sequence[int]): 真实单段攻击值。

    Returns:
        ArithmeticCase: 问题与完整计算过程。
    """
    attack = rng.choice(attack_values)
    hits = rng.randint(2, 4)
    total = attack * hits
    block = rng.randint(0, total)
    loss = max(total - block, 0)
    hp = rng.randint(20, 80)
    ask = rng.choice(
        [
            f"敌人意图'{attack}点伤害{hits}次'。你格挡{block}、HP{hp}。结束回合后HP是多少？",
            f"对面这一轮打{hits}段、每段{attack}点。我格挡{block}、HP{hp}，结束回合剩多少HP？",
        ]
    )
    if loss >= hp:
        return (
            ask,
            f"总伤 {attack}×{hits}={total}；透过 max(0, {total}−{block})={loss}≥HP{hp}：会死。",
        )
    return (
        ask,
        f"总伤 {attack}×{hits}={total}；透过 max(0, {total}−{block})={loss}；HP {hp}−{loss}={hp - loss}。",
    )


def _energy_math(
    rng: random.Random,
    attack_values: Sequence[int],
) -> ArithmeticCase | None:
    """生成连续打牌后的能量记账题。

    Args:
        rng (random.Random): 确定性随机源。
        attack_values (Sequence[int]): 为统一生成函数签名保留的实战攻击值。

    Returns:
        ArithmeticCase | None: 合法费用组合，或费用超限时返回 ``None`` 重试。
    """
    del attack_values
    energy = rng.randint(2, 5)
    costs = [rng.choice([0, 1, 1, 1, 2, 2, 3]) for _ in range(rng.randint(2, 4))]
    spent = sum(costs)
    if spent > energy:
        return None
    left = energy - spent
    costs_text = "、".join(f"{cost}费" for cost in costs)
    ask = rng.choice(
        [
            f"本回合你有{energy}点能量，依次打出{costs_text}的牌。还剩多少能量？",
            f"能量{energy}，已打出{len(costs)}张牌（{costs_text}）。还能剩几点能量？",
        ]
    )
    return (
        ask,
        f"能量记账：{energy}−({'+'.join(str(cost) for cost in costs)})={energy}−{spent}={left}。剩{left}点能量。",
    )


def _orb_focus(
    rng: random.Random,
    attack_values: Sequence[int],
) -> ArithmeticCase:
    """生成闪电、冰霜和玻璃充能球的集中加成题。

    Args:
        rng (random.Random): 确定性随机源。
        attack_values (Sequence[int]): 为统一生成函数签名保留的实战攻击值。

    Returns:
        ArithmeticCase: 问题与完整计算过程。
    """
    del attack_values
    orbs = [
        ("闪电", 3, 8, 1, 1, "点伤害"),
        ("冰霜", 2, 5, 1, 1, "点格挡"),
        ("玻璃", 4, 8, 1, 2, "点伤害"),
    ]
    name, passive, evoke, passive_coefficient, evoke_coefficient, unit = rng.choice(
        orbs
    )
    passive_mode = rng.choice([True, False])
    if passive_mode:
        base = passive
        coefficient = passive_coefficient
        verb = "被动"
        timing = "回合末被动"
    else:
        base = evoke
        coefficient = evoke_coefficient
        verb = "激发"
        timing = "激发时"
    focus = rng.randint(1, 5)
    orb_count = rng.randint(1, 6)
    gain = focus * coefficient
    per_orb = base + gain
    total = per_orb * orb_count
    ask = (
        f"你有{orb_count}个{name}充能球，{verb}基值都是{base}，集中为{focus}。"
        f"它们在{timing}总共造成或获得多少？"
    )
    equation = (
        f"{base}+({focus}×{coefficient})={base}+{gain}={per_orb}"
        if coefficient != 1
        else f"{base}+{focus}={per_orb}"
    )
    steps = (
        f"集中{focus}加成后每球为{equation}；{orb_count}个球合计"
        f"{per_orb}×{orb_count}={total}。{verb}总计为{total}{unit}。"
    )
    return ask, steps


_GENERATORS: dict[str, ArithmeticGenerator] = {
    "block_math": _block_math,
    "lethal": _lethal,
    "status_math": _status_math,
    "multihit": _multihit,
    "energy_math": _energy_math,
    "orb_focus": _orb_focus,
}
