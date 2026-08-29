"""运行未见问法知识召回与组合算术生成探针。"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .encoding import SftTrainingError
from .trainer import resolve_device

_INTEGER = re.compile(r"\d+")
_COMPOSITIONAL_BUCKETS = {
    "block_math",
    "energy_math",
    "lethal",
    "multihit",
    "orb_focus",
    "status_math",
}


@dataclass(frozen=True, slots=True)
class KnowledgeProbe:
    """保存一条知识或组合推理探针。

    Args:
        kind (str): ``form_holdout`` 或具体组合算术类别。
        prompt (str): 训练集中未出现的评测问题。
        reference (str): 用于确定数字或核心文本的参考答案。
        category (str | None): 知识实体类别。
        object_id (str | None): 知识实体稳定 ID。
        answer_choice (str | None): 可选的“会死/不会死”等结论。
        answer_numbers (tuple[int, ...]): 构建期确定的最终答案数字。
    """

    kind: str
    prompt: str
    reference: str
    category: str | None = None
    object_id: str | None = None
    answer_choice: str | None = None
    answer_numbers: tuple[int, ...] = ()


def score_recall_answer(answer: str, reference: str) -> bool:
    """以忽略空白的完整参考答案匹配提供保守、无假阳性的评分。

    Args:
        answer (str): 模型生成答案。
        reference (str): 探针参考答案。

    Returns:
        bool: 回答是否完整复现参考事实文本。
    """
    return re.sub(r"\s+", "", answer) == re.sub(r"\s+", "", reference)


def score_compositional_answer(answer: str, probe: Mapping[str, Any]) -> bool:
    """按构建期标出的最终数字与生死结论评分组合题。

    Args:
        answer (str): 模型生成答案。
        probe (Mapping[str, Any]): 含 reference 和可选答案标记的探针。

    Returns:
        bool: 结论与最终数字是否全部正确。
    """
    choice = probe.get("answer_choice")
    if choice == "不会死" and "不会死" not in answer and "能活" not in answer:
        return False
    if choice == "会死" and ("会死" not in answer or "不会死" in answer):
        return False
    expected = probe.get("answer_numbers")
    if expected:
        expected_numbers = tuple(str(value) for value in expected)
        answer_numbers = _integers(answer)
        return tuple(answer_numbers[-len(expected_numbers) :]) == expected_numbers
    if choice:
        return True
    return sorted(_integers(answer)) == sorted(_integers(str(probe["reference"])))


def load_knowledge_probes(
    root: Path,
    *,
    limit: int | None = None,
) -> list[KnowledgeProbe]:
    """从新版 eval 目录树读取知识与算术 SFT 行并保持分层截断。

    Args:
        root (Path): SFT ``eval`` 分卷目录。
        limit (int | None): 可选总题数；按原类别比例保留组合题。

    Raises:
        SftTrainingError: 文件、字段、limit 或题目数量无效。
        OSError: 无法读取探针文件。

    Returns:
        list[KnowledgeProbe]: 保持稳定顺序的全部题目。
    """
    if limit is not None and limit <= 0:
        raise SftTrainingError("知识探针 limit 必须为正数")
    root = Path(root)
    if not root.is_dir():
        raise SftTrainingError(f"SFT eval 目录不存在: {root}")
    recall: list[KnowledgeProbe] = []
    compositional: list[KnowledgeProbe] = []
    for path in sorted(root.rglob("*.jsonl")):
        relative = path.relative_to(root)
        if relative.parts[0] in {"combat", "strategy"}:
            continue
        loaded = _read_sft_probe_file(path)
        if relative.parts[0] == "arithmetic":
            compositional.extend(loaded)
        else:
            recall.extend(loaded)
    probes = recall + compositional
    if not probes:
        raise SftTrainingError(f"知识探针为空: {root}")
    if limit is not None and limit < len(probes):
        compositional_count = max(
            1,
            round(limit * len(compositional) / len(probes)),
        )
        probes = (
            recall[: limit - compositional_count] + compositional[:compositional_count]
        )
    return probes


def run_knowledge_evaluation(
    model_path: Path,
    eval_root: Path,
    output_path: Path,
    *,
    device: str = "auto",
    limit: int | None = None,
    minimum_new_tokens: int = 96,
) -> dict[str, object]:
    """用合并 Hugging Face 模型贪心生成完整知识探针报告。

    Args:
        model_path (Path): 待评测的本地合并模型目录。
        eval_root (Path): 新版 SFT eval 分卷目录。
        output_path (Path): 完整 JSON 报告路径。
        device (str): ``auto``、``mps`` 或 ``cpu``。
        limit (int | None): 可选的分层冒烟题数。
        minimum_new_tokens (int): 短参考答案的最小生成预算。

    Raises:
        SftTrainingError: 参数、模型或探针不符合评测契约。
        OSError: 无法读取模型或写入报告。

    Returns:
        dict[str, object]: 可直接比较轮次的汇总指标。
    """
    if minimum_new_tokens <= 0:
        raise SftTrainingError("minimum_new_tokens 必须为正数")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    probes = load_knowledge_probes(eval_root, limit=limit)
    resolved_device = resolve_device(device)
    dtype = torch.bfloat16 if resolved_device == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.to(resolved_device)
    model.eval()
    results: list[dict[str, object]] = []
    started = time.monotonic()
    for probe in probes:
        messages = [{"role": "user", "content": f"Q: {probe.prompt}\nA:"}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=False,
        ).to(resolved_device)
        reference_tokens = len(
            tokenizer(probe.reference, add_special_tokens=False).input_ids
        )
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max(minimum_new_tokens, reference_tokens + 32),
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        answer = tokenizer.decode(
            generated_ids[0, input_ids.shape[-1] :],
            skip_special_tokens=True,
        ).strip()
        bucket = probe.kind
        probe_mapping = {
            "reference": probe.reference,
            "answer_choice": probe.answer_choice,
            "answer_numbers": probe.answer_numbers,
        }
        passed = (
            score_compositional_answer(answer, probe_mapping)
            if probe.kind in _COMPOSITIONAL_BUCKETS
            else score_recall_answer(answer, probe.reference)
        )
        results.append(
            {
                "bucket": bucket,
                "category": probe.category,
                "object_id": probe.object_id,
                "prompt": probe.prompt,
                "reference": probe.reference,
                "answer": answer,
                "passed": passed,
            }
        )
        if resolved_device == "mps":
            torch.mps.empty_cache()
    summary = _summarize_results(
        results,
        model_path=Path(model_path),
        elapsed_seconds=time.monotonic() - started,
    )
    report = _report_payload(summary, results)
    _write_json(output_path, report)
    return summary


def _read_sft_probe_file(path: Path) -> list[KnowledgeProbe]:
    """把一个 eval SFT JSONL 转换为知识评测题。

    Args:
        path (Path): 新版 eval 树中的知识或算术 JSONL。

    Raises:
        SftTrainingError: 某行字段缺失或 JSON 无效。
        OSError: 文件无法读取。

    Returns:
        list[KnowledgeProbe]: 保持源文件顺序的题目。
    """
    probes: list[KnowledgeProbe] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SftTrainingError(f"无效知识探针: {path}:{line_number}") from exc
            if not isinstance(value, Mapping):
                raise SftTrainingError(f"知识探针行不是对象: {path}:{line_number}")
            messages = value.get("messages")
            if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
                raise SftTrainingError(f"知识评测行缺少 messages: {path}:{line_number}")
            users = [
                message.get("content")
                for message in messages
                if isinstance(message, Mapping) and message.get("role") == "user"
            ]
            answers = [
                message.get("content")
                for message in messages
                if isinstance(message, Mapping) and message.get("role") == "assistant"
            ]
            if (
                len(users) != 1
                or len(answers) != 1
                or not isinstance(users[0], str)
                or not isinstance(answers[0], str)
            ):
                raise SftTrainingError(f"知识评测消息无效: {path}:{line_number}")
            category = _optional_string(value.get("category"))
            object_id = _optional_string(value.get("object_id"))
            if category is None or object_id is None:
                raise SftTrainingError(f"知识评测身份无效: {path}:{line_number}")
            prompt = re.sub(r"^\s*Q:\s*", "", users[0])
            prompt = re.sub(r"\s*A:\s*$", "", prompt).strip()
            reference = answers[0].strip()
            kind = object_id if category == "arithmetic" else "form_holdout"
            answer_numbers = value.get("answer_numbers", ())
            if not isinstance(answer_numbers, Sequence) or isinstance(
                answer_numbers, (str, bytes)
            ):
                raise SftTrainingError(f"知识探针答案数字无效: {path}:{line_number}")
            probes.append(
                KnowledgeProbe(
                    kind=kind,
                    prompt=prompt,
                    reference=reference,
                    category=category,
                    object_id=object_id,
                    answer_choice=_optional_string(value.get("answer_choice")),
                    answer_numbers=tuple(int(item) for item in answer_numbers),
                )
            )
    return probes


def _report_payload(
    summary: Mapping[str, object],
    results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """构造包含全部逐题证据的可审计评测报告。

    Args:
        summary (Mapping[str, object]): 聚合指标。
        results (Sequence[Mapping[str, object]]): 所有探针的逐题结果。

    Returns:
        dict[str, object]: 汇总与完整结果列表。
    """
    return {
        "summary": dict(summary),
        "results": [dict(row) for row in results],
    }


def _summarize_results(
    results: Sequence[Mapping[str, object]],
    *,
    model_path: Path,
    elapsed_seconds: float,
) -> dict[str, object]:
    """汇总总体、分桶和知识类别准确率。

    Args:
        results (Sequence[Mapping[str, object]]): 全部逐题结果。
        model_path (Path): 用于来源追溯的模型目录。
        elapsed_seconds (float): 完整生成耗时。

    Returns:
        dict[str, object]: 可用于比较不同训练轮次的稳定指标。
    """
    summary: dict[str, object] = {
        "model": str(model_path),
        "questions": len(results),
        "elapsed_seconds": elapsed_seconds,
    }
    for bucket in ("form_holdout", *_COMPOSITIONAL_BUCKETS):
        subset = [row for row in results if row["bucket"] == bucket]
        if subset:
            summary[bucket] = {
                "questions": len(subset),
                "accuracy": sum(row["passed"] is True for row in subset) / len(subset),
            }
    compositional = [row for row in results if row["bucket"] in _COMPOSITIONAL_BUCKETS]
    if compositional:
        summary["compositional_total"] = {
            "questions": len(compositional),
            "accuracy": sum(row["passed"] is True for row in compositional)
            / len(compositional),
        }
    categories = {
        str(row["category"])
        for row in results
        if row["bucket"] == "form_holdout" and row.get("category") is not None
    }
    summary["form_holdout_by_category"] = {
        category: sum(
            row["passed"] is True
            for row in results
            if row["bucket"] == "form_holdout" and row["category"] == category
        )
        / sum(
            row["bucket"] == "form_holdout" and row["category"] == category
            for row in results
        )
        for category in sorted(categories)
    }
    return summary


def _integers(text: str) -> list[str]:
    """返回文本中按出现顺序提取的十进制数字字符串。

    Args:
        text (str): 任意模型答案或参考答案。

    Returns:
        list[str]: 保留原始数字值的字符串数组。
    """
    return _INTEGER.findall(text or "")


def _optional_string(value: object) -> str | None:
    """把可选 JSON 字段限制为字符串或 ``None``。

    Args:
        value (object): JSON 解码后的任意字段。

    Raises:
        SftTrainingError: 非空字段不是字符串。

    Returns:
        str | None: 规范化后的可选字符串。
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise SftTrainingError("知识探针可选字段不是字符串")
    return value


def _write_json(path: Path, value: object) -> None:
    """以统一缩进写入 UTF-8 JSON 报告。

    Args:
        path (Path): 目标报告路径。
        value (object): 可 JSON 序列化的报告。

    Returns:
        None: 报告完整写入后返回。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
