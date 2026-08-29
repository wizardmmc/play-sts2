"""提供 SFT teacher-forced 指标与可复查的生成式评测。"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import dataset_split_path, validate_sft_dataset
from .encoding import (
    IGNORE_LABEL,
    SftConfig,
    SftTrainingError,
    TokenizedSample,
    load_tokenized_samples,
    normalize_messages,
)
from .trainer import resolve_device

_ACTION_LINE = re.compile(r"ACTION: [a-z_]+(?: \d+){0,2}")


@dataclass(frozen=True, slots=True)
class MaskedTokenStats:
    """保存一批 assistant token 的 teacher-forced 统计。

    Args:
        loss_sum (float): 所有监督 token 的交叉熵总和。
        correct (int): 贪心 token 预测正确的数量。
        tokens (int): 参与指标计算的监督 token 数。
    """

    loss_sum: float
    correct: int
    tokens: int


@dataclass(frozen=True, slots=True)
class GenerationScore:
    """描述一条生成结果的最小离线评分。

    Args:
        exact_match (bool): 去除首尾空白后是否与目标完全相同。
        action_shape_valid (bool | None): 人类行为是否为单行 ``ACTION:`` 外形；
            知识问答没有此指标。
        action_legal (bool | None): 人类行为是否落在样本记录的精确合法动作集合；
            知识问答没有此指标。
        action_failure (str | None): 空输出、格式、动作名或参数失败类别。
    """

    exact_match: bool
    action_shape_valid: bool | None
    action_legal: bool | None
    action_failure: str | None


@dataclass(frozen=True, slots=True)
class GeneratedReply:
    """保存模型生成文本及其是否因 token 预算截断。

    Args:
        text (str): 去除特殊 token 后的 assistant 文本。
        truncated (bool): 生成未遇到 EOS 就耗尽预算时为真。
    """

    text: str
    truncated: bool


def _resolve_evaluation_split(split: str) -> str:
    """把公开分卷名称转换成内部兼容键。

    Args:
        split (str): ``validation``/``dev`` 或 ``eval``/``test``。

    Raises:
        SftTrainingError: 分卷不支持。

    Returns:
        str: ``dev`` 或 ``test``。
    """
    aliases = {"validation": "dev", "dev": "dev", "eval": "test", "test": "test"}
    try:
        return aliases[split]
    except KeyError as exc:
        raise SftTrainingError(f"评测只允许 validation/eval: {split}") from exc


def _manifest_split_files(
    manifest: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    """从数据 manifest 取得某棵分卷目录的全部文件记录。

    Args:
        manifest (Mapping[str, Any]): 已通过完整性校验的数据清单。
        split (str): 内部 ``dev`` 或 ``test`` 分卷键。

    Raises:
        SftTrainingError: manifest 缺少文件映射。

    Returns:
        dict[str, Any]: 相对路径到行数与摘要记录的映射。
    """
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise SftTrainingError("SFT manifest 缺少 files")
    prefix = dataset_split_path(Path(), split).as_posix() + "/"
    return {
        str(name): record
        for name, record in files.items()
        if str(name).startswith(prefix)
    }


def chunked_masked_stats(
    lm_head: Any,
    hidden: Any,
    input_ids: Any,
    labels: Any,
    *,
    chunk_size: int,
) -> MaskedTokenStats:
    """在不物化完整 logits 的情况下计算移位 CE 与 token accuracy。

    Args:
        lm_head (Any): 模型的词表投影层。
        hidden (Any): decoder 的完整 hidden state。
        input_ids (Any): 完整对话 token ID。
        labels (Any): 非 assistant 位置为 ``-100`` 的标签。
        chunk_size (int): 每次词表投影的最大序列长度。

    Raises:
        SftTrainingError: 分块长度无效。

    Returns:
        MaskedTokenStats: 可跨样本累加的损失、正确数与 token 数。
    """
    from torch.nn import functional

    if chunk_size <= 0:
        raise SftTrainingError("logits chunk_size 必须为正数")
    loss_sum = 0.0
    correct = 0
    tokens = 0
    last_predictor = hidden.shape[1] - 1
    for start in range(0, last_predictor, chunk_size):
        end = min(start + chunk_size, last_predictor)
        logits = lm_head(hidden[:, start:end]).float()
        targets = input_ids[:, start + 1 : end + 1]
        keep = labels[:, start + 1 : end + 1].ne(IGNORE_LABEL)
        if not keep.any().item():
            continue
        selected_logits = logits[keep]
        selected_targets = targets[keep]
        loss_sum += float(
            functional.cross_entropy(
                selected_logits,
                selected_targets,
                reduction="sum",
            ).item()
        )
        correct += int(
            (selected_logits.argmax(dim=-1) == selected_targets).sum().item()
        )
        tokens += int(selected_targets.numel())
    return MaskedTokenStats(loss_sum, correct, tokens)


def evaluate_tokenized_samples(
    model: Any,
    samples: Sequence[TokenizedSample],
    *,
    device: str,
    chunk_size: int,
) -> dict[str, int | float]:
    """计算一组 token 化样本的 assistant-only teacher-forced 指标。

    Args:
        model (Any): 已加载 adapter 的 PyTorch 因果语言模型。
        samples (Sequence[TokenizedSample]): 待评测的 prompt-loss-masked 样本。
        device (str): PyTorch 设备字符串。
        chunk_size (int): 词表投影的序列分块长度。

    Raises:
        SftTrainingError: 样本为空、模型结构不支持或监督 token 为零。

    Returns:
        dict[str, int | float]: 样本数、监督 token、loss、困惑度和准确率。
    """
    import torch

    if not samples:
        raise SftTrainingError("teacher-forced 评测样本为空")
    decoder_getter = getattr(model, "get_decoder", None)
    decoder = decoder_getter() if callable(decoder_getter) else None
    lm_head = getattr(model, "lm_head", None)
    if decoder is None or lm_head is None:
        raise SftTrainingError("模型没有可分块评测的 decoder/lm_head")
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for sample in samples:
            input_ids = torch.tensor(
                [sample.input_ids],
                dtype=torch.long,
                device=device,
            )
            labels = torch.tensor(
                [sample.labels],
                dtype=torch.long,
                device=device,
            )
            hidden = decoder(
                input_ids=input_ids,
                attention_mask=None,
            ).last_hidden_state
            stats = chunked_masked_stats(
                lm_head,
                hidden,
                input_ids,
                labels,
                chunk_size=chunk_size,
            )
            total_loss += stats.loss_sum
            total_correct += stats.correct
            total_tokens += stats.tokens
            if device == "mps":
                torch.mps.empty_cache()
    if total_tokens == 0:
        raise SftTrainingError("teacher-forced 评测没有监督 token")
    mean_loss = total_loss / total_tokens
    return {
        "rows": len(samples),
        "supervised_tokens": total_tokens,
        "mean_loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "token_accuracy": total_correct / total_tokens,
    }


def evaluate_sft_loss(
    config: SftConfig,
    adapter_path: Path,
    split: str,
    *,
    max_samples: int | None = None,
    output_path: Path | None = None,
) -> dict[str, object]:
    """加载 LoRA adapter 并执行 assistant-only teacher-forced 评测。

    Args:
        config (SftConfig): 训练和模型配置。
        adapter_path (Path): 待评测的 PEFT adapter 目录。
        split (str): ``dev`` 或 ``test`` 数据分卷。
        max_samples (int | None): 可选的评测样本上限。
        output_path (Path | None): 可选聚合 JSON 报告路径。

    Raises:
        SftTrainingError: 分卷、样本数量或模型结构无效。
        OSError: 无法读取模型数据或写入报告。

    Returns:
        dict[str, object]: 含来源、耗时和 teacher-forced 指标的报告。
    """
    resolved_split = _resolve_evaluation_split(split)
    if max_samples is not None and max_samples <= 0:
        raise SftTrainingError("max_samples 必须为正数")
    dataset_manifest = validate_sft_dataset(config.dataset_root)
    adapter_path = Path(adapter_path)
    if output_path is None:
        output_path = (
            config.runs_root.parent / "eval" / f"{adapter_path.name}-{split}-loss.json"
        )
    tokenizer, model, device = _load_evaluation_model(config, adapter_path)
    samples = load_tokenized_samples(
        dataset_split_path(config.dataset_root, resolved_split),
        tokenizer,
        max_length=config.max_length,
    )
    if max_samples is not None:
        samples = samples[:max_samples]
    started = time.monotonic()
    summary: dict[str, object] = dict(
        evaluate_tokenized_samples(
            model,
            samples,
            device=device,
            chunk_size=config.logits_chunk_size,
        )
    )
    summary.update(
        {
            "base_model": str(config.base_model),
            "adapter": str(adapter_path),
            "split": split,
            "output": str(output_path),
            "device": device,
            "elapsed_seconds": time.monotonic() - started,
            "dataset_files": _manifest_split_files(dataset_manifest, resolved_split),
        }
    )
    _write_json(output_path, summary)
    return summary


def score_generation(
    *,
    expected: str,
    generated: str,
    source: str,
    available_actions: Sequence[str] = (),
    legal_actions: Sequence[str] = (),
) -> GenerationScore:
    """计算不依赖模型库的生成结果指标。

    Args:
        expected (str): 数据集中的目标 assistant 回复。
        generated (str): 模型实际生成的回复。
        source (str): 当前样本来源。
        available_actions (Sequence[str]): 当前向模型开放的动作名称。
        legal_actions (Sequence[str]): 已按当前状态索引展开的规范动作行。

    Returns:
        GenerationScore: 精确匹配与可选动作外形指标。
    """
    expected = expected.strip()
    generated = generated.strip()
    if source != "human_play":
        return GenerationScore(expected == generated, None, None, None)
    if not generated:
        return GenerationScore(False, False, False, "empty_output")
    if _ACTION_LINE.fullmatch(generated) is None:
        return GenerationScore(
            expected == generated,
            False,
            False,
            "invalid_format",
        )
    action_name = generated.split()[1]
    if action_name not in available_actions:
        return GenerationScore(
            expected == generated,
            True,
            False,
            "unavailable_action",
        )
    if generated not in legal_actions:
        return GenerationScore(
            expected == generated,
            True,
            False,
            "invalid_parameters",
        )
    return GenerationScore(expected == generated, True, True, None)


def decode_generation(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    eos_token_id: int,
) -> GeneratedReply:
    """解码新增 token，并标记没有生成 EOS 的预算截断。

    Args:
        tokenizer (Any): 提供 ``decode`` 的 Hugging Face tokenizer。
        token_ids (Sequence[int]): 不含 prompt 的模型新增 token。
        eos_token_id (int): 当前 tokenizer 的 assistant 结束 token ID。

    Returns:
        GeneratedReply: 清理后的文本与截断标记。
    """
    ids = [int(token_id) for token_id in token_ids]
    return GeneratedReply(
        text=str(tokenizer.decode(ids, skip_special_tokens=True)).strip(),
        truncated=eos_token_id not in ids,
    )


def evaluate_rows(
    dataset_path: Path,
    generate: Callable[[list[dict[str, str]]], GeneratedReply],
    *,
    output_path: Path,
    max_samples: int | None = None,
) -> dict[str, int | float]:
    """递归读取分卷目录并逐行生成可复查的对照结果。

    Args:
        dataset_path (Path): 待读取的 SFT 目录或兼容单个 JSONL。
        generate (Callable[[list[dict[str, str]]], GeneratedReply]): 接收
            prompt 消息并返回模型文本与截断状态的生成函数。
        output_path (Path): 逐样本评测 JSONL 路径。
        max_samples (int | None): 可选的样本数量上限。

    Raises:
        SftTrainingError: 数据行无效或没有可评测样本。
        OSError: 无法读取分卷或写入结果。

    Returns:
        dict[str, int | float]: 样本数、精确匹配率与动作外形通过率。
    """
    if max_samples is not None and max_samples <= 0:
        raise SftTrainingError("max_samples 必须为正数")
    records: list[dict[str, object]] = []
    paths = (
        sorted(Path(dataset_path).rglob("*.jsonl"))
        if Path(dataset_path).is_dir()
        else [Path(dataset_path)]
    )
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SftTrainingError(
                        f"无效评测 JSONL: {path}:{line_number}"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise SftTrainingError(f"评测行不是对象: {path}:{line_number}")
                sample_id = row.get("sample_id")
                source = row.get("source")
                messages = row.get("messages")
                if (
                    not isinstance(sample_id, str)
                    or not isinstance(source, str)
                    or not isinstance(messages, Sequence)
                    or isinstance(messages, (str, bytes))
                ):
                    raise SftTrainingError(f"评测行缺少样本字段: {path}:{line_number}")
                normalized = normalize_messages(messages, sample_id)
                expected = normalized[-1]["content"]
                reply = generate(normalized[:-1])
                available_actions: Sequence[str] = ()
                legal_actions: Sequence[str] = ()
                if source == "human_play":
                    available_actions = _row_string_sequence(
                        row,
                        "available_actions",
                        path=path,
                        line_number=line_number,
                    )
                    legal_actions = _row_string_sequence(
                        row,
                        "legal_actions",
                        path=path,
                        line_number=line_number,
                    )
                score = score_generation(
                    expected=expected,
                    generated=reply.text,
                    source=source,
                    available_actions=available_actions,
                    legal_actions=legal_actions,
                )
                records.append(
                    {
                        "sample_id": sample_id,
                        "source": source,
                        "expected": expected,
                        "generated": reply.text,
                        "truncated": reply.truncated,
                        "exact_match": score.exact_match,
                        "action_shape_valid": score.action_shape_valid,
                        "action_legal": score.action_legal,
                        "action_failure": score.action_failure,
                    }
                )
                if max_samples is not None and len(records) >= max_samples:
                    break
        if max_samples is not None and len(records) >= max_samples:
            break
    if not records:
        raise SftTrainingError(f"评测分卷为空: {dataset_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    exact_matches = sum(record["exact_match"] is True for record in records)
    action_records = [
        record for record in records if record["action_shape_valid"] is not None
    ]
    valid_actions = sum(
        record["action_shape_valid"] is True for record in action_records
    )
    legal_actions = sum(record["action_legal"] is True for record in action_records)
    failures = Counter(
        str(record["action_failure"])
        for record in action_records
        if record["action_failure"] is not None
    )
    return {
        "samples": len(records),
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / len(records),
        "action_samples": len(action_records),
        "valid_action_shapes": valid_actions,
        "valid_action_shape_rate": (
            valid_actions / len(action_records) if action_records else 0.0
        ),
        "legal_actions": legal_actions,
        "legal_action_rate": (
            legal_actions / len(action_records) if action_records else 0.0
        ),
        "empty_action_outputs": failures["empty_output"],
        "invalid_action_formats": failures["invalid_format"],
        "unavailable_actions": failures["unavailable_action"],
        "invalid_action_parameters": failures["invalid_parameters"],
        "truncated_samples": sum(record["truncated"] is True for record in records),
    }


def _row_string_sequence(
    row: Mapping[str, Any],
    field: str,
    *,
    path: Path,
    line_number: int,
) -> tuple[str, ...]:
    """读取行为评测行中必需的字符串数组。

    Args:
        row (Mapping[str, Any]): 当前评测数据行。
        field (str): 待读取的字段名称。
        path (Path): 用于错误定位的 JSONL 路径。
        line_number (int): 当前数据行号。

    Raises:
        SftTrainingError: 字段缺失、不是数组或包含非字符串值。

    Returns:
        tuple[str, ...]: 已验证的字符串序列。
    """
    values = row.get(field)
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or any(not isinstance(value, str) for value in values)
    ):
        raise SftTrainingError(f"行为评测行缺少 {field}: {path}:{line_number}")
    return tuple(values)


def evaluate_sft(
    config: SftConfig,
    adapter_path: Path,
    split: str,
    *,
    max_samples: int | None = None,
    output_path: Path | None = None,
    temperature: float = 0.0,
) -> dict[str, object]:
    """加载 LoRA adapter，执行独立生成式评测。

    Args:
        config (SftConfig): 训练和模型配置。
        adapter_path (Path): 待评测的 PEFT adapter 目录。
        split (str): ``dev`` 或 ``test`` 数据分卷。
        max_samples (int | None): 可选的评测样本上限。
        output_path (Path | None): 可选逐样本 JSONL 路径。
        temperature (float): ``0`` 使用贪心生成，正数启用对应温度的采样。

    Raises:
        SftTrainingError: 分卷无效、为空或设备不可用。
        OSError: 无法读取模型数据或写入报告。

    Returns:
        dict[str, object]: 含模型来源和聚合生成指标的评测清单。
    """
    import torch

    resolved_split = _resolve_evaluation_split(split)
    if not math.isfinite(temperature) or temperature < 0:
        raise SftTrainingError("temperature 必须为非负有限数")
    dataset_manifest = validate_sft_dataset(config.dataset_root)
    adapter_path = Path(adapter_path)
    if output_path is None:
        output_path = (
            config.runs_root.parent
            / "eval"
            / _generation_output_name(adapter_path.name, split, temperature)
        )
    tokenizer, model, device = _load_evaluation_model(config, adapter_path)

    def generate(messages: list[dict[str, str]]) -> GeneratedReply:
        """为一组 prompt 消息生成确定性 assistant 回复。

        Args:
            messages (list[dict[str, str]]): 不含目标 assistant 的消息。

        Returns:
            GeneratedReply: 去除特殊 token 后的文本与截断状态。
        """
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=False,
        ).to(device)
        with torch.no_grad():
            generation_options: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "max_new_tokens": config.eval_max_new_tokens,
                "do_sample": temperature > 0,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if temperature > 0:
                generation_options["temperature"] = temperature
            generated_ids = model.generate(
                **generation_options,
            )
        continuation = generated_ids[0, input_ids.shape[-1] :]
        return decode_generation(
            tokenizer,
            continuation,
            eos_token_id=tokenizer.eos_token_id,
        )

    summary: dict[str, object] = dict(
        evaluate_rows(
            dataset_split_path(config.dataset_root, resolved_split),
            generate,
            output_path=output_path,
            max_samples=max_samples,
        )
    )
    summary.update(
        {
            "base_model": str(config.base_model),
            "adapter": str(adapter_path),
            "split": split,
            "output": str(output_path),
            "device": device,
            "temperature": temperature,
            "enable_thinking": False,
            "max_retries": 0,
            "dataset_files": _manifest_split_files(dataset_manifest, resolved_split),
        }
    )
    _write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def _generation_output_name(
    adapter_name: str,
    split: str,
    temperature: float,
) -> str:
    """生成不会让不同采样温度互相覆盖的默认报告文件名。

    Args:
        adapter_name (str): 当前 LoRA adapter 目录名。
        split (str): 用户请求的公开评测分卷名。
        temperature (float): 本轮生成温度。

    Returns:
        str: 包含稳定温度标签的 JSONL 文件名。
    """
    temperature_label = format(temperature, ".12g").replace(".", "p")
    return f"{adapter_name}-{split}-t{temperature_label}.jsonl"


def _load_evaluation_model(
    config: SftConfig,
    adapter_path: Path,
) -> tuple[Any, Any, str]:
    """加载共用 tokenizer、基座与只读 PEFT adapter。

    Args:
        config (SftConfig): 本地模型与设备配置。
        adapter_path (Path): 待加载的 adapter 目录。

    Returns:
        tuple[Any, Any, str]: tokenizer、已装配模型与解析后的设备。
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.base_model),
        local_files_only=True,
        trust_remote_code=False,
    )
    device = resolve_device(config.device)
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        str(config.base_model),
        dtype=dtype,
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _write_json(path: Path, value: object) -> None:
    """以统一缩进写入 UTF-8 JSON。

    Args:
        path (Path): 目标 JSON 路径。
        value (object): 可 JSON 序列化的值。

    Returns:
        None: 文件写入完成后返回。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
