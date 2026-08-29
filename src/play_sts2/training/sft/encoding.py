"""提供 SFT 配置、消息校验与 prompt-loss-mask 编码。"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

IGNORE_LABEL = -100
_ASSISTANT_BRANCH = '{%- elif message.role == "assistant" %}'
_TOOL_BRANCH = '{%- elif message.role == "tool" %}'
_GENERATION_MARKERS = (
    (
        "+ '\\n</think>\\n\\n' + content }}",
        "+ '\\n</think>\\n\\n' }}{% generation %}{{ content }}{% endgeneration %}",
    ),
    (
        "+ '\\n' + content }}",
        "+ '\\n' }}{% generation %}{{ content }}{% endgeneration %}",
    ),
    (
        "{{- '<|im_end|>\\n' }}",
        "{% generation %}{{- '<|im_end|>' }}{% endgeneration %}{{ '\\n' }}",
    ),
)


class SftTrainingError(RuntimeError):
    """表示 SFT 配置、样本或运行状态不符合训练契约。"""


@dataclass(frozen=True, slots=True)
class SftConfig:
    """保存一次 Qwen LoRA SFT 使用的稳定配置。

    Args:
        base_model (Path): 本地 Hugging Face 基座模型目录。
        dataset_root (Path): 含训练、验证和最终测试分卷的数据集目录。
        adapter_root (Path): 最终 LoRA adapter 的父目录。
        runs_root (Path): 训练过程记录的父目录。
        device (str): ``auto``、``mps`` 或 ``cpu``。
        epochs (int): 训练数据遍历次数。
        learning_rate (float): AdamW 峰值学习率。
        max_length (int): 接受的最大完整样本 token 数。
        gradient_accumulation_steps (int): 每次优化累积的样本数。
        seed (int): 样本洗牌和 PyTorch 使用的随机种子。
        lora_rank (int): LoRA 矩阵秩。
        lora_alpha (int): LoRA 缩放参数。
        max_grad_norm (float): 梯度裁剪上限。
        eval_max_new_tokens (int): 单条生成式评测的最大新 token 数。
        warmup_steps (int): 线性学习率预热的优化步数。
        logits_chunk_size (int): 分块词表投影与交叉熵的序列长度。
        checkpoint_steps (int): 覆盖 ``checkpoint-last`` 的优化步间隔；零为关闭。
        init_adapter (Path | None): 可选的近似续训 LoRA adapter。
    """

    base_model: Path
    dataset_root: Path
    adapter_root: Path
    runs_root: Path
    device: str
    epochs: int
    learning_rate: float
    max_length: int
    gradient_accumulation_steps: int
    seed: int
    lora_rank: int
    lora_alpha: int
    max_grad_norm: float = 1.0
    eval_max_new_tokens: int = 256
    warmup_steps: int = 4
    logits_chunk_size: int = 2048
    checkpoint_steps: int = 2000
    init_adapter: Path | None = None


@dataclass(frozen=True, slots=True)
class TokenizedSample:
    """保存一条只监督 assistant 回复的 token 化样本。

    Args:
        sample_id (str): 派生数据中的稳定样本 ID。
        source (str): ``web_wiki``、``mod_export`` 或 ``human_play``。
        input_ids (tuple[int, ...]): 完整对话 token ID。
        labels (tuple[int, ...]): 非 assistant 位置为 ``-100`` 的训练标签。
        training_epoch (int | None): 知识问法指定的训练轮次；锚点样本为空。
    """

    sample_id: str
    source: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    training_epoch: int | None = None

    @property
    def supervised_tokens(self) -> int:
        """返回参与交叉熵计算的 token 数。

        Returns:
            int: labels 中不等于 ``-100`` 的位置数。
        """
        return sum(label != IGNORE_LABEL for label in self.labels)


def load_sft_config(path: Path) -> SftConfig:
    """从扁平 TOML 文件读取类型化 SFT 配置。

    Args:
        path (Path): 项目内可读的训练配置文件。

    Raises:
        SftTrainingError: 字段缺失、类型错误或数值不合法。
        OSError: 配置文件无法读取。
        tomllib.TOMLDecodeError: TOML 语法无效。

    Returns:
        SftConfig: 已完成数值校验的训练配置。
    """
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    try:
        init_adapter = data.get("init_adapter")
        config = SftConfig(
            base_model=Path(data["base_model"]),
            dataset_root=Path(data["dataset_root"]),
            adapter_root=Path(data["adapter_root"]),
            runs_root=Path(data["runs_root"]),
            device=str(data["device"]),
            epochs=int(data["epochs"]),
            learning_rate=float(data["learning_rate"]),
            max_length=int(data["max_length"]),
            gradient_accumulation_steps=int(data["gradient_accumulation_steps"]),
            seed=int(data["seed"]),
            lora_rank=int(data["lora_rank"]),
            lora_alpha=int(data["lora_alpha"]),
            max_grad_norm=float(data.get("max_grad_norm", 1.0)),
            eval_max_new_tokens=int(data.get("eval_max_new_tokens", 256)),
            warmup_steps=int(data.get("warmup_steps", 4)),
            logits_chunk_size=int(data.get("logits_chunk_size", 2048)),
            checkpoint_steps=int(data.get("checkpoint_steps", 2000)),
            init_adapter=Path(init_adapter) if init_adapter else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SftTrainingError(f"无效 SFT 配置: {exc}") from exc
    positive = {
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "max_length": config.max_length,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "max_grad_norm": config.max_grad_norm,
        "eval_max_new_tokens": config.eval_max_new_tokens,
        "logits_chunk_size": config.logits_chunk_size,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if config.warmup_steps < 0:
        invalid.append("warmup_steps")
    if config.checkpoint_steps < 0:
        invalid.append("checkpoint_steps")
    if invalid or config.device not in {"auto", "mps", "cpu"}:
        detail = ", ".join(invalid) if invalid else f"device={config.device}"
        raise SftTrainingError(f"SFT 配置值无效: {detail}")
    return config


def encode_messages(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    sample_id: str,
    source: str,
    max_length: int,
    training_epoch: int | None = None,
) -> TokenizedSample:
    """用模型原生模板编码一个独立 assistant 回复。

    生成提示与完整回答必须保持 token 前缀关系，以屏蔽 system/user
    并只监督末尾 assistant。

    Args:
        tokenizer (Any): 支持 ``apply_chat_template`` 的 Hugging Face tokenizer。
        messages (Sequence[Mapping[str, str]]): 数据集中的完整聊天消息。
        sample_id (str): 用于错误定位的稳定样本 ID。
        source (str): 样本来源类别。
        max_length (int): 允许的最大完整序列长度。
        training_epoch (int | None): 当前问法指定的训练轮次。

    Raises:
        SftTrainingError: 消息形状、模板边界或序列长度不符合约定。

    Returns:
        TokenizedSample: 含 assistant-loss-mask 的 token 化样本。
    """
    normalized = normalize_messages(messages, sample_id)
    input_ids = _render_messages(tokenizer, normalized, generation_prompt=False)
    if len(input_ids) > max_length:
        raise SftTrainingError(
            f"{sample_id}: {len(input_ids)} token 超过 max_length={max_length}"
        )
    native_labels = _assistant_labels_from_template(
        tokenizer,
        normalized,
        input_ids,
        sample_id,
    )
    if native_labels is not None:
        return TokenizedSample(
            sample_id=sample_id,
            source=source,
            input_ids=tuple(input_ids),
            labels=tuple(native_labels),
            training_epoch=training_epoch,
        )
    prompt_ids = _render_messages(
        tokenizer,
        normalized[:-1],
        generation_prompt=True,
    )
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise SftTrainingError(f"{sample_id}: 生成提示不是完整回答的前缀")
    if len(input_ids) == len(prompt_ids):
        raise SftTrainingError(f"{sample_id}: assistant 回复没有监督 token")
    labels = [IGNORE_LABEL] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    return TokenizedSample(
        sample_id=sample_id,
        source=source,
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        training_epoch=training_epoch,
    )


@lru_cache(maxsize=4)
def mark_generation_spans(chat_template: str) -> str:
    """只在 Qwen assistant 内容与 ``im_end`` 周围加入 generation 标记。

    标记只服务于 Hugging Face 的 ``assistant_masks`` 通道，不允许改变渲染
    token。每项手术都要求在 assistant 分支内恰好命中一次，模板升级漂移时
    立即失败。

    Args:
        chat_template (str): tokenizer 使用的原始 Jinja chat template。

    Raises:
        SftTrainingError: assistant 分支缺失或目标片段命中数不为一。

    Returns:
        str: 加入 ``{% generation %}`` 标记的等价模板。
    """
    try:
        begin = chat_template.index(_ASSISTANT_BRANCH)
        end = chat_template.index(_TOOL_BRANCH, begin)
    except ValueError as exc:
        raise SftTrainingError("Qwen chat template 的 assistant 分支发生漂移") from exc
    head = chat_template[:begin]
    branch = chat_template[begin:end]
    tail = chat_template[end:]
    for original, marked in _GENERATION_MARKERS:
        occurrences = branch.count(original)
        if occurrences != 1:
            raise SftTrainingError(
                "Qwen chat template 的 assistant 掩码片段发生漂移: "
                f"{original!r} 命中 {occurrences} 次"
            )
        branch = branch.replace(original, marked)
    return head + branch + tail


def _assistant_labels_from_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    input_ids: Sequence[int],
    sample_id: str,
) -> list[int] | None:
    """优先用 generation 标记读取 Qwen 原生 assistant token 掩码。

    Args:
        tokenizer (Any): 可能带 ``chat_template`` 的 tokenizer。
        messages (Sequence[Mapping[str, str]]): 已规范化的完整聊天消息。
        input_ids (Sequence[int]): 原模板渲染得到的 token ID。
        sample_id (str): 用于错误定位的样本 ID。

    Raises:
        SftTrainingError: 标记模板改变渲染、掩码错位或掩码为空。

    Returns:
        list[int] | None: 原生 assistant labels；测试替身无模板时返回 ``None``。
    """
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        return None
    encoded = tokenizer.apply_chat_template(
        messages,
        chat_template=mark_generation_spans(chat_template),
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    marked_ids = [int(token_id) for token_id in encoded["input_ids"]]
    mask = [int(keep) for keep in encoded["assistant_masks"]]
    if marked_ids != list(input_ids):
        raise SftTrainingError(f"{sample_id}: assistant 标记模板改变了渲染 token")
    if len(mask) != len(marked_ids):
        raise SftTrainingError(f"{sample_id}: assistant 掩码与 token 长度不一致")
    labels = [
        token_id if keep else IGNORE_LABEL for token_id, keep in zip(marked_ids, mask)
    ]
    if all(label == IGNORE_LABEL for label in labels):
        raise SftTrainingError(f"{sample_id}: assistant 原生掩码没有监督 token")
    return labels


def load_tokenized_samples(
    path: Path,
    tokenizer: Any,
    *,
    max_length: int,
) -> list[TokenizedSample]:
    """递归读取可读 SFT 目录或单个 JSONL 并编码全部样本。

    Args:
        path (Path): 数据集分卷目录或兼容的单个 JSONL。
        tokenizer (Any): 目标模型 tokenizer。
        max_length (int): 允许的最大完整序列长度。

    Raises:
        SftTrainingError: JSONL 行或消息不符合训练契约，或分卷为空。
        OSError: 数据文件无法读取。

    Returns:
        list[TokenizedSample]: 保持原始行顺序的 token 化样本。
    """
    samples: list[TokenizedSample] = []
    for jsonl_path in _sft_jsonl_paths(path):
        with jsonl_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SftTrainingError(
                        f"无效 SFT JSONL: {jsonl_path}:{line_number}"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise SftTrainingError(
                        f"SFT 行不是对象: {jsonl_path}:{line_number}"
                    )
                sample_id = row.get("sample_id")
                source = row.get("source")
                messages = row.get("messages")
                training_epoch = row.get("training_epoch")
                if (
                    not isinstance(sample_id, str)
                    or not isinstance(source, str)
                    or not isinstance(messages, Sequence)
                    or isinstance(messages, (str, bytes))
                ):
                    raise SftTrainingError(
                        f"SFT 行缺少样本字段: {jsonl_path}:{line_number}"
                    )
                if training_epoch is not None and (
                    not isinstance(training_epoch, int)
                    or isinstance(training_epoch, bool)
                    or training_epoch not in {1, 2, 3, 4, 5}
                ):
                    raise SftTrainingError(
                        f"SFT 行 training_epoch 无效: {jsonl_path}:{line_number}"
                    )
                samples.append(
                    encode_messages(
                        tokenizer,
                        messages,
                        sample_id=sample_id,
                        source=source,
                        max_length=max_length,
                        training_epoch=training_epoch,
                    )
                )
    if not samples:
        raise SftTrainingError(f"SFT 分卷为空: {path}")
    return samples


def _sft_jsonl_paths(path: Path) -> list[Path]:
    """解析单文件或目录形式的 SFT 输入。

    Args:
        path (Path): JSONL 文件或包含 JSONL 的目录。

    Raises:
        SftTrainingError: 路径既不是 JSONL 文件也不是目录。

    Returns:
        list[Path]: 按相对路径稳定排序的 JSONL 文件。
    """
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.jsonl"))
    raise SftTrainingError(f"SFT 分卷不存在: {path}")


def normalize_messages(
    messages: Sequence[Mapping[str, str]],
    sample_id: str,
) -> list[dict[str, str]]:
    """校验训练消息必须是独立的单轮回复。

    Args:
        messages (Sequence[Mapping[str, str]]): 原始数据集消息。
        sample_id (str): 用于错误定位的稳定样本 ID。

    Raises:
        SftTrainingError: 字段无效或角色不是支持的单轮结构。

    Returns:
        list[dict[str, str]]: 只保留 role 与 content 的普通字典。
    """
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role") if isinstance(message, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise SftTrainingError(f"{sample_id}: 消息字段无效")
        normalized.append({"role": role, "content": content})
    roles = tuple(message["role"] for message in normalized)
    if roles not in {
        ("user", "assistant"),
        ("system", "user", "assistant"),
    }:
        raise SftTrainingError(f"{sample_id}: 只接受独立的单轮样本")
    return normalized


def config_as_json(config: SftConfig) -> dict[str, object]:
    """把包含 Path 的配置转换成 JSON 对象。

    Args:
        config (SftConfig): 类型化训练配置。

    Returns:
        dict[str, object]: 可直接写入 JSON 的有效配置。
    """
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _render_messages(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    generation_prompt: bool,
) -> list[int]:
    """用固定 Qwen 训练参数渲染一段聊天消息。

    Args:
        tokenizer (Any): 支持 Hugging Face chat template 的 tokenizer。
        messages (Sequence[Mapping[str, str]]): 当前消息前缀。
        generation_prompt (bool): 是否追加 assistant 生成前缀。

    Returns:
        list[int]: 普通整数 token ID。
    """
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generation_prompt,
        enable_thinking=False,
        return_dict=False,
    )
    return [int(token_id) for token_id in rendered]
