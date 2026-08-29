"""验证 SFT 配置、目录读取、消息编码与 assistant-only loss mask。"""

import json
from pathlib import Path

import pytest

from play_sts2.training.sft import (
    SftConfig,
    SftTrainingError,
    encode_messages,
    load_sft_config,
    load_tokenized_samples,
)


class FakeTokenizer:
    """模拟 Qwen 模板的生成提示与完整回答前缀关系。"""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        return_dict: bool,
    ) -> list[int]:
        """返回便于断言监督边界的固定 token。

        Args:
            messages (list[dict[str, str]]): 待渲染的聊天消息。
            tokenize (bool): 是否直接返回 token ID。
            add_generation_prompt (bool): 是否追加 assistant 生成前缀。
            enable_thinking (bool): 是否启用思考块。
            return_dict (bool): 是否返回带命名字段的编码对象。

        Raises:
            AssertionError: 调用没有采用训练约定的模板参数。

        Returns:
            list[int]: 模拟的模板 token ID。
        """
        assert tokenize is True
        assert enable_thinking is False
        assert return_dict is False
        if add_generation_prompt:
            assert messages[-1]["role"] == "user"
            return [10, 11, 12]
        assert messages[-1]["role"] == "assistant"
        return [10, 11, 12, 20, 21]


class DriftedTokenizer(FakeTokenizer):
    """模拟生成提示不再是完整回答前缀的漂移模板。"""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        return_dict: bool,
    ) -> list[int]:
        """在完整回答中改写一个前缀 token。

        Args:
            messages (list[dict[str, str]]): 待渲染的聊天消息。
            tokenize (bool): 是否直接返回 token ID。
            add_generation_prompt (bool): 是否追加 assistant 生成前缀。
            enable_thinking (bool): 是否启用思考块。
            return_dict (bool): 是否返回带命名字段的编码对象。

        Returns:
            list[int]: 前缀故意不一致的 token ID。
        """
        result = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            return_dict=return_dict,
        )
        return result if add_generation_prompt else [10, 99, *result[2:]]


def test_load_tokenized_samples_recurses_directory_tree(tmp_path: Path) -> None:
    """训练编码器应按路径稳定递归读取分卷目录。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Raises:
        AssertionError: 子目录样本被漏读或顺序不稳定。

    Returns:
        None: 此测试只使用固定 tokenizer。
    """
    root = tmp_path / "train"
    for relative, sample_id in (
        ("cards/ZAP.jsonl", "knowledge/ZAP"),
        ("strategy/RUN.jsonl", "human/RUN/1"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "sample_id": sample_id,
            "source": ("human_play" if sample_id.startswith("human") else "knowledge"),
            "messages": [
                {"role": "user", "content": "状态"},
                {"role": "assistant", "content": "答案"},
            ],
        }
        if sample_id.startswith("knowledge"):
            row["training_epoch"] = 2
        path.write_text(
            json.dumps(row, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    samples = load_tokenized_samples(root, FakeTokenizer(), max_length=32)

    assert [sample.sample_id for sample in samples] == [
        "knowledge/ZAP",
        "human/RUN/1",
    ]
    assert [sample.training_epoch for sample in samples] == [2, None]


def test_load_sft_config_reads_minimal_toml(tmp_path: Path) -> None:
    """读取项目路径和最小 LoRA 超参数。

    Args:
        tmp_path (Path): Pytest 提供的隔离配置目录。

    Raises:
        AssertionError: TOML 字段没有进入类型化配置。

    Returns:
        None: 此测试只检查配置读取。
    """
    config_path = tmp_path / "sft.toml"
    config_path.write_text(
        """base_model = "models/base/qwen3.5-4b"
        dataset_root = "data/datasets/sft"
        adapter_root = "models/adapters"
        runs_root = "runs/sft"
        device = "cpu"
        epochs = 2
        learning_rate = 0.0001
        max_length = 1024
        gradient_accumulation_steps = 4
        seed = 7
        lora_rank = 8
        lora_alpha = 16
        warmup_steps = 8
        logits_chunk_size = 512
        checkpoint_steps = 2000
        init_adapter = "models/adapters/e2"
        """,
        encoding="utf-8",
    )

    config = load_sft_config(config_path)

    assert config == SftConfig(
        base_model=Path("models/base/qwen3.5-4b"),
        dataset_root=Path("data/datasets/sft"),
        adapter_root=Path("models/adapters"),
        runs_root=Path("runs/sft"),
        device="cpu",
        epochs=2,
        learning_rate=0.0001,
        max_length=1024,
        gradient_accumulation_steps=4,
        seed=7,
        lora_rank=8,
        lora_alpha=16,
        warmup_steps=8,
        logits_chunk_size=512,
        checkpoint_steps=2000,
        init_adapter=Path("models/adapters/e2"),
    )


def test_load_sft_config_accepts_explicit_cuda_device(tmp_path: Path) -> None:
    """生成式评测应能复用显式指定 CUDA 卡号的 SFT 配置。

    Args:
        tmp_path (Path): Pytest 提供的隔离配置目录。

    Raises:
        AssertionError: 通用 SFT 配置加载器拒绝合法的 CUDA 设备。

    Returns:
        None: 此测试只检查设备配置契约。
    """
    config_path = tmp_path / "sft-cuda.toml"
    config_path.write_text(
        """base_model = "models/base/qwen3.5-4b"
        dataset_root = "data/datasets/sft"
        adapter_root = "models/adapters"
        runs_root = "runs/sft"
        device = "cuda:2"
        epochs = 2
        learning_rate = 0.0001
        max_length = 1024
        gradient_accumulation_steps = 4
        seed = 7
        lora_rank = 8
        lora_alpha = 16
        """,
        encoding="utf-8",
    )

    config = load_sft_config(config_path)

    assert config.device == "cuda:2"


def test_encode_messages_only_supervises_final_assistant() -> None:
    """prompt token 使用忽略标签，assistant 内容和结尾参与损失。

    Raises:
        AssertionError: labels 没有从真实生成边界开始。

    Returns:
        None: 此测试只检查 token 监督范围。
    """
    sample = encode_messages(
        FakeTokenizer(),
        [
            {"role": "system", "content": "规则"},
            {"role": "user", "content": "状态"},
            {"role": "assistant", "content": "ACTION: end_turn"},
        ],
        sample_id="human/RUN/1",
        source="human_play",
        max_length=32,
    )

    assert sample.input_ids == (10, 11, 12, 20, 21)
    assert sample.labels == (-100, -100, -100, 20, 21)
    assert sample.supervised_tokens == 2


def test_encode_messages_rejects_cross_action_history() -> None:
    """编码层拒绝把多个人类动作重新拼成增长历史。

    Raises:
        AssertionError: 多轮行为样本没有在 token 化前被拒绝。

    Returns:
        None: 此测试只检查 stateless 训练边界。
    """
    with pytest.raises(SftTrainingError, match="只接受独立的单轮样本"):
        encode_messages(
            object(),
            [
                {"role": "system", "content": "规则"},
                {"role": "user", "content": "状态1"},
                {"role": "assistant", "content": "ACTION: play_card 0"},
                {"role": "user", "content": "状态2"},
                {"role": "assistant", "content": "ACTION: end_turn"},
            ],
            sample_id="human/RUN/battle-1",
            source="human_play",
            max_length=32,
        )


def test_encode_messages_rejects_template_prefix_drift() -> None:
    """模板边界漂移时拒绝构造可能错位的监督标签。

    Raises:
        AssertionError: 漂移没有触发明确的训练数据错误。

    Returns:
        None: 此测试只检查失败语义。
    """
    with pytest.raises(SftTrainingError, match="生成提示不是完整回答的前缀"):
        encode_messages(
            DriftedTokenizer(),
            [
                {"role": "user", "content": "状态"},
                {"role": "assistant", "content": "ACTION: end_turn"},
            ],
            sample_id="human/RUN/1",
            source="human_play",
            max_length=32,
        )
