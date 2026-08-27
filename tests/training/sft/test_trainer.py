"""验证 LoRA 优化、分块 loss、checkpoint 与 adapter 发布契约。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from play_sts2.training.sft import (
    ChunkedCrossEntropy,
    MaskedTokenStats,
    SftTrainingError,
    TokenizedSample,
    attach_lora,
    chunked_masked_stats,
    optimize,
    publish_adapter,
    save_last_checkpoint,
)


def test_optimize_updates_a_tiny_causal_model(tmp_path: Path) -> None:
    """核心训练循环执行梯度累积并实际更新可训练参数。

    Args:
        tmp_path (Path): Pytest 提供的隔离日志目录。

    Raises:
        AssertionError: 优化步数、日志或参数更新不符合约定。

    Returns:
        None: 此测试只运行一个微型 PyTorch 模型。
    """
    torch = pytest.importorskip("torch")

    class TinyCausalModel(torch.nn.Module):
        """提供与 Transformers 因果语言模型相同的 loss 接口。"""

        def __init__(self) -> None:
            """创建一个四 token 词表的线性语言模型。"""
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 4)
            self.head = torch.nn.Linear(4, 4)

        def forward(
            self,
            *,
            input_ids: object,
            attention_mask: object,
            labels: object,
            use_cache: bool,
        ) -> SimpleNamespace:
            """计算标准向后错一位的交叉熵。

            Args:
                input_ids (object): 输入 token 张量。
                attention_mask (object): 当前样本的有效 token 掩码。
                labels (object): 含 ``-100`` 忽略位的监督标签。
                use_cache (bool): 训练时是否使用 KV cache。

            Returns:
                SimpleNamespace: 含标量 ``loss`` 的模型输出。
            """
            assert attention_mask is not None
            assert use_cache is False
            hidden = self.embedding(input_ids)
            logits = self.head(hidden)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 4),
                labels[:, 1:].reshape(-1),
            )
            return SimpleNamespace(loss=loss)

    model = TinyCausalModel()
    before = model.head.weight.detach().clone()
    samples = [
        TokenizedSample("a", "human_play", (0, 1, 2), (-100, 1, 2)),
        TokenizedSample("b", "web_wiki", (0, 2, 3), (-100, 2, 3)),
    ]

    checkpoints: list[int] = []
    result = optimize(
        model,
        samples,
        device="cpu",
        epochs=1,
        learning_rate=0.1,
        gradient_accumulation_steps=2,
        seed=7,
        trace_path=tmp_path / "metrics.jsonl",
        checkpoint_steps=1,
        checkpoint=checkpoints.append,
    )

    assert result.optimizer_steps == 1
    assert result.samples_seen == 2
    assert checkpoints == [1]
    assert not torch.equal(before, model.head.weight)
    trace = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    assert trace["samples"] == 2
    assert trace["loss"] > 0


def test_chunked_cross_entropy_matches_masked_teacher_forced_stats() -> None:
    """分块训练 loss 与 teacher-forced 评估共享相同的移位和掩码口径。

    Raises:
        AssertionError: 分块边界、监督 token 数或指标计算发生漂移。

    Returns:
        None: 此测试只运行一个微型词表投影层。
    """
    torch = pytest.importorskip("torch")
    head = torch.nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(3))
    hidden = torch.tensor(
        [[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 1.0]]],
        requires_grad=True,
    )
    input_ids = torch.tensor([[0, 1, 2, 1]])
    labels = torch.tensor([[-100, 1, -100, 1]])

    loss = ChunkedCrossEntropy(head, chunk_size=2)(hidden, input_ids, labels)
    stats = chunked_masked_stats(head, hidden.detach(), input_ids, labels, chunk_size=2)

    assert stats == MaskedTokenStats(
        loss_sum=pytest.approx(float(loss.detach()) * 2),
        correct=0,
        tokens=2,
    )
    loss.backward()
    assert hidden.grad is not None


def test_optimize_rejects_non_positive_step_limit(tmp_path: Path) -> None:
    """零步或负数上限不会意外执行一次参数更新。

    Args:
        tmp_path (Path): Pytest 提供的隔离日志目录。

    Raises:
        AssertionError: 非正步数没有在训练前被拒绝。

    Returns:
        None: 此测试不需要构造真实模型。
    """
    with pytest.raises(SftTrainingError, match="max_steps 必须为正数"):
        optimize(
            object(),
            [TokenizedSample("a", "human_play", (0, 1), (-100, 1))],
            device="cpu",
            epochs=1,
            learning_rate=0.1,
            gradient_accumulation_steps=1,
            seed=7,
            trace_path=tmp_path / "metrics.jsonl",
            max_steps=0,
        )


def test_attach_lora_seeds_adapter_initialization() -> None:
    """相同训练 seed 不受调用前全局随机状态影响。

    Raises:
        AssertionError: 两次 LoRA A 的初始值不同。

    Returns:
        None: 此测试使用一个微型 Llama 骨干。
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    def build_base() -> object:
        """用固定权重创建微型因果语言模型。

        Returns:
            object: 仅含一层注意力的 Llama 模型。
        """
        torch.manual_seed(101)
        return transformers.LlamaForCausalLM(
            transformers.LlamaConfig(
                vocab_size=16,
                max_position_embeddings=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=1,
                num_key_value_heads=1,
            )
        )

    torch.manual_seed(111)
    first = attach_lora(
        build_base(),
        rank=2,
        alpha=4,
        seed=7,
        target_modules=("q_proj",),
    )
    torch.manual_seed(222)
    second = attach_lora(
        build_base(),
        rank=2,
        alpha=4,
        seed=7,
        target_modules=("q_proj",),
    )
    first_lora = next(
        parameter for name, parameter in first.named_parameters() if "lora_A" in name
    )
    second_lora = next(
        parameter for name, parameter in second.named_parameters() if "lora_A" in name
    )

    assert torch.equal(first_lora, second_lora)


def test_publish_adapter_does_not_leave_partial_destination(tmp_path: Path) -> None:
    """tokenizer 保存失败时最终 adapter 目录保持不存在。

    Args:
        tmp_path (Path): Pytest 提供的隔离模型目录。

    Raises:
        AssertionError: 发布失败后仍留下最终目录。

    Returns:
        None: 此测试只检查原子发布边界。
    """

    class FakeModel:
        """模拟能够写出 adapter 权重的 PEFT 模型。"""

        def save_pretrained(
            self,
            directory: Path,
            *,
            safe_serialization: bool,
        ) -> None:
            """在暂存目录写入一个权重占位文件。

            Args:
                directory (Path): adapter 暂存目录。
                safe_serialization (bool): 是否要求 safetensors。

            Returns:
                None: 权重占位文件写入后返回。
            """
            assert safe_serialization is True
            Path(directory, "adapter_model.safetensors").write_bytes(b"weights")

    class FailingTokenizer:
        """模拟 tokenizer 文件保存失败。"""

        def save_pretrained(self, _directory: Path) -> None:
            """抛出预期的磁盘错误。

            Args:
                _directory (Path): adapter 暂存目录。

            Raises:
                OSError: 固定模拟保存失败。
            """
            raise OSError("disk full")

    destination = tmp_path / "adapters/demo"
    with pytest.raises(OSError, match="disk full"):
        publish_adapter(FakeModel(), FailingTokenizer(), destination, {"run": "demo"})

    assert not destination.exists()


def test_save_last_checkpoint_replaces_one_fixed_directory(tmp_path: Path) -> None:
    """周期保存原子切换固定链接，不按步数积累 adapter 副本。

    Args:
        tmp_path (Path): Pytest 提供的隔离运行目录。

    Raises:
        AssertionError: 旧 checkpoint、交换目录或新增权重不符合约定。

    Returns:
        None: 此测试只检查 checkpoint 的目录发布语义。
    """

    class FakeCheckpointModel:
        """写出可区分的新 checkpoint 内容。"""

        def save_pretrained(
            self,
            directory: Path,
            *,
            safe_serialization: bool,
        ) -> None:
            """把新权重标记写入暂存目录。

            Args:
                directory (Path): 自动创建的 checkpoint 暂存目录。
                safe_serialization (bool): 是否要求安全权重格式。

            Returns:
                None: 标记写入完成后返回。
            """
            assert safe_serialization is True
            Path(directory, "adapter_model.safetensors").write_text(
                "new",
                encoding="utf-8",
            )

    destination = tmp_path / "run/checkpoint-last"

    save_last_checkpoint(FakeCheckpointModel(), destination)
    save_last_checkpoint(FakeCheckpointModel(), destination)

    assert destination.is_symlink()
    assert (destination / "adapter_model.safetensors").read_text() == "new"
    assert not (destination.parent / ".checkpoint-last-previous").exists()
    assert len(list(destination.parent.iterdir())) == 2


def test_save_last_checkpoint_keeps_previous_target_during_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """指针原子替换失败时旧 checkpoint 始终可读。

    Args:
        tmp_path (Path): Pytest 提供的隔离运行目录。
        monkeypatch (pytest.MonkeyPatch): 用于模拟原子指针切换失败。

    Raises:
        AssertionError: 发布窗口令固定 checkpoint 消失或变成半成品。

    Returns:
        None: 此测试只检查失败时的可恢复性。
    """

    class MarkerModel:
        """写出调用方指定的 checkpoint 标记。"""

        def __init__(self, marker: str) -> None:
            """保存本次写入使用的标记。

            Args:
                marker (str): 权重占位内容。
            """
            self._marker = marker

        def save_pretrained(
            self,
            directory: Path,
            *,
            safe_serialization: bool,
        ) -> None:
            """把标记写入完整暂存目录。

            Args:
                directory (Path): checkpoint 暂存目录。
                safe_serialization (bool): 是否要求安全权重格式。

            Returns:
                None: 标记写入完成后返回。
            """
            assert safe_serialization is True
            Path(directory, "adapter_model.safetensors").write_text(
                self._marker,
                encoding="utf-8",
            )

    destination = tmp_path / "run/checkpoint-last"
    save_last_checkpoint(MarkerModel("old"), destination)

    def fail_replace(source: Path, target: Path) -> None:
        """确认旧链接仍可读取后模拟原子替换失败。

        Args:
            source (Path): 新 checkpoint 临时链接。
            target (Path): 固定 checkpoint 链接。

        Raises:
            OSError: 固定模拟发布失败。
        """
        assert Path(source).is_symlink()
        assert Path(target, "adapter_model.safetensors").read_text() == "old"
        raise OSError("simulated pointer failure")

    monkeypatch.setattr("play_sts2.training.sft.trainer.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated pointer failure"):
        save_last_checkpoint(MarkerModel("new"), destination)

    assert (destination / "adapter_model.safetensors").read_text() == "old"
