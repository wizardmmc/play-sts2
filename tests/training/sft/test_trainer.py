"""验证 LoRA 优化、分块 loss、checkpoint 与 adapter 发布契约。"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from play_sts2.training.sft import (
    ChunkedCrossEntropy,
    MaskedTokenStats,
    OptimizationCheckpoint,
    SftTrainingError,
    TokenizedSample,
    adapter_source_files,
    attach_lora,
    chunked_masked_stats,
    optimize,
    publish_adapter,
    save_last_checkpoint,
)
from play_sts2.training.sft import trainer as trainer_module


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

    checkpoints: list[OptimizationCheckpoint] = []
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
    assert [checkpoint.optimizer_steps for checkpoint in checkpoints] == [1]
    assert not torch.equal(before, model.head.weight)
    trace = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    assert trace["samples"] == 2
    assert trace["loss"] > 0


def test_training_rejects_base_model_changed_after_fingerprint(
    tmp_path: Path,
) -> None:
    """训练清单不能把旧摘要绑定到随后被改写的基座。

    Args:
        tmp_path (Path): Pytest 提供的隔离模型目录。

    Raises:
        AssertionError: 同尺寸改写并恢复 mtime 后仍被视为同一基座。

    Returns:
        None: 此测试只覆盖训练前后的来源身份校验。
    """
    model_root = tmp_path / "base"
    model_root.mkdir()
    config = model_root / "config.json"
    weights = model_root / "model.safetensors"
    config.write_text("{}\n", encoding="utf-8")
    weights.write_text("before\n", encoding="utf-8")
    files = trainer_module.model_source_files(model_root)
    snapshot = trainer_module.snapshot_files(files)
    previous = weights.stat()
    weights.write_text("after!\n", encoding="utf-8")
    trainer_module.os.utime(
        weights,
        ns=(previous.st_atime_ns, previous.st_mtime_ns),
    )

    with pytest.raises(SftTrainingError, match="训练基座发生变化"):
        trainer_module._require_unchanged_model_sources(
            model_root,
            files,
            snapshot,
        )


def test_training_rejects_dataset_changed_after_fingerprint(tmp_path: Path) -> None:
    """数据分卷加载期间变化时不得继续使用旧来源声明。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 数据被替换后仍通过加载字节一致性检查。

    Returns:
        None: 此测试只覆盖通用来源快照校验。
    """
    train = tmp_path / "train.jsonl"
    train.write_text("before\n", encoding="utf-8")
    files = {"train.jsonl": train}
    snapshot = trainer_module.snapshot_files(files)
    train.write_text("after!\n", encoding="utf-8")

    with pytest.raises(SftTrainingError, match="训练数据集发生变化"):
        trainer_module._require_unchanged_sources(
            "训练数据集",
            files,
            files,
            snapshot,
        )


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


def test_optimize_rejects_non_fp32_trainable_parameters(tmp_path: Path) -> None:
    """LoRA 等可训练参数不是 FP32 时必须在创建优化器前失败。

    Args:
        tmp_path (Path): Pytest 提供的隔离日志目录。

    Raises:
        AssertionError: BF16 可训练参数仍被训练循环接受。

    Returns:
        None: 此测试用一个标量参数验证训练精度硬边界。
    """
    torch = pytest.importorskip("torch")

    class NonFp32TrainableModel(torch.nn.Module):
        """模拟被错误保留为 BF16 的 LoRA 参数。"""

        def __init__(self, dtype: object = torch.bfloat16) -> None:
            """创建一个指定精度的可训练标量。

            Args:
                dtype (object): 标量参数的 PyTorch dtype。
            """
            super().__init__()
            self.adapter_weight = torch.nn.Parameter(torch.tensor(1.0, dtype=dtype))

        def forward(
            self,
            *,
            input_ids: object,
            attention_mask: object,
            labels: object,
            use_cache: bool,
        ) -> SimpleNamespace:
            """返回与输入无关但可反传的标量损失。

            Args:
                input_ids (object): 未参与该精度测试的 token。
                attention_mask (object): 未参与该精度测试的掩码。
                labels (object): 未参与该精度测试的标签。
                use_cache (bool): 训练时是否使用 KV cache。

            Returns:
                SimpleNamespace: 含可反传 loss 的模型输出。
            """
            assert input_ids is not None
            assert attention_mask is not None
            assert labels is not None
            assert use_cache is False
            return SimpleNamespace(loss=self.adapter_weight.float().square())

    with pytest.raises(SftTrainingError, match="FP32"):
        optimize(
            NonFp32TrainableModel(),
            [TokenizedSample("a", "human_play", (0, 1), (-100, 1))],
            device="cpu",
            epochs=1,
            learning_rate=0.1,
            gradient_accumulation_steps=1,
            seed=7,
            trace_path=tmp_path / "metrics.jsonl",
        )

    class CastingOnMoveModel(NonFp32TrainableModel):
        """模拟在最后一次设备迁移时意外降低 adapter 精度的模型。"""

        def __init__(self) -> None:
            """先以合法 FP32 创建可训练参数。"""
            super().__init__(torch.float32)

        def to(self, device: object) -> "CastingOnMoveModel":
            """迁移设备时故意把参数降为 BF16。

            Args:
                device (object): 目标 PyTorch 设备。

            Returns:
                CastingOnMoveModel: 已完成错误降精度的自身。
            """
            super().to(device=device, dtype=torch.bfloat16)
            return self

    with pytest.raises(SftTrainingError, match="FP32"):
        optimize(
            CastingOnMoveModel(),
            [TokenizedSample("a", "human_play", (0, 1), (-100, 1))],
            device="cpu",
            epochs=1,
            learning_rate=0.1,
            gradient_accumulation_steps=1,
            seed=7,
            trace_path=tmp_path / "cast-on-move.jsonl",
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
        build_base().to(dtype=torch.bfloat16),
        rank=2,
        alpha=4,
        seed=7,
        target_modules=("q_proj",),
    )
    torch.manual_seed(222)
    second = attach_lora(
        build_base().to(dtype=torch.bfloat16),
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
    assert first_lora.dtype == torch.float32


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


def test_optimize_exact_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    """恢复优化器、洗牌位置与随机状态后应复现不中断训练。

    Args:
        tmp_path (Path): Pytest 提供的隔离日志目录。

    Raises:
        AssertionError: 恢复后的步数、轨迹或最终权重与连续训练不同。

    Returns:
        None: 此测试使用微型 CPU 模型验证精确续训。
    """
    torch = pytest.importorskip("torch")

    class TinyCausalModel(torch.nn.Module):
        """提供确定性的微型因果语言模型。"""

        def __init__(self) -> None:
            """创建嵌入层与词表投影层。"""
            super().__init__()
            self.embedding = torch.nn.Embedding(5, 4)
            self.head = torch.nn.Linear(4, 5)

        def forward(
            self,
            *,
            input_ids: object,
            attention_mask: object,
            labels: object,
            use_cache: bool,
        ) -> SimpleNamespace:
            """计算标准右移一位的交叉熵。

            Args:
                input_ids (object): 输入 token 张量。
                attention_mask (object): 有效 token 掩码。
                labels (object): 带忽略位的监督标签。
                use_cache (bool): 训练时是否使用 KV cache。

            Returns:
                SimpleNamespace: 含标量 loss 的模型输出。
            """
            assert attention_mask is not None
            assert use_cache is False
            logits = self.head(self.embedding(input_ids))
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 5),
                labels[:, 1:].reshape(-1),
            )
            return SimpleNamespace(loss=loss)

    samples = [
        TokenizedSample("a", "human_play", (0, 1, 2), (-100, 1, 2)),
        TokenizedSample("b", "human_play", (0, 2, 3), (-100, 2, 3)),
        TokenizedSample("c", "human_play", (0, 3, 4), (-100, 3, 4)),
    ]
    torch.manual_seed(101)
    uninterrupted = TinyCausalModel()
    full = optimize(
        uninterrupted,
        samples,
        device="cpu",
        epochs=2,
        learning_rate=0.05,
        gradient_accumulation_steps=1,
        seed=7,
        trace_path=tmp_path / "full.jsonl",
    )

    torch.manual_seed(101)
    resumed = TinyCausalModel()
    checkpoints: list[OptimizationCheckpoint] = []
    checkpoint_weights: list[dict[str, object]] = []

    def capture_checkpoint(checkpoint: OptimizationCheckpoint) -> None:
        """同时保存优化状态和对应的微型模型权重。

        Args:
            checkpoint (OptimizationCheckpoint): 第 2 步的完整优化状态。

        Returns:
            None: 权重与优化状态副本保存完成后返回。
        """
        checkpoints.append(copy.deepcopy(checkpoint))
        checkpoint_weights.append(
            {
                name: tensor.detach().clone()
                for name, tensor in resumed.state_dict().items()
            }
        )

    first = optimize(
        resumed,
        samples,
        device="cpu",
        epochs=2,
        learning_rate=0.05,
        gradient_accumulation_steps=1,
        seed=7,
        trace_path=tmp_path / "resumed.jsonl",
        checkpoint_steps=2,
        checkpoint=capture_checkpoint,
        max_steps=3,
    )
    resumed.load_state_dict(checkpoint_weights[-1])
    second = optimize(
        resumed,
        samples,
        device="cpu",
        epochs=2,
        learning_rate=0.05,
        gradient_accumulation_steps=1,
        seed=7,
        trace_path=tmp_path / "resumed.jsonl",
        resume_from=checkpoints[-1],
    )

    assert first.optimizer_steps == 3
    assert second == full
    assert [
        json.loads(line)["step"]
        for line in (tmp_path / "resumed.jsonl").read_text().splitlines()
    ] == [1, 2, 3, 4, 5, 6]
    for full_parameter, resumed_parameter in zip(
        uninterrupted.parameters(), resumed.parameters(), strict=True
    ):
        assert torch.equal(full_parameter, resumed_parameter)


@pytest.mark.parametrize(
    "content",
    (
        '{"step": 1}\n',
        '{"step": 1}\nnot-json\n',
        '{"step": 1}\n{"step": 3}\n',
    ),
)
def test_exact_resume_rejects_missing_or_malformed_trace_prefix(
    tmp_path: Path,
    content: str,
) -> None:
    """checkpoint 之前的指标行缺失、损坏或错序时必须拒绝恢复。

    Args:
        tmp_path (Path): Pytest 提供的隔离日志目录。
        content (str): 模拟的损坏指标轨迹。

    Raises:
        AssertionError: 无法证明对应权重的指标前缀仍被接受。

    Returns:
        None: 此测试只检查精确恢复的指标前缀。
    """
    trace = tmp_path / "metrics.jsonl"
    trace.write_text(content, encoding="utf-8")

    with pytest.raises(SftTrainingError, match="指标轨迹"):
        trainer_module._prepare_resume_trace(trace, 2, True)


def test_exact_resume_rejects_changed_resolved_device_or_dtype() -> None:
    """配置同为 auto 时也必须拒绝跨设备或精度的伪精确恢复。

    Raises:
        AssertionError: 已解析运行时不同仍被视为精确恢复。

    Returns:
        None: 此测试只检查运行时血缘比较。
    """
    with pytest.raises(SftTrainingError, match="设备或精度"):
        trainer_module._validate_execution_runtime(
            {
                "device": "mps",
                "base_model_dtype": "bfloat16",
                "adapter_dtype": "float32",
            },
            {
                "device": "cpu",
                "base_model_dtype": "float32",
                "adapter_dtype": "float32",
            },
        )


def test_training_run_lock_rejects_concurrent_same_name(tmp_path: Path) -> None:
    """同名训练运行持锁期间第二个进程入口必须立即失败。

    Args:
        tmp_path (Path): Pytest 提供的隔离 runs 目录。

    Raises:
        AssertionError: 同名运行可以同时取得排他锁。

    Returns:
        None: 此测试只检查同名运行的并发保护。
    """
    with (
        trainer_module._exclusive_run_lock(tmp_path, "20260828-e3"),
        pytest.raises(SftTrainingError, match="正在运行"),
        trainer_module._exclusive_run_lock(tmp_path, "20260828-e3"),
    ):
        pytest.fail("同名运行不应取得第二把锁")


def test_adapter_fingerprint_excludes_unused_template_files(tmp_path: Path) -> None:
    """父 adapter 指纹只锁定实际加载的权重和 PEFT 配置。

    Args:
        tmp_path (Path): Pytest 提供的隔离 adapter 目录。

    Raises:
        AssertionError: tokenizer 或 chat template 被错误纳入父 adapter 硬锁。

    Returns:
        None: 此测试只检查来源文件选择。
    """
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "chat_template.jinja",
        "tokenizer.json",
    ):
        (adapter / name).write_text(name, encoding="utf-8")

    assert set(adapter_source_files(adapter)) == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }


def test_training_run_name_requires_date_prefix() -> None:
    """训练目录名必须以八位日期开头，保证按名称即可按时间排序。

    Raises:
        AssertionError: 后缀日期仍被接受，或合法日期前缀被拒绝。

    Returns:
        None: 此测试只检查运行名契约。
    """
    trainer_module._validate_run_name("20260828-e3-smoke")
    with pytest.raises(SftTrainingError, match="YYYYMMDD"):
        trainer_module._validate_run_name("e3-smoke-20260828")


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

    save_last_checkpoint(FakeCheckpointModel(), destination, {"optimizer_steps": 1})
    save_last_checkpoint(FakeCheckpointModel(), destination, {"optimizer_steps": 2})

    assert destination.is_symlink()
    assert (destination / "adapter_model.safetensors").read_text() == "new"
    state = pytest.importorskip("torch").load(
        destination / "training_state.pt",
        weights_only=True,
    )
    assert state["optimizer_steps"] == 2
    assert not (destination.parent / ".checkpoint-last-previous").exists()


def test_save_last_checkpoint_keeps_generation_for_relative_run_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """相对 runs 根目录不得因路径表示不同而删除刚发布的 checkpoint。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。
        monkeypatch (pytest.MonkeyPatch): 用于把工作目录切到隔离目录。

    Raises:
        AssertionError: 固定符号链接指向的当前代被误当成旧代清理。

    Returns:
        None: 此测试只覆盖真实配置使用相对路径的发布方式。
    """

    class RelativeCheckpointModel:
        """写出可辨认 adapter 文件的 checkpoint 模型替身。"""

        def save_pretrained(
            self,
            directory: Path,
            *,
            safe_serialization: bool,
        ) -> None:
            """在暂存目录写入最小权重文件。

            Args:
                directory (Path): checkpoint 暂存目录。
                safe_serialization (bool): 是否要求安全序列化格式。

            Returns:
                None: 标记文件写完后返回。
            """
            assert safe_serialization is True
            (Path(directory) / "adapter_model.safetensors").write_text(
                "relative",
                encoding="utf-8",
            )

    monkeypatch.chdir(tmp_path)
    destination = Path("runs/sft/relative/checkpoint-last")

    save_last_checkpoint(
        RelativeCheckpointModel(),
        destination,
        {"optimizer_steps": 1},
    )

    assert destination.is_dir()
    assert (destination / "adapter_model.safetensors").read_text() == "relative"
    assert (destination / "training_state.pt").is_file()
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
    save_last_checkpoint(MarkerModel("old"), destination, {"optimizer_steps": 1})

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
        save_last_checkpoint(MarkerModel("new"), destination, {"optimizer_steps": 2})

    assert (destination / "adapter_model.safetensors").read_text() == "old"
