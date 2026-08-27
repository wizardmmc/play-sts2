"""验证 LoRA 合并产物的校验、追溯与原子发布。"""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from play_sts2.training.sft import SftConfig, SftTrainingError, attach_lora
from play_sts2.training.sft import merge as merge_module


class FakeArtifact:
    """模拟 Hugging Face 模型或 tokenizer 的保存接口。"""

    def __init__(self, filename: str, *, fail: bool = False) -> None:
        """保存测试产物名与可选失败开关。

        Args:
            filename (str): ``save_pretrained`` 创建的文件名。
            fail (bool): 为真时模拟保存中断。
        """
        self.filename = filename
        self.fail = fail

    def save_pretrained(self, destination: Path, **_kwargs: Any) -> None:
        """在目标目录写入一个小型占位产物。

        Args:
            destination (Path): 合并实现创建的暂存目录。
            **_kwargs (Any): 真实 Hugging Face 保存参数。

        Raises:
            OSError: 测试要求模拟写入失败时抛出。

        Returns:
            None: 占位文件写入后返回。
        """
        if self.fail:
            raise OSError("simulated save failure")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / self.filename).write_text("artifact\n", encoding="utf-8")


def test_merge_sft_adapter_publishes_traceable_default_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认目录包含完整模型、tokenizer 与来源哈希。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。
        monkeypatch (pytest.MonkeyPatch): 用于替换真实模型加载。

    Raises:
        AssertionError: 默认命名、文件保存或 manifest 不符合契约。

    Returns:
        None: 此测试不加载真实模型。
    """
    config, adapter = _merge_fixture(tmp_path)
    monkeypatch.setattr(
        merge_module,
        "_load_merged_artifacts",
        lambda _base, _adapter: (
            FakeArtifact("model.safetensors"),
            FakeArtifact("tokenizer.json"),
            {"transformers": "test", "peft": "test", "torch": "test"},
        ),
    )

    result = merge_module.merge_sft_adapter(config, adapter)

    destination = tmp_path / "models/merged/demo-merged"
    assert result["output"] == str(destination)
    assert (destination / "model.safetensors").is_file()
    assert (destination / "tokenizer.json").is_file()
    manifest = json.loads(
        (destination / "merge_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["merge"] == "peft.merge_and_unload(safe_merge=True)"
    assert manifest["base_model"] == str(config.base_model.resolve())
    assert manifest["adapter"] == str(adapter.resolve())
    assert manifest["tokenizer_source"] == str(config.base_model.resolve())
    assert set(manifest["source_sha256"]) == {
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "adapter/train_manifest.json",
        "base/config.json",
        "base/model.safetensors",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    }
    assert manifest["lineage"]["checks"] == ["declared_local_model"]
    assert not list(destination.parent.glob(".demo-merged-*"))


def test_merge_sft_adapter_refuses_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """已有目标不会被一次新合并静默覆盖。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。
        monkeypatch (pytest.MonkeyPatch): 用于确认拒绝发生在模型加载前。

    Raises:
        AssertionError: 合并实现加载了模型或改写已有目录。

    Returns:
        None: 此测试只验证破坏性操作边界。
    """
    config, adapter = _merge_fixture(tmp_path)
    destination = tmp_path / "models/merged/demo-merged"
    destination.mkdir(parents=True)
    marker = destination / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")
    monkeypatch.setattr(
        merge_module,
        "_load_merged_artifacts",
        lambda _base, _adapter: pytest.fail("不应加载模型"),
    )

    with pytest.raises(SftTrainingError, match="合并输出已存在"):
        merge_module.merge_sft_adapter(config, adapter)

    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_merge_sft_adapter_cleans_staging_after_save_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """保存中断不会留下看似可用的目标或暂存目录。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。
        monkeypatch (pytest.MonkeyPatch): 用于注入保存失败模型。

    Raises:
        AssertionError: 失败后仍残留半成品目录。

    Returns:
        None: 此测试只验证原子发布失败路径。
    """
    config, adapter = _merge_fixture(tmp_path)
    monkeypatch.setattr(
        merge_module,
        "_load_merged_artifacts",
        lambda _base, _adapter: (
            FakeArtifact("model.safetensors", fail=True),
            FakeArtifact("tokenizer.json"),
            {},
        ),
    )

    with pytest.raises(OSError, match="simulated save failure"):
        merge_module.merge_sft_adapter(config, adapter)

    merged_root = tmp_path / "models/merged"
    assert not (merged_root / "demo-merged").exists()
    assert not list(merged_root.glob(".demo-merged-*"))


def test_merge_sft_adapter_does_not_replace_late_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """发布瞬间出现的同名目录也不会被 rename 覆盖。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。
        monkeypatch (pytest.MonkeyPatch): 用于在发布前制造目标竞态。

    Raises:
        AssertionError: 竞态目录被覆盖、删除或留下暂存目录。

    Returns:
        None: 此测试覆盖初始 exists 检查后的竞态窗口。
    """
    config, adapter = _merge_fixture(tmp_path)
    destination = tmp_path / "models/merged/demo-merged"
    monkeypatch.setattr(
        merge_module,
        "_load_merged_artifacts",
        lambda _base, _adapter: (
            FakeArtifact("model.safetensors"),
            FakeArtifact("tokenizer.json"),
            {},
        ),
    )
    original_write_json = merge_module._write_json

    def create_racing_destination(path: Path, value: dict[str, Any]) -> None:
        """写完清单后模拟另一进程抢先创建最终目录。

        Args:
            path (Path): 合并清单路径。
            value (dict[str, Any]): 合并清单内容。

        Returns:
            None: 清单和竞态标记写入后返回。
        """
        original_write_json(path, value)
        destination.mkdir()
        (destination / "keep.txt").write_text("keep\n", encoding="utf-8")

    monkeypatch.setattr(merge_module, "_write_json", create_racing_destination)

    with pytest.raises(SftTrainingError, match="合并输出已存在"):
        merge_module.merge_sft_adapter(config, adapter)

    assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep\n"
    assert not list(destination.parent.glob(".demo-merged-*"))


def test_merge_sft_adapter_rejects_wrong_declared_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同架构但不同权重的基座不能与旧 adapter 静默合并。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。
        monkeypatch (pytest.MonkeyPatch): 用于确认拒绝发生在模型加载前。

    Raises:
        AssertionError: 不同基座通过血缘校验或触发了模型加载。

    Returns:
        None: 此测试使用旧清单的本地基座回溯路径。
    """
    config, adapter = _merge_fixture(tmp_path)
    wrong_base = tmp_path / "models/base/wrong"
    wrong_base.mkdir()
    (wrong_base / "config.json").write_text("{}\n", encoding="utf-8")
    (wrong_base / "model.safetensors").write_text(
        "different weights\n",
        encoding="utf-8",
    )
    _write_fixture_json(
        adapter / "adapter_config.json",
        {"base_model_name_or_path": str(wrong_base)},
    )
    _write_fixture_json(
        adapter / "train_manifest.json", {"base_model": str(wrong_base)}
    )
    monkeypatch.setattr(
        merge_module,
        "_load_merged_artifacts",
        lambda _base, _adapter: pytest.fail("不应加载模型"),
    )

    with pytest.raises(SftTrainingError, match="基座血缘不一致"):
        merge_module.merge_sft_adapter(config, adapter)


def test_merge_sft_adapter_accepts_training_manifest_base_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """新训练清单可用完整基座指纹证明已移动模型的血缘。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。
        monkeypatch (pytest.MonkeyPatch): 用于替换真实模型加载。

    Raises:
        AssertionError: 有效指纹未通过或未记录指纹校验方式。

    Returns:
        None: 此测试模拟原训练路径已经不可访问。
    """
    config, adapter = _merge_fixture(tmp_path)
    hashes = merge_module.sha256_files(
        merge_module.model_source_files(config.base_model)
    )
    _write_fixture_json(
        adapter / "adapter_config.json",
        {"base_model_name_or_path": "Qwen/moved-base"},
    )
    _write_fixture_json(
        adapter / "train_manifest.json",
        {
            "base_model": "missing/original-base",
            "base_model_sha256": hashes,
        },
    )
    monkeypatch.setattr(
        merge_module,
        "_load_merged_artifacts",
        lambda _base, _adapter: (
            FakeArtifact("model.safetensors"),
            FakeArtifact("tokenizer.json"),
            {},
        ),
    )

    merge_module.merge_sft_adapter(config, adapter)

    manifest = json.loads(
        (tmp_path / "models/merged/demo-merged/merge_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["lineage"]["checks"] == ["train_manifest_sha256"]


def test_merge_sft_adapter_rejects_sources_changed_during_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """合并期间发生变化的输入不能发布带过时摘要的模型。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。
        monkeypatch (pytest.MonkeyPatch): 用于在发布前改写来源权重。

    Raises:
        AssertionError: 变化后的来源仍被发布或留下暂存目录。

    Returns:
        None: 此测试覆盖摘要计算后的来源变更窗口。
    """
    config, adapter = _merge_fixture(tmp_path)
    source = adapter / "adapter_model.safetensors"
    source_stat = source.stat()
    monkeypatch.setattr(
        merge_module,
        "_load_merged_artifacts",
        lambda _base, _adapter: (
            FakeArtifact("model.safetensors"),
            FakeArtifact("tokenizer.json"),
            {},
        ),
    )
    original_write_json = merge_module._write_json

    def mutate_source(path: Path, value: dict[str, Any]) -> None:
        """写完清单后改写已取摘要的 adapter 权重。

        Args:
            path (Path): 合并清单路径。
            value (dict[str, Any]): 合并清单内容。

        Returns:
            None: 清单和来源改写完成后返回。
        """
        original_write_json(path, value)
        source.write_text("changed\n", encoding="utf-8")
        os.utime(
            source,
            ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
        )

    monkeypatch.setattr(merge_module, "_write_json", mutate_source)

    with pytest.raises(SftTrainingError, match="合并期间发生变化"):
        merge_module.merge_sft_adapter(config, adapter)

    merged_root = tmp_path / "models/merged"
    assert not (merged_root / "demo-merged").exists()
    assert not list(merged_root.glob(".demo-merged-*"))


def test_linux_publish_fails_closed_without_exclusive_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux 缺少排他 rename 时不会退回有竞态的普通 rename。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。
        monkeypatch (pytest.MonkeyPatch): 用于模拟 libc 不导出 ``renameat2``。

    Raises:
        AssertionError: 实现退回普通 rename 或移动了暂存目录。

    Returns:
        None: 此测试不依赖当前宿主操作系统。
    """
    source = tmp_path / "staging"
    destination = tmp_path / "published"
    source.mkdir()
    monkeypatch.setattr(merge_module.sys, "platform", "linux")
    monkeypatch.setattr(
        merge_module,
        "_renameat2_no_replace",
        lambda _source, _destination: False,
    )

    with pytest.raises(SftTrainingError, match="不支持排他原子发布"):
        merge_module._publish_directory_no_replace(source, destination)

    assert source.is_dir()
    assert not destination.exists()


def test_load_merged_artifacts_merges_real_tiny_lora(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实 Transformers 与 PEFT 调用能把 LoRA 增量并入基座权重。

    Args:
        tmp_path (Path): Pytest 提供的隔离模型目录。
        monkeypatch (pytest.MonkeyPatch): 用于以轻量 tokenizer 替代词表资产。

    Raises:
        AssertionError: 合并调用失败、仍残留 PEFT 包装或权重没有变化。

    Returns:
        None: 此测试使用单层微型 Llama，不下载外部模型。
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    base = tmp_path / "tiny-base"
    adapter = tmp_path / "tiny-adapter"
    model = transformers.LlamaForCausalLM(
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
    model.save_pretrained(base)
    original = model.model.layers[0].self_attn.q_proj.weight.detach().to(torch.bfloat16)
    model = transformers.LlamaForCausalLM.from_pretrained(base)
    peft_model = attach_lora(
        model,
        rank=2,
        alpha=4,
        seed=7,
        target_modules=("q_proj",),
    )
    with torch.no_grad():
        for name, parameter in peft_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                parameter.fill_(0.25)
    peft_model.save_pretrained(adapter)

    tokenizer = object()

    def load_stub_tokenizer(
        _cls: type[object],
        *_args: Any,
        **_kwargs: Any,
    ) -> object:
        """忽略真实 tokenizer 参数并返回固定对象。

        Args:
            _cls (type[object]): 被替换类方法接收的 tokenizer 类。
            *_args (Any): Transformers 位置参数。
            **_kwargs (Any): Transformers 关键字参数。

        Returns:
            object: 测试使用的 tokenizer 占位对象。
        """
        return tokenizer

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(load_stub_tokenizer),
    )

    merged, actual_tokenizer, versions = merge_module._load_merged_artifacts(
        base,
        adapter,
    )

    merged_weight = merged.model.layers[0].self_attn.q_proj.weight.detach()
    assert actual_tokenizer is tokenizer
    assert not hasattr(merged.model.layers[0].self_attn.q_proj, "lora_A")
    assert not torch.equal(original, merged_weight)
    assert {"peft", "torch", "transformers"} <= versions.keys()


def _merge_fixture(tmp_path: Path) -> tuple[SftConfig, Path]:
    """创建含最小来源文件的合并测试目录。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Returns:
        tuple[SftConfig, Path]: 可用配置和 adapter 目录。
    """
    base = tmp_path / "models/base/qwen"
    adapter = tmp_path / "models/adapters/demo"
    base.mkdir(parents=True)
    adapter.mkdir(parents=True)
    (base / "config.json").write_text("{}\n", encoding="utf-8")
    (base / "model.safetensors").write_text("base weights\n", encoding="utf-8")
    (base / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (base / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    _write_fixture_json(
        adapter / "adapter_config.json",
        {"base_model_name_or_path": str(base)},
    )
    (adapter / "adapter_model.safetensors").write_text("weights\n", encoding="utf-8")
    _write_fixture_json(adapter / "train_manifest.json", {"base_model": str(base)})
    config = SftConfig(
        base_model=base,
        dataset_root=tmp_path / "dataset",
        adapter_root=tmp_path / "models/adapters",
        runs_root=tmp_path / "runs",
        device="cpu",
        epochs=1,
        learning_rate=1e-4,
        max_length=1024,
        gradient_accumulation_steps=1,
        seed=1,
        lora_rank=16,
        lora_alpha=32,
    )
    return config, adapter


def _write_fixture_json(path: Path, value: dict[str, Any]) -> None:
    """写入合并测试使用的最小 JSON 文件。

    Args:
        path (Path): 目标文件路径。
        value (dict[str, Any]): 可序列化的测试值。

    Returns:
        None: JSON 写入完成后返回。
    """
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
