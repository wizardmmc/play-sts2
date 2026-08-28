"""验证本地 MLX 模型的准备、服务和真实协议冒烟边界。"""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

_TEMPLATE_FINGERPRINTS = {
    "disabled": "a" * 64,
    "enabled": "b" * 64,
}


def test_prepare_model_converts_hf_model_to_mlx_8bit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """转换时使用独立视图修正 Qwen 类型，并生成 8-bit MLX 目录。

    Args:
        tmp_path (Path): Pytest 提供的隔离临时目录。
        monkeypatch (pytest.MonkeyPatch): 用于替换昂贵的外部转换进程。

    Raises:
        AssertionError: 转换参数、源文件保护或目标目录不符合约定。

    Returns:
        None: 此测试只验证模型准备边界。
    """
    from play_sts2.inference import local_model

    _stub_dynamic_template(local_model, monkeypatch)
    source = tmp_path / "merged"
    target = tmp_path / "serving/model-mlx-8bit"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_text"}),
        encoding="utf-8",
    )
    (source / "model.safetensors").write_bytes(b"weights")
    merge_manifest = _write_merge_manifest(source)
    (source / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 248044}),
        encoding="utf-8",
    )
    (source / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<|im_end|>"}),
        encoding="utf-8",
    )
    (source / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": 248044, "content": "<|endoftext|>"},
                    {"id": 248046, "content": "<|im_end|>"},
                ]
            }
        ),
        encoding="utf-8",
    )

    def convert(command: list[str], *, check: bool) -> None:
        """校验转换命令并创建最小生产等价输出。

        Args:
            command (list[str]): ``mlx_lm.convert`` 的完整命令参数。
            check (bool): 子进程失败时是否应抛出异常。

        Raises:
            AssertionError: 命令没有启用约定的 MLX 8-bit 参数。

        Returns:
            None: 此替身把转换结果写入命令指定目录。
        """
        assert check is True
        assert command[1:4] == ["-m", "mlx_lm", "convert"]
        assert command[-5:] == ["-q", "--q-bits", "8", "--q-group-size", "64"]
        hf_dir = Path(command[command.index("--hf-path") + 1])
        mlx_dir = Path(command[command.index("--mlx-path") + 1])
        converted_config = json.loads(
            (hf_dir / "config.json").read_text(encoding="utf-8")
        )
        assert converted_config["model_type"] == "qwen3_5"
        assert converted_config["eos_token_id"] == 248046
        converted_generation = json.loads(
            (hf_dir / "generation_config.json").read_text(encoding="utf-8")
        )
        assert converted_generation["eos_token_id"] == 248046
        assert (hf_dir / "model.safetensors").read_bytes() == b"weights"
        mlx_dir.mkdir()
        (mlx_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5",
                    "eos_token_id": 248046,
                    "quantization": {
                        "bits": 8,
                        "group_size": 64,
                        "mode": "affine",
                    },
                }
            ),
            encoding="utf-8",
        )
        (mlx_dir / "model.safetensors").write_bytes(b"quantized")

    monkeypatch.setattr(local_model.subprocess, "run", convert)

    result = local_model.prepare_model(source, target, artifact_id="demo-e3")

    assert result == target
    assert json.loads((source / "config.json").read_text(encoding="utf-8")) == {
        "model_type": "qwen3_5_text"
    }
    assert json.loads(
        (source / "generation_config.json").read_text(encoding="utf-8")
    ) == {"eos_token_id": 248044}
    assert (target / "model.safetensors").read_bytes() == b"quantized"
    assert json.loads(
        (target / "serving_manifest.json").read_text(encoding="utf-8")
    ) == {
        "schema_version": 1,
        "artifact_id": "demo-e3",
        "source_model": str(source.resolve()),
        "source_merge_manifest_sha256": hashlib.sha256(merge_manifest).hexdigest(),
        "quantization": {"bits": 8, "group_size": 64},
        "eos_token_id": 248046,
        "thinking_template_sha256": _TEMPLATE_FINGERPRINTS,
    }


def test_prepare_model_does_not_publish_failed_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """转换进程失败时不发布不完整目标目录。

    Args:
        tmp_path (Path): Pytest 提供的隔离临时目录。
        monkeypatch (pytest.MonkeyPatch): 用于模拟转换进程失败。

    Raises:
        AssertionError: 转换失败后仍留下目标目录。

    Returns:
        None: 此测试只验证转换的原子失败边界。
    """
    from play_sts2.inference import local_model

    _stub_dynamic_template(local_model, monkeypatch)
    source = tmp_path / "merged"
    target = tmp_path / "serving/model-mlx-8bit"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    _write_merge_manifest(source)
    (source / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<|im_end|>"}),
        encoding="utf-8",
    )
    (source / "tokenizer.json").write_text(
        json.dumps({"added_tokens": [{"id": 3, "content": "<|im_end|>"}]}),
        encoding="utf-8",
    )

    def fail(command: list[str], *, check: bool) -> None:
        """模拟 MLX 转换进程失败。

        Args:
            command (list[str]): ``mlx_lm convert`` 的完整命令参数。
            check (bool): 子进程失败时是否应抛出异常。

        Raises:
            subprocess.CalledProcessError: 始终模拟转换失败。

        Returns:
            None: 此替身不会生成模型文件。
        """
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(local_model.subprocess, "run", fail)

    with pytest.raises(subprocess.CalledProcessError):
        local_model.prepare_model(source, target, artifact_id="demo-e3")

    assert not target.exists()


def test_prepare_model_does_not_replace_destination_created_during_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """初检后竞态创建的空目录也不能被普通 rename 静默替换。

    Args:
        tmp_path (Path): Pytest 提供的隔离模型目录。
        monkeypatch (pytest.MonkeyPatch): 用于注入模拟竞态的转换进程。

    Returns:
        None: 此测试只检查转换产物的原子发布边界。
    """
    from play_sts2.inference import local_model

    _stub_dynamic_template(local_model, monkeypatch)
    source = tmp_path / "merged/demo-e3-merged"
    target = tmp_path / "serving/demo-e3-mlx-8bit"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    _write_merge_manifest(source)
    (source / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<|im_end|>"}),
        encoding="utf-8",
    )
    (source / "tokenizer.json").write_text(
        json.dumps({"added_tokens": [{"id": 3, "content": "<|im_end|>"}]}),
        encoding="utf-8",
    )

    def convert(command: list[str], *, check: bool) -> None:
        """模拟转换完成后由竞态方抢先创建目标目录。

        Args:
            command (list[str]): 待模拟执行的 MLX 转换命令。
            check (bool): 是否要求子进程成功。

        Raises:
            AssertionError: 调用方未要求检查转换进程结果。

        Returns:
            None: 此替身只构造转换结果与竞态目录。
        """
        assert check is True
        converted = Path(command[command.index("--mlx-path") + 1])
        converted.mkdir()
        (converted / "config.json").write_text(
            json.dumps(
                {
                    "eos_token_id": 3,
                    "quantization": {"bits": 8, "group_size": 64},
                }
            ),
            encoding="utf-8",
        )
        target.mkdir(parents=True)

    monkeypatch.setattr(local_model.subprocess, "run", convert)

    with pytest.raises(FileExistsError):
        local_model.prepare_model(source, target, artifact_id="demo-e3")

    assert target.is_dir()
    assert not (target / "serving_manifest.json").exists()


def test_serve_model_enables_prefix_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型服务默认监听 8900 并保留十个提示词缓存槽。

    Args:
        tmp_path (Path): Pytest 提供的隔离临时目录。
        monkeypatch (pytest.MonkeyPatch): 用于替换阻塞的模型服务进程。

    Raises:
        AssertionError: 服务命令遗漏模型、端口或缓存参数。

    Returns:
        None: 此测试只验证服务启动边界。
    """
    from play_sts2.inference import local_model

    _stub_dynamic_template(local_model, monkeypatch)
    source = tmp_path / "merged/demo-e3-merged"
    model_dir = tmp_path / "serving/demo-e3-mlx-8bit"
    source.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    merge_manifest = _write_merge_manifest(source)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "eos_token_id": 3,
                "quantization": {"bits": 8, "group_size": 64},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "serving_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_id": "demo-e3",
                "source_model": str(source.resolve()),
                "source_merge_manifest_sha256": hashlib.sha256(
                    merge_manifest
                ).hexdigest(),
                "quantization": {"bits": 8, "group_size": 64},
                "eos_token_id": 3,
                "thinking_template_sha256": _TEMPLATE_FINGERPRINTS,
            }
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    def run(command: list[str], *, check: bool) -> None:
        """记录即将执行的模型服务命令。

        Args:
            command (list[str]): ``mlx_lm.server`` 的完整命令参数。
            check (bool): 子进程失败时是否应抛出异常。

        Returns:
            None: 此替身不启动真实阻塞服务。
        """
        assert check is True
        commands.append(command)

    monkeypatch.setattr(local_model.subprocess, "run", run)
    monkeypatch.chdir(tmp_path)
    relative_model_dir = model_dir.relative_to(tmp_path)

    local_model.serve_model(
        relative_model_dir,
        artifact_id="demo-e3",
        merged_model=source,
    )

    assert commands == [
        [
            local_model.sys.executable,
            "-m",
            "mlx_lm",
            "server",
            "--model",
            str(model_dir.resolve()),
            "--port",
            "8900",
            "--prompt-cache-size",
            "10",
        ]
    ]


@pytest.mark.parametrize(
    ("artifact_id", "mutate_source", "message"),
    [
        ("wrong-e3", False, "artifact_id 不一致"),
        ("demo-e3", True, "merge_manifest 摘要不一致"),
    ],
)
def test_serve_model_rejects_stale_or_wrong_artifact_before_starting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_id: str,
    mutate_source: bool,
    message: str,
) -> None:
    """服务目录身份或合并来源变化时必须在启动子进程前失败。

    Args:
        tmp_path (Path): Pytest 提供的隔离模型目录。
        monkeypatch (pytest.MonkeyPatch): 用于拦截服务启动进程。
        artifact_id (str): 参数化的期望制品标识。
        mutate_source (bool): 是否篡改合并来源清单。
        message (str): 期望的身份校验错误文本。

    Returns:
        None: 此测试只检查服务启动前的制品身份门禁。
    """
    from play_sts2.inference import local_model

    source = tmp_path / "merged/demo-e3-merged"
    model_dir = tmp_path / "serving/demo-e3-mlx-8bit"
    source.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    merge_manifest = _write_merge_manifest(source)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "eos_token_id": 3,
                "quantization": {"bits": 8, "group_size": 64},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "serving_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_id": "demo-e3",
                "source_model": str(source.resolve()),
                "source_merge_manifest_sha256": hashlib.sha256(
                    merge_manifest
                ).hexdigest(),
                "quantization": {"bits": 8, "group_size": 64},
                "eos_token_id": 3,
                "thinking_template_sha256": _TEMPLATE_FINGERPRINTS,
            }
        ),
        encoding="utf-8",
    )
    if mutate_source:
        (source / "merge_manifest.json").write_text(
            '{"schema_version": 2}\n', encoding="utf-8"
        )

    def unexpected_run(_command: list[str], *, check: bool) -> None:
        """在身份校验失败后意外启动服务时立即失败。

        Args:
            _command (list[str]): 本不应执行的服务启动命令。
            check (bool): 子进程错误检查标记。

        Raises:
            AssertionError: 该替身被调用即表明身份门禁失效。

        Returns:
            None: 该替身始终抛出异常，不会正常返回。
        """
        raise AssertionError("身份校验失败时不应启动模型服务")

    monkeypatch.setattr(local_model.subprocess, "run", unexpected_run)

    with pytest.raises(local_model.ServingModelIdentityError, match=message):
        local_model.serve_model(
            model_dir,
            artifact_id=artifact_id,
            merged_model=source,
        )


def test_prepare_model_rejects_merge_manifest_from_different_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """目录被改名不能把 e2 合并结果重新贴成 e3 artifact。"""
    from play_sts2.inference import local_model

    source = tmp_path / "merged/demo-e3-merged"
    target = tmp_path / "serving/demo-e3-mlx-8bit"
    source.mkdir(parents=True)
    _write_merge_manifest(source, artifact_id="demo-e2")
    monkeypatch.setattr(
        local_model.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("血缘不符时不应启动转换"),
    )

    with pytest.raises(
        local_model.ServingModelIdentityError,
        match="adapter 与 artifact_id 不一致",
    ):
        local_model.prepare_model(source, target, artifact_id="demo-e3")


def test_serve_model_rejects_actual_quantization_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """清单声称 8-bit、实际 MLX config 为 4-bit 时拒绝启动。"""
    from play_sts2.inference import local_model

    source = tmp_path / "merged/demo-e3-merged"
    model_dir = tmp_path / "serving/demo-e3-mlx-8bit"
    source.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    merge_manifest = _write_merge_manifest(source)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "eos_token_id": 3,
                "quantization": {"bits": 4, "group_size": 64},
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "serving_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_id": "demo-e3",
                "source_model": str(source.resolve()),
                "source_merge_manifest_sha256": hashlib.sha256(
                    merge_manifest
                ).hexdigest(),
                "quantization": {"bits": 8, "group_size": 64},
                "eos_token_id": 3,
                "thinking_template_sha256": _TEMPLATE_FINGERPRINTS,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        local_model.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("量化配置不符时不应启动服务"),
    )

    with pytest.raises(
        local_model.ServingModelIdentityError,
        match="量化配置不一致",
    ):
        local_model.serve_model(
            model_dir,
            artifact_id="demo-e3",
            merged_model=source,
        )


def test_thinking_template_fingerprints_reject_static_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """true/false 渲染相同的静态 no-think 模板不能冒充双模式服务件。

    Args:
        tmp_path (Path): Pytest 提供的隔离模型目录。
        monkeypatch (pytest.MonkeyPatch): 用于注入静态 tokenizer。

    Returns:
        None: 此测试只检查 thinking 模板的功能指纹。
    """
    from play_sts2.inference import local_model

    class StaticTokenizer:
        """模拟不响应 thinking 开关的静态 tokenizer。"""

        has_thinking = True
        think_start = "<think>"
        think_end = "</think>"

        def apply_chat_template(self, *_args: object, **kwargs: object) -> object:
            """无论 thinking 开关如何都返回同一模板和 token。

            Args:
                *_args (object): 未使用的对话模板位置参数。
                **kwargs (object): 包含 ``tokenize`` 的模板选项。

            Returns:
                object: 固定的模板文本或 token 序列。
            """
            if kwargs.get("tokenize") is True:
                return [1, 2, 3]
            return "<think>\n\n</think>\n"

    monkeypatch.setattr(
        local_model,
        "_load_tokenizer",
        lambda _path: StaticTokenizer(),
        raising=False,
    )

    with pytest.raises(
        local_model.ServingModelIdentityError,
        match="enable_thinking 没有改变模板渲染",
    ):
        local_model._thinking_template_fingerprints(tmp_path)


def test_thinking_template_fingerprints_include_token_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """文本模板相同但 tokenizer 映射变化时，功能指纹也必须变化。

    Args:
        tmp_path (Path): Pytest 提供的隔离模型目录。
        monkeypatch (pytest.MonkeyPatch): 用于注入可变 tokenizer。

    Returns:
        None: 此测试只检查 token ID 是否进入功能指纹。
    """
    from play_sts2.inference import local_model

    class DynamicTokenizer:
        """模拟文本分支稳定但 token 映射可变化的 tokenizer。"""

        has_thinking = True
        think_start = "<think>"
        think_end = "</think>"

        def __init__(self, offset: int) -> None:
            """保存用于构造 token 指纹的偏移量。

            Args:
                offset (int): 注入 token 序列的可变偏移量。

            Returns:
                None: 此方法只初始化测试 tokenizer。
            """
            self.offset = offset

        def apply_chat_template(self, *_args: object, **kwargs: object) -> object:
            """按 thinking 开关和偏移量返回模板或 token。

            Args:
                *_args (object): 未使用的对话模板位置参数。
                **kwargs (object): 包含 thinking 和 tokenize 开关的模板选项。

            Returns:
                object: 对应开关的模板文本或带偏移的 token 序列。
            """
            enabled = kwargs["enable_thinking"] is True
            if kwargs.get("tokenize") is True:
                return [self.offset, 1 if enabled else 2]
            return "<think>\n" if enabled else "<think>\n\n</think>\n"

    tokenizer = DynamicTokenizer(10)
    monkeypatch.setattr(local_model, "_load_tokenizer", lambda _path: tokenizer)
    first = local_model._thinking_template_fingerprints(tmp_path)
    tokenizer.offset = 20

    second = local_model._thinking_template_fingerprints(tmp_path)

    assert first != second


def test_repository_qwen_template_has_distinct_thinking_branches() -> None:
    """本地真实 Qwen tokenizer 的动态模板满足 MLX 双模式契约。"""
    from play_sts2.inference import local_model

    model_dir = Path("models/base/qwen3.5-4b")
    if not model_dir.is_dir():
        pytest.skip("本地 Qwen 基座不存在")
    pytest.importorskip("mlx_lm")

    fingerprints = local_model._thinking_template_fingerprints(model_dir)

    assert fingerprints["disabled"] != fingerprints["enabled"]


@pytest.mark.parametrize(
    "source_sha256",
    [
        {
            "base/config.json": "not-a-sha256",
            "adapter/train_manifest.json": "b" * 64,
            "tokenizer/tokenizer.json": "c" * 64,
        },
        {
            "adapter/train_manifest.json": "b" * 64,
            "tokenizer/tokenizer.json": "c" * 64,
        },
    ],
)
def test_merge_identity_rejects_malformed_or_incomplete_source_digests(
    tmp_path: Path,
    source_sha256: dict[str, str],
) -> None:
    """merge 清单必须用真实 SHA-256 覆盖 base、adapter、tokenizer 三类来源。"""
    from play_sts2.inference import local_model

    source = tmp_path / "models/merged/demo-e3-merged"
    source.mkdir(parents=True)
    (source / "merge_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "adapter": str(tmp_path / "models/adapters/demo-e3"),
                "output": str(source.resolve()),
                "source_sha256": source_sha256,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(local_model.ServingModelIdentityError, match="来源摘要"):
        local_model._validate_merge_model_identity(source, "demo-e3")


def _write_merge_manifest(source: Path, *, artifact_id: str = "demo-e3") -> bytes:
    """写入当前 SFT merge 发布的最小身份字段。"""
    payload = {
        "schema_version": 1,
        "adapter": str(source.parent.parent / "adapters" / artifact_id),
        "output": str(source.resolve()),
        "tokenizer_source": str(source.parent.parent / "adapters" / artifact_id),
        "source_sha256": {
            "base/config.json": "a" * 64,
            "adapter/train_manifest.json": "b" * 64,
            "tokenizer/tokenizer.json": "c" * 64,
        },
    }
    content = (json.dumps(payload, sort_keys=True) + "\n").encode()
    (source / "merge_manifest.json").write_bytes(content)
    return content


def _stub_dynamic_template(
    local_model: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """让非 tokenizer 单元测试聚焦各自边界。"""
    monkeypatch.setattr(
        local_model,
        "_thinking_template_fingerprints",
        lambda _path: dict(_TEMPLATE_FINGERPRINTS),
        raising=False,
    )
