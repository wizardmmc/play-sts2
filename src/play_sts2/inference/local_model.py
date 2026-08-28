"""准备并验证供本机 Apple Silicon 使用的 MLX 模型。"""

import ctypes
import errno
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

DEFAULT_MODEL_PORT = 8900
DEFAULT_PROMPT_CACHE_SIZE = 10
SERVING_MANIFEST_NAME = "serving_manifest.json"
_MERGE_MANIFEST_NAME = "merge_manifest.json"


class ServingModelIdentityError(RuntimeError):
    """表示 MLX 服务件无法证明来自配置指定的合并模型。"""


def prepare_model(source: Path, target: Path, *, artifact_id: str) -> Path:
    """把合并后的 Hugging Face 模型转换为 MLX 8-bit 模型。

    Args:
        source (Path): 合并后的 Hugging Face 模型目录。
        target (Path): 转换完成后保存 MLX 模型的目录。
        artifact_id (str): 配置声明的不可变模型产物标识。

    Raises:
        FileExistsError: 目标目录已经存在。
        subprocess.CalledProcessError: ``mlx_lm.convert`` 转换失败。

    Returns:
        Path: 转换完成的目标目录。
    """
    if target.exists():
        raise FileExistsError(target)

    source = source.resolve()
    merge_manifest_path = source / _MERGE_MANIFEST_NAME
    if not merge_manifest_path.is_file():
        raise ServingModelIdentityError(
            f"合并模型缺少 {_MERGE_MANIFEST_NAME}: {source}"
        )
    _validate_merge_model_identity(source, artifact_id)
    merge_manifest_sha256 = _sha256(merge_manifest_path)
    thinking_template_sha256 = _thinking_template_fingerprints(source)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") == "qwen3_5_text":
        config["model_type"] = "qwen3_5"
    eos_token_id = _eos_token_id(source)
    if eos_token_id is not None:
        config["eos_token_id"] = eos_token_id

    generation_config_path = source / "generation_config.json"
    generation_config = None
    if generation_config_path.is_file():
        generation_config = json.loads(
            generation_config_path.read_text(encoding="utf-8")
        )
        if eos_token_id is not None:
            generation_config["eos_token_id"] = eos_token_id

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}-",
        dir=target.parent,
    ) as temporary:
        temporary_dir = Path(temporary)
        hf_view = temporary_dir / "hf"
        converted = temporary_dir / "mlx"
        hf_view.mkdir()
        replaced_files = {"config.json"}
        if generation_config is not None:
            replaced_files.add("generation_config.json")
        _link_model_files(source, hf_view, replaced_files)
        _write_json(hf_view / "config.json", config)
        if generation_config is not None:
            _write_json(hf_view / "generation_config.json", generation_config)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mlx_lm",
                "convert",
                "--hf-path",
                str(hf_view),
                "--mlx-path",
                str(converted),
                "-q",
                "--q-bits",
                "8",
                "--q-group-size",
                "64",
            ],
            check=True,
        )
        manifest = {
            "schema_version": 1,
            "artifact_id": artifact_id,
            "source_model": str(source),
            "source_merge_manifest_sha256": merge_manifest_sha256,
            "quantization": {"bits": 8, "group_size": 64},
            "eos_token_id": eos_token_id,
            "thinking_template_sha256": thinking_template_sha256,
        }
        _write_json(converted / SERVING_MANIFEST_NAME, manifest)
        _validate_serving_parameters(converted, manifest)
        _validate_thinking_template(converted, manifest)
        _publish_directory_no_replace(converted, target)
    return target


def serve_model(
    model_dir: Path,
    *,
    artifact_id: str,
    merged_model: Path,
    port: int = DEFAULT_MODEL_PORT,
    prompt_cache_size: int = DEFAULT_PROMPT_CACHE_SIZE,
) -> None:
    """以前台进程启动本地 MLX OpenAI-compatible 服务。

    Args:
        model_dir (Path): 已转换的 MLX 模型目录。
        artifact_id (str): 配置期望加载的产物标识。
        merged_model (Path): 配置期望的合并模型来源目录。
        port (int): HTTP 服务监听端口。
        prompt_cache_size (int): 服务保留的提示词前缀缓存槽位数。

    Raises:
        subprocess.CalledProcessError: 模型服务异常退出。

    Returns:
        None: 此函数会阻塞到模型服务退出。
    """
    validate_serving_model(
        model_dir,
        artifact_id=artifact_id,
        merged_model=merged_model,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_lm",
            "server",
            "--model",
            str(Path(model_dir).resolve()),
            "--port",
            str(port),
            "--prompt-cache-size",
            str(prompt_cache_size),
        ],
        check=True,
    )


def validate_serving_model(
    model_dir: Path,
    *,
    artifact_id: str,
    merged_model: Path,
) -> dict[str, object]:
    """用小型来源清单验证 MLX 目录身份，不扫描模型权重。"""
    model_dir = Path(model_dir)
    merged_model = Path(merged_model).resolve()
    manifest_path = model_dir / SERVING_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ServingModelIdentityError(f"服务模型清单无效: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ServingModelIdentityError(f"服务模型清单无效: {manifest_path}")
    if manifest.get("artifact_id") != artifact_id:
        raise ServingModelIdentityError("服务模型 artifact_id 不一致")
    if manifest.get("source_model") != str(merged_model):
        raise ServingModelIdentityError("服务模型合并来源路径不一致")

    merge_manifest_path = merged_model / _MERGE_MANIFEST_NAME
    if not merge_manifest_path.is_file():
        raise ServingModelIdentityError(
            f"合并模型缺少 {_MERGE_MANIFEST_NAME}: {merged_model}"
        )
    if manifest.get("source_merge_manifest_sha256") != _sha256(merge_manifest_path):
        raise ServingModelIdentityError("服务模型 merge_manifest 摘要不一致")
    _validate_merge_model_identity(merged_model, artifact_id)
    _validate_serving_parameters(model_dir, manifest)
    _validate_thinking_template(model_dir, manifest)
    return manifest


def _validate_merge_model_identity(source: Path, artifact_id: str) -> None:
    """核对当前 SFT merge 清单声明的输出目录与 adapter 身份。"""
    manifest_path = source / _MERGE_MANIFEST_NAME
    manifest = _read_json_object(manifest_path, "合并模型清单")
    if manifest.get("schema_version") != 1:
        raise ServingModelIdentityError(f"合并模型清单无效: {manifest_path}")
    declared_output = manifest.get("output")
    if (
        not isinstance(declared_output, str)
        or Path(declared_output).resolve() != source
    ):
        raise ServingModelIdentityError("合并模型清单的输出目录不一致")
    adapter = manifest.get("adapter")
    if not isinstance(adapter, str) or Path(adapter).name != artifact_id:
        raise ServingModelIdentityError("合并模型 adapter 与 artifact_id 不一致")
    source_sha256 = manifest.get("source_sha256")
    if not isinstance(source_sha256, dict) or not source_sha256:
        raise ServingModelIdentityError("合并模型清单缺少来源摘要")
    if any(
        not isinstance(name, str)
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
        for name, digest in source_sha256.items()
    ):
        raise ServingModelIdentityError("合并模型清单来源摘要格式无效")
    missing_categories = [
        category
        for category in ("base", "adapter", "tokenizer")
        if not any(name.startswith(f"{category}/") for name in source_sha256)
    ]
    if missing_categories:
        raise ServingModelIdentityError(
            "合并模型清单来源摘要缺少类别: " + ", ".join(missing_categories)
        )


def _validate_serving_parameters(
    model_dir: Path,
    manifest: dict[str, object],
) -> None:
    """核对 MLX 实际配置与服务清单记录的量化和 EOS 参数。"""
    config = _read_json_object(model_dir / "config.json", "MLX 模型配置")
    actual_quantization = config.get("quantization")
    expected_quantization = manifest.get("quantization")
    if (
        not isinstance(actual_quantization, Mapping)
        or not isinstance(expected_quantization, Mapping)
        or any(
            actual_quantization.get(key) != value
            for key, value in expected_quantization.items()
        )
    ):
        raise ServingModelIdentityError("服务模型量化配置不一致")
    expected_eos = manifest.get("eos_token_id")
    if config.get("eos_token_id") != expected_eos:
        raise ServingModelIdentityError("服务模型 EOS 配置不一致")
    generation_path = model_dir / "generation_config.json"
    if generation_path.is_file():
        generation = _read_json_object(generation_path, "MLX 生成配置")
        if generation.get("eos_token_id") != expected_eos:
            raise ServingModelIdentityError("服务模型生成 EOS 配置不一致")


def _validate_thinking_template(
    model_dir: Path,
    manifest: dict[str, object],
) -> None:
    """核对转换后的 tokenizer 仍保留 manifest 绑定的双模式模板。"""
    actual = _thinking_template_fingerprints(model_dir)
    if actual != manifest.get("thinking_template_sha256"):
        raise ServingModelIdentityError("服务模型 thinking 模板指纹不一致")


def _thinking_template_fingerprints(model_dir: Path) -> dict[str, str]:
    """实际渲染固定对话，证明模板的 thinking 开关可用并生成小型指纹。"""
    try:
        tokenizer = _load_tokenizer(model_dir)
        if getattr(tokenizer, "has_thinking", False) is not True:
            raise ServingModelIdentityError("tokenizer 不支持 thinking 状态")
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "state"},
        ]
        disabled = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        enabled = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
        )
        if not isinstance(disabled, str) or not isinstance(enabled, str):
            raise ServingModelIdentityError("thinking 模板没有返回文本")
        if disabled == enabled:
            raise ServingModelIdentityError("enable_thinking 没有改变模板渲染")
        think_start = getattr(tokenizer, "think_start", None)
        think_end = getattr(tokenizer, "think_end", None)
        if (
            not isinstance(think_start, str)
            or not isinstance(think_end, str)
            or not enabled.rstrip().endswith(think_start)
            or not disabled.rstrip().endswith(think_end)
        ):
            raise ServingModelIdentityError("thinking 模板尾部状态不符合 MLX 契约")
        disabled_tokens = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
        enabled_tokens = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=True,
        )
        if not _token_ids(disabled_tokens) or not _token_ids(enabled_tokens):
            raise ServingModelIdentityError("thinking 模板没有返回有效 token IDs")
    except ServingModelIdentityError:
        raise
    except Exception as exc:
        raise ServingModelIdentityError(
            f"无法验证 thinking 模板: {Path(model_dir)}"
        ) from exc
    return {
        "disabled": _template_digest(disabled, disabled_tokens),
        "enabled": _template_digest(enabled, enabled_tokens),
    }


def _token_ids(value: object) -> bool:
    """判断模板 token 化结果是非空且不含布尔别名的整数列表。"""
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(token, int) and not isinstance(token, bool) for token in value
        )
    )


def _template_digest(text: str, tokens: object) -> str:
    """对固定模板的文本与 token IDs 一起生成稳定功能指纹。"""
    payload = json.dumps(
        {"text": text, "tokens": tokens},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_tokenizer(model_dir: Path) -> object:
    """延迟导入 inference 依赖并加载与 MLX server 相同的 tokenizer 包装。"""
    from mlx_lm.utils import load_tokenizer

    return load_tokenizer(str(model_dir))


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    """用平台原生排他 rename 原子发布服务目录。"""
    if sys.platform == "darwin":
        library = ctypes.CDLL(None, use_errno=True)
        renamex = library.renamex_np
        renamex.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex.restype = ctypes.c_int
        result = renamex(
            os.fsencode(source),
            os.fsencode(destination),
            0x00000004,
        )
    elif sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOTSUP, "系统不支持排他原子发布", str(destination))
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
    else:
        raise OSError(errno.ENOTSUP, "系统不支持排他原子发布", str(destination))
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    """读取身份敏感的小型 JSON 对象并统一领域错误。"""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ServingModelIdentityError(f"{label}无效: {path}") from exc
    if not isinstance(value, dict):
        raise ServingModelIdentityError(f"{label}无效: {path}")
    return value


def _eos_token_id(source: Path) -> int | None:
    """读取 tokenizer 为结束标记分配的真实 token ID。

    Args:
        source (Path): Hugging Face 模型目录。

    Returns:
        int | None: 结束标记的 token ID；模型未声明时返回 ``None``。
    """
    tokenizer_config = json.loads(
        (source / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    eos_token = tokenizer_config.get("eos_token")
    tokenizer = json.loads((source / "tokenizer.json").read_text(encoding="utf-8"))
    for token in tokenizer.get("added_tokens", ()):
        if token.get("content") == eos_token:
            return int(token["id"])
    return None


def _sha256(path: Path) -> str:
    """计算一个小型清单文件的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: object) -> None:
    """以项目统一格式写入一个临时 JSON 文件。

    Args:
        path (Path): 待写入的临时文件路径。
        data (object): 可被 JSON 序列化的数据。

    Returns:
        None: 数据会覆盖写入指定文件。
    """
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _link_model_files(
    source: Path,
    target: Path,
    excluded: set[str],
) -> None:
    """为转换器创建不修改源模型的轻量目录视图。

    Args:
        source (Path): 原始 Hugging Face 模型目录。
        target (Path): 已创建的临时视图目录。
        excluded (set[str]): 将由临时修正版替代的文件名。

    Returns:
        None: 除配置外的模型文件会以符号链接加入临时视图。
    """
    for child in source.iterdir():
        if child.name not in excluded:
            (target / child.name).symlink_to(
                child,
                target_is_directory=child.is_dir(),
            )
