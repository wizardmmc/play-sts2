"""把训练完成的 LoRA adapter 安全合并为独立 Hugging Face 模型。"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .encoding import SftConfig, SftTrainingError
from .provenance import (
    model_source_files,
    sha256_files,
    snapshot_files,
    sources_unchanged,
    tokenizer_source_files,
)


def merge_sft_adapter(
    config: SftConfig,
    adapter: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    """校验血缘、合并一个 LoRA adapter 并原子发布完整模型。

    Args:
        config (SftConfig): 提供本地基座模型与标准模型目录的 SFT 配置。
        adapter (Path): 含 PEFT 权重和训练清单的 adapter 目录。
        output (Path | None): 可选目标目录；省略时写入同级 ``models/merged``。

    Raises:
        SftTrainingError: 输入不完整、血缘不符、来源变化或目标已存在。
        OSError: 模型读取、保存或原子发布失败。

    Returns:
        dict[str, Any]: 合并输出目录、来源目录和固定合并方法。
    """
    adapter = Path(adapter)
    destination = (
        Path(output) if output is not None else _default_output(config, adapter)
    )
    tokenizer_source = _tokenizer_source(config.base_model, adapter)
    source_files = _validate_merge_paths(
        config.base_model,
        adapter,
        tokenizer_source,
        destination,
    )
    source_snapshot = snapshot_files(source_files)
    source_hashes = sha256_files(source_files)
    _require_unchanged_sources(
        config.base_model,
        adapter,
        tokenizer_source,
        source_files,
        source_snapshot,
    )
    lineage = _validate_adapter_lineage(
        config.base_model,
        adapter,
        _base_hashes(source_hashes),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged_model, tokenizer, versions = _load_merged_artifacts(
        config.base_model,
        adapter,
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}-",
            dir=destination.parent,
        )
    )
    try:
        merged_model.save_pretrained(
            staging,
            safe_serialization=True,
            max_shard_size="10GB",
        )
        tokenizer.save_pretrained(staging)
        manifest = _merge_manifest(
            config.base_model,
            adapter,
            tokenizer_source,
            destination,
            source_hashes,
            lineage,
            versions,
        )
        _write_json(staging / "merge_manifest.json", manifest)
        _require_unchanged_sources(
            config.base_model,
            adapter,
            tokenizer_source,
            source_files,
            source_snapshot,
        )
        _publish_directory_no_replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output": str(destination),
        "base_model": str(config.base_model),
        "adapter": str(adapter),
        "merge": "peft.merge_and_unload(safe_merge=True)",
    }


def _default_output(config: SftConfig, adapter: Path) -> Path:
    """从标准 adapter 根目录推导默认合并目录。

    Args:
        config (SftConfig): 含 ``adapter_root`` 的训练配置。
        adapter (Path): 本次合并的 adapter 目录。

    Returns:
        Path: ``models/merged/<adapter>-merged`` 风格的目标目录。
    """
    return config.adapter_root.parent / "merged" / f"{adapter.name}-merged"


def _tokenizer_source(base_model: Path, adapter: Path) -> Path:
    """选择训练时随 adapter 发布的 tokenizer，缺失时回退到基座。

    Args:
        base_model (Path): 本地 Hugging Face 基座模型目录。
        adapter (Path): PEFT adapter 目录。

    Returns:
        Path: 应加载并记录指纹的 tokenizer 目录。
    """
    return adapter if (adapter / "tokenizer_config.json").is_file() else base_model


def _validate_merge_paths(
    base_model: Path,
    adapter: Path,
    tokenizer_source: Path,
    destination: Path,
) -> dict[str, Path]:
    """校验合并输入、完整来源清单和目标边界。

    Args:
        base_model (Path): 本地 Hugging Face 基座模型目录。
        adapter (Path): PEFT adapter 目录。
        tokenizer_source (Path): 实际采用的 tokenizer 目录。
        destination (Path): 尚未创建的合并目标目录。

    Raises:
        SftTrainingError: 必需文件缺失、目标已存在或路径重叠。

    Returns:
        dict[str, Path]: 需要取摘要并监测变化的全部来源文件。
    """
    base_model = Path(base_model)
    adapter = Path(adapter)
    destination = Path(destination)
    if destination.exists():
        raise SftTrainingError(f"合并输出已存在: {destination}")
    resolved_destination = destination.resolve()
    for source in (base_model.resolve(), adapter.resolve()):
        if (
            resolved_destination == source
            or resolved_destination in source.parents
            or source in resolved_destination.parents
        ):
            raise SftTrainingError(f"合并输出不能与来源重叠: {destination}")

    source_files = _merge_source_files(base_model, adapter, tokenizer_source)
    missing = [path for path in source_files.values() if not path.is_file()]
    if missing:
        raise SftTrainingError(f"合并来源文件缺失: {', '.join(map(str, missing))}")
    base_files = model_source_files(base_model)
    if "config.json" not in base_files or not _has_model_weights(base_files):
        raise SftTrainingError(f"基座模型配置或权重不完整: {base_model}")
    if not tokenizer_source_files(tokenizer_source):
        raise SftTrainingError(f"tokenizer 来源文件缺失: {tokenizer_source}")
    return source_files


def _merge_source_files(
    base_model: Path,
    adapter: Path,
    tokenizer_source: Path,
) -> dict[str, Path]:
    """收集合并结果所依赖的基座、adapter 与 tokenizer 文件。

    Args:
        base_model (Path): 本地 Hugging Face 基座模型目录。
        adapter (Path): PEFT adapter 目录。
        tokenizer_source (Path): 实际采用的 tokenizer 目录。

    Returns:
        dict[str, Path]: 带来源前缀的稳定文件映射。
    """
    adapter_weights = adapter / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        adapter_weights = adapter / "adapter_model.bin"
    files = {
        f"base/{name}": path for name, path in model_source_files(base_model).items()
    }
    files.update(
        {
            "adapter/adapter_config.json": adapter / "adapter_config.json",
            f"adapter/{adapter_weights.name}": adapter_weights,
            "adapter/train_manifest.json": adapter / "train_manifest.json",
        }
    )
    files.update(
        {
            f"tokenizer/{name}": path
            for name, path in tokenizer_source_files(tokenizer_source).items()
        }
    )
    return dict(sorted(files.items()))


def _has_model_weights(files: Mapping[str, Path]) -> bool:
    """判断本地模型文件映射是否包含至少一个权重文件。

    Args:
        files (Mapping[str, Path]): ``model_source_files`` 的结果。

    Returns:
        bool: 包含 safetensors 或 PyTorch 权重时为真。
    """
    return any(name.endswith((".safetensors", ".bin")) for name in files)


def _base_hashes(source_hashes: Mapping[str, str]) -> dict[str, str]:
    """从完整来源摘要中提取基座模型相对文件摘要。

    Args:
        source_hashes (Mapping[str, str]): 带来源前缀的摘要映射。

    Returns:
        dict[str, str]: 可与训练清单直接比较的基座摘要。
    """
    prefix = "base/"
    return {
        name.removeprefix(prefix): digest
        for name, digest in source_hashes.items()
        if name.startswith(prefix)
    }


def _validate_adapter_lineage(
    base_model: Path,
    adapter: Path,
    selected_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """证明 adapter 所声明的基座与本次选择的基座内容一致。

    新训练清单直接比较权重摘要；旧清单则对仍可访问的声明路径计算
    同一组模型文件摘要，避免仅凭架构相同而合并错误 checkpoint。

    Args:
        base_model (Path): 本次选择的本地基座模型目录。
        adapter (Path): 待合并 adapter 目录。
        selected_hashes (Mapping[str, str]): 已计算的本次基座文件摘要。

    Raises:
        SftTrainingError: 清单摘要不符、声明路径内容不符或无法证明血缘。

    Returns:
        dict[str, Any]: 写入合并清单的声明与校验方式。
    """
    adapter_config = _read_json_object(adapter / "adapter_config.json")
    train_manifest = _read_json_object(adapter / "train_manifest.json")
    checks: list[str] = []
    expected = train_manifest.get("base_model_sha256")
    if expected is not None:
        if not isinstance(expected, dict) or not all(
            isinstance(name, str) and isinstance(digest, str)
            for name, digest in expected.items()
        ):
            raise SftTrainingError("adapter 训练清单的基座指纹格式无效")
        if dict(expected) != dict(selected_hashes):
            raise SftTrainingError("adapter 与所选基座血缘不一致: 权重指纹不同")
        checks.append("train_manifest_sha256")

    declarations = _base_declarations(adapter_config, train_manifest)
    local_cache: dict[Path, dict[str, str]] = {}
    selected_root = Path(base_model).resolve()
    local_match = False
    for declaration in declarations:
        declared_root = _existing_local_directory(declaration)
        if declared_root is None:
            continue
        if declared_root == selected_root:
            local_match = True
            continue
        declared_files = model_source_files(declared_root)
        if "config.json" not in declared_files or not _has_model_weights(
            declared_files
        ):
            raise SftTrainingError(
                f"adapter 声明的基座不完整，无法校验血缘: {declared_root}"
            )
        if declared_root not in local_cache:
            local_cache[declared_root] = sha256_files(declared_files)
        declared_hashes = local_cache[declared_root]
        if declared_hashes != dict(selected_hashes):
            raise SftTrainingError(f"adapter 与所选基座血缘不一致: {declared_root}")
        local_match = True
    if local_match:
        checks.append("declared_local_model")
    if not checks:
        raise SftTrainingError("无法从 adapter 清单证明所选基座的血缘")
    return {
        "checks": checks,
        "declared_base_models": declarations,
    }


def _base_declarations(
    adapter_config: Mapping[str, Any],
    train_manifest: Mapping[str, Any],
) -> list[str]:
    """提取 PEFT 配置和新旧训练清单里的基座声明。

    Args:
        adapter_config (Mapping[str, Any]): 已解析的 PEFT 配置。
        train_manifest (Mapping[str, Any]): 已解析的训练清单。

    Returns:
        list[str]: 去重并保持稳定顺序的非空基座声明。
    """
    values = (
        adapter_config.get("base_model_name_or_path"),
        train_manifest.get("base_model"),
        train_manifest.get("init_model"),
    )
    return list(
        dict.fromkeys(value for value in values if isinstance(value, str) and value)
    )


def _existing_local_directory(value: str) -> Path | None:
    """把存在的本地基座声明解析为绝对目录。

    Args:
        value (str): 清单中的路径或远端模型标识。

    Returns:
        Path | None: 本地目录存在时返回其真实路径，否则返回 ``None``。
    """
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_dir() else None


def _read_json_object(path: Path) -> dict[str, Any]:
    """读取并确认来源 JSON 的顶层是对象。

    Args:
        path (Path): 待读取的配置或清单。

    Raises:
        SftTrainingError: JSON 顶层不是对象。
        OSError: 文件无法读取。
        json.JSONDecodeError: 文件不是有效 JSON。

    Returns:
        dict[str, Any]: 已解析对象。
    """
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SftTrainingError(f"合并来源 JSON 顶层必须是对象: {path}")
    return value


def _require_unchanged_sources(
    base_model: Path,
    adapter: Path,
    tokenizer_source: Path,
    expected_files: Mapping[str, Path],
    expected_snapshot: Mapping[str, tuple[int, int, int, int, int]],
) -> None:
    """拒绝摘要计算后文件集合或状态发生变化的合并来源。

    Args:
        base_model (Path): 本地 Hugging Face 基座模型目录。
        adapter (Path): PEFT adapter 目录。
        tokenizer_source (Path): 实际采用的 tokenizer 目录。
        expected_files (Mapping[str, Path]): 取摘要时的来源文件集合。
        expected_snapshot (Mapping[str, tuple[int, int, int, int, int]]): 初始文件状态。

    Raises:
        SftTrainingError: 来源集合、tokenizer 选择或文件状态发生变化。

    Returns:
        None: 所有来源保持不变时返回。
    """
    try:
        current_tokenizer = _tokenizer_source(base_model, adapter)
        current_files = _merge_source_files(base_model, adapter, current_tokenizer)
        unchanged = (
            current_tokenizer.resolve() == Path(tokenizer_source).resolve()
            and current_files == dict(expected_files)
            and sources_unchanged(expected_files, expected_snapshot)
        )
    except OSError:
        unchanged = False
    if not unchanged:
        raise SftTrainingError("合并来源在合并期间发生变化，已拒绝发布")


def _load_merged_artifacts(
    base_model: Path,
    adapter: Path,
) -> tuple[Any, Any, dict[str, str]]:
    """在 CPU 上加载基座和 adapter，并执行 PEFT 安全合并。

    Args:
        base_model (Path): 本地 Hugging Face 基座模型目录。
        adapter (Path): 待合并的 PEFT adapter 目录。

    Returns:
        tuple[Any, Any, dict[str, str]]: 合并模型、tokenizer 和依赖版本。
    """
    import torch
    from peft import PeftModel
    from peft import __version__ as peft_version
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import __version__ as transformers_version

    tokenizer = AutoTokenizer.from_pretrained(
        str(_tokenizer_source(base_model, adapter)),
        local_files_only=True,
        trust_remote_code=False,
    )
    base = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    merged = peft_model.merge_and_unload(safe_merge=True)
    merged.eval()
    versions = {
        "peft": peft_version,
        "torch": torch.__version__,
        "transformers": transformers_version,
    }
    return merged, tokenizer, versions


def _merge_manifest(
    base_model: Path,
    adapter: Path,
    tokenizer_source: Path,
    destination: Path,
    source_hashes: Mapping[str, str],
    lineage: Mapping[str, Any],
    versions: Mapping[str, str],
) -> dict[str, Any]:
    """构建完整输入指纹和血缘均已验证的合并清单。

    Args:
        base_model (Path): 本地基座模型目录。
        adapter (Path): PEFT adapter 目录。
        tokenizer_source (Path): 实际采用的 tokenizer 目录。
        destination (Path): 最终合并目录。
        source_hashes (Mapping[str, str]): 合并前计算的全部来源摘要。
        lineage (Mapping[str, Any]): adapter 基座血缘校验记录。
        versions (Mapping[str, str]): 实际合并依赖版本。

    Returns:
        dict[str, Any]: 可直接序列化为 JSON 的合并清单。
    """
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "base_model": str(Path(base_model).resolve()),
        "adapter": str(Path(adapter).resolve()),
        "tokenizer_source": str(Path(tokenizer_source).resolve()),
        "output": str(Path(destination).resolve()),
        "dtype": "bfloat16",
        "merge": "peft.merge_and_unload(safe_merge=True)",
        "source_sha256": dict(sorted(source_hashes.items())),
        "lineage": dict(lineage),
        "versions": dict(sorted(versions.items())),
    }


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    """原子发布目录，并保证竞态创建的目标绝不被替换。

    macOS 与 Linux 分别使用原生排他 rename；其他平台使用保守回退。

    Args:
        source (Path): 已完整写入的同文件系统暂存目录。
        destination (Path): 必须尚不存在的最终目录。

    Raises:
        SftTrainingError: 最终目录已由其他进程创建。
        OSError: 原子重命名失败。

    Returns:
        None: 目录成功发布后返回。
    """
    if sys.platform == "darwin":
        _renamex_no_replace(source, destination)
        return
    if sys.platform.startswith("linux"):
        if _renameat2_no_replace(source, destination):
            return
        raise SftTrainingError("当前 Linux libc 不支持排他原子发布，已拒绝降级")
    raise SftTrainingError(f"当前系统不支持排他原子发布: {sys.platform}")


def _renamex_no_replace(source: Path, destination: Path) -> None:
    """用 macOS ``renamex_np(RENAME_EXCL)`` 排他发布目录。

    Args:
        source (Path): 暂存目录。
        destination (Path): 最终目录。

    Raises:
        SftTrainingError: 目标已存在。
        OSError: 系统调用失败。

    Returns:
        None: 排他重命名成功后返回。
    """
    library = ctypes.CDLL(None, use_errno=True)
    renamex = library.renamex_np
    renamex.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
    renamex.restype = ctypes.c_int
    result = renamex(os.fsencode(source), os.fsencode(destination), 0x00000004)
    if result != 0:
        _raise_rename_error(destination)


def _renameat2_no_replace(source: Path, destination: Path) -> bool:
    """尝试用 Linux ``renameat2(RENAME_NOREPLACE)`` 排他发布目录。

    Args:
        source (Path): 暂存目录。
        destination (Path): 最终目录。

    Raises:
        SftTrainingError: 目标已存在。
        OSError: 系统调用存在但执行失败。

    Returns:
        bool: 系统提供该调用且发布成功时为真，不提供时为假。
    """
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        return False
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
    if result != 0:
        _raise_rename_error(destination)
    return True


def _raise_rename_error(destination: Path) -> None:
    """把原生排他 rename 的 errno 转换为领域错误或 OSError。

    Args:
        destination (Path): 用于错误上下文的最终目录。

    Raises:
        SftTrainingError: errno 表示目标已经存在。
        OSError: 其他系统错误。
    """
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise SftTrainingError(f"合并输出已存在: {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """写入带稳定缩进和末尾换行的 UTF-8 JSON。

    Args:
        path (Path): 输出 JSON 路径。
        value (Mapping[str, Any]): 可序列化的清单内容。

    Returns:
        None: 文件完整写入后返回。
    """
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
