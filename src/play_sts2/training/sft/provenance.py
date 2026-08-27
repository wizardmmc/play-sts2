"""提供 SFT 模型来源文件的稳定指纹与变更检测。"""

import hashlib
from collections.abc import Mapping
from pathlib import Path

FileStat = tuple[int, int, int, int, int]


def model_source_files(root: Path) -> dict[str, Path]:
    """列出决定一个本地 Hugging Face 模型身份的配置与权重。

    Args:
        root (Path): 本地模型目录。

    Returns:
        dict[str, Path]: 以相对文件名为键的模型来源文件。
    """
    root = Path(root)
    names = {
        "config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
    paths = [root / name for name in names]
    paths.extend(root.glob("*.safetensors"))
    paths.extend(root.glob("pytorch_model*.bin"))
    return {path.name: path for path in sorted(set(paths)) if path.is_file()}


def tokenizer_source_files(root: Path) -> dict[str, Path]:
    """列出决定 tokenizer 和聊天模板身份的本地文件。

    Args:
        root (Path): tokenizer 所在目录。

    Returns:
        dict[str, Path]: 以相对文件名为键的 tokenizer 来源文件。
    """
    root = Path(root)
    names = (
        "chat_template.jinja",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    )
    return {name: root / name for name in names if (root / name).is_file()}


def sha256_files(files: Mapping[str, Path]) -> dict[str, str]:
    """流式计算一组来源文件的 SHA-256。

    Args:
        files (Mapping[str, Path]): 稳定名称到文件路径的映射。

    Returns:
        dict[str, str]: 按名称排序的十六进制摘要。
    """
    return {name: _sha256(path) for name, path in sorted(files.items())}


def snapshot_files(files: Mapping[str, Path]) -> dict[str, FileStat]:
    """记录一组文件的设备、inode、长度、修改时间与状态变更时间。

    Args:
        files (Mapping[str, Path]): 稳定名称到文件路径的映射。

    Returns:
        dict[str, FileStat]: 可用于加载前后快速对比的文件状态。
    """
    return {
        name: (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        for name, path in sorted(files.items())
        for stat in (path.stat(),)
    }


def sources_unchanged(
    files: Mapping[str, Path],
    expected: Mapping[str, FileStat],
) -> bool:
    """判断合并期间来源文件集合及状态是否保持不变。

    Args:
        files (Mapping[str, Path]): 当前来源文件映射。
        expected (Mapping[str, FileStat]): 合并前记录的文件状态。

    Returns:
        bool: 文件集合与每个文件状态都相同时为真。
    """
    try:
        return snapshot_files(files) == dict(expected)
    except OSError:
        return False


def _sha256(path: Path) -> str:
    """流式计算单个文件的 SHA-256。

    Args:
        path (Path): 待摘要文件。

    Returns:
        str: 小写十六进制 SHA-256。
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
