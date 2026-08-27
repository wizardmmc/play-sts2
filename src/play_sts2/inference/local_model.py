"""准备并验证供本机 Apple Silicon 使用的 MLX 模型。"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_MODEL_PORT = 8900
DEFAULT_PROMPT_CACHE_SIZE = 10


def prepare_model(source: Path, target: Path) -> Path:
    """把合并后的 Hugging Face 模型转换为 MLX 8-bit 模型。

    Args:
        source (Path): 合并后的 Hugging Face 模型目录。
        target (Path): 转换完成后保存 MLX 模型的目录。

    Raises:
        FileExistsError: 目标目录已经存在。
        subprocess.CalledProcessError: ``mlx_lm.convert`` 转换失败。

    Returns:
        Path: 转换完成的目标目录。
    """
    if target.exists():
        raise FileExistsError(target)

    source = source.resolve()
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
        converted.rename(target)
    return target


def serve_model(
    model_dir: Path,
    *,
    port: int = DEFAULT_MODEL_PORT,
    prompt_cache_size: int = DEFAULT_PROMPT_CACHE_SIZE,
) -> None:
    """以前台进程启动本地 MLX OpenAI-compatible 服务。

    Args:
        model_dir (Path): 已转换的 MLX 模型目录。
        port (int): HTTP 服务监听端口。
        prompt_cache_size (int): 服务保留的提示词前缀缓存槽位数。

    Raises:
        subprocess.CalledProcessError: 模型服务异常退出。

    Returns:
        None: 此函数会阻塞到模型服务退出。
    """
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_lm",
            "server",
            "--model",
            str(model_dir),
            "--port",
            str(port),
            "--prompt-cache-size",
            str(prompt_cache_size),
        ],
        check=True,
    )


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
