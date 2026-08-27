"""验证本地 MLX 模型的准备、服务和真实协议冒烟边界。"""

import json
import subprocess
from pathlib import Path

import pytest


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

    source = tmp_path / "merged"
    target = tmp_path / "serving/model-mlx-8bit"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_text"}),
        encoding="utf-8",
    )
    (source / "model.safetensors").write_bytes(b"weights")
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
                    "quantization": {"bits": 8, "group_size": 64},
                }
            ),
            encoding="utf-8",
        )
        (mlx_dir / "model.safetensors").write_bytes(b"quantized")

    monkeypatch.setattr(local_model.subprocess, "run", convert)

    result = local_model.prepare_model(source, target)

    assert result == target
    assert json.loads((source / "config.json").read_text(encoding="utf-8")) == {
        "model_type": "qwen3_5_text"
    }
    assert json.loads(
        (source / "generation_config.json").read_text(encoding="utf-8")
    ) == {"eos_token_id": 248044}
    assert (target / "model.safetensors").read_bytes() == b"quantized"


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

    source = tmp_path / "merged"
    target = tmp_path / "serving/model-mlx-8bit"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
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
        local_model.prepare_model(source, target)

    assert not target.exists()


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

    model_dir = tmp_path / "model"
    model_dir.mkdir()
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

    local_model.serve_model(model_dir)

    assert commands == [
        [
            local_model.sys.executable,
            "-m",
            "mlx_lm",
            "server",
            "--model",
            str(model_dir),
            "--port",
            "8900",
            "--prompt-cache-size",
            "10",
        ]
    ]
