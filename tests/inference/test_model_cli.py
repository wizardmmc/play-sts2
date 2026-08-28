"""验证本地模型命令的用户可观察行为。"""

import importlib
import json
from pathlib import Path

import pytest

from play_sts2.inference import ModelReply


def test_main_prepares_artifact_selected_by_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无模型路径参数时，prepare 只使用配置声明的完整 artifact 身份。

    Args:
        tmp_path (Path): Pytest 提供的隔离配置目录。
        monkeypatch (pytest.MonkeyPatch): 用于替换真实模型转换实现。

    Raises:
        AssertionError: prepare 默认路径仍指向第一轮或复用同一输出目录。

    Returns:
        None: 此测试只解析命令行默认值。
    """
    cli = importlib.import_module("play_sts2.inference.cli")
    config_path, source, target = _write_inference_config(tmp_path)
    calls: list[tuple[Path, Path, str]] = []

    def prepare(source_arg: Path, target_arg: Path, *, artifact_id: str) -> Path:
        """记录 prepare 选择的完整模型制品身份。

        Args:
            source_arg (Path): 合并后模型的输入路径。
            target_arg (Path): MLX 服务制品的输出路径。
            artifact_id (str): 完整模型制品标识。

        Returns:
            Path: 未执行转换的测试目标路径。
        """
        calls.append((source_arg, target_arg, artifact_id))
        return target_arg

    monkeypatch.setattr(cli, "prepare_model", prepare)

    exit_code = cli.main(["prepare", "--config", str(config_path)])

    assert exit_code == 0
    assert calls == [(source, target, "demo-e3")]


def test_main_prepares_requested_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """prepare 子命令把指定合并模型写入指定服务目录。

    Args:
        tmp_path (Path): Pytest 提供的隔离临时目录。
        monkeypatch (pytest.MonkeyPatch): 用于替换昂贵的外部转换进程。
        capsys (pytest.CaptureFixture[str]): 用于读取命令标准输出。

    Raises:
        AssertionError: CLI 没有完成转换或没有报告目标目录。

    Returns:
        None: 此测试只验证模型准备命令。
    """
    cli = importlib.import_module("play_sts2.inference.cli")
    local_model = importlib.import_module("play_sts2.inference.local_model")
    config_path, source, target = _write_inference_config(tmp_path)
    source.mkdir(parents=True)
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_text"}),
        encoding="utf-8",
    )
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "merge_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "adapter": str(tmp_path / "models/adapters/demo-e3"),
                "output": str(source.resolve()),
                "tokenizer_source": str(tmp_path / "models/adapters/demo-e3"),
                "source_sha256": {
                    "base/config.json": "a" * 64,
                    "adapter/train_manifest.json": "b" * 64,
                    "tokenizer/tokenizer.json": "c" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    (source / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<|im_end|>"}),
        encoding="utf-8",
    )
    (source / "tokenizer.json").write_text(
        json.dumps({"added_tokens": [{"id": 248046, "content": "<|im_end|>"}]}),
        encoding="utf-8",
    )

    def convert(command: list[str], *, check: bool) -> None:
        """创建 CLI 转换命令指定的最小服务模型。

        Args:
            command (list[str]): ``mlx_lm.convert`` 的完整命令参数。
            check (bool): 子进程失败时是否应抛出异常。

        Returns:
            None: 此替身写入最小转换结果。
        """
        assert check is True
        output = Path(command[command.index("--mlx-path") + 1])
        output.mkdir()
        (output / "config.json").write_text(
            json.dumps(
                {
                    "eos_token_id": 248046,
                    "quantization": {"bits": 8, "group_size": 64},
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(local_model.subprocess, "run", convert)
    monkeypatch.setattr(
        local_model,
        "_thinking_template_fingerprints",
        lambda _path: {"disabled": "a" * 64, "enabled": "b" * 64},
    )

    exit_code = cli.main(["prepare", "--config", str(config_path)])

    assert exit_code == 0
    assert target.is_dir()
    assert capsys.readouterr().out == f"模型准备完成: {target.resolve()}\n"


def test_main_forwards_serve_options_and_stops_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """serve 子命令转发参数，并把用户中断视为正常停止。

    Args:
        monkeypatch (pytest.MonkeyPatch): 用于替换阻塞的模型服务进程。

    Raises:
        AssertionError: 服务参数或中断返回码不符合约定。

    Returns:
        None: 此测试只验证服务命令编排。
    """
    cli = importlib.import_module("play_sts2.inference.cli")
    config_path, source, target = _write_inference_config(tmp_path)
    calls: list[tuple[Path, str, Path, int, int]] = []

    def serve(
        model_dir: Path,
        *,
        artifact_id: str,
        merged_model: Path,
        port: int,
        prompt_cache_size: int,
    ) -> None:
        """记录服务参数后模拟用户中断。

        Args:
            model_dir (Path): CLI 传入的模型目录。
            port (int): CLI 传入的监听端口。
            prompt_cache_size (int): CLI 传入的缓存槽位数。

        Raises:
            KeyboardInterrupt: 始终模拟用户按下 Ctrl-C。

        Returns:
            None: 此替身不会启动真实服务。
        """
        calls.append((model_dir, artifact_id, merged_model, port, prompt_cache_size))
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "serve_model", serve)

    exit_code = cli.main(
        [
            "serve",
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 130
    assert calls == [(target, "demo-e3", source, 9000, 6)]


def test_main_forwards_smoke_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """smoke 子命令转发服务地址和可选模型名。

    Args:
        monkeypatch (pytest.MonkeyPatch): 用于替换真实模型请求。
        capsys (pytest.CaptureFixture[str]): 用于读取命令标准输出。

    Raises:
        AssertionError: 请求参数或成功输出不符合约定。

    Returns:
        None: 此测试只验证冒烟命令编排。
    """
    cli = importlib.import_module("play_sts2.inference.cli")
    config_path, source, target = _write_inference_config(tmp_path)
    calls: list[tuple[object, ...]] = []

    def smoke(
        base_url: str,
        *,
        artifact_id: str,
        merged_model: Path,
        serving_model: Path,
        enable_thinking: bool,
        max_tokens: int,
        temperature: float,
        model: str | None = None,
    ) -> ModelReply:
        """记录冒烟参数并返回合法模型回复。

        Args:
            base_url (str): CLI 传入的服务根地址。
            model (str | None): CLI 传入的模型名称。

        Returns:
            ModelReply: 用于 CLI 输出的固定合法回复。
        """
        calls.append(
            (
                base_url,
                artifact_id,
                merged_model,
                serving_model,
                enable_thinking,
                max_tokens,
                temperature,
                model,
            )
        )
        return ModelReply("ACTION: end_turn")

    monkeypatch.setattr(cli, "smoke_model", smoke)

    exit_code = cli.main(["smoke", "--config", str(config_path), "--profile", "think"])

    assert exit_code == 0
    assert calls == [
        (
            "http://127.0.0.1:9000",
            "demo-e3",
            source,
            target,
            True,
            512,
            0.2,
            None,
        )
    ]
    assert capsys.readouterr().out == "模型协议验证通过: ACTION: end_turn\n"


def test_model_smoke_rejects_request_model_override() -> None:
    """本地模型 CLI 不允许请求级 model 绕过配置中的目录身份。"""
    cli = importlib.import_module("play_sts2.inference.cli")

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["smoke", "--model", "wrong-model"])


def _write_inference_config(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    """写入 CLI 测试共用的完整推理配置。"""
    root = tmp_path
    source = root / "models/merged/demo-e3-merged"
    target = root / "models/serving/demo-e3-mlx-8bit"
    config_path = root / "inference.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""
artifact_id = "demo-e3"
merged_model = {json.dumps(str(source))}
serving_model = {json.dumps(str(target))}
default_profile = "no-think"

[server]
base_url = "http://127.0.0.1:9000"
port = 9000
prompt_cache_size = 6

[profiles.no-think]
enable_thinking = false
max_tokens = 128
temperature = 0.0

[profiles.think]
enable_thinking = true
max_tokens = 512
temperature = 0.2
""".strip(),
        encoding="utf-8",
    )
    return config_path, source, target
