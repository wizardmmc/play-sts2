"""验证本地模型命令的用户可观察行为。"""

import importlib
import json
from pathlib import Path

import pytest

from play_sts2.inference import ModelReply


def test_prepare_defaults_to_round_two_merged_model() -> None:
    """默认转换目标跟随用户选定的 e2 合并模型且不覆盖 e1 服务件。

    Raises:
        AssertionError: prepare 默认路径仍指向第一轮或复用同一输出目录。

    Returns:
        None: 此测试只解析命令行默认值。
    """
    cli = importlib.import_module("play_sts2.inference.cli")

    args = cli.build_parser().parse_args(["prepare"])

    assert args.source == Path("models/merged/sft-clean-20260827-native-r16-e2-merged")
    assert args.output == Path(
        "models/serving/sft-clean-20260827-native-r16-e2-mlx-8bit"
    )


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
    source = tmp_path / "merged"
    target = tmp_path / "serving"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_text"}),
        encoding="utf-8",
    )
    (source / "model.safetensors").write_bytes(b"weights")
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
        (output / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(local_model.subprocess, "run", convert)

    exit_code = cli.main(["prepare", "--source", str(source), "--output", str(target)])

    assert exit_code == 0
    assert target.is_dir()
    assert capsys.readouterr().out == f"模型准备完成: {target.resolve()}\n"


def test_main_forwards_serve_options_and_stops_cleanly(
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
    calls: list[tuple[Path, int, int]] = []

    def serve(
        model_dir: Path,
        *,
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
        calls.append((model_dir, port, prompt_cache_size))
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "serve_model", serve)

    exit_code = cli.main(
        [
            "serve",
            "--model-dir",
            "models/custom",
            "--port",
            "9000",
            "--prompt-cache-size",
            "4",
        ]
    )

    assert exit_code == 130
    assert calls == [(Path("models/custom"), 9000, 4)]


def test_main_forwards_smoke_options(
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
    calls: list[tuple[str, str | None]] = []

    def smoke(base_url: str, *, model: str | None = None) -> ModelReply:
        """记录冒烟参数并返回合法模型回复。

        Args:
            base_url (str): CLI 传入的服务根地址。
            model (str | None): CLI 传入的模型名称。

        Returns:
            ModelReply: 用于 CLI 输出的固定合法回复。
        """
        calls.append((base_url, model))
        return ModelReply("ACTION: end_turn")

    monkeypatch.setattr(cli, "smoke_model", smoke)

    exit_code = cli.main(
        ["smoke", "--base-url", "http://127.0.0.1:9000", "--model", "qwen"]
    )

    assert exit_code == 0
    assert calls == [("http://127.0.0.1:9000", "qwen")]
    assert capsys.readouterr().out == "模型协议验证通过: ACTION: end_turn\n"
