"""验证训练命令行对 SFT 训练与评测的路由。"""

from pathlib import Path

from play_sts2.training import cli


def test_merge_sft_command_uses_config_and_optional_output(
    monkeypatch: object,
    capsys: object,
) -> None:
    """合并子命令把配置、adapter 与目标目录交给发布实现。

    Args:
        monkeypatch (object): Pytest 提供的属性替换工具。
        capsys (object): Pytest 提供的标准输出捕获工具。

    Raises:
        AssertionError: CLI 没有按契约传递合并参数。

    Returns:
        None: 此测试不加载真实模型。
    """
    config = object()
    calls: list[tuple[object, Path, Path | None]] = []
    monkeypatch.setattr(cli, "load_sft_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "merge_sft_adapter",
        lambda value, adapter, output=None: (
            calls.append((value, adapter, output)) or {"output": str(output)}
        ),
    )

    result = cli.main(
        [
            "merge-sft",
            "--adapter",
            "models/adapters/e3",
            "--output",
            "models/merged/e3-merged",
        ]
    )

    assert result == 0
    assert calls == [
        (
            config,
            Path("models/adapters/e3"),
            Path("models/merged/e3-merged"),
        )
    ]
    assert '"output": "models/merged/e3-merged"' in capsys.readouterr().out


def test_train_sft_command_passes_config_name_and_step_limit(
    monkeypatch: object,
    capsys: object,
) -> None:
    """训练子命令读取配置并把冒烟步数传给训练器。

    Args:
        monkeypatch (object): Pytest 提供的属性替换工具。
        capsys (object): Pytest 提供的标准输出捕获工具。

    Raises:
        AssertionError: CLI 参数没有按契约传递。

    Returns:
        None: 此测试不加载真实模型。
    """
    config = object()
    calls: list[tuple[object, str, int | None, bool]] = []
    monkeypatch.setattr(cli, "load_sft_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "train_sft",
        lambda value, name, max_steps=None, exact_resume=False: (
            calls.append((value, name, max_steps, exact_resume))
            or {"optimizer_steps": 1}
        ),
    )

    result = cli.main(
        [
            "sft",
            "--config",
            "configs/sft.toml",
            "--name",
            "20260828-smoke",
            "--max-steps",
            "1",
            "--resume",
        ]
    )

    assert result == 0
    assert calls == [(config, "20260828-smoke", 1, True)]
    assert '"optimizer_steps": 1' in capsys.readouterr().out


def test_eval_sft_command_uses_named_split(
    monkeypatch: object,
    capsys: object,
) -> None:
    """评测子命令加载指定 adapter 和数据分卷。

    Args:
        monkeypatch (object): Pytest 提供的属性替换工具。
        capsys (object): Pytest 提供的标准输出捕获工具。

    Raises:
        AssertionError: CLI 没有把评测参数传给实现。

    Returns:
        None: 此测试不加载真实模型。
    """
    config = object()
    calls: list[tuple[object, Path, str, int | None, Path | None]] = []
    monkeypatch.setattr(cli, "load_sft_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "evaluate_sft",
        lambda value, adapter, split, max_samples=None, output_path=None: (
            calls.append((value, adapter, split, max_samples, output_path))
            or {"samples": 2}
        ),
    )

    result = cli.main(
        [
            "eval-sft",
            "--config",
            "configs/sft.toml",
            "--adapter",
            "models/adapters/demo",
            "--split",
            "test",
            "--max-samples",
            "2",
            "--output",
            "runs/eval/demo.jsonl",
        ]
    )

    assert result == 0
    assert calls == [
        (
            config,
            Path("models/adapters/demo"),
            "test",
            2,
            Path("runs/eval/demo.jsonl"),
        )
    ]
    assert '"samples": 2' in capsys.readouterr().out


def test_eval_sft_loss_command_routes_teacher_forced_report(
    monkeypatch: object,
    capsys: object,
) -> None:
    """loss 子命令独立路由 teacher-forced 评测而不触发生成。

    Args:
        monkeypatch (object): Pytest 提供的属性替换工具。
        capsys (object): Pytest 提供的标准输出捕获工具。

    Raises:
        AssertionError: adapter、分卷或报告路径没有按契约传递。

    Returns:
        None: 此测试不加载真实模型。
    """
    config = object()
    calls: list[tuple[object, Path, str, int | None, Path | None]] = []
    monkeypatch.setattr(cli, "load_sft_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "evaluate_sft_loss",
        lambda value, adapter, split, max_samples=None, output_path=None: (
            calls.append((value, adapter, split, max_samples, output_path))
            or {"mean_loss": 0.5}
        ),
    )

    result = cli.main(
        [
            "eval-sft-loss",
            "--adapter",
            "models/adapters/e2",
            "--split",
            "dev",
            "--max-samples",
            "3",
            "--output",
            "runs/eval/e2-dev-loss.json",
        ]
    )

    assert result == 0
    assert calls == [
        (
            config,
            Path("models/adapters/e2"),
            "dev",
            3,
            Path("runs/eval/e2-dev-loss.json"),
        )
    ]
    assert '"mean_loss": 0.5' in capsys.readouterr().out


def test_eval_sft_knowledge_command_uses_migrated_probe_set(
    monkeypatch: object,
    capsys: object,
) -> None:
    """知识子命令把合并模型与迁移探针传给独立生成评分器。

    Args:
        monkeypatch (object): Pytest 提供的属性替换工具。
        capsys (object): Pytest 提供的标准输出捕获工具。

    Raises:
        AssertionError: 模型、探针或分层 limit 没有按契约传递。

    Returns:
        None: 此测试不加载真实模型。
    """
    calls: list[tuple[Path, Path, Path, str, int | None, int]] = []
    monkeypatch.setattr(
        cli,
        "run_knowledge_evaluation",
        lambda model, probes, output, device, limit, minimum_new_tokens: (
            calls.append(
                (
                    model,
                    probes,
                    output,
                    device,
                    limit,
                    minimum_new_tokens,
                )
            )
            or {"questions": 10}
        ),
    )

    result = cli.main(
        [
            "eval-sft-knowledge",
            "--model",
            "models/merged/e2",
            "--limit",
            "10",
        ]
    )

    assert result == 0
    assert calls == [
        (
            Path("models/merged/e2"),
            Path("data/datasets/sft/eval/knowledge"),
            Path("runs/eval/e2-knowledge.json"),
            "auto",
            10,
            96,
        )
    ]
    assert '"questions": 10' in capsys.readouterr().out
