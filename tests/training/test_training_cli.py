"""验证训练命令行对 SFT 训练与评测的路由。"""

from pathlib import Path

import pytest

from play_sts2.training import cli


def test_training_commands_use_grouped_config_defaults() -> None:
    """训练与评测命令默认读取 ``configs/sft`` 下的本地配置。

    Returns:
        None: 此测试防止目录重构后 CLI 继续访问旧扁平路径。
    """
    parser = cli.build_parser()

    assert parser.parse_args(["sft", "--name", "20260830-demo"]).config == Path(
        "configs/sft/sft.toml"
    )
    assert parser.parse_args(["sft-cuda", "--name", "20260830-demo"]).config == Path(
        "configs/sft/sft-cuda.toml"
    )
    for command in ("merge-sft", "eval-sft", "eval-sft-loss"):
        assert parser.parse_args(
            [command, "--adapter", "models/adapters/demo"]
        ).config == Path("configs/sft/sft.toml")


def test_collect_rl_battle_command_routes_remote_policy_and_game_workers(
    monkeypatch: object,
    capsys: object,
) -> None:
    """战斗采样命令把场景、多个游戏端点和冻结策略交给 collector。

    Args:
        monkeypatch (object): Pytest 提供的属性替换工具。
        capsys (object): Pytest 提供的标准输出捕获工具。

    Raises:
        AssertionError: CLI 丢失 worker、policy version 或采样预算。

    Returns:
        None: 此测试不连接真实游戏或 A100 服务。
    """
    calls: list[dict[str, object]] = []

    def collect(**kwargs: object) -> dict[str, object]:
        """记录 collector 参数并返回最小摘要。

        Args:
            **kwargs (object): CLI 传入的采样参数。

        Returns:
            dict[str, object]: 可直接输出为 JSON 的采样摘要。
        """
        calls.append(kwargs)
        return {"group_id": kwargs["group_id"], "arms": kwargs["group_size"]}

    monkeypatch.setattr(cli, "collect_battle_rollout_group", collect)

    result = cli.main(
        [
            "collect-rl-battle",
            "--scenario",
            "configs/scenarios/cultists.json",
            "--game-url",
            "http://127.0.0.1:8080",
            "--game-url",
            "http://127.0.0.1:8081",
            "--model-url",
            "http://127.0.0.1:8900",
            "--policy-model",
            "policy-test",
            "--vllm-logprobs-mode",
            "processed_logprobs",
            "--structured-output-backend",
            "xgrammar",
            "--structured-output-version",
            "0.1.33",
            "--group-id",
            "battle-demo-001",
            "--output",
            "runs/rl/battle-demo-001.json",
            "--temperature",
            "0.8",
        ]
    )

    assert result == 0
    assert calls == [
        {
            "scenario_path": Path("configs/scenarios/cultists.json"),
            "game_urls": (
                "http://127.0.0.1:8080",
                "http://127.0.0.1:8081",
            ),
            "model_url": "http://127.0.0.1:8900",
            "policy_model": "policy-test",
            "vllm_logprobs_mode": "processed_logprobs",
            "structured_output_backend": "xgrammar",
            "structured_output_version": "0.1.33",
            "group_id": "battle-demo-001",
            "output_path": Path("runs/rl/battle-demo-001.json"),
            "group_size": 8,
            "max_tokens": 128,
            "temperature": 0.8,
            "infrastructure_attempts": 3,
        }
    ]
    assert '"arms": 8' in capsys.readouterr().out


def test_collect_rl_tree_command_routes_checkpoint_two_games_and_a100(
    monkeypatch: object,
    capsys: object,
) -> None:
    """Tree 命令应把原生 checkpoint、两个本地端口和远程 E6 policy 传入。

    Args:
        monkeypatch (object): Pytest 提供的属性替换工具。
        capsys (object): Pytest 提供的标准输出捕获工具。

    Returns:
        None: 此测试不启动游戏或 A100 服务。
    """
    calls: list[dict[str, object]] = []

    def collect(**kwargs: object) -> dict[str, object]:
        """记录 Tree collector 参数。

        Args:
            **kwargs (object): CLI 解析后的完整参数。

        Returns:
            dict[str, object]: 最小可序列化摘要。
        """
        calls.append(kwargs)
        return {"group_id": kwargs["group_id"], "arms": kwargs["group_size"]}

    monkeypatch.setattr(cli, "collect_tree_rollout_group", collect)

    result = cli.main(
        [
            "collect-rl-tree",
            "--checkpoint",
            "runs/rl/tree/checkpoint",
            "--executable",
            "/game/SlayTheSpire2",
            "--profile",
            "e2e/fixtures/profile",
            "--home-root",
            "runs/rl/tree/homes",
            "--port",
            "8080",
            "--port",
            "8081",
            "--strategy-model-url",
            "http://127.0.0.1:8900",
            "--battle-model-url",
            "http://127.0.0.1:8900",
            "--strategy-policy-model",
            "e6-feasibility",
            "--battle-policy-model",
            "e6-feasibility",
            "--vllm-logprobs-mode",
            "processed_logprobs",
            "--structured-output-backend",
            "xgrammar",
            "--structured-output-version",
            "0.1.33",
            "--group-id",
            "tree-demo-001",
            "--output",
            "runs/rl/tree/tree-demo-001.json",
        ]
    )

    assert result == 0
    assert calls == [
        {
            "checkpoint_path": Path("runs/rl/tree/checkpoint"),
            "executable": Path("/game/SlayTheSpire2"),
            "profile": Path("e2e/fixtures/profile"),
            "home_root": Path("runs/rl/tree/homes"),
            "ports": (8080, 8081),
            "strategy_model_url": "http://127.0.0.1:8900",
            "battle_model_url": "http://127.0.0.1:8900",
            "strategy_policy_model": "e6-feasibility",
            "battle_policy_model": "e6-feasibility",
            "vllm_logprobs_mode": "processed_logprobs",
            "structured_output_backend": "xgrammar",
            "structured_output_version": "0.1.33",
            "group_id": "tree-demo-001",
            "output_path": Path("runs/rl/tree/tree-demo-001.json"),
            "group_size": 8,
            "max_tokens": 128,
            "temperature": 0.8,
            "max_macro_checkpoints": 2,
        }
    ]
    assert '"arms": 8' in capsys.readouterr().out


def test_tree_grpo_command_routes_engineering_smoke(
    monkeypatch: object,
    capsys: object,
) -> None:
    """Tree 训练命令应读取配置并只执行一次命名 smoke。

    Args:
        monkeypatch (object): Pytest 提供的属性替换工具。
        capsys (object): Pytest 提供的标准输出捕获工具。

    Returns:
        None: 此测试不加载真实模型。
    """
    calls: list[object] = []
    sentinel = object()

    def load(path: Path) -> object:
        """记录配置路径并返回哨兵配置。

        Args:
            path (Path): CLI 传入的配置路径。

        Returns:
            object: 哨兵配置。
        """
        calls.append(path)
        return sentinel

    def train(config: object, name: str) -> dict[str, object]:
        """记录训练参数并返回最小摘要。

        Args:
            config (object): 已加载配置。
            name (str): smoke 名称。

        Returns:
            dict[str, object]: 可序列化摘要。
        """
        calls.extend((config, name))
        return {"optimizer_steps": 1, "name": name}

    monkeypatch.setattr(cli, "load_tree_grpo_config", load)
    monkeypatch.setattr(cli, "train_tree_grpo", train)

    result = cli.main(
        [
            "tree-grpo",
            "--config",
            "configs/rl/tree-grpo.toml",
            "--name",
            "tree-feasibility",
        ]
    )

    assert result == 0
    assert calls == [
        Path("configs/rl/tree-grpo.toml"),
        sentinel,
        "tree-feasibility",
    ]
    assert '"optimizer_steps": 1' in capsys.readouterr().out


def test_eval_tree_regret_command_routes_frozen_policy(
    monkeypatch: object,
    capsys: object,
) -> None:
    """Tree regret 命令应在落盘兄弟组上查询指定冻结 policy。

    Args:
        monkeypatch (object): Pytest 提供的属性替换工具。
        capsys (object): Pytest 提供的标准输出捕获工具。

    Returns:
        None: 此测试不连接推理服务。
    """
    calls: list[dict[str, object]] = []

    def evaluate(**kwargs: object) -> dict[str, object]:
        """记录 regret 评估参数。

        Args:
            **kwargs (object): CLI 传入的评估参数。

        Returns:
            dict[str, object]: 最小 regret 摘要。
        """
        calls.append(kwargs)
        return {"regret": 0.25}

    monkeypatch.setattr(cli, "evaluate_tree_branch_regret", evaluate)

    result = cli.main(
        [
            "eval-tree-regret",
            "--rollout",
            "runs/rl/tree/heldout.json",
            "--model-url",
            "http://127.0.0.1:8900",
            "--policy-model",
            "strategy-after",
            "--output",
            "runs/rl/tree/regret-after.json",
        ]
    )

    assert result == 0
    assert calls == [
        {
            "rollout_path": Path("runs/rl/tree/heldout.json"),
            "model_url": "http://127.0.0.1:8900",
            "policy_model": "strategy-after",
            "output_path": Path("runs/rl/tree/regret-after.json"),
            "max_tokens": 128,
        }
    ]
    assert '"regret": 0.25' in capsys.readouterr().out


def test_label_rl_dagger_command_routes_rollout_and_teacher_game(
    monkeypatch: object,
    capsys: object,
) -> None:
    """DAgger 命令应把学生 group、教师游戏和抽样预算交给标注器。

    Args:
        monkeypatch (object): Pytest 提供的属性替换工具。
        capsys (object): Pytest 提供的标准输出捕获工具。

    Returns:
        None: 此测试不启动真实游戏或 Solver。
    """
    calls: list[dict[str, object]] = []

    def label(**kwargs: object) -> dict[str, object]:
        """记录 CLI 参数并返回最小摘要。

        Args:
            **kwargs (object): CLI 传入的标注参数。

        Returns:
            dict[str, object]: 可序列化的标签摘要。
        """
        calls.append(kwargs)
        return {"labels": kwargs["max_labels"], "disagreements": 3}

    monkeypatch.setattr(cli, "label_dagger_rollout_group", label)

    result = cli.main(
        [
            "label-rl-dagger",
            "--rollout",
            "runs/rl/group.json",
            "--game-url",
            "http://127.0.0.1:8084",
            "--output",
            "runs/rl/dagger/labels.jsonl",
            "--max-labels",
            "8",
            "--selection-seed",
            "17",
            "--search-timeout",
            "140",
        ]
    )

    assert result == 0
    assert calls == [
        {
            "rollout_path": Path("runs/rl/group.json"),
            "game_url": "http://127.0.0.1:8084",
            "output_path": Path("runs/rl/dagger/labels.jsonl"),
            "max_labels": 8,
            "selection_seed": 17,
            "search_timeout": 140.0,
        }
    ]
    assert '"disagreements": 3' in capsys.readouterr().out


def test_build_sft_command_passes_explicit_mix_recipe(
    monkeypatch: object,
    capsys: object,
) -> None:
    """数据构建命令应把显式混合配方传给构建器。

    Args:
        monkeypatch (object): Pytest 提供的属性替换工具。
        capsys (object): Pytest 提供的标准输出捕获工具。

    Raises:
        AssertionError: CLI 未解析或未传递混合配方。

    Returns:
        None: 此测试不读取真实训练数据。
    """
    calls: list[Path | None] = []

    class Result:
        """提供 CLI 输出需要的最小构建结果。"""

        output_root = Path("data/datasets/sft")
        train_count = 10
        dev_count = 2
        test_count = 1

    def build(**kwargs: object) -> Result:
        """记录 CLI 传入的混合配方。

        Args:
            **kwargs (object): 数据构建关键字参数。

        Returns:
            Result: 最小构建结果。
        """
        calls.append(kwargs.get("mix_config_path"))
        return Result()

    monkeypatch.setattr(cli, "build_sft_dataset", build)

    result = cli.main(["build-sft", "--mix", "configs/sft/mix.toml"])

    assert result == 0
    assert calls == [Path("configs/sft/mix.toml")]
    assert '"train": 10' in capsys.readouterr().out


def test_build_sft_command_passes_dagger_root(
    monkeypatch: object,
) -> None:
    """SFT 构建命令应把旁路标签根传给数据聚合器。

    Args:
        monkeypatch (object): Pytest 提供的属性替换工具。

    Returns:
        None: 此测试只验证 CLI 到构建器的参数边界。
    """
    calls: list[tuple[Path | None, object]] = []

    class Result:
        """提供 CLI 输出需要的最小构建结果。"""

        output_root = Path("data/datasets/dagger-smoke/sft")
        train_count = 1
        dev_count = 0
        test_count = 0

    def build(**kwargs: object) -> Result:
        """记录 DAgger 根并返回最小结果。

        Args:
            **kwargs (object): CLI 传入的数据构建参数。

        Returns:
            Result: 最小构建结果。
        """
        calls.append((kwargs.get("dagger_root"), kwargs.get("dagger_policy_version")))
        return Result()

    monkeypatch.setattr(cli, "build_sft_dataset", build)

    result = cli.main(
        [
            "build-sft",
            "--dagger-root",
            "runs/rl/dagger",
            "--dagger-policy-model",
            "policy-test",
        ]
    )

    assert result == 0
    assert calls == [(Path("runs/rl/dagger"), "policy-test")]


def test_build_sft_command_passes_independent_run_splits(
    monkeypatch: object,
    capsys: object,
) -> None:
    """数据构建命令应把跨 raw 根实验名册传给构建器。

    Args:
        monkeypatch (object): Pytest 提供的属性替换工具。
        capsys (object): Pytest 提供的标准输出捕获工具。

    Raises:
        AssertionError: CLI 未解析或未传递独立分卷文件。

    Returns:
        None: 此测试不读取真实训练数据。
    """
    calls: list[Path | None] = []

    class Result:
        """提供 CLI 输出需要的最小构建结果。"""

        output_root = Path("data/datasets/experiment/sft")
        train_count = 12
        dev_count = 3
        test_count = 2

    def build(**kwargs: object) -> Result:
        """记录 CLI 传入的分卷文件。

        Args:
            **kwargs (object): 数据构建关键字参数。

        Returns:
            Result: 最小构建结果。
        """
        calls.append(kwargs.get("run_splits_path"))
        return Result()

    monkeypatch.setattr(cli, "build_sft_dataset", build)

    result = cli.main(
        [
            "build-sft",
            "--run-splits",
            "data/raw/human_combat_solver/experiment-splits.json",
        ]
    )

    assert result == 0
    assert calls == [Path("data/raw/human_combat_solver/experiment-splits.json")]
    assert '"train": 12' in capsys.readouterr().out


def test_build_sft_command_rejects_split_file_mixed_with_run_flags() -> None:
    """CLI 应在调用构建器前拒绝两套分卷输入同时出现。

    Raises:
        AssertionError: 冲突参数没有触发 argparse 失败。

    Returns:
        None: 此测试只检查命令行参数边界。
    """
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "build-sft",
                "--run-splits",
                "data/raw/human_combat_solver/experiment-splits.json",
                "--train-run",
                "FLAG-RUN",
            ]
        )

    assert exc_info.value.code == 2


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
            "models/adapters/parent",
            "--output",
            "models/merged/parent-merged",
        ]
    )

    assert result == 0
    assert calls == [
        (
            config,
            Path("models/adapters/parent"),
            Path("models/merged/parent-merged"),
        )
    ]
    assert '"output": "models/merged/parent-merged"' in capsys.readouterr().out


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
            "configs/sft/sft.toml",
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


def test_train_sft_cuda_command_uses_separate_backend(
    monkeypatch: object,
    capsys: object,
) -> None:
    """CUDA 子命令应只调用独立 CUDA 配置与训练入口。

    Args:
        monkeypatch (object): Pytest 提供的属性替换工具。
        capsys (object): Pytest 提供的标准输出捕获工具。

    Raises:
        AssertionError: CUDA 命令误用了 Mac 训练入口或没有传递续训参数。

    Returns:
        None: 此测试不访问真实 GPU。
    """
    config = object()
    calls: list[tuple[object, str, int | None, bool]] = []
    monkeypatch.setattr(cli, "load_cuda_sft_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "train_sft_cuda",
        lambda value, name, max_steps=None, exact_resume=False: (
            calls.append((value, name, max_steps, exact_resume))
            or {"device": "cuda:0", "optimizer_steps": 1}
        ),
    )

    result = cli.main(
        [
            "sft-cuda",
            "--config",
            "configs/sft/sft-cuda.toml",
            "--name",
            "20260828-cuda-smoke",
            "--max-steps",
            "1",
            "--resume",
        ]
    )

    assert result == 0
    assert calls == [(config, "20260828-cuda-smoke", 1, True)]
    assert '"device": "cuda:0"' in capsys.readouterr().out


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
    calls: list[tuple[object, Path, str, int | None, Path | None, float]] = []
    monkeypatch.setattr(cli, "load_sft_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "evaluate_sft",
        lambda value, adapter, split, max_samples=None, output_path=None, temperature=0.0: (
            calls.append((value, adapter, split, max_samples, output_path, temperature))
            or {"samples": 2}
        ),
    )

    result = cli.main(
        [
            "eval-sft",
            "--config",
            "configs/sft/sft.toml",
            "--adapter",
            "models/adapters/demo",
            "--split",
            "eval",
            "--max-samples",
            "2",
            "--output",
            "runs/eval/demo.jsonl",
            "--temperature",
            "0.8",
        ]
    )

    assert result == 0
    assert calls == [
        (
            config,
            Path("models/adapters/demo"),
            "eval",
            2,
            Path("runs/eval/demo.jsonl"),
            0.8,
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
            "validation",
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
            "validation",
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
            Path("data/datasets/sft/eval"),
            Path("runs/eval/e2-knowledge.json"),
            "auto",
            10,
            96,
        )
    ]
    assert '"questions": 10' in capsys.readouterr().out
