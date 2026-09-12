"""验证阶段七组合战略训练配置。"""

from pathlib import Path
from types import SimpleNamespace

import pytest


def test_load_strategy_config_keeps_fixed_source_weights(tmp_path: Path) -> None:
    """GiGPO 与 Tree 应按固定数据源权重组合而不是按 token 数竞争。

    Args:
        tmp_path (Path): 临时配置目录。

    Returns:
        None: 默认两个有效数据源各占一半。
    """
    from play_sts2.training.rl.orchestration import load_strategy_grpo_config

    path = tmp_path / "strategy.toml"
    path.write_text(
        """base_model = "models/merged/frozen"
init_adapter = "models/adapters/qwen3.5-s0"
policy_model = "qwen3.5-s0"
run_role = "engineering_smoke"
gigpo_path = "runs/cycle/backbone.json"
tree_root = "runs/cycle/tree"
adapter_root = "models/adapters"
runs_root = "runs/train"
device = "cuda:1"
learning_rate = 0.00001
max_length = 12288
max_grad_norm = 1.0
logits_chunk_size = 128
seed = 20260830
clip = 0.2
kl_beta = 0.02
""",
        encoding="utf-8",
    )

    config = load_strategy_grpo_config(path)

    assert config.backbone_loss_weight == pytest.approx(0.5)
    assert config.tree_loss_weight == pytest.approx(0.5)
    assert config.device == "cuda:1"


def test_combined_strategy_groups_require_complete_same_receipt() -> None:
    """同一步战略更新不能混入不同战斗 policy 或 thinking 配置。

    Returns:
        None: 完整 ``S/B/profile/environment`` 收据参与准入。
    """
    from play_sts2.training.rl import GrpoTrainingError
    from play_sts2.training.rl.orchestration.strategy_trainer import (
        _validate_strategy_group_receipts,
    )

    environment = {
        "game_version": "v0.111.0",
        "mod_version": "mod-test",
        "protocol_version": "protocol-test",
        "structured_output_backend": "xgrammar",
        "structured_output_version": "0.1.33",
    }
    backbone = SimpleNamespace(
        policy_version="qwen3.5-s0",
        battle_policy_version="qwen3.5-b0",
        generation_profile={"temperature": 0.8, "thinking_enabled": False},
        environment=environment,
    )
    matching_tree = SimpleNamespace(
        policy_version="qwen3.5-s0",
        battle_policy_version="qwen3.5-b0",
        generation_profile={"temperature": 0.8, "thinking_enabled": False},
        environment=dict(environment),
    )

    _validate_strategy_group_receipts(
        (backbone, matching_tree),
        expected_strategy="qwen3.5-s0",
    )

    mismatched_tree = SimpleNamespace(
        **{
            **matching_tree.__dict__,
            "generation_profile": {
                "temperature": 0.8,
                "thinking_enabled": None,
            },
        }
    )
    with pytest.raises(GrpoTrainingError, match="B_n"):
        _validate_strategy_group_receipts(
            (backbone, mismatched_tree),
            expected_strategy="qwen3.5-s0",
        )


def test_strategy_update_gate_uses_post_step_metrics() -> None:
    """战略 adapter 只能在更新后 KL 与 ratio 均通过时发布。

    Returns:
        None: 预更新恒为零的 KL 不能替代 post-step 硬门控。
    """
    from play_sts2.training.rl.orchestration.strategy_trainer import (
        _post_update_metrics_safe,
    )

    assert _post_update_metrics_safe({"kl": 0.01, "ratio_mean": 1.1})
    assert not _post_update_metrics_safe({"kl": 0.06, "ratio_mean": 1.0})
    assert not _post_update_metrics_safe({"kl": 0.01, "ratio_mean": 2.1})


def test_strategy_tensorboard_payload_separates_backbone_tree_and_post_step() -> None:
    """战略曲线应区分两个数据源和真正约束发布的更新后指标。

    Returns:
        None: 不能用 pre-step KL 覆盖 post-step KL。
    """
    from play_sts2.training.rl.orchestration.strategy_trainer import (
        strategy_tensorboard_payload,
    )

    payload = strategy_tensorboard_payload(
        {
            "backbone_metric": {"policy_loss": 0.2, "kl": 0.0},
            "tree_metric": {"policy_loss": -0.1, "kl": 0.0},
            "post_backbone_metric": {"kl": 0.01, "ratio_mean": 1.1},
            "post_tree_metric": {"kl": 0.02, "ratio_mean": 0.9},
            "grad_norm": 0.5,
        }
    )

    assert payload["strategy/backbone"]["policy_loss"] == 0.2
    assert payload["strategy/post_backbone"]["kl"] == 0.01
    assert payload["strategy/post_tree"]["ratio_mean"] == 0.9
    assert payload["strategy"]["grad_norm"] == 0.5


def test_load_strategy_config_reads_optimizer_steps_bounds(tmp_path: Path) -> None:
    """多 epoch 更新的步数必须在 1 到 4 之间且可从 TOML 读取。

    Args:
        tmp_path (Path): 临时配置目录。

    Returns:
        None: 默认一步；越界值直接拒绝。
    """
    from play_sts2.training.rl.learner import GrpoTrainingError
    from play_sts2.training.rl.orchestration import load_strategy_grpo_config

    base = """base_model = "models/merged/frozen"
init_adapter = "models/adapters/qwen3.5-s0"
policy_model = "qwen3.5-s0"
run_role = "engineering_smoke"
gigpo_path = "runs/cycle/backbone.json"
tree_root = "runs/cycle/tree"
adapter_root = "models/adapters"
runs_root = "runs/train"
device = "cuda:0"
learning_rate = 0.00001
max_length = 12288
seed = 20260830
"""
    path = tmp_path / "default.toml"
    path.write_text(base, encoding="utf-8")
    assert load_strategy_grpo_config(path).optimizer_steps == 1

    path = tmp_path / "three.toml"
    path.write_text(base + "optimizer_steps = 3\n", encoding="utf-8")
    assert load_strategy_grpo_config(path).optimizer_steps == 3

    path = tmp_path / "zero.toml"
    path.write_text(base + "optimizer_steps = 0\n", encoding="utf-8")
    with pytest.raises(GrpoTrainingError):
        load_strategy_grpo_config(path)

    path = tmp_path / "five.toml"
    path.write_text(base + "optimizer_steps = 5\n", encoding="utf-8")
    with pytest.raises(GrpoTrainingError):
        load_strategy_grpo_config(path)


def test_multi_group_config_requires_divisible_single_pass(tmp_path: Path) -> None:
    """多组模式必须整除、步数等于组数除以每组步长，且与单组路径互斥。

    Args:
        tmp_path (Path): 临时配置目录。

    Returns:
        None: 合法多组配置通过；不整除、步数不符、双写或双空均拒绝。
    """
    from play_sts2.training.rl.learner import GrpoTrainingError
    from play_sts2.training.rl.orchestration import load_strategy_grpo_config

    base = """base_model = "models/merged/frozen"
init_adapter = "models/adapters/qwen3.5-s1"
policy_model = "qwen3.5-e7-sgroup-next1"
run_role = "engineering_smoke"
tree_root = "runs/cycle/tree"
adapter_root = "models/adapters"
runs_root = "runs/train"
device = "cuda:0"
learning_rate = 0.0001
max_length = 12288
seed = 20260909
"""
    valid = (
        base
        + 'gigpo_paths = ["runs/a/backbone.json", "runs/b/backbone.json"]\n'
        + "groups_per_step = 1\noptimizer_steps = 2\n"
    )
    path = tmp_path / "multi.toml"
    path.write_text(valid, encoding="utf-8")
    config = load_strategy_grpo_config(path)
    assert config.gigpo_paths == (
        Path("runs/a/backbone.json"),
        Path("runs/b/backbone.json"),
    )
    assert config.groups_per_step == 1

    bad_indivisible = (
        base
        + 'gigpo_paths = ["runs/a/backbone.json", "runs/b/backbone.json", "runs/c/backbone.json"]\n'
        + "groups_per_step = 2\noptimizer_steps = 2\n"
    )
    path = tmp_path / "indivisible.toml"
    path.write_text(bad_indivisible, encoding="utf-8")
    with pytest.raises(GrpoTrainingError, match="多组模式"):
        load_strategy_grpo_config(path)

    bad_steps = (
        base
        + 'gigpo_paths = ["runs/a/backbone.json", "runs/b/backbone.json"]\n'
        + "groups_per_step = 1\noptimizer_steps = 4\n"
    )
    path = tmp_path / "steps-mismatch.toml"
    path.write_text(bad_steps, encoding="utf-8")
    with pytest.raises(GrpoTrainingError, match="多组模式"):
        load_strategy_grpo_config(path)

    both = (
        base
        + 'gigpo_path = "runs/a/backbone.json"\n'
        + 'gigpo_paths = ["runs/a/backbone.json", "runs/b/backbone.json"]\n'
        + "groups_per_step = 1\noptimizer_steps = 2\n"
    )
    path = tmp_path / "both.toml"
    path.write_text(both, encoding="utf-8")
    with pytest.raises(GrpoTrainingError, match="二选一"):
        load_strategy_grpo_config(path)

    path = tmp_path / "neither.toml"
    path.write_text(base, encoding="utf-8")
    with pytest.raises(GrpoTrainingError, match="二选一"):
        load_strategy_grpo_config(path)


def test_backbone_step_plans_chunk_multi_and_repeat_single() -> None:
    """步骤计划在多组模式切块单遍、单组模式重复同组。

    Returns:
        None: 36 组 3 组/步得 12 步全数据一遍；单组 3 epoch 得三步同索引。
    """
    from play_sts2.training.rl.orchestration.strategy_trainer import (
        _backbone_step_plans,
    )

    multi = _backbone_step_plans(
        group_count=36, groups_per_step=3, single_group_epochs=1
    )
    assert len(multi) == 12
    assert multi[0] == (0, 1, 2)
    assert multi[-1] == (33, 34, 35)
    assert sorted(index for step in multi for index in step) == list(range(36))

    single = _backbone_step_plans(
        group_count=1, groups_per_step=1, single_group_epochs=3
    )
    assert single == [(0,), (0,), (0,)]
