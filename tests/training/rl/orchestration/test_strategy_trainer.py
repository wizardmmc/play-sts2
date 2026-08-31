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
