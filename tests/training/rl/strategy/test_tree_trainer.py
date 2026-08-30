"""验证 Tree-GRPO E6 可行性训练配置与边界。"""

from pathlib import Path

import pytest


def test_load_tree_config_requires_engineering_smoke_and_single_cuda(
    tmp_path: Path,
) -> None:
    """第六阶段配置应固定 E6 父 policy、单卡和工程 smoke 身份。

    Args:
        tmp_path (Path): 临时 TOML 目录。

    Returns:
        None: 配置解析为一组 Tree update 所需字段。
    """
    from play_sts2.training.rl.strategy.trainer import load_tree_grpo_config

    path = tmp_path / "tree.toml"
    path.write_text(_config_text(), encoding="utf-8")

    config = load_tree_grpo_config(path)

    assert config.base_model == Path("/models/qwen3.5-4b")
    assert config.init_adapter == Path("/models/e6")
    assert config.policy_model == "e6-feasibility"
    assert config.rollout_path == Path("/runs/tree.json")
    assert config.device == "cuda:2"
    assert config.run_role == "engineering_smoke"


def test_tree_config_rejects_formal_role_in_e6_feasibility(tmp_path: Path) -> None:
    """E6 尚未成为最终父模型时不得把 Tree smoke 标成正式训练。

    Args:
        tmp_path (Path): 临时 TOML 目录。

    Returns:
        None: formal 角色在加载阶段被拒绝。
    """
    from play_sts2.training.rl.learner import GrpoTrainingError
    from play_sts2.training.rl.strategy.trainer import load_tree_grpo_config

    path = tmp_path / "tree.toml"
    path.write_text(
        _config_text().replace("engineering_smoke", "formal"),
        encoding="utf-8",
    )

    with pytest.raises(GrpoTrainingError, match="engineering_smoke"):
        load_tree_grpo_config(path)


def _config_text() -> str:
    """返回可读的单卡 Tree smoke 配置。

    Returns:
        str: 完整 TOML 文本。
    """
    return """
base_model = "/models/qwen3.5-4b"
init_adapter = "/models/e6"
policy_model = "e6-feasibility"
run_role = "engineering_smoke"
rollout_path = "/runs/tree.json"
adapter_root = "/models/adapters"
runs_root = "/runs/train"
device = "cuda:2"
learning_rate = 0.00001
max_length = 12288
max_grad_norm = 1.0
logits_chunk_size = 128
seed = 20260830
clip = 0.2
kl_beta = 0.02
""".strip()
