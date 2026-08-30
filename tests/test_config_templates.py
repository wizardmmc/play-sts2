"""验证可提交配置模板与生产加载器保持兼容。"""

from pathlib import Path

from play_sts2.inference.config import load_inference_config
from play_sts2.training.rl.strategy import load_tree_grpo_config
from play_sts2.training.sft import load_sft_config, load_sft_mix
from play_sts2.training.sft_cuda import load_cuda_sft_config


def test_checked_in_toml_templates_load_without_local_configs() -> None:
    """所有 TOML 模板都应直接通过对应的生产配置加载器。

    Returns:
        None: 此测试保证新环境复制模板后不会先遇到结构错误。
    """
    repository = Path(__file__).resolve().parents[1]
    templates = repository / "configs-template"

    inference = load_inference_config(templates / "inference/inference.toml")
    sft = load_sft_config(templates / "sft/sft.toml")
    cuda = load_cuda_sft_config(templates / "sft/sft-cuda.toml")
    mix = load_sft_mix(templates / "sft/mix.toml")
    tree = load_tree_grpo_config(templates / "rl/tree-grpo.toml")

    assert inference.default_profile == "no-think"
    assert sft.device == "auto"
    assert cuda.device == "cuda:0"
    assert mix.human_train_action_limits["play_card"] == 600
    assert tree.run_role == "engineering_smoke"
    assert tree.device == "cuda:0"
