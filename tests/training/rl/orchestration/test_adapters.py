"""验证双 residual 的 composite serving LoRA 数学。"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_KEY = "base_model.model.model.layers.0.linear_attn.in_proj_a"


def test_compose_serving_adapter_preserves_sum_of_two_lora_deltas(
    tmp_path: Path,
) -> None:
    """rank 拼接后的单 LoRA 应与 SFT LoRA 加 residual 的函数完全相同。

    Args:
        tmp_path (Path): 两个输入 adapter 与输出目录。

    Returns:
        None: 手工矩阵前向在浮点容差内一致，且输出键使用 vLLM 模块路径。
    """
    from play_sts2.training.rl.orchestration import compose_serving_adapter

    parent = _adapter(
        tmp_path / "parent",
        a=torch.tensor([[1.0, 2.0]]),
        b=torch.tensor([[3.0]]),
        alpha=2,
    )
    residual = _adapter(
        tmp_path / "residual",
        a=torch.tensor([[4.0, 5.0]]),
        b=torch.tensor([[6.0]]),
        alpha=1,
        reverse_targets=True,
    )

    output = tmp_path / "composite"
    compose_serving_adapter(
        parent_adapter=parent,
        residual_adapter=residual,
        output=output,
        name="qwen3.5-s1",
    )

    state = load_file(output / "adapter_model.safetensors")
    config = json.loads((output / "adapter_config.json").read_text(encoding="utf-8"))
    a = state[
        f"{_KEY}.lora_A.weight".replace("model.layers", "language_model.model.layers")
    ]
    b = state[
        f"{_KEY}.lora_B.weight".replace("model.layers", "language_model.model.layers")
    ]
    assert set(state) == {
        f"{_KEY}.lora_A.weight".replace("model.layers", "language_model.model.layers"),
        f"{_KEY}.lora_B.weight".replace("model.layers", "language_model.model.layers"),
    }
    vector = torch.tensor([0.5, -1.0])
    expected = 2.0 * (torch.tensor([[3.0]]) @ torch.tensor([[1.0, 2.0]]) @ vector)
    expected += torch.tensor([[6.0]]) @ torch.tensor([[4.0, 5.0]]) @ vector
    actual = config["lora_alpha"] / config["r"] * (b @ a @ vector)

    assert config["r"] == 2
    assert config["lora_alpha"] == 2
    assert torch.allclose(actual, expected)


def _adapter(
    root: Path,
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: int,
    reverse_targets: bool = False,
) -> Path:
    """写入一个单层 rank-1 测试 LoRA。

    Args:
        root (Path): adapter 目录。
        a (torch.Tensor): LoRA A 矩阵。
        b (torch.Tensor): LoRA B 矩阵。
        alpha (int): LoRA alpha。
        reverse_targets (bool): 是否用相反顺序保存等价目标模块集合。

    Returns:
        Path: 完整 adapter 目录。
    """
    root.mkdir()
    (root / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 1,
                "lora_alpha": alpha,
                "use_rslora": False,
                "use_dora": False,
                "lora_dropout": 0.0,
                "target_modules": (
                    ["other", "in_proj_a"]
                    if reverse_targets
                    else ["in_proj_a", "other"]
                ),
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            f"{_KEY}.lora_A.weight": a,
            f"{_KEY}.lora_B.weight": b,
        },
        root / "adapter_model.safetensors",
    )
    return root
