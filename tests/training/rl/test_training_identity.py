"""验证正式 RL 在更新之前拒绝模型路径或行为分布错配。"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_formal_training_rejects_bare_base_before_model_loading(tmp_path: Path) -> None:
    """正式 RL 的 base 必须有原生 SFT 合并收据，不能只靠目录别名。"""
    from play_sts2.training.rl import GrpoTrainingError
    from play_sts2.training.rl.battle_trainer import validate_training_policy_identity

    config = SimpleNamespace(
        run_role="formal",
        init_adapter=Path("models/policy-3"),
        policy_model="policy-3",
        base_model=tmp_path,
    )
    with pytest.raises(GrpoTrainingError, match="SFT"):
        validate_training_policy_identity(config, check_base=True)
    (tmp_path / "merge_manifest.json").write_text(
        '{"adapter":"models/adapters/sft-e7","merge":"peft.merge_and_unload(safe_merge=True)"}'
    )
    validate_training_policy_identity(config, check_base=True)


@pytest.mark.parametrize("kind", ["battle", "strategy"])
def test_formal_config_rejects_old_adapter_under_new_policy_name(
    tmp_path: Path, kind: str
) -> None:
    """S20/B20 标签不能让 S3/B3 初始化通过正式 learner 准入。"""
    from play_sts2.training.rl import (
        GrpoTrainingError,
        load_battle_grpo_config,
        load_strategy_grpo_config,
    )

    path = tmp_path / "config.toml"
    path.write_text("""base_model = "models/base/e7"
init_adapter = "models/adapters/policy-3"
policy_model = "policy-20"
run_role = "formal"
adapter_root = "models/adapters"
runs_root = "runs/train"
device = "cuda:0"
learning_rate = 0.0001
max_length = 12288
seed = 42
rollout_root = "runs/battle"
reward_scheme = "core_no_turn"
gigpo_path = "runs/backbone.json"
tree_root = "runs/tree"
""")
    loader = load_battle_grpo_config if kind == "battle" else load_strategy_grpo_config
    with pytest.raises(GrpoTrainingError, match="init_adapter"):
        loader(path)
    path.write_text(
        path.read_text().replace(
            'policy_model = "policy-20"', 'policy_model = "policy-3"'
        )
    )
    assert loader(path).policy_model == "policy-3"


@pytest.mark.parametrize("ratio", [0.5, 1.5, float("nan")])
def test_initial_ratio_mismatch_cannot_change_model(
    ratio: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """即便已算出梯度，首组明显错配也必须在 optimizer.step 前停止。"""
    from play_sts2.training.rl import GrpoTrainingError, battle_trainer

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def backward(*args: object, **kwargs: object) -> dict[str, float]:
        """隔离外部模型前向，提供可实际改变参数的梯度和错配指标。"""
        model.weight.grad = torch.ones_like(model.weight)
        return {"ratio_mean": ratio, "kl": 0.0}

    monkeypatch.setattr(battle_trainer, "backward_grpo_group", backward)
    with pytest.raises(GrpoTrainingError, match="ratio"):
        battle_trainer.optimize_grpo_group(
            model,
            optimizer,
            battle_trainer.FrozenAnchor(model),
            [None] * 8,
            [],
            dagger_cursor=0,
            config=SimpleNamespace(dagger_weight=0, max_grad_norm=1),
            device="cpu",
            verify_initial_policy=True,
        )
    assert model.weight.item() == 1.0


def test_final_battle_gate_restores_parent_on_large_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """最后一次更新越界时必须恢复父权重，不能发布无最终门禁的候选。"""
    from play_sts2.training.rl import GrpoTrainingError, battle_trainer

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    anchor = battle_trainer.FrozenAnchor(model)
    model.weight.data.fill_(2.0)
    monkeypatch.setattr(
        battle_trainer,
        "evaluate_grpo_group",
        lambda *args, **kwargs: {"kl": 0.1, "ratio_mean": 1.0},
    )
    with pytest.raises(GrpoTrainingError, match="post-step"):
        battle_trainer.verify_final_battle_update(
            model,
            anchor,
            SimpleNamespace(arms=()),
            config=None,
            device="cpu",
        )
    assert model.weight.item() == 1.0
