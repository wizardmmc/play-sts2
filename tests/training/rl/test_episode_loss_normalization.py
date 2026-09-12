"""用真实 autograd 验证多步终局信用不会惩罚所有轨迹共有的前缀。"""

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from play_sts2.training.rl import battle_trainer
from play_sts2.training.rl.learner import GrpoArm, GrpoSequence


def test_group_normalization_cancels_shared_prefix_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """长成功轨迹和短失败轨迹共有的动作应相消，成功后的独有动作仍获正信用。"""
    model = torch.nn.ParameterDict(
        {
            "prefix": torch.nn.Parameter(torch.tensor(0.0)),
            "later": torch.nn.Parameter(torch.tensor(0.0)),
        }
    )

    def logprobs(
        model: object, sequence: GrpoSequence, **kwargs: object
    ) -> torch.Tensor:
        """只替代大模型前向，保留伯努利策略的实际可微 log-prob。"""
        return torch.nn.functional.logsigmoid(
            model["prefix" if sequence.step_index == 0 else "later"]
        ).reshape(1)

    monkeypatch.setattr(battle_trainer, "_sequence_logprobs", logprobs)
    arms = []
    for index in range(8):
        advantage = 1.0 if index < 4 else -1.0
        sequences = tuple(
            GrpoSequence(
                arm_index=index,
                step_index=step,
                input_ids=(10, 11),
                assistant_mask=(False, True),
                behavior_logprobs=(-math.log(2),),
                allowed_token_ids=((11, 12),),
                temperature=1.0,
                advantage=advantage,
            )
            for step in range(2 if index < 4 else 1)
        )
        arms.append(GrpoArm(index, advantage, advantage, sequences))
    config = SimpleNamespace(
        logits_chunk_size=1, clip=0.2, kl_beta=0.0, loss_normalization="group_tokens"
    )
    battle_trainer.backward_grpo_group(
        model,
        battle_trainer.FrozenAnchor(model),
        arms,
        config=config,
        device="cpu",
    )
    assert model["prefix"].grad.item() == pytest.approx(0.0, abs=1e-7)
    assert model["later"].grad.item() == pytest.approx(-1.0 / 6.0, abs=1e-7)


@pytest.mark.parametrize("kind", ["battle", "strategy"])
def test_normalization_config_is_explicit_and_validated(
    tmp_path: Path, kind: str
) -> None:
    """两类 learner 都须读取实验选项，未知值不能静默回到历史公式。"""
    from play_sts2.training.rl import (
        GrpoTrainingError,
        load_battle_grpo_config,
        load_strategy_grpo_config,
    )

    template = Path(__file__).parents[3] / "configs-template/rl" / f"{kind}-grpo.toml"
    text = "\n".join(
        line
        for line in template.read_text().splitlines()
        if not line.startswith("loss_normalization")
    )
    path = tmp_path / "config.toml"
    path.write_text(text + '\nloss_normalization = "group_tokens"\n')
    loader = load_battle_grpo_config if kind == "battle" else load_strategy_grpo_config
    assert loader(path).loss_normalization == "group_tokens"
    path.write_text(text + '\nloss_normalization = "unknown"\n')
    with pytest.raises(GrpoTrainingError):
        loader(path)


def test_legacy_checkpoint_cannot_resume_with_changed_loss(tmp_path: Path) -> None:
    """缺少新字段的旧 checkpoint 只按 per_arm 恢复，不能中途改变目标。"""
    from play_sts2.training.rl import GrpoTrainingError, load_battle_grpo_config

    template = Path(__file__).parents[3] / "configs-template/rl/battle-grpo.toml"
    config = replace(load_battle_grpo_config(template), loss_normalization="per_arm")
    saved = battle_trainer._config_json(config)
    saved.pop("loss_normalization")
    (tmp_path / "config.json").write_text(json.dumps(saved))
    paths = [Path("group.json")]
    (tmp_path / "inputs.json").write_text(
        json.dumps(
            {"policy_model": config.policy_model, "rollout_paths": ["group.json"]}
        )
    )
    battle_trainer._validate_resume_inputs(config, tmp_path, paths)
    with pytest.raises(GrpoTrainingError, match="配置"):
        battle_trainer._validate_resume_inputs(
            replace(config, loss_normalization="group_tokens"), tmp_path, paths
        )


def test_group_denominator_preserves_tree_importance_mass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不同长度的分层计划仍保持来源 loss_scale，发布门禁采用相同 token 权重。"""
    model = torch.nn.ParameterDict(
        {
            "root": torch.nn.Parameter(torch.tensor(math.log(4.0))),
            "later": torch.nn.Parameter(torch.tensor(0.0)),
        }
    )

    def logprobs(
        model: object, sequence: GrpoSequence, **kwargs: object
    ) -> torch.Tensor:
        """根域是概率0.2/0.8的二选一，后续动作初始概率0.5。"""
        value = (
            model["later"]
            if sequence.step_index
            else model["root"] * (-1 if sequence.arm_index == 0 else 1)
        )
        return torch.nn.functional.logsigmoid(value).reshape(1)

    monkeypatch.setattr(battle_trainer, "_sequence_logprobs", logprobs)
    arms = []
    for index, length in enumerate((1, 3)):
        advantage = -1.0 if index == 0 else 1.0
        sequences = tuple(
            GrpoSequence(
                arm_index=index,
                step_index=step,
                input_ids=(10, 11),
                assistant_mask=(False, True),
                behavior_logprobs=(
                    math.log(0.2 if index == 0 else 0.8) if step == 0 else -math.log(2),
                ),
                allowed_token_ids=((11, 12),),
                temperature=1.0,
                advantage=advantage,
            )
            for step in range(length)
        )
        arms.append(
            GrpoArm(
                index,
                advantage,
                advantage,
                sequences,
                proposal_probability=0.5,
                recompute_root_probability=True,
            )
        )
    anchor = battle_trainer.FrozenAnchor(model)
    config = SimpleNamespace(
        logits_chunk_size=1, clip=0.2, kl_beta=0.0, loss_normalization="group_tokens"
    )
    battle_trainer.backward_grpo_group(
        model, anchor, arms, config=config, device="cpu", loss_scale=0.25
    )
    assert model["root"].grad.item() == pytest.approx(-0.25 * 0.32 / 2.6, abs=1e-7)
    assert model["later"].grad.item() == pytest.approx(-0.25 * 0.8 / 2.6, abs=1e-7)
    model["later"].data.fill_(math.log(3.0))
    metric = battle_trainer.evaluate_grpo_group(
        model, anchor, arms, config=config, device="cpu"
    )
    assert metric["ratio_mean"] == pytest.approx(1.0 + 0.8 / 2.6, abs=1e-6)
