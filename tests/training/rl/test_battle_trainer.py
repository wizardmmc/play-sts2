"""验证战斗 GRPO 训练配置、冻结锚与精确 checkpoint。"""

from pathlib import Path

import pytest
import torch

from play_sts2.training import rl
from play_sts2.training.rl.battle_trainer import (
    _checkpoint_due,
    _validate_dagger_versions,
)


class _TinyAdapter(torch.nn.Module):
    """提供一个可修改的微型可训练 adapter。"""

    def __init__(self) -> None:
        """创建确定值的单参数模块。"""
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))


def test_load_battle_grpo_config(tmp_path: Path) -> None:
    """配置加载器应完整校验训练、DAgger 和输出边界。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Returns:
        None: 配置与默认值符合单卡训练契约。
    """
    path = tmp_path / "battle.toml"
    path.write_text(
        """base_model = "models/base/model"
init_adapter = "models/adapters/parent"
policy_model = "policy-test"
run_role = "engineering_smoke"
rollout_root = "runs/rl/groups"
dagger_root = "runs/rl/dagger/labels.jsonl"
adapter_root = "models/adapters"
runs_root = "runs/rl/train"
device = "cuda:3"
learning_rate = 0.00001
max_length = 12288
max_grad_norm = 1.0
logits_chunk_size = 128
checkpoint_groups = 1
seed = 20260830
clip = 0.2
kl_beta = 0.02
dagger_weight = 0.1
dagger_samples_per_group = 2
reward_scheme = "core"
potion_cost = 0.25
""",
        encoding="utf-8",
    )

    config = rl.load_battle_grpo_config(path)

    assert config.device == "cuda:3"
    assert config.init_adapter == Path("models/adapters/parent")
    assert config.run_role == "engineering_smoke"
    assert config.dagger_weight == pytest.approx(0.1)
    assert config.reward_scheme == "core"


def test_frozen_anchor_swap_restores_current_trainable_state() -> None:
    """冻结锚前向结束后必须恢复当前 learner 参数。

    Returns:
        None: 锚状态保持初值，当前状态在上下文退出后恢复。
    """
    model = _TinyAdapter()
    anchor = rl.FrozenAnchor(model)
    with torch.no_grad():
        model.adapter.add_(10.0)

    with anchor.applied():
        assert model.adapter.tolist() == pytest.approx([1.0, 2.0])
    assert model.adapter.tolist() == pytest.approx([11.0, 12.0])
    assert anchor.state_dict()["adapter"].tolist() == pytest.approx([1.0, 2.0])


def test_grpo_checkpoint_round_trip() -> None:
    """checkpoint 应保存组游标、优化器、随机状态和冻结锚。

    Returns:
        None: payload 能无损恢复精确训练状态。
    """
    state = rl.BattleGrpoCheckpoint(
        next_group_index=2,
        optimizer_steps=2,
        dagger_cursor=4,
        torch_rng_state=torch.tensor([1, 2], dtype=torch.uint8),
        device_rng_state=None,
        optimizer_state={"state": {}, "param_groups": []},
        anchor_state={"adapter": torch.tensor([1.0])},
    )

    restored = rl.BattleGrpoCheckpoint.from_payload(state.to_payload())

    assert restored.next_group_index == 2
    assert restored.optimizer_steps == 2
    assert restored.dagger_cursor == 4
    assert restored.anchor_state["adapter"].tolist() == [1.0]


def test_dagger_versions_reject_old_game_labels() -> None:
    """相同 policy 名也不能让旧游戏版本标签进入 DAgger CE。

    Returns:
        None: DAgger 正式边界固定为 v0.111.0。
    """
    rows = (
        {
            "game_version": "v0.107.1",
            "mod_version": "mod-test",
            "protocol_version": "protocol-test",
            "harness_version": "harness-test",
        },
    )

    with pytest.raises(rl.GrpoTrainingError, match="v0.111.0"):
        _validate_dagger_versions(rows)


def test_pause_forces_checkpoint_before_interval() -> None:
    """max_groups 主动暂停不能等到周期整除才保存恢复点。

    Returns:
        None: group 1 在 interval 10 下暂停仍必须 checkpoint。
    """
    assert _checkpoint_due(1, total_groups=12, interval=10, pausing=True)
    assert not _checkpoint_due(1, total_groups=12, interval=10, pausing=False)
