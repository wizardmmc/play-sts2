"""验证战斗 GRPO 训练配置、冻结锚与精确 checkpoint。"""

import io
import json
from pathlib import Path

import pytest
import torch

from play_sts2.training import rl
from play_sts2.training.rl.battle_trainer import (
    _checkpoint_due,
    _importance_estimator_weights,
    _record_battle_training_metrics,
    _validate_dagger_versions,
)


class _TinyAdapter(torch.nn.Module):
    """提供一个可修改的微型可训练 adapter。"""

    def __init__(self) -> None:
        """创建确定值的单参数模块。"""
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))


class _MetricsWriter:
    """记录 battle trainer 交给 TensorBoard 的标量对象。"""

    def __init__(self) -> None:
        """初始化空调用记录。"""
        self.calls: list[tuple[dict[str, object], int]] = []

    def write(self, metrics: dict[str, object], *, step: int) -> None:
        """保存一次 TensorBoard 写入。

        Args:
            metrics (dict[str, object]): 当前指标对象。
            step (int): optimizer step。

        Returns:
            None: 调用已记录。
        """
        self.calls.append((metrics, step))


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
    assert config.checkpoint_groups == 50


def test_formal_battle_training_rejects_tiny_checkpoint_interval(
    tmp_path: Path,
) -> None:
    """正式长训不能每个 group 重写一次完整训练 checkpoint。

    Args:
        tmp_path (Path): 临时配置目录。

    Returns:
        None: 小于二十个完整 group 的间隔只允许工程恢复测试使用。
    """
    path = tmp_path / "formal.toml"
    path.write_text(
        """base_model = "models/base/model"
init_adapter = "models/adapters/parent"
policy_model = "policy-test"
run_role = "formal"
rollout_root = "runs/rl/groups"
adapter_root = "models/adapters"
runs_root = "runs/rl/train"
device = "cuda:0"
learning_rate = 0.00001
max_length = 12288
checkpoint_groups = 1
seed = 20260830
clip = 0.2
kl_beta = 0.02
dagger_weight = 0.0
dagger_samples_per_group = 1
reward_scheme = "core"
potion_cost = 0.25
""",
        encoding="utf-8",
    )

    with pytest.raises(rl.GrpoTrainingError, match="checkpoint_groups"):
        rl.load_battle_grpo_config(path)


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
    assert not _checkpoint_due(12, total_groups=12, interval=10, pausing=False)


def test_importance_estimator_does_not_self_normalize_sample_weights() -> None:
    """分层 Tree 应使用 ``importance/K`` 而非除以当前样本权重和。

    Returns:
        None: 权重和可以偏离一，保留无偏 proposal correction 定义。
    """
    weights = _importance_estimator_weights((0.5, 1.0, 2.0))

    assert weights == pytest.approx((1 / 6, 1 / 3, 2 / 3))
    assert sum(weights) == pytest.approx(7 / 6)


def test_battle_metrics_share_one_jsonl_and_tensorboard_record() -> None:
    """每个 optimizer step 应把同一口径同时写入 JSONL 与 TensorBoard。

    Returns:
        None: 两个输出使用同一个 record，不触发完整 checkpoint。
    """
    trace = io.StringIO()
    writer = _MetricsWriter()
    record = {
        "step": 3,
        "group_id": "battle-003",
        "reward_mean": -1.5,
        "policy_loss": 0.25,
        "kl": 0.01,
    }

    _record_battle_training_metrics(trace, writer, record)

    assert json.loads(trace.getvalue()) == record
    assert writer.calls == [({"battle": record}, 3)]
