"""定义第三阶段可审计的战斗终局奖励消融。"""

from dataclasses import dataclass
from typing import Literal

from .contracts import BattleReward, RewardComponent

BattleRewardScheme = Literal[
    "core",
    "core_no_turn",
    "potion_cost",
    "legacy_remaining_potion",
]
_MODEL_ERROR_REWARD_FLOOR = -25.0


@dataclass(frozen=True, slots=True)
class BattleRewardInput:
    """保存一场战斗奖励所需的最小终局事实。

    Args:
        entry_hp (int): 战斗入口当前生命。
        max_hp (int): 战斗入口最大生命。
        final_hp (int): 正常离场生命；死亡或模型失败为零。
        turns (int): 本场出现的最大回合编号。
        cleared (bool): 是否正常击败全部敌人。
        died (bool): 是否因角色生命归零正常结束战斗。
        model_error (bool): 是否因格式、截断、合法动作或动作上限中止。
        potions_entry (int): 战斗入口非空药水数。
        potions_used (int): 学生轨迹实际使用药水数。
    """

    entry_hp: int
    max_hp: int
    final_hp: int
    turns: int
    cleared: bool
    died: bool
    model_error: bool = False
    potions_entry: int = 0
    potions_used: int = 0


def score_battle_reward(
    inputs: BattleRewardInput,
    *,
    scheme: BattleRewardScheme,
    potion_cost: float = 0.25,
) -> BattleReward:
    """计算一个终局奖励方案的总值与分量。

    Args:
        inputs (BattleRewardInput): 当前战斗的终局事实。
        scheme (BattleRewardScheme): 待评估的固定奖励方案。
        potion_cost (float): ``potion_cost`` 每瓶药水的资源成本。

    Raises:
        ValueError: 生命、结果、回合、药水或方案不满足契约。

    Returns:
        BattleReward: 可用于组内相对 advantage 的标量及审计分量。
    """
    _validate_reward_input(inputs, potion_cost)
    hp_end = 0 if inputs.died or inputs.model_error else inputs.final_hp
    hp_term = max(-1.0, min(0.25, (hp_end - inputs.entry_hp) / inputs.entry_hp))
    clear_value = 3.0 if inputs.cleared else 0.0
    hp_value = 6.0 * hp_term
    death_value = -3.0 if inputs.died or inputs.model_error else 0.0
    turn_value = (
        0.0 if scheme == "core_no_turn" or inputs.model_error else -0.05 * inputs.turns
    )
    model_error_value = (
        _MODEL_ERROR_REWARD_FLOOR - clear_value - hp_value - death_value - turn_value
        if inputs.model_error
        else 0.0
    )
    potion_value = 0.0
    if scheme == "potion_cost":
        potion_value = -potion_cost * inputs.potions_used
    elif scheme == "legacy_remaining_potion":
        lost_fraction = max(0.0, (inputs.entry_hp - hp_end) / inputs.entry_hp)
        potion_gate = min(1.0, lost_fraction / 0.30)
        remaining_fraction = (
            max(0, inputs.potions_entry - inputs.potions_used) / inputs.potions_entry
            if inputs.potions_entry
            else 0.0
        )
        potion_value = -3.0 * remaining_fraction * potion_gate
    components = (
        RewardComponent(name="clear", value=clear_value),
        RewardComponent(name="hp_delta", value=hp_value),
        RewardComponent(name="death", value=death_value),
        RewardComponent(name="turns", value=turn_value),
        RewardComponent(name="model_error", value=model_error_value),
        RewardComponent(name="potions", value=potion_value),
    )
    return BattleReward(
        scheme=scheme,
        total=sum(component.value for component in components),
        components=components,
    )


def _validate_reward_input(inputs: BattleRewardInput, potion_cost: float) -> None:
    """验证奖励事实不会产生含糊或非有限计算。

    Args:
        inputs (BattleRewardInput): 待检查终局事实。
        potion_cost (float): 待检查每瓶药水成本。

    Raises:
        ValueError: 任一数值、结果标记或药水计数无效。

    Returns:
        None: 所有字段满足奖励边界时返回。
    """
    if inputs.entry_hp <= 0 or inputs.max_hp < inputs.entry_hp:
        raise ValueError("战斗奖励入口生命值无效")
    if inputs.final_hp < 0:
        raise ValueError("战斗奖励终局生命值无效")
    if sum((inputs.cleared, inputs.died, inputs.model_error)) != 1:
        raise ValueError("战斗奖励必须且只能声明通关、死亡或模型失败")
    if inputs.turns < 1:
        raise ValueError("战斗奖励回合数必须大于零")
    if inputs.potions_entry < 0 or inputs.potions_used < 0:
        raise ValueError("战斗奖励药水计数无效")
    if potion_cost < 0:
        raise ValueError("药水成本不能为负数")
