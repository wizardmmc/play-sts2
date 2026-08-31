"""把完整游戏中的真实战斗入口投影为可重放 scenario。"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ....scenario import BattleScenario, BattleSnapshot
from ....scenario.verifier import capture_battle_snapshot


@dataclass(frozen=True, slots=True)
class BackboneBattleCandidate:
    """保存 backbone 暴露的一场可重放战斗入口。

    Args:
        battle_index (int): 当前 episode 内战斗序号。
        floor (int): 战斗入口楼层。
        is_elite (bool): 遭遇 ID 是否明确为精英。
        is_boss (bool): 遭遇 ID 是否与本幕 Boss 一致或明确为 Boss。
        scenario (BattleScenario): 调试重放配置。
        entry_snapshot (BattleSnapshot): 原始完整游戏的玩家可见战斗入口。
        outcome (str): backbone 中该场战斗的离场结果。
        hp_loss_ratio (float): 入口到离场的可见生命损失比例。
    """

    battle_index: int
    floor: int
    is_elite: bool
    is_boss: bool
    scenario: BattleScenario
    entry_snapshot: BattleSnapshot
    outcome: str = "pending"
    hp_loss_ratio: float = 0.0


def extract_battle_candidate(
    state: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    seed: str,
    battle_index: int,
) -> BackboneBattleCandidate:
    """从真实战斗入口构造 scenario 和原始可见快照。

    遭遇 ID 只用于本地调试重放，不进入模型 observation。其余牌组、遗物、药水和
    生命值均来自玩家可见 ``run`` 状态；后续 collector 会把重放入口与这里的
    ``entry_snapshot`` 逐字段比较。

    Args:
        state (Mapping[str, Any]): 当前第一回合可决策战斗状态。
        audit (Mapping[str, Any]): 同 revision 的本地 checkpoint 审计。
        seed (str): 当前完整游戏种子。
        battle_index (int): episode 内战斗序号。

    Raises:
        ValueError: run、审计或 loadout 字段不足以构造确定性场景。
        ScenarioVerificationError: 原始入口无法形成玩家可见快照。

    Returns:
        BackboneBattleCandidate: 不包含牌序或 RNG 审计的重放候选。
    """
    run = state.get("run")
    combat_audit = audit.get("combat")
    if not isinstance(run, Mapping) or not isinstance(combat_audit, Mapping):
        raise TypeError("战斗候选缺少 run 或 combat 审计")
    encounter_id = combat_audit.get("encounter_id")
    if not isinstance(encounter_id, str) or not encounter_id:
        raise ValueError("战斗候选缺少 encounter ID")
    deck = tuple(_card_token(card) for card in _mapping_items(run, "deck"))
    relics = tuple(_relic_token(relic) for relic in _mapping_items(run, "relics"))
    potion_items = _mapping_items(run, "potions")
    potions = tuple(
        str(item["potion_id"])
        if isinstance(item.get("potion_id"), str) and item.get("potion_id")
        else None
        for item in potion_items
    )
    character_id = _required_text(run, "character_id")
    floor = _required_integer(run, "floor")
    current_hp = _required_integer(run, "current_hp")
    max_hp = _required_integer(run, "max_hp")
    ascension = _required_integer(run, "ascension")
    scenario = BattleScenario(
        character_id=character_id,
        seed=seed,
        floor=floor,
        encounter_id=encounter_id,
        deck=deck,
        relics=relics,
        potions=potions,
        potion_slots=len(potions),
        current_hp=current_hp,
        max_hp=max_hp,
        ascension=ascension,
    )
    boss_id = run.get("boss_id")
    return BackboneBattleCandidate(
        battle_index=battle_index,
        floor=floor,
        is_elite=encounter_id.endswith("_ELITE"),
        is_boss=encounter_id.endswith("_BOSS") or encounter_id == boss_id,
        scenario=scenario,
        entry_snapshot=capture_battle_snapshot(state),
    )


def _card_token(card: Mapping[str, Any]) -> str:
    """把一张可见牌组卡牌转成 loadout token。

    Args:
        card (Mapping[str, Any]): ``run.deck`` 中的卡牌。

    Returns:
        str: 保留升级和附魔层数的 loadout token。
    """
    token = _required_text(card, "card_id")
    upgrade = _required_integer(card, "upgrade_level")
    if upgrade:
        token += f"+{upgrade}"
    enchantment = card.get("enchantment_id")
    if enchantment is not None:
        if not isinstance(enchantment, str) or not enchantment:
            raise ValueError("战斗候选卡牌附魔 ID 无效")
        token += f"@{enchantment}"
        amount = card.get("enchantment_amount")
        if amount is not None:
            if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
                raise ValueError("战斗候选卡牌附魔层数无效")
            token += f":{amount}"
    return token


def _relic_token(relic: Mapping[str, Any]) -> str:
    """把一件可见遗物转成 loadout token。

    Args:
        relic (Mapping[str, Any]): ``run.relics`` 中的遗物。

    Returns:
        str: 保留熔毁状态的 loadout token。
    """
    token = _required_text(relic, "relic_id")
    return f"{token}:m" if relic.get("is_melted") is True else token


def _mapping_items(
    value: Mapping[str, Any], field: str
) -> tuple[Mapping[str, Any], ...]:
    """读取一个只含对象的数组字段。

    Args:
        value (Mapping[str, Any]): 当前映射。
        field (str): 数组字段名。

    Raises:
        ValueError: 字段不是对象数组或为空。

    Returns:
        tuple[Mapping[str, Any], ...]: 校验后的数组。
    """
    raw = value.get(field)
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(item, Mapping) for item in raw)
    ):
        raise ValueError(f"战斗候选字段必须是非空对象数组: {field}")
    return tuple(raw)


def _required_text(value: Mapping[str, Any], field: str) -> str:
    """读取非空文本字段。

    Args:
        value (Mapping[str, Any]): 当前映射。
        field (str): 字段名。

    Raises:
        ValueError: 字段不是非空字符串。

    Returns:
        str: 校验后的文本。
    """
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise ValueError(f"战斗候选字段必须是非空字符串: {field}")
    return item


def _required_integer(value: Mapping[str, Any], field: str) -> int:
    """读取非负整数字段。

    Args:
        value (Mapping[str, Any]): 当前映射。
        field (str): 字段名。

    Raises:
        ValueError: 字段不是非负整数。

    Returns:
        int: 校验后的整数。
    """
    item = value.get(field)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise ValueError(f"战斗候选字段必须是非负整数: {field}")
    return item
