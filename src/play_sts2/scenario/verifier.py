"""核对场景装载结果并捕获战斗入口快照。"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..harness import ObservationError, build_observation, system_prompt
from .models import (
    BattleScenario,
    BattleSnapshot,
    CardSnapshot,
    EnemySnapshot,
    IntentSnapshot,
    ModelInputSnapshot,
)

_CARD_TOKEN = re.compile(
    r"^(?P<id>[A-Z0-9_]+)(?:\+(?P<upgrade>[1-9][0-9]*))?"
    r"(?:@(?P<enchantment>[A-Z0-9_]+)"
    r"(?::(?P<enchantment_amount>[1-9][0-9]*))?)?"
    r"(?:x(?P<count>[1-9][0-9]*))?$"
)


class ScenarioVerificationError(RuntimeError):
    """表示游戏状态与请求的战斗场景不一致。"""


def verify_battle_scenario(
    scenario: BattleScenario,
    state: Mapping[str, Any],
    *,
    expected_snapshot: BattleSnapshot | None = None,
) -> BattleSnapshot:
    """核对场景配置并返回确定性相关的战斗入口快照。

    Args:
        scenario (BattleScenario): 请求创建的完整战斗场景。
        state (Mapping[str, Any]): Mod 返回的战斗入口状态。
        expected_snapshot (BattleSnapshot | None): 可选的首次复现基准快照。

    Raises:
        ScenarioVerificationError: 状态不在战斗中、装载不符或快照不同。

    Returns:
        BattleSnapshot: 当前敌人、生命、意图和手牌顺序的不可变快照。
    """
    if state.get("screen") != "COMBAT" or state.get("in_combat") is not True:
        raise ScenarioVerificationError("场景没有进入战斗")
    run = _mapping(state.get("run"), "run")
    _verify_run(scenario, run)
    snapshot = capture_battle_snapshot(state)
    if expected_snapshot is not None and snapshot != expected_snapshot:
        raise ScenarioVerificationError("初始战斗快照不一致")
    return snapshot


def capture_battle_snapshot(state: Mapping[str, Any]) -> BattleSnapshot:
    """提取任一战斗决策点的敌人、意图和手牌快照。

    该函数不核对装载配置，因此可用于比较固定动作脚本下的第二回合等后续
    决策点。

    Args:
        state (Mapping[str, Any]): Mod 返回的完整战斗状态。

    Raises:
        ScenarioVerificationError: 状态不在战斗中或战斗字段形态无效。

    Returns:
        BattleSnapshot: 当前战斗决策点的不可变快照。
    """
    if state.get("screen") != "COMBAT" or state.get("in_combat") is not True:
        raise ScenarioVerificationError("状态不在战斗中")
    return _capture_snapshot(state, _mapping(state.get("combat"), "combat"))


def _verify_run(scenario: BattleScenario, run: Mapping[str, Any]) -> None:
    """核对角色、进阶、牌组、遗物、药水和生命值。

    Args:
        scenario (BattleScenario): 请求创建的完整战斗场景。
        run (Mapping[str, Any]): Mod 状态中的新局载荷。

    Raises:
        ScenarioVerificationError: 任一装载字段与场景不一致。

    Returns:
        None: 全部字段一致时返回。
    """
    _expect(run.get("character_id"), scenario.character_id, "角色")
    _expect(run.get("ascension"), scenario.ascension, "进阶")

    actual_deck = tuple(
        (
            card.get("card_id"),
            _integer(card.get("upgrade_level"), "牌组升级等级"),
            _optional_text(card.get("enchantment_id"), "牌组附魔 ID"),
            _optional_integer(card.get("enchantment_amount"), "牌组附魔层数"),
        )
        for card in _mapping_items(run.get("deck"), "牌组")
    )
    _expect(actual_deck, _expand_deck(scenario.deck), "牌组")

    actual_relics = tuple(
        (relic.get("relic_id"), relic.get("is_melted") is True)
        for relic in _mapping_items(run.get("relics"), "遗物")
    )
    expected_relics = tuple(
        (token.removesuffix(":m"), token.endswith(":m")) for token in scenario.relics
    )
    _expect(actual_relics, expected_relics, "遗物")

    potion_slots = _mapping_items(run.get("potions"), "药水栏")
    if scenario.potion_slots is not None:
        _expect(len(potion_slots), scenario.potion_slots, "药水栏容量")
        actual_potions = tuple(
            (
                _integer(potion.get("index"), "药水栏索引"),
                _optional_text(potion.get("potion_id"), "药水 ID"),
            )
            for potion in potion_slots
        )
        expected_slots = scenario.potions + (None,) * (
            scenario.potion_slots - len(scenario.potions)
        )
        _expect(
            actual_potions,
            tuple(enumerate(expected_slots)),
            "药水栏",
        )
    else:
        actual_potions = tuple(
            potion.get("potion_id")
            for potion in potion_slots
            if potion.get("occupied") is True or potion.get("potion_id") is not None
        )
        _expect(actual_potions, scenario.potions, "药水")

    if scenario.current_hp is not None:
        _expect(run.get("current_hp"), scenario.current_hp, "当前生命值")
    if scenario.max_hp is not None:
        _expect(run.get("max_hp"), scenario.max_hp, "最大生命值")


def _capture_snapshot(
    state: Mapping[str, Any],
    combat: Mapping[str, Any],
) -> BattleSnapshot:
    """从战斗载荷提取稳定且与 RNG 相关的入口字段。

    Args:
        state (Mapping[str, Any]): 完整游戏状态。
        combat (Mapping[str, Any]): 状态中的战斗载荷。

    Raises:
        ScenarioVerificationError: 回合、手牌、敌人或意图字段形态无效。

    Returns:
        BattleSnapshot: 当前战斗入口的不可变快照。
    """
    turn = state.get("turn")
    if isinstance(turn, bool) or not isinstance(turn, int):
        raise ScenarioVerificationError("战斗回合不可用")
    hand = tuple(
        CardSnapshot(
            index=_integer(card.get("index"), "手牌索引"),
            card_id=_text(card.get("card_id"), "手牌 ID"),
            upgrade_level=_integer(card.get("upgrade_level"), "手牌升级等级"),
            enchantment_id=_optional_text(
                card.get("enchantment_id"), "手牌附魔 ID"
            ),
            enchantment_amount=_optional_integer(
                card.get("enchantment_amount"), "手牌附魔层数"
            ),
        )
        for card in _mapping_items(combat.get("hand"), "手牌")
    )
    enemies = tuple(
        EnemySnapshot(
            index=_integer(enemy.get("index"), "敌人索引"),
            enemy_id=_text(enemy.get("enemy_id"), "敌人 ID"),
            current_hp=_integer(enemy.get("current_hp"), "敌人当前生命值"),
            max_hp=_integer(enemy.get("max_hp"), "敌人最大生命值"),
            move_id=_optional_text(enemy.get("move_id"), "敌人招式 ID"),
            intents=tuple(
                _intent_snapshot(intent)
                for intent in _mapping_items(enemy.get("intents"), "敌人意图")
            ),
        )
        for enemy in _mapping_items(combat.get("enemies"), "敌人")
    )
    if not enemies:
        raise ScenarioVerificationError("战斗中没有敌人")
    try:
        observation = build_observation(state)
    except ObservationError as exc:
        raise ScenarioVerificationError("战斗入口无法生成模型观测") from exc
    model_input = ModelInputSnapshot(
        system=system_prompt(observation.layer, state),
        user=observation.text,
        available_actions=observation.available_actions,
    )
    return BattleSnapshot(
        turn=turn,
        enemies=enemies,
        hand=hand,
        model_input=model_input,
    )


def _intent_snapshot(intent: Mapping[str, Any]) -> IntentSnapshot:
    """把一个 Mod 意图载荷转换为不可变快照。

    Args:
        intent (Mapping[str, Any]): Mod 返回的一项敌人意图。

    Raises:
        ScenarioVerificationError: 意图字段类型无效。

    Returns:
        IntentSnapshot: 保留类型、标签和伤害信息的意图快照。
    """
    return IntentSnapshot(
        intent_type=_text(intent.get("intent_type"), "意图类型"),
        label=_optional_text(intent.get("label"), "意图标签"),
        damage=_optional_integer(intent.get("damage"), "意图伤害"),
        hits=_optional_integer(intent.get("hits"), "意图次数"),
        total_damage=_optional_integer(intent.get("total_damage"), "意图总伤害"),
        status_card_count=_optional_integer(
            intent.get("status_card_count"),
            "意图状态牌数量",
        ),
    )


def _expand_deck(
    tokens: Sequence[str],
) -> tuple[tuple[str, int, str | None, int | None], ...]:
    """把紧凑的 ``loadout`` 牌组 token 展开为有序牌列表。

    Args:
        tokens (Sequence[str]): 已由 ``BattleScenario`` 校验的牌组 token。

    Raises:
        ScenarioVerificationError: token 未满足场景模型的语法约束。

    Returns:
        tuple[tuple[str, int, str | None, int | None], ...]: 有序的卡牌 ID、
            升级等级、附魔 ID 与附魔层数。
    """
    cards: list[tuple[str, int, str | None, int | None]] = []
    for token in tokens:
        match = _CARD_TOKEN.fullmatch(token)
        if match is None:
            raise ScenarioVerificationError(f"牌组 token 无效: {token}")
        enchantment = match.group("enchantment")
        card = (
            match.group("id"),
            int(match.group("upgrade") or 0),
            enchantment,
            int(match.group("enchantment_amount") or 1)
            if enchantment is not None
            else None,
        )
        cards.extend((card,) * int(match.group("count") or 1))
    return tuple(cards)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    """读取一个必需的对象字段。

    Args:
        value (object): 待校验的原始值。
        field (str): 用于错误信息的字段名称。

    Raises:
        ScenarioVerificationError: 原始值不是对象。

    Returns:
        Mapping[str, Any]: 校验后的对象。
    """
    if not isinstance(value, Mapping):
        raise ScenarioVerificationError(f"{field} 状态不可用")
    return value


def _mapping_items(value: object, field: str) -> tuple[Mapping[str, Any], ...]:
    """读取一个由对象组成的必需列表字段。

    Args:
        value (object): 待校验的原始值。
        field (str): 用于错误信息的字段名称。

    Raises:
        ScenarioVerificationError: 原始值不是列表或包含非对象元素。

    Returns:
        tuple[Mapping[str, Any], ...]: 校验并冻结后的对象序列。
    """
    if not isinstance(value, list) or any(
        not isinstance(item, Mapping) for item in value
    ):
        raise ScenarioVerificationError(f"{field} 状态不可用")
    return tuple(value)


def _text(value: object, field: str) -> str:
    """读取一个必需的非空文本字段。

    Args:
        value (object): 待校验的原始值。
        field (str): 用于错误信息的字段名称。

    Raises:
        ScenarioVerificationError: 原始值不是非空字符串。

    Returns:
        str: 校验后的文本。
    """
    if not isinstance(value, str) or not value:
        raise ScenarioVerificationError(f"{field} 不可用")
    return value


def _optional_text(value: object, field: str) -> str | None:
    """读取一个允许为空的文本字段。

    Args:
        value (object): 待校验的原始值。
        field (str): 用于错误信息的字段名称。

    Raises:
        ScenarioVerificationError: 非空值不是字符串。

    Returns:
        str | None: 校验后的文本或 ``None``。
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ScenarioVerificationError(f"{field} 不可用")
    return value


def _integer(value: object, field: str) -> int:
    """读取一个必需的整数字段。

    Args:
        value (object): 待校验的原始值。
        field (str): 用于错误信息的字段名称。

    Raises:
        ScenarioVerificationError: 原始值不是整数。

    Returns:
        int: 校验后的整数。
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScenarioVerificationError(f"{field} 不可用")
    return value


def _optional_integer(value: object, field: str) -> int | None:
    """读取一个允许为空的整数字段。

    Args:
        value (object): 待校验的原始值。
        field (str): 用于错误信息的字段名称。

    Raises:
        ScenarioVerificationError: 非空值不是整数。

    Returns:
        int | None: 校验后的整数或 ``None``。
    """
    if value is None:
        return None
    return _integer(value, field)


def _expect(actual: object, expected: object, field: str) -> None:
    """比较一个场景字段的实际值和期望值。

    Args:
        actual (object): Mod 状态中的实际值。
        expected (object): 场景声明的期望值。
        field (str): 用于错误信息的字段名称。

    Raises:
        ScenarioVerificationError: 实际值和期望值不同。

    Returns:
        None: 两个值相等时返回。
    """
    if actual != expected:
        raise ScenarioVerificationError(
            f"{field}不一致: 期望 {expected!r}，实际 {actual!r}"
        )
