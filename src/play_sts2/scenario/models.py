"""定义可重复战斗场景及其不可变状态快照。"""

import re
from dataclasses import dataclass
from typing import Any

_CARD_TOKEN = re.compile(r"^[A-Z0-9_]+(?:\+[1-9][0-9]*)?(?:x[1-9][0-9]*)?$")
_RELIC_TOKEN = re.compile(r"^[A-Z0-9_]+(?::m)?$")
_MODEL_ID = re.compile(r"^[A-Z0-9_]+$")
_GAME_SEED = re.compile(r"^[0-9ABCDEFGHJKLMNPQRSTUVWXYZ]{10}$")


@dataclass(frozen=True, slots=True)
class BattleScenario:
    """描述一场可重复创建的战斗入口。

    ``floor`` 是原战斗的总层数，只参与遭遇 RNG 种子的计算。调试入口不会
    伪造地图历史，因此新局状态中的 ``run.floor`` 仍保持真实值。

    Args:
        character_id (str): 使用的角色稳定 ID。
        seed (str): 新局使用的游戏种子。
        floor (int): 原战斗的总层数，用于确定遭遇 RNG。
        encounter_id (str): 待进入的遭遇稳定 ID。
        deck (tuple[str, ...]): ``loadout`` 语法表示的完整牌组。
        relics (tuple[str, ...]): ``loadout`` 语法表示的完整遗物列表。
        potions (tuple[str, ...]): 按药水栏顺序授予的药水 ID。
        potion_slots (int | None): 可选的药水栏容量。
        current_hp (int | None): 战斗开场效果结算后设置的当前生命值。
        max_hp (int | None): 与当前生命值一起设置的最大生命值。
        ascension (int): 新局使用的进阶等级。
    """

    character_id: str
    seed: str
    floor: int
    encounter_id: str
    deck: tuple[str, ...]
    relics: tuple[str, ...]
    potions: tuple[str, ...] = ()
    potion_slots: int | None = None
    current_hp: int | None = None
    max_hp: int | None = None
    ascension: int = 0

    def __post_init__(self) -> None:
        """拒绝无法安全翻译为 Mod 控制台命令的场景值。

        Raises:
            ValueError: ID、装载 token 或数值边界无效。

        Returns:
            None: 所有场景字段合法时返回。
        """
        if not _MODEL_ID.fullmatch(self.character_id):
            raise ValueError(f"角色 ID 无效: {self.character_id}")
        if not _GAME_SEED.fullmatch(self.seed):
            raise ValueError("游戏种子必须是 10 位 STS2 种子字符")
        if self.floor < 0:
            raise ValueError("总层数不能小于 0")
        if not _MODEL_ID.fullmatch(self.encounter_id):
            raise ValueError(f"遭遇 ID 无效: {self.encounter_id}")
        if not self.deck or any(not _CARD_TOKEN.fullmatch(card) for card in self.deck):
            raise ValueError("牌组不能为空，且每张牌必须使用 loadout token 语法")
        if not self.relics or any(
            not _RELIC_TOKEN.fullmatch(relic) for relic in self.relics
        ):
            raise ValueError("遗物列表不能为空，且必须使用 loadout token 语法")
        if any(not _MODEL_ID.fullmatch(potion) for potion in self.potions):
            raise ValueError("药水必须使用稳定 ID")
        if self.potion_slots is not None and not 1 <= self.potion_slots <= 10:
            raise ValueError("药水栏容量必须在 1 到 10 之间")
        if not 0 <= self.ascension <= 20:
            raise ValueError("进阶等级必须在 0 到 20 之间")
        if self.current_hp is not None and self.current_hp < 1:
            raise ValueError("当前生命值必须大于 0")
        if self.max_hp is not None:
            if self.current_hp is None:
                raise ValueError("设置最大生命值时必须同时设置当前生命值")
            if self.max_hp < self.current_hp:
                raise ValueError("最大生命值不能小于当前生命值")


@dataclass(frozen=True, slots=True)
class CardSnapshot:
    """表示战斗手牌中的一张牌。

    Args:
        index (int): 当前手牌顺序索引。
        card_id (str): 卡牌稳定 ID。
        upgrade_level (int): 卡牌已经升级的具体次数。
    """

    index: int
    card_id: str
    upgrade_level: int


@dataclass(frozen=True, slots=True)
class IntentSnapshot:
    """表示敌人当前展示的一项意图。

    Args:
        intent_type (str): 意图类型。
        label (str | None): 游戏展示的意图标签。
        damage (int | None): 单次伤害。
        hits (int | None): 攻击次数。
        total_damage (int | None): 总伤害。
        status_card_count (int | None): 将加入牌堆的状态牌数量。
    """

    intent_type: str
    label: str | None
    damage: int | None
    hits: int | None
    total_damage: int | None
    status_card_count: int | None


@dataclass(frozen=True, slots=True)
class EnemySnapshot:
    """表示战斗入口中的一个敌人。

    Args:
        index (int): 敌人在当前战斗中的目标索引。
        enemy_id (str): 敌人稳定 ID。
        current_hp (int): 当前生命值。
        max_hp (int): 最大生命值。
        move_id (str | None): 当前招式稳定 ID。
        intents (tuple[IntentSnapshot, ...]): 当前展示的完整意图。
    """

    index: int
    enemy_id: str
    current_hp: int
    max_hp: int
    move_id: str | None
    intents: tuple[IntentSnapshot, ...]


@dataclass(frozen=True, slots=True)
class BattleSnapshot:
    """保存判断战斗入口能否复现所需的 RNG 可见状态。

    Args:
        turn (int): 当前战斗回合。
        enemies (tuple[EnemySnapshot, ...]): 敌人组成、生命与意图。
        hand (tuple[CardSnapshot, ...]): 当前手牌及其顺序。
    """

    turn: int
    enemies: tuple[EnemySnapshot, ...]
    hand: tuple[CardSnapshot, ...]


@dataclass(frozen=True, slots=True)
class ScenarioResetResult:
    """返回已经核对的战斗状态和入口快照。

    Args:
        state (dict[str, Any]): 可直接交给战斗 Runner 的完整 Mod 状态。
        snapshot (BattleSnapshot): 可供后续采样复现核对的入口快照。
    """

    state: dict[str, Any]
    snapshot: BattleSnapshot
