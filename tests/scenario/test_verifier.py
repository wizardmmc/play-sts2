"""验证战斗场景状态核对与确定性快照。"""

import pytest

from play_sts2.scenario import (
    BattleScenario,
    ScenarioVerificationError,
    verify_battle_scenario,
)


def test_battle_scenario_rejects_empty_relics() -> None:
    """拒绝当前 ``loadout`` 无法表达的空遗物列表。

    Raises:
        AssertionError: 场景模型接受了无法精确装载的遗物配置。

    Returns:
        None: 此测试仅验证场景模型与 Mod 命令能力一致。
    """
    with pytest.raises(ValueError, match="遗物列表不能为空"):
        BattleScenario(
            character_id="DEFECT",
            seed="ABCDEF1234",
            floor=7,
            encounter_id="CULTISTS_NORMAL",
            deck=("ZAP",),
            relics=(),
        )


def test_verify_battle_scenario_returns_rng_snapshot() -> None:
    """完整核对装载结果，并提取敌人、手牌和意图快照。

    Raises:
        AssertionError: 场景字段未被核对，或快照遗漏 RNG 相关状态。

    Returns:
        None: 此测试仅验证状态核对的成功路径。
    """
    scenario = BattleScenario(
        character_id="DEFECT",
        seed="ABCDEF1234",
        floor=7,
        encounter_id="CULTISTS_NORMAL",
        deck=("ZAP+1", "STRIKE_DEFECTx2"),
        relics=("CRACKED_CORE", "ORICHALCUM:m"),
        potions=("FIRE_POTION",),
        potion_slots=3,
        current_hp=41,
        max_hp=70,
        ascension=2,
    )
    state = _combat_state()

    snapshot = verify_battle_scenario(scenario, state)

    assert snapshot.turn == 1
    assert [(card.card_id, card.upgrade_level) for card in snapshot.hand] == [
        ("STRIKE_DEFECT", 0),
        ("ZAP", 1),
    ]
    assert [enemy.enemy_id for enemy in snapshot.enemies] == ["DAMP_CULTIST"]
    assert snapshot.enemies[0].current_hp == 48
    assert snapshot.enemies[0].move_id == "INCANTATION"
    assert snapshot.enemies[0].intents[0].intent_type == "Buff"
    assert snapshot.enemies[0].intents[0].status_card_count == 2


def test_verify_battle_scenario_distinguishes_upgrade_levels() -> None:
    """拒绝仅有升级布尔值相同、具体升级等级不同的牌组。

    Raises:
        AssertionError: 核对器把 ``+1`` 与 ``+2`` 误判为相同牌组。

    Returns:
        None: 此测试仅验证多级升级的精确核对。
    """
    scenario = BattleScenario(
        character_id="DEFECT",
        seed="ABCDEF1234",
        floor=7,
        encounter_id="CULTISTS_NORMAL",
        deck=("ZAP+2",),
        relics=("CRACKED_CORE", "ORICHALCUM:m"),
        potions=("FIRE_POTION",),
        potion_slots=3,
        current_hp=41,
        max_hp=70,
        ascension=2,
    )
    state = _combat_state()
    state["run"]["deck"] = [
        {"index": 0, "card_id": "ZAP", "upgraded": True, "upgrade_level": 1}
    ]
    state["combat"]["hand"] = [
        {"index": 0, "card_id": "ZAP", "upgraded": True, "upgrade_level": 1}
    ]

    with pytest.raises(ScenarioVerificationError, match="牌组不一致"):
        verify_battle_scenario(scenario, state)


def test_verify_battle_scenario_rejects_different_initial_hand() -> None:
    """对照基准快照时拒绝不同的初始手牌。

    Raises:
        AssertionError: 核对器没有报告初始手牌不一致。

    Returns:
        None: 此测试仅验证复现实验的失败关闭边界。
    """
    scenario = BattleScenario(
        character_id="DEFECT",
        seed="ABCDEF1234",
        floor=7,
        encounter_id="CULTISTS_NORMAL",
        deck=("ZAP+1", "STRIKE_DEFECTx2"),
        relics=("CRACKED_CORE", "ORICHALCUM:m"),
        potions=("FIRE_POTION",),
        potion_slots=3,
        current_hp=41,
        max_hp=70,
        ascension=2,
    )
    baseline = verify_battle_scenario(scenario, _combat_state())
    changed = _combat_state()
    changed["combat"]["hand"].reverse()

    with pytest.raises(ScenarioVerificationError, match="初始战斗快照不一致"):
        verify_battle_scenario(scenario, changed, expected_snapshot=baseline)


def _combat_state() -> dict[str, object]:
    """创建一份同时覆盖装载和战斗快照字段的真实协议形态。

    Returns:
        dict[str, object]: 可供场景核对器解析的战斗状态。
    """
    return {
        "screen": "COMBAT",
        "in_combat": True,
        "turn": 1,
        "available_actions": ["play_card", "end_turn", "save_and_quit"],
        "run": {
            "character_id": "DEFECT",
            "ascension": 2,
            "floor": 0,
            "current_hp": 41,
            "max_hp": 70,
            "deck": [
                {
                    "index": 0,
                    "card_id": "ZAP",
                    "upgraded": True,
                    "upgrade_level": 1,
                },
                {
                    "index": 1,
                    "card_id": "STRIKE_DEFECT",
                    "upgraded": False,
                    "upgrade_level": 0,
                },
                {
                    "index": 2,
                    "card_id": "STRIKE_DEFECT",
                    "upgraded": False,
                    "upgrade_level": 0,
                },
            ],
            "relics": [
                {"index": 0, "relic_id": "CRACKED_CORE", "is_melted": False},
                {"index": 1, "relic_id": "ORICHALCUM", "is_melted": True},
            ],
            "potions": [
                {"index": 0, "potion_id": "FIRE_POTION", "occupied": True},
                {"index": 1, "potion_id": None, "occupied": False},
                {"index": 2, "potion_id": None, "occupied": False},
            ],
        },
        "combat": {
            "player": {
                "current_hp": 41,
                "max_hp": 70,
                "energy": 3,
                "block": 0,
            },
            "hand": [
                {
                    "index": 0,
                    "card_id": "STRIKE_DEFECT",
                    "upgraded": False,
                    "upgrade_level": 0,
                },
                {
                    "index": 1,
                    "card_id": "ZAP",
                    "upgraded": True,
                    "upgrade_level": 1,
                },
            ],
            "enemies": [
                {
                    "index": 0,
                    "enemy_id": "DAMP_CULTIST",
                    "current_hp": 48,
                    "max_hp": 48,
                    "move_id": "INCANTATION",
                    "intents": [
                        {
                            "index": 0,
                            "intent_type": "Buff",
                            "label": None,
                            "damage": None,
                            "hits": None,
                            "total_damage": None,
                            "status_card_count": 2,
                        }
                    ],
                }
            ],
        },
    }
