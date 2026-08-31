"""验证完整游戏暴露的战斗入口可投影为确定性 scenario。"""


def test_extract_battle_candidate_preserves_visible_loadout_and_encounter() -> None:
    """scenario 应保留升级、附魔、熔毁遗物、空药水槽和教师遭遇 ID。

    Returns:
        None: 调试重放所需字段从真实入口逐项投影。
    """
    from play_sts2.training.rl.gigpo.scenario import extract_battle_candidate

    state = _battle_state()
    candidate = extract_battle_candidate(
        state,
        {"combat": {"encounter_id": "TEST_ELITE"}},
        seed="ABCDEF1234",
        battle_index=2,
    )

    assert candidate.scenario.encounter_id == "TEST_ELITE"
    assert candidate.scenario.deck == (
        "STRIKE_DEFECT+1",
        "ZAP@NIMBLE:2",
    )
    assert candidate.scenario.relics == ("CRACKED_CORE", "CANDLE:m")
    assert candidate.scenario.potions == ("FIRE_POTION", None)
    assert candidate.scenario.potion_slots == 2
    assert candidate.is_elite is True
    assert candidate.entry_snapshot.turn == 1


def _battle_state() -> dict[str, object]:
    """构造 scenario 投影需要的最小真实战斗入口。

    Returns:
        dict[str, object]: 含 Harness 可渲染字段的战斗状态。
    """
    return {
        "screen": "COMBAT",
        "in_combat": True,
        "turn": 1,
        "available_actions": ["end_turn", "play_card"],
        "run": {
            "character_id": "DEFECT",
            "ascension": 0,
            "floor": 8,
            "current_hp": 50,
            "max_hp": 75,
            "deck": [
                {
                    "card_id": "STRIKE_DEFECT",
                    "upgrade_level": 1,
                    "enchantment_id": None,
                    "enchantment_amount": None,
                },
                {
                    "card_id": "ZAP",
                    "upgrade_level": 0,
                    "enchantment_id": "NIMBLE",
                    "enchantment_amount": 2,
                },
            ],
            "relics": [
                {"relic_id": "CRACKED_CORE", "is_melted": False},
                {"relic_id": "CANDLE", "is_melted": True},
            ],
            "potions": [
                {"index": 0, "potion_id": "FIRE_POTION", "occupied": True},
                {"index": 1, "potion_id": None, "occupied": False},
            ],
        },
        "combat": {
            "player": {
                "current_hp": 50,
                "max_hp": 75,
                "block": 0,
                "energy": 3,
                "stars": 0,
                "focus": 0,
                "powers": [],
                "orbs": [],
            },
            "hand": [
                {
                    "index": 0,
                    "card_id": "STRIKE_DEFECT",
                    "name": "打击+",
                    "upgrade_level": 1,
                    "enchantment_id": None,
                    "enchantment_amount": None,
                    "energy_cost": 1,
                    "star_cost": 0,
                    "resolved_rules_text": "造成9点伤害。",
                    "playable": True,
                    "requires_target": True,
                    "valid_target_indices": [0],
                }
            ],
            "draw_count": 1,
            "discard_count": 0,
            "enemies": [
                {
                    "index": 0,
                    "enemy_id": "TEST_ENEMY",
                    "name": "测试敌人",
                    "current_hp": 20,
                    "max_hp": 20,
                    "block": 0,
                    "is_alive": True,
                    "is_hittable": True,
                    "move_id": "ATTACK",
                    "powers": [],
                    "intents": [
                        {
                            "intent_type": "Attack",
                            "label": "5",
                            "damage": 5,
                            "hits": 1,
                            "total_damage": 5,
                            "status_card_count": None,
                        }
                    ],
                }
            ],
        },
    }
