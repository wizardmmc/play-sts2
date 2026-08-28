"""验证完整游戏状态会转换成稳定、可读的模型观测。"""

import importlib
from typing import Any

import pytest


@pytest.mark.parametrize(
    ("state", "expected_fragments"),
    [
        (
            {
                "screen": "REST",
                "available_actions": ["choose_rest_option"],
                "rest": {
                    "options": [
                        {
                            "index": 0,
                            "title": "休息",
                            "description": "回复生命。",
                            "is_enabled": True,
                        }
                    ]
                },
            },
            ("=== 休息处 ===", "[0] 休息", "回复生命。"),
        ),
        (
            {
                "screen": "SHOP",
                "available_actions": ["buy_card", "remove_card_at_shop", "proceed"],
                "shop": {
                    "is_open": True,
                    "cards": [
                        {
                            "index": 0,
                            "name": "眼部攻击",
                            "energy_cost": 0,
                            "price": 45,
                            "is_stocked": True,
                            "enough_gold": True,
                        }
                    ],
                    "relics": [],
                    "potions": [],
                    "card_removal": {
                        "price": 75,
                        "available": True,
                        "used": False,
                        "enough_gold": True,
                    },
                },
            },
            (
                "=== 商店（库存已打开）===",
                "[0]眼部攻击(0费)",
                "45 金币",
                "删牌: 75 金币",
            ),
        ),
        (
            {
                "screen": "CHEST",
                "available_actions": ["choose_treasure_relic"],
                "chest": {
                    "is_opened": True,
                    "relic_options": [
                        {
                            "index": 0,
                            "name": "锚",
                            "rarity": "Common",
                            "description": "战斗开始时获得格挡。",
                        }
                    ],
                },
            },
            ("=== 宝箱（已打开）===", "[0] 锚", "战斗开始时获得格挡。"),
        ),
        (
            {
                "screen": "BUNDLE_SELECTION",
                "available_actions": ["choose_bundle"],
                "bundles": [
                    {
                        "index": 0,
                        "cards": [{"index": 0, "name": "打击", "energy_cost": 1}],
                    }
                ],
            },
            ("=== 选择卡牌包 ===", "卡牌包 [0]", "打击"),
        ),
        (
            {
                "screen": "CRYSTAL_SPHERE",
                "available_actions": ["choose_crystal_sphere_cell"],
                "crystal_sphere": {
                    "divinations_remaining": 2,
                    "tool": "big",
                    "clickable_cells": [{"index": 3, "x": 2, "y": 1}],
                    "revealed_items": [{"kind": "gold", "x": 0, "y": 0}],
                },
            },
            ("=== 水晶球 ===", "剩余占卜: 2", "[3] 坐标 (2, 1)", "gold @ (0, 0)"),
        ),
        (
            {
                "screen": "MODAL",
                "available_actions": ["confirm_modal", "dismiss_modal"],
                "modal": {
                    "type_name": "NConfirmationModal",
                    "underlying_screen": "SHOP",
                    "confirm_label": "确认",
                    "dismiss_label": "取消",
                },
            },
            ("=== 确认弹窗 ===", "来源页面: SHOP", "确认: 确认", "取消: 取消"),
        ),
        (
            {
                "screen": "CARDS_VIEW",
                "available_actions": ["close_cards_view"],
                "cards_view": {
                    "prompt": "查看牌组",
                    "cards": [{"index": 0, "name": "防御", "energy_cost": 1}],
                },
            },
            ("=== 查看牌组 ===", "查看牌组", "[0]防御(1费)"),
        ),
        (
            {
                "screen": "TIMELINE",
                "available_actions": ["choose_timeline_epoch"],
                "timeline": {
                    "slots": [
                        {
                            "index": 1,
                            "title": "第一纪元",
                            "state": "obtained",
                            "is_actionable": True,
                        }
                    ]
                },
            },
            ("=== 时间线 ===", "[1] 第一纪元", "obtained", "可选择"),
        ),
        (
            {
                "screen": "UNKNOWN",
                "available_actions": ["proceed"],
            },
            ("=== 房间结算 ===", "继续进入下一状态"),
        ),
    ],
)
def test_build_observation_renders_supported_strategic_screen(
    state: dict[str, Any],
    expected_fragments: tuple[str, ...],
) -> None:
    """Mod 已支持的战略页面都展示作出选择所需的信息。

    Args:
        state (dict[str, Any]): 当前战略页面的特有状态。
        expected_fragments (tuple[str, ...]): 手工确定的必要观测片段。

    Raises:
        AssertionError: 页面无法渲染、归属错误或缺少决策信息。

    Returns:
        None: 此测试只验证战略页面观测。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "in_combat": False,
        "run": {
            "character_name": "故障机器人",
            "current_hp": 70,
            "max_hp": 75,
            "gold": 99,
            "deck": [],
            "relics": [],
            "potions": [],
        },
        **state,
    }

    observation = harness.build_observation(state)

    assert observation.layer is harness.HarnessLayer.STRATEGIC
    assert all(fragment in observation.text for fragment in expected_fragments)


def test_shop_observation_explains_that_reopening_does_not_refresh_stock() -> None:
    """商店没有任何可购买项目时明确提示重开库存不会刷新。

    Raises:
        AssertionError: 观测没有向模型说明零购买力和真实离开语义。

    Returns:
        None: 此测试只验证商店观测，不替模型选择离开动作。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "SHOP",
        "available_actions": ["open_shop_inventory", "proceed"],
        "run": {"gold": 0},
        "shop": {
            "is_open": False,
            "cards": [
                {
                    "index": 0,
                    "name": "眼部攻击",
                    "price": 45,
                    "is_stocked": True,
                    "enough_gold": False,
                }
            ],
            "relics": [],
            "potions": [],
            "card_removal": {
                "price": 75,
                "available": True,
                "used": False,
                "enough_gold": False,
            },
        },
    }

    observation = harness.build_observation(state)

    assert "当前没有任何可购买项目" in observation.text
    assert "重新打开库存不会刷新商品" in observation.text
    assert "ACTION: proceed" in observation.text
    assert observation.available_actions == ("open_shop_inventory", "proceed")


def test_reward_observation_does_not_repeat_identical_description() -> None:
    """奖励名称与描述同值时只展示一次业务信息。

    Raises:
        AssertionError: 金币或卡牌奖励仍被重复拼接。

    Returns:
        None: 此测试只验证奖励行去重。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "REWARD",
        "in_combat": False,
        "available_actions": ["claim_reward"],
        "run": {"relics": [], "potions": [], "deck": []},
        "reward": {
            "rewards": [
                {
                    "index": 0,
                    "name": "11金币",
                    "description": "11金币",
                    "effect_description": "",
                },
                {
                    "index": 1,
                    "name": "将一张牌添加到你的牌组。",
                    "description": "将一张牌添加到你的牌组。",
                    "effect_description": "",
                },
            ]
        },
    }

    observation = harness.build_observation(state)

    assert "[0] 11金币\n" in observation.text
    assert "[0] 11金币 | 11金币" not in observation.text
    assert observation.text.count("将一张牌添加到你的牌组。") == 1


def test_card_selection_hides_prompt_copied_from_first_card_rules() -> None:
    """历史 raw 的首张卡说明不能伪装成选牌页面提示。

    Raises:
        AssertionError: 首张卡牌规则在候选列表之前重复出现。

    Returns:
        None: 此测试只验证有证据的历史数据降级。
    """
    harness = importlib.import_module("play_sts2.harness")
    duplicated_rules = "将一张随机能力牌加入你的手牌。这张牌在本回合内免费打出。 消耗。"
    state = {
        "screen": "CARD_SELECTION",
        "in_combat": False,
        "available_actions": ["choose_reward_card"],
        "run": {"relics": [], "potions": [], "deck": []},
        "selection": {
            "prompt": duplicated_rules,
            "cards": [
                {
                    "index": 0,
                    "name": "白噪声",
                    "energy_cost": 1,
                    "resolved_rules_text": duplicated_rules,
                },
                {
                    "index": 1,
                    "name": "引雷针",
                    "energy_cost": 1,
                    "resolved_rules_text": (
                        "获得4点格挡。在下2个回合开始时，生成1个闪电充能球。"
                    ),
                },
            ],
        },
    }

    observation = harness.build_observation(state)

    assert observation.text.count(duplicated_rules) == 1
    assert "[0]白噪声(1费)" in observation.text


def test_combat_card_selection_keeps_current_combat_context() -> None:
    """全息影像选弃牌时保留决定候选牌价值所需的当前战斗状态。"""
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "CARD_SELECTION",
        "in_combat": True,
        "turn": 3,
        "available_actions": ["select_deck_card", "skip_card_selection"],
        "run": {"relics": [], "potions": []},
        "combat": {
            "player": {
                "current_hp": 41,
                "max_hp": 75,
                "block": 4,
                "energy": 1,
                "stars": 0,
            },
            "enemies": [
                {
                    "index": 0,
                    "name": "邪教徒",
                    "current_hp": 19,
                    "max_hp": 48,
                    "block": 0,
                    "intents": [{"intent_type": "Attack", "total_damage": 6}],
                }
            ],
            "hand": [
                {
                    "index": 0,
                    "name": "防御",
                    "energy_cost": 1,
                    "resolved_rules_text": "获得5点格挡。",
                }
            ],
            "draw_count": 3,
            "discard_count": 2,
        },
        "agent_view": {
            "combat": {
                "draw": [{"line": "打击 [1费]：造成6点伤害。"}],
                "discard": [
                    {"line": "冷静头脑 [1费]：生成1个冰霜充能球。"},
                    {"line": "电击+ [0费]：生成1个闪电充能球。"},
                ],
                "exhaust": [],
            }
        },
        "selection": {
            "kind": "deck_card_select",
            "prompt": "选择一张牌加入你的手牌",
            "cards": [
                {
                    "index": 0,
                    "name": "冷静头脑",
                    "energy_cost": 1,
                    "resolved_rules_text": "生成1个冰霜充能球。抽1张牌。",
                },
                {
                    "index": 1,
                    "name": "电击",
                    "upgraded": True,
                    "energy_cost": 0,
                    "resolved_rules_text": "生成1个闪电充能球。",
                },
            ],
        },
    }

    observation = harness.build_observation(state)

    assert observation.layer is harness.HarnessLayer.BATTLE
    assert "玩家: HP 41/75 | 格挡4 | 能量1 | 星能0" in observation.text
    assert "敌[0] 邪教徒: HP 19/48 | 格挡0" in observation.text
    assert "手牌:\n  [0]防御(1费) 获得5点格挡。" in observation.text
    assert "弃牌堆（2张）" in observation.text
    assert "=== 选择卡牌 ===\n选择一张牌加入你的手牌" in observation.text
    assert "[0]冷静头脑(1费)" in observation.text
    assert "[1]电击+(0费)" in observation.text
    assert observation.text.endswith(
        "可执行动作:\n- select_deck_card(option_index)\n- skip_card_selection"
    )


def test_rest_observation_renders_all_relic_added_options() -> None:
    """帐篷与铲子增加的休息处选项由通用列表完整展示。

    Raises:
        AssertionError: Harness 把休息处硬编码成休息和锻造两项。

    Returns:
        None: 此测试刻画已有的通用休息处契约。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "REST",
        "in_combat": False,
        "available_actions": ["choose_rest_option"],
        "run": {
            "relics": [{"name": "帐篷"}, {"name": "铲子"}],
            "potions": [],
            "deck": [],
        },
        "rest": {
            "options": [
                {"index": 0, "option_id": "HEAL", "title": "休息"},
                {"index": 1, "option_id": "SMITH", "title": "锻造"},
                {"index": 2, "option_id": "DIG", "title": "挖掘"},
                {"index": 3, "option_id": "LEAVE", "title": "离开"},
            ]
        },
    }

    observation = harness.build_observation(state)

    assert "[0] 休息" in observation.text
    assert "[1] 锻造" in observation.text
    assert "[2] 挖掘" in observation.text
    assert "[3] 离开" in observation.text
    assert "- choose_rest_option(option_index)" in observation.text


def test_build_observation_renders_combat_decision() -> None:
    """战斗观测展示资源、敌人意图、手牌索引和合法动作。

    Raises:
        AssertionError: 模型缺少战斗决策所需信息或看见越权动作。

    Returns:
        None: 此测试只验证生成的战斗观测。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "COMBAT",
        "in_combat": True,
        "turn": 2,
        "available_actions": [
            "save_and_quit",
            "play_card",
            "use_potion",
            "discard_potion",
            "end_turn",
        ],
        "run": {
            "character_name": "故障机器人",
            "ascension": 2,
            "act_id": "0",
            "floor": 3,
            "current_hp": 44,
            "max_hp": 75,
            "gold": 99,
            "deck": [
                {
                    "index": 0,
                    "name": "打击",
                    "card_type": "Attack",
                    "energy_cost": 1,
                },
                {
                    "index": 1,
                    "name": "防御",
                    "card_type": "Skill",
                    "energy_cost": 1,
                },
            ],
            "relics": [
                {
                    "index": 0,
                    "name": "破损核心",
                    "description": "战斗开始时生成1个闪电充能球。",
                }
            ],
            "potions": [
                {
                    "index": 0,
                    "name": "火焰药水",
                    "description": "对一个敌人造成20点伤害。",
                    "occupied": True,
                    "can_use": True,
                    "valid_target_indices": [0],
                }
            ],
        },
        "combat": {
            "player": {
                "current_hp": 44,
                "max_hp": 75,
                "block": 6,
                "energy": 2,
                "stars": 1,
                "focus": 2,
                "orb_capacity": 3,
                "empty_orb_slots": 2,
                "cards_played_this_turn": 3,
                "attacks_played_this_turn": 1,
                "skills_played_this_turn": 2,
                "card_play_counters_reliable": True,
                "powers": [
                    {
                        "name": "集中",
                        "amount": 2,
                        "power_id": "FOCUS_POWER",
                    },
                    {"name": "敏捷", "amount": 1},
                ],
                "orbs": [
                    {
                        "slot_index": 0,
                        "name": "闪电",
                        "passive_value": 3,
                        "evoke_value": 8,
                    }
                ],
            },
            "enemies": [
                {
                    "index": 0,
                    "name": "邪教徒",
                    "current_hp": 30,
                    "max_hp": 48,
                    "block": 0,
                    "powers": [{"name": "力量", "amount": 2}],
                    "intents": [
                        {
                            "intent_type": "Attack",
                            "label": "6",
                            "total_damage": 6,
                        }
                    ],
                },
                {
                    "index": 1,
                    "name": "树枝史莱姆",
                    "current_hp": 8,
                    "max_hp": 8,
                    "block": 0,
                    "intents": [
                        {
                            "intent_type": "StatusCard",
                            "label": "1",
                            "status_card_count": 1,
                        }
                    ],
                },
            ],
            "hand": [
                {
                    "index": 0,
                    "name": "打击",
                    "energy_cost": 1,
                    "star_cost": 0,
                    "resolved_rules_text": "造成[blue]6[/blue]点伤害。",
                    "target_type": "AnyEnemy",
                    "playable": True,
                    "valid_target_indices": [0],
                },
                {
                    "index": 1,
                    "name": "防御",
                    "energy_cost": 1,
                    "star_cost": 0,
                    "resolved_rules_text": "获得5点格挡。",
                    "target_type": "Self",
                    "playable": True,
                    "valid_target_indices": [],
                },
                {
                    "index": 2,
                    "name": "凡庸",
                    "energy_cost": -1,
                    "star_cost": 0,
                    "resolved_rules_text": "不能被打出。",
                    "target_type": "Self",
                    "playable": False,
                    "unplayable_reason": "unplayable",
                    "valid_target_indices": [],
                },
            ],
            "draw_count": 4,
            "discard_count": 2,
            "lethal_risks": [
                {
                    "risk_id": "incoming_damage",
                    "damage_after_block": 6,
                    "player_hp": 5,
                    "will_kill_player": True,
                }
            ],
        },
        "agent_view": {
            "combat": {
                "draw": [
                    {"line": "打击*2 [1费]：造成6点伤害。"},
                    {"line": "电击 [1费]：生成1个闪电充能球。"},
                ],
                "discard": [{"line": "防御*2 [1费]：获得5点格挡。"}],
                "exhaust": [{"line": "白噪声 [1费]：加入一张能力牌。"}],
            }
        },
    }

    observation = harness.build_observation(state)

    assert observation.layer is harness.HarnessLayer.BATTLE
    assert observation.available_actions == (
        "play_card",
        "use_potion",
        "discard_potion",
        "end_turn",
    )
    assert observation.text.startswith("玩家: HP 44/75 | 格挡6 | 能量2 | 星能1")
    assert "角色:" not in observation.text
    assert "金币:" not in observation.text
    assert "遗物:" not in observation.text
    assert "牌组 2 张" not in observation.text
    assert "=== 战斗" not in observation.text
    assert "玩家: HP 44/75 | 格挡6 | 能量2 | 星能1" in observation.text
    assert "    buff: 集中2 | 敏捷1" in observation.text
    assert observation.text.count("集中2") == 1
    assert (
        "    充能球(FIFO最旧→最新;下一激发=[0]): "
        "1/3 [0]闪电(被动3/激发8)" in observation.text
    )
    assert "  敌[0] 邪教徒: HP 30/48 | 格挡0" in observation.text
    assert "    buff: 力量2" in observation.text
    assert "    意图:攻击6" in observation.text
    assert "  敌[1] 树枝史莱姆: HP 8/8 | 格挡0" in observation.text
    assert "    意图:塞状态牌1" in observation.text
    assert "  [0]打击(1费)<AnyEnemy> 造成6点伤害。" in observation.text
    assert "  [1]防御(1费) 获得5点格挡。" in observation.text
    assert "  [2]凡庸 不能被打出。 (不可使用: unplayable)" in observation.text
    assert "造成6点伤害。 | 目标: [0]" not in observation.text
    assert "-1费" not in observation.text
    assert (
        "[0] 火焰药水 | 可使用 | 对一个敌人造成20点伤害。 | 目标: [0]"
        in observation.text
    )
    assert "对一个敌人造成20点伤害。" in observation.text
    assert "本回合已打出: 卡牌 3 | 攻击 1 | 技能 2" in observation.text
    assert (
        "抽牌堆（4张）:\n"
        "  打击*2(1费) 造成6点伤害。\n"
        "  电击(1费) 生成1个闪电充能球。" in observation.text
    )
    assert "弃牌堆（2张）:\n  防御*2(1费) 获得5点格挡。" in observation.text
    assert "消耗牌堆（1张）:\n  白噪声(1费) 加入一张能力牌。" in observation.text
    assert "遗物效果:" not in observation.text
    assert "危险: 预计承受6点未格挡伤害，足以致命。" in observation.text
    assert observation.text.endswith(
        "可执行动作:\n"
        "- play_card(card_index, target_index)\n"
        "- use_potion(option_index, target_index)\n"
        "- discard_potion(option_index)\n"
        "- end_turn"
    )


def test_battle_observation_hides_dead_enemies_without_renumbering_targets() -> None:
    """死亡敌人不再干扰模型选择，存活敌人的 Mod 目标索引保持不变。"""
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "COMBAT",
        "in_combat": True,
        "available_actions": ["play_card", "end_turn"],
        "run": {"relics": [], "potions": []},
        "combat": {
            "player": {
                "current_hp": 70,
                "max_hp": 75,
                "block": 0,
                "energy": 3,
                "stars": 0,
            },
            "enemies": [
                {
                    "index": 0,
                    "name": "已死亡敌人",
                    "current_hp": 0,
                    "max_hp": 20,
                    "block": 0,
                    "is_alive": False,
                },
                {
                    "index": 1,
                    "name": "存活敌人",
                    "current_hp": 20,
                    "max_hp": 20,
                    "block": 0,
                    "is_alive": True,
                },
            ],
            "hand": [
                {
                    "index": 0,
                    "name": "打击",
                    "energy_cost": 1,
                    "target_type": "AnyEnemy",
                    "requires_target": True,
                    "valid_target_indices": [1],
                    "playable": True,
                }
            ],
        },
    }

    observation = harness.build_observation(state)

    assert "已死亡敌人" not in observation.text
    assert "敌[1] 存活敌人" in observation.text


def test_combat_observation_renders_generic_card_and_relic_ui_state() -> None:
    """游戏提供的高亮与遗物 UI 状态按通用字段进入战斗观测。

    Raises:
        AssertionError: Harness 丢失金红高亮、计数、激活或禁用状态。

    Returns:
        None: 此测试只核对与具体卡牌、遗物 ID 无关的展示契约。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "COMBAT",
        "in_combat": True,
        "turn": 1,
        "available_actions": ["play_card", "end_turn"],
        "run": {
            "relics": [
                {
                    "index": 0,
                    "name": "双截棍",
                    "show_counter": True,
                    "counter_value": 9,
                    "status": "Active",
                    "is_used_up": False,
                },
                {
                    "index": 1,
                    "name": "翼靴",
                    "show_counter": True,
                    "counter_value": 2,
                    "status": "Disabled",
                    "is_used_up": True,
                },
                {
                    "index": 2,
                    "name": "普通遗物",
                    "show_counter": False,
                    "status": "Normal",
                    "is_used_up": False,
                },
            ],
            "potions": [],
        },
        "combat": {
            "player": {
                "current_hp": 70,
                "max_hp": 75,
                "block": 0,
                "energy": 3,
                "stars": 0,
            },
            "enemies": [],
            "hand": [
                {
                    "index": 0,
                    "name": "正面条件牌",
                    "energy_cost": 0,
                    "star_cost": 0,
                    "resolved_rules_text": "造成5点伤害。",
                    "target_type": "AnyEnemy",
                    "playable": True,
                    "should_glow_gold": True,
                    "should_glow_red": False,
                },
                {
                    "index": 1,
                    "name": "负面限制牌",
                    "energy_cost": 1,
                    "star_cost": 0,
                    "resolved_rules_text": "不能被打出。",
                    "target_type": "Self",
                    "playable": False,
                    "unplayable_reason": "unplayable",
                    "should_glow_gold": False,
                    "should_glow_red": True,
                },
            ],
            "draw_count": 0,
            "discard_count": 0,
        },
    }

    observation = harness.build_observation(state)

    assert "遗物UI:\n  [0] 双截棍〔计数 9；已高亮〕" in observation.text
    assert "  [1] 翼靴〔计数 2；已禁用；已耗尽〕" in observation.text
    assert "普通遗物" not in observation.text
    assert (
        "[0]正面条件牌(0费)<AnyEnemy> 造成5点伤害。 "
        "〔金光：有利条件满足〕" in observation.text
    )
    assert (
        "[1]负面限制牌(1费) 不能被打出。 "
        "〔红光：不利条件生效〕 (不可使用: unplayable)" in observation.text
    )


def test_combat_observation_does_not_invent_missing_relic_counter() -> None:
    """遗物声明显示计数但没有数值时，不把缺失值伪装成零。

    Raises:
        AssertionError: Harness 输出了未由 Mod 提供的计数值。

    Returns:
        None: 此测试验证计数缺失与其他 UI 状态相互独立。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "COMBAT",
        "in_combat": True,
        "available_actions": ["end_turn"],
        "run": {
            "relics": [
                {
                    "index": 0,
                    "name": "缺失计数",
                    "show_counter": True,
                    "status": "Normal",
                },
                {
                    "index": 1,
                    "name": "仅有高亮",
                    "show_counter": True,
                    "counter_value": None,
                    "status": "Active",
                },
            ],
            "potions": [],
        },
        "combat": {
            "player": {
                "current_hp": 70,
                "max_hp": 75,
                "block": 0,
                "energy": 3,
                "stars": 0,
            },
            "enemies": [],
            "hand": [],
            "draw_count": 0,
            "discard_count": 0,
        },
    }

    observation = harness.build_observation(state)

    assert "缺失计数" not in observation.text
    assert "[1] 仅有高亮〔已高亮〕" in observation.text
    assert "计数 0" not in observation.text
    assert "计数 None" not in observation.text


def test_combat_observation_translates_every_recorded_intent_type() -> None:
    """人类 raw 中出现过的非攻击意图全部渲染为中文。

    Raises:
        AssertionError: 结构化意图枚举直接泄漏为英文。

    Returns:
        None: 此测试只验证当前录制语料的完整意图词表。
    """
    harness = importlib.import_module("play_sts2.harness")
    intent_names = {
        "Buff": "增强",
        "Debuff": "削弱",
        "Defend": "防御",
        "StatusCard": "塞状态牌",
        "Summon": "召唤",
        "Heal": "治疗",
        "CardDebuff": "诅咒卡牌",
        "DebuffStrong": "强力削弱",
        "Sleep": "沉睡",
        "Stun": "眩晕",
        "DeathBlow": "处决",
        "Escape": "逃跑",
    }
    state = {
        "screen": "COMBAT",
        "in_combat": True,
        "turn": 1,
        "available_actions": ["end_turn"],
        "run": {"relics": [], "potions": []},
        "combat": {
            "player": {
                "current_hp": 75,
                "max_hp": 75,
                "block": 0,
                "energy": 3,
                "stars": 0,
            },
            "enemies": [
                {
                    "index": index,
                    "name": f"测试敌人{index}",
                    "current_hp": 10,
                    "max_hp": 10,
                    "block": 0,
                    "intents": [{"intent_type": intent_type}],
                }
                for index, intent_type in enumerate(intent_names)
            ],
            "hand": [],
        },
    }

    observation = harness.build_observation(state)

    for intent_type, expected in intent_names.items():
        assert f"意图:{expected}" in observation.text
        assert f"意图:{intent_type}" not in observation.text


def test_build_observation_renders_complete_strategic_map_context() -> None:
    """地图决策包含整局资源、已走路径、可达统计和完整邻接图。

    Raises:
        AssertionError: 战略模型看到的地图快照不足以比较完整路线。

    Returns:
        None: 此测试只验证地图决策所需的全局上下文。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "MAP",
        "in_combat": False,
        "available_actions": ["choose_map_node"],
        "run": {
            "act_id": "0",
            "ascension": 3,
            "ascension_effects": [
                {"name": "精英蜂拥", "description": "精英敌人出现更加频繁。"}
            ],
            "boss_id": "THE_KIN_BOSS",
            "character_name": "故障机器人",
            "current_hp": 60,
            "max_hp": 75,
            "gold": 110,
            "floor": 2,
            "potions": [
                {"index": 0, "occupied": False},
                {
                    "index": 1,
                    "occupied": True,
                    "name": "火焰药水",
                    "description": "造成20点伤害。",
                },
            ],
            "relics": [
                {
                    "index": 0,
                    "name": "破损核心",
                    "description": "生成1个闪电充能球。",
                    "stack_count": 2,
                }
            ],
            "deck": [
                {
                    "index": 0,
                    "name": "打击",
                    "card_type": "Attack",
                    "energy_cost": 1,
                    "upgraded": False,
                },
                {
                    "index": 1,
                    "name": "打击",
                    "card_type": "Attack",
                    "energy_cost": 1,
                    "upgraded": False,
                },
                {
                    "index": 2,
                    "name": "电击",
                    "card_type": "Skill",
                    "energy_cost": 0,
                    "upgraded": True,
                },
            ],
        },
        "map": {
            "current_node": {"row": 1, "col": 3},
            "boss_node": {"row": 4, "col": 3},
            "available_nodes": [
                {"index": 0, "row": 2, "col": 2, "node_type": "Monster"},
                {"index": 1, "row": 2, "col": 4, "node_type": "Unknown"},
            ],
            "nodes": [
                {
                    "row": 0,
                    "col": 3,
                    "node_type": "Ancient",
                    "visited": True,
                    "children": [{"row": 1, "col": 3}],
                },
                {
                    "row": 1,
                    "col": 3,
                    "node_type": "Monster",
                    "visited": True,
                    "is_current": True,
                    "children": [{"row": 2, "col": 2}, {"row": 2, "col": 4}],
                },
                {
                    "row": 2,
                    "col": 2,
                    "node_type": "Monster",
                    "children": [{"row": 3, "col": 3}],
                },
                {
                    "row": 2,
                    "col": 4,
                    "node_type": "Unknown",
                    "children": [{"row": 3, "col": 3}],
                },
                {
                    "row": 3,
                    "col": 3,
                    "node_type": "RestSite",
                    "children": [{"row": 4, "col": 3}],
                },
                {
                    "row": 4,
                    "col": 3,
                    "node_type": "Boss",
                    "is_boss": True,
                    "children": [],
                },
            ],
        },
    }

    observation = harness.build_observation(state)

    assert "【第0幕】" in observation.text
    assert (
        "本幕Boss: 同族小队 (THE_KIN_BOSS) | 组成: 同族信徒、同族神官"
        in observation.text
    )
    assert "难度3: 精英蜂拥（精英敌人出现更加频繁。）" in observation.text
    assert "【当前状态】" in observation.text
    assert "HP 60/75 | 金币110 | 第2层" in observation.text
    assert "药水栏 1/2: [0] - [1] 火焰药水（造成20点伤害。）" in observation.text
    assert "遗物: 破损核心×2（生成1个闪电充能球。）" in observation.text
    assert "牌组 3 张（升级1）" in observation.text
    assert "- 打击 x2（1费攻击）" in observation.text
    assert "- 电击+ x1（0费技能）" in observation.text
    assert "位置: 行1 列3 | 已走: 古(行0)→敌(行1)" in observation.text
    assert "[0] 行2列2 敌 | 距Boss 2步 | 后续: 火 → 王" in observation.text
    assert "[1] 行2列4 ? | 距Boss 2步 | 后续: 火 → 王" in observation.text
    assert "本幕可达: ?×1 敌×1 火×1 王×1 | 最深可达 2步" in observation.text
    assert "=== 全图(坐标邻接, 未走层) ===" in observation.text
    assert "(1,3)@敌→(2,2)敌 (2,4)?" in observation.text
    assert "(3,3)火→(4,3)王" in observation.text


@pytest.mark.parametrize(
    ("boss_id", "expected"),
    [
        ("KAISER_CRAB_BOSS", "组成: 碾碎爪、火箭"),
        ("QUEEN_BOSS", "组成: 女王、火炬头聚合体"),
        ("THE_KIN_BOSS", "组成: 同族信徒、同族神官"),
    ],
)
def test_strategic_context_exposes_multi_monster_boss_encounters(
    boss_id: str,
    expected: str,
) -> None:
    """十二个遭遇 ID 中的复合遭遇显式覆盖十四个生产 Boss 单位。

    Args:
        boss_id (str): 游戏 ``run.boss_id`` 返回的遭遇 ID。
        expected (str): 该遭遇中必须展示的怪物组成。

    Raises:
        AssertionError: 复合遭遇隐藏了目录中的 Boss 怪物单位。

    Returns:
        None: 此测试只核对 Boss 遭遇与怪物单位的映射。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "MAP",
        "in_combat": False,
        "available_actions": ["choose_map_node"],
        "run": {
            "boss_id": boss_id,
            "relics": [],
            "potions": [],
            "deck": [],
        },
        "map": {
            "available_nodes": [
                {"index": 0, "row": 1, "col": 0, "node_type": "Monster"}
            ]
        },
    }

    observation = harness.build_observation(state)

    assert expected in observation.text


def test_build_observation_hides_unverified_historical_card_counters() -> None:
    """旧 raw 没有可靠性标记时不把恒为零的计数伪装成事实。

    Raises:
        AssertionError: 历史 Mod 的占位计数泄漏进模型观测。

    Returns:
        None: 此测试只验证旧录制数据的诚实降级。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "COMBAT",
        "in_combat": True,
        "available_actions": ["end_turn"],
        "run": {"act_id": 0},
        "combat": {
            "player": {
                "current_hp": 10,
                "max_hp": 10,
                "block": 0,
                "energy": 0,
                "stars": 0,
                "cards_played_this_turn": 0,
                "attacks_played_this_turn": 0,
                "skills_played_this_turn": 0,
            },
            "enemies": [],
            "hand": [],
        },
    }

    observation = harness.build_observation(state)

    assert "本回合已打出" not in observation.text


@pytest.mark.parametrize(
    ("state", "expected_snippets", "expected_actions"),
    [
        (
            {
                "screen": "EVENT",
                "in_combat": False,
                "available_actions": ["save_and_quit", "choose_event_option"],
                "event": {
                    "title": "涅奥",
                    "description": (
                        "[font_size=18]选择一份[gold]馈赠[/gold]。[/font_size]"
                    ),
                    "options": [
                        {
                            "index": 0,
                            "title": "熔岩石",
                            "description": "Boss额外掉落2件遗物。",
                            "is_locked": False,
                        },
                        {
                            "index": 1,
                            "title": "锁定选项",
                            "description": "尚不可用。",
                            "is_locked": True,
                        },
                    ],
                },
            },
            (
                "=== 事件: 涅奥 ===",
                "选择一份馈赠。",
                "[0] 熔岩石 | Boss额外掉落2件遗物。",
                "[1] 锁定选项 | 已锁定 | 尚不可用。",
            ),
            ("choose_event_option",),
        ),
        (
            {
                "screen": "MAP",
                "in_combat": False,
                "available_actions": ["save_and_quit", "choose_map_node"],
                "map": {
                    "available_nodes": [
                        {"index": 0, "row": 4, "col": 1, "node_type": "Monster"},
                        {"index": 1, "row": 4, "col": 3, "node_type": "RestSite"},
                    ]
                },
            },
            (
                "=== 地图 ===",
                "[0] 第 4 行，第 1 列 | 普通敌人",
                "[1] 第 4 行，第 3 列 | 休息处",
            ),
            ("choose_map_node",),
        ),
        (
            {
                "screen": "REWARD",
                "in_combat": False,
                "available_actions": ["save_and_quit", "claim_reward", "proceed"],
                "reward": {
                    "rewards": [
                        {
                            "index": 0,
                            "name": "奥术卷轴",
                            "effect_description": "获得一张[gold]稀有牌[/gold]。",
                            "claimable": True,
                        },
                        {
                            "index": 1,
                            "name": "金币",
                            "effect_description": "获得20金币。",
                            "claimable": False,
                        },
                    ],
                    "card_options": [],
                    "alternatives": [],
                },
            },
            (
                "=== 奖励 ===",
                "[0] 奥术卷轴 | 获得一张稀有牌。",
                "[1] 金币 | 暂不可领取 | 获得20金币。",
                "- claim_reward(option_index)",
            ),
            ("claim_reward", "proceed"),
        ),
        (
            {
                "screen": "CARD_SELECTION",
                "in_combat": False,
                "available_actions": [
                    "save_and_quit",
                    "choose_reward_card",
                    "select_deck_card",
                    "skip_reward_cards",
                ],
                "selection": {
                    "prompt": "选择[blue]1[/blue]张牌。",
                    "selected_count": 0,
                    "min_select": 0,
                    "max_select": 1,
                    "cards": [
                        {
                            "index": 0,
                            "name": "球状闪电",
                            "upgraded": True,
                            "energy_cost": 1,
                            "star_cost": 0,
                            "resolved_rules_text": "造成7点伤害。生成1个闪电充能球。",
                            "selected": False,
                        }
                    ],
                },
            },
            (
                "=== 选择卡牌 ===",
                "选择1张牌。",
                "[0]球状闪电+(1费) 造成7点伤害。生成1个闪电充能球。",
            ),
            ("choose_reward_card", "skip_reward_cards"),
        ),
    ],
)
def test_build_observation_renders_strategic_decisions(
    state: dict[str, Any],
    expected_snippets: tuple[str, ...],
    expected_actions: tuple[str, ...],
) -> None:
    """常见战略屏幕展示带稳定索引的候选项和模型动作。

    Args:
        state (dict[str, Any]): Mod 返回的战略决策状态。
        expected_snippets (tuple[str, ...]): 观测中必须出现的可读信息。
        expected_actions (tuple[str, ...]): 当前模型可选择的战略动作。

    Raises:
        AssertionError: 战略观测遗漏选项、泄漏标记或动作过滤错误。

    Returns:
        None: 此测试只验证生成的战略观测。
    """
    harness = importlib.import_module("play_sts2.harness")
    state["run"] = {
        "character_name": "故障机器人",
        "ascension": 3,
        "act_id": "0",
        "floor": 5,
        "current_hp": 60,
        "max_hp": 75,
        "gold": 99,
        "relics": [{"index": 0, "name": "破损核心"}],
        "potions": [{"index": 0, "name": "空", "occupied": False}],
        "deck": [
            {"index": 0, "name": "打击", "upgraded": False},
            {"index": 1, "name": "打击", "upgraded": False},
            {"index": 2, "name": "电击", "upgraded": True},
        ],
    }

    observation = harness.build_observation(state)

    assert observation.layer is harness.HarnessLayer.STRATEGIC
    assert observation.available_actions == expected_actions
    assert "遗物: 破损核心" in observation.text
    assert "药水栏 0/1: [0] -" in observation.text
    assert "牌组 3 张（升级1）" in observation.text
    assert "- 打击 x2（未知）" in observation.text
    assert "- 电击+ x1（未知）" in observation.text
    assert "[font_size" not in observation.text
    for snippet in expected_snippets:
        assert snippet in observation.text


def test_build_observation_rejects_state_without_model_decision() -> None:
    """过渡状态不会生成空洞观测并误唤醒模型。

    Returns:
        None: 此测试只验证无决策状态会明确失败。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "CAPSTONE_SELECTION",
        "in_combat": False,
        "available_actions": ["choose_capstone_option"],
    }

    with pytest.raises(harness.ObservationError, match="无需模型决策"):
        harness.build_observation(state)
