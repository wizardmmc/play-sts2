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
            ("=== 商店（库存已打开）===", "[0] 眼部攻击", "45 金币", "删牌: 75 金币"),
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
            ("=== 查看牌组 ===", "查看牌组", "[0] 防御"),
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
            "potions": [
                {
                    "index": 0,
                    "name": "火焰药水",
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
                    "playable": True,
                    "valid_target_indices": [0],
                },
                {
                    "index": 1,
                    "name": "防御",
                    "energy_cost": 1,
                    "star_cost": 0,
                    "resolved_rules_text": "获得5点格挡。",
                    "playable": True,
                    "valid_target_indices": [],
                },
                {
                    "index": 2,
                    "name": "凡庸",
                    "energy_cost": -1,
                    "star_cost": 0,
                    "resolved_rules_text": "不能被打出。",
                    "playable": False,
                    "unplayable_reason": "unplayable",
                    "valid_target_indices": [],
                },
            ],
            "draw_count": 4,
            "discard_count": 2,
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
    assert "角色: 故障机器人 | 进阶: 2 | 第 1 幕 | 第 3 层" in observation.text
    assert (
        "玩家: 生命 44/75 | 格挡 6 | 能量 2 | 星能 1 | 集中 2 | "
        "充能球槽 1/3" in observation.text
    )
    assert "[0] 邪教徒 | 生命 30/48 | 格挡 0 | 意图: 攻击 6" in observation.text
    assert "[1] 树枝史莱姆 | 生命 8/8 | 格挡 0 | 意图: 塞入状态牌 1" in (
        observation.text
    )
    assert "[0] 打击 | 1 能量 | 造成6点伤害。 | 目标: [0]" in observation.text
    assert "[2] 凡庸 | 不能被打出。 | 不可使用: unplayable" in observation.text
    assert "-1 能量" not in observation.text
    assert "[0] 火焰药水 | 可使用 | 目标: [0]" in observation.text
    assert "牌堆: 抽牌 4 | 弃牌 2" in observation.text
    assert observation.text.endswith(
        "可执行动作:\n- play_card\n- use_potion\n- discard_potion\n- end_turn"
    )


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
                "已选 0 | 至少 0 | 至多 1",
                "[0] 球状闪电+ | 1 能量 | 造成7点伤害。生成1个闪电充能球。",
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
    assert "遗物: [0] 破损核心" in observation.text
    assert "药水: [0] 空" in observation.text
    assert "牌组: 打击×2，电击+" in observation.text
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
