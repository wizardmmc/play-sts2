"""验证战略观测的完整玩家可见状态。"""

import importlib


def test_event_decision_includes_complete_remaining_map() -> None:
    """非 Proceed-only 的事件决策必须同时显示当前坐标和完整剩余地图。

    Returns:
        None: 删除非 MAP 页面地图拼接会使本测试失败。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = _strategic_state()
    state.update(
        {
            "screen": "EVENT",
            "available_actions": ["choose_event_option"],
            "event": {
                "event_id": "TEST_EVENT",
                "title": "测试事件",
                "description": "选择一条路。",
                "options": [
                    {
                        "index": 0,
                        "title": "继续",
                        "description": "获得奖励。",
                    }
                ],
            },
        }
    )

    observation = harness.build_observation(state)

    assert "=== 事件: 测试事件 ===" in observation.text
    assert "当前位置: 行1 列1" in observation.text
    assert "当前节点后续: 行2列0 ?、行2列2 火" in observation.text
    assert "本幕可达: ?×1 火×1 王×1 | 最深可达 1步" in observation.text
    assert "=== 全图(坐标邻接, 未走层) ===" in observation.text
    assert "(1,1)@敌→(2,0)? (2,2)火" in observation.text
    assert "(3,1)王→" not in observation.text


def test_strategic_context_preserves_card_instances_and_relic_runtime_state() -> None:
    """牌组实例、角色容量和遗物动态状态不能被分组或静态说明抹掉。

    Returns:
        None: 任一实例索引、中文附魔、当前规则或失效状态丢失都会失败。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = _strategic_state()
    state.update(
        {
            "screen": "REST",
            "available_actions": ["choose_rest_option"],
            "rest": {
                "options": [
                    {
                        "index": 0,
                        "option_id": "HEAL",
                        "title": "休息",
                        "description": "回复生命。",
                        "is_enabled": True,
                    }
                ]
            },
        }
    )

    observation = harness.build_observation(state)

    assert "角色: 故障机器人 | 最大能量3 | 基础充能球槽3" in observation.text
    assert "牌组 2 张（升级1，附魔1）" in observation.text
    assert "- [0] 打击+（1费攻击）：造成9点伤害。" in observation.text
    assert "- [1] 防御（1费技能）〔附魔：伶俐×2〕：获得7点格挡。" in observation.text
    assert "STRIKE_DEFECT" not in observation.text
    assert "DEFEND_DEFECT" not in observation.text
    assert "升级等级" not in observation.text
    assert "永久数值" not in observation.text
    assert "- [0] 羽翼之靴〔计数2〕：可无视路线。" in observation.text
    assert "- [1] 骨茶〔已耗尽；当前禁用〕" in observation.text
    assert "接下来1场战斗" not in observation.text
    assert "- [2] 蜡制小血瓶〔已熔毁；效果失效；当前禁用〕" in observation.text
    assert "战斗开始时回复2点生命" not in observation.text
    assert (
        "- [3] 南瓜蜡烛〔计数0；当前禁用〕：熄灭后可以在休息处添火。"
        in observation.text
    )


def test_proceed_only_decision_does_not_repeat_complete_map() -> None:
    """只有一个语义 Proceed 的页面只保留位置摘要，不重复整张地图。

    Returns:
        None: 已确认的 Proceed 例外保持紧凑。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = _strategic_state()
    state.update(
        {
            "screen": "UNKNOWN",
            "available_actions": ["proceed"],
        }
    )

    observation = harness.build_observation(state)

    assert "位置: 行1 列1" in observation.text
    assert "=== 全图(坐标邻接, 未走层) ===" not in observation.text
    assert observation.available_actions == ("proceed",)


def _strategic_state() -> dict[str, object]:
    """返回由 v0.111.0 raw 字段裁剪出的战略状态。

    Returns:
        dict[str, object]: 含逐卡、遗物动态字段和完整地图的状态。
    """
    return {
        "in_combat": False,
        "run": {
            "character_id": "DEFECT",
            "character_name": "故障机器人",
            "ascension": 3,
            "act_id": "0",
            "boss_id": "TEST_SUBJECT_BOSS",
            "floor": 5,
            "current_hp": 60,
            "max_hp": 75,
            "gold": 99,
            "max_energy": 3,
            "base_orb_slots": 3,
            "potions": [],
            "deck": [
                {
                    "index": 0,
                    "card_id": "STRIKE_DEFECT",
                    "name": "打击",
                    "upgraded": True,
                    "upgrade_level": 1,
                    "enchantment_id": None,
                    "enchantment_amount": None,
                    "card_type": "Attack",
                    "energy_cost": 1,
                    "resolved_rules_text": "造成9点伤害。",
                    "dynamic_values": [
                        {
                            "name": "Damage",
                            "base_value": 9,
                            "current_value": 9,
                            "enchanted_value": 9,
                            "is_modified": False,
                        }
                    ],
                },
                {
                    "index": 1,
                    "card_id": "DEFEND_DEFECT",
                    "name": "防御",
                    "upgraded": False,
                    "upgrade_level": 0,
                    "enchantment_id": "ADROIT",
                    "enchantment_name": "伶俐",
                    "enchantment_amount": 2,
                    "enchantment_description": "获得2点格挡。",
                    "card_type": "Skill",
                    "energy_cost": 1,
                    "resolved_rules_text": "获得7点格挡。",
                    "dynamic_values": [
                        {
                            "name": "Block",
                            "base_value": 5,
                            "current_value": 7,
                            "enchanted_value": 7,
                            "is_modified": True,
                        }
                    ],
                },
            ],
            "relics": [
                {
                    "index": 0,
                    "relic_id": "WINGED_BOOTS",
                    "name": "羽翼之靴",
                    "description": "可无视路线。",
                    "show_counter": True,
                    "counter_value": 2,
                    "status": "Normal",
                    "is_used_up": False,
                    "is_melted": False,
                },
                {
                    "index": 1,
                    "relic_id": "BONE_TEA",
                    "name": "骨茶",
                    "description": "接下来1场战斗升级初始手牌。",
                    "show_counter": False,
                    "counter_value": None,
                    "status": "Disabled",
                    "is_used_up": True,
                    "is_melted": False,
                },
                {
                    "index": 2,
                    "relic_id": "BLOOD_VIAL",
                    "name": "蜡制小血瓶",
                    "description": "战斗开始时回复2点生命。",
                    "show_counter": False,
                    "counter_value": None,
                    "status": "Disabled",
                    "is_used_up": False,
                    "is_melted": True,
                },
                {
                    "index": 3,
                    "relic_id": "PUMPKIN_CANDLE",
                    "name": "南瓜蜡烛",
                    "description": "熄灭后可以在休息处添火。",
                    "show_counter": True,
                    "counter_value": 0,
                    "status": "Disabled",
                    "is_used_up": False,
                    "is_melted": False,
                },
            ],
        },
        "map": {
            "current_node": {"row": 1, "col": 1},
            "available_nodes": [],
            "nodes": [
                {
                    "row": 1,
                    "col": 1,
                    "node_type": "Monster",
                    "visited": True,
                    "is_current": True,
                    "children": [{"row": 2, "col": 0}, {"row": 2, "col": 2}],
                },
                {
                    "row": 2,
                    "col": 0,
                    "node_type": "Unknown",
                    "children": [{"row": 3, "col": 1}],
                },
                {
                    "row": 2,
                    "col": 2,
                    "node_type": "RestSite",
                    "children": [{"row": 3, "col": 1}],
                },
                {
                    "row": 3,
                    "col": 1,
                    "node_type": "Boss",
                    "is_boss": True,
                    "children": [],
                },
            ],
        },
    }
