"""验证战略 Harness 的紧凑中文展示契约。"""

import importlib


def test_strategic_context_uses_chinese_vertical_item_lists() -> None:
    """Boss、难度、药水与牌组应使用中文逐行格式且不重复内部字段。

    Returns:
        None: 英文 ID、横向长句或重复升级信息都会使测试失败。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "EVENT",
        "in_combat": False,
        "available_actions": ["choose_event_option"],
        "run": {
            "act_id": "0",
            "boss_id": "KAISER_CRAB_BOSS",
            "character_id": "DEFECT",
            "character_name": "故障机器人",
            "ascension": 2,
            "ascension_effects": [
                {"name": "精英蜂拥", "description": "精英敌人出现更加频繁。"},
                {"name": "旅途劳顿", "description": "回复量降低至80%。"},
            ],
            "current_hp": 60,
            "max_hp": 75,
            "gold": 99,
            "floor": 2,
            "max_energy": 3,
            "base_orb_slots": 3,
            "potions": [
                {
                    "index": 0,
                    "occupied": True,
                    "name": "敏捷药水",
                    "description": "获得2点敏捷。",
                },
                {"index": 1, "occupied": False},
                {
                    "index": 2,
                    "occupied": True,
                    "name": "火焰药水",
                    "description": "造成20点伤害。",
                },
            ],
            "relics": [],
            "deck": [
                {
                    "index": 0,
                    "card_id": "STRIKE_DEFECT",
                    "name": "打击",
                    "upgraded": True,
                    "upgrade_level": 1,
                    "card_type": "Attack",
                    "energy_cost": 1,
                    "resolved_rules_text": "造成9点伤害。",
                },
                {
                    "index": 1,
                    "card_id": "DEFEND_DEFECT",
                    "name": "防御",
                    "upgraded": False,
                    "card_type": "Skill",
                    "energy_cost": 1,
                    "enchantment_id": "NIMBLE",
                    "enchantment_amount": 2,
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
        },
        "event": {
            "title": "测试事件",
            "options": [{"index": 0, "title": "继续"}],
        },
    }

    text = harness.build_observation(state).text

    assert "本幕Boss: 帝皇蟹 | 组成: 碾碎爪、火箭" in text
    assert (
        "难度2:\n- 精英蜂拥：精英敌人出现更加频繁。\n- 旅途劳顿：回复量降低至80%。"
        in text
    )
    assert (
        "药水栏 2/3:\n"
        "- [0] 敏捷药水：获得2点敏捷。\n"
        "- [1] 空\n"
        "- [2] 火焰药水：造成20点伤害。" in text
    )
    assert "角色: 故障机器人 | 最大能量3 | 基础充能球槽3" in text
    assert "- [0] 打击+（1费攻击）：造成9点伤害。" in text
    assert "- [1] 防御（1费技能）〔附魔：灵巧×2〕：获得7点格挡。" in text
    assert "KAISER_CRAB_BOSS" not in text
    assert "STRIKE_DEFECT" not in text
    assert "DEFEND_DEFECT" not in text
    assert "NIMBLE" not in text
    assert "升级等级" not in text
    assert "永久数值" not in text
    assert "Block=" not in text


def test_upgrade_selection_shows_real_upgraded_card_in_deck_format() -> None:
    """升级选牌应只显示 Mod 返回的升级后效果并沿用牌组行格式。

    Returns:
        None: 当前效果被冒充为升级效果或候选格式分叉时测试失败。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "CARD_SELECTION",
        "in_combat": False,
        "available_actions": ["select_deck_card"],
        "run": _empty_run(),
        "selection": {
            "kind": "deck_upgrade_select",
            "prompt": "选择1张牌来升级。",
            "cards": [
                {
                    "index": 0,
                    "name": "打击",
                    "upgraded": False,
                    "card_type": "Attack",
                    "energy_cost": 1,
                    "resolved_rules_text": "造成6点伤害。",
                    "upgrade_preview": {
                        "index": 0,
                        "name": "打击",
                        "upgraded": True,
                        "card_type": "Attack",
                        "energy_cost": 1,
                        "resolved_rules_text": "造成9点伤害。",
                    },
                }
            ],
        },
    }

    text = harness.build_observation(state).text

    assert "选择1张牌来升级。以下展示升级后效果：" in text
    assert "- [0] 打击+（1费攻击）：造成9点伤害。" in text
    assert "造成6点伤害。" not in text


def test_legacy_upgrade_selection_labels_unavailable_preview_as_current() -> None:
    """旧录像缺少升级预览时必须明确把候选规则称为当前效果。

    Returns:
        None: 当前效果再次被标题冒充为升级后效果时测试失败。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "CARD_SELECTION",
        "in_combat": False,
        "available_actions": ["select_deck_card"],
        "run": _empty_run(),
        "selection": {
            "kind": "deck_upgrade_select",
            "prompt": "选择1张牌来升级。",
            "cards": [
                {
                    "index": 0,
                    "name": "打击",
                    "upgraded": False,
                    "card_type": "Attack",
                    "energy_cost": 1,
                    "resolved_rules_text": "造成6点伤害。",
                }
            ],
        },
    }

    text = harness.build_observation(state).text

    assert "旧录像未保存升级预览，以下展示当前效果：" in text
    assert "- [0] 打击（1费攻击）：造成6点伤害。" in text
    assert "以下展示升级后效果" not in text


def test_legacy_upgrade_selection_without_prompt_still_labels_current_effect() -> None:
    """旧升级录像即使没有页面提示，也必须声明候选只是当前效果。

    Returns:
        None: 空 prompt 绕过历史降级说明时测试失败。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "CARD_SELECTION",
        "in_combat": False,
        "available_actions": ["select_deck_card"],
        "run": _empty_run(),
        "selection": {
            "kind": "deck_upgrade_select",
            "prompt": "",
            "cards": [
                {
                    "index": 0,
                    "name": "打击",
                    "card_type": "Attack",
                    "energy_cost": 1,
                    "resolved_rules_text": "造成6点伤害。",
                }
            ],
        },
    }

    text = harness.build_observation(state).text

    assert "旧录像未保存升级预览，以下展示当前效果：" in text
    assert "- [0] 打击（1费攻击）：造成6点伤害。" in text


def test_strategic_card_format_preserves_energy_and_star_costs() -> None:
    """战略卡牌行必须同时保留普通、X 和星能费用。

    Returns:
        None: 任一种玩家可见费用在统一格式中丢失时测试失败。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "REWARD",
        "in_combat": False,
        "available_actions": ["choose_reward_card"],
        "run": _empty_run(),
        "reward": {
            "card_options": [
                {
                    "index": 0,
                    "name": "七星",
                    "card_type": "Attack",
                    "energy_cost": 2,
                    "star_cost": 7,
                    "resolved_rules_text": "造成伤害。",
                },
                {
                    "index": 1,
                    "name": "星尘",
                    "card_type": "Skill",
                    "energy_cost": 0,
                    "star_costs_x": True,
                    "resolved_rules_text": "获得星能。",
                },
                {
                    "index": 2,
                    "name": "聚变",
                    "card_type": "Skill",
                    "costs_x": True,
                    "star_cost": 2,
                    "resolved_rules_text": "生成充能球。",
                },
            ]
        },
    }

    text = harness.build_observation(state).text

    assert "- [0] 七星（2费+7星能攻击）：造成伤害。" in text
    assert "- [1] 星尘（0费+X星能技能）：获得星能。" in text
    assert "- [2] 聚变（X费+2星能技能）：生成充能球。" in text


def test_shop_inventory_uses_the_same_vertical_item_format() -> None:
    """商店卡牌、遗物与药水应共享逐行价格、状态和效果布局。

    Returns:
        None: 任一库存类别继续使用管道拼接旧格式时测试失败。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "SHOP",
        "in_combat": False,
        "available_actions": ["buy_card", "buy_relic", "buy_potion", "proceed"],
        "run": _empty_run(),
        "shop": {
            "is_open": True,
            "cards": [
                {
                    "index": 0,
                    "name": "球状闪电",
                    "card_id": "BALL_LIGHTNING",
                    "card_type": "Attack",
                    "energy_cost": 1,
                    "resolved_rules_text": "造成7点伤害。生成1个闪电充能球。",
                    "price": 75,
                    "is_stocked": True,
                    "enough_gold": True,
                }
            ],
            "relics": [
                {
                    "index": 0,
                    "name": "数据磁盘",
                    "description": "每场战斗开始时获得1点集中。",
                    "price": 150,
                    "is_stocked": True,
                    "enough_gold": False,
                }
            ],
            "potions": [
                {
                    "index": 0,
                    "name": "火焰药水",
                    "description": "造成20点伤害。",
                    "price": 50,
                    "is_stocked": True,
                    "enough_gold": True,
                }
            ],
            "card_removal": {
                "price": 75,
                "available": True,
                "used": False,
                "enough_gold": True,
            },
        },
    }

    text = harness.build_observation(state).text

    assert (
        "卡牌:\n- [0] 球状闪电（1费攻击）〔75金币〕："
        "造成7点伤害。生成1个闪电充能球。" in text
    )
    assert (
        "遗物:\n- [0] 数据磁盘〔150金币；金币不足〕："
        "每场战斗开始时获得1点集中。" in text
    )
    assert "药水:\n- [0] 火焰药水〔50金币〕：造成20点伤害。" in text
    assert "删牌〔75金币；可用〕" in text
    assert "BALL_LIGHTNING" not in text


def test_strategic_cards_view_uses_the_deck_card_format() -> None:
    """战略牌组浏览页也应复用牌组卡牌行而非战斗手牌格式。

    Returns:
        None: 浏览页重新泄漏英文 ID 或产生第二套卡牌格式时测试失败。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "CARDS_VIEW",
        "in_combat": False,
        "available_actions": ["close_cards_view"],
        "run": _empty_run(),
        "cards_view": {
            "prompt": "查看牌组",
            "cards": [
                {
                    "index": 0,
                    "card_id": "DEFEND_DEFECT",
                    "name": "防御",
                    "upgraded": True,
                    "card_type": "Skill",
                    "energy_cost": 1,
                    "resolved_rules_text": "获得8点格挡。",
                }
            ],
        },
    }

    text = harness.build_observation(state).text

    assert "- [0] 防御+（1费技能）：获得8点格挡。" in text
    assert "DEFEND_DEFECT" not in text
    assert "升级等级" not in text


def _empty_run() -> dict[str, object]:
    """返回战略格式测试共用的最小整局状态。

    Returns:
        dict[str, object]: 不引入无关牌组、遗物或药水文本的状态。
    """
    return {
        "act_id": "0",
        "ascension": 0,
        "current_hp": 60,
        "max_hp": 75,
        "gold": 99,
        "floor": 2,
        "potions": [],
        "relics": [],
        "deck": [],
    }
