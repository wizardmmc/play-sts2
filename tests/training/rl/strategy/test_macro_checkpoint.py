"""验证战略宏 checkpoint 和纯 UI 自动动作。"""


def test_map_with_multiple_destinations_is_macro_checkpoint() -> None:
    """真实地图分叉应进入 Tree 候选池并保留语义类型。

    Returns:
        None: 三个目的地形成一个 MAP 宏 checkpoint。
    """
    from play_sts2.training.rl.strategy.macro import classify_macro_checkpoint

    checkpoint = classify_macro_checkpoint(
        {
            "screen": "MAP",
            "available_actions": ["save_and_quit", "choose_map_node"],
            "map": {
                "available_nodes": [
                    {"index": 0, "row": 2, "col": 1},
                    {"index": 1, "row": 2, "col": 3},
                    {"index": 2, "row": 2, "col": 5},
                ]
            },
        }
    )

    assert checkpoint is not None
    assert checkpoint.kind == "map"
    assert checkpoint.option_ids == ("map:2:1", "map:2:3", "map:2:5")


def test_single_proceed_event_is_automatic_not_macro() -> None:
    """已完成事件的唯一 Proceed 不应消耗模型调用或 Tree 预算。

    Returns:
        None: 自动动作使用真实 option index，节点不进入候选池。
    """
    from play_sts2.training.rl.strategy.macro import (
        automatic_strategic_action,
        classify_macro_checkpoint,
    )

    state = {
        "screen": "EVENT",
        "available_actions": ["save_and_quit", "choose_event_option"],
        "event": {
            "options": [
                {
                    "index": 4,
                    "text_key": "PROCEED",
                    "is_proceed": True,
                    "is_locked": False,
                }
            ]
        },
    }

    assert classify_macro_checkpoint(state) is None
    action = automatic_strategic_action(state)
    assert action is not None
    assert action.name == "choose_event_option"
    assert action.parameters == {"option_index": 4}


def test_reward_claim_order_is_not_mistaken_for_macro_choice() -> None:
    """金币与药水领取顺序不能伪装成两个战略计划。

    Returns:
        None: 没有卡牌选择入口时普通奖励页不是宏 checkpoint。
    """
    from play_sts2.training.rl.strategy.macro import classify_macro_checkpoint

    state = {
        "screen": "REWARD",
        "available_actions": ["claim_reward", "proceed"],
        "reward": {
            "rewards": [
                {"index": 0, "reward_type": "Gold", "claimable": True},
                {"index": 1, "reward_type": "Potion", "claimable": True},
            ]
        },
    }

    assert classify_macro_checkpoint(state) is None


def test_single_shop_purchase_and_leave_is_macro_checkpoint() -> None:
    """唯一可买项与结束购物仍是两个语义后继。

    Returns:
        None: 低金币商店不会因只剩一件可买物而漏出 Tree 候选池。
    """
    from play_sts2.training.rl.strategy.macro import classify_macro_checkpoint

    state = {
        "screen": "SHOP",
        "available_actions": ["buy_card", "close_shop_inventory"],
        "run": {"act_id": "0", "floor": 5},
        "shop": {
            "cards": [
                {
                    "index": 0,
                    "card_id": "ZAP",
                    "is_stocked": True,
                    "enough_gold": True,
                }
            ],
            "relics": [],
            "potions": [],
            "card_removal": {"available": False},
        },
    }

    checkpoint = classify_macro_checkpoint(state)

    assert checkpoint is not None
    assert checkpoint.option_ids == ("shop:card:0:ZAP", "shop:leave")
