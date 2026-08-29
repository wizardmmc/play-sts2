"""验证模型动作文本与 Mod 请求参数之间的严格转换。"""

import importlib

import pytest


@pytest.mark.parametrize(
    ("text", "available_actions", "name", "parameters"),
    [
        (
            "ACTION: play_card 3 1",
            ["play_card", "end_turn"],
            "play_card",
            {"card_index": 3, "target_index": 1},
        ),
        (
            "ACTION: use_potion 0",
            ["use_potion", "end_turn"],
            "use_potion",
            {"option_index": 0},
        ),
        (
            "ACTION: choose_reward_alternative 1",
            ["choose_reward_alternative", "skip_reward_cards"],
            "choose_reward_alternative",
            {"option_index": 1},
        ),
        (
            "ACTION: choose_rest_option 1 2",
            ["choose_rest_option"],
            "choose_rest_option",
            {"option_index": 1, "target_index": 2},
        ),
        (
            "ACTION: end_turn",
            ["play_card", "end_turn"],
            "end_turn",
            {},
        ),
    ],
)
def test_parse_action_maps_canonical_text_to_mod_parameters(
    text: str,
    available_actions: list[str],
    name: str,
    parameters: dict[str, int],
) -> None:
    """规范动作行按动作签名转换为 Mod 请求参数。

    Args:
        text (str): 模型输出的单行规范动作。
        available_actions (list[str]): 当前状态开放的动作名称。
        name (str): 期望的稳定动作名称。
        parameters (dict[str, int]): 期望的 Mod 参数。

    Raises:
        AssertionError: 动作名称或参数映射不符合 Harness 契约。

    Returns:
        None: 此测试只验证解析后的结构化动作。
    """
    harness = importlib.import_module("play_sts2.harness")

    action = harness.parse_action(text, available_actions)

    assert action == harness.HarnessAction(name=name, parameters=parameters)
    assert harness.format_action(action) == text


def test_parse_action_uses_final_reasoning_line_when_thinking_content_is_empty() -> (
    None
):
    """思考模板未关闭时，只接受 reasoning 最后一行的规范动作。

    Raises:
        AssertionError: 空最终内容无法使用 reasoning 末尾的严格动作行。

    Returns:
        None: 此测试覆盖 Qwen 思考模板的真实分离响应。
    """
    harness = importlib.import_module("play_sts2.harness")

    action = harness.parse_action(
        "",
        ["play_card", "end_turn"],
        reasoning="先比较敌人意图。\n\nACTION: play_card 1 0",
    )

    assert action == harness.HarnessAction(
        name="play_card",
        parameters={"card_index": 1, "target_index": 0},
    )


def test_parse_action_does_not_replace_nonempty_content_with_reasoning() -> None:
    """非空最终输出不合法时，不得改用 reasoning 中的动作绕过协议。

    Returns:
        None: 此测试钉住 reasoning 回退只适用于空最终内容。
    """
    harness = importlib.import_module("play_sts2.harness")

    with pytest.raises(harness.ActionParseError):
        harness.parse_action(
            "我无法决定。",
            ["end_turn"],
            reasoning="ACTION: end_turn",
        )


@pytest.mark.parametrize(
    ("text", "available_actions"),
    [
        ("我建议先防御。\nACTION: play_card 1", ["play_card"]),
        ("ACTION: end_turn", ["play_card"]),
        ("ACTION: play_card", ["play_card"]),
        ("ACTION: end_turn 0", ["end_turn"]),
        ("ACTION: choose_map_node -1", ["choose_map_node"]),
    ],
)
def test_parse_action_rejects_noncanonical_or_unavailable_output(
    text: str,
    available_actions: list[str],
) -> None:
    """拒绝解释性文本、非法动作和错误参数数量。

    Args:
        text (str): 不符合严格输出契约的模型回复。
        available_actions (list[str]): 当前状态开放的动作名称。

    Returns:
        None: 此测试只验证非法输出会明确失败。
    """
    harness = importlib.import_module("play_sts2.harness")

    with pytest.raises(harness.ActionParseError):
        harness.parse_action(text, available_actions)


def test_legal_action_lines_enumerates_playable_cards_and_usable_potions() -> None:
    """合法动作集合应排除不可用索引，并展开必须目标的卡牌和药水。

    Raises:
        AssertionError: 不可用项目进入评测动作域，或合法目标没有完整展开。

    Returns:
        None: 此测试钉住 no-thinking 评测使用的实际动作边界。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "COMBAT",
        "in_combat": True,
        "available_actions": [
            "save_and_quit",
            "play_card",
            "use_potion",
            "discard_potion",
            "end_turn",
        ],
        "combat": {
            "hand": [
                {
                    "index": 0,
                    "playable": True,
                    "requires_target": True,
                    "valid_target_indices": [1],
                },
                {
                    "index": 1,
                    "playable": False,
                    "requires_target": False,
                    "valid_target_indices": [],
                },
            ]
        },
        "run": {
            "potions": [
                {
                    "index": 0,
                    "can_use": True,
                    "can_discard": True,
                    "requires_target": False,
                    "valid_target_indices": [],
                },
                {
                    "index": 1,
                    "can_use": True,
                    "can_discard": True,
                    "requires_target": True,
                    "valid_target_indices": [1, 2],
                },
                {
                    "index": 2,
                    "can_use": False,
                    "can_discard": False,
                    "requires_target": False,
                    "valid_target_indices": [],
                },
            ]
        },
    }

    assert harness.legal_action_lines(state) == (
        "ACTION: play_card 0 1",
        "ACTION: use_potion 0",
        "ACTION: use_potion 1 1",
        "ACTION: use_potion 1 2",
        "ACTION: discard_potion 0",
        "ACTION: discard_potion 1",
        "ACTION: end_turn",
    )


def test_legal_action_lines_falls_back_to_occupied_potions_when_flags_are_stale() -> (
    None
):
    """Mod 已开放用药但槽位标志全部滞后时，应保留占用槽位及其目标。

    Raises:
        AssertionError: 已被真实录制动作证明可用的药水从合法集合消失。

    Returns:
        None: 此测试复现人类轨迹中的全局动作与槽位标志不一致。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "COMBAT",
        "in_combat": True,
        "available_actions": ["use_potion"],
        "run": {
            "potions": [
                {
                    "index": 0,
                    "occupied": True,
                    "can_use": False,
                    "requires_target": True,
                    "valid_target_indices": [2],
                },
                {
                    "index": 1,
                    "occupied": False,
                    "can_use": False,
                    "requires_target": False,
                    "valid_target_indices": [],
                },
            ]
        },
    }

    assert harness.legal_action_lines(state) == ("ACTION: use_potion 0 2",)


def test_legal_action_lines_excludes_inactive_timeline_slots() -> None:
    """时间线动作只应展开 Mod 标记为可选择的纪元槽位。

    Raises:
        AssertionError: 不可操作的时间线索引进入合法动作集合。

    Returns:
        None: 此测试核对字段名与可读观测使用的契约一致。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "TIMELINE",
        "available_actions": ["choose_timeline_epoch"],
        "timeline": {
            "slots": [
                {"index": 0, "is_actionable": False},
                {"index": 1, "is_actionable": True},
            ]
        },
    }

    assert harness.legal_action_lines(state) == ("ACTION: choose_timeline_epoch 1",)


def test_legal_action_lines_obeys_reward_and_multiselect_business_rules() -> None:
    """替代奖励和已满多选应使用与 Mod 执行动作相同的业务边界。

    Raises:
        AssertionError: 专用跳过项或超过多选上限的卡牌被误记为合法。

    Returns:
        None: 此测试覆盖 Mod 会明确拒绝的两类参数。
    """
    harness = importlib.import_module("play_sts2.harness")
    reward_state = {
        "screen": "CARD_SELECTION",
        "available_actions": ["choose_reward_alternative", "skip_reward_cards"],
        "reward": {
            "alternatives": [
                {"index": 0, "option_id": "Skip"},
                {"index": 1, "option_id": "SACRIFICE"},
            ]
        },
    }
    selection_state = {
        "screen": "CARD_SELECTION",
        "in_combat": False,
        "available_actions": ["select_deck_card", "confirm_selection"],
        "selection": {
            "kind": "deck_card_select",
            "selected_count": 2,
            "max_select": 2,
            "cards": [
                {"index": 0, "selected": True},
                {"index": 1, "selected": True},
                {"index": 2, "selected": False},
            ],
        },
    }
    combat_hand_state = {
        "screen": "CARD_SELECTION",
        "in_combat": True,
        "available_actions": ["select_deck_card", "confirm_selection"],
        "selection": {
            "kind": "combat_hand_select",
            "selected_count": 1,
            "max_select": 1,
            "cards": [{"index": 0}, {"index": 1}],
        },
    }

    assert harness.legal_action_lines(reward_state) == (
        "ACTION: choose_reward_alternative 1",
        "ACTION: skip_reward_cards",
    )
    assert harness.legal_action_lines(selection_state) == (
        "ACTION: select_deck_card 0",
        "ACTION: select_deck_card 1",
        "ACTION: confirm_selection",
    )
    assert harness.legal_action_lines(combat_hand_state) == (
        "ACTION: select_deck_card 0",
        "ACTION: select_deck_card 1",
        "ACTION: confirm_selection",
    )


def test_legal_action_lines_uses_real_character_select_signatures() -> None:
    """角色锁定和无参数进阶动作应符合开局 Mod 契约。

    Raises:
        AssertionError: 锁定角色被展开，或进阶动作被错误要求索引。

    Returns:
        None: 此测试钉住公共动作签名的一致性。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "screen": "CHARACTER_SELECT",
        "available_actions": [
            "select_character",
            "increase_ascension",
            "decrease_ascension",
            "embark",
        ],
        "character_select": {
            "characters": [
                {"index": 0, "is_locked": False},
                {"index": 1, "is_locked": True},
            ]
        },
    }

    assert harness.legal_action_lines(state) == (
        "ACTION: select_character 0",
        "ACTION: increase_ascension",
        "ACTION: decrease_ascension",
        "ACTION: embark",
    )
    assert harness.action_signature("increase_ascension") == "increase_ascension"
