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
