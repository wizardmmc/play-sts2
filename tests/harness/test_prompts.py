"""验证 Harness 从独立资源目录加载模型提示词。"""

import importlib

import pytest


def test_system_prompt_loads_layer_specific_action_contracts() -> None:
    """战斗和战略层读取不同且包含规范动作契约的提示词。

    Raises:
        AssertionError: 提示词缺失、层级混用或没有声明动作输出契约。

    Returns:
        None: 此测试只验证提示词加载后的运行时内容。
    """
    harness = importlib.import_module("play_sts2.harness")

    battle = harness.system_prompt(harness.HarnessLayer.BATTLE)
    strategic = harness.system_prompt(harness.HarnessLayer.STRATEGIC)

    assert battle != strategic
    assert "战斗" in battle
    assert "战略" in strategic
    assert "ACTION:" in battle
    assert "ACTION:" in strategic
    assert "use_potion" in strategic
    assert "每次决策彼此独立" in battle
    assert "完整快照" in battle
    assert "金币是贯穿整局的资源" in strategic
    assert "可执行动作" in strategic


def test_battle_system_prompt_describes_relics_without_static_action_examples() -> None:
    """战斗 system 携带当前遗物效果，并把合法动作交给 user 观测。

    Raises:
        AssertionError: 遗物信息缺失或静态提示重复枚举动作签名。

    Returns:
        None: 此测试只验证动态战斗提示词。
    """
    harness = importlib.import_module("play_sts2.harness")
    state = {
        "turn": 2,
        "run": {
            "relics": [
                {
                    "index": 0,
                    "name": "破损核心",
                    "description": "战斗开始时[gold]生成[/gold]1个闪电充能球。",
                },
                {
                    "index": 1,
                    "name": "熔毁测试遗物",
                    "description": "本应不再生效。",
                    "is_melted": True,
                },
            ]
        },
    }

    prompt = harness.system_prompt(harness.HarnessLayer.BATTLE, state)

    assert "【遗物】" in prompt
    assert "【当前回合】2" in prompt
    assert "【当前回合】\n" not in prompt
    assert "- [0] 破损核心: 战斗开始时生成1个闪电充能球。" in prompt
    assert "- [1] 熔毁测试遗物（已熔毁，效果失效）" in prompt
    assert "本应不再生效" not in prompt
    assert "ACTION: play_card" not in prompt
    assert "ACTION: use_potion" not in prompt
    assert "观测末尾列出的可执行动作" in prompt


def test_system_prompt_rejects_transient_layer() -> None:
    """过渡层没有决策权，因此不能加载模型提示词。

    Returns:
        None: 此测试只验证过渡层会明确失败。
    """
    harness = importlib.import_module("play_sts2.harness")

    with pytest.raises(ValueError, match="过渡层"):
        harness.system_prompt(harness.HarnessLayer.TRANSIENT)
