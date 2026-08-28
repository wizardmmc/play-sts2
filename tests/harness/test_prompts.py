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


def test_battle_system_prompt_describes_relics_without_volatile_turn() -> None:
    """战斗 system 携带当前遗物效果，但不混入每回合变化的状态。

    Raises:
        AssertionError: 遗物信息缺失、混入动态回合或重复枚举动作签名。

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
    assert "【当前回合】" not in prompt
    assert "- [0] 破损核心: 战斗开始时生成1个闪电充能球。" in prompt
    assert "- [1] 熔毁测试遗物（已熔毁，效果失效）" in prompt
    assert "本应不再生效" not in prompt
    assert "ACTION: play_card" not in prompt
    assert "ACTION: use_potion" not in prompt
    assert "观测末尾列出的可执行动作" in prompt


def test_battle_system_prompt_preserves_resource_icon_meaning() -> None:
    """遗物说明中的通用能量和星能图标不会被清理成空白。

    Raises:
        AssertionError: system 丢失图标表达的数值或资源类型。

    Returns:
        None: 此测试只验证与具体遗物 ID 无关的富文本清理。
    """
    harness = importlib.import_module("play_sts2.harness")
    energy_icon = (
        "[img]res://images/packed/sprite_fonts/colorless_energy_icon.png[/img]"
    )
    bare_energy_icon = (
        "res://images/packed/sprite_fonts/defect_energy_icon.png"
    )
    star_icon = "[img]res://images/packed/sprite_fonts/star_icon.png[/img]"
    unknown_icon = "[img]res://images/icons/unknown_resource.webp[/img]"
    state = {
        "run": {
            "relics": [
                {
                    "index": 0,
                    "name": "通用能量测试",
                    "description": f"满足条件时获得{energy_icon}。",
                },
                {
                    "index": 1,
                    "name": "带数字测试",
                    "description": f"战斗开始时获得[blue]4{energy_icon}[/blue]。",
                },
                {
                    "index": 2,
                    "name": "连续星能测试",
                    "description": f"战斗开始时获得{star_icon}{star_icon}{star_icon}。",
                },
                {
                    "index": 3,
                    "name": "裸路径测试",
                    "description": f"满足条件时获得{bare_energy_icon}。",
                },
                {
                    "index": 4,
                    "name": "数字优先测试",
                    "description": f"满足条件时获得4{energy_icon}{energy_icon}。",
                },
                {
                    "index": 5,
                    "name": "未知资源测试",
                    "description": f"满足条件时获得{unknown_icon}。",
                },
            ]
        }
    }

    prompt = harness.system_prompt(harness.HarnessLayer.BATTLE, state)

    assert "满足条件时获得1点能量。" in prompt
    assert "战斗开始时获得4点能量。" in prompt
    assert "战斗开始时获得3点星能。" in prompt
    assert prompt.count("满足条件时获得1点能量。") == 2
    assert "满足条件时获得4点能量。" in prompt
    assert "满足条件时获得〔未知图标: unknown_resource〕。" in prompt
    assert "res://" not in prompt


def test_system_prompt_rejects_transient_layer() -> None:
    """过渡层没有决策权，因此不能加载模型提示词。

    Returns:
        None: 此测试只验证过渡层会明确失败。
    """
    harness = importlib.import_module("play_sts2.harness")

    with pytest.raises(ValueError, match="过渡层"):
        harness.system_prompt(harness.HarnessLayer.TRANSIENT)
