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


def test_system_prompt_rejects_transient_layer() -> None:
    """过渡层没有决策权，因此不能加载模型提示词。

    Returns:
        None: 此测试只验证过渡层会明确失败。
    """
    harness = importlib.import_module("play_sts2.harness")

    with pytest.raises(ValueError, match="过渡层"):
        harness.system_prompt(harness.HarnessLayer.TRANSIENT)
