"""约束可提交到 Git 的公开 E2E 测试存档。"""

import json
from pathlib import Path
from typing import Any

_PROFILE = Path(__file__).parents[1] / "e2e/fixtures/profile"


def _read_json(path: Path) -> dict[str, Any]:
    """读取测试存档中的 JSON 对象。

    Args:
        path (Path): 待读取的存档文件。

    Raises:
        AssertionError: 文件根节点不是 JSON 对象。

    Returns:
        dict[str, Any]: 解析后的存档对象。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_profile_uses_synthetic_identity_and_neutral_history() -> None:
    """公开 fixture 只保留解锁数据，不携带个人身份和游玩历史。

    Raises:
        AssertionError: fixture 含随机身份、真实时间或非零游玩统计。

    Returns:
        None: 此测试仅验证公开测试档的数据边界。
    """
    progress = _read_json(_PROFILE / "progress.save")

    assert progress["unique_id"] == "E2E"
    assert all(epoch["obtain_date"] == 0 for epoch in progress["epochs"])
    assert progress["total_playtime"] == 0
    assert progress["floors_climbed"] == 0
    assert progress["current_score"] == 0
    assert all(character["playtime"] == 0 for character in progress["character_stats"])
    assert all(
        character["total_wins"] == 0 for character in progress["character_stats"]
    )
    assert all(
        character["total_losses"] == 0 for character in progress["character_stats"]
    )


def test_profile_keeps_the_full_unlock_state() -> None:
    """公开 fixture 保留当前游戏版本所需的完整解锁状态。

    Raises:
        AssertionError: 角色进阶或任一发现列表不完整。

    Returns:
        None: 此测试仅验证 E2E 可用的解锁范围。
    """
    progress = _read_json(_PROFILE / "progress.save")
    ascension_by_character = {
        character["id"]: character["max_ascension"]
        for character in progress["character_stats"]
    }

    assert ascension_by_character == {
        "CHARACTER.DEFECT": 10,
        "CHARACTER.IRONCLAD": 10,
        "CHARACTER.NECROBINDER": 10,
        "CHARACTER.REGENT": 10,
        "CHARACTER.SILENT": 10,
    }
    assert len(progress["epochs"]) == 57
    assert all(epoch["state"] == "revealed" for epoch in progress["epochs"])
    assert len(progress["discovered_cards"]) == 578
    assert len(progress["discovered_relics"]) == 297
    assert len(progress["discovered_potions"]) == 65
    assert len(progress["discovered_events"]) == 57
    assert len(progress["discovered_acts"]) == 4


def test_profile_contains_no_current_run() -> None:
    """公开 fixture 不包含从个人游戏复制来的进行中局面。

    Raises:
        AssertionError: fixture 中存在 ``current_run.save``。

    Returns:
        None: 此测试仅验证测试档从干净主菜单启动。
    """
    assert not list(_PROFILE.rglob("current_run.save"))


def test_profile_contains_no_local_identity_or_path() -> None:
    """公开 fixture 不包含账号标识或本机绝对路径。

    Raises:
        AssertionError: fixture 出现常见账号字段、邮箱或用户目录路径。

    Returns:
        None: 此测试仅验证可公开提交的数据范围。
    """
    progress = _read_json(_PROFILE / "progress.save")
    preferences = _read_json(_PROFILE / "prefs.save")
    settings = _read_json(_PROFILE / "settings.save")
    serialized = json.dumps(
        [progress, preferences, settings],
        ensure_ascii=False,
    ).casefold()

    for key in ("steam_id", "account_id", "player_name", "user_name", "email"):
        assert f'"{key}"' not in serialized
    assert "/users/" not in serialized
    assert "@" not in serialized


def test_profile_enables_only_the_agent_mod() -> None:
    """公开 fixture 关闭共享存档并只启用 Agent Mod。

    Raises:
        AssertionError: fixture 会加载额外 Mod 或统一存档功能。

    Returns:
        None: 此测试仅验证 E2E 的 Mod 隔离设置。
    """
    settings = _read_json(_PROFILE / "settings.save")
    mod_list = settings["mod_settings"]["mod_list"]
    mod_states = {mod["id"]: mod["is_enabled"] for mod in mod_list}

    assert settings["mod_settings"]["mods_enabled"] is True
    assert mod_states == {
        "STS2AIAgent": True,
        "STS2-RitsuLib": False,
        "CombatSolver": False,
        "UnifiedSavePath": False,
    }


def test_profile_enables_native_fast_mode_without_data_upload() -> None:
    """公开 fixture 默认使用游戏原生加速且关闭数据上传。

    Raises:
        AssertionError: fixture 未启用原生 fast mode 或仍允许遥测上传。

    Returns:
        None: 此测试只验证正式启动命令使用的 profile 偏好。
    """
    preferences = _read_json(_PROFILE / "prefs.save")

    assert preferences["fast_mode"] == "fast"
    assert preferences["upload_data"] is False
