"""验证真实游戏 E2E 启动器的就绪条件。"""

import json
from pathlib import Path

import pytest

from . import conftest


def _write_test_profile(profile: Path, mod_list: list[dict[str, object]]) -> None:
    """写入不依赖完整性清单的最小测试档。

    Args:
        profile (Path): 待创建的测试档目录。
        mod_list (list[dict[str, object]]): 要写入设置文件的 Mod 状态列表。

    Returns:
        None: 测试档与清单写入完成后返回。
    """
    profile.mkdir()
    settings = json.dumps(
        {
            "mod_settings": {
                "mods_enabled": True,
                "mod_list": mod_list,
            }
        }
    ).encode()
    progress = b'{"unique_id":"E2E"}'
    (profile / "settings.save").write_bytes(settings)
    (profile / "progress.save").write_bytes(progress)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("headless", ["/game/sts2", "--force-steam=off", "--headless"]),
        ("headed", ["/game/sts2", "--force-steam=off"]),
    ],
)
def test_game_command_matches_requested_mode(
    mode: str,
    expected: list[str],
) -> None:
    """根据启动模式决定是否传入无头参数。

    Args:
        mode (str): 待验证的游戏启动模式。
        expected (list[str]): 期望生成的进程参数。

    Raises:
        AssertionError: 生成的进程参数与启动模式不匹配。

    Returns:
        None: 此测试仅验证启动参数生成规则。
    """
    assert conftest._game_command(Path("/game/sts2"), mode) == expected


def test_game_command_rejects_unknown_mode() -> None:
    """拒绝未定义的游戏启动模式。

    Raises:
        AssertionError: 未知模式没有触发 ``ValueError``。

    Returns:
        None: 此测试仅验证启动模式的输入边界。
    """
    with pytest.raises(ValueError, match="未知的 STS2 启动模式"):
        conftest._game_command(Path("/game/sts2"), "invalid")


def test_stage_test_profile_copies_only_non_steam_save_files(tmp_path: Path) -> None:
    """只把专用测试档复制到隔离 HOME 的非 Steam 布局。

    Args:
        tmp_path (Path): Pytest 提供的单测临时目录。

    Raises:
        AssertionError: 文件内容、目标位置或隔离边界不符合预期。

    Returns:
        None: 此测试仅验证专用测试档的落盘结果。
    """
    profile = tmp_path / "profile"
    _write_test_profile(
        profile,
        [
            {"id": "UnifiedSavePath", "is_enabled": False},
            {"id": "STS2AIAgent", "is_enabled": True},
        ],
    )
    isolated_home = tmp_path / "home"

    conftest._stage_test_profile(profile, isolated_home)

    data_root = isolated_home / "Library/Application Support/SlayTheSpire2"
    expected_files = {
        data_root / "default/1/settings.save": (profile / "settings.save").read_bytes(),
        data_root / "default/1/modded/profile1/saves/progress.save": (
            b'{"unique_id":"E2E"}'
        ),
    }
    assert {path for path in data_root.rglob("*") if path.is_file()} == set(
        expected_files
    )
    for path, expected_content in expected_files.items():
        assert path.read_bytes() == expected_content
    assert not list(data_root.rglob("current_run.save"))


@pytest.mark.parametrize(
    "mod_list",
    [
        [
            {"id": "UnifiedSavePath", "is_enabled": False},
            {"id": "STS2AIAgent", "is_enabled": True},
            {"id": "OtherMod", "is_enabled": True},
        ],
        [
            {"id": "UnifiedSavePath", "is_enabled": False},
            {"id": "STS2AIAgent", "is_enabled": True},
            {"id": "STS2AIAgent", "is_enabled": True},
        ],
    ],
)
def test_stage_test_profile_rejects_unsafe_mod_list(
    tmp_path: Path,
    mod_list: list[dict[str, object]],
) -> None:
    """在启动游戏前拒绝额外启用或 ID 重复的 Mod。

    Args:
        tmp_path (Path): Pytest 提供的单测临时目录。
        mod_list (list[dict[str, object]]): 不满足隔离要求的 Mod 状态列表。

    Raises:
        AssertionError: 不安全的 Mod 列表未被拒绝。

    Returns:
        None: 此测试仅验证启动前 Mod 白名单。
    """
    profile = tmp_path / "profile"
    _write_test_profile(profile, mod_list)

    with pytest.raises(ValueError, match="专用测试档的 Mod 配置无效"):
        conftest._stage_test_profile(profile, tmp_path / "home")


def test_isolated_mod_configuration_requires_disabled_unified_save_path(
    tmp_path: Path,
) -> None:
    """只接受关闭统一存档且仅加载 Agent 的运行时日志。

    Args:
        tmp_path (Path): Pytest 提供的单测临时目录。

    Raises:
        AssertionError: 不安全的 Mod 加载记录未被拒绝。

    Returns:
        None: 此测试仅验证运行时隔离守卫。
    """
    log_path = tmp_path / "game.log"
    log_path.write_text(
        (
            "[INFO] Steam initialization skipped (editor mode). Use --force-steam "
            "to enable.\n"
            "[INFO] Skipping loading mod UnifiedSavePath, it is set to disabled "
            "in settings\n"
            "[INFO] Finished mod initialization for 'STS2 AI Agent' "
            "(STS2AIAgent).\n"
            "[INFO]  --- RUNNING MODDED! --- Loaded 1 mods (2 total)"
        ),
        encoding="utf-8",
    )
    conftest._verify_isolated_mod_configuration(log_path)

    log_path.write_text(
        "[INFO] Finished mod initialization for 'STS2 AI Agent' (STS2AIAgent).",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="无法确认测试存档隔离"):
        conftest._verify_isolated_mod_configuration(log_path)

    log_path.write_text(
        (
            "[INFO] Steam initialization skipped (editor mode). Use --force-steam "
            "to enable.\n"
            "[INFO] Skipping loading mod UnifiedSavePath, it is set to disabled "
            "in settings\n"
            "[INFO] Finished mod initialization for 'STS2 AI Agent' "
            "(STS2AIAgent).\n"
            "[INFO]  --- RUNNING MODDED! --- Loaded 1 mods (2 total)\n"
            "[INFO] Steamworks initialization succeeded!"
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="无法确认测试存档隔离"):
        conftest._verify_isolated_mod_configuration(log_path)


def test_main_menu_readiness_requires_expected_action() -> None:
    """只有主菜单已经暴露开角色选择动作时才视为就绪。

    Raises:
        AssertionError: 启动器错误接受过渡状态或拒绝可操作主菜单。

    Returns:
        None: 此测试仅验证游戏实例的就绪边界。
    """
    assert not conftest._main_menu_is_ready(
        {
            "ok": True,
            "data": {
                "screen": "UNKNOWN",
                "available_actions": [],
            },
        }
    )
    assert not conftest._main_menu_is_ready(
        {
            "ok": True,
            "data": {
                "screen": "MAIN_MENU",
                "available_actions": ["continue_run"],
            },
        }
    )
    assert conftest._main_menu_is_ready(
        {
            "ok": True,
            "data": {
                "screen": "MAIN_MENU",
                "available_actions": ["open_character_select", "open_timeline"],
            },
        }
    )
