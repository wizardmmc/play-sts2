"""验证可供正式命令与 E2E 共用的隔离游戏启动基础。"""

import importlib
import json
from pathlib import Path
from typing import Any

import pytest


def test_default_app_is_project_pinned_v01071_runtime() -> None:
    """默认启动器不能继续指向会被 Steam 自动更新的安装目录。

    Raises:
        AssertionError: 默认应用路径未指向项目专用 v0.107.1 副本。

    Returns:
        None: 此测试只检查启动器的默认路径。
    """
    launcher = importlib.import_module("play_sts2.game_launcher")

    expected = (
        Path(__file__).resolve().parents[1]
        / ".runtime/SlayTheSpire2-v0.107.1/SlayTheSpire2.app"
    )
    assert launcher.DEFAULT_APP == expected


class ExitedProcess:
    """模拟已经自行退出但仍需回收的游戏进程。"""

    pid = 42

    def __init__(self) -> None:
        """初始化等待计数。

        Returns:
            None: 此方法只初始化测试替身。
        """
        self.wait_calls = 0

    def poll(self) -> int:
        """返回已退出状态。

        Returns:
            int: 模拟游戏正常退出的状态码。
        """
        return 0

    def wait(self, *, timeout: int) -> int:
        """记录领导进程被回收。

        Args:
            timeout (int): 允许回收进程的最长秒数。

        Returns:
            int: 模拟游戏正常退出的状态码。
        """
        assert timeout == 10
        self.wait_calls += 1
        return 0


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
    """正式启动器根据模式生成与 E2E 一致的游戏参数。

    Args:
        mode (str): 待验证的游戏显示模式。
        expected (list[str]): 期望传给游戏进程的完整参数。

    Raises:
        AssertionError: 启动参数没有正确区分无头和有头模式。

    Returns:
        None: 此测试只验证公开的启动命令构造函数。
    """
    launcher = importlib.import_module("play_sts2.game_launcher")

    assert launcher.game_command(Path("/game/sts2"), mode) == expected


def test_game_command_rejects_unknown_mode() -> None:
    """正式启动器拒绝未定义的显示模式。

    Raises:
        AssertionError: 未知模式没有触发明确错误。

    Returns:
        None: 此测试只验证启动模式输入边界。
    """
    launcher = importlib.import_module("play_sts2.game_launcher")

    with pytest.raises(ValueError, match="未知的 STS2 启动模式"):
        launcher.game_command(Path("/game/sts2"), "invalid")


def test_stop_game_does_not_signal_an_exited_process(monkeypatch: Any) -> None:
    """游戏已自行退出时只回收进程，不再向旧 PID 发信号。

    Args:
        monkeypatch (Any): Pytest 提供的属性替换工具。

    Raises:
        AssertionError: 清理函数仍终止已退出的进程组或未回收进程。

    Returns:
        None: 此测试只验证正常退出路径的清理边界。
    """
    launcher = importlib.import_module("play_sts2.game_launcher")
    process = ExitedProcess()

    def fail_killpg(_pid: int, _signal: int) -> None:
        """在清理函数错误发送信号时立即失败。

        Args:
            _pid (int): 不应使用的进程组 ID。
            _signal (int): 不应发送的信号。

        Raises:
            AssertionError: 此替身被调用时始终抛出。

        Returns:
            None: 此函数不会正常返回。
        """
        raise AssertionError("不应终止已退出的进程组")

    monkeypatch.setattr(launcher.os, "killpg", fail_killpg)

    launcher._stop_game(process)

    assert process.wait_calls == 1


def test_stage_profile_copies_only_isolated_non_steam_files(tmp_path: Path) -> None:
    """正式启动器只复制允许进入隔离 HOME 的存档文件。

    Args:
        tmp_path (Path): Pytest 提供的单测临时目录。

    Raises:
        AssertionError: 启动器泄漏其他存档或复制到 Steam 路径。

    Returns:
        None: 此测试只验证隔离存档的真实落盘结果。
    """
    launcher = importlib.import_module("play_sts2.game_launcher")
    profile = tmp_path / "profile"
    _write_profile(profile)
    isolated_home = tmp_path / "home"

    launcher.stage_profile(profile, isolated_home)

    data_root = isolated_home / "Library/Application Support/SlayTheSpire2"
    expected_files = {
        data_root / "default/1/settings.save": (profile / "settings.save").read_bytes(),
        data_root / "default/1/modded/profile1/saves/progress.save": (
            profile / "progress.save"
        ).read_bytes(),
        data_root / "default/1/modded/profile1/saves/prefs.save": (
            profile / "prefs.save"
        ).read_bytes(),
    }
    assert {path for path in data_root.rglob("*") if path.is_file()} == set(
        expected_files
    )
    for path, expected in expected_files.items():
        assert path.read_bytes() == expected


def test_stage_profile_accepts_exact_combat_solver_teacher_mods(
    tmp_path: Path,
) -> None:
    """教师存档允许同时启用 Agent、RitsuLib 与 CombatSolver。

    Args:
        tmp_path (Path): Pytest 提供的单测临时目录。

    Raises:
        AssertionError: 精确教师 Mod 白名单仍被启动器拒绝。

    Returns:
        None: 此测试只验证教师 profile 的启动前边界。
    """
    launcher = importlib.import_module("play_sts2.game_launcher")
    profile = tmp_path / "profile"
    profile.mkdir()
    settings = {
        "mod_settings": {
            "mods_enabled": True,
            "mod_list": [
                {"id": "UnifiedSavePath", "is_enabled": False},
                {"id": "STS2AIAgent", "is_enabled": True},
                {"id": "STS2-RitsuLib", "is_enabled": True},
                {"id": "CombatSolver", "is_enabled": True},
            ],
        }
    }
    (profile / "settings.save").write_text(json.dumps(settings), encoding="utf-8")
    (profile / "progress.save").write_text('{"unique_id":"TEST"}', encoding="utf-8")
    (profile / "prefs.save").write_text(
        '{"fast_mode":"fast","upload_data":false}',
        encoding="utf-8",
    )

    enabled_mods = launcher.stage_profile(profile, tmp_path / "home")

    assert enabled_mods == frozenset({"STS2AIAgent", "STS2-RitsuLib", "CombatSolver"})


def test_teacher_log_verification_requires_all_three_mods(tmp_path: Path) -> None:
    """教师启动日志必须证明三个预期 Mod 都完成初始化。

    Args:
        tmp_path (Path): Pytest 提供的单测临时目录。

    Raises:
        AssertionError: 教师日志漏掉依赖 Mod 后仍通过验证。

    Returns:
        None: 此测试只验证启动后的教师 Mod 白名单。
    """
    launcher = importlib.import_module("play_sts2.game_launcher")
    log_path = tmp_path / "game.log"
    expected_mods = frozenset({"STS2AIAgent", "STS2-RitsuLib", "CombatSolver"})
    log_path.write_text(
        (
            "[INFO] Steam initialization skipped (editor mode). Use --force-steam "
            "to enable.\n"
            "[INFO] Skipping loading mod UnifiedSavePath, it is set to disabled "
            "in settings\n"
            "[INFO] Finished mod initialization for 'STS2 AI Agent' "
            "(STS2AIAgent).\n"
            "[INFO] Finished mod initialization for 'RitsuLib' "
            "(STS2-RitsuLib).\n"
            "[INFO] Finished mod initialization for '战斗路线求解器' "
            "(CombatSolver).\n"
            "[INFO]  --- RUNNING MODDED! --- Loaded 3 mods (4 total)"
        ),
        encoding="utf-8",
    )

    launcher._verify_isolated_mod_configuration(log_path, expected_mods)

    log_path.write_text(
        log_path.read_text(encoding="utf-8").replace(
            "[INFO] Finished mod initialization for 'RitsuLib' (STS2-RitsuLib).\n",
            "",
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="无法确认游戏隔离"):
        launcher._verify_isolated_mod_configuration(log_path, expected_mods)


def test_repository_combat_solver_profile_is_stageable(tmp_path: Path) -> None:
    """仓库内教师 profile 应能直接用于隔离启动器。

    Args:
        tmp_path (Path): Pytest 提供的单测临时目录。

    Raises:
        AssertionError: 教师 profile 缺文件、启用错误 Mod 或无法复制。

    Returns:
        None: 此测试验证仓库交付的真实 profile。
    """
    launcher = importlib.import_module("play_sts2.game_launcher")
    profile = Path(__file__).resolve().parents[1] / "e2e/fixtures/combat-solver-profile"

    enabled_mods = launcher.stage_profile(profile, tmp_path / "home")

    assert enabled_mods == frozenset({"STS2AIAgent", "STS2-RitsuLib", "CombatSolver"})


def _write_profile(profile: Path) -> None:
    """写入只启用 Agent Mod 的最小隔离存档模板。

    Args:
        profile (Path): 待创建的存档模板目录。

    Returns:
        None: 设置与进度文件写入完成后返回。
    """
    profile.mkdir()
    settings = {
        "mod_settings": {
            "mods_enabled": True,
            "mod_list": [
                {"id": "UnifiedSavePath", "is_enabled": False},
                {"id": "STS2AIAgent", "is_enabled": True},
            ],
        }
    }
    (profile / "settings.save").write_text(json.dumps(settings), encoding="utf-8")
    (profile / "progress.save").write_text(
        '{"unique_id":"TEST"}',
        encoding="utf-8",
    )
    (profile / "prefs.save").write_text(
        '{"fast_mode":"fast","upload_data":false}',
        encoding="utf-8",
    )
