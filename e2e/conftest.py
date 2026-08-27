"""为 E2E 测试启动并清理使用专用存档的 STS2 实例。"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

_DEFAULT_APP = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common"
    / "Slay the Spire 2/SlayTheSpire2.app"
)
_DEFAULT_PORT = 18081
_STARTUP_TIMEOUT_SECONDS = 120.0
_DATA_ROOT = Path("Library/Application Support/SlayTheSpire2")
_DEFAULT_PROFILE = Path(__file__).parent / "fixtures/profile"


@dataclass(frozen=True, slots=True)
class RunningGame:
    """记录 E2E 测试独占的游戏实例。

    Args:
        base_url (str): Agent Mod HTTP 服务根地址。
        home (Path): 游戏使用的隔离 HOME 目录。
    """

    base_url: str
    home: Path


def pytest_addoption(parser: pytest.Parser) -> None:
    """注册真实游戏 E2E 的启用参数与启动模式。

    Args:
        parser (pytest.Parser): Pytest 命令行参数解析器。

    Returns:
        None: 此函数注册 E2E 开关与无头、有头模式选项。
    """
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="启动真实 STS2 并运行 E2E 测试",
    )
    parser.addoption(
        "--sts2-mode",
        choices=("headless", "headed"),
        default="headless",
        help="选择 STS2 启动模式，默认为 headless",
    )


@pytest.fixture
def running_game(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Iterator[RunningGame]:
    """为当前测试提供使用专用存档的独立游戏实例。

    Args:
        request (pytest.FixtureRequest): 当前 Pytest 测试请求。
        tmp_path (Path): 当前测试独占的临时目录。

    Raises:
        ValueError: ``STS2_E2E_PORT`` 不是有效整数。
        FileNotFoundError: 找不到游戏文件或专用测试存档。
        RuntimeError: 端口已占用、Mod 隔离失败或游戏提前退出。
        TimeoutError: 游戏未能在限定时间内到达可操作主菜单。

    Yields:
        RunningGame: 已就绪的游戏实例及其隔离 HOME。
    """
    if not request.config.getoption("--run-e2e"):
        pytest.skip("使用 --run-e2e 才会启动真实游戏")
    if sys.platform != "darwin":
        pytest.skip("当前 E2E 启动器仅支持 macOS")

    app_path = Path(os.environ.get("STS2_APP_PATH", _DEFAULT_APP))
    executable = app_path / "Contents/MacOS/Slay the Spire 2"
    port = int(os.environ.get("STS2_E2E_PORT", _DEFAULT_PORT))
    mode = request.config.getoption("--sts2-mode")
    isolated_home = tmp_path / "home"
    isolated_home.mkdir()

    with _game(executable, port, isolated_home, _DEFAULT_PROFILE, mode) as game:
        yield game


@contextmanager
def _game(
    executable: Path,
    port: int,
    isolated_home: Path,
    profile: Path,
    mode: str,
) -> Iterator[RunningGame]:
    """启动一个由当前测试进程独占的隔离游戏实例。

    Args:
        executable (Path): STS2 的 macOS 可执行文件。
        port (int): Mod HTTP 服务监听端口。
        isolated_home (Path): 游戏实例使用的临时 HOME 目录。
        profile (Path): 专用测试存档目录。
        mode (str): ``headless`` 或 ``headed`` 启动模式。

    Raises:
        FileNotFoundError: 游戏文件或专用测试存档不存在。
        RuntimeError: 端口已占用、Mod 隔离失败或游戏提前退出。
        TimeoutError: 游戏未能在限定时间内到达可操作主菜单。
        ValueError: 启动模式或专用测试存档配置无效。

    Yields:
        RunningGame: 游戏到达可操作主菜单后的实例信息。
    """
    if not executable.is_file():
        raise FileNotFoundError(f"找不到 STS2 可执行文件: {executable}")

    steam_app_id = executable.parent / "steam_appid.txt"
    if not steam_app_id.is_file():
        raise FileNotFoundError(f"找不到游戏启动所需文件: {steam_app_id}")
    if _port_is_open(port):
        raise RuntimeError(f"E2E 端口已被占用: {port}")

    _stage_test_profile(profile, isolated_home)

    base_url = f"http://127.0.0.1:{port}"
    log_path = isolated_home / f"{mode}.log"
    environment = os.environ.copy()
    environment["HOME"] = str(isolated_home)
    environment["STS2_API_PORT"] = str(port)
    environment["STS2_ENABLE_DEBUG_ACTIONS"] = "1"

    with log_path.open("ab", buffering=0) as log_file:
        process = subprocess.Popen(
            _game_command(executable, mode),
            cwd=executable.parent,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        try:
            _wait_until_ready(process, base_url, log_path)
            _verify_isolated_mod_configuration(log_path)
            yield RunningGame(base_url=base_url, home=isolated_home)
        finally:
            _stop_game(process)


def _game_command(executable: Path, mode: str) -> list[str]:
    """生成禁用 Steam 且符合指定显示模式的游戏进程参数。

    Args:
        executable (Path): STS2 可执行文件路径。
        mode (str): ``headless`` 或 ``headed`` 启动模式。

    Raises:
        ValueError: ``mode`` 不是受支持的启动模式。

    Returns:
        list[str]: 包含 Steam 隔离参数的子进程参数列表。
    """
    command = [str(executable), "--force-steam=off"]
    if mode == "headless":
        return [*command, "--headless"]
    if mode == "headed":
        return command
    raise ValueError(f"未知的 STS2 启动模式: {mode}")


def _stage_test_profile(profile: Path, isolated_home: Path) -> None:
    """把受控测试档写入隔离 HOME 的非 Steam 布局。

    ``UnifiedSavePath`` 必须关闭，否则 Modded 游戏会回到共享存档目录，破坏
    隔离边界。

    Args:
        profile (Path): 包含 ``settings.save`` 和 ``progress.save`` 的测试档目录。
        isolated_home (Path): 当前 E2E 游戏独占的 HOME 目录。

    Raises:
        FileNotFoundError: 专用测试档缺少必需文件。
        ValueError: HOME 非空或 Mod 配置不满足隔离要求。
        OSError: 无法创建目录、读取或复制测试档。

    Returns:
        None: 专用非 Steam 存档布局准备完成后返回。
    """
    settings = profile / "settings.save"
    progress = profile / "progress.save"
    for path in (settings, progress):
        if not path.is_file():
            raise FileNotFoundError(f"专用测试档缺少文件: {path}")
    if isolated_home.exists() and any(isolated_home.iterdir()):
        raise ValueError(f"隔离 HOME 必须为空: {isolated_home}")

    try:
        payload = json.loads(settings.read_text(encoding="utf-8"))
        mod_settings = payload["mod_settings"]
        mod_list = mod_settings["mod_list"]
        if not isinstance(mod_settings, Mapping) or not isinstance(mod_list, list):
            raise TypeError
        mod_states: dict[str, bool] = {}
        for mod in mod_list:
            if not isinstance(mod, Mapping):
                raise TypeError
            mod_id = mod.get("id")
            is_enabled = mod.get("is_enabled")
            if (
                not isinstance(mod_id, str)
                or not isinstance(is_enabled, bool)
                or mod_id in mod_states
            ):
                raise TypeError
            mod_states[mod_id] = is_enabled
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"专用测试档的 Mod 配置无效: {settings}") from exc
    enabled_mods = {mod_id for mod_id, is_enabled in mod_states.items() if is_enabled}
    if (
        mod_settings.get("mods_enabled") is not True
        or enabled_mods != {"STS2AIAgent"}
        or mod_states.get("UnifiedSavePath") is not False
    ):
        raise ValueError(
            "专用测试档的 Mod 配置无效：必须仅启用 Agent 并关闭 UnifiedSavePath"
        )

    data_root = isolated_home / _DATA_ROOT
    targets = (
        (settings, data_root / "default/1/settings.save"),
        (progress, data_root / "default/1/modded/profile1/saves/progress.save"),
    )
    for source, target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _stop_game(process: subprocess.Popen[bytes]) -> None:
    """终止游戏创建的独立进程组并回收领导进程。

    Args:
        process (subprocess.Popen[bytes]): 使用独立会话启动的游戏进程。

    Raises:
        subprocess.TimeoutExpired: 进程组终止后领导进程仍无法回收。

    Returns:
        None: 游戏进程组已退出且领导进程已回收后返回。
    """
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def _verify_isolated_mod_configuration(log_path: Path) -> None:
    """从启动日志确认 Steam 与统一存档关闭且只加载 Agent Mod。

    Args:
        log_path (Path): 当前游戏进程的完整启动日志。

    Raises:
        RuntimeError: 日志无法读取或缺少任一隔离证据。

    Returns:
        None: 三项运行时隔离证据齐全后返回。
    """
    expected_lines = (
        "[INFO] Steam initialization skipped (editor mode). Use --force-steam to enable.",
        "[INFO] Skipping loading mod UnifiedSavePath, it is set to disabled in settings",
        "[INFO] Finished mod initialization for 'STS2 AI Agent' (STS2AIAgent).",
        "[INFO]  --- RUNNING MODDED! --- Loaded 1 mods (",
    )
    forbidden_lines = (
        "Steamworks initialization succeeded!",
        "Syncing cloud save files to the local save directory",
        "save_dir=user://steam/",
    )
    try:
        log = log_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"无法读取游戏启动日志: {log_path}") from exc
    if any(expected not in log for expected in expected_lines) or any(
        forbidden in log for forbidden in forbidden_lines
    ):
        raise RuntimeError(
            "无法确认测试存档隔离：Steam 与 UnifiedSavePath 必须关闭，"
            "且只允许 Agent Mod"
        )


def _wait_until_ready(
    process: subprocess.Popen[bytes],
    base_url: str,
    log_path: Path,
) -> None:
    """等待游戏实例的 HTTP 服务和可操作主菜单同时就绪。

    Args:
        process (subprocess.Popen[bytes]): 当前测试启动的游戏进程。
        base_url (str): Mod HTTP 服务根地址。
        log_path (Path): 游戏标准输出日志路径。

    Raises:
        RuntimeError: 游戏在主菜单就绪前退出。
        TimeoutError: 游戏未在限定时间内到达可操作主菜单。

    Returns:
        None: Mod 健康且主菜单可操作后返回。
    """
    deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS

    with httpx.Client(timeout=1.0) as client:
        while time.monotonic() < deadline:
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"STS2 提前退出，返回码 {return_code}\n"
                    f"日志末尾:\n{_read_log_tail(log_path)}"
                )

            try:
                health_response = client.get(f"{base_url}/health")
                state_response = client.get(f"{base_url}/state")
                if (
                    health_response.status_code == 200
                    and state_response.status_code == 200
                    and _main_menu_is_ready(state_response.json())
                ):
                    return
            except (httpx.HTTPError, ValueError):
                pass

            time.sleep(1.0)

    raise TimeoutError(
        f"STS2 在 {_STARTUP_TIMEOUT_SECONDS:.0f} 秒内未就绪\n"
        f"日志末尾:\n{_read_log_tail(log_path)}"
    )


def _main_menu_is_ready(payload: object) -> bool:
    """判断状态响应是否已经到达可操作的干净主菜单。

    Args:
        payload (object): 尚未校验的 ``/state`` JSON 响应。

    Returns:
        bool: 主菜单已暴露 ``open_character_select`` 时为 ``True``。
    """
    if not isinstance(payload, Mapping) or payload.get("ok") is not True:
        return False

    data = payload.get("data")
    if not isinstance(data, Mapping):
        return False

    actions = data.get("available_actions")
    return (
        data.get("screen") == "MAIN_MENU"
        and isinstance(actions, list)
        and "open_character_select" in actions
    )


def _port_is_open(port: int) -> bool:
    """检查本机回环地址上的端口是否已有监听者。

    Args:
        port (int): 待检查的 TCP 端口。

    Returns:
        bool: 端口可连接时为 ``True``，否则为 ``False``。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.2)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def _read_log_tail(path: Path, limit: int = 4000) -> str:
    """读取游戏日志末尾，供启动失败时诊断。

    Args:
        path (Path): 游戏日志路径。
        limit (int): 最多读取的末尾字节数。

    Returns:
        str: 使用替换策略解码的日志末尾文本。
    """
    if not path.is_file():
        return "<日志文件不存在>"

    with path.open("rb") as log_file:
        log_file.seek(0, os.SEEK_END)
        size = log_file.tell()
        log_file.seek(max(0, size - limit))
        return log_file.read().decode("utf-8", errors="replace")
