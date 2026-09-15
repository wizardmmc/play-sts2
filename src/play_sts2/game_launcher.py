"""启动使用隔离存档和独立进程组的 STS2 实例。"""

import json
import os
import shutil
import signal
import socket
import subprocess
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx

DEFAULT_APP = (
    Path(__file__).resolve().parents[2]
    / ".runtime/SlayTheSpire2-v0.107.1/SlayTheSpire2.app"
)
DEFAULT_PORT = 8080
DEFAULT_PROFILE = Path(__file__).resolve().parents[2] / "e2e/fixtures/profile"
_STARTUP_TIMEOUT_SECONDS = 120.0
_DATA_ROOT = Path("Library/Application Support/SlayTheSpire2")
_AGENT_MODS = frozenset({"STS2AIAgent"})
_COMBAT_SOLVER_MODS = frozenset({"STS2AIAgent", "STS2-RitsuLib", "CombatSolver"})
_ALLOWED_MOD_SETS = frozenset({_AGENT_MODS, _COMBAT_SOLVER_MODS})


@dataclass(frozen=True, slots=True)
class RunningGame:
    """记录已经就绪的隔离游戏实例。

    Args:
        base_url (str): Agent Mod HTTP 服务根地址。
        home (Path): 游戏独占的隔离 HOME 目录。
        log_path (Path): 当前游戏进程的标准输出日志。
    """

    base_url: str
    home: Path
    log_path: Path
    _process: subprocess.Popen[bytes] = field(repr=False)

    def wait(self) -> int:
        """阻塞等待游戏进程自行退出。

        Returns:
            int: 游戏进程的退出状态码。
        """
        return self._process.wait()

    def stop(self) -> None:
        """停止本实例拥有的游戏进程，供预算截止与上下文清理共用。

        Returns:
            None: 已退出的实例不再发送信号，可安全重复调用。
        """
        _stop_game(self._process)


@contextmanager
def launch_game(
    executable: Path,
    *,
    port: int,
    home: Path,
    profile: Path,
    mode: str,
    enable_debug_actions: bool = False,
    run_save: Path | None = None,
    startup_deadline: float | None = None,
) -> Iterator[RunningGame]:
    """启动并在离开上下文时清理一个隔离游戏实例。

    Args:
        executable (Path): STS2 的 macOS 可执行文件。
        port (int): Agent Mod HTTP 服务监听端口。
        home (Path): 当前实例独占且必须为空的 HOME 目录。
        profile (Path): 只包含安全设置、偏好与进度的存档模板目录。
        mode (str): ``headless`` 或 ``headed`` 启动模式。
        enable_debug_actions (bool): 是否开放场景重置使用的调试动作。
        run_save (Path | None): 可选的原生 ``current_run.save`` checkpoint。
        startup_deadline (float | None): 可选的 ``time.monotonic()`` 绝对启动
            截止时间，用于共享外部实验预算；不延长默认启动等待上限。

    Raises:
        FileNotFoundError: 游戏文件或隔离存档模板不存在。
        RuntimeError: 端口占用、隔离核对失败或游戏提前退出。
        TimeoutError: 游戏未在限定时间内进入可操作主菜单。
        ValueError: 启动模式或存档模板不满足隔离要求。

    Yields:
        RunningGame: 已进入可操作主菜单的游戏实例。
    """
    if startup_deadline is not None and time.monotonic() >= startup_deadline:
        raise TimeoutError("游戏启动前共享预算已到")
    if not executable.is_file():
        raise FileNotFoundError(f"找不到 STS2 可执行文件: {executable}")
    if not (executable.parent / "steam_appid.txt").is_file():
        raise FileNotFoundError("找不到游戏启动所需文件: steam_appid.txt")
    if _port_is_open(port):
        raise RuntimeError(f"Agent Mod 端口已被占用: {port}")

    enabled_mods = stage_profile(profile, home, run_save=run_save)
    base_url = f"http://127.0.0.1:{port}"
    log_path = home / f"{mode}.log"
    environment = os.environ.copy()
    environment["HOME"] = str(home)
    environment["STS2_API_PORT"] = str(port)
    environment.pop("STS2_ENABLE_DEBUG_ACTIONS", None)
    if enable_debug_actions:
        environment["STS2_ENABLE_DEBUG_ACTIONS"] = "1"

    with log_path.open("ab", buffering=0) as log_file:
        process = subprocess.Popen(
            game_command(executable, mode),
            cwd=executable.parent,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            if startup_deadline is None:
                _wait_until_ready(process, base_url, log_path)
            else:
                _wait_until_ready(
                    process, base_url, log_path, startup_deadline=startup_deadline
                )
            _verify_isolated_mod_configuration(log_path, enabled_mods)
            if startup_deadline is not None and time.monotonic() >= startup_deadline:
                raise TimeoutError("游戏启动完成时共享预算已到")
            yield RunningGame(base_url, home, log_path, process)
        finally:
            _stop_game(process)


def game_command(executable: Path, mode: str) -> list[str]:
    """生成禁用 Steam 且符合指定显示模式的游戏参数。

    Args:
        executable (Path): STS2 可执行文件路径。
        mode (str): ``headless`` 或 ``headed`` 启动模式。

    Raises:
        ValueError: ``mode`` 不是受支持的启动模式。

    Returns:
        list[str]: 可直接交给子进程的完整参数。
    """
    command = [str(executable), "--force-steam=off"]
    if mode == "headless":
        return [*command, "--headless"]
    if mode == "headed":
        return command
    raise ValueError(f"未知的 STS2 启动模式: {mode}")


def stage_profile(
    profile: Path,
    home: Path,
    *,
    run_save: Path | None = None,
) -> frozenset[str]:
    """把受控存档模板写入隔离 HOME 的非 Steam 路径。

    Args:
        profile (Path): 包含设置、偏好和进度文件的模板目录。
        home (Path): 当前游戏独占且必须为空的 HOME 目录。
        run_save (Path | None): 可选的原生整局存档 checkpoint。

    Raises:
        FileNotFoundError: 存档模板缺少必需文件。
        ValueError: HOME 非空或 Mod 配置不满足隔离要求。
        OSError: 无法读取或复制存档文件。

    Returns:
        frozenset[str]: 三个允许的存档文件复制完成后，返回预期启用的
            Mod ID。
    """
    settings = profile / "settings.save"
    preferences = profile / "prefs.save"
    progress = profile / "progress.save"
    for path in (settings, preferences, progress):
        if not path.is_file():
            raise FileNotFoundError(f"隔离存档模板缺少文件: {path}")
    if run_save is not None and not run_save.is_file():
        raise FileNotFoundError(f"找不到整局 checkpoint: {run_save}")
    if home.exists() and any(home.iterdir()):
        raise ValueError(f"隔离 HOME 必须为空: {home}")

    enabled_mods = _validate_profile(settings)
    data_root = home / _DATA_ROOT
    targets = [
        (settings, data_root / "default/1/settings.save"),
        (progress, data_root / "default/1/modded/profile1/saves/progress.save"),
        (preferences, data_root / "default/1/modded/profile1/saves/prefs.save"),
    ]
    if run_save is not None:
        targets.append(
            (
                run_save,
                data_root / "default/1/modded/profile1/saves/current_run.save",
            )
        )
    for source, target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return enabled_mods


def _validate_profile(settings: Path) -> frozenset[str]:
    """确认模板关闭共享存档且使用已知的精确 Mod 白名单。

    Args:
        settings (Path): 待检查的 ``settings.save`` 文件。

    Raises:
        ValueError: 设置内容无法证明存档和 Mod 已隔离。

    Returns:
        frozenset[str]: 模板明确启用的 Mod ID。
    """
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
        raise ValueError(f"隔离存档模板的 Mod 配置无效: {settings}") from exc

    enabled_mods = frozenset(
        mod_id for mod_id, enabled in mod_states.items() if enabled
    )
    if (
        mod_settings.get("mods_enabled") is not True
        or enabled_mods not in _ALLOWED_MOD_SETS
        or mod_states.get("UnifiedSavePath") is not False
    ):
        raise ValueError(
            "隔离存档模板必须仅启用 Agent，或精确启用教师 Mod，并关闭 UnifiedSavePath"
        )
    return enabled_mods


def _wait_until_ready(
    process: subprocess.Popen[bytes],
    base_url: str,
    log_path: Path,
    *,
    startup_deadline: float | None = None,
) -> None:
    """等待 Mod HTTP 服务与可操作主菜单同时就绪。

    Args:
        process (subprocess.Popen[bytes]): 当前游戏进程。
        base_url (str): Agent Mod HTTP 服务根地址。
        log_path (Path): 游戏标准输出日志路径。
        startup_deadline (float | None): 外部共享预算的单调时钟截止时间。

    Raises:
        RuntimeError: 游戏在主菜单就绪前退出。
        TimeoutError: 游戏未在限定时间内到达可操作主菜单。

    Returns:
        None: 主菜单已经可以开始或继续游戏时返回。
    """
    deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    if startup_deadline is not None:
        deadline = min(deadline, startup_deadline)
    with httpx.Client(timeout=1.0) as client:
        while time.monotonic() < deadline:
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"STS2 提前退出，返回码 {return_code}\n"
                    f"日志末尾:\n{_read_log_tail(log_path)}"
                )
            try:
                health = client.get(f"{base_url}/health")
                state = client.get(f"{base_url}/state")
                if (
                    health.status_code == 200
                    and state.status_code == 200
                    and _main_menu_is_ready(state.json())
                ):
                    if startup_deadline is not None and time.monotonic() >= deadline:
                        break
                    return
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(1.0)

    limit = (
        f"{_STARTUP_TIMEOUT_SECONDS:.0f} 秒"
        if startup_deadline is None
        else "默认启动上限与共享预算的较早截止时间"
    )
    raise TimeoutError(
        f"STS2 在 {limit} 内未就绪\n日志末尾:\n{_read_log_tail(log_path)}"
    )


def _main_menu_is_ready(payload: object) -> bool:
    """判断状态响应是否已经到达可操作主菜单。

    Args:
        payload (object): 尚未校验的 ``/state`` JSON 响应。

    Returns:
        bool: 主菜单已经开放开局或续局动作时为 ``True``。
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
        and bool({"open_character_select", "continue_run"}.intersection(actions))
    )


def _verify_isolated_mod_configuration(
    log_path: Path,
    expected_mods: frozenset[str] = _AGENT_MODS,
) -> None:
    """从游戏日志核对 Steam、共享存档和 Mod 白名单。

    Args:
        log_path (Path): 当前游戏进程的标准输出日志路径。
        expected_mods (frozenset[str]): profile 中预期启用的精确 Mod ID。

    Raises:
        RuntimeError: 日志缺少任一隔离证据。

    Returns:
        None: 四项隔离证据全部成立时返回。
    """
    expected = [
        "[INFO] Steam initialization skipped (editor mode). Use --force-steam to enable.",
        "[INFO] Skipping loading mod UnifiedSavePath, it is set to disabled in settings",
        f"[INFO]  --- RUNNING MODDED! --- Loaded {len(expected_mods)} mods (",
    ]
    expected.extend(f"({mod_id})." for mod_id in sorted(expected_mods))
    forbidden = (
        "Steamworks initialization succeeded!",
        "Syncing cloud save files to the local save directory",
        "save_dir=user://steam/",
    )
    try:
        log = log_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"无法读取游戏启动日志: {log_path}") from exc
    if (
        expected_mods not in _ALLOWED_MOD_SETS
        or any(line not in log for line in expected)
        or any(line in log for line in forbidden)
    ):
        raise RuntimeError(
            "无法确认游戏隔离：Steam 与 UnifiedSavePath 必须关闭，且加载的 Mod "
            "必须与 profile 白名单一致"
        )


def _stop_game(process: subprocess.Popen[bytes]) -> None:
    """终止游戏的独立进程组并回收领导进程。

    Args:
        process (subprocess.Popen[bytes]): 使用独立会话启动的游戏进程。

    Raises:
        subprocess.TimeoutExpired: 进程组终止后领导进程仍无法回收。

    Returns:
        None: 游戏进程组退出并被回收后返回。
    """
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def _port_is_open(port: int) -> bool:
    """检查回环地址上的端口是否已有监听者。

    Args:
        port (int): 待检查的 TCP 端口。

    Returns:
        bool: 端口可连接时为 ``True``。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.2)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def _read_log_tail(path: Path, limit: int = 4000) -> str:
    """读取游戏日志末尾供启动失败诊断。

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
