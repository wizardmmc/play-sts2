"""提供从命令行启动人类轨迹录制的入口。"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from play_sts2.client import GameClient

from .recorder import HumanRunRecorder

_DEFAULT_BASE_URL = "http://127.0.0.1:8080"
_DEFAULT_OUTPUT_ROOT = Path("data/raw")
_DEFAULT_CHECK_INTERVAL = 0.1
_DEFAULT_RUNTIME_RECEIPT = Path(".runtime/SlayTheSpire2-v0.111.0/runtime-receipt.json")
_SOLVER_PRESETS: dict[str, dict[str, int | str]] = {
    "low": {
        "preset": "low",
        "settings_source": "recorder_argument",
        "short_time_budget_ms": 2_000,
        "deep_time_budget_ms": 20_000,
        "short_node_budget": 1_000,
        "deep_node_budget": 4_000,
        "memory_budget_bytes": 4_000_000_000,
    },
    "medium": {
        "preset": "medium",
        "settings_source": "recorder_argument",
        "short_time_budget_ms": 5_000,
        "deep_time_budget_ms": 60_000,
        "short_node_budget": 1_200,
        "deep_node_budget": 6_000,
        "memory_budget_bytes": 6_000_000_000,
    },
    "high": {
        "preset": "high",
        "settings_source": "recorder_argument",
        "short_time_budget_ms": 8_000,
        "deep_time_budget_ms": 120_000,
        "short_node_budget": 2_400,
        "deep_node_budget": 12_000,
        "memory_budget_bytes": 8_000_000_000,
    },
}


def main(argv: Sequence[str] | None = None) -> int:
    """连接已运行的 STS2，并把遇到的第一局保存为人类轨迹。

    Args:
        argv (Sequence[str] | None): 不含程序名的命令行参数；为 ``None`` 时
            使用当前进程参数。

    Returns:
        int: 命令成功时返回 ``0``。
    """
    args = _parser().parse_args(argv)
    recording_context = _recording_context(
        args.source,
        args.runtime_receipt,
        args.solver_preset,
    )
    with GameClient(args.base_url) as client:
        result = HumanRunRecorder(
            client,
            args.output_root,
            source=args.source,
            recording_context=recording_context,
            check_interval=args.check_interval,
        ).record()

    if result is not None:
        print(f"录制完成: {result.run_dir.resolve()}")
    return 0


def _parser() -> argparse.ArgumentParser:
    """创建人类轨迹录制命令的参数解析器。

    Returns:
        argparse.ArgumentParser: 包含连接、目录和轮询参数的解析器。
    """
    parser = argparse.ArgumentParser(
        description="连接已运行的 STS2，并录制遇到的第一局人类游玩轨迹。",
    )
    parser.add_argument(
        "--base-url",
        default=_DEFAULT_BASE_URL,
        help=f"Agent Mod 服务地址，默认为 {_DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
        help=f"原始数据根目录，默认为 {_DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--source",
        choices=("human", "human_combat_solver"),
        default="human",
        help="整局录制来源；人类操作战略且 Solver 接管战斗时选 human_combat_solver",
    )
    parser.add_argument(
        "--runtime-receipt",
        type=Path,
        default=_DEFAULT_RUNTIME_RECEIPT,
        help="人机协作录制使用的版本化教师运行时 receipt",
    )
    parser.add_argument(
        "--solver-preset",
        choices=tuple(_SOLVER_PRESETS),
        default="medium",
        help="CombatSolver 当前使用的性能预设，默认为 medium",
    )
    parser.add_argument(
        "--check-interval",
        "--poll-interval",
        dest="check_interval",
        type=float,
        default=_DEFAULT_CHECK_INTERVAL,
        help=f"事件队列检查间隔秒数，默认为 {_DEFAULT_CHECK_INTERVAL}",
    )
    return parser


def _recording_context(
    source: str,
    runtime_receipt: Path,
    solver_preset: str,
) -> dict[str, Any] | None:
    """为人机协作轨迹读取固定运行时版本和 Solver 预算。

    Args:
        source (str): 整局录制来源。
        runtime_receipt (Path): 教师运行时安装凭据。
        solver_preset (str): 当前选择的 Solver 性能档位。

    Raises:
        TypeError: receipt 不是对象或缺少游戏、Mod 版本信息。
        OSError: receipt 无法读取。
        json.JSONDecodeError: receipt 不是合法 JSON。

    Returns:
        dict[str, Any] | None: 人机协作环境信息；纯人工录制返回 ``None``。
    """
    if source != "human_combat_solver":
        return None
    receipt = json.loads(runtime_receipt.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise TypeError(f"教师运行时 receipt 顶层不是对象: {runtime_receipt}")
    game = receipt.get("game")
    mods = receipt.get("mods")
    if not isinstance(game, dict) or not isinstance(mods, dict):
        raise TypeError(f"教师运行时 receipt 缺少 game 或 mods: {runtime_receipt}")
    required_mods = ("STS2AIAgent", "STS2-RitsuLib", "CombatSolver")
    versioned_mods: dict[str, dict[str, str]] = {}
    for mod_name in required_mods:
        mod = mods.get(mod_name)
        version = mod.get("version") if isinstance(mod, dict) else None
        if not isinstance(version, str) or not version:
            raise ValueError(f"教师运行时 receipt 缺少 {mod_name} 版本")
        versioned_mods[mod_name] = {"version": version}
    if not isinstance(game.get("version"), str) or not isinstance(
        game.get("branch"), str
    ):
        raise TypeError(f"教师运行时 receipt 缺少游戏版本或分支: {runtime_receipt}")
    return {
        "game": {
            "version": game["version"],
            "branch": game["branch"],
        },
        "mods": versioned_mods,
        "solver": dict(_SOLVER_PRESETS[solver_preset]),
    }
