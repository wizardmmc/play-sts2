"""写出不包含 checkpoint 隐藏审计的 terminal Tree group。"""

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from ..contracts import (
    ACTION_CONSTRAINT_MODE,
    BEHAVIOR_LOGPROBS_MODE,
    RL_GAME_VERSION,
    STRUCTURED_OUTPUT_BACKEND,
)
from .contracts import TerminalTreeGroup


def write_terminal_tree_group(
    group: TerminalTreeGroup,
    path: Path,
    *,
    environment: Mapping[str, str],
) -> Path:
    """持久化一个分层 terminal Tree group。

    Args:
        group (TerminalTreeGroup): 已通过 proposal 与终局准入的分支组。
        path (Path): 输出 JSON 路径。
        environment (Mapping[str, str]): 游戏、Mod、协议与 xgrammar 收据。

    Raises:
        ValueError: 环境收据不是固定 v0.111.0/xgrammar 契约。

    Returns:
        Path: 实际输出路径。
    """
    if (
        environment.get("game_version") != RL_GAME_VERSION
        or environment.get("structured_output_backend") != STRUCTURED_OUTPUT_BACKEND
        or not environment.get("mod_version")
        or not environment.get("protocol_version")
        or not environment.get("structured_output_version")
    ):
        raise ValueError("terminal Tree group 环境收据无效")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "terminal_tree_group",
        "strategy_policy_version": group.branches[0].strategy_policy_version,
        "battle_policy_version": group.branches[0].battle_policy_version,
        "behavior_logprobs_mode": BEHAVIOR_LOGPROBS_MODE,
        "action_constraint_mode": ACTION_CONSTRAINT_MODE,
        "environment": dict(environment),
        "battles": [
            {"arm_index": branch.arm_index, **asdict(candidate)}
            for branch in group.branches
            for candidate in branch.battle_candidates
        ],
        **asdict(group),
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
