"""读写不包含原生隐藏审计的 GiGPO 八局组。"""

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
from .contracts import GigpoGroup


def write_gigpo_group(
    group: GigpoGroup,
    path: Path,
    *,
    environment: Mapping[str, str],
) -> Path:
    """写出可传给 A100 learner 的 GiGPO 文件。

    Args:
        group (GigpoGroup): 已通过八局准入且只含 anchor ID 的 group。
        path (Path): 输出 JSON 路径。
        environment (Mapping[str, str]): 游戏、Mod、协议与 xgrammar 收据。

    Raises:
        ValueError: 环境不是固定 v0.111.0/xgrammar 契约。

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
        raise ValueError("GiGPO group 环境收据无效")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "gigpo_group",
        "strategy_policy_version": group.episodes[0].strategy_policy_version,
        "battle_policy_version": group.episodes[0].battle_policy_version,
        "behavior_logprobs_mode": BEHAVIOR_LOGPROBS_MODE,
        "action_constraint_mode": ACTION_CONSTRAINT_MODE,
        "environment": dict(environment),
        **asdict(group),
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
