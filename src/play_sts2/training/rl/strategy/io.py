"""读写不含隐藏 checkpoint 审计的 Tree-GRPO group。"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..contracts import (
    ACTION_CONSTRAINT_MODE,
    BEHAVIOR_LOGPROBS_MODE,
    RL_GAME_VERSION,
    STRUCTURED_OUTPUT_BACKEND,
)
from .contracts import TreeRolloutGroup


def write_tree_rollout_group(
    group: TreeRolloutGroup,
    path: Path,
    *,
    environment: dict[str, str],
) -> Path:
    """写出可传给 A100 learner 的玩家可见 Tree group。

    Args:
        group (TreeRolloutGroup): 已通过 K=8 准入的兄弟 suffix 组。
        path (Path): 输出 JSON 文件。
        environment (dict[str, str]): 游戏、Mod、协议和 xgrammar 收据。

    Raises:
        ValueError: 环境收据不是固定 v0.111.0/xgrammar 契约。
        OSError: 无法创建目录或写入 JSON。

    Returns:
        Path: 已写入的 group 文件。
    """
    if (
        environment.get("game_version") != RL_GAME_VERSION
        or environment.get("structured_output_backend") != STRUCTURED_OUTPUT_BACKEND
        or not environment.get("mod_version")
        or not environment.get("protocol_version")
        or not environment.get("structured_output_version")
    ):
        raise ValueError("Tree group 环境收据无效")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    screen = {
        "map": "MAP",
        "card_selection": "CARD_SELECTION",
        "card_reward": "REWARD",
        "event": "EVENT",
        "rest": "REST",
        "shop": "SHOP",
    }[group.checkpoint_kind]
    payload: dict[str, Any] = {
        "format": "tree_grpo_group",
        "group_id": group.group_id,
        "checkpoint": {
            "kind": group.checkpoint_kind,
            "screen": screen,
            "option_ids": list(group.checkpoint_option_ids),
            "policy_text": group.checkpoint_policy_text,
        },
        "strategy_policy_version": group.strategy_policy_version,
        "battle_policy_version": group.battle_policy_version,
        "behavior_logprobs_mode": BEHAVIOR_LOGPROBS_MODE,
        "action_constraint_mode": ACTION_CONSTRAINT_MODE,
        "generation_profile": asdict(group.generation_profile),
        "max_macro_checkpoints": group.max_macro_checkpoints,
        "environment": dict(environment),
        "arms": [asdict(arm) for arm in group.arms],
        "returns": list(group.returns),
        "advantages": list(group.advantages),
        "plan_counts": [list(item) for item in group.plan_counts],
        "successor_counts": [list(item) for item in group.successor_counts],
        "successor_returns": [list(item) for item in group.successor_returns],
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
