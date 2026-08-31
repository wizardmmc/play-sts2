"""把双 policy 完整游戏投影为 GiGPO episode 与候选入口。"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ....client import Health
from ....harness import HarnessLayer, format_action
from ....runtime import DecisionGenerationProfile, RunOutcome, RunResult
from .contracts import GigpoEpisode, GigpoStep
from .scenario import BackboneBattleCandidate


@dataclass(frozen=True, slots=True)
class BackboneCheckpointCandidate:
    """保存完整游戏中已经发布的一个原生宏 checkpoint。

    Args:
        index (int): 当前 episode 内候选序号。
        kind (str): map、card_reward、event、rest 或 shop。
        floor (int): 捕获楼层。
        option_ids (tuple[str, ...]): 玩家可见宏选项。
        policy_text (str): 入口玩家可见战略观测。
        path (Path | None): 被墙钟策略选中并发布时的原生 checkpoint 目录；仅记录
            候选元数据时为空。
    """

    index: int
    kind: str
    floor: int
    option_ids: tuple[str, ...]
    policy_text: str
    path: Path | None


@dataclass(frozen=True, slots=True)
class BackboneEpisodeDraft:
    """保存一条完整 episode 及仅留在 Mac 的候选和审计。

    Args:
        episode (GigpoEpisode): 不含隐藏状态的训练 episode。
        anchor_audits (tuple[Mapping[str, Any], ...]): 与战略步骤对齐的原生审计。
        checkpoints (tuple[BackboneCheckpointCandidate, ...]): 可供 Tree 选择的入口。
        battles (tuple[BackboneBattleCandidate, ...]): 可供战斗刷新选择的入口。
        health (Health): 当前游戏、Mod 与协议收据。
    """

    episode: GigpoEpisode
    anchor_audits: tuple[Mapping[str, Any], ...]
    checkpoints: tuple[BackboneCheckpointCandidate, ...]
    battles: tuple[BackboneBattleCandidate, ...]
    health: Health


def build_gigpo_episode(
    result: RunResult,
    *,
    arm_index: int,
    worker_id: str,
    seed: str,
    character_id: str,
    ascension: int,
    strategy_policy_version: str,
    battle_policy_version: str,
    anchor_audits: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
) -> GigpoEpisode:
    """把 Runtime 完整结果转换为只训练战略 token 的 episode。

    Args:
        result (RunResult): 双 provider 完成的整局结果。
        arm_index (int): 同种子组内序号。
        worker_id (str): 本地 worker 名称。
        seed (str): 游戏种子。
        character_id (str): 角色稳定 ID。
        ascension (int): 进阶等级。
        strategy_policy_version (str): 期望战略 residual 名。
        battle_policy_version (str): 期望战斗 residual 名。
        anchor_audits (Sequence[Mapping[str, Any]]): 与战略决策逐项对齐的审计。
        elapsed_seconds (float): 本局游戏墙钟。

    Raises:
        ValueError: policy、步骤、token、终局或审计不满足冻结合同。

    Returns:
        GigpoEpisode: 可进入同种子八局组的训练 episode。
    """
    strategic = tuple(
        decision.step
        for decision in result.decisions
        if decision.layer is HarnessLayer.STRATEGIC
    )
    battles = tuple(
        decision.step
        for decision in result.decisions
        if decision.layer is HarnessLayer.BATTLE
    )
    if len(strategic) != len(anchor_audits):
        raise ValueError(
            "GiGPO 战略决策与精确审计没有逐项对齐: "
            f"decisions={len(strategic)}, audits={len(anchor_audits)}"
        )
    if any(
        step.retry_errors
        or len(step.replies) != 1
        or step.reply.model != battle_policy_version
        for step in battles
    ):
        raise ValueError("GiGPO 整局混入不同 battle policy 或模型重试")
    projected = []
    profiles = set()
    for index, step in enumerate(strategic):
        if step.retry_errors or len(step.replies) != 1:
            raise ValueError("GiGPO 战略 rollout 不允许动作内模型重试")
        reply = step.reply
        action = format_action(step.action)
        if (
            reply.model != strategy_policy_version
            or reply.text != action
            or not reply.token_ids
            or len(reply.token_ids) != len(reply.behavior_logprobs)
            or any(
                not math.isfinite(value) or value > 0
                for value in reply.behavior_logprobs
            )
            or not step.response_choices
            or action not in step.response_choices
            or not reply.finish_reason
            or step.generation_profile is None
        ):
            raise ValueError("GiGPO 战略步骤缺少真实行为策略 token 事实")
        profiles.add(step.generation_profile)
        projected.append(
            GigpoStep(
                index=index,
                messages=step.messages,
                reply_text=reply.text,
                action=action,
                token_ids=reply.token_ids,
                behavior_logprobs=reply.behavior_logprobs,
                response_choices=step.response_choices,
                finish_reason=reply.finish_reason,
                anchor_id=None,
            )
        )
    if len(profiles) != 1:
        raise ValueError("GiGPO episode 缺少统一战略生成参数")
    final_run = result.final_state.get("run")
    if not isinstance(final_run, Mapping):
        raise TypeError("GiGPO episode 终局缺少 run 状态")
    final_floor = _non_negative_integer(final_run, "floor")
    max_hp = _positive_integer(final_run, "max_hp")
    victory = result.outcome is RunOutcome.VICTORY
    final_hp = _non_negative_integer(final_run, "current_hp") if victory else 0
    if final_hp > max_hp or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("GiGPO episode 终局生命或墙钟无效")
    profile = next(iter(profiles))
    if not isinstance(profile, DecisionGenerationProfile):
        raise TypeError("GiGPO episode 生成参数类型无效")
    return GigpoEpisode(
        arm_index=arm_index,
        worker_id=worker_id,
        seed=seed,
        character_id=character_id,
        ascension=ascension,
        strategy_policy_version=strategy_policy_version,
        battle_policy_version=battle_policy_version,
        generation_profile=profile,
        steps=tuple(projected),
        victory=victory,
        final_floor=final_floor,
        final_hp=final_hp,
        max_hp=max_hp,
        bosses_cleared=result.bosses_cleared,
        battle_count=result.battle_count,
        elapsed_seconds=elapsed_seconds,
        model_error=result.model_error,
        failure_reason=result.failure_reason,
    )


def _non_negative_integer(value: Mapping[str, Any], field: str) -> int:
    """读取非负整数字段。

    Args:
        value (Mapping[str, Any]): 当前映射。
        field (str): 字段名。

    Raises:
        ValueError: 字段不是非负整数。

    Returns:
        int: 校验后的整数。
    """
    item = value.get(field)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise ValueError(f"GiGPO 终局字段必须是非负整数: {field}")
    return item


def _positive_integer(value: Mapping[str, Any], field: str) -> int:
    """读取正整数字段。

    Args:
        value (Mapping[str, Any]): 当前映射。
        field (str): 字段名。

    Raises:
        ValueError: 字段不是正整数。

    Returns:
        int: 校验后的整数。
    """
    item = _non_negative_integer(value, field)
    if item <= 0:
        raise ValueError(f"GiGPO 终局字段必须是正整数: {field}")
    return item
