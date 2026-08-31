"""定义同种子八局 GiGPO 的数据合同与分层相对优势。"""

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from ....inference import ChatMessage
from ....runtime import DecisionGenerationProfile


class GigpoGroupRejected(ValueError):
    """表示八条完整 episode 不能形成可信的 GiGPO group。"""


@dataclass(frozen=True, slots=True)
class GigpoStep:
    """保存一条整局 backbone 中的可训练战略动作。

    Args:
        index (int): 当前 episode 内战略动作序号。
        messages (tuple[ChatMessage, ...]): 实际发送的 stateless 消息。
        reply_text (str): 模型返回的规范动作文本。
        action (str): 已执行的规范动作。
        token_ids (tuple[int, ...]): assistant completion token。
        behavior_logprobs (tuple[float, ...]): 处理后的旧策略逐 token 概率。
        response_choices (tuple[str, ...]): xgrammar 使用的完整动作候选。
        finish_reason (str): 推理服务终止原因。
        anchor_id (str | None): 本地按完整原生审计相等分配的自然 anchor。
    """

    index: int
    messages: tuple[ChatMessage, ...]
    reply_text: str
    action: str
    token_ids: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    response_choices: tuple[str, ...]
    finish_reason: str
    anchor_id: str | None


@dataclass(frozen=True, slots=True)
class GigpoEpisode:
    """保存一条由冻结双 policy 完成的完整游戏。

    Args:
        arm_index (int): 同种子组内序号。
        worker_id (str): 本地游戏 worker 名称。
        seed (str): 本组共享游戏种子。
        character_id (str): 角色稳定 ID。
        ascension (int): 进阶等级。
        strategy_policy_version (str): 冻结战略 residual 名。
        battle_policy_version (str): 冻结战斗 residual 名。
        generation_profile (DecisionGenerationProfile): 战略采样参数。
        steps (tuple[GigpoStep, ...]): 本局全部战略模型动作。
        victory (bool): 是否整局胜利。
        final_floor (int): 终局楼层。
        final_hp (int): 终局当前生命，死亡为零。
        max_hp (int): 终局最大生命。
        bosses_cleared (int): 本局实际击败 Boss 数。
        battle_count (int): 本局完成的战斗数。
        elapsed_seconds (float): 游戏墙钟时间。
        model_error (bool): 是否由策略非法输出、截断或动作上限结束。
        failure_reason (str | None): 可读策略失败原因。
    """

    arm_index: int
    worker_id: str
    seed: str
    character_id: str
    ascension: int
    strategy_policy_version: str
    battle_policy_version: str
    generation_profile: DecisionGenerationProfile
    steps: tuple[GigpoStep, ...]
    victory: bool
    final_floor: int
    final_hp: int
    max_hp: int
    bosses_cleared: int
    battle_count: int
    elapsed_seconds: float
    model_error: bool = False
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class GigpoAnchorCensus:
    """保存不含隐藏状态内容的精确自然 anchor 覆盖统计。

    Args:
        groups (int): 至少包含两个步骤的精确 anchor 组数。
        repeated_steps (int): 落入这些组的战略步骤数。
        total_steps (int): 全部可训练战略步骤数。
        coverage (float): ``repeated_steps / total_steps``。
    """

    groups: int
    repeated_steps: int
    total_steps: int
    coverage: float


@dataclass(frozen=True, slots=True)
class GigpoGroup:
    """保存同种子八局及 episode/milestone 相对优势。

    Args:
        group_id (str): 当前 backbone group 名称。
        episodes (tuple[GigpoEpisode, ...]): 按 arm 排列的八条完整游戏。
        terminal_returns (tuple[float, ...]): 每局胜利终局分量。
        progress_returns (tuple[float, ...]): 每局 Boss、楼层与 HP 进度分量。
        episode_advantages (tuple[float, ...]): 终局组中心化优势。
        milestone_advantages (tuple[float, ...]): 进度组中心化优势。
        advantages (tuple[float, ...]): 广播给本局战略 token 的总优势。
        lambda_milestone (float): 进度课程权重。
        normalization (Literal["one", "std"]): 优势归一化方式。
        anchor_census (GigpoAnchorCensus): 精确自然 anchor 覆盖统计。
    """

    group_id: str
    episodes: tuple[GigpoEpisode, ...]
    terminal_returns: tuple[float, ...]
    progress_returns: tuple[float, ...]
    episode_advantages: tuple[float, ...]
    milestone_advantages: tuple[float, ...]
    advantages: tuple[float, ...]
    lambda_milestone: float
    normalization: Literal["one", "std"]
    anchor_census: GigpoAnchorCensus


def build_gigpo_group(
    *,
    group_id: str,
    episodes: Sequence[GigpoEpisode],
    anchor_audits: Sequence[Sequence[Mapping[str, Any]]],
    lambda_milestone: float,
    normalization: Literal["one", "std"],
) -> GigpoGroup:
    """核对八局冻结策略并计算 episode/milestone 相对优势。

    Args:
        group_id (str): 当前 backbone group 名称。
        episodes (Sequence[GigpoEpisode]): 八条完整游戏。
        anchor_audits (Sequence[Sequence[Mapping[str, Any]]]): 与战略步骤逐项对齐的
            本地隐藏审计；只用于完整对象相等比较，不写入返回对象。
        lambda_milestone (float): 进度课程权重。
        normalization (Literal["one", "std"]): 使用常数一或组标准差归一化。

    Raises:
        GigpoGroupRejected: K、种子、策略、终局、token 或审计不满足合同。

    Returns:
        GigpoGroup: 不含隐藏审计内容的可训练八局组。
    """
    ordered = tuple(sorted(episodes, key=lambda episode: episode.arm_index))
    if not group_id or len(ordered) != 8:
        raise GigpoGroupRejected("GiGPO 必须包含同种子 K=8 完整 episode")
    if tuple(episode.arm_index for episode in ordered) != tuple(range(8)):
        raise GigpoGroupRejected("GiGPO arm_index 必须连续且唯一")
    _require_single(ordered, "seed", "GiGPO group 混入不同 seed")
    _require_single(ordered, "character_id", "GiGPO group 混入不同角色")
    _require_single(ordered, "ascension", "GiGPO group 混入不同进阶")
    _require_single(
        ordered,
        "strategy_policy_version",
        "GiGPO group 混入不同 strategy policy",
    )
    _require_single(
        ordered,
        "battle_policy_version",
        "GiGPO group 混入不同 battle policy",
    )
    _require_single(
        ordered,
        "generation_profile",
        "GiGPO group 混入不同战略生成参数",
    )
    if not math.isfinite(lambda_milestone) or lambda_milestone < 0:
        raise GigpoGroupRejected("GiGPO milestone 权重无效")
    if normalization not in {"one", "std"}:
        raise GigpoGroupRejected("GiGPO normalization 无效")
    if len(anchor_audits) != 8:
        raise GigpoGroupRejected("GiGPO anchor 审计没有与八局对齐")
    for episode, audits in zip(ordered, anchor_audits, strict=True):
        _validate_episode(episode)
        if len(audits) != len(episode.steps):
            raise GigpoGroupRejected("GiGPO anchor 审计没有与战略步骤对齐")

    anchored, census = _assign_exact_anchors(ordered, anchor_audits)
    terminal = tuple(4.0 if episode.victory else 0.0 for episode in anchored)
    progress = tuple(_progress_return(episode) for episode in anchored)
    episode_advantages = _center(terminal, normalization)
    milestone_advantages = _center(progress, normalization)
    advantages = tuple(
        episode + lambda_milestone * milestone
        for episode, milestone in zip(
            episode_advantages,
            milestone_advantages,
            strict=True,
        )
    )
    if not any(value != 0.0 for value in advantages):
        raise GigpoGroupRejected("GiGPO group 的战略 advantage 全为零")
    return GigpoGroup(
        group_id=group_id,
        episodes=anchored,
        terminal_returns=terminal,
        progress_returns=progress,
        episode_advantages=episode_advantages,
        milestone_advantages=milestone_advantages,
        advantages=advantages,
        lambda_milestone=lambda_milestone,
        normalization=normalization,
        anchor_census=census,
    )


def _require_single(
    episodes: Sequence[GigpoEpisode],
    field: str,
    message: str,
) -> None:
    """要求全部 episode 的指定字段完全一致且非空。

    Args:
        episodes (Sequence[GigpoEpisode]): 待核对 episode。
        field (str): 数据类字段名。
        message (str): 失败错误文本。

    Raises:
        GigpoGroupRejected: 字段出现多个值或字符串值为空。

    Returns:
        None: 字段一致时返回。
    """
    values = {getattr(episode, field) for episode in episodes}
    value = next(iter(values)) if len(values) == 1 else None
    if len(values) != 1 or isinstance(value, str) and not value:
        raise GigpoGroupRejected(message)


def _validate_episode(episode: GigpoEpisode) -> None:
    """核对一条完整 episode 的终局与 token 事实。

    Args:
        episode (GigpoEpisode): 待核对完整游戏。

    Raises:
        GigpoGroupRejected: 终局字段、步骤或 token 元数据无效。

    Returns:
        None: episode 可用于训练时返回。
    """
    if (
        not episode.steps
        or episode.final_floor < 0
        or episode.max_hp <= 0
        or not 0 <= episode.final_hp <= episode.max_hp
        or episode.bosses_cleared < 0
        or episode.battle_count < 0
        or not math.isfinite(episode.elapsed_seconds)
        or episode.elapsed_seconds < 0
        or episode.model_error != (episode.failure_reason is not None)
    ):
        raise GigpoGroupRejected("GiGPO episode 的终局或计数字段无效")
    if not episode.victory and episode.final_hp != 0:
        raise GigpoGroupRejected("GiGPO 死亡 episode 的 final_hp 必须为零")
    if tuple(step.index for step in episode.steps) != tuple(range(len(episode.steps))):
        raise GigpoGroupRejected("GiGPO 战略 step index 必须连续")
    for step in episode.steps:
        if (
            len(step.messages) != 2
            or tuple(message.role for message in step.messages) != ("system", "user")
            or step.reply_text != step.action
            or not step.token_ids
            or len(step.token_ids) != len(step.behavior_logprobs)
            or any(
                not math.isfinite(value) or value > 0
                for value in step.behavior_logprobs
            )
            or not step.response_choices
            or step.action not in step.response_choices
            or not step.finish_reason
        ):
            raise GigpoGroupRejected("GiGPO 战略步骤缺少真实行为策略 token 事实")


def _progress_return(episode: GigpoEpisode) -> float:
    """计算一局 Boss、楼层与极小 HP 进度分量。

    Args:
        episode (GigpoEpisode): 已核对完整游戏。

    Returns:
        float: 文档冻结的 episode-level ``R_progress``。
    """
    hp = episode.final_hp / episode.max_hp * 0.001 if episode.victory else 0.0
    return episode.bosses_cleared + episode.final_floor * 0.01 + hp


def _center(
    values: Sequence[float],
    normalization: Literal["one", "std"],
) -> tuple[float, ...]:
    """按配置中心化一组回报。

    Args:
        values (Sequence[float]): 八条 episode 回报。
        normalization (Literal["one", "std"]): 分母为一或总体标准差。

    Raises:
        GigpoGroupRejected: ``std`` 模式下回报没有方差。

    Returns:
        tuple[float, ...]: 组均值为零的相对优势。
    """
    mean = statistics.fmean(values)
    denominator = 1.0
    if normalization == "std":
        denominator = statistics.pstdev(values)
        if denominator == 0:
            raise GigpoGroupRejected("GiGPO std normalization 遇到零方差")
    return tuple((value - mean) / denominator for value in values)


def _assign_exact_anchors(
    episodes: Sequence[GigpoEpisode],
    anchor_audits: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[tuple[GigpoEpisode, ...], GigpoAnchorCensus]:
    """用原生审计与完整 policy state 相等分配自然 anchor。

    Args:
        episodes (Sequence[GigpoEpisode]): 已按 arm 排列的八局。
        anchor_audits (Sequence[Sequence[Mapping[str, Any]]]): 每个战略步骤的原生审计。

    Returns:
        tuple[tuple[GigpoEpisode, ...], GigpoAnchorCensus]: 带可读 anchor ID 的
        episode 和覆盖统计。
    """
    locations: list[tuple[int, int, Mapping[str, Any], GigpoStep]] = []
    for episode_index, audits in enumerate(anchor_audits):
        for step_index, audit in enumerate(audits):
            locations.append(
                (
                    episode_index,
                    step_index,
                    audit,
                    episodes[episode_index].steps[step_index],
                )
            )
    assignments: dict[tuple[int, int], str] = {}
    consumed: set[tuple[int, int]] = set()
    group_index = 0
    for episode_index, step_index, audit, step in locations:
        location = (episode_index, step_index)
        if location in consumed:
            continue
        matches = [
            (other_episode, other_step)
            for other_episode, other_step, other_audit, other_policy_step in locations
            if (other_episode, other_step) not in consumed
            and other_audit == audit
            and other_policy_step.messages == step.messages
            and other_policy_step.response_choices == step.response_choices
        ]
        if len(matches) < 2:
            consumed.add(location)
            continue
        anchor_id = f"anchor-{group_index:03d}"
        group_index += 1
        for match in matches:
            assignments[match] = anchor_id
            consumed.add(match)
    anchored = []
    for episode_index, episode in enumerate(episodes):
        steps = tuple(
            replace(
                step,
                anchor_id=assignments.get((episode_index, step.index)),
            )
            for step in episode.steps
        )
        anchored.append(replace(episode, steps=steps))
    repeated_steps = len(assignments)
    total_steps = sum(len(episode.steps) for episode in episodes)
    census = GigpoAnchorCensus(
        groups=group_index,
        repeated_steps=repeated_steps,
        total_steps=total_steps,
        coverage=repeated_steps / total_steps if total_steps else 0.0,
    )
    return tuple(anchored), census
