"""把整局 GiGPO 文件投影为共用 token-level GRPO 训练组。"""

import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..contracts import ACTION_CONSTRAINT_MODE, BEHAVIOR_LOGPROBS_MODE, RL_GAME_VERSION
from ..learner import (
    GrpoTrainingError,
    GrpoTrainingGroup,
    XGrammarChoiceMasker,
    build_grpo_arm,
)


def load_gigpo_training_group(
    path: Path,
    tokenizer: Any,
    *,
    max_length: int,
    choice_masker: Any | None = None,
) -> GrpoTrainingGroup:
    """读取八局 group 并只把每局总优势广播到该局战略步骤。

    Args:
        path (Path): Mac collector 写出的 GiGPO JSON。
        tokenizer (Any): 与冻结 rollout policy 共用的 tokenizer。
        max_length (int): 单步 stateless 输入最大 token 数。
        choice_masker (Any | None): 测试可注入的 grammar 支持集构造器。

    Raises:
        GrpoTrainingError: 环境、策略、episode、优势或 token 不满足合同。

    Returns:
        GrpoTrainingGroup: 可与 terminal Tree 组合反向传播的八臂组。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("format") != "gigpo_group":
        raise GrpoTrainingError("GiGPO group 必须是已知 JSON 对象")
    if "anchor_audits" in payload or "audit" in payload:
        raise GrpoTrainingError("GiGPO group 不得包含原生隐藏审计")
    group_id = _required_text(payload, "group_id")
    strategy_policy = _required_text(payload, "strategy_policy_version")
    battle_policy = _required_text(payload, "battle_policy_version")
    environment = payload.get("environment")
    if (
        not isinstance(environment, Mapping)
        or environment.get("game_version") != RL_GAME_VERSION
        or not isinstance(environment.get("mod_version"), str)
        or not isinstance(environment.get("protocol_version"), str)
        or environment.get("structured_output_backend") != "xgrammar"
        or not isinstance(environment.get("structured_output_version"), str)
    ):
        raise GrpoTrainingError(f"GiGPO group 缺少固定 {RL_GAME_VERSION} 环境收据")
    if payload.get("behavior_logprobs_mode") != BEHAVIOR_LOGPROBS_MODE:
        raise GrpoTrainingError("GiGPO group 缺少 processed behavior log-prob")
    if payload.get("action_constraint_mode") != ACTION_CONSTRAINT_MODE:
        raise GrpoTrainingError("GiGPO group 缺少 structured choice 约束")
    if payload.get("normalization") not in {"one", "std"}:
        raise GrpoTrainingError("GiGPO normalization 无效")
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list) or len(raw_episodes) != 8:
        raise GrpoTrainingError("GiGPO group 必须恰好包含八条 episode")
    ordered = sorted(
        raw_episodes, key=lambda item: _required_integer(item, "arm_index")
    )
    if [_required_integer(item, "arm_index") for item in ordered] != list(range(8)):
        raise GrpoTrainingError("GiGPO arm_index 必须连续且唯一")
    profiles = []
    seeds = set()
    for episode in ordered:
        if (
            episode.get("strategy_policy_version") != strategy_policy
            or episode.get("battle_policy_version") != battle_policy
        ):
            raise GrpoTrainingError("GiGPO episode 混入不同双 policy")
        seeds.add(_required_text(episode, "seed"))
        profile = episode.get("generation_profile")
        if not isinstance(profile, Mapping):
            raise GrpoTrainingError("GiGPO episode 缺少生成参数")
        profiles.append(dict(profile))
        steps = episode.get("steps")
        if not isinstance(steps, list) or not steps:
            raise GrpoTrainingError("GiGPO episode 没有战略步骤")
        if [_required_integer(step, "index") for step in steps] != list(
            range(len(steps))
        ):
            raise GrpoTrainingError("GiGPO 战略 step index 必须连续")
    if len(seeds) != 1 or any(profile != profiles[0] for profile in profiles[1:]):
        raise GrpoTrainingError("GiGPO episode 混入不同 seed 或生成参数")
    try:
        temperature = float(profiles[0]["temperature"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GrpoTrainingError("GiGPO temperature 无效") from exc
    if not math.isfinite(temperature) or temperature <= 0:
        raise GrpoTrainingError("GiGPO temperature 必须为正")
    saved_advantages = _number_array(payload, "advantages", expected=8)
    saved_terminal = _number_array(payload, "terminal_returns", expected=8)
    saved_progress = _number_array(payload, "progress_returns", expected=8)
    saved_episode_advantages = _number_array(
        payload,
        "episode_advantages",
        expected=8,
    )
    saved_milestone_advantages = _number_array(
        payload,
        "milestone_advantages",
        expected=8,
    )
    try:
        lambda_milestone = float(payload["lambda_milestone"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GrpoTrainingError("GiGPO lambda_milestone 无效") from exc
    normalization = str(payload["normalization"])
    terminal = tuple(
        4.0 if episode.get("victory") is True else 0.0 for episode in ordered
    )
    progress = tuple(_episode_progress_return(episode) for episode in ordered)
    episode_advantages = _center_returns(terminal, normalization)
    milestone_advantages = _center_returns(progress, normalization)
    advantages = tuple(
        episode + lambda_milestone * milestone
        for episode, milestone in zip(
            episode_advantages,
            milestone_advantages,
            strict=True,
        )
    )
    arrays = (
        ("terminal return", saved_terminal, terminal),
        ("progress return", saved_progress, progress),
        ("episode advantage", saved_episode_advantages, episode_advantages),
        (
            "milestone advantage",
            saved_milestone_advantages,
            milestone_advantages,
        ),
        ("总 advantage", saved_advantages, advantages),
    )
    for name, saved, recomputed in arrays:
        if any(
            not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)
            for actual, expected in zip(saved, recomputed, strict=True)
        ):
            raise GrpoTrainingError(f"GiGPO 保存的{name}与 episode 事实不一致")
    if choice_masker is None:
        choice_masker = XGrammarChoiceMasker(
            tokenizer,
            expected_version=str(environment["structured_output_version"]),
        )
    rewards = tuple(
        terminal_value + progress_value
        for terminal_value, progress_value in zip(terminal, progress, strict=True)
    )
    arms = tuple(
        build_grpo_arm(
            episode,
            tokenizer,
            advantage=advantages[index],
            reward=rewards[index],
            max_length=max_length,
            temperature=temperature,
            choice_masker=choice_masker,
        )
        for index, episode in enumerate(ordered)
    )
    return GrpoTrainingGroup(
        group_id=group_id,
        policy_version=strategy_policy,
        reward_scheme="episode_terminal_progress",
        rewards=rewards,
        advantages=advantages,
        arms=arms,
        battle_policy_version=battle_policy,
        generation_profile=dict(profiles[0]),
        environment=dict(environment),
    )


def _episode_progress_return(episode: Mapping[str, Any]) -> float:
    """从落盘终局事实重算单局 milestone return。

    Args:
        episode (Mapping[str, Any]): 单条完整 GiGPO episode。

    Raises:
        GrpoTrainingError: Boss、楼层或生命字段无效。

    Returns:
        float: 文档冻结的 Boss、楼层和胜局 HP 进度分量。
    """
    bosses = _required_integer(episode, "bosses_cleared")
    floor = _required_integer(episode, "final_floor")
    max_hp = _required_integer(episode, "max_hp")
    final_hp = _required_integer(episode, "final_hp")
    if max_hp <= 0 or final_hp > max_hp:
        raise GrpoTrainingError("GiGPO episode 生命字段无效")
    hp = final_hp / max_hp * 0.001 if episode.get("victory") is True else 0.0
    return bosses + floor * 0.01 + hp


def _center_returns(values: tuple[float, ...], normalization: str) -> tuple[float, ...]:
    """按 group 声明重新中心化回报。

    Args:
        values (tuple[float, ...]): 八条 episode 回报。
        normalization (str): ``one`` 或 ``std``。

    Raises:
        GrpoTrainingError: std 模式的回报没有方差。

    Returns:
        tuple[float, ...]: learner 独立重算的相对优势。
    """
    denominator = 1.0
    if normalization == "std":
        denominator = statistics.pstdev(values)
        if denominator == 0:
            raise GrpoTrainingError("GiGPO std normalization 遇到零方差")
    mean = statistics.fmean(values)
    return tuple((value - mean) / denominator for value in values)


def _number_array(
    payload: Mapping[str, Any],
    field: str,
    *,
    expected: int,
) -> tuple[float, ...]:
    """读取固定长度的有限浮点数组。

    Args:
        payload (Mapping[str, Any]): GiGPO 顶层对象。
        field (str): 数组字段名。
        expected (int): 精确长度。

    Raises:
        GrpoTrainingError: 字段缺失、长度或数值无效。

    Returns:
        tuple[float, ...]: 校验后的数值。
    """
    raw = payload.get(field)
    if not isinstance(raw, list) or len(raw) != expected:
        raise GrpoTrainingError(f"GiGPO {field} 无效")
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise GrpoTrainingError(f"GiGPO {field} 无效") from exc
    if any(not math.isfinite(value) for value in values):
        raise GrpoTrainingError(f"GiGPO {field} 必须有限")
    return values


def _required_text(value: Mapping[object, object], field: str) -> str:
    """读取非空字符串字段。

    Args:
        value (Mapping[object, object]): 当前 JSON 对象。
        field (str): 字段名。

    Raises:
        GrpoTrainingError: 字段不是非空字符串。

    Returns:
        str: 校验后的文本。
    """
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise GrpoTrainingError(f"GiGPO 字段必须是非空字符串: {field}")
    return item


def _required_integer(value: object, field: str) -> int:
    """读取非负整数字段。

    Args:
        value (object): 当前 JSON 对象。
        field (str): 字段名。

    Raises:
        GrpoTrainingError: 对象或字段不是非负整数。

    Returns:
        int: 校验后的整数。
    """
    if not isinstance(value, Mapping):
        raise GrpoTrainingError("GiGPO JSON 项必须是对象")
    item = value.get(field)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise GrpoTrainingError(f"GiGPO 字段必须是非负整数: {field}")
    return item
