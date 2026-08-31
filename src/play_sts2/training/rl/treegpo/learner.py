"""把 terminal Tree group 投影为带 proposal 收据的 GRPO 训练组。"""

import json
import math
import statistics
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..contracts import ACTION_CONSTRAINT_MODE, BEHAVIOR_LOGPROBS_MODE, RL_GAME_VERSION
from ..learner import (
    GrpoTrainingError,
    GrpoTrainingGroup,
    XGrammarChoiceMasker,
    build_grpo_arm,
)


def load_terminal_tree_training_group(
    path: Path,
    tokenizer: Any,
    *,
    max_length: int,
    choice_masker: Any | None = None,
) -> GrpoTrainingGroup:
    """读取二至四条 terminal 分支并重算未标准化 branch advantage。

    Args:
        path (Path): Mac collector 写出的 terminal Tree JSON。
        tokenizer (Any): 与冻结 rollout policy 共用的 tokenizer。
        max_length (int): 单步 stateless 输入最大 token 数。
        choice_masker (Any | None): 测试可注入的 grammar 支持集构造器。

    Raises:
        GrpoTrainingError: 环境、入口、proposal、终局、return 或 token 无效。

    Returns:
        GrpoTrainingGroup: 可与 GiGPO 组合反向传播的 Tree group。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("format") != "terminal_tree_group"
    ):
        raise GrpoTrainingError("terminal Tree group 必须是已知 JSON 对象")
    if "audit" in payload:
        raise GrpoTrainingError("terminal Tree group 不得包含 checkpoint 隐藏审计")
    group_id = _required_text(payload, "group_id")
    strategy_policy = _required_text(payload, "strategy_policy_version")
    battle_policy = _required_text(payload, "battle_policy_version")
    checkpoint_policy_text = _required_text(payload, "checkpoint_policy_text")
    sampling_mode = payload.get("sampling_mode")
    option_ids = payload.get("checkpoint_option_ids")
    if sampling_mode not in {"stratified", "policy_iid"}:
        raise GrpoTrainingError("terminal Tree sampling mode 无效")
    if not isinstance(option_ids, list) or any(
        not isinstance(value, str) or not value for value in option_ids
    ):
        raise GrpoTrainingError("terminal Tree checkpoint options 无效")
    environment = payload.get("environment")
    if (
        not isinstance(environment, Mapping)
        or environment.get("game_version") != RL_GAME_VERSION
        or not isinstance(environment.get("mod_version"), str)
        or not isinstance(environment.get("protocol_version"), str)
        or environment.get("structured_output_backend") != "xgrammar"
        or not isinstance(environment.get("structured_output_version"), str)
    ):
        raise GrpoTrainingError(
            f"terminal Tree group 缺少固定 {RL_GAME_VERSION} 环境收据"
        )
    if payload.get("behavior_logprobs_mode") != BEHAVIOR_LOGPROBS_MODE:
        raise GrpoTrainingError("terminal Tree 缺少 processed behavior log-prob")
    if payload.get("action_constraint_mode") != ACTION_CONSTRAINT_MODE:
        raise GrpoTrainingError("terminal Tree 缺少 structured choice 约束")
    raw_branches = payload.get("branches")
    if not isinstance(raw_branches, list) or not 2 <= len(raw_branches) <= 4:
        raise GrpoTrainingError("terminal Tree 必须包含二至四条 branches")
    ordered = sorted(
        raw_branches, key=lambda item: _required_integer(item, "arm_index")
    )
    if [_required_integer(item, "arm_index") for item in ordered] != list(
        range(len(ordered))
    ):
        raise GrpoTrainingError("terminal Tree arm_index 必须连续且唯一")
    profiles = []
    for branch in ordered:
        if (
            branch.get("strategy_policy_version") != strategy_policy
            or branch.get("battle_policy_version") != battle_policy
            or branch.get("horizon_reason") not in {"victory", "died", "model_error"}
            or (branch.get("horizon_reason") == "model_error")
            != isinstance(branch.get("failure_reason"), str)
        ):
            raise GrpoTrainingError("terminal Tree branch 混入不同 policy 或非终局")
        profile = branch.get("generation_profile")
        if not isinstance(profile, Mapping):
            raise GrpoTrainingError("terminal Tree branch 缺少生成参数")
        profiles.append(dict(profile))
        steps = branch.get("steps")
        plan_step_count = _required_integer(branch, "plan_step_count")
        if (
            not isinstance(steps, list)
            or plan_step_count <= 0
            or plan_step_count > len(steps)
            or "\n".join(
                _required_text(step, "action") for step in steps[:plan_step_count]
            )
            != _required_text(branch, "plan_id")
        ):
            raise GrpoTrainingError("terminal Tree 入口宏计划无效")
        first = steps[0]
        messages = first.get("messages") if isinstance(first, Mapping) else None
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or not isinstance(messages[1], Mapping)
            or messages[1].get("content") != checkpoint_policy_text
        ):
            raise GrpoTrainingError("terminal Tree 玩家可见入口不一致")
    if any(profile != profiles[0] for profile in profiles[1:]):
        raise GrpoTrainingError("terminal Tree 混入不同生成参数")
    try:
        temperature = float(profiles[0]["temperature"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GrpoTrainingError("terminal Tree temperature 无效") from exc
    if not math.isfinite(temperature) or temperature <= 0:
        raise GrpoTrainingError("terminal Tree temperature 必须为正")

    rewards = tuple(_branch_return(branch) for branch in ordered)
    mean = statistics.fmean(rewards)
    advantages = tuple(value - mean for value in rewards)
    saved_returns = _number_array(payload, "returns", len(ordered))
    saved_advantages = _number_array(payload, "advantages", len(ordered))
    if any(
        not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)
        for actual, expected in zip(saved_returns, rewards, strict=True)
    ) or any(
        not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)
        for actual, expected in zip(saved_advantages, advantages, strict=True)
    ):
        raise GrpoTrainingError("terminal Tree 保存值与 learner 重算不一致")
    if not any(value != 0 for value in advantages):
        raise GrpoTrainingError("terminal Tree return 没有方差")
    if choice_masker is None:
        choice_masker = XGrammarChoiceMasker(
            tokenizer,
            expected_version=str(environment["structured_output_version"]),
        )
    arms = []
    for index, branch in enumerate(ordered):
        proposal = _positive_probability(branch, "proposal_probability")
        recompute = branch.get("recompute_root_probability") is True
        if sampling_mode == "stratified":
            expected_q = 1.0 / len(ordered)
            if (
                len(option_ids) != len(ordered)
                or not recompute
                or not math.isclose(
                    proposal,
                    expected_q,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                raise GrpoTrainingError("terminal Tree stratified proposal 无效")
        elif recompute:
            raise GrpoTrainingError("terminal Tree policy_iid 不应重复重要性校正")
        projected = dict(branch)
        projected["steps"] = branch["steps"][: int(branch["plan_step_count"])]
        arm = build_grpo_arm(
            projected,
            tokenizer,
            advantage=advantages[index],
            reward=rewards[index],
            max_length=max_length,
            temperature=temperature,
            choice_masker=choice_masker,
        )
        arms.append(
            replace(
                arm,
                proposal_probability=proposal,
                recompute_root_probability=recompute,
            )
        )
    return GrpoTrainingGroup(
        group_id=group_id,
        policy_version=strategy_policy,
        reward_scheme="terminal_strategy",
        rewards=rewards,
        advantages=advantages,
        arms=tuple(arms),
        battle_policy_version=battle_policy,
        generation_profile=dict(profiles[0]),
        environment=dict(environment),
    )


def _branch_return(branch: Mapping[str, Any]) -> float:
    """读取一个有限 terminal return。

    Args:
        branch (Mapping[str, Any]): 单条分支 JSON。

    Raises:
        GrpoTrainingError: return 对象或数值无效。

    Returns:
        float: 完整战略回报。
    """
    value = branch.get("continuation_return")
    if not isinstance(value, Mapping):
        raise GrpoTrainingError("terminal Tree branch 缺少 return")
    try:
        total = float(value["total"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GrpoTrainingError("terminal Tree branch return 无效") from exc
    if not math.isfinite(total):
        raise GrpoTrainingError("terminal Tree branch return 必须有限")
    return total


def _number_array(
    payload: Mapping[str, Any],
    field: str,
    expected: int,
) -> tuple[float, ...]:
    """读取固定长度的有限数值数组。

    Args:
        payload (Mapping[str, Any]): group 顶层对象。
        field (str): 数组字段名。
        expected (int): 精确长度。

    Raises:
        GrpoTrainingError: 字段、长度或数值无效。

    Returns:
        tuple[float, ...]: 校验后的数组。
    """
    raw = payload.get(field)
    if not isinstance(raw, list) or len(raw) != expected:
        raise GrpoTrainingError(f"terminal Tree {field} 无效")
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise GrpoTrainingError(f"terminal Tree {field} 无效") from exc
    if any(not math.isfinite(value) for value in values):
        raise GrpoTrainingError(f"terminal Tree {field} 必须有限")
    return values


def _positive_probability(value: Mapping[str, Any], field: str) -> float:
    """读取 ``(0, 1]`` 内的 proposal probability。

    Args:
        value (Mapping[str, Any]): 当前 branch。
        field (str): 概率字段名。

    Raises:
        GrpoTrainingError: 数值不在有效范围。

    Returns:
        float: 校验后的概率。
    """
    try:
        probability = float(value[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise GrpoTrainingError("terminal Tree proposal probability 无效") from exc
    if not math.isfinite(probability) or not 0 < probability <= 1:
        raise GrpoTrainingError("terminal Tree proposal probability 无效")
    return probability


def _required_text(value: object, field: str) -> str:
    """读取非空字符串字段。

    Args:
        value (object): 当前 JSON 对象。
        field (str): 字段名。

    Raises:
        GrpoTrainingError: 对象或字段无效。

    Returns:
        str: 校验后的文本。
    """
    if not isinstance(value, Mapping):
        raise GrpoTrainingError("terminal Tree JSON 项必须是对象")
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise GrpoTrainingError(f"terminal Tree 字段必须是非空字符串: {field}")
    return item


def _required_integer(value: object, field: str) -> int:
    """读取非负整数字段。

    Args:
        value (object): 当前 JSON 对象。
        field (str): 字段名。

    Raises:
        GrpoTrainingError: 对象或字段无效。

    Returns:
        int: 校验后的整数。
    """
    if not isinstance(value, Mapping):
        raise GrpoTrainingError("terminal Tree JSON 项必须是对象")
    item = value.get(field)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise GrpoTrainingError(f"terminal Tree 字段必须是非负整数: {field}")
    return item
