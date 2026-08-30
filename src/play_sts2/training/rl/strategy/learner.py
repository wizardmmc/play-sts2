"""把 Tree-GRPO group 投影到共用 token-level learner。"""

import json
import math
import statistics
from collections import Counter
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


def load_tree_training_group(
    path: Path,
    tokenizer: Any,
    *,
    max_length: int,
    choice_masker: Any | None = None,
) -> GrpoTrainingGroup:
    """读取 Tree group，重算 branch advantage 并构造战略 token 序列。

    Args:
        path (Path): 本地 collector 写出的 Tree group JSON。
        tokenizer (Any): 与冻结 rollout policy 相同的 tokenizer。
        max_length (int): 单步 prompt 与 completion 最大 token 数。
        choice_masker (Any | None): 测试可注入的 xgrammar 支持集构造器。

    Raises:
        GrpoTrainingError: 环境、checkpoint、策略、K、return 或 token 不符合契约。
        OSError: Tree group 无法读取。
        json.JSONDecodeError: 文件不是合法 JSON。

    Returns:
        GrpoTrainingGroup: 可交给共用 PPO/KL 优化器的八臂战略组。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("format") != "tree_grpo_group":
        raise GrpoTrainingError("Tree group 必须是已知 JSON 对象")
    group_id = _required_text(payload, "group_id")
    strategy_policy = _required_text(payload, "strategy_policy_version")
    battle_policy = _required_text(payload, "battle_policy_version")
    if not battle_policy:
        raise GrpoTrainingError("Tree group 缺少冻结 battle policy")
    environment = payload.get("environment")
    if (
        not isinstance(environment, Mapping)
        or environment.get("game_version") != RL_GAME_VERSION
        or not isinstance(environment.get("mod_version"), str)
        or not isinstance(environment.get("protocol_version"), str)
        or environment.get("structured_output_backend") != "xgrammar"
        or not isinstance(environment.get("structured_output_version"), str)
    ):
        raise GrpoTrainingError(f"Tree group 缺少固定 {RL_GAME_VERSION} 环境收据")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise GrpoTrainingError("Tree group 缺少 checkpoint 入口")
    if "audit" in checkpoint:
        raise GrpoTrainingError("Tree group 不得包含 checkpoint 隐藏审计")
    if (
        not isinstance(checkpoint.get("kind"), str)
        or not isinstance(checkpoint.get("screen"), str)
        or not isinstance(checkpoint.get("policy_text"), str)
        or not isinstance(checkpoint.get("option_ids"), list)
    ):
        raise GrpoTrainingError("Tree checkpoint 可见入口字段无效")
    if payload.get("behavior_logprobs_mode") != BEHAVIOR_LOGPROBS_MODE:
        raise GrpoTrainingError("Tree group 缺少 processed behavior log-prob")
    if payload.get("action_constraint_mode") != ACTION_CONSTRAINT_MODE:
        raise GrpoTrainingError("Tree group 缺少 structured choice 约束")
    generation_profile = payload.get("generation_profile")
    if not isinstance(generation_profile, Mapping):
        raise GrpoTrainingError("Tree group 缺少生成参数")
    try:
        temperature = float(generation_profile["temperature"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GrpoTrainingError("Tree group temperature 无效") from exc
    if not math.isfinite(temperature) or temperature <= 0:
        raise GrpoTrainingError("Tree group temperature 必须为正")
    max_macro_checkpoints = _required_integer(payload, "max_macro_checkpoints")
    if max_macro_checkpoints <= 0:
        raise GrpoTrainingError("Tree group horizon 必须为正")

    raw_arms = payload.get("arms")
    if not isinstance(raw_arms, list) or len(raw_arms) != 8:
        raise GrpoTrainingError("Tree group 必须恰好包含 K=8 arms")
    ordered = sorted(raw_arms, key=lambda arm: _required_integer(arm, "arm_index"))
    if [_required_integer(arm, "arm_index") for arm in ordered] != list(range(8)):
        raise GrpoTrainingError("Tree arm_index 必须连续且唯一")
    first_messages: object | None = None
    first_choices: object | None = None
    checkpoint_policy_text = str(checkpoint["policy_text"])
    for arm in ordered:
        if (
            arm.get("strategy_policy_version") != strategy_policy
            or arm.get("battle_policy_version") != battle_policy
            or arm.get("behavior_logprobs_mode") != BEHAVIOR_LOGPROBS_MODE
            or arm.get("action_constraint_mode") != ACTION_CONSTRAINT_MODE
            or arm.get("generation_profile") != generation_profile
            or arm.get("max_macro_checkpoints") != max_macro_checkpoints
        ):
            raise GrpoTrainingError("Tree arm 混入不同 policy、生成参数或 horizon")
        steps = arm.get("steps")
        plan_step_count = _required_integer(arm, "plan_step_count")
        if (
            not isinstance(steps, list)
            or plan_step_count <= 0
            or plan_step_count > len(steps)
            or "\n".join(
                _required_text(step, "action") for step in steps[:plan_step_count]
            )
            != _required_text(arm, "plan_id")
        ):
            raise GrpoTrainingError("Tree arm 的完整宏计划轨迹无效")
        first_step = steps[0]
        if not isinstance(first_step, Mapping):
            raise GrpoTrainingError("Tree 首步必须是对象")
        messages = first_step.get("messages")
        choices = first_step.get("response_choices")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or not isinstance(messages[1], Mapping)
            or messages[1].get("role") != "user"
            or messages[1].get("content") != checkpoint_policy_text
        ):
            raise GrpoTrainingError("Tree group 的玩家可见入口无效")
        if first_messages is None:
            first_messages = messages
            first_choices = choices
        elif messages != first_messages or choices != first_choices:
            raise GrpoTrainingError("Tree group 的玩家可见入口或首步候选不一致")
    successor_groups: dict[str, list[Mapping[str, Any]]] = {}
    for arm in ordered:
        successor_groups.setdefault(
            _required_text(arm, "macro_successor_text"), []
        ).append(arm)
    successor_counts = Counter(
        {successor: len(arms) for successor, arms in successor_groups.items()}
    )
    checkpoint_options = checkpoint["option_ids"]
    repeated_successors = sum(count >= 2 for count in successor_counts.values())
    if len(successor_counts) < 2 or (
        len(checkpoint_options) <= 4 and repeated_successors < 2
    ):
        raise GrpoTrainingError("Tree group 缺少两个重复覆盖的宏计划")

    rewards = tuple(_arm_return(arm) for arm in ordered)
    successor_means = {
        successor: statistics.fmean(_arm_return(arm) for arm in arms)
        for successor, arms in successor_groups.items()
    }
    branch_values = tuple(successor_means.values())
    reward_mean = statistics.fmean(branch_values)
    reward_std = statistics.pstdev(branch_values)
    if reward_std == 0.0:
        raise GrpoTrainingError("Tree group 的计划平均 continuation return 没有方差")
    advantages = tuple(
        (successor_means[_required_text(arm, "macro_successor_text")] - reward_mean)
        / reward_std
        for arm in ordered
    )
    successor_labels = {
        successor: _required_text(arms[0], "plan_id")
        for successor, arms in successor_groups.items()
    }
    expected_counts = tuple(
        (successor_labels[successor], float(len(arms)))
        for successor, arms in successor_groups.items()
    )
    expected_means = tuple(
        (successor_labels[successor], value)
        for successor, value in successor_means.items()
    )
    _validate_saved_values(payload.get("returns"), rewards, "returns")
    _validate_saved_values(payload.get("advantages"), advantages, "advantages")
    _validate_saved_named_values(
        payload.get("successor_counts"), expected_counts, "successor_counts"
    )
    _validate_saved_named_values(
        payload.get("successor_returns"), expected_means, "successor_returns"
    )
    schemes = {
        _required_text(arm.get("continuation_return"), "scheme") for arm in ordered
    }
    if len(schemes) != 1:
        raise GrpoTrainingError("Tree arms 混入不同 return scheme")
    if choice_masker is None:
        choice_masker = XGrammarChoiceMasker(
            tokenizer,
            expected_version=str(environment["structured_output_version"]),
        )
    training_arms = []
    for index, arm in enumerate(ordered):
        projected = dict(arm)
        steps = arm["steps"]
        plan_step_count = int(arm["plan_step_count"])
        projected["steps"] = steps[:plan_step_count]
        training_arms.append(
            build_grpo_arm(
                projected,
                tokenizer,
                advantage=advantages[index],
                reward=rewards[index],
                max_length=max_length,
                temperature=temperature,
                choice_masker=choice_masker,
            )
        )
    arms = tuple(training_arms)
    return GrpoTrainingGroup(
        group_id=group_id,
        policy_version=strategy_policy,
        reward_scheme=schemes.pop(),
        rewards=rewards,
        advantages=advantages,
        arms=arms,
    )


def _arm_return(arm: Mapping[str, Any]) -> float:
    """读取一条 arm 的有限 continuation return。

    Args:
        arm (Mapping[str, Any]): Tree arm JSON。

    Raises:
        GrpoTrainingError: return 对象或总值无效。

    Returns:
        float: 当前 arm 标量 return。
    """
    value = arm.get("continuation_return")
    if not isinstance(value, Mapping):
        raise GrpoTrainingError("Tree arm 缺少 continuation return")
    try:
        total = float(value["total"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GrpoTrainingError("Tree arm return 无效") from exc
    if not math.isfinite(total):
        raise GrpoTrainingError("Tree arm return 必须有限")
    return total


def _validate_saved_values(
    raw: object,
    expected: tuple[float, ...],
    field: str,
) -> None:
    """核对 collector 保存的 return 或 advantage 与 learner 重算一致。

    Args:
        raw (object): JSON 中的数值数组。
        expected (tuple[float, ...]): learner 重算值。
        field (str): 报错字段名称。

    Raises:
        GrpoTrainingError: 数组长度、数值或重算结果不一致。

    Returns:
        None: 每项在浮点容差内一致时返回。
    """
    if not isinstance(raw, list) or len(raw) != len(expected):
        raise GrpoTrainingError(f"Tree group {field} 无效")
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise GrpoTrainingError(f"Tree group {field} 无效") from exc
    if any(
        not math.isfinite(actual)
        or not math.isclose(actual, target, rel_tol=1e-6, abs_tol=1e-6)
        for actual, target in zip(values, expected, strict=True)
    ):
        raise GrpoTrainingError(f"Tree group {field} 与 learner 重算不一致")


def _validate_saved_named_values(
    raw: object,
    expected: tuple[tuple[str, float], ...],
    field: str,
) -> None:
    """核对带代表计划名的计数或平均 return 收据。

    Args:
        raw (object): JSON 中的 ``[name, value]`` 数组。
        expected (tuple[tuple[str, float], ...]): learner 从 arms 重算的值。
        field (str): 报错字段名称。

    Raises:
        GrpoTrainingError: 名称、长度、数值或顺序与重算结果不一致。

    Returns:
        None: 全部命名数值一致时返回。
    """
    if not isinstance(raw, list) or len(raw) != len(expected):
        raise GrpoTrainingError(f"Tree group {field} 无效")
    for item, (expected_name, expected_value) in zip(raw, expected, strict=True):
        if not isinstance(item, list) or len(item) != 2 or item[0] != expected_name:
            raise GrpoTrainingError(f"Tree group {field} 无效")
        try:
            actual = float(item[1])
        except (TypeError, ValueError) as exc:
            raise GrpoTrainingError(f"Tree group {field} 无效") from exc
        if not math.isfinite(actual) or not math.isclose(
            actual,
            expected_value,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise GrpoTrainingError(f"Tree group {field} 与 learner 重算不一致")


def _required_text(value: object, field: str) -> str:
    """读取 JSON 对象中的非空字符串。

    Args:
        value (object): 当前 JSON 对象。
        field (str): 字段名。

    Raises:
        GrpoTrainingError: 对象或字符串字段无效。

    Returns:
        str: 校验后的字符串。
    """
    if not isinstance(value, Mapping):
        raise GrpoTrainingError("Tree JSON 项必须是对象")
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise GrpoTrainingError(f"Tree 字段必须是非空字符串: {field}")
    return item


def _required_integer(value: object, field: str) -> int:
    """读取 JSON 对象中的非负整数。

    Args:
        value (object): 当前 JSON 对象。
        field (str): 字段名。

    Raises:
        GrpoTrainingError: 对象或整数字段无效。

    Returns:
        int: 校验后的整数。
    """
    if not isinstance(value, Mapping):
        raise GrpoTrainingError("Tree JSON 项必须是对象")
    item = value.get(field)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise GrpoTrainingError(f"Tree 字段必须是非负整数: {field}")
    return item
