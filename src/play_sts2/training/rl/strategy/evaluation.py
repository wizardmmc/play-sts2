"""评估冻结战略 policy 在 held-out Tree group 上的分支 regret。"""

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ....inference import ChatMessage, OpenAICompatibleProvider


def evaluate_tree_branch_regret(
    *,
    rollout_path: Path,
    model_url: str,
    policy_model: str,
    output_path: Path,
    max_tokens: int = 128,
) -> dict[str, Any]:
    """查询一个冻结 policy 并计算相对最佳已采兄弟计划的 regret。

    Args:
        rollout_path (Path): held-out K=8 Tree group JSON。
        model_url (str): A100 OpenAI-compatible 推理地址。
        policy_model (str): 待评估战略 policy 名称。
        output_path (Path): regret 报告路径。
        max_tokens (int): 结构化回复 token 上限。

    Raises:
        TypeError: arm、计划、return 或消息对象类型无效。
        ValueError: group、入口步骤、模型动作或 return 不可用于 regret。
        OSError: group 或报告无法读写。

    Returns:
        dict[str, Any]: 所选计划、兄弟均值、最佳均值与 regret。
    """
    payload = json.loads(Path(rollout_path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("format") != "tree_grpo_group":
        raise ValueError("held-out Tree group 格式无效")
    arms = payload.get("arms")
    if not isinstance(arms, list) or len(arms) != 8:
        raise ValueError("held-out Tree group 必须包含 K=8 arms")
    first_step = _first_step(arms[0])
    messages = _messages(first_step.get("messages"))
    raw_choices = first_step.get("response_choices")
    if (
        not isinstance(raw_choices, list)
        or not raw_choices
        or not all(isinstance(choice, str) and choice for choice in raw_choices)
    ):
        raise ValueError("held-out Tree group 缺少入口结构化动作候选")
    choices = tuple(raw_choices)
    with OpenAICompatibleProvider(
        model_url,
        model=policy_model,
        enable_thinking=False,
    ) as provider:
        reply = provider.chat(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            response_choices=choices,
        )
    selected_plan = reply.text
    plan_returns: dict[str, list[float]] = defaultdict(list)
    for arm in arms:
        if not isinstance(arm, Mapping):
            raise TypeError("held-out Tree arm 必须是对象")
        plan = _first_step(arm).get("action")
        continuation = arm.get("continuation_return")
        if not isinstance(plan, str) or not isinstance(continuation, Mapping):
            raise TypeError("held-out Tree arm 缺少计划或 return")
        try:
            score = float(continuation["total"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("held-out Tree arm return 无效") from exc
        if not math.isfinite(score):
            raise ValueError("held-out Tree arm return 必须有限")
        plan_returns[plan].append(score)
    if selected_plan not in plan_returns:
        raise ValueError("policy 选择的计划没有已采兄弟 return，无法计算 regret")
    plan_means = {
        plan: sum(values) / len(values) for plan, values in plan_returns.items()
    }
    best_plan = max(plan_means, key=plan_means.__getitem__)
    selected_mean = plan_means[selected_plan]
    best_mean = plan_means[best_plan]
    checkpoint = payload.get("checkpoint")
    checkpoint_kind = (
        checkpoint.get("kind") if isinstance(checkpoint, Mapping) else None
    )
    report: dict[str, Any] = {
        "group_id": payload.get("group_id"),
        "checkpoint_kind": checkpoint_kind,
        "policy_model": policy_model,
        "selected_plan": selected_plan,
        "selected_mean_return": selected_mean,
        "best_sampled_plan": best_plan,
        "best_sampled_mean_return": best_mean,
        "regret": max(0.0, best_mean - selected_mean),
        "sampled_plan_means": plan_means,
        "sampled_plan_counts": {
            plan: len(values) for plan, values in plan_returns.items()
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _first_step(arm: object) -> Mapping[str, Any]:
    """读取第一条 arm 的入口战略步骤。

    Args:
        arm (object): Tree arm JSON。

    Raises:
        TypeError: arm 不是对象。
        ValueError: arm 或步骤结构无效。

    Returns:
        Mapping[str, Any]: 入口战略步骤。
    """
    if not isinstance(arm, Mapping):
        raise TypeError("held-out Tree arm 必须是对象")
    steps = arm.get("steps")
    if not isinstance(steps, list) or not steps or not isinstance(steps[0], Mapping):
        raise ValueError("held-out Tree arm 缺少入口战略步骤")
    return steps[0]


def _messages(raw: object) -> tuple[ChatMessage, ...]:
    """把落盘消息还原为推理 Provider 输入。

    Args:
        raw (object): Tree step 的消息数组。

    Raises:
        TypeError: 消息项不是对象。
        ValueError: 消息不是 system/user 两条非空文本。

    Returns:
        tuple[ChatMessage, ...]: 可直接重新查询冻结 policy 的消息。
    """
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError("held-out Tree 入口消息必须为 stateless system/user")
    messages: list[ChatMessage] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise TypeError("held-out Tree 消息必须是对象")
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user"} or not isinstance(content, str):
            raise ValueError("held-out Tree 消息角色或内容无效")
        messages.append(ChatMessage(role=role, content=content))
    if tuple(message.role for message in messages) != ("system", "user"):
        raise ValueError("held-out Tree 入口消息必须按 system/user 排列")
    return tuple(messages)
