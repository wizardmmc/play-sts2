"""构造单战斗 GRPO 训练批并提供 token 级损失数学。"""

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .contracts import ACTION_CONSTRAINT_MODE, BEHAVIOR_LOGPROBS_MODE, RL_GAME_VERSION
from .reward import BattleRewardInput, BattleRewardScheme, score_battle_reward


class GrpoTrainingError(RuntimeError):
    """表示 rollout、token 或训练状态不满足 GRPO 契约。"""


@dataclass(frozen=True, slots=True)
class GrpoSequence:
    """保存一个 stateless 决策步骤的 token 训练事实。

    Args:
        arm_index (int): 来源 arm 序号。
        step_index (int): 来源战斗步骤序号。
        input_ids (tuple[int, ...]): prompt 与 rollout completion token。
        assistant_mask (tuple[bool, ...]): 仅 completion token 为真。
        behavior_logprobs (tuple[float, ...]): rollout 时逐 token 真实旧概率。
        allowed_token_ids (tuple[tuple[int, ...], ...]): xgrammar 每步允许的 token。
        temperature (float): rollout 时结构化采样使用的温度。
        advantage (float): 当前 arm 的组相对优势。
    """

    arm_index: int
    step_index: int
    input_ids: tuple[int, ...]
    assistant_mask: tuple[bool, ...]
    behavior_logprobs: tuple[float, ...]
    allowed_token_ids: tuple[tuple[int, ...], ...]
    temperature: float
    advantage: float


@dataclass(frozen=True, slots=True)
class GrpoArm:
    """保存一条完整学生 arm 的训练序列。

    Args:
        arm_index (int): 来源 arm 序号。
        reward (float): 当前奖励方案下的终局回报。
        advantage (float): 组内标准化相对优势。
        sequences (tuple[GrpoSequence, ...]): 按步骤排列的 stateless 序列。
        proposal_probability (float): 当前 arm 根计划在 proposal 下的概率。
        recompute_root_probability (bool): 是否用冻结父策略重算根动作概率和
            PPO behavior 概率。
    """

    arm_index: int
    reward: float
    advantage: float
    sequences: tuple[GrpoSequence, ...]
    proposal_probability: float = 1.0
    recompute_root_probability: bool = False

    @property
    def supervised_tokens(self) -> int:
        """返回当前 arm 的 rollout completion token 数。

        Returns:
            int: 所有步骤 assistant mask 的真值总数。
        """
        return sum(sum(sequence.assistant_mask) for sequence in self.sequences)


@dataclass(frozen=True, slots=True)
class GrpoTrainingGroup:
    """保存一个同入口、同策略的可训练 GRPO group。

    Args:
        group_id (str): 来源 group 标识。
        policy_version (str): rollout 冻结学生 policy。
        reward_scheme (str): 本批重算使用的奖励方案。
        rewards (tuple[float, ...]): 按 arm 排列的终局回报。
        advantages (tuple[float, ...]): 按总体标准差归一的组相对优势。
        arms (tuple[GrpoArm, ...]): 可供 learner 顺序反向传播的完整 arms。
        battle_policy_version (str | None): 战略数据对应的冻结战斗 policy。
        generation_profile (Mapping[str, Any] | None): rollout 的完整生成参数。
        environment (Mapping[str, Any] | None): 游戏、Mod 与约束运行时收据。
    """

    group_id: str
    policy_version: str
    reward_scheme: str
    rewards: tuple[float, ...]
    advantages: tuple[float, ...]
    arms: tuple[GrpoArm, ...]
    battle_policy_version: str | None = None
    generation_profile: Mapping[str, Any] | None = None
    environment: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class GrpoLossResult:
    """保存一条序列的可读 GRPO 损失诊断。

    Args:
        total_loss (float): policy loss 加权 sampled KL。
        policy_loss (float): clipped PPO 负目标。
        kl (float): Schulman 非负 sampled KL。
        ratio_mean (float): 新旧策略逐 token ratio 均值。
    """

    total_loss: float
    policy_loss: float
    kl: float
    ratio_mean: float


class XGrammarChoiceMasker:
    """使用与 vLLM 0.19 相同的 xgrammar choice EBNF 生成 token bitmask。"""

    def __init__(self, tokenizer: Any, *, expected_version: str) -> None:
        """为当前 tokenizer 建立带缓存的 grammar compiler。

        Args:
            tokenizer (Any): rollout 与 learner 共用的 Hugging Face tokenizer。
            expected_version (str): group 收据声明的 xgrammar 版本。

        Raises:
            GrpoTrainingError: 本地 xgrammar 与采样运行时版本不一致。
        """
        import xgrammar

        actual_version = version("xgrammar")
        if expected_version != actual_version:
            raise GrpoTrainingError(
                "GRPO xgrammar 版本与采样运行时不一致: "
                f"expected={expected_version}, actual={actual_version}"
            )
        self._xgrammar = xgrammar
        self._vocab_size = len(tokenizer)
        tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(
            tokenizer,
            vocab_size=self._vocab_size,
        )
        self._compiler = xgrammar.GrammarCompiler(tokenizer_info)

    def allowed_token_ids(
        self,
        choices: Sequence[str],
        completion_ids: Sequence[int],
    ) -> tuple[tuple[int, ...], ...]:
        """沿实际 completion 前缀读取每一步 xgrammar 允许的 token 集。

        Args:
            choices (Sequence[str]): 当前结构化完整动作候选。
            completion_ids (Sequence[int]): vLLM 实际生成 token。

        Raises:
            GrpoTrainingError: grammar 不能接受实际 rollout token。

        Returns:
            tuple[tuple[int, ...], ...]: 与 completion 等长的允许 token ID。
        """
        import torch

        grammar = "root ::= " + " | ".join(
            f'"{_escape_ebnf_choice(choice)}"' for choice in choices
        )
        matcher = self._xgrammar.GrammarMatcher(self._compiler.compile_grammar(grammar))
        allowed_steps = []
        for token in completion_ids:
            bitmask = self._xgrammar.allocate_token_bitmask(1, self._vocab_size)
            if not matcher.fill_next_token_bitmask(bitmask, 0):
                raise GrpoTrainingError("xgrammar 无法生成下一 token bitmask")
            masked = torch.zeros((1, self._vocab_size), dtype=torch.float32)
            self._xgrammar.apply_token_bitmask_inplace(
                masked,
                bitmask,
                vocab_size=self._vocab_size,
            )
            allowed = tuple(
                int(index)
                for index in torch.isfinite(masked[0]).nonzero().flatten().tolist()
            )
            if int(token) not in allowed or not matcher.accept_token(int(token)):
                raise GrpoTrainingError("rollout token 不属于 xgrammar choice 支持集")
            allowed_steps.append(allowed)
        return tuple(allowed_steps)


def load_grpo_training_group(
    path: Path,
    tokenizer: Any,
    *,
    reward_scheme: BattleRewardScheme,
    max_length: int,
    potion_cost: float = 0.25,
    choice_masker: Any | None = None,
    allow_zero_variance: bool = False,
) -> GrpoTrainingGroup:
    """从完整 rollout JSON 重算奖励并构造 stateless token 序列。

    Args:
        path (Path): 第一阶段写出的 battle group JSON。
        tokenizer (Any): 与 rollout policy 相同基座的 tokenizer。
        reward_scheme (BattleRewardScheme): 本轮奖励消融方案。
        max_length (int): 单步 prompt 加 completion 的最大 token 数。
        potion_cost (float): 固定药水成本方案的每瓶成本。
        choice_masker (Any | None): 测试可注入的 choice bitmask 构造器。
        allow_zero_variance (bool): 是否为独立 DAgger 构造零优势训练组。

    Raises:
        GrpoTrainingError: group、策略、token、行为概率或奖励方差无效。
        OSError: 文件无法读取。
        json.JSONDecodeError: 文件不是合法 JSON。

    Returns:
        GrpoTrainingGroup: 可直接交给单卡 learner 的训练批。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise GrpoTrainingError("GRPO group 必须是 JSON 对象")
    group_id = _required_text(payload, "group_id")
    policy_version = _required_text(payload, "policy_version")
    environment = payload.get("environment")
    if (
        not isinstance(environment, Mapping)
        or environment.get("game_version") != RL_GAME_VERSION
        or not isinstance(environment.get("mod_version"), str)
        or not isinstance(environment.get("protocol_version"), str)
        or environment.get("structured_output_backend") != "xgrammar"
        or not isinstance(environment.get("structured_output_version"), str)
    ):
        raise GrpoTrainingError(f"GRPO group 缺少固定 {RL_GAME_VERSION} 游戏运行时收据")
    if payload.get("behavior_logprobs_mode") != BEHAVIOR_LOGPROBS_MODE:
        raise GrpoTrainingError("GRPO group 缺少 processed behavior log-probs")
    if payload.get("action_constraint_mode") != ACTION_CONSTRAINT_MODE:
        raise GrpoTrainingError("GRPO group 缺少结构化动作约束")
    raw_rollouts = payload.get("rollouts")
    if not isinstance(raw_rollouts, list) or len(raw_rollouts) != 8:
        raise GrpoTrainingError("GRPO group 必须恰好包含八条 arms")
    ordered = sorted(
        raw_rollouts, key=lambda value: _required_integer(value, "arm_index")
    )
    if [int(value["arm_index"]) for value in ordered] != list(range(len(ordered))):
        raise GrpoTrainingError("GRPO arm_index 必须连续且唯一")
    generation_profile = payload.get("generation_profile")
    if not isinstance(generation_profile, Mapping):
        raise GrpoTrainingError("GRPO group 缺少 generation profile")
    _validate_group_rollouts(
        payload,
        ordered,
        policy_version=policy_version,
        generation_profile=generation_profile,
    )
    rewards = tuple(
        _rollout_reward(value, policy_version, reward_scheme, potion_cost)
        for value in ordered
    )
    reward_mean = statistics.fmean(rewards)
    reward_std = statistics.pstdev(rewards)
    if reward_std == 0.0 and not allow_zero_variance:
        raise GrpoTrainingError("GRPO group 奖励没有方差")
    advantages = (
        (0.0,) * len(rewards)
        if reward_std == 0.0
        else tuple((reward - reward_mean) / reward_std for reward in rewards)
    )
    try:
        temperature = float(generation_profile["temperature"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GrpoTrainingError("GRPO group temperature 无效") from exc
    if not math.isfinite(temperature) or temperature <= 0:
        raise GrpoTrainingError("GRPO group temperature 必须为正数")
    if choice_masker is None:
        choice_masker = XGrammarChoiceMasker(
            tokenizer,
            expected_version=str(environment["structured_output_version"]),
        )
    arms = tuple(
        build_grpo_arm(
            value,
            tokenizer,
            advantage=advantages[index],
            reward=rewards[index],
            max_length=max_length,
            temperature=temperature,
            choice_masker=choice_masker,
        )
        for index, value in enumerate(ordered)
    )
    return GrpoTrainingGroup(
        group_id=group_id,
        policy_version=policy_version,
        reward_scheme=reward_scheme,
        rewards=rewards,
        advantages=advantages,
        arms=arms,
        generation_profile=dict(generation_profile),
        environment=dict(environment),
    )


def _validate_group_rollouts(
    payload: Mapping[str, Any],
    rollouts: Sequence[Mapping[str, Any]],
    *,
    policy_version: str,
    generation_profile: Mapping[str, Any],
) -> None:
    """在 learner 边界重新验证八臂同入口与同采样配置。

    Args:
        payload (Mapping[str, Any]): group 顶层 JSON。
        rollouts (Sequence[Mapping[str, Any]]): 已按 arm_index 排序的八条 arm。
        policy_version (str): 顶层冻结策略名。
        generation_profile (Mapping[str, Any]): 顶层生成参数。

    Raises:
        GrpoTrainingError: 任一 arm 的入口、策略、模式、步骤或 profile 不一致。
    """
    entry_snapshot = payload.get("entry_snapshot")
    if not isinstance(entry_snapshot, Mapping):
        raise GrpoTrainingError("GRPO group 缺少入口 snapshot")
    for rollout in rollouts:
        if (
            rollout.get("policy_version") != policy_version
            or rollout.get("behavior_logprobs_mode") != BEHAVIOR_LOGPROBS_MODE
            or rollout.get("action_constraint_mode") != ACTION_CONSTRAINT_MODE
            or rollout.get("generation_profile") != generation_profile
            or rollout.get("entry_snapshot") != entry_snapshot
        ):
            raise GrpoTrainingError("GRPO arm 的入口、策略或采样配置不一致")
        steps = rollout.get("steps")
        if not isinstance(steps, list) or not steps:
            raise GrpoTrainingError("GRPO arm 没有步骤")
        indices = [_required_integer(step, "index") for step in steps]
        if indices != list(range(len(steps))):
            raise GrpoTrainingError("GRPO step index 必须连续且有序")
        for step in steps:
            if not isinstance(step, Mapping):
                raise GrpoTrainingError("GRPO step 必须是对象")
            choices = step.get("response_choices")
            if (
                not isinstance(step.get("finish_reason"), str)
                or not isinstance(choices, list)
                or not choices
            ):
                raise GrpoTrainingError("GRPO step 缺少 finish reason 或动作候选")


def grpo_sequence_loss(
    new_logprobs: Any,
    behavior_logprobs: Any,
    anchor_logprobs: Any,
    *,
    advantage: float,
    clip: float,
    kl_beta: float,
) -> GrpoLossResult:
    """计算一条序列的 token 级 clipped PPO 与 sampled KL。

    Args:
        new_logprobs (Any): 当前 policy 对 rollout token 的 log-prob。
        behavior_logprobs (Any): rollout 时保存的真实旧 policy log-prob。
        anchor_logprobs (Any): 冻结父 adapter 的 token log-prob。
        advantage (float): 当前 arm 的组相对优势。
        clip (float): PPO ratio 对称裁剪半径。
        kl_beta (float): sampled KL 权重。

    Raises:
        ValueError: 张量形状、裁剪、权重或数值无效。

    Returns:
        GrpoLossResult: 已 detach 的可读损失诊断。
    """
    total, policy, kl, ratio = _grpo_loss_tensors(
        new_logprobs,
        behavior_logprobs,
        anchor_logprobs,
        advantage=advantage,
        clip=clip,
        kl_beta=kl_beta,
    )
    return GrpoLossResult(
        total_loss=float(total.detach().cpu()),
        policy_loss=float(policy.detach().cpu()),
        kl=float(kl.detach().cpu()),
        ratio_mean=float(ratio.detach().cpu()),
    )


def stratified_importance_weight(
    anchor_logprobs: Any,
    *,
    proposal_probability: float,
) -> float:
    """计算强制根动作 ``pi_old(a|s) / q(a|s)``。

    Args:
        anchor_logprobs (Any): 冻结父策略在完整 xgrammar 支持集上的逐 token
            条件 log-prob。
        proposal_probability (float): 分层采样 proposal 的根动作概率。

    Raises:
        ValueError: proposal、log-prob 或最终权重不是有限正数。

    Returns:
        float: 可乘到当前 terminal branch policy loss 的重要性权重。
    """
    import torch

    if not math.isfinite(proposal_probability) or not 0 < proposal_probability <= 1:
        raise ValueError("Tree proposal probability 必须位于 (0, 1]")
    if anchor_logprobs.numel() == 0 or not torch.isfinite(anchor_logprobs).all().item():
        raise ValueError("Tree anchor log-prob 必须是有限非空张量")
    probability = float(torch.exp(anchor_logprobs.float().sum()).detach().cpu())
    weight = probability / proposal_probability
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("Tree importance weight 必须是有限正数")
    return weight


def compare_battle_reward_schemes(
    paths: Sequence[Path],
) -> dict[str, Any]:
    """对相同 rollout 离线比较第三阶段固定奖励方案。

    报告保留 arm 级回报和排序，便于识别均值相近但偏好方向不同的 reward hacking。
    此函数只重算奖励，不执行模型更新，也不把工程数据提升为正式结论。

    Args:
        paths (Sequence[Path]): 待比较的完整八臂 group JSON。

    Raises:
        GrpoTrainingError: 输入为空或任一 group 不满足 learner 契约。

    Returns:
        dict[str, Any]: 含每组各方案均值、标准差、arm 回报与排序的报告。
    """
    if not paths:
        raise GrpoTrainingError("奖励消融至少需要一个 rollout group")
    definitions = (
        ("core", "core", 0.25),
        ("core_no_turn", "core_no_turn", 0.25),
        ("potion_cost_0.1", "potion_cost", 0.1),
        ("potion_cost_0.25", "potion_cost", 0.25),
        ("legacy_remaining_potion", "legacy_remaining_potion", 0.25),
    )
    groups = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise GrpoTrainingError("奖励消融 group 必须是 JSON 对象")
        group_id = _required_text(payload, "group_id")
        policy_version = _required_text(payload, "policy_version")
        environment = payload.get("environment")
        if (
            not isinstance(environment, Mapping)
            or environment.get("game_version") != RL_GAME_VERSION
            or environment.get("structured_output_backend") != "xgrammar"
            or not isinstance(environment.get("structured_output_version"), str)
        ):
            raise GrpoTrainingError("奖励消融 group 缺少固定运行时收据")
        raw_rollouts = payload.get("rollouts")
        generation_profile = payload.get("generation_profile")
        if not isinstance(raw_rollouts, list) or len(raw_rollouts) != 8:
            raise GrpoTrainingError("奖励消融 group 必须恰好包含八条 arms")
        if not isinstance(generation_profile, Mapping):
            raise GrpoTrainingError("奖励消融 group 缺少 generation profile")
        ordered = sorted(
            raw_rollouts,
            key=lambda value: _required_integer(value, "arm_index"),
        )
        if [int(value["arm_index"]) for value in ordered] != list(range(8)):
            raise GrpoTrainingError("奖励消融 arm_index 必须连续且唯一")
        _validate_group_rollouts(
            payload,
            ordered,
            policy_version=policy_version,
            generation_profile=generation_profile,
        )
        schemes = {}
        for label, scheme, potion_cost in definitions:
            rewards = tuple(
                _rollout_reward(
                    rollout,
                    policy_version,
                    scheme,  # type: ignore[arg-type]
                    potion_cost,
                )
                for rollout in ordered
            )
            reward_std = statistics.pstdev(rewards)
            schemes[label] = {
                "mean": statistics.fmean(rewards),
                "std": reward_std,
                "trainable": reward_std > 0,
                "rewards": list(rewards),
                "arm_order": sorted(
                    range(len(rewards)),
                    key=lambda index: (-rewards[index], index),
                ),
            }
        groups.append(
            {
                "path": str(path),
                "group_id": group_id,
                "schemes": schemes,
            }
        )
    return {
        "format": "battle_reward_comparison",
        "selected_default": "core",
        "groups": groups,
    }


def _grpo_loss_tensors(
    new_logprobs: Any,
    behavior_logprobs: Any,
    anchor_logprobs: Any,
    *,
    advantage: float,
    clip: float,
    kl_beta: float,
) -> tuple[Any, Any, Any, Any]:
    """返回保持梯度的 GRPO 损失张量。

    Args:
        new_logprobs (Any): 当前 policy log-prob 张量。
        behavior_logprobs (Any): rollout behavior log-prob 张量。
        anchor_logprobs (Any): 冻结锚 log-prob 张量。
        advantage (float): arm 相对优势。
        clip (float): PPO 裁剪半径。
        kl_beta (float): KL 权重。

    Raises:
        ValueError: 输入形状、超参数或数值无效。

    Returns:
        tuple[Any, Any, Any, Any]: total、policy、KL 和 ratio 均值。
    """
    import torch

    if not 0 <= clip < 1 or kl_beta < 0 or not math.isfinite(advantage):
        raise ValueError("GRPO clip、KL 或 advantage 无效")
    if (
        new_logprobs.shape != behavior_logprobs.shape
        or new_logprobs.shape != anchor_logprobs.shape
        or new_logprobs.numel() == 0
    ):
        raise ValueError("GRPO token log-prob 形状不一致或为空")
    if not all(
        torch.isfinite(value).all().item()
        for value in (new_logprobs, behavior_logprobs, anchor_logprobs)
    ):
        raise ValueError("GRPO token log-prob 必须有限")
    ratio = torch.exp((new_logprobs - behavior_logprobs).clamp(max=30.0))
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    advantage_tensor = torch.as_tensor(
        advantage,
        dtype=new_logprobs.dtype,
        device=new_logprobs.device,
    )
    policy_loss = -torch.minimum(
        ratio * advantage_tensor,
        clipped * advantage_tensor,
    ).mean()
    delta = anchor_logprobs - new_logprobs
    kl = (torch.exp(delta.clamp(max=30.0)) - delta - 1.0).clamp(min=0.0).mean()
    return policy_loss + kl_beta * kl, policy_loss, kl, ratio.mean()


def build_grpo_arm(
    rollout: Mapping[str, Any],
    tokenizer: Any,
    *,
    advantage: float,
    reward: float,
    max_length: int,
    temperature: float,
    choice_masker: Any,
) -> GrpoArm:
    """构造一条 arm 的全部 stateless 决策序列。

    Args:
        rollout (Mapping[str, Any]): 一条 JSON rollout。
        tokenizer (Any): 生成 prompt token 的 tokenizer。
        advantage (float): 当前 arm 优势。
        reward (float): 当前 arm 回报。
        max_length (int): 单步最大 token 数。
        temperature (float): rollout 时结构化采样温度。
        choice_masker (Any): 与采样端一致的 grammar token bitmask 构造器。

    Raises:
        GrpoTrainingError: 任一步骤消息、token 或概率无效。

    Returns:
        GrpoArm: 可训练 arm。
    """
    arm_index = _required_integer(rollout, "arm_index")
    raw_steps = rollout.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise GrpoTrainingError("GRPO arm 没有步骤")
    sequences: list[GrpoSequence] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, Mapping):
            raise GrpoTrainingError("GRPO step 必须是对象")
        step_index = _required_integer(raw_step, "index")
        raw_messages = raw_step.get("messages")
        if not isinstance(raw_messages, list) or [
            message.get("role") if isinstance(message, Mapping) else None
            for message in raw_messages
        ] != ["system", "user"]:
            raise GrpoTrainingError("GRPO step 必须使用 stateless system/user 消息")
        messages = [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in raw_messages
        ]
        prompt_ids = tuple(
            int(token)
            for token in tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
        )
        raw_token_ids = raw_step.get("token_ids")
        raw_logprobs = raw_step.get("behavior_logprobs")
        if (
            not isinstance(raw_token_ids, list)
            or not isinstance(raw_logprobs, list)
            or not raw_token_ids
            or len(raw_token_ids) != len(raw_logprobs)
        ):
            raise GrpoTrainingError("GRPO step token 与 behavior log-prob 不对齐")
        completion_ids = tuple(int(token) for token in raw_token_ids)
        behavior_logprobs = tuple(float(value) for value in raw_logprobs)
        if any(not math.isfinite(value) or value > 0 for value in behavior_logprobs):
            raise GrpoTrainingError("GRPO behavior log-prob 无效")
        input_ids = (*prompt_ids, *completion_ids)
        if len(input_ids) > max_length:
            raise GrpoTrainingError("GRPO stateless step 超过 max_length")
        reply_text = raw_step.get("reply_text")
        action = raw_step.get("action")
        is_model_failure = raw_step.get("is_model_failure") is True
        decoded = tokenizer.decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if (
            not isinstance(reply_text, str)
            or decoded != reply_text
            or (is_model_failure and action != "")
            or (
                not is_model_failure
                and (not isinstance(action, str) or action != reply_text)
            )
        ):
            raise GrpoTrainingError("GRPO reply、action 与 completion token 不一致")
        raw_choices = raw_step.get("response_choices")
        if (
            not isinstance(raw_choices, list)
            or not raw_choices
            or any(not isinstance(choice, str) or not choice for choice in raw_choices)
        ):
            raise GrpoTrainingError("GRPO step 缺少结构化回复候选")
        allowed_token_ids = choice_masker.allowed_token_ids(
            tuple(raw_choices),
            completion_ids,
        )
        sequences.append(
            GrpoSequence(
                arm_index=arm_index,
                step_index=step_index,
                input_ids=input_ids,
                assistant_mask=(False,) * len(prompt_ids)
                + (True,) * len(completion_ids),
                behavior_logprobs=behavior_logprobs,
                allowed_token_ids=allowed_token_ids,
                temperature=temperature,
                advantage=advantage,
            )
        )
    return GrpoArm(
        arm_index=arm_index,
        reward=reward,
        advantage=advantage,
        sequences=tuple(sequences),
    )


def constrained_completion_logprobs(
    logits: Any,
    *,
    completion_ids: Sequence[int],
    allowed_token_ids: Sequence[Sequence[int]],
    temperature: float,
) -> Any:
    """在 xgrammar structured-choice bitmask 上计算逐 token 条件 log-prob。

    rollout 保存的是 vLLM 在温度和 choice grammar 之后的 processed log-prob。
    learner 因此也只在采样端 grammar matcher 允许的全词表 token 上归一化；不能
    用 choice 的单条 canonical tokenization 代替真实支持集。

    Args:
        logits (Any): 每个 completion token 对应的未归一化全词表 logits。
        completion_ids (Sequence[int]): 实际 rollout completion token。
        allowed_token_ids (Sequence[Sequence[int]]): 每一步 grammar 允许的 token ID。
        temperature (float): rollout 采样温度。

    Raises:
        ValueError: 形状、温度或实际 token 不符合 grammar bitmask。

    Returns:
        Any: 与 completion 等长、保持梯度的条件 log-prob 张量。
    """
    import torch

    actual = tuple(int(token) for token in completion_ids)
    allowed_steps = tuple(
        tuple(int(token) for token in allowed) for allowed in allowed_token_ids
    )
    if (
        logits.ndim != 2
        or logits.shape[0] != len(actual)
        or not actual
        or len(allowed_steps) != len(actual)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("structured-choice logits、候选或温度无效")
    values = []
    for index, selected in enumerate(actual):
        allowed = sorted(set(allowed_steps[index]))
        if selected not in allowed:
            raise ValueError("实际 completion token 不属于 grammar 支持集")
        allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=logits.device)
        selected_index = allowed.index(selected)
        normalized = torch.log_softmax(
            logits[index].index_select(0, allowed_tensor).float() / temperature,
            dim=0,
        )
        values.append(normalized[selected_index])
    return torch.stack(values)


def _escape_ebnf_choice(choice: str) -> str:
    """按 vLLM 0.19 ``choice_as_grammar`` 规则转义 EBNF 字符串。

    Args:
        choice (str): 一个完整结构化动作候选。

    Returns:
        str: 双引号与反斜线已转义的 EBNF 内容。
    """
    return choice.replace("\\", "\\\\").replace('"', '\\"')


def _rollout_reward(
    rollout: Mapping[str, Any],
    policy_version: str,
    scheme: BattleRewardScheme,
    potion_cost: float,
) -> float:
    """从一条 JSON arm 重算指定方案的终局回报。

    Args:
        rollout (Mapping[str, Any]): 一条完整学生 arm。
        policy_version (str): group 绑定的学生 policy。
        scheme (BattleRewardScheme): 奖励方案。
        potion_cost (float): 固定药水成本。

    Raises:
        GrpoTrainingError: policy、状态、动作或终局无效。

    Returns:
        float: 当前方案的标量回报。
    """
    if rollout.get("policy_version") != policy_version:
        raise GrpoTrainingError("GRPO group 混入不同 policy")
    steps = rollout.get("steps")
    final_state = rollout.get("final_state")
    if not isinstance(steps, list) or not steps or not isinstance(final_state, Mapping):
        raise GrpoTrainingError("GRPO arm 缺少步骤或终局")
    entry_state = (
        steps[0].get("before_state") if isinstance(steps[0], Mapping) else None
    )
    if not isinstance(entry_state, Mapping):
        raise GrpoTrainingError("GRPO arm 缺少入口状态")
    entry_run = entry_state.get("run")
    final_run = final_state.get("run")
    if not isinstance(entry_run, Mapping) or not isinstance(final_run, Mapping):
        raise GrpoTrainingError("GRPO arm 缺少生命值")
    entry_hp = _mapping_integer(entry_run, "current_hp")
    max_hp = _mapping_integer(entry_run, "max_hp")
    outcome = rollout.get("outcome")
    final_hp = (
        0 if outcome == "model_error" else _mapping_integer(final_run, "current_hp")
    )
    turns = max(
        _mapping_integer(step.get("before_state", {}), "turn")
        for step in steps
        if isinstance(step, Mapping)
    )
    potions = entry_run.get("potions")
    potions_entry = sum(
        isinstance(potion, Mapping)
        and (potion.get("occupied") is True or potion.get("potion_id") is not None)
        for potion in potions or ()
    )
    potions_used = sum(
        isinstance(step, Mapping)
        and str(step.get("action", "")).startswith("ACTION: use_potion ")
        for step in steps
    )
    reward = score_battle_reward(
        BattleRewardInput(
            entry_hp=entry_hp,
            max_hp=max_hp,
            final_hp=final_hp,
            turns=turns,
            cleared=outcome == "cleared",
            died=outcome == "died",
            model_error=outcome == "model_error",
            potions_entry=potions_entry,
            potions_used=potions_used,
        ),
        scheme=scheme,
        potion_cost=potion_cost,
    )
    return reward.total


def _required_text(value: Mapping[object, object], field: str) -> str:
    """读取 group JSON 的非空字符串字段。

    Args:
        value (Mapping[object, object]): 当前 JSON 对象。
        field (str): 字段名。

    Raises:
        GrpoTrainingError: 字段缺失或为空。

    Returns:
        str: 校验后的字符串。
    """
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise GrpoTrainingError(f"GRPO 字段必须是非空字符串: {field}")
    return item


def _required_integer(value: object, field: str) -> int:
    """读取 group JSON 的非负整数字段。

    Args:
        value (object): 当前 JSON 对象。
        field (str): 字段名。

    Raises:
        GrpoTrainingError: 对象或字段不是非负整数。

    Returns:
        int: 校验后的整数。
    """
    if not isinstance(value, Mapping):
        raise GrpoTrainingError("GRPO JSON 项必须是对象")
    item = value.get(field)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise GrpoTrainingError(f"GRPO 字段必须是非负整数: {field}")
    return item


def _mapping_integer(value: object, field: str) -> int:
    """读取状态映射中的非负整数。

    Args:
        value (object): 状态子对象。
        field (str): 字段名。

    Raises:
        GrpoTrainingError: 对象或字段无效。

    Returns:
        int: 校验后的整数。
    """
    return _required_integer(value, field)
