"""验证 GRPO 训练批、真实 behavior ratio 与冻结锚 KL。"""

import json
import math
from pathlib import Path

import pytest
import torch

from play_sts2.training import rl


class _PromptTokenizer:
    """为 rollout 消息返回固定生成前缀。"""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        return_dict: bool,
    ) -> list[int]:
        """核对 stateless 消息并返回手工前缀 token。

        Args:
            messages (list[dict[str, str]]): 当前单步 system/user 消息。
            tokenize (bool): 是否返回 token ID。
            add_generation_prompt (bool): 是否追加 assistant 起始标记。
            enable_thinking (bool): 是否启用 thinking。
            return_dict (bool): 是否返回带键的编码对象。

        Returns:
            list[int]: 固定的两个 prompt token。
        """
        roles = [message["role"] for message in messages]
        assert roles in (["system", "user"], ["system", "user", "assistant"])
        assert tokenize is True
        assert enable_thinking is False
        assert return_dict is False
        if roles[-1] != "assistant":
            assert add_generation_prompt is True
            return [10, 11]
        assert add_generation_prompt is False
        action = messages[-1]["content"]
        suffix = {
            "ACTION: play_card 0": [101, 102],
            "ACTION: end_turn": [103, 102],
        }[action]
        return [10, 11, *suffix]

    def decode(
        self,
        token_ids: tuple[int, ...],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        """把手工 completion token 还原为规范动作。

        Args:
            token_ids (tuple[int, ...]): 当前动作 token。
            skip_special_tokens (bool): 是否移除终止标记。
            clean_up_tokenization_spaces (bool): 是否清理空格。

        Returns:
            str: 与测试 JSON reply_text 一致的动作。
        """
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return {
            (101, 102): "ACTION: play_card 0",
            (103, 102): "ACTION: end_turn",
        }[tuple(token_ids)]


class _ChoiceMasker:
    """为测试 token 返回手工 structured-choice 支持集。"""

    def allowed_token_ids(
        self,
        choices: tuple[str, ...],
        completion_ids: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        """返回分叉首 token 与确定终止 token 的允许集合。

        Args:
            choices (tuple[str, ...]): 两个规范动作候选。
            completion_ids (tuple[int, ...]): 当前实际生成 token。

        Returns:
            tuple[tuple[int, ...], ...]: 与 completion 等长的允许 token。
        """
        assert choices == ("ACTION: play_card 0", "ACTION: end_turn")
        assert completion_ids in {(101, 102), (103, 102)}
        return ((101, 103), (102,))


def test_load_grpo_group_builds_stateless_assistant_token_sequences(
    tmp_path: Path,
) -> None:
    """rollout token 必须直接接在每步独立 prompt 后并保存真实旧概率。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Returns:
        None: 此测试固定训练张量的数据来源和掩码边界。
    """
    path = _write_group(tmp_path)

    group = rl.load_grpo_training_group(
        path,
        _PromptTokenizer(),
        reward_scheme="core",
        max_length=32,
        choice_masker=_ChoiceMasker(),
    )

    assert group.policy_version == "policy-test"
    assert group.rewards == pytest.approx((2.95, -9.05) * 4)
    assert group.advantages == pytest.approx((1.0, -1.0) * 4)
    first = group.arms[0].sequences[0]
    assert first.input_ids == (10, 11, 101, 102)
    assert first.assistant_mask == (False, False, True, True)
    assert first.behavior_logprobs == (-0.2, -0.4)
    assert first.allowed_token_ids == ((101, 103), (102,))
    assert first.temperature == pytest.approx(0.8)
    assert first.advantage == pytest.approx(1.0)
    assert first.step_index == 0


def test_grpo_loss_uses_rollout_behavior_logprobs_and_anchor_kl() -> None:
    """PPO ratio 必须来自 rollout p_old，不能使用当前前向 detach 冒充。

    Returns:
        None: 此测试以手算 token 数值固定 clipped objective 和 KL。
    """
    new_logprobs = torch.tensor([-0.1, -0.3])
    old_logprobs = torch.tensor([-0.2, -0.2])
    anchor_logprobs = torch.tensor([-0.15, -0.25])

    result = rl.grpo_sequence_loss(
        new_logprobs,
        old_logprobs,
        anchor_logprobs,
        advantage=1.5,
        clip=0.2,
        kl_beta=0.02,
    )

    ratios = torch.exp(new_logprobs - old_logprobs)
    clipped = torch.clamp(ratios, 0.8, 1.2)
    objective = torch.minimum(ratios * 1.5, clipped * 1.5).mean()
    delta = anchor_logprobs - new_logprobs
    kl = (torch.exp(delta) - delta - 1).mean()
    assert result.ratio_mean == pytest.approx(float(ratios.mean()))
    assert result.policy_loss == pytest.approx(float(-objective))
    assert result.kl == pytest.approx(float(kl))
    assert result.total_loss == pytest.approx(float(-objective + 0.02 * kl))
    assert not math.isclose(result.ratio_mean, 1.0)


def test_stratified_tree_importance_uses_full_root_action_probability() -> None:
    """强制根动作的校正权重应为冻结策略动作概率除以 proposal。

    Returns:
        None: 三个 token 的条件概率先相乘，再除以均匀 ``q``。
    """
    logprobs = torch.log(torch.tensor([0.5, 0.8, 1.0]))

    weight = rl.stratified_importance_weight(logprobs, proposal_probability=0.25)

    assert weight == pytest.approx(1.6)


def test_stratified_tree_importance_rejects_invalid_proposal() -> None:
    """proposal 为零或冻结动作概率非有限时必须停止更新。

    Returns:
        None: 两类无效输入都不会静默截断成可训练权重。
    """
    with pytest.raises(ValueError, match="proposal"):
        rl.stratified_importance_weight(torch.tensor([-0.2]), proposal_probability=0.0)
    with pytest.raises(ValueError, match="log-prob"):
        rl.stratified_importance_weight(
            torch.tensor([float("nan")]), proposal_probability=0.5
        )


def test_load_grpo_group_rejects_arm_profile_drift(tmp_path: Path) -> None:
    """顶层标签正确也不能掩盖某条 arm 混入 raw log-prob 或不同入口。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Returns:
        None: learner 在模型加载前重新执行严格八臂准入。
    """
    path = _write_group(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rollouts"][3]["behavior_logprobs_mode"] = "raw_logprobs"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(rl.GrpoTrainingError, match="入口、策略或采样配置"):
        rl.load_grpo_training_group(
            path,
            _PromptTokenizer(),
            reward_scheme="core",
            max_length=32,
            choice_masker=_ChoiceMasker(),
        )


def test_load_grpo_group_rejects_failed_step_with_executed_action(
    tmp_path: Path,
) -> None:
    """模型失败 step 不能同时声称已经执行了非空规范动作。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Returns:
        None: learner 二次准入与 rollout contract 保持一致。
    """
    path = _write_group(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rollouts"][0]["steps"][0]["is_model_failure"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(rl.GrpoTrainingError, match="reply、action"):
        rl.load_grpo_training_group(
            path,
            _PromptTokenizer(),
            reward_scheme="core",
            max_length=32,
            choice_masker=_ChoiceMasker(),
        )


def test_constrained_logprobs_normalize_only_over_choice_trie() -> None:
    """新策略概率必须复现 structured choice 和采样温度的条件分布。

    Returns:
        None: 确认确定后缀概率为零，分叉 token 只在合法候选间归一。
    """
    logits = torch.zeros((2, 8), dtype=torch.float32)
    logits[0, 3] = 1.0
    logits[0, 4] = 3.0
    logits[0, 7] = 100.0
    logits[1, 5] = -10.0
    logits[1, 7] = 100.0

    actual = rl.constrained_completion_logprobs(
        logits,
        completion_ids=(3, 5),
        allowed_token_ids=((3, 4), (5,)),
        temperature=2.0,
    )

    expected_first = torch.log_softmax(torch.tensor([0.5, 1.5]), dim=0)[0]
    assert actual.tolist() == pytest.approx([float(expected_first), 0.0])


def test_compare_reward_schemes_keeps_arm_level_audit(tmp_path: Path) -> None:
    """奖励消融报告必须保留每个 arm 的回报与排序，不能只报组均值。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Returns:
        None: 报告覆盖主基线、无回合项、两档药水成本和旧负对照。
    """
    path = _write_group(tmp_path)

    report = rl.compare_battle_reward_schemes(
        (path,),
    )

    schemes = report["groups"][0]["schemes"]
    assert set(schemes) == {
        "core",
        "core_no_turn",
        "core_no_turn_boss_progress",
        "potion_cost_0.1",
        "potion_cost_0.25",
        "legacy_remaining_potion",
    }
    assert schemes["core"]["rewards"] == pytest.approx([2.95, -9.05] * 4)
    assert schemes["core"]["arm_order"] == [0, 2, 4, 6, 1, 3, 5, 7]
    assert report["selected_default"] == "core"


def _write_group(tmp_path: Path) -> Path:
    """写入两臂一动作的手算训练 group。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Returns:
        Path: 完整 group JSON 路径。
    """
    rollouts = []
    for arm_index, outcome in enumerate(("cleared", "died") * 4):
        final_hp = 40 if outcome == "cleared" else 0
        rollouts.append(
            {
                "arm_index": arm_index,
                "worker_id": f"worker-{arm_index}",
                "policy_version": "policy-test",
                "behavior_logprobs_mode": "processed_logprobs",
                "action_constraint_mode": "vllm_structured_choice",
                "generation_profile": {
                    "max_tokens": 128,
                    "temperature": 0.8,
                    "max_retries": 0,
                    "thinking_enabled": False,
                },
                "entry_snapshot": {"turn": 1, "model_input": "same"},
                "outcome": outcome,
                "final_state": {"run": {"current_hp": final_hp, "max_hp": 40}},
                "steps": [
                    {
                        "index": 0,
                        "before_state": {
                            "turn": 1,
                            "run": {
                                "current_hp": 40,
                                "max_hp": 40,
                                "potions": [],
                            },
                        },
                        "messages": [
                            {"role": "system", "content": "战斗系统"},
                            {"role": "user", "content": "战斗状态"},
                        ],
                        "action": (
                            "ACTION: play_card 0"
                            if outcome == "cleared"
                            else "ACTION: end_turn"
                        ),
                        "reply_text": (
                            "ACTION: play_card 0"
                            if outcome == "cleared"
                            else "ACTION: end_turn"
                        ),
                        "token_ids": (
                            [101, 102] if outcome == "cleared" else [103, 102]
                        ),
                        "behavior_logprobs": [-0.2, -0.4],
                        "response_choices": [
                            "ACTION: play_card 0",
                            "ACTION: end_turn",
                        ],
                        "finish_reason": "stop",
                    }
                ],
            }
        )
    path = tmp_path / "group.json"
    path.write_text(
        json.dumps(
            {
                "group_id": "group-test",
                "policy_version": "policy-test",
                "behavior_logprobs_mode": "processed_logprobs",
                "action_constraint_mode": "vllm_structured_choice",
                "generation_profile": {
                    "max_tokens": 128,
                    "temperature": 0.8,
                    "max_retries": 0,
                    "thinking_enabled": False,
                },
                "entry_snapshot": {"turn": 1, "model_input": "same"},
                "environment": {
                    "game_version": "v0.111.0",
                    "mod_version": "mod-test",
                    "protocol_version": "protocol-test",
                    "structured_output_backend": "xgrammar",
                    "structured_output_version": "0.1.33",
                },
                "rollouts": rollouts,
            }
        ),
        encoding="utf-8",
    )
    return path
