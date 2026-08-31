"""验证 terminal Tree learner 的可变分支数与重要性校正收据。"""

import json
from pathlib import Path

import pytest


class PromptTokenizer:
    """为三个事件动作提供固定 token 编解码。"""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        return_dict: bool,
    ) -> list[int]:
        """返回固定 prompt token。

        Args:
            messages (list[dict[str, str]]): stateless 消息。
            tokenize (bool): 是否 token 化。
            add_generation_prompt (bool): 是否追加 assistant 前缀。
            enable_thinking (bool): 是否启用思考。
            return_dict (bool): 是否返回映射。

        Returns:
            list[int]: 固定 prompt token。
        """
        assert [message["role"] for message in messages] == ["system", "user"]
        assert tokenize and add_generation_prompt and not enable_thinking
        assert return_dict is False
        return [10, 11]

    def decode(
        self,
        token_ids: tuple[int, ...],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        """把 completion 还原为事件动作。

        Args:
            token_ids (tuple[int, ...]): completion token。
            skip_special_tokens (bool): 是否移除特殊 token。
            clean_up_tokenization_spaces (bool): 是否清理空格。

        Returns:
            str: 规范事件动作。
        """
        assert skip_special_tokens and not clean_up_tokenization_spaces
        return f"ACTION: choose_event_option {token_ids[0] - 101}"


class ChoiceMasker:
    """返回三个事件动作的 grammar 支持集。"""

    def allowed_token_ids(
        self,
        choices: tuple[str, ...],
        completion_ids: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        """允许三个动作首 token。

        Args:
            choices (tuple[str, ...]): 完整事件动作。
            completion_ids (tuple[int, ...]): 实际 token。

        Returns:
            tuple[tuple[int, ...], ...]: grammar 支持集。
        """
        assert len(choices) == 3
        assert len(completion_ids) == 1
        return ((101, 102, 103),)


def test_terminal_tree_loader_marks_forced_root_for_anchor_recompute(
    tmp_path: Path,
) -> None:
    """分层强制根动作应保留 q，并要求优化器重算冻结根概率。

    Args:
        tmp_path (Path): 临时 group 路径。

    Returns:
        None: 三条 arm 都只训练入口计划且携带 ``q=1/3``。
    """
    from play_sts2.training.rl.treegpo import load_terminal_tree_training_group

    group = load_terminal_tree_training_group(
        _write_group(tmp_path),
        PromptTokenizer(),
        max_length=32,
        choice_masker=ChoiceMasker(),
    )

    assert len(group.arms) == 3
    assert all(len(arm.sequences) == 1 for arm in group.arms)
    assert all(arm.recompute_root_probability for arm in group.arms)
    assert all(arm.proposal_probability == pytest.approx(1 / 3) for arm in group.arms)
    assert group.advantages == pytest.approx((-1.0, 0.0, 1.0))


def test_terminal_tree_loader_rejects_saved_advantage_drift(tmp_path: Path) -> None:
    """A100 learner 必须从 terminal return 重算未标准化 branch advantage。

    Args:
        tmp_path (Path): 临时 group 路径。

    Returns:
        None: collector 伪造的 advantage 不会直接进入梯度。
    """
    from play_sts2.training.rl.learner import GrpoTrainingError
    from play_sts2.training.rl.treegpo import load_terminal_tree_training_group

    path = _write_group(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["advantages"][2] = 9.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GrpoTrainingError, match="重算"):
        load_terminal_tree_training_group(
            path,
            PromptTokenizer(),
            max_length=32,
            choice_masker=ChoiceMasker(),
        )


def _write_group(tmp_path: Path) -> Path:
    """写入三分支均匀 proposal 的 terminal Tree 文件。

    Args:
        tmp_path (Path): 输出目录。

    Returns:
        Path: group JSON 路径。
    """
    choices = [f"ACTION: choose_event_option {index}" for index in range(3)]
    branches = []
    for index, value in enumerate((0.0, 1.0, 2.0)):
        action = choices[index]
        branches.append(
            {
                "arm_index": index,
                "worker_id": f"worker-{index}",
                "strategy_policy_version": "qwen3.5-s0",
                "battle_policy_version": "qwen3.5-b0",
                "generation_profile": {
                    "max_tokens": 128,
                    "temperature": 0.8,
                    "max_retries": 0,
                    "thinking_enabled": False,
                },
                "plan_id": action,
                "plan_step_count": 1,
                "steps": [
                    {
                        "index": 0,
                        "messages": [
                            {"role": "system", "content": "战略系统"},
                            {"role": "user", "content": "事件入口"},
                        ],
                        "reply_text": action,
                        "action": action,
                        "token_ids": [101 + index],
                        "behavior_logprobs": [0.0],
                        "response_choices": choices,
                        "finish_reason": "stop",
                    },
                    {
                        "index": 1,
                        "messages": [
                            {"role": "system", "content": "战略系统"},
                            {"role": "user", "content": "后续地图"},
                        ],
                        "reply_text": "ACTION: choose_event_option 0",
                        "action": "ACTION: choose_event_option 0",
                        "token_ids": [101],
                        "behavior_logprobs": [-0.2],
                        "response_choices": choices,
                        "finish_reason": "stop",
                    },
                ],
                "horizon_reason": "victory" if index == 2 else "died",
                "continuation_return": {
                    "scheme": "engineering_terminal_milestone",
                    "total": value,
                    "components": [],
                },
                "proposal_probability": 1 / 3,
                "recompute_root_probability": True,
                "elapsed_seconds": 1.0,
            }
        )
    payload = {
        "format": "terminal_tree_group",
        "group_id": "tree-terminal",
        "checkpoint_kind": "event",
        "checkpoint_option_ids": ["event:0", "event:1", "event:2"],
        "checkpoint_policy_text": "事件入口",
        "sampling_mode": "stratified",
        "strategy_policy_version": "qwen3.5-s0",
        "battle_policy_version": "qwen3.5-b0",
        "behavior_logprobs_mode": "processed_logprobs",
        "action_constraint_mode": "vllm_structured_choice",
        "environment": {
            "game_version": "v0.111.0",
            "mod_version": "mod-test",
            "protocol_version": "protocol-test",
            "structured_output_backend": "xgrammar",
            "structured_output_version": "0.1.33",
        },
        "branches": branches,
        "returns": [0.0, 1.0, 2.0],
        "advantages": [-1.0, 0.0, 1.0],
    }
    path = tmp_path / "terminal-tree.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
