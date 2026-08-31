"""验证 GiGPO 文件只把整局优势广播到本局战略 token。"""

import json
from pathlib import Path

import pytest


class PromptTokenizer:
    """为测试战略动作提供固定 chat template token。"""

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
            messages (list[dict[str, str]]): stateless system/user 消息。
            tokenize (bool): 是否返回 token。
            add_generation_prompt (bool): 是否追加生成前缀。
            enable_thinking (bool): 是否启用思考。
            return_dict (bool): 是否返回映射。

        Returns:
            list[int]: 两个固定 prompt token。
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
        """把动作 token 还原为规范动作。

        Args:
            token_ids (tuple[int, ...]): completion token。
            skip_special_tokens (bool): 是否移除特殊 token。
            clean_up_tokenization_spaces (bool): 是否清理空格。

        Returns:
            str: 规范地图动作。
        """
        assert skip_special_tokens and not clean_up_tokenization_spaces
        return f"ACTION: choose_map_node {token_ids[0] - 100}"


class ChoiceMasker:
    """返回测试动作对应的完整 grammar 支持集。"""

    def allowed_token_ids(
        self,
        choices: tuple[str, ...],
        completion_ids: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        """允许八个手工动作 token。

        Args:
            choices (tuple[str, ...]): 全部动作行。
            completion_ids (tuple[int, ...]): 实际动作 token。

        Returns:
            tuple[tuple[int, ...], ...]: 单步 grammar 支持集。
        """
        assert len(choices) == 8
        assert len(completion_ids) == 1
        return (tuple(range(100, 108)),)


def test_gigpo_loader_broadcasts_each_episode_advantage(tmp_path: Path) -> None:
    """每个 episode 的战略步骤只能收到该局的相对优势。

    Args:
        tmp_path (Path): 临时 group 文件目录。

    Returns:
        None: 八条 arm 的单步 advantage 与文件重算一致。
    """
    from play_sts2.training.rl.gigpo import load_gigpo_training_group

    path = _write_group(tmp_path)
    group = load_gigpo_training_group(
        path,
        PromptTokenizer(),
        max_length=32,
        choice_masker=ChoiceMasker(),
    )

    assert len(group.arms) == 8
    assert group.policy_version == "qwen3.5-s0"
    expected = tuple(1.01 * (index - 3.5) for index in range(8))
    assert tuple(arm.advantage for arm in group.arms) == pytest.approx(expected)
    assert tuple(arm.sequences[0].advantage for arm in group.arms) == pytest.approx(
        expected
    )


def test_gigpo_loader_rejects_hidden_anchor_audit(tmp_path: Path) -> None:
    """传给 A100 的 GiGPO 文件不得携带原生隐藏审计。

    Args:
        tmp_path (Path): 临时 group 文件目录。

    Returns:
        None: 隐藏审计字段出现时 learner 明确拒绝。
    """
    from play_sts2.training.rl.gigpo import load_gigpo_training_group
    from play_sts2.training.rl.learner import GrpoTrainingError

    path = _write_group(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["anchor_audits"] = [{"rng": 1}]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GrpoTrainingError, match="隐藏审计"):
        load_gigpo_training_group(
            path,
            PromptTokenizer(),
            max_length=32,
            choice_masker=ChoiceMasker(),
        )


def test_gigpo_loader_recomputes_saved_components(tmp_path: Path) -> None:
    """learner 不能信任与 episode 终局事实不一致的派生优势数组。

    Args:
        tmp_path (Path): 临时 group 文件目录。

    Returns:
        None: 即使分量代数仍自洽，篡改的回报也会被拒绝。
    """
    from play_sts2.training.rl.gigpo import load_gigpo_training_group
    from play_sts2.training.rl.learner import GrpoTrainingError

    path = _write_group(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["progress_returns"][0] += 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GrpoTrainingError, match="progress return"):
        load_gigpo_training_group(
            path,
            PromptTokenizer(),
            max_length=32,
            choice_masker=ChoiceMasker(),
        )


def _write_group(tmp_path: Path) -> Path:
    """写入八局单步的最小 GiGPO 文件。

    Args:
        tmp_path (Path): 输出目录。

    Returns:
        Path: group JSON 路径。
    """
    progress = tuple(index + (index + 1) * 0.01 for index in range(8))
    mean_progress = sum(progress) / len(progress)
    advantages = tuple(value - mean_progress for value in progress)
    choices = [f"ACTION: choose_map_node {index}" for index in range(8)]
    episodes = []
    for index in range(8):
        action = choices[index]
        episodes.append(
            {
                "arm_index": index,
                "worker_id": f"worker-{index % 4}",
                "seed": "ABCDEF1234",
                "character_id": "DEFECT",
                "ascension": 0,
                "strategy_policy_version": "qwen3.5-s0",
                "battle_policy_version": "qwen3.5-b0",
                "generation_profile": {
                    "max_tokens": 128,
                    "temperature": 0.8,
                    "max_retries": 0,
                    "thinking_enabled": False,
                },
                "steps": [
                    {
                        "index": 0,
                        "messages": [
                            {"role": "system", "content": "战略系统"},
                            {"role": "user", "content": "地图入口"},
                        ],
                        "reply_text": action,
                        "action": action,
                        "token_ids": [100 + index],
                        "behavior_logprobs": [-0.2],
                        "response_choices": choices,
                        "finish_reason": "stop",
                        "anchor_id": None,
                    }
                ],
                "victory": False,
                "final_floor": index + 1,
                "final_hp": 0,
                "max_hp": 75,
                "bosses_cleared": index,
                "battle_count": 1,
                "elapsed_seconds": 1.0,
            }
        )
    payload = {
        "format": "gigpo_group",
        "group_id": "cycle-000-backbone",
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
        "episodes": episodes,
        "terminal_returns": [0.0] * 8,
        "progress_returns": list(progress),
        "episode_advantages": [0.0] * 8,
        "milestone_advantages": list(advantages),
        "advantages": list(advantages),
        "lambda_milestone": 1.0,
        "normalization": "one",
        "anchor_census": {
            "groups": 0,
            "repeated_steps": 0,
            "total_steps": 8,
            "coverage": 0.0,
        },
    }
    path = tmp_path / "gigpo.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
