"""验证 Tree group 到通用 token-level GRPO learner 的投影。"""

import json
from pathlib import Path

import pytest


class PromptTokenizer:
    """为 Tree 测试返回固定 prompt 与 completion token。"""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        return_dict: bool,
    ) -> list[int]:
        """把 stateless 战略消息编码为固定 prompt token。

        Args:
            messages (list[dict[str, str]]): system/user 消息。
            tokenize (bool): 必须请求 token。
            add_generation_prompt (bool): 必须追加生成前缀。
            enable_thinking (bool): 工程 Tree 关闭 thinking。
            return_dict (bool): 必须直接返回 token 列表。

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
        """把测试 completion token 还原为地图动作。

        Args:
            token_ids (tuple[int, ...]): 两个动作 token。
            skip_special_tokens (bool): 必须移除特殊 token。
            clean_up_tokenization_spaces (bool): 禁止额外清理。

        Returns:
            str: 规范地图动作。
        """
        assert skip_special_tokens and not clean_up_tokenization_spaces
        return {
            (101, 102): "ACTION: choose_map_node 0",
            (103, 102): "ACTION: choose_map_node 1",
        }[tuple(token_ids)]


class ChoiceMasker:
    """返回两个地图动作的手工 xgrammar 支持集。"""

    def allowed_token_ids(
        self,
        choices: tuple[str, ...],
        completion_ids: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        """为首 token 分叉和确定尾 token 返回支持集。

        Args:
            choices (tuple[str, ...]): 完整地图动作候选。
            completion_ids (tuple[int, ...]): 当前实际动作 token。

        Returns:
            tuple[tuple[int, ...], ...]: 每步允许 token。
        """
        assert choices == (
            "ACTION: choose_map_node 0",
            "ACTION: choose_map_node 1",
        )
        assert completion_ids in {(101, 102), (103, 102)}
        return ((101, 103), (102,))


def test_load_tree_group_builds_generic_grpo_sequences(tmp_path: Path) -> None:
    """Tree learner 应复用同一 token-level PPO/KL 结构且只训练战略步骤。

    Args:
        tmp_path (Path): 临时 Tree group 路径。

    Returns:
        None: 八臂 return、advantage、mask 和 behavior log-prob 均正确。
    """
    from play_sts2.training.rl.strategy.learner import load_tree_training_group

    path = _write_tree_group(tmp_path)

    group = load_tree_training_group(
        path,
        PromptTokenizer(),
        max_length=32,
        choice_masker=ChoiceMasker(),
    )

    assert group.policy_version == "e6-feasibility"
    assert group.reward_scheme == "engineering_terminal_milestone"
    assert group.rewards == pytest.approx((1.0,) * 4 + (2.0,) * 4)
    assert group.advantages == pytest.approx((-1.0,) * 4 + (1.0,) * 4)
    first = group.arms[0].sequences[0]
    assert first.input_ids == (10, 11, 101, 102)
    assert first.assistant_mask == (False, False, True, True)
    assert first.behavior_logprobs == (-0.2, -0.1)
    assert first.allowed_token_ids == ((101, 103), (102,))


def test_tree_loader_rejects_hidden_checkpoint_audit(tmp_path: Path) -> None:
    """隐藏 RNG 审计不得进入发往 A100 learner 的 Tree group。

    Args:
        tmp_path (Path): 临时 Tree group 路径。

    Returns:
        None: checkpoint 中出现 audit 字段时 loader 明确拒绝。
    """
    from play_sts2.training.rl.learner import GrpoTrainingError
    from play_sts2.training.rl.strategy.learner import load_tree_training_group

    path = _write_tree_group(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checkpoint"]["audit"] = {"run_rng": "hidden"}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GrpoTrainingError, match="隐藏审计"):
        load_tree_training_group(
            path,
            PromptTokenizer(),
            max_length=32,
            choice_masker=ChoiceMasker(),
        )


def test_tree_loader_rechecks_same_visible_entry(tmp_path: Path) -> None:
    """learner 不应只相信 collector 声明的同 checkpoint。

    Args:
        tmp_path (Path): 临时 Tree group 路径。

    Returns:
        None: 任一 arm 的首步玩家消息漂移都会被拒绝。
    """
    from play_sts2.training.rl.learner import GrpoTrainingError
    from play_sts2.training.rl.strategy.learner import load_tree_training_group

    path = _write_tree_group(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["arms"][7]["steps"][0]["messages"][1]["content"] = "另一个地图入口"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GrpoTrainingError, match="玩家可见入口"):
        load_tree_training_group(
            path,
            PromptTokenizer(),
            max_length=32,
            choice_masker=ChoiceMasker(),
        )


def test_tree_loader_allows_unique_successors_for_large_action_domain(
    tmp_path: Path,
) -> None:
    """大动作域由冻结 policy 自然采样时不强求重复每个宏计划。

    Args:
        tmp_path (Path): 临时 Tree group 路径。

    Returns:
        None: 超过四个入口候选且有多个语义后继时可以进入 learner。
    """
    from play_sts2.training.rl.strategy.learner import load_tree_training_group

    path = _write_tree_group(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checkpoint"]["option_ids"] = [f"shop:{index}" for index in range(5)]
    for index, arm in enumerate(payload["arms"]):
        arm["macro_successor_text"] = f"unique-successor-{index}"
    payload["successor_counts"] = [[arm["plan_id"], 1] for arm in payload["arms"]]
    payload["successor_returns"] = [
        [arm["plan_id"], arm["continuation_return"]["total"]] for arm in payload["arms"]
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    group = load_tree_training_group(
        path,
        PromptTokenizer(),
        max_length=32,
        choice_masker=ChoiceMasker(),
    )

    assert len(group.arms) == 8


def test_tree_loader_rejects_horizon_drift(tmp_path: Path) -> None:
    """A100 learner 应复验每条 arm 的 horizon 收据。

    Args:
        tmp_path (Path): 临时 Tree group 路径。

    Returns:
        None: 一个 arm 使用不同后继宏节点预算时被拒绝。
    """
    from play_sts2.training.rl.learner import GrpoTrainingError
    from play_sts2.training.rl.strategy.learner import load_tree_training_group

    path = _write_tree_group(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["arms"][7]["max_macro_checkpoints"] = 3
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GrpoTrainingError, match="horizon"):
        load_tree_training_group(
            path,
            PromptTokenizer(),
            max_length=32,
            choice_masker=ChoiceMasker(),
        )


def test_tree_loader_recomputes_plan_mean_advantage(tmp_path: Path) -> None:
    """learner 应复算语义计划均值而非使用每条 suffix 的幸运回报。

    Args:
        tmp_path (Path): 临时 Tree group 路径。

    Returns:
        None: 同计划四条 arm 得到相同 advantage。
    """
    from play_sts2.training.rl.strategy.learner import load_tree_training_group

    path = _write_tree_group(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = [0.0, 2.0, 0.0, 2.0, 2.0, 4.0, 2.0, 4.0]
    for arm, value in zip(payload["arms"], values, strict=True):
        arm["continuation_return"]["total"] = value
    payload["returns"] = values
    payload["advantages"] = [-1.0] * 4 + [1.0] * 4
    payload["successor_returns"] = [
        ["ACTION: choose_map_node 0", 1.0],
        ["ACTION: choose_map_node 1", 3.0],
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    group = load_tree_training_group(
        path,
        PromptTokenizer(),
        max_length=32,
        choice_masker=ChoiceMasker(),
    )

    assert group.advantages == pytest.approx((-1.0,) * 4 + (1.0,) * 4)


def test_tree_loader_trains_only_entry_macro_plan_steps(tmp_path: Path) -> None:
    """branch advantage 不应广播给宏计划结束后的 continuation 动作。

    Args:
        tmp_path (Path): 临时 Tree group 路径。

    Returns:
        None: 原始 arm 有两个步骤但 learner 只保留 plan_step_count 个序列。
    """
    from play_sts2.training.rl.strategy.learner import load_tree_training_group

    path = _write_tree_group(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for arm in payload["arms"]:
        continuation = dict(arm["steps"][0])
        continuation["index"] = 1
        arm["steps"].append(continuation)
    path.write_text(json.dumps(payload), encoding="utf-8")

    group = load_tree_training_group(
        path,
        PromptTokenizer(),
        max_length=32,
        choice_masker=ChoiceMasker(),
    )

    assert all(len(arm.sequences) == 1 for arm in group.arms)


def _write_tree_group(tmp_path: Path) -> Path:
    """写入两个计划各四条 suffix 的完整 Tree JSON。

    Args:
        tmp_path (Path): 输出父目录。

    Returns:
        Path: Tree group 文件。
    """
    arms = []
    for index in range(8):
        first = index < 4
        action = "ACTION: choose_map_node 0" if first else "ACTION: choose_map_node 1"
        arms.append(
            {
                "arm_index": index,
                "worker_id": f"worker-{index % 2}",
                "strategy_policy_version": "e6-feasibility",
                "battle_policy_version": "e6-feasibility",
                "behavior_logprobs_mode": "processed_logprobs",
                "action_constraint_mode": "vllm_structured_choice",
                "generation_profile": {
                    "max_tokens": 128,
                    "temperature": 0.8,
                    "max_retries": 0,
                    "thinking_enabled": False,
                },
                "max_macro_checkpoints": 2,
                "plan_id": action,
                "plan_step_count": 1,
                "macro_successor_text": f"successor:{action}",
                "horizon_reason": "next_macro_checkpoint",
                "continuation_return": {
                    "scheme": "engineering_terminal_milestone",
                    "total": 1.0 if first else 2.0,
                    "components": [],
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
                        "token_ids": [101, 102] if first else [103, 102],
                        "behavior_logprobs": [-0.2, -0.1],
                        "response_choices": [
                            "ACTION: choose_map_node 0",
                            "ACTION: choose_map_node 1",
                        ],
                        "finish_reason": "stop",
                    }
                ],
            }
        )
    path = tmp_path / "tree-group.json"
    path.write_text(
        json.dumps(
            {
                "format": "tree_grpo_group",
                "group_id": "tree-test",
                "checkpoint": {
                    "kind": "map",
                    "screen": "MAP",
                    "option_ids": ["map:2:1", "map:2:3"],
                    "policy_text": "地图入口",
                },
                "strategy_policy_version": "e6-feasibility",
                "battle_policy_version": "e6-feasibility",
                "behavior_logprobs_mode": "processed_logprobs",
                "action_constraint_mode": "vllm_structured_choice",
                "generation_profile": {
                    "max_tokens": 128,
                    "temperature": 0.8,
                    "max_retries": 0,
                    "thinking_enabled": False,
                },
                "max_macro_checkpoints": 2,
                "environment": {
                    "game_version": "v0.111.0",
                    "mod_version": "mod-test",
                    "protocol_version": "protocol-test",
                    "structured_output_backend": "xgrammar",
                    "structured_output_version": "0.1.33",
                },
                "arms": arms,
                "returns": [1.0] * 4 + [2.0] * 4,
                "advantages": [-1.0] * 4 + [1.0] * 4,
                "successor_counts": [
                    ["ACTION: choose_map_node 0", 4],
                    ["ACTION: choose_map_node 1", 4],
                ],
                "successor_returns": [
                    ["ACTION: choose_map_node 0", 1.0],
                    ["ACTION: choose_map_node 1", 2.0],
                ],
            }
        ),
        encoding="utf-8",
    )
    return path
