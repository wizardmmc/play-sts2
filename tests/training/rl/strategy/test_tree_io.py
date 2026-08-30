"""验证 Tree group JSON 只包含训练所需的玩家可见事实。"""

import json
from pathlib import Path


def test_write_tree_group_keeps_receipt_and_excludes_hidden_audit(
    tmp_path: Path,
) -> None:
    """落盘组应含 v0.111.0/xgrammar 收据但绝不复制 checkpoint audit。

    Args:
        tmp_path (Path): Tree group 输出目录。

    Returns:
        None: JSON 可供远端 learner 使用且不含隐藏 RNG。
    """
    from play_sts2.training.rl.strategy.contracts import build_tree_rollout_group
    from play_sts2.training.rl.strategy.io import write_tree_rollout_group

    group = build_tree_rollout_group(
        group_id="tree-io",
        checkpoint_kind="map",
        checkpoint_option_ids=("map:2:1", "map:2:3"),
        checkpoint_policy_text="地图入口",
        arms=tuple(
            _arm(
                index,
                "ACTION: choose_map_node 0"
                if index < 4
                else "ACTION: choose_map_node 1",
                1.0 if index < 4 else 2.0,
            )
            for index in range(8)
        ),
        expected_size=8,
    )
    output = tmp_path / "tree.json"

    write_tree_rollout_group(
        group,
        output,
        environment={
            "game_version": "v0.111.0",
            "mod_version": "mod-test",
            "protocol_version": "protocol-test",
            "structured_output_backend": "xgrammar",
            "structured_output_version": "0.1.33",
        },
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["format"] == "tree_grpo_group"
    assert payload["checkpoint"] == {
        "kind": "map",
        "screen": "MAP",
        "option_ids": ["map:2:1", "map:2:3"],
        "policy_text": "地图入口",
    }
    assert payload["environment"]["game_version"] == "v0.111.0"
    assert payload["behavior_logprobs_mode"] == "processed_logprobs"
    assert payload["action_constraint_mode"] == "vllm_structured_choice"
    assert payload["max_macro_checkpoints"] == 2
    assert "audit" not in output.read_text(encoding="utf-8")


def _arm(index: int, plan: str, value: float):
    """构造 Tree IO 使用的最小合法 arm。

    Args:
        index (int): arm 序号。
        plan (str): 首个地图动作。
        value (float): continuation return。

    Returns:
        TreeRolloutArm: 带完整 token 元数据的 arm。
    """
    from play_sts2.inference import ChatMessage
    from play_sts2.runtime import DecisionGenerationProfile
    from play_sts2.training.rl.strategy.contracts import (
        StrategicReturn,
        StrategicReturnComponent,
        TreeRolloutArm,
        TreeRolloutStep,
    )

    return TreeRolloutArm(
        arm_index=index,
        worker_id=f"worker-{index % 2}",
        strategy_policy_version="e6-feasibility",
        battle_policy_version="e6-feasibility",
        behavior_logprobs_mode="processed_logprobs",
        action_constraint_mode="vllm_structured_choice",
        generation_profile=DecisionGenerationProfile(128, 0.8, 0, False),
        max_macro_checkpoints=2,
        plan_id=plan,
        plan_step_count=1,
        macro_successor_text=f"successor:{plan}",
        steps=(
            TreeRolloutStep(
                index=0,
                messages=(
                    ChatMessage("system", "战略系统"),
                    ChatMessage("user", "地图入口"),
                ),
                reply_text=plan,
                action=plan,
                token_ids=(100 + index, 200),
                behavior_logprobs=(-0.2, -0.1),
                response_choices=(
                    "ACTION: choose_map_node 0",
                    "ACTION: choose_map_node 1",
                ),
                finish_reason="stop",
            ),
        ),
        final_state={"run": {"floor": 3}},
        horizon_reason="next_macro_checkpoint",
        continuation_return=StrategicReturn(
            "engineering_terminal_milestone",
            value,
            (StrategicReturnComponent("test", value),),
        ),
        elapsed_seconds=1.0,
    )
