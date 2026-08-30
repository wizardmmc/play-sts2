"""验证 Tree collector 在启动游戏前拒绝危险拓扑。"""

from pathlib import Path

import pytest


def _arguments() -> dict[str, object]:
    """返回不触碰文件系统的最小 Tree collector 参数。

    Returns:
        dict[str, object]: 测试按需覆盖的入口参数。
    """
    return {
        "checkpoint_path": Path("missing-checkpoint"),
        "executable": Path("missing-game"),
        "profile": Path("missing-profile"),
        "home_root": Path("missing-homes"),
        "ports": (8080, 8081),
        "strategy_model_url": "http://127.0.0.1:8900",
        "battle_model_url": "http://127.0.0.1:8900",
        "strategy_policy_model": "e6-feasibility",
        "battle_policy_model": "e6-feasibility",
        "vllm_logprobs_mode": "processed_logprobs",
        "structured_output_backend": "xgrammar",
        "structured_output_version": "0.1.33",
        "group_id": "tree-test",
        "output_path": Path("missing-output.json"),
    }


def test_tree_entrypoint_requires_exactly_two_distinct_local_ports() -> None:
    """本轮可行性测试必须使用两个真实且不同的本地游戏端口。

    Returns:
        None: 重复端口在读取 checkpoint 前被拒绝。
    """
    from play_sts2.training.rl.strategy.entrypoint import collect_tree_rollout_group

    arguments = _arguments()
    arguments["ports"] = (8080, 8080)

    with pytest.raises(ValueError, match="两个不同的本地游戏端口"):
        collect_tree_rollout_group(**arguments)


def test_tree_entrypoint_rejects_raw_behavior_logprobs() -> None:
    """Tree importance ratio 只能使用 vLLM processed log-prob。

    Returns:
        None: raw log-prob 在启动游戏前被拒绝。
    """
    from play_sts2.training.rl.strategy.entrypoint import collect_tree_rollout_group

    arguments = _arguments()
    arguments["vllm_logprobs_mode"] = "raw_logprobs"

    with pytest.raises(ValueError, match="processed_logprobs"):
        collect_tree_rollout_group(**arguments)
