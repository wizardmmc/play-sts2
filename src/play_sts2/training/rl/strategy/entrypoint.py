"""连接两个本地游戏与 A100 policy 收集一个 Tree K=8 group。"""

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ....checkpoint import load_strategic_checkpoint
from ..contracts import BEHAVIOR_LOGPROBS_MODE, STRUCTURED_OUTPUT_BACKEND
from ..entrypoint import validate_rl_game_health
from .collector import TreeGroupCollector
from .io import write_tree_rollout_group
from .worker import GameTreeRolloutWorker


def collect_tree_rollout_group(
    *,
    checkpoint_path: Path,
    executable: Path,
    profile: Path,
    home_root: Path,
    ports: tuple[int, ...],
    strategy_model_url: str,
    battle_model_url: str,
    strategy_policy_model: str,
    battle_policy_model: str,
    vllm_logprobs_mode: str,
    structured_output_backend: str,
    structured_output_version: str,
    group_id: str,
    output_path: Path,
    group_size: int = 8,
    max_tokens: int = 128,
    temperature: float = 0.8,
    max_macro_checkpoints: int = 2,
) -> dict[str, Any]:
    """收集两个本地实例共享的冻结 Tree suffix group。

    Args:
        checkpoint_path (Path): 第五阶段发布的原生 checkpoint 目录。
        executable (Path): v0.111.0 游戏可执行文件。
        profile (Path): 关闭 Steam 和共享存档的 profile。
        home_root (Path): 当前 group 全部 arm HOME 的父目录。
        ports (tuple[int, ...]): 本轮必须恰好两个不同本地端口。
        strategy_model_url (str): A100 战略推理地址。
        battle_model_url (str): A100 战斗推理地址。
        strategy_policy_model (str): 冻结战略模型名。
        battle_policy_model (str): 冻结战斗模型名。
        vllm_logprobs_mode (str): 必须为 processed 行为概率。
        structured_output_backend (str): 必须为 xgrammar。
        structured_output_version (str): 服务端 xgrammar 包版本。
        group_id (str): 当前 Tree group 标识。
        output_path (Path): 不含隐藏审计的 group JSON。
        group_size (int): 固定总预算 K=8。
        max_tokens (int): 单次战略/战斗回复 token 预算。
        temperature (float): 冻结 policy 采样温度。
        max_macro_checkpoints (int): suffix 后继宏节点 horizon。

    Raises:
        ValueError: 拓扑、行为概率、结构化后端或 policy 名无效。
        TreeGroupRejected: 八臂计划覆盖或 return 方差不满足准入。

    Returns:
        dict[str, Any]: group、计划覆盖、return 与输出路径摘要。
    """
    if len(ports) != 2 or len(set(ports)) != 2:
        raise ValueError("Tree 可行性测试要求两个不同的本地游戏端口")
    if vllm_logprobs_mode != BEHAVIOR_LOGPROBS_MODE:
        raise ValueError("Tree collector 必须使用 processed_logprobs")
    if structured_output_backend != STRUCTURED_OUTPUT_BACKEND:
        raise ValueError("Tree collector 必须使用 xgrammar")
    if (
        group_size != 8
        or not strategy_policy_model
        or not battle_policy_model
        or not structured_output_version
        or not group_id
    ):
        raise ValueError("Tree collector 的 K、policy、版本和 group_id 必须有效")

    checkpoint = load_strategic_checkpoint(checkpoint_path)
    kind = {
        "MAP": "map",
        "REWARD": "card_reward",
        "CARD_SELECTION": "card_selection",
        "EVENT": "event",
        "REST": "rest",
        "SHOP": "shop",
    }[checkpoint.entry.screen]
    workers = tuple(
        GameTreeRolloutWorker(
            worker_id=f"worker-{index}",
            executable=executable,
            profile=profile,
            home_root=home_root / f"worker-{index}",
            port=port,
            checkpoint=checkpoint,
            strategy_model_url=strategy_model_url,
            battle_model_url=battle_model_url,
            strategy_policy_version=strategy_policy_model,
            battle_policy_version=battle_policy_model,
            max_tokens=max_tokens,
            temperature=temperature,
            max_macro_checkpoints=max_macro_checkpoints,
        )
        for index, port in enumerate(ports)
    )
    group = TreeGroupCollector(workers, group_size=group_size).collect(
        group_id=group_id,
        checkpoint_kind=kind,
        checkpoint_option_ids=checkpoint.entry.option_ids,
        checkpoint_policy_text=checkpoint.entry.policy_text,
    )
    healths = [worker.last_health for worker in workers]
    if any(health is None for health in healths):
        raise ValueError("Tree collector 缺少本地游戏运行时收据")
    environment = {
        **validate_rl_game_health([health for health in healths if health is not None]),
        "structured_output_backend": structured_output_backend,
        "structured_output_version": structured_output_version,
    }
    output = write_tree_rollout_group(
        group,
        output_path,
        environment=environment,
    )
    return {
        "group_id": group.group_id,
        "strategy_policy_version": group.strategy_policy_version,
        "battle_policy_version": group.battle_policy_version,
        "generation_profile": asdict(group.generation_profile),
        "max_macro_checkpoints": group.max_macro_checkpoints,
        "arms": len(group.arms),
        "plan_counts": list(group.plan_counts),
        "successor_counts": list(group.successor_counts),
        "return_mean": sum(group.returns) / len(group.returns),
        "return_min": min(group.returns),
        "return_max": max(group.returns),
        "environment": environment,
        "output": str(output),
    }
