"""从原生宏 checkpoint 收集分层 terminal Tree group。"""

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ....checkpoint import load_strategic_checkpoint
from ..contracts import BEHAVIOR_LOGPROBS_MODE, STRUCTURED_OUTPUT_BACKEND
from ..entrypoint import validate_rl_game_health
from ..strategy import GameTreeRolloutWorker
from .collector import TerminalTreeCollector
from .io import write_terminal_tree_group


def collect_terminal_tree_group(
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
    max_tokens: int = 128,
    temperature: float = 0.8,
    max_model_steps: int = 400,
) -> dict[str, Any]:
    """从一个真实 checkpoint 采集二至四条 terminal branches。

    只有二至四岔 MAP 逐动作分层强制；事件、奖励、火堆、商店等组合宏从冻结战略
    policy 独立采四条完整计划。所有 continuation 使用同一个冻结战斗 residual。

    Args:
        checkpoint_path (Path): backbone 发布的原生 checkpoint 目录。
        executable (Path): v0.111.0 游戏可执行文件。
        profile (Path): 关闭 Steam 与共享存档的 profile。
        home_root (Path): terminal branches 的隔离 HOME 父目录。
        ports (tuple[int, ...]): 阶段七允许的一至两个不同本地端口。
        strategy_model_url (str): A100 战略推理地址。
        battle_model_url (str): A100 战斗推理地址。
        strategy_policy_model (str): 冻结战略 residual 名。
        battle_policy_model (str): 冻结战斗 residual 名。
        vllm_logprobs_mode (str): 必须为 processed 行为概率。
        structured_output_backend (str): 必须为 xgrammar。
        structured_output_version (str): 服务端 xgrammar 包版本。
        group_id (str): 当前 terminal Tree group 名称。
        output_path (Path): 不含隐藏审计的 group JSON。
        max_tokens (int): 单次回复 token 预算。
        temperature (float): 冻结采样温度。
        max_model_steps (int): 每条 terminal suffix 的战略模型动作总上限。

    Raises:
        ValueError: 拓扑、策略或服务端概率合同无效。

    Returns:
        dict[str, Any]: proposal、分支回报、墙钟和输出摘要。
    """
    if not 1 <= len(ports) <= 2 or len(set(ports)) != len(ports):
        raise ValueError("阶段七 terminal Tree 只允许一至两个本地游戏端口")
    if vllm_logprobs_mode != BEHAVIOR_LOGPROBS_MODE:
        raise ValueError("terminal Tree 必须使用 processed_logprobs")
    if structured_output_backend != STRUCTURED_OUTPUT_BACKEND:
        raise ValueError("terminal Tree 必须使用 xgrammar")
    if not all(
        (
            structured_output_version,
            group_id,
            strategy_policy_model,
            battle_policy_model,
        )
    ):
        raise ValueError("terminal Tree policy、版本和 group_id 必须有效")
    checkpoint = load_strategic_checkpoint(checkpoint_path)
    kind = {
        "MAP": "map",
        "REWARD": "card_reward",
        "CARD_SELECTION": "card_selection",
        "EVENT": "event",
        "REST": "rest",
        "SHOP": "shop",
    }[checkpoint.entry.screen]
    root_actions = checkpoint.entry.legal_actions
    sampling_mode = _terminal_sampling_mode(kind, root_actions)
    simple = sampling_mode == "stratified"
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
            max_macro_checkpoints=1,
            continue_to_terminal=True,
            max_model_steps=max_model_steps,
        )
        for index, port in enumerate(ports)
    )
    group = TerminalTreeCollector(workers).collect(
        group_id=group_id,
        checkpoint_kind=kind,
        checkpoint_policy_text=checkpoint.entry.policy_text,
        root_actions=root_actions if simple else (),
        sampling_mode=sampling_mode,
    )
    healths = [
        worker.last_health for worker in workers if worker.last_health is not None
    ]
    environment = {
        **validate_rl_game_health(healths),
        "structured_output_backend": structured_output_backend,
        "structured_output_version": structured_output_version,
    }
    output = write_terminal_tree_group(group, output_path, environment=environment)
    return {
        "group_id": group.group_id,
        "sampling_mode": group.sampling_mode,
        "branches": len(group.branches),
        "strategy_policy_version": group.branches[0].strategy_policy_version,
        "battle_policy_version": group.branches[0].battle_policy_version,
        "returns": list(group.returns),
        "advantages": list(group.advantages),
        "proposal_probabilities": [
            branch.proposal_probability for branch in group.branches
        ],
        "elapsed_seconds": sum(branch.elapsed_seconds for branch in group.branches),
        "environment": environment,
        "generation_profile": asdict(group.branches[0].generation_profile),
        "output": str(output),
    }


def _terminal_sampling_mode(kind: str, root_actions: tuple[str, ...]) -> str:
    """区分单步地图根动作与需要完整计划采样的组合宏。

    Args:
        kind (str): checkpoint 类型。
        root_actions (tuple[str, ...]): 入口完整合法动作行。

    Returns:
        str: 地图二至四选一使用分层 proposal，其余使用 policy iid 计划。
    """
    if kind == "map" and 2 <= len(root_actions) <= 4:
        return "stratified"
    return "policy_iid"
