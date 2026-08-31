"""连接本地无头游戏与 A100 双 residual 收集 GiGPO 八局组。"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..contracts import BEHAVIOR_LOGPROBS_MODE, STRUCTURED_OUTPUT_BACKEND
from ..entrypoint import validate_rl_game_health
from .collector import BackboneGroupCollector
from .io import write_gigpo_group
from .worker import GameBackboneWorker


def collect_gigpo_group(
    *,
    executable: Path,
    profile: Path,
    home_root: Path,
    checkpoint_root: Path,
    ports: tuple[int, ...],
    strategy_model_url: str,
    battle_model_url: str,
    strategy_policy_model: str,
    battle_policy_model: str,
    seed: str,
    character_id: str,
    ascension: int,
    vllm_logprobs_mode: str,
    structured_output_backend: str,
    structured_output_version: str,
    group_id: str,
    output_path: Path,
    candidates_path: Path,
    group_size: int = 8,
    max_tokens: int = 128,
    temperature: float = 0.8,
    lambda_milestone: float = 1.0,
    normalization: str = "one",
) -> dict[str, Any]:
    """收集同 seed 八条完整游戏并写出训练组和本地候选清单。

    Args:
        executable (Path): v0.111.0 游戏可执行文件。
        profile (Path): 关闭 Steam 与共享存档的 profile。
        home_root (Path): 全部隔离 HOME 父目录。
        checkpoint_root (Path): 原生宏 checkpoint 输出根。
        ports (tuple[int, ...]): 阶段七允许的一至两个不同本地端口。
        strategy_model_url (str): A100 战略推理地址。
        battle_model_url (str): A100 战斗推理地址。
        strategy_policy_model (str): 冻结战略 residual 名。
        battle_policy_model (str): 冻结战斗 residual 名。
        seed (str): 当前组共享游戏种子。
        character_id (str): 角色稳定 ID。
        ascension (int): 进阶等级。
        vllm_logprobs_mode (str): 必须为 processed 行为概率。
        structured_output_backend (str): 必须为 xgrammar。
        structured_output_version (str): 服务端 xgrammar 包版本。
        group_id (str): 当前 backbone group 名称。
        output_path (Path): 不含隐藏审计的 GiGPO JSON。
        candidates_path (Path): 只留在本地的 checkpoint 与战斗候选清单。
        group_size (int): 固定八条完整游戏。
        max_tokens (int): 单次回复 token 预算。
        temperature (float): 冻结采样温度。
        lambda_milestone (float): episode-level 进度课程权重。
        normalization (str): ``one`` 或 ``std``。

    Raises:
        ValueError: 拓扑、策略、行为概率或 structured output 无效。

    Returns:
        dict[str, Any]: outcome、anchor、候选数量与输出路径摘要。
    """
    if not 1 <= len(ports) <= 2 or len(set(ports)) != len(ports):
        raise ValueError("阶段七 GiGPO 只允许一至两个本地游戏端口")
    if vllm_logprobs_mode != BEHAVIOR_LOGPROBS_MODE:
        raise ValueError("GiGPO collector 必须使用 processed_logprobs")
    if structured_output_backend != STRUCTURED_OUTPUT_BACKEND:
        raise ValueError("GiGPO collector 必须使用 xgrammar")
    if (
        group_size != 8
        or not strategy_policy_model
        or not battle_policy_model
        or not structured_output_version
        or not group_id
    ):
        raise ValueError("GiGPO collector 的 K、policy、版本和 group_id 必须有效")
    workers = tuple(
        GameBackboneWorker(
            worker_id=f"worker-{index}",
            executable=executable,
            profile=profile,
            home_root=home_root / f"worker-{index}",
            checkpoint_root=checkpoint_root / f"worker-{index}",
            port=port,
            strategy_model_url=strategy_model_url,
            battle_model_url=battle_model_url,
            strategy_policy_version=strategy_policy_model,
            battle_policy_version=battle_policy_model,
            seed=seed,
            character_id=character_id,
            ascension=ascension,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for index, port in enumerate(ports)
    )
    collected = BackboneGroupCollector(workers, group_size=group_size).collect(
        group_id=group_id,
        lambda_milestone=lambda_milestone,
        normalization=normalization,
    )
    environment = {
        **validate_rl_game_health([draft.health for draft in collected.drafts]),
        "structured_output_backend": structured_output_backend,
        "structured_output_version": structured_output_version,
    }
    output = write_gigpo_group(collected.group, output_path, environment=environment)
    candidates = {
        "format": "stage7_candidates",
        "group_id": group_id,
        "strategy_policy_version": strategy_policy_model,
        "battle_policy_version": battle_policy_model,
        "checkpoints": [
            {
                "arm_index": draft.episode.arm_index,
                **asdict(candidate),
                "path": str(candidate.path) if candidate.path is not None else None,
            }
            for draft in collected.drafts
            for candidate in draft.checkpoints
        ],
        "battles": [
            {
                "arm_index": draft.episode.arm_index,
                **asdict(candidate),
            }
            for draft in collected.drafts
            for candidate in draft.battles
        ],
    }
    candidate_output = Path(candidates_path)
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    candidate_output.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "group_id": group_id,
        "episodes": len(collected.group.episodes),
        "victories": sum(episode.victory for episode in collected.group.episodes),
        "mean_floor": sum(episode.final_floor for episode in collected.group.episodes)
        / len(collected.group.episodes),
        "anchor_census": asdict(collected.group.anchor_census),
        "checkpoint_candidates": len(candidates["checkpoints"]),
        "captured_checkpoints": sum(
            item["path"] is not None for item in candidates["checkpoints"]
        ),
        "battle_candidates": len(candidates["battles"]),
        "environment": environment,
        "output": str(output),
        "candidates": str(candidate_output),
    }
