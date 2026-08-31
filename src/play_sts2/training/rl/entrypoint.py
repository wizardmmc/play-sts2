"""连接本地游戏 workers 与远程冻结策略收集一个战斗 group。"""

from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from ...client import GameClient, Health
from ...inference import OpenAICompatibleProvider
from ...runtime import BattleRunner
from ...scenario import BattleResetter
from .collector import BattleGroupCollector, GameBattleRolloutWorker
from .contracts import (
    BEHAVIOR_LOGPROBS_MODE,
    RL_GAME_VERSION,
    STRUCTURED_OUTPUT_BACKEND,
)
from .dagger import (
    DaggerReplayLabeler,
    load_dagger_rollout_source,
    select_dagger_candidates,
    summarize_dagger_unsupported,
    write_dagger_labels,
)
from .io import (
    load_battle_scenario,
    load_battle_snapshot,
    write_battle_rollout_group,
)


def collect_battle_rollout_group(
    *,
    scenario_path: Path,
    game_urls: tuple[str, ...],
    model_url: str,
    policy_model: str,
    vllm_logprobs_mode: str,
    structured_output_backend: str,
    structured_output_version: str,
    group_id: str,
    output_path: Path,
    group_size: int = 8,
    max_tokens: int = 128,
    temperature: float = 0.8,
    infrastructure_attempts: int = 3,
    expected_snapshot_path: Path | None = None,
    allow_zero_variance: bool = False,
) -> dict[str, object]:
    """从多个隔离游戏实例收集并写出一个严格同入口 group。

    Args:
        scenario_path (Path): 确定性战斗场景 JSON。
        game_urls (tuple[str, ...]): 一到四个本地 Agent Mod 服务地址。
        model_url (str): A100 上的 vLLM-compatible 服务地址。
        policy_model (str): 请求和响应共同绑定的冻结模型版本。
        vllm_logprobs_mode (str): vLLM 启动时使用的 log-prob 模式声明。
        structured_output_backend (str): vLLM 实际选择的 grammar backend。
        structured_output_version (str): grammar backend 的精确包版本。
        group_id (str): 当前 group 的稳定标识。
        output_path (Path): 完整 group JSON 输出路径。
        group_size (int): 同状态 arm 数，默认 8。
        max_tokens (int): 每个动作允许生成的最大 token 数。
        temperature (float): rollout 行为策略采样温度。
        infrastructure_attempts (int): 单条 arm 的基础设施总尝试数。
        expected_snapshot_path (Path | None): 可选的 backbone 原始入口快照；提供时
            第一条 reset 也必须逐字段一致。
        allow_zero_variance (bool): 是否为独立 DAgger 保存零优势完整组。

    Raises:
        ValueError: 游戏 worker 数、地址或 vLLM 行为概率模式不满足契约。
        BattleGroupRejected: 完成的 arms 不满足同状态准入条件。
        RolloutInfrastructureError: 基础设施重采后仍无法完成。

    Returns:
        dict[str, object]: 供 CLI 输出的 group 摘要。
    """
    if not 1 <= len(game_urls) <= 4:
        raise ValueError("战斗 RL 采样需要一到四个游戏 worker")
    normalized_game_urls = tuple(url.rstrip("/") for url in game_urls)
    if any(not url for url in normalized_game_urls):
        raise ValueError("游戏 worker 地址不能为空")
    if len(set(normalized_game_urls)) != len(normalized_game_urls):
        raise ValueError("游戏 worker 地址不能重复")
    if vllm_logprobs_mode != BEHAVIOR_LOGPROBS_MODE:
        raise ValueError("战斗 RL 要求 vLLM 使用 --logprobs-mode processed_logprobs")
    if (
        structured_output_backend != STRUCTURED_OUTPUT_BACKEND
        or not structured_output_version.strip()
    ):
        raise ValueError("战斗 RL 要求显式声明 xgrammar backend 与精确版本")
    scenario = load_battle_scenario(scenario_path)
    expected_snapshot = (
        load_battle_snapshot(expected_snapshot_path)
        if expected_snapshot_path is not None
        else None
    )
    with ExitStack() as stack:
        workers: list[GameBattleRolloutWorker] = []
        healths: list[Health] = []
        for index, game_url in enumerate(normalized_game_urls):
            game = stack.enter_context(GameClient(game_url))
            healths.append(game.health())
            provider = stack.enter_context(
                OpenAICompatibleProvider(
                    model_url,
                    model=policy_model,
                    enable_thinking=False,
                    capture_token_metadata=True,
                )
            )
            workers.append(
                GameBattleRolloutWorker(
                    worker_id=f"worker-{index}",
                    resetter=BattleResetter(game),
                    behavior_logprobs_mode=vllm_logprobs_mode,
                    runner=BattleRunner(
                        game,
                        provider,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        max_retries=0,
                        max_conflict_retries=3,
                        constrain_actions=True,
                    ),
                )
            )
        environment = {
            **validate_rl_game_health(healths),
            "structured_output_backend": structured_output_backend,
            "structured_output_version": structured_output_version,
        }
        group = BattleGroupCollector(
            workers,
            group_size=group_size,
            infrastructure_attempts=infrastructure_attempts,
            allow_zero_variance=allow_zero_variance,
        ).collect(
            scenario,
            group_id=group_id,
            expected_snapshot=expected_snapshot,
        )
    output = write_battle_rollout_group(
        output_path,
        group,
        environment=environment,
    )
    return {
        "group_id": group.group_id,
        "policy_version": group.policy_version,
        "behavior_logprobs_mode": group.behavior_logprobs_mode,
        "action_constraint_mode": group.action_constraint_mode,
        "generation_profile": asdict(group.generation_profile),
        "arms": len(group.rollouts),
        "reward_mean": group.reward_mean,
        "reward_std": group.reward_std,
        "environment": environment,
        "output": str(output),
    }


def validate_rl_game_health(
    healths: tuple[Health, ...] | list[Health],
) -> dict[str, str]:
    """确认所有本地 worker 都是同一套固定 RL 游戏运行时。

    Args:
        healths (tuple[Health, ...] | list[Health]): 每个游戏 worker 的健康收据。

    Raises:
        ValueError: worker 为空、不是 v0.111.0 或 Mod/协议版本不一致。

    Returns:
        dict[str, str]: 可写入 rollout group 的统一环境版本。
    """
    if not healths:
        raise ValueError("战斗 RL 缺少游戏运行时收据")
    versions = {
        (health.game_version, health.mod_version, health.protocol_version)
        for health in healths
    }
    if len(versions) != 1:
        raise ValueError("战斗 RL worker 的游戏、Mod 或协议版本不一致")
    game_version, mod_version, protocol_version = versions.pop()
    if game_version != RL_GAME_VERSION:
        raise ValueError(f"战斗 RL 只允许 {RL_GAME_VERSION}，实际为 {game_version}")
    return {
        "game_version": game_version,
        "mod_version": mod_version,
        "protocol_version": protocol_version,
    }


def label_dagger_rollout_group(
    *,
    rollout_path: Path,
    game_url: str,
    output_path: Path,
    max_labels: int = 8,
    selection_seed: int = 0,
    search_timeout: float = 135.0,
) -> dict[str, object]:
    """在教师游戏中重放学生前缀并写出旁路 DAgger 标签。

    Args:
        rollout_path (Path): 已通过准入的学生 battle group JSON。
        game_url (str): 加载 CombatSolver 教师 Mod 的本地游戏地址。
        output_path (Path): 不含隐藏状态的标签 JSONL 输出路径。
        max_labels (int): 本批最多标注的学生状态数。
        selection_seed (int): 普通状态抽样种子。
        search_timeout (float): 每次 Solver 搜索最长秒数。

    Raises:
        DaggerContractError: 学生 group、重放或教师动作不满足标签契约。
        httpx.HTTPStatusError: 教师游戏或 Solver 端点失败。

    Returns:
        dict[str, object]: 标签数、分歧数、选择原因和输出路径摘要。
    """
    source = load_dagger_rollout_source(rollout_path)
    candidates = select_dagger_candidates(
        source,
        max_labels=max_labels,
        seed=selection_seed,
    )
    with GameClient(game_url.rstrip("/")) as game:
        labels = DaggerReplayLabeler(
            game,
            resetter=BattleResetter(game),
            search_timeout=search_timeout,
            harness_version=version("play-sts2"),
        ).label(source, candidates)
    unsupported_reasons = summarize_dagger_unsupported(source)
    output = write_dagger_labels(
        output_path,
        labels,
        selection_seed=selection_seed,
        selection_budget=max_labels,
        unsupported_reasons=unsupported_reasons,
    )
    return {
        "group_id": source.group_id,
        "student_policy_version": source.policy_version,
        "labels": len(labels),
        "agreements": sum(label.agrees for label in labels),
        "disagreements": sum(not label.agrees for label in labels),
        "selection_reasons": dict(
            sorted(Counter(label.selection_reason for label in labels).items())
        ),
        "unsupported_reasons": unsupported_reasons,
        "output": str(output),
    }
