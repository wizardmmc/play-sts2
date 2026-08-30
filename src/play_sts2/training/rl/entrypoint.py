"""连接本地游戏 workers 与远程冻结策略收集一个战斗 group。"""

from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

from ...client import GameClient
from ...inference import OpenAICompatibleProvider
from ...runtime import BattleRunner
from ...scenario import BattleResetter
from .collector import BattleGroupCollector, GameBattleRolloutWorker
from .contracts import BEHAVIOR_LOGPROBS_MODE
from .io import load_battle_scenario, write_battle_rollout_group


def collect_battle_rollout_group(
    *,
    scenario_path: Path,
    game_urls: tuple[str, ...],
    model_url: str,
    policy_model: str,
    vllm_logprobs_mode: str,
    group_id: str,
    output_path: Path,
    group_size: int = 8,
    max_tokens: int = 128,
    temperature: float = 0.8,
    infrastructure_attempts: int = 3,
) -> dict[str, object]:
    """从多个隔离游戏实例收集并写出一个严格同入口 group。

    Args:
        scenario_path (Path): 确定性战斗场景 JSON。
        game_urls (tuple[str, ...]): 一到四个本地 Agent Mod 服务地址。
        model_url (str): A100 上的 vLLM-compatible 服务地址。
        policy_model (str): 请求和响应共同绑定的冻结模型版本。
        vllm_logprobs_mode (str): vLLM 启动时使用的 log-prob 模式声明。
        group_id (str): 当前 group 的稳定标识。
        output_path (Path): 完整 group JSON 输出路径。
        group_size (int): 同状态 arm 数，默认 8。
        max_tokens (int): 每个动作允许生成的最大 token 数。
        temperature (float): rollout 行为策略采样温度。
        infrastructure_attempts (int): 单条 arm 的基础设施总尝试数。

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
    scenario = load_battle_scenario(scenario_path)
    with ExitStack() as stack:
        workers: list[GameBattleRolloutWorker] = []
        for index, game_url in enumerate(normalized_game_urls):
            game = stack.enter_context(GameClient(game_url))
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
        group = BattleGroupCollector(
            workers,
            group_size=group_size,
            infrastructure_attempts=infrastructure_attempts,
        ).collect(scenario, group_id=group_id)
    output = write_battle_rollout_group(output_path, group)
    return {
        "group_id": group.group_id,
        "policy_version": group.policy_version,
        "behavior_logprobs_mode": group.behavior_logprobs_mode,
        "action_constraint_mode": group.action_constraint_mode,
        "generation_profile": asdict(group.generation_profile),
        "arms": len(group.rollouts),
        "reward_mean": group.reward_mean,
        "reward_std": group.reward_std,
        "output": str(output),
    }
