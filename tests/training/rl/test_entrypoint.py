"""验证真实战斗采样入口在连接外部进程前拒绝危险配置。"""

from pathlib import Path

import pytest

from play_sts2.client import Health
from play_sts2.training.rl import collect_battle_rollout_group
from play_sts2.training.rl.entrypoint import validate_rl_game_health


def test_entrypoint_rejects_duplicate_game_urls() -> None:
    """同一游戏端点不能伪装成两个隔离 worker。

    Returns:
        None: 此测试保证重复 URL 不会产生并发 reset/run 竞争。
    """
    with pytest.raises(ValueError, match="游戏 worker 地址不能重复"):
        collect_battle_rollout_group(
            scenario_path=Path("not-read.json"),
            game_urls=(
                "http://127.0.0.1:8080",
                "http://127.0.0.1:8080/",
            ),
            model_url="http://127.0.0.1:8900",
            policy_model="policy-test",
            vllm_logprobs_mode="processed_logprobs",
            structured_output_backend="xgrammar",
            structured_output_version="0.1.33",
            group_id="duplicate-workers",
            output_path=Path("not-written.json"),
        )


def test_entrypoint_rejects_raw_vllm_logprobs() -> None:
    """温度采样必须保存处理后的真实行为分布概率。

    Returns:
        None: 此测试阻止把 vLLM 默认 raw log-prob 当成 behavior log-prob。
    """
    with pytest.raises(ValueError, match="processed_logprobs"):
        collect_battle_rollout_group(
            scenario_path=Path("not-read.json"),
            game_urls=("http://127.0.0.1:8080",),
            model_url="http://127.0.0.1:8900",
            policy_model="policy-test",
            vllm_logprobs_mode="raw_logprobs",
            structured_output_backend="xgrammar",
            structured_output_version="0.1.33",
            group_id="raw-logprobs",
            output_path=Path("not-written.json"),
        )


def test_entrypoint_rejects_non_rl_game_version() -> None:
    """RL collector 必须在连接后立刻拒绝非 0.111.0 游戏实例。

    Returns:
        None: 旧兼容副本不能产生看似有效的正式 rollout。
    """
    health = Health(
        service="sts2-ai-agent",
        mod_version="mod-test",
        protocol_version="protocol-test",
        game_version="v0.107.1",
        status="ready",
    )

    with pytest.raises(ValueError, match="v0.111.0"):
        validate_rl_game_health([health])
