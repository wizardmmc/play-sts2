"""验证真实战斗采样入口在连接外部进程前拒绝危险配置。"""

from pathlib import Path

import pytest

from play_sts2.training.rl import collect_battle_rollout_group


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
            group_id="raw-logprobs",
            output_path=Path("not-written.json"),
        )
