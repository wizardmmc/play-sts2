"""提供连接当前 STS2 战斗与 Qwen 服务的命令行入口。"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..client import GameClient
from ..inference import OpenAICompatibleProvider
from ..inference.config import DEFAULT_INFERENCE_CONFIG, load_inference_config
from .battle import BattleRunner
from .model_smoke import validate_model_service

_DEFAULT_GAME_URL = "http://127.0.0.1:8080"


def main(argv: Sequence[str] | None = None) -> int:
    """连接已运行的游戏与模型服务，并完成当前一场战斗。

    Args:
        argv (Sequence[str] | None): 不含程序名的命令行参数；为 ``None`` 时
            使用当前进程参数。

    Returns:
        int: 战斗正常离场时返回 ``0``。
    """
    args = _parser().parse_args(argv)
    config = load_inference_config(args.inference_config)
    profile = config.profile(args.profile)
    model_url = args.model_url or config.base_url
    validate_model_service(
        model_url,
        artifact_id=config.artifact_id,
        merged_model=config.merged_model,
        serving_model=config.serving_model,
        enable_thinking=profile.enable_thinking,
    )
    with (
        GameClient(args.game_url) as game,
        OpenAICompatibleProvider(
            model_url,
            model=str(config.serving_model.resolve()),
            enable_thinking=profile.enable_thinking,
        ) as provider,
    ):
        result = BattleRunner(
            game,
            provider,
            max_tokens=profile.max_tokens,
            temperature=profile.temperature,
        ).run()

    print(f"战斗结束: {result.outcome.value}，执行 {len(result.steps)} 个动作")
    return 0


def _parser() -> argparse.ArgumentParser:
    """创建当前战斗命令的参数解析器。

    Returns:
        argparse.ArgumentParser: 包含游戏、模型服务和模型名参数的解析器。
    """
    parser = argparse.ArgumentParser(
        description="连接已运行的 STS2 与 Qwen 服务，完成当前一场战斗。",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--game-url",
        default=_DEFAULT_GAME_URL,
        help=f"Agent Mod 服务地址，默认为 {_DEFAULT_GAME_URL}",
    )
    parser.add_argument(
        "--model-url",
        help="覆盖推理配置中的 OpenAI-compatible 服务地址",
    )
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=DEFAULT_INFERENCE_CONFIG,
        help=f"推理配置，默认为 {DEFAULT_INFERENCE_CONFIG}",
    )
    parser.add_argument(
        "--profile",
        help="生成 profile；省略时使用推理配置默认值",
    )
    return parser
