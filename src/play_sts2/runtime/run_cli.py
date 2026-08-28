"""提供连接 STS2 与 Qwen 并完成整局的命令行入口。"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..client import GameClient
from ..inference import OpenAICompatibleProvider
from ..inference.config import DEFAULT_INFERENCE_CONFIG, load_inference_config
from ..run_start import resume_run, start_run
from .model_smoke import validate_model_service
from .run import RunRunner

_DEFAULT_GAME_URL = "http://127.0.0.1:8080"
_DEFAULT_CHARACTER = "DEFECT"


def main(argv: Sequence[str] | None = None) -> int:
    """开始新局或续玩当前局，并持续运行模型直到游戏结束。

    Args:
        argv (Sequence[str] | None): 不含程序名的命令行参数；为 ``None`` 时
            使用当前进程参数。

    Returns:
        int: 游戏正常到达终局时返回 ``0``。
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
        initial_state = (
            resume_run(game)
            if args.resume
            else start_run(
                game,
                args.character,
                seed=args.seed,
                ascension=args.ascension,
            )
        )
        result = RunRunner(
            game,
            provider,
            max_tokens=profile.max_tokens,
            temperature=profile.temperature,
        ).run(initial_state)

    print(
        f"整局结束: {result.outcome.value}，经历 {result.battle_count} 场战斗，"
        f"执行 {len(result.decisions)} 个动作"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    """创建完整一局命令的参数解析器。

    Returns:
        argparse.ArgumentParser: 包含游戏、模型、开局和续局参数的解析器。
    """
    parser = argparse.ArgumentParser(
        description="连接已运行的 STS2 与 Qwen 服务，自主完成一局游戏。",
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
    parser.add_argument(
        "--character",
        default=_DEFAULT_CHARACTER,
        help=f"新局使用的角色稳定 ID，默认为 {_DEFAULT_CHARACTER}",
    )
    parser.add_argument(
        "--ascension",
        type=int,
        default=0,
        help="新局使用的进阶等级，默认为 0",
    )
    parser.add_argument(
        "--seed",
        help="新局使用的可选游戏种子",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="续玩游戏中已经存在的当前局，不执行主菜单开局流程",
    )
    return parser
