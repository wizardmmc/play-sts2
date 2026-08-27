"""提供连接当前 STS2 战斗与 Qwen 服务的命令行入口。"""

import argparse
from collections.abc import Sequence

from ..client import GameClient
from ..inference import OpenAICompatibleProvider
from .battle import BattleRunner

_DEFAULT_GAME_URL = "http://127.0.0.1:8080"
_DEFAULT_MODEL_URL = "http://127.0.0.1:8900"


def main(argv: Sequence[str] | None = None) -> int:
    """连接已运行的游戏与模型服务，并完成当前一场战斗。

    Args:
        argv (Sequence[str] | None): 不含程序名的命令行参数；为 ``None`` 时
            使用当前进程参数。

    Returns:
        int: 战斗正常离场时返回 ``0``。
    """
    args = _parser().parse_args(argv)
    with (
        GameClient(args.game_url) as game,
        OpenAICompatibleProvider(args.model_url, model=args.model) as provider,
    ):
        result = BattleRunner(game, provider).run()

    print(f"战斗结束: {result.outcome.value}，执行 {len(result.steps)} 个动作")
    return 0


def _parser() -> argparse.ArgumentParser:
    """创建当前战斗命令的参数解析器。

    Returns:
        argparse.ArgumentParser: 包含游戏、模型服务和模型名参数的解析器。
    """
    parser = argparse.ArgumentParser(
        description="连接已运行的 STS2 与 Qwen 服务，完成当前一场战斗。",
    )
    parser.add_argument(
        "--game-url",
        default=_DEFAULT_GAME_URL,
        help=f"Agent Mod 服务地址，默认为 {_DEFAULT_GAME_URL}",
    )
    parser.add_argument(
        "--model-url",
        default=_DEFAULT_MODEL_URL,
        help=f"OpenAI-compatible 模型服务地址，默认为 {_DEFAULT_MODEL_URL}",
    )
    parser.add_argument(
        "--model",
        help="服务端要求的模型名称；本地单模型服务通常可以省略",
    )
    return parser
