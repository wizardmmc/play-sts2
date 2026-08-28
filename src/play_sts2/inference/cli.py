"""提供本地模型准备、服务和冒烟检查命令。"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from play_sts2.runtime.model_smoke import smoke_model

from .config import DEFAULT_INFERENCE_CONFIG, load_inference_config
from .local_model import prepare_model, serve_model


def build_parser() -> argparse.ArgumentParser:
    """构造本地模型命令行解析器。

    Returns:
        argparse.ArgumentParser: 包含 prepare、serve 和 smoke 子命令的解析器。
    """
    parser = argparse.ArgumentParser(prog="play-sts2-model")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="转换合并模型为 MLX 8-bit")
    _add_config_argument(prepare)

    serve = commands.add_parser("serve", help="启动本地模型服务")
    _add_config_argument(serve)

    smoke = commands.add_parser("smoke", help="验证模型服务与 Harness 协议")
    _add_config_argument(smoke)
    smoke.add_argument("--profile", help="生成 profile；省略时使用配置默认值")
    return parser


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    """为模型子命令添加统一推理配置参数。"""
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_INFERENCE_CONFIG,
        help=f"推理配置，默认为 {DEFAULT_INFERENCE_CONFIG}",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """执行用户选择的本地模型命令。

    Args:
        argv (Sequence[str] | None): 不含程序名的命令行参数。

    Returns:
        int: 成功时返回零。
    """
    args = build_parser().parse_args(argv)
    config = load_inference_config(args.config)
    if args.command == "prepare":
        output = prepare_model(
            config.merged_model,
            config.serving_model,
            artifact_id=config.artifact_id,
        )
        print(f"模型准备完成: {output.resolve()}")
    elif args.command == "serve":
        try:
            serve_model(
                config.serving_model,
                artifact_id=config.artifact_id,
                merged_model=config.merged_model,
                port=config.port,
                prompt_cache_size=config.prompt_cache_size,
            )
        except KeyboardInterrupt:
            return 130
    else:
        profile = config.profile(args.profile)
        reply = smoke_model(
            config.base_url,
            artifact_id=config.artifact_id,
            merged_model=config.merged_model,
            serving_model=config.serving_model,
            enable_thinking=profile.enable_thinking,
            max_tokens=profile.max_tokens,
            temperature=profile.temperature,
        )
        print(f"模型协议验证通过: {reply.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
