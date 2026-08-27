"""提供本地模型准备、服务和冒烟检查命令。"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from play_sts2.runtime.model_smoke import smoke_model

from .local_model import prepare_model, serve_model

DEFAULT_SOURCE = Path("models/merged/sft-clean-20260827-native-r16-e1-merged")
DEFAULT_OUTPUT = Path("models/serving/sft-clean-20260827-native-r16-e1-mlx-8bit")


def build_parser() -> argparse.ArgumentParser:
    """构造本地模型命令行解析器。

    Returns:
        argparse.ArgumentParser: 包含 prepare、serve 和 smoke 子命令的解析器。
    """
    parser = argparse.ArgumentParser(prog="play-sts2-model")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="转换合并模型为 MLX 8-bit")
    prepare.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    serve = commands.add_parser("serve", help="启动本地模型服务")
    serve.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT)
    serve.add_argument("--port", type=int, default=8900)
    serve.add_argument("--prompt-cache-size", type=int, default=10)

    smoke = commands.add_parser("smoke", help="验证模型服务与 Harness 协议")
    smoke.add_argument("--base-url", default="http://127.0.0.1:8900")
    smoke.add_argument("--model")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行用户选择的本地模型命令。

    Args:
        argv (Sequence[str] | None): 不含程序名的命令行参数。

    Returns:
        int: 成功时返回零。
    """
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        output = prepare_model(args.source, args.output)
        print(f"模型准备完成: {output.resolve()}")
    elif args.command == "serve":
        try:
            serve_model(
                args.model_dir,
                port=args.port,
                prompt_cache_size=args.prompt_cache_size,
            )
        except KeyboardInterrupt:
            return 130
    else:
        reply = smoke_model(args.base_url, model=args.model)
        print(f"模型协议验证通过: {reply.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
