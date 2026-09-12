"""为单独标注的训练试验启用显式 FLA，不改变默认训练后端。"""

import argparse
import inspect
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


def make_trial_kernel(
    target: Callable[..., Any], receipt: dict[str, Any]
) -> Callable[..., Any]:
    """包装诊断中已验证的 FLA 调用，并拒绝丢弃有效参数。

    Args:
        target (Callable[..., Any]): 显式 FLA 内核。
        receipt (dict[str, Any]): 记录调用次数及被过滤的关闭选项。

    Returns:
        Callable[..., Any]: 可注入模型模块的内核包装。
    """
    supported = inspect.signature(target).parameters

    def kernel(*args: Any, **kwargs: Any) -> Any:
        """保留支持参数，只过滤值为 None 或 False 的未知选项。"""
        unsupported = {k: v for k, v in kwargs.items() if k not in supported}
        for key, value in unsupported.items():
            if value is not None and value is not False:
                raise ValueError(f"FLA 不支持有效参数: {key}")
        receipt["dropped_inactive_kwargs"] = sorted(
            set(receipt["dropped_inactive_kwargs"]) | set(unsupported)
        )
        receipt["kernel_calls"] += 1
        return target(*args, **{k: v for k, v in kwargs.items() if k in supported})

    return kernel


def validate_trial_job(argv: list[str]) -> None:
    """要求从父 adapter 开始新试验，禁止跨后端精确 resume。"""
    if not argv or argv[0] not in {"battle-grpo", "strategy-grpo"}:
        raise ValueError("FLA 试验只接受明确的 GRPO 训练 job")
    if any(arg == "--resume" or arg.startswith("--resume=") for arg in argv):
        raise ValueError("独立 FLA 试验不允许 --resume；必须从父 adapter 开始")


def main() -> None:
    """执行独立训练 job，并将后端身份和完成状态写入独立收据。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    validate_trial_job(job["argv"])
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    from ...cli import main as training_main
    from .. import battle_trainer

    receipt = {
        "backend": "explicit_fla_trial",
        "job": str(args.job),
        "argv": job["argv"],
        "status": "running",
        "equivalent_to_native": False,
        "kernel_calls": 0,
        "dropped_inactive_kwargs": [],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    original_hidden = battle_trainer._hidden_and_head
    installed = []

    def hidden_and_head(model: Any, input_ids: Any) -> Any:
        """在模型加载完成后的首次前向安装内核，避免初始化覆盖补丁。"""
        if not installed:
            base = model.get_base_model()
            module = sys.modules[type(base.model.layers[0].linear_attn).__module__]
            installed.append((module, module.torch_chunk_gated_delta_rule))
            module.torch_chunk_gated_delta_rule = make_trial_kernel(
                chunk_gated_delta_rule, receipt
            )
        return original_hidden(model, input_ids)

    battle_trainer._hidden_and_head = hidden_and_head
    try:
        training_main(job["argv"])
        if not receipt["kernel_calls"]:
            raise RuntimeError("试验没有实际进入 FLA 内核")
        receipt["status"] = "completed_requires_review_and_game_evaluation"
    except BaseException:
        receipt["status"] = "failed"
        raise
    finally:
        battle_trainer._hidden_and_head = original_hidden
        for module, original in installed:
            module.torch_chunk_gated_delta_rule = original
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
