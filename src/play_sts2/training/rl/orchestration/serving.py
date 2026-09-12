"""核对 serving 实物与在线模型绑定，并提供受检服务启动入口。"""

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import httpx


def validate_serving_job(job: dict[str, Any], *, root: Path) -> dict[str, str]:
    """检查 LoRA 别名与现有 composition 收据中的 residual 身份一致。

    Args:
        job (dict[str, Any]): vLLM argv 启动配置。
        root (Path): 服务进程工作目录。

    Raises:
        ValueError: LoRA 被关闭、别名错配或没有可验证的组合产物。

    Returns:
        dict[str, str]: 可与在线模型列表比较的名称到路径映射。
    """
    argv = job.get("argv", [])
    if "--enable-lora" not in argv or "--lora-modules" not in argv:
        raise ValueError("RL serving 必须显式加载组合 LoRA")
    bindings = {}
    for item in argv[argv.index("--lora-modules") + 1 :]:
        if item.startswith("--"):
            break
        name, separator, adapter_path = item.partition("=")
        if not separator or name in bindings:
            raise ValueError(f"无效或重复的 LoRA 绑定: {item}")
        adapter = root / adapter_path
        manifest = json.loads((adapter / "serving_manifest.json").read_text())
        if (
            manifest.get("name") != name
            or Path(manifest.get("residual_adapter", "")).name != name
        ):
            raise ValueError(f"serving 别名 {name} 与实际 residual 不一致: {manifest}")
        if not (adapter / "adapter_model.safetensors").is_file():
            raise ValueError(f"serving 权重不存在: {adapter}")
        bindings[name] = adapter_path
    if not bindings:
        raise ValueError("RL serving 没有 LoRA 绑定")
    return bindings


def validate_serving_models(
    payload: dict[str, Any], expected: dict[str, str]
) -> dict[str, str]:
    """拒绝在线模型名字正确但实际 root 指向旧权重的服务。

    Args:
        payload (dict[str, Any]): 在线 GET /v1/models 的实际响应。
        expected (dict[str, str]): 启动前通过实物检查的名称与路径。

    Raises:
        ValueError: 需要的模型缺失或实际 root 不一致。

    Returns:
        dict[str, str]: 已逐个核实的在线绑定。
    """
    actual = {row["id"]: row.get("root") for row in payload.get("data", [])}
    for name, path in expected.items():
        if actual.get(name) != path:
            raise ValueError(
                f"serving {name} root 错配: expected={path}, actual={actual.get(name)}"
            )
    return {name: actual[name] for name in expected}


def check_online_serving(model_url: str, expected: dict[str, str]) -> dict[str, str]:
    """读取实际模型列表并在采样前核对所有所需权重路径。

    Args:
        model_url (str): vLLM 服务根 URL。
        expected (dict[str, str]): 已验证的模型绑定。

    Returns:
        dict[str, str]: 通过检查的在线绑定。
    """
    response = httpx.get(f"{model_url.rstrip('/')}/v1/models", timeout=15)
    response.raise_for_status()
    return validate_serving_models(response.json(), expected)


def main() -> None:
    """验证 job；可保存收据、核查在线服务或启动经过实物检查的 vLLM。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--check-url")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    bindings = validate_serving_job(job, root=Path.cwd())
    if args.check_url:
        check_online_serving(args.check_url, bindings)
    result = {
        "job": str(args.job),
        "bindings": bindings,
        "online": bool(args.check_url),
    }
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if args.serve:
        sys.argv = ["vllm", *job["argv"]]
        runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
