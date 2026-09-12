"""从已选真实入口采集较大的独立战斗批次，并逐组保存有效量与拒绝原因。"""

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from ....game_launcher import launch_game
from ..collector import RolloutInfrastructureError
from ..contracts import BattleGroupRejected
from ..entrypoint import collect_battle_rollout_group
from ..learner import compare_battle_reward_schemes
from .serving import check_online_serving


def collect_selected_battles(
    *,
    selection_path: Path,
    output_root: Path,
    target_groups: int,
    executable: Path,
    profile: Path,
    ports: tuple[int, ...],
    model_url: str,
    expected_bindings: dict[str, str],
    reward_scheme: str = "core",
) -> dict[str, Any]:
    """依次采不同入口，达到有效组目标或耗尽已选入口后结束。

    Args:
        selection_path (Path): 现有选择器输出的 selection.json。
        output_root (Path): 新批次目录；不覆盖或混入历史 rollout。
        target_groups (int): 希望采满的有效 K=8 组数。
        executable (Path): 固定版本游戏程序。
        profile (Path): 隔离游戏配置模板。
        ports (tuple[int, ...]): 一至两个游戏端口。
        model_url (str): 冻结推理服务地址。
        expected_bindings (dict[str, str]): 已核实实物的模型路径映射。
        reward_scheme (str): 有效组配额使用的实际 learner 奖励方案。

    Raises:
        ValueError: 预算、拓扑或实际服务身份错误。

    Returns:
        dict[str, Any]: 有效组、拒绝组、墙钟和是否完成目标的持久化报告；
        游戏启动超时按基础设施拒绝记账，不中断批次。
    """
    if target_groups < 1 or not 1 <= len(ports) <= 2 or len(set(ports)) != len(ports):
        raise ValueError("战斗批次要求正数目标及一至两个不同端口")
    if reward_scheme not in {"core", "core_no_turn", "core_no_turn_boss_progress"}:
        raise ValueError("战斗批次只支持 core 或 core_no_turn 系方案")
    selection = json.loads(selection_path.read_text())
    policy = selection["battle_policy_version"]
    if policy not in expected_bindings:
        raise ValueError(f"缺少战斗 policy 实物绑定: {policy}")
    binding = {policy: expected_bindings[policy]}
    check_online_serving(model_url, binding)
    # 游戏进程有独立 cwd，HOME 必须使用绝对路径。
    output_root = output_root.resolve()
    executable = executable.resolve()
    profile = profile.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "battle").mkdir()
    report: dict[str, Any] = {
        "status": "collecting",
        "policy_model": policy,
        "serving": binding,
        "selection": str(selection_path),
        "target_groups": target_groups,
        "accepted_groups": 0,
        "attempted_groups": 0,
        "accepted": [],
        "rejected": [],
        "reward_scheme": reward_scheme,
    }
    started = time.monotonic()
    for index, row in enumerate(selection["selected"]):
        if report["accepted_groups"] >= target_groups:
            break
        check_online_serving(model_url, binding)
        report["attempted_groups"] += 1
        try:
            result = _collect_one(
                scenario_path=Path(row["scenario"]),
                output_path=output_root / "battle" / f"group-{index:03d}.json",
                home_root=output_root / "homes" / f"group-{index:03d}",
                executable=executable,
                profile=profile,
                ports=ports,
                model_url=model_url,
                policy_model=policy,
                group_id=f"{output_root.name}-battle-{index:03d}",
            )
            check_online_serving(model_url, binding)
            if reward_scheme != "core":
                output_path = Path(str(result["output"]))
                comparison = compare_battle_reward_schemes([output_path])
                reward = comparison["groups"][0]["schemes"][reward_scheme]
                if not reward["trainable"]:
                    rejected_root = output_root / "rejected"
                    rejected_root.mkdir(exist_ok=True)
                    output_path.rename(rejected_root / output_path.name)
                    raise BattleGroupRejected(f"{reward_scheme} 重算后奖励没有方差")
                result["training_reward_mean"] = reward["mean"]
                result["training_reward_std"] = reward["std"]
        except (BattleGroupRejected, RolloutInfrastructureError, TimeoutError) as exc:
            report["rejected"].append(
                {
                    "scenario": row["scenario"],
                    "type": type(exc).__name__,
                    "reason": str(exc),
                }
            )
        else:
            report["accepted"].append(result)
            report["accepted_groups"] += 1
        report["elapsed_seconds"] = time.monotonic() - started
        (output_root / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "accepted_groups",
                        "attempted_groups",
                        "elapsed_seconds",
                    )
                }
            ),
            flush=True,
        )
    report["status"] = (
        "completed"
        if report["accepted_groups"] >= target_groups
        else "insufficient_groups"
    )
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report


def _collect_one(
    *,
    scenario_path: Path,
    output_path: Path,
    home_root: Path,
    executable: Path,
    profile: Path,
    ports: tuple[int, ...],
    model_url: str,
    policy_model: str,
    group_id: str,
) -> dict[str, object]:
    """每个 K=8 组使用新游戏进程，避免长时间 reset 后的已知超时。

    Args:
        scenario_path (Path): 独立合成入口的完整场景配置。
        output_path (Path): 通过准入后写入的 rollout 文件。
        home_root (Path): 本组独占 HOME 的父目录。
        executable (Path): 固定版本游戏程序。
        profile (Path): 隔离配置模板。
        ports (tuple[int, ...]): worker 端口。
        model_url (str): 冻结服务地址。
        policy_model (str): 当前战斗 residual 名称。
        group_id (str): 本组唯一可读名称。

    Returns:
        dict[str, object]: 现有严格 collector 的完整结果。
    """
    with ExitStack() as stack:
        games = [
            stack.enter_context(
                launch_game(
                    executable,
                    port=port,
                    home=home_root / f"worker-{index}",
                    profile=profile,
                    mode="headless",
                    enable_debug_actions=True,
                )
            )
            for index, port in enumerate(ports)
        ]
        return collect_battle_rollout_group(
            scenario_path=scenario_path,
            output_path=output_path,
            game_urls=tuple(game.base_url for game in games),
            model_url=model_url,
            policy_model=policy_model,
            group_id=group_id,
            vllm_logprobs_mode="processed_logprobs",
            structured_output_backend="xgrammar",
            structured_output_version="0.1.33",
        )


def main() -> None:
    """从已验证服务收据启动有明确有效组目标的战斗采集批次。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--serving-receipt", type=Path, required=True)
    parser.add_argument("--target-groups", type=int, default=16)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--port", type=int, action="append", required=True)
    parser.add_argument("--model-url", required=True)
    parser.add_argument(
        "--reward-scheme",
        choices=("core", "core_no_turn", "core_no_turn_boss_progress"),
        default="core_no_turn",
    )
    args = parser.parse_args()
    receipt = json.loads(args.serving_receipt.read_text())
    result = collect_selected_battles(
        selection_path=args.selection,
        output_root=args.output_root,
        target_groups=args.target_groups,
        executable=args.executable,
        profile=args.profile,
        ports=tuple(args.port),
        model_url=args.model_url,
        expected_bindings=receipt["bindings"],
        reward_scheme=args.reward_scheme,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
