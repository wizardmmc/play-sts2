"""在相同合成入口比较学生与 Solver，保存完整诊断轨迹而不写训练分卷。"""

import argparse
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from ....client import GameClient
from ....game_launcher import launch_game
from ....inference import ModelReply, OpenAICompatibleProvider
from ....runtime import BattleRunner
from ....runtime.battle import BattlePolicyFailure
from ....scenario import BattleResetter
from ..entrypoint import validate_rl_game_health
from ..io import load_battle_scenario, load_battle_snapshot
from .serving import check_online_serving


class SolverUnsupportedState(ValueError):
    """表示现有建议接口未导出当前决策所需的教师计划。"""


class SolverProvider:
    """把当前 revision 的合法 Solver 建议交给普通执行器。"""

    def __init__(
        self, game: GameClient, deadline: float, *, max_queries: int | None = None
    ) -> None:
        """保存本场游戏与墙钟截止时间。"""
        self.game = game
        self.deadline = deadline
        self.max_queries = max_queries
        self.queries = 0

    def chat(
        self,
        messages: object,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        response_choices: object = None,
    ) -> ModelReply:
        """只接受当前状态中可执行的教师动作。

        Raises:
            TimeoutError: 单场诊断预算耗尽。
            ValueError: 教师回复过期或动作不合法。

        Returns:
            ModelReply: 通过身份与合法域校验的教师动作。
        """
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Solver 单场诊断预算耗尽")
        if response_choices is not None and len(response_choices) == 1:
            return ModelReply(text=response_choices[0], model="forced-single-action")
        if self.max_queries is not None and self.queries >= self.max_queries:
            raise RuntimeError("Solver查询额度耗尽")
        self.queries += 1
        if response_choices and any(
            action.startswith("ACTION: select_deck_card ")
            for action in response_choices
        ):
            raise SolverUnsupportedState("当前 Solver 接口未导出动作附带的选牌计划")
        state = self.game.state()
        suggestion = self.game.solver_suggestion(
            expected_state_revision=state["state_revision"],
            timeout=min(45.0, remaining),
        )
        if suggestion.state_revision != state["state_revision"]:
            raise ValueError("Solver 建议 revision 与请求不一致")
        if response_choices is None or suggestion.action not in response_choices:
            raise ValueError(f"Solver 建议不在当前合法动作域: {suggestion.action!r}")
        return ModelReply(
            text=suggestion.action, model=f"CombatSolver-{suggestion.solver_version}"
        )


class RecordingProvider:
    """每步即时落盘模型可见输入和回复，使超时仍有可复盘证据。"""

    def __init__(self, provider: Any, path: Path, deadline: float) -> None:
        """保存被包装的 Provider、日志路径与单场截止时间。"""
        self.provider = provider
        self.path = path
        self.deadline = deadline
        self.thinking_enabled = getattr(provider, "thinking_enabled", None)

    def chat(self, messages: Any, **kwargs: Any) -> ModelReply:
        """在调用前保存观测，调用后保存回复或错误。

        Raises:
            TimeoutError: 单场预算耗尽。

        Returns:
            ModelReply: 原始 Provider 的回复。
        """
        if time.monotonic() >= self.deadline:
            raise TimeoutError("诊断单场预算耗尽")
        with self.path.open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(
                    {"messages": [asdict(m) for m in messages], **kwargs},
                    ensure_ascii=False,
                )
                + "\n"
            )
            output.flush()
            try:
                reply = self.provider.chat(messages, **kwargs)
            except Exception as exc:
                output.write(json.dumps({"error": str(exc)}, ensure_ascii=False) + "\n")
                raise
            output.write(
                json.dumps({"reply": asdict(reply)}, ensure_ascii=False) + "\n"
            )
        return reply


def write_json(path: Path, payload: Any) -> None:
    """以同目录替换写入当前结果，避免读取到半份状态文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def http_failure_status(agent: str, path: str) -> str:
    """仅把教师搜索端点自身的失败隔离，不掩盖模型服务或入口异常。"""
    return (
        "teacher_error"
        if agent == "solver" and path == "/solver/suggest"
        else "inconclusive"
    )


def new_game_home(root: Path) -> Path:
    """为每次进程启动创建空隔离 HOME，保留先前会话的游戏日志。"""
    homes = root / "homes"
    homes.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="session-", dir=homes))


def load_budget(root: Path, total_seconds: float) -> dict[str, float]:
    """保存首次启动的墙钟截止时间，续跑只消费剩余预算。"""
    path = root / "budget.json"
    if path.exists():
        return json.loads(path.read_text())
    started = time.time()
    budget = {"started_at": started, "deadline_at": started + total_seconds}
    write_json(path, budget)
    return budget


def select_trial_path(
    root: Path, agent: str, repeat: int, *, retry_incomplete: bool = False
) -> Path:
    """保留失败尝试，只有显式重试时才为未完成场次选择新目录。"""
    base = root / f"{agent}-{repeat}"
    attempts = sorted(
        root.glob(f"{base.name}-attempt-*"), key=lambda p: int(p.name.rsplit("-", 1)[1])
    )
    latest = attempts[-1] if attempts else base
    if not latest.exists() or not retry_incomplete:
        return latest
    receipt = latest / "result.json"
    if receipt.exists() and json.loads(receipt.read_text()).get("status") in {
        "completed",
        "policy_failure",
        "unsupported_teacher_state",
        "teacher_error",
    }:
        return latest
    number = int(latest.name.rsplit("-", 1)[1]) + 1 if attempts else 2
    return root / f"{base.name}-attempt-{number}"


def run_trial(
    game: GameClient,
    student: OpenAICompatibleProvider,
    *,
    plan: dict[str, Any],
    case: dict[str, Any],
    root: Path,
    agent: str,
    repeat: int,
    trial_root: Path,
    total_deadline: float,
) -> dict[str, Any]:
    """重建同一入口并执行一场，基础设施失败与真实死亡分别记账。

    Returns:
        dict[str, Any]: 含实际入口、完成状态和完整动作轨迹路径的收据。
    """
    case_root = root / case["name"]
    trial_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {
        "case": case["name"],
        "agent": agent,
        "repeat": repeat,
        "attempt_directory": str(trial_root),
        "status": "running",
        "counts_as_goal_victory": False,
        "purpose": "teacher_capability_diagnostic_only",
    }
    write_json(trial_root / "result.json", record)
    try:
        record["serving_before"] = check_online_serving(
            plan["model_url"], plan["bindings"]
        )
        snapshot_path = case_root / "paired-entry.json"
        expected = (
            load_battle_snapshot(snapshot_path) if snapshot_path.exists() else None
        )
        scenario = load_battle_scenario(Path(case["scenario"]))
        reset = BattleResetter(game).reset(scenario, expected_snapshot=expected)
        if expected is None:
            write_json(snapshot_path, asdict(reset.snapshot))
        write_json(trial_root / "initial-state.json", reset.state)
        deadline = min(total_deadline, time.monotonic() + plan["trial_seconds"])
        provider = (
            SolverProvider(game, deadline, max_queries=plan.get("max_queries"))
            if agent == "solver"
            else student
        )
        recorder = RecordingProvider(provider, trial_root / "requests.jsonl", deadline)
        result = BattleRunner(
            game,
            recorder,
            max_tokens=128,
            temperature=0.0
            if agent == "solver"
            else plan.get("student_temperature", 0.8),
            max_retries=0,
            max_conflict_retries=2,
            constrain_actions=True,
        ).run(reset.state)
        write_json(trial_root / "trajectory.json", asdict(result))
        record.update(
            status="completed",
            outcome=result.outcome.value,
            initial_hp=reset.state["run"]["current_hp"],
            final_hp=result.final_state["run"]["current_hp"],
            max_hp=result.final_state["run"]["max_hp"],
            steps=len(result.steps),
            potions_used=sum(step.action.name == "use_potion" for step in result.steps),
            trajectory=str(trial_root / "trajectory.json"),
        )
    except SolverUnsupportedState as exc:
        record.update(status="unsupported_teacher_state", error=str(exc))
    except httpx.HTTPStatusError as exc:
        record.update(
            status=http_failure_status(agent, exc.request.url.path),
            error_type=type(exc).__name__,
            error=str(exc),
            error_response=exc.response.text,
        )
    except BattlePolicyFailure as exc:
        write_json(trial_root / "failure-steps.json", [asdict(s) for s in exc.steps])
        record.update(
            status="policy_failure", error_type=type(exc).__name__, error=str(exc)
        )
    except (RuntimeError, ValueError, TimeoutError, OSError, httpx.HTTPError) as exc:
        record.update(
            status="inconclusive", error_type=type(exc).__name__, error=str(exc)
        )
    finally:
        try:
            record["serving_after"] = check_online_serving(
                plan["model_url"], plan["bindings"]
            )
        except (ValueError, httpx.HTTPError) as exc:
            record.update(status="inconclusive", identity_error=str(exc))
        record["elapsed_seconds"] = time.monotonic() - started
        write_json(trial_root / "result.json", record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


def main() -> None:
    """执行有限教师筛查；学生可由student_model指定，旧计划默认B3。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--retry-incomplete", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(plan["output_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cases = [c for c in plan["cases"] if not args.case or c["name"] in args.case]
    if not cases:
        raise ValueError("没有匹配的诊断入口")
    budget = load_budget(root, plan["total_seconds"])
    deadline = time.monotonic() + budget["deadline_at"] - time.time()
    report_path = root / "report.json"
    report = (
        json.loads(report_path.read_text()) if report_path.exists() else {"results": []}
    )
    report.update(status="running", plan=str(args.plan), **budget)
    report.pop("error", None)
    report.pop("error_type", None)
    write_json(root / "report.json", report)
    try:
        if time.monotonic() >= deadline:
            raise TimeoutError("本轮首次启动的总预算已耗尽")
        with (
            launch_game(
                Path(plan["executable"]).resolve(),
                port=plan["port"],
                home=new_game_home(root),
                profile=Path(plan["profile"]).resolve(),
                mode="headless",
                enable_debug_actions=True,
            ) as running,
            GameClient(running.base_url) as game,
            OpenAICompatibleProvider(
                plan["model_url"],
                model=plan.get("student_model", "qwen3.5-e7-b3"),
                enable_thinking=False,
                capture_token_metadata=True,
                timeout=45,
            ) as student,
        ):
            report["environment"] = validate_rl_game_health([game.health()])
            for case in cases:
                for repeat, agent in enumerate(plan["agent_order"]):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("本轮诊断总预算耗尽")
                    trial_root = select_trial_path(
                        root / case["name"],
                        agent,
                        repeat,
                        retry_incomplete=args.retry_incomplete,
                    )
                    result_path = trial_root / "result.json"
                    if result_path.exists():
                        record = json.loads(result_path.read_text())
                    else:
                        record = run_trial(
                            game,
                            student,
                            plan=plan,
                            case=case,
                            root=root,
                            agent=agent,
                            repeat=repeat,
                            trial_root=trial_root,
                            total_deadline=deadline,
                        )
                    report["results"] = [
                        item
                        for item in report["results"]
                        if (item["case"], item["agent"], item["repeat"])
                        != (case["name"], agent, repeat)
                    ]
                    report["results"].append(record)
                    write_json(root / "report.json", report)
                    if record["status"] not in {
                        "completed",
                        "policy_failure",
                        "unsupported_teacher_state",
                        "teacher_error",
                    }:
                        raise RuntimeError("入口或服务异常，先检查当前收据再继续")
            report["status"] = "completed"
    except Exception as exc:
        report.update(status="stopped", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.time() - budget["started_at"]
        write_json(root / "report.json", report)


if __name__ == "__main__":
    main()
