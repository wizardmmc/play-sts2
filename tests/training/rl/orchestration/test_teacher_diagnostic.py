"""验证教师诊断不会执行过期或超出合法域的建议。"""

import json
import time
from types import SimpleNamespace

import pytest

from play_sts2.training.rl.orchestration.teacher_diagnostic import (
    SolverProvider,
    SolverUnsupportedState,
    http_failure_status,
    load_budget,
    new_game_home,
    select_trial_path,
)


class SuggestionGame:
    """提供绑定 revision 的离线建议，隔离真实搜索开销。"""

    def __init__(self, action: str, revision: int = 7) -> None:
        """保存需要验证的教师回复。"""
        self.action = action
        self.revision = revision

    def state(self) -> dict:
        """返回当前决策 revision。"""
        return {"state_revision": 7}

    def solver_suggestion(self, **kwargs: object) -> SimpleNamespace:
        """返回独立于请求参数的建议，允许测试错误绑定。"""
        return SimpleNamespace(
            action=self.action, state_revision=self.revision, solver_version="test"
        )


@pytest.mark.parametrize(
    ("action", "revision"), [("ACTION: end_turn", 6), ("ACTION: play_card 9", 7)]
)
def test_solver_rejects_stale_or_illegal_suggestion(action: str, revision: int) -> None:
    """过期 revision 或非法动作不能变成可执行教师标签。"""
    provider = SolverProvider(SuggestionGame(action, revision), time.monotonic() + 60)
    with pytest.raises(ValueError):
        provider.chat([], response_choices=["ACTION: end_turn", "ACTION: play_card 0"])


def test_budget_also_applies_to_forced_actions() -> None:
    """单合法动作也不能让已经超时的实验继续运行。"""
    provider = SolverProvider(SuggestionGame("ACTION: end_turn"), time.monotonic() - 1)
    with pytest.raises(TimeoutError):
        provider.chat([], response_choices=["ACTION: end_turn"])


def test_valid_solver_action_remains_bound_to_teacher() -> None:
    """通过检查的建议保留实际教师身份。"""
    provider = SolverProvider(SuggestionGame("ACTION: end_turn"), time.monotonic() + 60)
    reply = provider.chat(
        [], response_choices=["ACTION: end_turn", "ACTION: play_card 0"]
    )
    assert reply.text == "ACTION: end_turn"
    assert reply.model == "CombatSolver-test"


def test_selection_without_exported_plan_is_not_a_teacher_loss() -> None:
    """选牌计划未导出时明确拒绝，不猜索引或伪造教师败局。"""
    provider = SolverProvider(
        SuggestionGame("ACTION: play_card 0"), time.monotonic() + 60
    )
    with pytest.raises(SolverUnsupportedState):
        provider.chat(
            [],
            response_choices=[
                "ACTION: select_deck_card 0",
                "ACTION: select_deck_card 1",
            ],
        )


@pytest.mark.parametrize(
    ("agent", "path", "expected"),
    [
        ("solver", "/solver/suggest", "teacher_error"),
        ("solver", "/v1/models", "inconclusive"),
        ("b3", "/v1/chat/completions", "inconclusive"),
    ],
)
def test_teacher_http_failure_does_not_hide_serving_failure(
    agent, path, expected
) -> None:
    """教师搜索内部失败可跳过，模型身份或服务失败仍必须停止。"""
    assert http_failure_status(agent, path) == expected


def test_resume_preserves_original_deadline(tmp_path, monkeypatch) -> None:
    """重复启动不能重新获得完整预算。"""
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    first = load_budget(tmp_path, 300)
    monkeypatch.setattr(time, "time", lambda: 1200.0)
    resumed = load_budget(tmp_path, 300)
    assert first == resumed == {"started_at": 1000.0, "deadline_at": 1300.0}


def test_resume_uses_empty_home_and_retains_previous_evidence(tmp_path) -> None:
    """启动器要求空 HOME，续跑必须另建而不能清空旧日志。"""
    first = new_game_home(tmp_path)
    (first / "headless.log").write_text("old evidence")
    second = new_game_home(tmp_path)
    assert second != first
    assert list(second.iterdir()) == []
    assert (first / "headless.log").read_text() == "old evidence"


def test_retry_preserves_incomplete_attempt_and_skips_completed(tmp_path) -> None:
    """明确重试时另开目录，完整结果不重复执行。"""
    original = tmp_path / "solver-1"
    original.mkdir()
    receipt = original / "result.json"
    receipt.write_text(json.dumps({"status": "inconclusive"}))
    assert select_trial_path(tmp_path, "solver", 1) == original
    retry = select_trial_path(tmp_path, "solver", 1, retry_incomplete=True)
    assert retry == tmp_path / "solver-1-attempt-2"
    assert json.loads(receipt.read_text())["status"] == "inconclusive"
    retry.mkdir()
    (retry / "result.json").write_text(json.dumps({"status": "completed"}))
    assert select_trial_path(tmp_path, "solver", 1, retry_incomplete=True) == retry


@pytest.mark.parametrize("student_model", [None, "qwen3.5-a0-b3-real1"])
def test_main_binds_the_requested_student_model(tmp_path, monkeypatch, student_model):
    """显式real1不能静默变成B3，旧实验未配置时仍保留B3默认值。"""
    from contextlib import nullcontext

    from play_sts2.training.rl.orchestration import teacher_diagnostic as module

    plan = {
        "output_root": str(tmp_path / "run"),
        "cases": [{"name": "case"}],
        "total_seconds": 60,
        "executable": "game",
        "profile": "profile",
        "port": 8084,
        "model_url": "http://model",
        "agent_order": ["student"],
    }
    if student_model:
        plan["student_model"] = student_model
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr("sys.argv", ["diagnostic", "--plan", str(path)])
    monkeypatch.setattr(
        module,
        "launch_game",
        lambda *a, **kw: nullcontext(SimpleNamespace(base_url="http://game")),
    )
    monkeypatch.setattr(
        module,
        "GameClient",
        lambda *a: nullcontext(SimpleNamespace(health=lambda: None)),
    )
    monkeypatch.setattr(module, "validate_rl_game_health", lambda *a: {})
    observed = []

    def provider(*args, **kwargs):
        """记录真正用于构造学生提供者的模型身份。"""
        observed.append(kwargs["model"])
        return nullcontext(object())

    def trial(*args, **kwargs):
        """隔离真实游戏，仅让CLI完成一次学生调度。"""
        return {"case": "case", "agent": "student", "repeat": 0, "status": "completed"}

    monkeypatch.setattr(module, "OpenAICompatibleProvider", provider)
    monkeypatch.setattr(module, "run_trial", trial)
    module.main()
    assert observed == [student_model or "qwen3.5-e7-b3"]


@pytest.mark.parametrize(
    "agent,temperature,override",
    [
        ("student", 0.8, None),
        ("b3", 0.8, None),
        ("solver", 0.0, None),
        ("student", 0.0, 0.0),
    ],
)
def test_trial_preserves_student_sampling_temperature(
    tmp_path, monkeypatch, agent, temperature, override
):
    """学生改用通用名称后不能误用教师的确定性温度。"""
    from play_sts2.runtime import BattleOutcome, BattleResult
    from play_sts2.scenario.models import BattleSnapshot, ModelInputSnapshot
    from play_sts2.training.rl.orchestration import teacher_diagnostic as module

    state = {"run": {"current_hp": 40, "max_hp": 75}}
    snapshot = BattleSnapshot(1, (), (), ModelInputSnapshot("system", "user", ()))
    reset = SimpleNamespace(state=state, snapshot=snapshot)
    monkeypatch.setattr(module, "check_online_serving", lambda *a: {})
    monkeypatch.setattr(module, "load_battle_scenario", lambda *a: None)
    monkeypatch.setattr(
        module,
        "BattleResetter",
        lambda *a: SimpleNamespace(reset=lambda *a, **kw: reset),
    )
    observed = []

    def runner(*args, **kwargs):
        """记录战斗执行器实际收到的温度而不运行游戏。"""
        observed.append(kwargs["temperature"])
        return SimpleNamespace(
            run=lambda *a: BattleResult(BattleOutcome.CLEARED, (), state)
        )

    monkeypatch.setattr(module, "BattleRunner", runner)
    result = module.run_trial(
        object(),
        object(),
        plan={
            "model_url": "url",
            "bindings": {},
            "trial_seconds": 60,
            **({"student_temperature": override} if override is not None else {}),
        },
        case={"name": "case", "scenario": "scenario"},
        root=tmp_path,
        agent=agent,
        repeat=0,
        trial_root=tmp_path / "case" / "arm",
        total_deadline=time.monotonic() + 60,
    )
    assert result["status"] == "completed"
    assert observed == [temperature]


def test_solver_query_budget_stops_before_game_access():
    """查询尝试耗尽时不再访问游戏，singleton不消耗建议预算。"""
    from play_sts2.training.rl.orchestration.teacher_diagnostic import SolverProvider

    provider = SolverProvider(object(), time.monotonic() + 60, max_queries=0)
    assert (
        provider.chat([], response_choices=["ACTION: end_turn"]).text
        == "ACTION: end_turn"
    )
    with pytest.raises(RuntimeError, match="额度"):
        provider.chat([], response_choices=["ACTION: end_turn", "ACTION: play_card 0"])
