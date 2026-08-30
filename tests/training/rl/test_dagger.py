"""验证学生状态 DAgger 的抽样、标签与隐藏信息边界。"""

import json
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import pytest

from play_sts2.client import GameClient, Health, SolverSuggestion
from play_sts2.harness import HarnessLayer, build_observation, system_prompt
from play_sts2.inference import ChatMessage
from play_sts2.scenario import (
    BattleScenario,
    BattleSnapshot,
    ModelInputSnapshot,
    ScenarioResetResult,
)
from play_sts2.training import rl


def test_select_dagger_candidates_balances_death_uncertainty_and_ordinary(
    tmp_path: Path,
) -> None:
    """候选预算应同时保留死亡尾部、高不确定性和普通状态。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Returns:
        None: 此测试用手工 log-prob 固定三类状态的抽样结果。
    """
    source_path = _write_rollout_group(tmp_path)

    source = rl.load_dagger_rollout_source(source_path)
    candidates = rl.select_dagger_candidates(source, max_labels=3, seed=7)

    assert len(candidates) == 3
    assert {candidate.reason for candidate in candidates} == {
        "death_tail",
        "uncertainty",
        "ordinary",
    }
    assert len({candidate.label_id for candidate in candidates}) == 3
    uncertain = next(
        candidate for candidate in candidates if candidate.reason == "uncertainty"
    )
    assert (uncertain.arm_index, uncertain.step_index) == (1, 0)
    assert uncertain.uncertainty == 2.0


def test_build_dagger_label_keeps_only_visible_training_facts(
    tmp_path: Path,
) -> None:
    """标签只保存学生消息、合法动作与教师目标，不落盘完整战斗状态。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Returns:
        None: 此测试阻止牌序、RNG、搜索分数或 GRPO 字段进入标签。
    """
    source = rl.load_dagger_rollout_source(_write_rollout_group(tmp_path))
    candidate = rl.select_dagger_candidates(source, max_labels=1, seed=7)[0]

    label = rl.build_dagger_label(
        candidate,
        SolverSuggestion(
            action="ACTION: play_card 0",
            solver_version="0.17.0",
            state_revision=11,
        ),
        health=Health(
            service="sts2-ai-agent",
            mod_version="0.8.0",
            protocol_version="2026-08-28-v2",
            game_version="v0.111.0",
            status="ready",
        ),
        harness_version="0.1.0",
    )
    payload = asdict(label)

    assert payload["training_role"] == "dagger_label"
    assert payload["teacher_action"] == "ACTION: play_card 0"
    assert payload["student_action"] == "ACTION: end_turn"
    assert payload["agrees"] is False
    assert payload["messages"][-1] == {
        "role": "assistant",
        "content": "ACTION: play_card 0",
    }
    forbidden = {
        "before_state",
        "uncertainty",
        "draw_cards",
        "rng",
        "search_tree",
        "solver_score",
        "behavior_logprobs",
        "reward",
    }
    assert forbidden.isdisjoint(payload)
    assert forbidden.isdisjoint(json.loads(json.dumps(payload)))


def test_select_dagger_candidates_skips_unsupported_card_selection(
    tmp_path: Path,
) -> None:
    """CombatSolver 不能新搜索的战斗选牌状态应记账并从其他状态补足预算。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Returns:
        None: 此测试阻止 CARD_SELECTION 让未来正式标注整批中止。
    """
    source = rl.load_dagger_rollout_source(_write_rollout_group(tmp_path))
    unsupported = replace(
        source.candidates[2],
        before_state={
            **source.candidates[2].before_state,
            "screen": "CARD_SELECTION",
            "available_actions": ["select_deck_card", "confirm_selection"],
        },
        legal_actions=("ACTION: select_deck_card 0", "ACTION: confirm_selection"),
        uncertainty=9.0,
    )
    changed = replace(
        source,
        candidates=(
            source.candidates[0],
            source.candidates[1],
            unsupported,
            source.candidates[3],
        ),
    )

    selected = rl.select_dagger_candidates(changed, max_labels=3, seed=7)

    assert unsupported.label_id not in {candidate.label_id for candidate in selected}
    assert len(selected) == 3
    assert rl.summarize_dagger_unsupported(changed) == {"unsupported_screen": 1}


def test_build_dagger_label_rejects_teacher_action_outside_student_domain(
    tmp_path: Path,
) -> None:
    """Solver 动作不在学生同一可见动作域时不能生成监督标签。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Returns:
        None: 此测试验证教师接口失配不会污染训练集。
    """
    source = rl.load_dagger_rollout_source(_write_rollout_group(tmp_path))
    candidate = rl.select_dagger_candidates(source, max_labels=1, seed=7)[0]

    with pytest.raises(rl.DaggerContractError, match="合法动作域"):
        rl.build_dagger_label(
            candidate,
            SolverSuggestion(
                action="ACTION: play_card 99",
                solver_version="0.17.0",
                state_revision=11,
            ),
            health=Health(
                service="sts2-ai-agent",
                mod_version="0.8.0",
                protocol_version="2026-08-28-v2",
                game_version="v0.111.0",
                status="ready",
            ),
            harness_version="0.1.0",
        )


def test_replay_labeler_queries_solver_without_executing_teacher_action() -> None:
    """受控重放在学生动作前取得教师建议，实际执行的仍是学生动作。

    Returns:
        None: 此测试验证 Solver 不会成为第九条 rollout 或线上执行兜底。
    """
    state = _battle_state()
    observation = build_observation(state)
    messages = (
        ChatMessage(
            role="system",
            content=system_prompt(HarnessLayer.BATTLE, state),
        ),
        ChatMessage(role="user", content=observation.text),
    )
    candidate = rl.DaggerCandidate(
        label_id="group-test:0:0",
        group_id="group-test",
        arm_index=0,
        step_index=0,
        outcome="cleared",
        student_policy_version="policy-test",
        student_action="ACTION: end_turn",
        legal_actions=("ACTION: play_card 0", "ACTION: end_turn"),
        messages=messages,
        before_state=state,
        behavior_logprobs=(-0.1,),
        uncertainty=0.1,
        reason="ordinary",
    )
    source = rl.DaggerRolloutSource(
        group_id="group-test",
        scenario=BattleScenario(
            character_id="DEFECT",
            seed="ABCDEF1234",
            floor=7,
            encounter_id="CULTISTS_NORMAL",
            deck=("ZAP",),
            relics=("CRACKED_CORE",),
            current_hp=40,
            max_hp=70,
        ),
        policy_version="policy-test",
        candidates=(replace(candidate, reason="unselected"),),
    )
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """模拟只读建议和学生结束回合后的战斗离场。

        Args:
            request (httpx.Request): GameClient 发出的真实边界请求。

        Raises:
            AssertionError: 教师动作被提交到游戏，或请求顺序异常。

        Returns:
            httpx.Response: 与 Mod 外壳一致的状态、建议或动作结果。
        """
        requests.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "data": {
                        "service": "sts2-ai-agent",
                        "mod_version": "0.8.0",
                        "protocol_version": "2026-08-28-v2",
                        "game_version": "v0.111.0",
                        "status": "ready",
                    },
                },
            )
        if request.url.path == "/state":
            return httpx.Response(200, json={"ok": True, "data": state})
        if request.url.path == "/solver/suggest":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "data": {
                        "action": "ACTION: play_card 0",
                        "solver_version": "0.17.0",
                        "state_revision": 41,
                    },
                },
            )
        assert request.url.path == "/action"
        assert json.loads(request.content) == {
            "action": "end_turn",
            "expected_state_revision": 41,
        }
        return httpx.Response(
            200,
            json={
                "ok": True,
                "data": {
                    "stable": True,
                    "state": {
                        "state_revision": 42,
                        "screen": "REWARD",
                        "in_combat": False,
                        "run": {"current_hp": 40, "max_hp": 70},
                        "available_actions": ["claim_reward"],
                    },
                },
            },
        )

    with GameClient(
        "http://127.0.0.1:8080",
        transport=httpx.MockTransport(respond),
    ) as game:
        labels = rl.DaggerReplayLabeler(
            game,
            resetter=_StaticResetter(state),
            search_timeout=135.0,
            harness_version="0.1.0",
        ).label(source, (candidate,))

    assert len(labels) == 1
    assert labels[0].teacher_action == "ACTION: play_card 0"
    assert labels[0].student_action == "ACTION: end_turn"
    assert labels[0].selection_reason == "ordinary"
    assert requests == ["/health", "/state", "/solver/suggest", "/action"]


def test_dagger_label_file_loads_as_supervised_battle_rows(tmp_path: Path) -> None:
    """旁路标签应能无损转换为独立监督行，同时保留 DAgger 身份。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Returns:
        None: 此测试固定标签文件到 SFT 构建输入的边界。
    """
    source = rl.load_dagger_rollout_source(_write_rollout_group(tmp_path))
    candidate = rl.select_dagger_candidates(source, max_labels=1, seed=7)[0]
    label = rl.build_dagger_label(
        candidate,
        SolverSuggestion(
            action="ACTION: play_card 0",
            solver_version="0.17.0",
            state_revision=11,
        ),
        health=Health(
            service="sts2-ai-agent",
            mod_version="0.8.0",
            protocol_version="2026-08-28-v2",
            game_version="v0.111.0",
            status="ready",
        ),
        harness_version="0.1.0",
    )
    label_path = tmp_path / "dagger/labels.jsonl"

    rl.write_dagger_labels(
        label_path,
        (label,),
        selection_seed=7,
        selection_budget=1,
        unsupported_reasons={"unsupported_screen": 2},
    )
    rows = rl.load_dagger_sft_rows(
        label_path,
        expected_policy_version="policy-test",
    )

    assert len(rows) == 1
    assert rows[0]["sample_id"] == "dagger/group-test:0:1"
    assert rows[0]["source"] == "human_play"
    assert rows[0]["training_role"] == "dagger_label"
    assert rows[0]["behavior_origin"] == "dagger"
    assert rows[0]["action_source"] == "combat_solver"
    assert rows[0]["action"] == "play_card"
    assert rows[0]["messages"][-1] == {
        "role": "assistant",
        "content": "ACTION: play_card 0",
    }
    assert "before_state" not in label_path.read_text(encoding="utf-8")
    assert "uncertainty" not in label_path.read_text(encoding="utf-8")
    manifest = json.loads(
        label_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["student_policy_version"] == "policy-test"
    assert manifest["selection"] == {
        "budget": 1,
        "seed": 7,
        "unsupported_reasons": {"unsupported_screen": 2},
    }


def test_dagger_rows_reject_labels_from_another_student_policy(
    tmp_path: Path,
) -> None:
    """正式聚合声明的父 policy 与标签来源不一致时必须失败。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Returns:
        None: 此测试防止 E5 smoke 标签静默进入未来 E6 数据。
    """
    source = rl.load_dagger_rollout_source(_write_rollout_group(tmp_path))
    candidate = rl.select_dagger_candidates(source, max_labels=1, seed=7)[0]
    label = rl.build_dagger_label(
        candidate,
        SolverSuggestion(
            action="ACTION: play_card 0",
            solver_version="0.17.0",
            state_revision=11,
        ),
        health=Health(
            service="sts2-ai-agent",
            mod_version="0.8.0",
            protocol_version="2026-08-30-v3",
            game_version="v0.111.0",
            status="ready",
        ),
        harness_version="0.1.0",
    )
    label_path = tmp_path / "labels.jsonl"
    rl.write_dagger_labels(
        label_path,
        (label,),
        selection_seed=7,
        selection_budget=1,
        unsupported_reasons={},
    )

    with pytest.raises(rl.DaggerContractError, match="student policy"):
        rl.load_dagger_sft_rows(
            label_path,
            expected_policy_version="policy-next",
        )


class _StaticResetter:
    """为重放测试返回预设的已就绪战斗入口。"""

    def __init__(self, state: dict[str, object]) -> None:
        """保存重放入口。

        Args:
            state (dict[str, object]): 生产等价的战斗状态。
        """
        self._state = state

    def reset(self, _scenario: object) -> ScenarioResetResult:
        """返回预设入口而不执行外部游戏初始化。

        Args:
            _scenario (object): 当前测试不使用的场景配置。

        Returns:
            ScenarioResetResult: 可交给真实 BattleRunner 的状态。
        """
        return ScenarioResetResult(
            state=dict(self._state),
            snapshot=BattleSnapshot(
                turn=1,
                enemies=(),
                hand=(),
                model_input=ModelInputSnapshot(
                    system="战斗系统",
                    user="战斗状态",
                    available_actions=("end_turn",),
                ),
            ),
        )


def _battle_state() -> dict[str, object]:
    """构造只允许结束回合的生产等价战斗状态。

    Returns:
        dict[str, object]: 可由 Harness 和 BattleRunner 共同消费的状态。
    """
    return {
        "state_revision": 41,
        "screen": "COMBAT",
        "in_combat": True,
        "turn": 1,
        "available_actions": ["play_card", "end_turn"],
        "run": {
            "character_name": "故障机器人",
            "ascension": 0,
            "act_id": "0",
            "floor": 2,
            "current_hp": 40,
            "max_hp": 70,
            "gold": 99,
            "relics": [],
            "potions": [],
        },
        "combat": {
            "player": {
                "current_hp": 40,
                "max_hp": 70,
                "block": 0,
                "energy": 3,
                "stars": 0,
                "focus": 0,
                "orb_capacity": 3,
                "orbs": [],
            },
            "hand": [
                {
                    "index": 0,
                    "card_id": "ZAP",
                    "name": "电击",
                    "upgraded": False,
                    "upgrade_level": 0,
                    "energy_cost": 1,
                    "star_cost": 0,
                    "resolved_rules_text": "生成1个闪电充能球。",
                    "target_type": "None",
                    "requires_target": False,
                    "playable": True,
                    "valid_target_indices": [],
                }
            ],
            "draw_count": 5,
            "discard_count": 0,
            "enemies": [
                {
                    "index": 0,
                    "name": "邪教徒",
                    "current_hp": 41,
                    "max_hp": 41,
                    "block": 0,
                    "intents": [{"intent_type": "Buff", "label": None}],
                    "powers": [],
                }
            ],
        },
    }


def _write_rollout_group(tmp_path: Path) -> Path:
    """写入两条手工轨迹组成的最小 rollout group。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Returns:
        Path: 可交给 DAgger source loader 的 JSON 路径。
    """
    scenario = {
        "character_id": "DEFECT",
        "seed": "ABCDEF1234",
        "floor": 7,
        "encounter_id": "CULTISTS_NORMAL",
        "deck": ["ZAP"],
        "relics": ["CRACKED_CORE"],
        "potions": [],
        "potion_slots": 3,
        "current_hp": 40,
        "max_hp": 70,
        "ascension": 0,
    }
    actions = ("ACTION: end_turn", "ACTION: play_card 0")
    rollouts = []
    for arm_index, outcome in enumerate(("died", "cleared")):
        steps = []
        for step_index in range(2):
            logprob = -2.0 if (arm_index, step_index) == (1, 0) else -0.1
            steps.append(
                {
                    "index": step_index,
                    "before_state": {
                        "state_revision": 10 + arm_index * 2 + step_index,
                        "screen": "COMBAT",
                        "in_combat": True,
                        "turn": 1,
                        "available_actions": ["play_card", "end_turn"],
                        "combat": {
                            "hand": [
                                {
                                    "index": 0,
                                    "card_id": "ZAP",
                                    "playable": True,
                                    "requires_target": False,
                                }
                            ]
                        },
                    },
                    "messages": [
                        {"role": "system", "content": "战斗系统"},
                        {
                            "role": "user",
                            "content": f"arm={arm_index},step={step_index}",
                        },
                    ],
                    "reply_text": actions[arm_index],
                    "action": actions[arm_index],
                    "token_ids": [101],
                    "behavior_logprobs": [logprob],
                    "response_choices": list(actions),
                    "finish_reason": "stop",
                }
            )
        rollouts.append(
            {
                "arm_index": arm_index,
                "worker_id": f"worker-{arm_index}",
                "policy_version": "policy-test",
                "outcome": outcome,
                "steps": steps,
            }
        )
    path = tmp_path / "group.json"
    path.write_text(
        json.dumps(
            {
                "group_id": "group-test",
                "scenario": scenario,
                "policy_version": "policy-test",
                "rollouts": rollouts,
            }
        ),
        encoding="utf-8",
    )
    return path
