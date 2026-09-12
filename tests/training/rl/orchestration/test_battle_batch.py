"""验证扩大场景批次按有效组计数，拒绝组不能提前耗尽目标。"""

import json
import shutil
from pathlib import Path

import pytest


def test_batch_fills_valid_quota_after_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """首组零方差后继续采其他入口，满两个有效组后停止。"""
    from play_sts2.training.rl.contracts import BattleGroupRejected
    from play_sts2.training.rl.orchestration import battle_batch

    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "battle_policy_version": "b3",
                "selected": [
                    {"scenario": str(tmp_path / f"scene-{i}.json")} for i in range(4)
                ],
            }
        )
    )

    def collect_one(**kwargs: object) -> dict[str, object]:
        """替代外部游戏执行，仅模拟真实 collector 的成功与准入拒绝。"""
        if Path(kwargs["scenario_path"]).stem == "scene-0":
            raise BattleGroupRejected("战斗 group 的奖励没有方差")
        return {"arms": 8, "reward_std": 0.1, "output": str(kwargs["output_path"])}

    monkeypatch.setattr(battle_batch, "_collect_one", collect_one)
    monkeypatch.setattr(
        battle_batch, "check_online_serving", lambda *args: {"b3": "models/b3"}
    )
    result = battle_batch.collect_selected_battles(
        selection_path=selection,
        output_root=tmp_path / "batch",
        target_groups=2,
        executable=tmp_path / "game",
        profile=tmp_path / "profile",
        ports=(8081, 8083),
        model_url="http://localhost:8900",
        expected_bindings={"b3": "models/b3"},
    )
    assert result["accepted_groups"] == 2
    assert result["attempted_groups"] == 3
    assert result["status"] == "completed"
    assert (
        len(json.loads((tmp_path / "batch/report.json").read_text())["rejected"]) == 1
    )


def test_launch_timeout_is_rejected_and_batch_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """游戏启动超时记为基础设施拒绝，批次继续采满目标。"""
    from play_sts2.training.rl.orchestration import battle_batch

    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "battle_policy_version": "b3",
                "selected": [
                    {"scenario": str(tmp_path / f"scene-{i}.json")} for i in range(3)
                ],
            }
        )
    )

    def collect_one(**kwargs: object) -> dict[str, object]:
        """首个入口模拟 STS2 未就绪超时，其余入口正常返回有效组。"""
        if Path(kwargs["scenario_path"]).stem == "scene-0":
            raise TimeoutError("STS2 在 120 秒内未就绪")
        return {"arms": 8, "reward_std": 0.1, "output": str(kwargs["output_path"])}

    monkeypatch.setattr(battle_batch, "_collect_one", collect_one)
    monkeypatch.setattr(
        battle_batch, "check_online_serving", lambda *args: {"b3": "models/b3"}
    )
    result = battle_batch.collect_selected_battles(
        selection_path=selection,
        output_root=tmp_path / "batch",
        target_groups=1,
        executable=tmp_path / "game",
        profile=tmp_path / "profile",
        ports=(8081,),
        model_url="http://localhost:8900",
        expected_bindings={"b3": "models/b3"},
    )
    assert result["accepted_groups"] == 1
    assert result["status"] == "completed"
    assert result["rejected"][0]["type"] == "TimeoutError"
    assert result["rejected"][0]["reason"] == "STS2 在 120 秒内未就绪"


def test_no_turn_batch_does_not_count_death_timing_as_useful_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """实际全灭组在去回合配方下必须隔离，不能占有效训练配额。"""
    from play_sts2.training.rl.orchestration import battle_batch

    fixture = Path(__file__).parents[1] / "fixtures/battle-death-timing.json"
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "battle_policy_version": "qwen3.5-e7-b3",
                "selected": [{"scenario": "death-scene.json"}],
            }
        )
    )

    def collect_one(**kwargs: object) -> dict[str, object]:
        """复制真实组的奖励事实投影，保留实际死亡回合差异。"""
        shutil.copyfile(fixture, kwargs["output_path"])
        return {"arms": 8, "reward_std": 0.0165, "output": str(kwargs["output_path"])}

    monkeypatch.setattr(battle_batch, "_collect_one", collect_one)
    monkeypatch.setattr(battle_batch, "check_online_serving", lambda *args: {})
    result = battle_batch.collect_selected_battles(
        selection_path=selection,
        output_root=tmp_path / "batch",
        target_groups=1,
        executable=tmp_path / "game",
        profile=tmp_path / "profile",
        ports=(8085, 8086),
        model_url="http://localhost:18900",
        expected_bindings={"qwen3.5-e7-b3": "models/b3"},
        reward_scheme="core_no_turn",
    )
    assert result["accepted_groups"] == 0
    assert result["status"] == "insufficient_groups"
    assert not list((tmp_path / "batch/battle").glob("*.json"))
    assert (
        tmp_path / "batch/rejected/group-000.json"
    ).read_bytes() == fixture.read_bytes()
