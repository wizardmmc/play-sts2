"""验证 v0.111.0 真实战略状态满足 Harness 输入边界。"""

import json
from pathlib import Path

import pytest

from play_sts2 import start_run
from play_sts2.client import GameClient
from play_sts2.harness import build_observation
from play_sts2.recording import HumanRunWriter, RunMetadata
from play_sts2.transcription import render_run

from .conftest import RunningGame

pytestmark = pytest.mark.e2e


def test_rest_state_keeps_map_card_instances_and_dynamic_relic_options(
    running_game: RunningGame,
    tmp_path: Path,
) -> None:
    """非地图战略页仍导出完整地图、逐卡状态和游戏同步器的休息选项。

    Args:
        running_game (RunningGame): v0.111.0 隔离无头游戏实例。
        tmp_path (Path): 四层审计使用的临时 raw/transcript 根目录。

    Returns:
        None: Mod 四类玩家可见事实都能进入同一稳定 `/state`。
    """
    with GameClient(running_game.base_url) as game:
        health = game.health()
        state = start_run(game, "DEFECT", seed="ABCDEF1234", ascension=0)
        state = game.execute_action(
            "run_console_command",
            command=(
                "loadout cards=CHARGE_BATTERY@ADROIT:2,ZAP+1 "
                "relics=GIRYA,MEAT_CLEAVER,SHOVEL,MINIATURE_TENT,"
                "PUMPKIN_CANDLE,WINGED_BOOTS,TOY_BOX,BLOOD_VIAL:m"
            ),
        )["state"]
        state = game.execute_action(
            "run_console_command",
            command="room RestSite",
        )["state"]

    assert state["screen"] == "REST"
    assert health.game_version == "v0.111.0"
    assert health.mod_version == "0.8.0-rlsts2.50"
    assert state["map"]["current_node"] is not None
    assert len(state["map"]["nodes"]) == 59
    assert state["map"]["available_nodes"] == []
    assert any(node["node_type"] == "Unknown" for node in state["map"]["nodes"])
    assert state["run"]["character_id"] == "DEFECT"
    assert state["run"]["max_energy"] == 3
    assert state["run"]["base_orb_slots"] == 3
    assert state["run"]["deck"][0]["enchantment_id"] == "ADROIT"
    assert state["run"]["deck"][0]["enchantment_name"] == "伶俐"
    assert state["run"]["deck"][0]["enchantment_amount"] == 2
    assert state["run"]["deck"][0]["dynamic_values"]
    melted = next(
        relic for relic in state["run"]["relics"] if relic["name"] == "蜡制小血瓶"
    )
    assert melted["is_melted"] is True
    assert melted["status"] == "Disabled"
    relics = {relic["relic_id"]: relic for relic in state["run"]["relics"]}
    assert relics["WINGED_BOOTS"]["counter_value"] == 3
    assert relics["TOY_BOX"]["counter_value"] == 0
    assert relics["PUMPKIN_CANDLE"]["counter_value"] == 0
    assert relics["MINIATURE_TENT"]["is_used_up"] is False
    option_ids = {option["option_id"] for option in state["rest"]["options"]}
    assert {"HEAL", "SMITH", "LIFT", "COOK", "DIG", "KINDLE"} <= option_ids

    observation = build_observation(state)
    assert "=== 全图(坐标邻接, 未走层) ===" in observation.text
    assert "〔附魔：伶俐×2〕" in observation.text
    assert "蜡制小血瓶〔已熔毁；效果失效；当前禁用〕" in observation.text
    assert "本幕可达: 无" not in observation.text
    assert "[5] 添火 | 为南瓜蜡烛添加5层充能。" in observation.text

    dig_index = next(
        option["index"]
        for option in state["rest"]["options"]
        if option["option_id"] == "DIG"
    )
    writer = HumanRunWriter(
        tmp_path / "raw",
        RunMetadata(
            run_id="STRATEGIC-STATE-AUDIT",
            source="human",
            started_at="2026-08-30T05:00:00Z",
            character_id="DEFECT",
            seed="STRATEGIC-STATE-AUDIT",
            ascension=0,
        ),
    )
    raw_path = writer.append_decision(
        {
            "run_id": "STRATEGIC-STATE-AUDIT",
            "event_id": "strategic-state-rest",
            "observed_at": "2026-08-30T05:00:01Z",
            "before_state": state,
            "action": "choose_rest_option",
            "parameters": {"option_index": dig_index},
            "recorded_layer": "strategic",
        }
    )
    raw_row = json.loads(raw_path.read_text(encoding="utf-8"))
    assert raw_row["before_state"] == state
    writer.finalize(
        "interrupted",
        completed_at="2026-08-30T05:00:02Z",
    )
    transcript = render_run(writer.run_dir, tmp_path / "transcripts")
    transcript_text = (transcript.output_dir / "strategy/decisions.txt").read_text(
        encoding="utf-8"
    )
    assert "=== 全图(坐标邻接, 未走层) ===" in transcript_text
    assert "〔附魔：伶俐×2〕" in transcript_text
    assert "蜡制小血瓶〔已熔毁；效果失效；当前禁用〕" in transcript_text
    assert f"ACTION: choose_rest_option {dig_index}" in transcript_text

    with GameClient(running_game.base_url) as game:
        smith_index = next(
            option["index"]
            for option in state["rest"]["options"]
            if option["option_id"] == "SMITH"
        )
        upgrade_state = game.execute_action(
            "choose_rest_option",
            expected_state_revision=state["state_revision"],
            option_index=smith_index,
        )["state"]
        assert upgrade_state["screen"] == "CARD_SELECTION"
        assert upgrade_state["selection"]["kind"] == "deck_upgrade_select"
        charge_battery = next(
            card
            for card in upgrade_state["selection"]["cards"]
            if card["card_id"] == "CHARGE_BATTERY"
        )
        assert charge_battery["upgrade_preview"]["upgraded"] is True
        assert charge_battery["upgrade_preview"]["resolved_rules_text"]
        upgrade_observation = build_observation(upgrade_state)
        assert "以下展示升级后效果" in upgrade_observation.text
        assert "CHARGE_BATTERY" not in upgrade_observation.text
        refreshed_upgrade_state = game.state()
        original_charge_battery = next(
            card
            for card in refreshed_upgrade_state["run"]["deck"]
            if card["card_id"] == "CHARGE_BATTERY"
        )
        assert original_charge_battery["upgraded"] is False
        assert original_charge_battery["upgrade_level"] == 0
        assert (
            original_charge_battery["resolved_rules_text"]
            == (state["run"]["deck"][0]["resolved_rules_text"])
        )

        state = game.execute_action(
            "run_console_command",
            command="room RestSite",
        )["state"]
        state = game.execute_action(
            "run_console_command",
            command="scenariofight MOCK_MONSTER_ENCOUNTER floor=7",
        )["state"]
        state = game.execute_action(
            "run_console_command",
            command="win",
        )["state"]
        state = game.execute_action(
            "run_console_command",
            command="room RestSite",
        )["state"]
        pumpkin = next(
            relic
            for relic in state["run"]["relics"]
            if relic["relic_id"] == "PUMPKIN_CANDLE"
        )
        assert pumpkin["status"] == "Disabled"
        disabled_observation = build_observation(state)
        assert "南瓜蜡烛〔计数0；当前禁用〕" in disabled_observation.text
        assert "可以在休息处为其添火" in disabled_observation.text
        kindle_index = next(
            option["index"]
            for option in state["rest"]["options"]
            if option["option_id"] == "KINDLE"
        )
        state = game.execute_action(
            "choose_rest_option",
            expected_state_revision=state["state_revision"],
            option_index=kindle_index,
        )["state"]
    rekindled = next(
        relic
        for relic in state["run"]["relics"]
        if relic["relic_id"] == "PUMPKIN_CANDLE"
    )
    assert rekindled["counter_value"] == 5
    assert rekindled["status"] == "Normal"
