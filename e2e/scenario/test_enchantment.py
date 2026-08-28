"""验证真实游戏场景可重复装载附魔和药水槽位。"""

import pytest

from play_sts2.client import GameClient
from play_sts2.scenario import BattleResetter, BattleScenario

from ..conftest import RunningGame

pytestmark = pytest.mark.e2e


def test_battle_scenario_restores_enchantment_and_potion_slots(
    running_game: RunningGame,
) -> None:
    """场景装载恢复真实附魔，并保留药水栏中的空槽位置。"""
    scenario = BattleScenario(
        character_id="DEFECT",
        seed="ABCDEF1234",
        floor=7,
        encounter_id="MOCK_MONSTER_ENCOUNTER",
        deck=("CHARGE_BATTERY@NIMBLE:2x10",),
        relics=("CRACKED_CORE",),
        potions=(None, "FIRE_POTION", None),
        potion_slots=3,
        current_hp=70,
        max_hp=70,
    )

    with GameClient(running_game.base_url) as client:
        state = BattleResetter(client).reset(scenario).state

    assert all(card["enchantment_id"] == "NIMBLE" for card in state["run"]["deck"])
    assert all(card["enchantment_amount"] == 2 for card in state["run"]["deck"])
    assert [slot["potion_id"] for slot in state["run"]["potions"]] == [
        None,
        "FIRE_POTION",
        None,
    ]
    block_values = [
        value
        for card in state["combat"]["hand"]
        for value in card["dynamic_values"]
        if value["name"] == "Block"
    ]
    assert block_values
    assert all(value["enchanted_value"] == 9 for value in block_values)
