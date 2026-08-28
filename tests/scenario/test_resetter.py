"""验证从任意残局重置到确定性战斗场景的编排。"""

from typing import Any

from play_sts2.scenario import BattleResetter, BattleScenario


class ScriptedClient:
    """记录场景重置器发出的动作并返回对应状态。"""

    action_timeout = 1.0

    def __init__(self) -> None:
        """初始化一局仍在战斗中的残局。

        Returns:
            None: 此方法只准备测试状态与动作日志。
        """
        self.actions: list[tuple[str, dict[str, Any]]] = []
        self._battle_is_settling = False
        self._pending_states: list[dict[str, Any]] = []
        self._state = {
            "screen": "COMBAT",
            "in_combat": True,
            "available_actions": ["save_and_quit"],
            "run": {"character_id": "IRONCLAD"},
        }

    def state(self) -> dict[str, Any]:
        """返回当前脚本状态。

        Returns:
            dict[str, Any]: 当前测试游戏状态的浅副本。
        """
        if self._pending_states:
            self._state = self._pending_states.pop(0)
        elif self._battle_is_settling:
            self._battle_is_settling = False
            self._state["available_actions"] = [
                "play_card",
                "end_turn",
                "save_and_quit",
            ]
        return dict(self._state)

    def execute_action(self, action: str, **parameters: Any) -> dict[str, Any]:
        """执行重置流程需要的最小动作集合。

        Args:
            action (str): 待执行的游戏动作。
            parameters (Any): 动作携带的协议参数。

        Raises:
            AssertionError: 重置器发出不属于预期流程的动作。

        Returns:
            dict[str, Any]: 动作完成后的稳定状态结果。
        """
        self.actions.append((action, parameters))
        response_state: dict[str, Any] | None = None
        if action == "save_and_quit":
            response_state = dict(self._state)
            self._pending_states = [
                {"screen": "UNKNOWN", "available_actions": []},
                {
                    "screen": "MAIN_MENU",
                    "available_actions": ["abandon_run"],
                },
            ]
        elif action == "abandon_run":
            self._state = {
                "screen": "MAIN_MENU",
                "available_actions": ["confirm_modal"],
            }
        elif action == "confirm_modal":
            self._state = {
                "screen": "MAIN_MENU",
                "available_actions": ["open_character_select"],
            }
        elif action == "open_character_select":
            self._state = {
                "screen": "CHARACTER_SELECT",
                "available_actions": ["select_character", "set_seed", "embark"],
                "character_select": {
                    "selected_character_id": "IRONCLAD",
                    "ascension": 0,
                    "max_ascension": 0,
                    "seed": None,
                    "characters": [
                        {
                            "character_id": "DEFECT",
                            "index": 4,
                            "is_locked": False,
                        }
                    ],
                },
            }
        elif action == "select_character":
            self._state["character_select"]["selected_character_id"] = "DEFECT"
        elif action == "set_seed":
            self._state["character_select"]["seed"] = parameters["game_seed"]
        elif action == "embark":
            self._state = {
                "screen": "EVENT",
                "available_actions": ["choose_event_option", "save_and_quit"],
                "run": {
                    "character_id": "DEFECT",
                    "ascension": 0,
                    "floor": 0,
                    "potions": [],
                },
            }
        elif action == "choose_event_option":
            response_state = dict(self._state)
            map_state = {
                **self._state,
                "screen": "MAP",
                "available_actions": [
                    "choose_map_node",
                    "discard_potion",
                    "run_console_command",
                ],
            }
            self._pending_states = [
                {
                    **self._state,
                    "screen": "UNKNOWN",
                    "available_actions": [],
                },
                map_state,
            ]
        elif action == "discard_potion":
            self._state["run"]["potions"] = []
        elif action == "run_console_command":
            command = parameters["command"]
            if command.startswith("scenariofight"):
                self._state = _combat_state(current_hp=70, max_hp=70)
                self._state["available_actions"] = ["save_and_quit"]
                self._battle_is_settling = True
                response_state = self._state
            elif command.startswith("loadout hp="):
                assert "end_turn" in self._state["available_actions"]
                self._state = _combat_state(current_hp=41, max_hp=70)
            else:
                assert command.startswith("loadout cards=")
        else:
            raise AssertionError(f"预期外动作: {action}")
        pending = action in {"save_and_quit", "choose_event_option"} or (
            action == "run_console_command"
            and parameters["command"].startswith("scenariofight")
        )
        return {
            "action": action,
            "status": "pending" if pending else "completed",
            "stable": not pending,
            "state": response_state or self._state,
        }


def test_reset_replaces_stale_run_and_returns_verified_battle() -> None:
    """清理残局、结算初始事件、装载配置并进入已核对的战斗。

    Raises:
        AssertionError: 重置动作顺序、控制台命令或最终状态不正确。

    Returns:
        None: 此测试仅验证完整的场景重置闭环。
    """
    client = ScriptedClient()
    scenario = BattleScenario(
        character_id="DEFECT",
        seed="ABCDEF1234",
        floor=7,
        encounter_id="CULTISTS_NORMAL",
        deck=("ZAP+1", "STRIKE_DEFECTx2"),
        relics=("CRACKED_CORE",),
        potion_slots=1,
        current_hp=41,
        max_hp=70,
    )

    result = BattleResetter(client).reset(scenario)

    assert result.state["screen"] == "COMBAT"
    assert result.snapshot.enemies[0].enemy_id == "DAMP_CULTIST"
    assert client.actions == [
        ("save_and_quit", {}),
        ("abandon_run", {}),
        ("confirm_modal", {}),
        ("open_character_select", {}),
        ("select_character", {"option_index": 4}),
        ("set_seed", {"game_seed": "ABCDEF1234"}),
        ("embark", {}),
        (
            "run_console_command",
            {
                "command": (
                    "loadout cards=ZAP+1,STRIKE_DEFECTx2 "
                    "relics=CRACKED_CORE potions=_ potion_slots=1"
                )
            },
        ),
        (
            "run_console_command",
            {"command": "scenariofight CULTISTS_NORMAL floor=7"},
        ),
        ("run_console_command", {"command": "loadout hp=41/70"}),
    ]


def _combat_state(*, current_hp: int, max_hp: int) -> dict[str, Any]:
    """创建脚本重置器最终应返回的战斗状态。

    Args:
        current_hp (int): 当前生命值。
        max_hp (int): 最大生命值。

    Returns:
        dict[str, Any]: 已装载场景内容的战斗状态。
    """
    return {
        "screen": "COMBAT",
        "in_combat": True,
        "turn": 1,
        "available_actions": ["play_card", "end_turn", "save_and_quit"],
        "run": {
            "character_id": "DEFECT",
            "ascension": 0,
            "floor": 0,
            "current_hp": current_hp,
            "max_hp": max_hp,
            "deck": [
                {
                    "index": 0,
                    "card_id": "ZAP",
                    "upgraded": True,
                    "upgrade_level": 1,
                },
                {
                    "index": 1,
                    "card_id": "STRIKE_DEFECT",
                    "upgraded": False,
                    "upgrade_level": 0,
                },
                {
                    "index": 2,
                    "card_id": "STRIKE_DEFECT",
                    "upgraded": False,
                    "upgrade_level": 0,
                },
            ],
            "relics": [{"index": 0, "relic_id": "CRACKED_CORE", "is_melted": False}],
            "potions": [
                {"index": 0, "potion_id": None, "occupied": False},
            ],
        },
        "combat": {
            "player": {
                "current_hp": current_hp,
                "max_hp": max_hp,
                "energy": 3,
                "block": 0,
            },
            "hand": [
                {
                    "index": 0,
                    "card_id": "STRIKE_DEFECT",
                    "upgraded": False,
                    "upgrade_level": 0,
                },
                {
                    "index": 1,
                    "card_id": "ZAP",
                    "upgraded": True,
                    "upgrade_level": 1,
                },
            ],
            "enemies": [
                {
                    "index": 0,
                    "enemy_id": "DAMP_CULTIST",
                    "current_hp": 48,
                    "max_hp": 48,
                    "move_id": "INCANTATION",
                    "intents": [{"index": 0, "intent_type": "Buff"}],
                }
            ],
        },
    }
