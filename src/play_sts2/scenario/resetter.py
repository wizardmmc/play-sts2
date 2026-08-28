"""把任意现有游戏状态重置为已核对的战斗场景。"""

import time
from collections.abc import Mapping
from typing import Any

from ..client import GameClient
from ..run_start import start_run
from .models import BattleScenario, BattleSnapshot, ScenarioResetResult
from .verifier import verify_battle_scenario


class BattleResetError(RuntimeError):
    """表示现有游戏状态无法被重置为新战斗场景。"""


class BattleResetter:
    """编排残局清理、精确装载、战斗跳转和入口核对。"""

    def __init__(self, client: GameClient) -> None:
        """保存不由重置器拥有生命周期的游戏客户端。

        Args:
            client (GameClient): 已连接到启用开发动作的 STS2 Mod 客户端。

        Returns:
            None: 此方法只保存游戏客户端引用。
        """
        self._client = client

    def reset(
        self,
        scenario: BattleScenario,
        *,
        expected_snapshot: BattleSnapshot | None = None,
    ) -> ScenarioResetResult:
        """从任意残局创建一场可直接交给模型或采样器的战斗。

        Args:
            scenario (BattleScenario): 待创建的完整战斗入口。
            expected_snapshot (BattleSnapshot | None): 可选的首次采样基准快照。

        Raises:
            BattleResetError: 无法清理旧局、排队动作超时或响应缺少状态。
            RunStartError: 新局角色、种子或进阶无法正确设置。
            ScenarioVerificationError: 最终装载或战斗入口与场景不一致。
            httpx.HTTPStatusError: Mod 拒绝任一步骤。

        Returns:
            ScenarioResetResult: 已核对的完整状态和战斗入口快照。
        """
        self._clear_existing_run()
        state = start_run(
            self._client,
            scenario.character_id,
            seed=scenario.seed,
            ascension=scenario.ascension,
        )
        state = self._execute_console(_loadout_command(scenario))
        state = self._execute_console(
            f"scenariofight {scenario.encounter_id} floor={scenario.floor}"
        )
        state = self._await_battle_ready(state)
        state = self._execute_console(
            f"loadout hp={scenario.current_hp}/{scenario.max_hp}"
        )
        state = self._await_battle_ready(state)
        snapshot = verify_battle_scenario(
            scenario,
            state,
            expected_snapshot=expected_snapshot,
        )
        return ScenarioResetResult(state=state, snapshot=snapshot)

    def _clear_existing_run(self) -> None:
        """依次存退、放弃并确认现有残局，直到回到干净主菜单。

        Raises:
            BattleResetError: 当前状态没有通向干净主菜单的已知动作。

        Returns:
            None: 游戏已允许打开角色选择页时返回。
        """
        state = self._client.state()
        while True:
            actions = state.get("available_actions")
            if not isinstance(actions, list):
                raise BattleResetError("当前状态没有可用动作列表")
            if (
                state.get("screen") == "MAIN_MENU"
                and "open_character_select" in actions
            ):
                return
            action = next(
                (
                    candidate
                    for candidate in (
                        "return_to_main_menu",
                        "save_and_quit",
                        "abandon_run",
                        "confirm_modal",
                    )
                    if candidate in actions
                ),
                None,
            )
            if action is None:
                raise BattleResetError("无法清理当前游戏状态")
            result = self._client.execute_action(action)
            state = _action_state(result, action)
            if result.get("stable") is not True:
                state = self._await_cleanup_phase(state, action)

    def _execute_console(self, command: str) -> dict[str, Any]:
        """执行一条开发控制台命令并提取其稳定状态。

        Args:
            command (str): 待交给 Mod 的完整控制台命令。

        Raises:
            BattleResetError: 命令响应没有可用状态。

        Returns:
            dict[str, Any]: 控制台命令完成后的稳定游戏状态。
        """
        return _action_state(
            self._client.execute_action("run_console_command", command=command),
            command,
        )

    def _await_cleanup_phase(
        self,
        candidate: Mapping[str, Any],
        action: str,
    ) -> dict[str, Any]:
        """等待已排队的残局清理动作到达明确的下一操作阶段。

        Args:
            candidate (Mapping[str, Any]): 动作响应携带的首份状态。
            action (str): 已提交且正在等待完成的动作名称。

        Raises:
            BattleResetError: 动作超时内没有到达预期的下一清理阶段。

        Returns:
            dict[str, Any]: 已暴露下一项清理或开局动作的游戏状态。
        """
        next_actions = {
            "return_to_main_menu": {"abandon_run", "open_character_select"},
            "save_and_quit": {"abandon_run", "open_character_select"},
            "abandon_run": {"confirm_modal", "open_character_select"},
            "confirm_modal": {"open_character_select"},
        }[action]
        deadline = time.monotonic() + self._client.action_timeout
        state = dict(candidate)
        while time.monotonic() < deadline:
            actions = state.get("available_actions")
            if isinstance(actions, list) and next_actions.intersection(actions):
                return state
            state = self._client.state()
            if time.monotonic() < deadline:
                time.sleep(0.2)
        raise BattleResetError(f"等待动作状态迁移超时: {action}")

    def _await_battle_ready(
        self,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        """等待开场动作队列结束并暴露玩家战斗动作。

        Args:
            candidate (Mapping[str, Any]): 控制台命令返回的首份战斗状态。

        Raises:
            BattleResetError: 动作超时内没有到达玩家可决策的战斗状态。

        Returns:
            dict[str, Any]: 已暴露结束回合动作的战斗状态。
        """
        deadline = time.monotonic() + self._client.action_timeout
        state = dict(candidate)
        while time.monotonic() < deadline:
            actions = state.get("available_actions")
            if (
                state.get("screen") == "COMBAT"
                and state.get("in_combat") is True
                and isinstance(actions, list)
                and "end_turn" in actions
            ):
                return state
            state = self._client.state()
            if time.monotonic() < deadline:
                time.sleep(0.2)
        raise BattleResetError("等待战斗入口可操作状态超时")


def _loadout_command(scenario: BattleScenario) -> str:
    """把场景装载字段翻译为一条原子的 ``loadout`` 命令。

    Args:
        scenario (BattleScenario): 待创建的完整战斗入口。

    Returns:
        str: 不包含生命值的控制台装载命令。
    """
    parts = ["loadout", f"cards={','.join(scenario.deck)}"]
    if scenario.relics:
        parts.append(f"relics={','.join(scenario.relics)}")
    if scenario.potion_slots is not None:
        potion_slots = scenario.potions + (None,) * (
            scenario.potion_slots - len(scenario.potions)
        )
        parts.append(
            "potions=" + ",".join(potion or "_" for potion in potion_slots)
        )
    elif scenario.potions:
        parts.append(f"potions={','.join(scenario.potions)}")
    if scenario.potion_slots is not None:
        parts.append(f"potion_slots={scenario.potion_slots}")
    return " ".join(parts)


def _action_state(result: Mapping[str, Any], action: str) -> dict[str, Any]:
    """从动作结果中提取一份稳定状态。

    Args:
        result (Mapping[str, Any]): Mod 返回的动作结果。
        action (str): 用于错误信息的动作或控制台命令。

    Raises:
        BattleResetError: 动作既未完成也未排队，或状态对象缺失。

    Returns:
        dict[str, Any]: 动作响应携带的最新游戏状态。
    """
    state = result.get("state")
    accepted = result.get("stable") is True or result.get("status") == "pending"
    if not accepted or not isinstance(state, Mapping):
        raise BattleResetError(f"动作结果不可用: {action}")
    return dict(state)
