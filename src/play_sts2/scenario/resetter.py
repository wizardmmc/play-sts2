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
        state = self._settle_opening(state)
        state = self._clear_potions(state)
        state = self._execute_console(_loadout_command(scenario))
        state = self._execute_console(
            f"scenariofight {scenario.encounter_id} floor={scenario.floor}"
        )
        state = self._await_battle_ready(state)
        if scenario.current_hp is not None:
            hp = str(scenario.current_hp)
            if scenario.max_hp is not None:
                hp += f"/{scenario.max_hp}"
            state = self._execute_console(f"loadout hp={hp}")
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

    def _settle_opening(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """在 Neow 初始事件中选择固定选项，随后交给装载覆盖其奖励。

        Args:
            state (Mapping[str, Any]): 新局建立后的第一份稳定状态。

        Raises:
            BattleResetError: 初始事件动作没有返回稳定状态。

        Returns:
            dict[str, Any]: 初始事件结算后的状态，或原本非事件状态。
        """
        if state.get("screen") != "EVENT":
            return dict(state)
        actions = state.get("available_actions")
        if not isinstance(actions, list) or "choose_event_option" not in actions:
            raise BattleResetError("初始事件没有可选项")
        result = self._client.execute_action("choose_event_option", option_index=0)
        settled = _action_state(
            result,
            "choose_event_option",
        )
        if result.get("stable") is not True:
            settled = self._await_opening_settled(settled)
        return settled

    def _clear_potions(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """清空初始事件可能赠送的药水，避免污染精确装载。

        Args:
            state (Mapping[str, Any]): 初始事件结算后的状态。

        Raises:
            BattleResetError: 药水栏形态无效或丢弃动作没有稳定完成。

        Returns:
            dict[str, Any]: 所有已占用药水槽被清空后的状态。
        """
        current = dict(state)
        run = current.get("run")
        if not isinstance(run, Mapping):
            raise BattleResetError("新局状态不可用")
        potions = run.get("potions")
        if potions is None:
            return current
        if not isinstance(potions, list):
            raise BattleResetError("药水栏状态不可用")
        occupied = [
            potion.get("index")
            for potion in potions
            if isinstance(potion, Mapping)
            and (potion.get("occupied") is True or potion.get("potion_id") is not None)
        ]
        if any(
            isinstance(index, bool) or not isinstance(index, int) for index in occupied
        ):
            raise BattleResetError("药水栏索引不可用")
        for index in sorted(occupied, reverse=True):
            result = self._client.execute_action("discard_potion", option_index=index)
            current = _action_state(
                result,
                "discard_potion",
            )
            if result.get("stable") is not True:
                current = self._await_potion_discarded(current, index)
        return current

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

    def _await_opening_settled(
        self,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        """等待 Neow 初始事件完成并回到可选择节点的地图。

        Args:
            candidate (Mapping[str, Any]): 事件选项响应携带的首份状态。

        Raises:
            BattleResetError: 动作超时内没有回到可操作地图。

        Returns:
            dict[str, Any]: 已允许选择地图节点的稳定新局状态。
        """
        deadline = time.monotonic() + self._client.action_timeout
        state = dict(candidate)
        while time.monotonic() < deadline:
            actions = state.get("available_actions")
            if (
                state.get("screen") == "MAP"
                and isinstance(actions, list)
                and "choose_map_node" in actions
            ):
                return state
            state = self._client.state()
            if time.monotonic() < deadline:
                time.sleep(0.2)
        raise BattleResetError("等待初始事件结算超时")

    def _await_potion_discarded(
        self,
        candidate: Mapping[str, Any],
        option_index: int,
    ) -> dict[str, Any]:
        """等待指定药水槽变为空槽。

        Args:
            candidate (Mapping[str, Any]): 丢弃响应携带的首份状态。
            option_index (int): 正在等待清空的药水槽索引。

        Raises:
            BattleResetError: 动作超时内目标药水槽仍被占用。

        Returns:
            dict[str, Any]: 目标药水槽已经清空的游戏状态。
        """
        deadline = time.monotonic() + self._client.action_timeout
        state = dict(candidate)
        while time.monotonic() < deadline:
            run = state.get("run")
            potions = run.get("potions") if isinstance(run, Mapping) else None
            if isinstance(potions, list) and not any(
                isinstance(potion, Mapping)
                and potion.get("index") == option_index
                and (
                    potion.get("occupied") is True
                    or potion.get("potion_id") is not None
                )
                for potion in potions
            ):
                return state
            state = self._client.state()
            if time.monotonic() < deadline:
                time.sleep(0.2)
        raise BattleResetError(f"等待药水槽清空超时: {option_index}")

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
    if scenario.potions:
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
