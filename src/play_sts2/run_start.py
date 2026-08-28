"""从干净主菜单选择角色并开始一局游戏。"""

import time
from collections.abc import Mapping
from typing import Any

from .client import GameClient, ProtocolError


class RunStartError(RuntimeError):
    """表示当前游戏状态无法完成可验证的开局流程。"""


def start_run(
    client: GameClient,
    character_id: str,
    *,
    seed: str | None = None,
    ascension: int | None = None,
) -> dict[str, Any]:
    """从干净主菜单选择指定角色并开始新局。

    Args:
        client (GameClient): 已连接到可操作主菜单的游戏客户端。
        character_id (str): Mod 暴露的角色稳定 ID，例如 ``DEFECT``。
        seed (str | None): 可选的游戏种子；提供时在开局前写入角色选择页。
        ascension (int | None): 可选的目标进阶等级；提供时在开局前逐级调整。

    Raises:
        RunStartError: 必需动作、角色信息或动作后的身份校验不成立。
        httpx.HTTPStatusError: Mod 拒绝流程中的动作请求。
        ProtocolError: Mod 返回不符合客户端协议的响应。

    Returns:
        dict[str, Any]: Mod 在 ``embark`` 完成后返回的新局状态。
    """
    state = client.state()
    if state.get("screen") != "MAIN_MENU":
        raise RunStartError("游戏不在主菜单")

    _require_action(state, "open_character_select")
    state = _action_state(
        client.execute_action(
            "open_character_select",
            expected_state_revision=_state_revision(state),
        ),
        "open_character_select",
    )

    character_select = state.get("character_select")
    if not isinstance(character_select, Mapping):
        raise RunStartError("角色选择状态不可用")
    characters = character_select.get("characters")
    if not isinstance(characters, list):
        raise RunStartError("角色列表不可用")

    matches = [
        character
        for character in characters
        if isinstance(character, Mapping)
        and character.get("character_id") == character_id
    ]
    if len(matches) != 1:
        raise RunStartError(f"无法唯一找到角色: {character_id}")

    selected = matches[0]
    if selected.get("is_locked") is True:
        raise RunStartError(f"角色已锁定: {character_id}")

    option_index = selected.get("index")
    if isinstance(option_index, bool) or not isinstance(option_index, int):
        raise RunStartError(f"角色索引无效: {character_id}")

    _require_action(state, "select_character")
    state = _action_state(
        client.execute_action(
            "select_character",
            expected_state_revision=_state_revision(state),
            option_index=option_index,
        ),
        "select_character",
    )
    character_select = state.get("character_select")
    if (
        not isinstance(character_select, Mapping)
        or character_select.get("selected_character_id") != character_id
    ):
        raise RunStartError(f"角色选择结果不匹配: {character_id}")

    if ascension is not None:
        state = _adjust_ascension(client, state, ascension)
    if seed is not None:
        _require_action(state, "set_seed")
        state = _action_state(
            client.execute_action(
                "set_seed",
                expected_state_revision=_state_revision(state),
                game_seed=seed,
            ),
            "set_seed",
        )

    _require_action(state, "embark")
    revision = _state_revision(state)
    deadline = time.monotonic() + client.action_timeout
    state = _await_run_state(
        client,
        client.execute_action("embark", expected_state_revision=revision),
        deadline,
        "embark",
        after_revision=revision,
    )
    run = state.get("run")
    if not isinstance(run, Mapping) or run.get("character_id") != character_id:
        raise RunStartError(f"新局角色身份不匹配: {character_id}")
    if ascension is not None and run.get("ascension") != ascension:
        raise RunStartError(f"新局进阶等级不匹配: {ascension}")
    return state


def resume_run(client: GameClient) -> dict[str, Any]:
    """续玩当前进行中的一局，或从主菜单恢复保存局。

    Args:
        client (GameClient): 已连接到当前游戏实例的客户端。

    Raises:
        RunStartError: 当前既不在局中，也没有可执行的续局动作。
        httpx.HTTPStatusError: Mod 拒绝续局或状态读取请求。
        ProtocolError: Mod 返回不符合客户端协议的响应。

    Returns:
        dict[str, Any]: 已经进入当前局的稳定游戏状态。
    """
    state = client.state()
    if isinstance(state.get("run"), Mapping):
        return state
    if state.get("screen") != "MAIN_MENU":
        raise RunStartError("游戏既不在局中，也不在主菜单")

    _require_action(state, "continue_run")
    revision = _state_revision(state)
    deadline = time.monotonic() + client.action_timeout
    state = _await_run_state(
        client,
        client.execute_action("continue_run", expected_state_revision=revision),
        deadline,
        "continue_run",
        after_revision=revision,
    )
    if not isinstance(state.get("run"), Mapping):
        raise RunStartError("续局后没有取得运行状态")
    return state


def _adjust_ascension(
    client: GameClient,
    state: Mapping[str, Any],
    target: int,
) -> dict[str, Any]:
    """把角色选择页的进阶等级逐级调整到目标值。

    Args:
        client (GameClient): 已连接到角色选择页的游戏客户端。
        state (Mapping[str, Any]): 当前角色选择状态。
        target (int): 期望在新局中使用的进阶等级。

    Raises:
        RunStartError: 进阶字段缺失、目标越界或所需调整动作不可用。

    Returns:
        dict[str, Any]: 已到达目标进阶等级的角色选择状态。
    """
    current_state = dict(state)
    character_select = current_state.get("character_select")
    if not isinstance(character_select, Mapping):
        raise RunStartError("角色选择状态不可用")
    current = character_select.get("ascension")
    maximum = character_select.get("max_ascension")
    if (
        isinstance(current, bool)
        or not isinstance(current, int)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
    ):
        raise RunStartError("进阶状态不可用")
    if target < 0 or target > maximum:
        raise RunStartError(f"目标进阶等级不可用: {target}")

    while current != target:
        action = "increase_ascension" if current < target else "decrease_ascension"
        _require_action(current_state, action)
        current_state = _action_state(
            client.execute_action(
                action,
                expected_state_revision=_state_revision(current_state),
            ),
            action,
        )
        character_select = current_state.get("character_select")
        if not isinstance(character_select, Mapping):
            raise RunStartError("角色选择状态不可用")
        next_level = character_select.get("ascension")
        if isinstance(next_level, bool) or not isinstance(next_level, int):
            raise RunStartError("进阶状态不可用")
        if abs(next_level - current) != 1:
            raise RunStartError("进阶调整结果无效")
        current = next_level
    return current_state


def _require_action(state: Mapping[str, Any], action: str) -> None:
    """确认当前状态明确允许执行指定动作。

    Args:
        state (Mapping[str, Any]): 当前游戏状态。
        action (str): 流程下一步要求的动作名称。

    Raises:
        RunStartError: 动作列表缺失或未包含指定动作。

    Returns:
        None: 动作存在时返回。
    """
    actions = state.get("available_actions")
    if not isinstance(actions, list) or action not in actions:
        raise RunStartError(f"当前状态不允许动作: {action}")


def _action_state(result: Mapping[str, Any], action: str) -> dict[str, Any]:
    """从已完成动作的结果中提取稳定游戏状态。

    Args:
        result (Mapping[str, Any]): Mod 返回的动作结果对象。
        action (str): 产生该结果的动作名称。

    Raises:
        RunStartError: 动作尚未稳定完成，或结果没有状态对象。

    Returns:
        dict[str, Any]: 动作完成后的稳定游戏状态。
    """
    state = result.get("state")
    if result.get("stable") is not True or not isinstance(state, Mapping):
        raise RunStartError(f"动作未返回稳定状态: {action}")
    return dict(state)


def _await_run_state(
    client: GameClient,
    result: Mapping[str, Any],
    deadline: float,
    action: str,
    *,
    after_revision: int,
) -> dict[str, Any]:
    """等待已排队的开局或续局动作产生运行状态。

    Args:
        client (GameClient): 已提交开局动作的游戏客户端。
        result (Mapping[str, Any]): 开局或续局动作的即时结果。
        deadline (float): 动作 HTTP 请求与状态等待共享的单调时钟截止点。
        action (str): 当前等待的动作名称。
        after_revision (int): 提交动作前已经处理的状态版本。

    Raises:
        RunStartError: 动作既未完成也未排队，或新局状态等待超时。
        httpx.HTTPStatusError: 状态事件连接被 Mod 拒绝。
        ProtocolError: 状态事件违反客户端协议。

    Returns:
        dict[str, Any]: 已包含 ``run`` 对象的游戏状态。
    """
    if result.get("stable") is True:
        return _action_state(result, action)
    if result.get("status") != "pending":
        raise RunStartError(f"动作未返回稳定状态: {action}")

    raw_state = result.get("state")
    state = dict(raw_state) if isinstance(raw_state, Mapping) else None
    revision = _state_revision(state) if state is not None else after_revision
    while True:
        if state is not None and isinstance(state.get("run"), Mapping):
            return state
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RunStartError(f"等待运行状态超时: {action}")
        try:
            state = client.wait_for_state(
                after_revision=revision,
                timeout=remaining,
            )
        except TimeoutError as exc:
            raise RunStartError(f"等待运行状态超时: {action}") from exc
        revision = _state_revision(state)


def _state_revision(state: Mapping[str, Any]) -> int:
    """读取开局动作并发保护所需的状态 revision。"""
    revision = state.get("state_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ProtocolError("game state is missing a valid state_revision")
    return revision
