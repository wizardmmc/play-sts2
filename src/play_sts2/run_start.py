"""从干净主菜单选择角色并开始一局游戏。"""

import time
from collections.abc import Mapping
from typing import Any

from .client import GameClient


class RunStartError(RuntimeError):
    """表示当前游戏状态无法完成可验证的开局流程。"""


def start_run(client: GameClient, character_id: str) -> dict[str, Any]:
    """从干净主菜单选择指定角色并开始新局。

    Args:
        client (GameClient): 已连接到可操作主菜单的游戏客户端。
        character_id (str): Mod 暴露的角色稳定 ID，例如 ``DEFECT``。

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
        client.execute_action("open_character_select"),
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
        client.execute_action("select_character", option_index=option_index),
        "select_character",
    )
    character_select = state.get("character_select")
    if (
        not isinstance(character_select, Mapping)
        or character_select.get("selected_character_id") != character_id
    ):
        raise RunStartError(f"角色选择结果不匹配: {character_id}")

    _require_action(state, "embark")
    deadline = time.monotonic() + client.action_timeout
    state = _await_embark_state(
        client,
        client.execute_action("embark"),
        deadline,
    )
    run = state.get("run")
    if not isinstance(run, Mapping) or run.get("character_id") != character_id:
        raise RunStartError(f"新局角色身份不匹配: {character_id}")
    return state


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


def _await_embark_state(
    client: GameClient,
    result: Mapping[str, Any],
    deadline: float,
) -> dict[str, Any]:
    """等待已排队的开局动作产生可验证的新局状态。

    Args:
        client (GameClient): 已提交开局动作的游戏客户端。
        result (Mapping[str, Any]): ``embark`` 动作的即时结果。
        deadline (float): 动作 HTTP 请求与状态等待共享的单调时钟截止点。

    Raises:
        RunStartError: 动作既未完成也未排队，或新局状态等待超时。
        httpx.HTTPStatusError: 轮询状态时 Mod 返回非成功 HTTP 状态码。
        ProtocolError: 轮询状态时 Mod 返回不符合客户端协议的响应。

    Returns:
        dict[str, Any]: 已包含 ``run`` 对象的新局状态。
    """
    if result.get("stable") is True:
        return _action_state(result, "embark")
    if result.get("status") != "pending":
        raise RunStartError("动作未返回稳定状态: embark")

    while time.monotonic() < deadline:
        state = client.state()
        if isinstance(state.get("run"), Mapping):
            return state
        time.sleep(0.2)
    raise RunStartError("等待新局状态超时: embark")
