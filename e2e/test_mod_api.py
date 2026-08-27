"""验证真实 STS2、Mod HTTP API 与 Python 客户端的完整动作链路。"""

import pytest

from play_sts2 import start_run
from play_sts2.client import GameClient

from .conftest import RunningGame

pytestmark = pytest.mark.e2e


def test_game_mod_action_contract(running_game: RunningGame) -> None:
    """使用专用测试档在真实游戏中开启故障机器人新局。

    Args:
        running_game (RunningGame): 测试进程启动的隔离游戏实例。

    Raises:
        AssertionError: 真实 Mod 响应、开局状态或存档隔离不满足预期。

    Returns:
        None: 此测试仅验证完整游戏动作链路。
    """
    with GameClient(running_game.base_url) as client:
        health = client.health()
        state = client.state()
        actions = client.available_actions()
        run_state = start_run(client, "DEFECT")

    assert health.service == "sts2-ai-agent"
    assert health.status == "ready"
    assert isinstance(state.get("state_version"), int)
    assert isinstance(state.get("screen"), str)
    assert isinstance(state.get("available_actions"), list)
    assert actions.screen == state["screen"]
    assert [action.name for action in actions.actions] == state["available_actions"]
    assert run_state["run"]["character_id"] == "DEFECT"

    data_root = running_game.home / "Library/Application Support/SlayTheSpire2"
    expected_run_save = data_root / "default/1/modded/profile1/saves/current_run.save"
    assert set(data_root.rglob("current_run.save")) == {expected_run_save}
    assert not (data_root / "steam").exists()
