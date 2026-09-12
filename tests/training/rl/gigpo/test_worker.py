"""验证 GiGPO backbone 的原生 checkpoint 物化边界。"""

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class AuditGame:
    """只提供 backbone 状态钩子所需的隐藏审计。"""

    def checkpoint_audit(self) -> dict[str, Any]:
        """返回战斗奖励后地图对应的 Monster 房间审计。

        Returns:
            dict[str, Any]: 含当前房间类型的最小审计。
        """
        return {
            "screen": "MAP",
            "run_id": "WORKER-SEED",
            "run": {"current_room": {"room_type": "Monster"}},
        }


def test_backbone_does_not_materialize_post_combat_map_checkpoint(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """战斗奖励后的地图候选不能保存为不可精确恢复的原生 checkpoint。

    Args:
        monkeypatch (Any): Pytest 提供的属性替换工具。
        tmp_path (Path): Pytest 提供的临时目录。

    Returns:
        None: 候选仍被记录，但路径为空且不会进入后续 Tree 物化池。
    """
    worker = importlib.import_module("play_sts2.training.rl.gigpo.worker")

    def capture_checkpoint(*_args: Any, destination: Path, **_kwargs: Any) -> Any:
        """返回可观察的伪 checkpoint 路径。

        Args:
            _args (Any): 被忽略的位置参数。
            destination (Path): 生产代码请求写入的位置。
            _kwargs (Any): 被忽略的其余参数。

        Returns:
            Any: 仅含 ``root`` 的捕获结果。
        """
        return SimpleNamespace(root=destination)

    monkeypatch.setattr(worker, "capture_strategic_checkpoint", capture_checkpoint)
    monkeypatch.setattr(
        worker,
        "restore_strategic_checkpoint",
        lambda _game, _checkpoint: _map_state(),
    )
    capture = worker._BackboneStateCapture(
        game=AuditGame(),
        home=tmp_path / "home",
        checkpoint_root=tmp_path / "checkpoints",
        seed="WORKER-SEED",
        early_target_ordinal=0,
        late_target_ordinal=0,
    )

    capture(_map_state())

    assert len(capture.checkpoints) == 1
    assert capture.checkpoints[0].kind == "map"
    assert capture.checkpoints[0].path is None


def _map_state() -> dict[str, Any]:
    """返回战斗奖励后可供模型选择路线的地图状态。

    Returns:
        dict[str, Any]: 含两个后继节点的完整最小战略状态。
    """
    return {
        "state_revision": 17,
        "screen": "MAP",
        "in_combat": False,
        "available_actions": ["save_and_quit", "choose_map_node"],
        "run": {
            "act_id": "0",
            "boss_id": "KAISER_CRAB_BOSS",
            "ascension": 0,
            "floor": 3,
            "current_hp": 60,
            "max_hp": 75,
            "gold": 114,
            "potions": [],
            "relics": [],
            "deck": [],
        },
        "map": {
            "current_node": {"row": 2, "col": 3},
            "available_nodes": [
                {"index": 0, "row": 3, "col": 2, "node_type": "Monster"},
                {"index": 1, "row": 3, "col": 4, "node_type": "Event"},
            ],
        },
    }


def test_late_checkpoint_budget_can_be_reserved_for_act_two(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """第一幕中段不消耗晚期预算，到第二幕后只物化一次。"""
    worker = importlib.import_module("play_sts2.training.rl.gigpo.worker")
    game = SimpleNamespace(
        checkpoint_audit=lambda: {"run": {"current_room": {"room_type": "Event"}}}
    )
    floors = []

    def capture_checkpoint(game: Any, *, destination: Path, **kwargs: Any) -> Any:
        """记录真实触发物化的楼层，代替外部游戏存退。"""
        floors.append(game.current_state["run"]["floor"])
        return SimpleNamespace(root=destination, state=game.current_state)

    monkeypatch.setattr(worker, "capture_strategic_checkpoint", capture_checkpoint)
    monkeypatch.setattr(
        worker,
        "restore_strategic_checkpoint",
        lambda game, checkpoint: checkpoint.state,
    )
    capture = worker._BackboneStateCapture(
        game=game,
        home=tmp_path / "home",
        checkpoint_root=tmp_path / "checkpoints",
        seed="WORKER-SEED",
        early_target_ordinal=0,
        late_target_ordinal=0,
        late_checkpoint_min_floor=18,
    )
    for floor in (3, 11, 18, 19):
        state = _map_state()
        state["run"].update(floor=floor, act_id="1" if floor >= 18 else "0")
        state["map"]["current_node"]["row"] = (floor - 1) % 17
        for node in state["map"]["available_nodes"]:
            node["row"] = floor % 17
        game.current_state = state
        capture(state)
    assert floors == [3, 18]
    assert [item.path is not None for item in capture.checkpoints] == [
        True,
        False,
        True,
        False,
    ]
