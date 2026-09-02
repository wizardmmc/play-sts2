"""验证战略 checkpoint 的文件、入口快照与稳定选项契约。"""

import importlib
from pathlib import Path
from typing import Any

import pytest

from play_sts2.client import Health
from play_sts2.harness import HarnessAction


class CaptureGame:
    """提供 checkpoint 捕获所需的真实客户端边界替身。"""

    def __init__(self, state: dict[str, Any], audit: dict[str, Any]) -> None:
        """保存动作前状态与隐藏审计。

        Args:
            state (dict[str, Any]): 原生保存前的战略决策状态。
            audit (dict[str, Any]): 与状态同一时刻的隐藏环境审计。

        Returns:
            None: 此方法只保存测试输入。
        """
        self._state = state
        self._audit = audit
        self.actions: list[tuple[str, dict[str, Any]]] = []

    def health(self) -> Health:
        """返回固定的 v0.111.0 游戏与 Mod 身份。

        Returns:
            Health: checkpoint 元数据使用的运行时身份。
        """
        return Health(
            service="sts2-ai-agent",
            mod_version="0.8.0-rlsts2.50",
            protocol_version="2026-08-30-v3",
            game_version="v0.111.0",
            status="ready",
        )

    def state(self) -> dict[str, Any]:
        """返回原生保存前的稳定战略状态。

        Returns:
            dict[str, Any]: 动作前状态副本。
        """
        return dict(self._state)

    def checkpoint_audit(self) -> dict[str, Any]:
        """返回不进入模型消息的隐藏环境审计。

        Returns:
            dict[str, Any]: RNG 与房间状态。
        """
        return dict(self._audit)

    def execute_action(self, action: str, **parameters: Any) -> dict[str, Any]:
        """记录原生存退并返回稳定主菜单。

        Args:
            action (str): 必须是 ``save_and_quit``。
            parameters (Any): 必须绑定动作前 revision。

        Returns:
            dict[str, Any]: 原生动作成功后的主菜单状态。
        """
        self.actions.append((action, dict(parameters)))
        return {
            "action": action,
            "stable": True,
            "state": {
                "screen": "MAIN_MENU",
                "available_actions": ["continue_run"],
            },
        }


class RestoreGame(CaptureGame):
    """模拟从主菜单原生 continue 后回到 checkpoint 入口。"""

    action_timeout = 30.0

    def state(self) -> dict[str, Any]:
        """返回带原生续局动作的主菜单。

        Returns:
            dict[str, Any]: 可恢复保存局的主菜单状态。
        """
        return {
            "state_revision": 30,
            "screen": "MAIN_MENU",
            "available_actions": ["continue_run"],
        }

    def execute_action(self, action: str, **parameters: Any) -> dict[str, Any]:
        """执行原生 continue 并返回捕获时的战略入口。

        Args:
            action (str): 必须是 ``continue_run``。
            parameters (Any): 必须绑定主菜单 revision。

        Returns:
            dict[str, Any]: 已恢复的稳定战略状态。
        """
        self.actions.append((action, dict(parameters)))
        return {
            "action": action,
            "stable": True,
            "state": dict(self._state),
        }


class PrefinishedEventRestoreGame(RestoreGame):
    """模拟原生存档先恢复已完成事件、再进入目标地图。"""

    def __init__(
        self,
        target_state: dict[str, Any],
        audit: dict[str, Any],
    ) -> None:
        """保存最终目标状态和已完成事件过渡页。

        Args:
            target_state (dict[str, Any]): 语义 Proceed 后的地图状态。
            audit (dict[str, Any]): 地图入口隐藏审计。

        Returns:
            None: 此方法构造原生恢复序列。
        """
        super().__init__(target_state, audit)
        self._continued = False

    def execute_action(self, action: str, **parameters: Any) -> dict[str, Any]:
        """先恢复已完成事件，再执行唯一继续选项进入地图。

        Args:
            action (str): ``continue_run`` 或 ``choose_event_option``。
            parameters (Any): 当前页面要求的 revision 与选项索引。

        Returns:
            dict[str, Any]: 对应动作后的稳定状态。
        """
        self.actions.append((action, dict(parameters)))
        if not self._continued:
            self._continued = True
            return {
                "action": action,
                "stable": True,
                "state": _finished_event_state(),
            }
        return {
            "action": action,
            "stable": True,
            "state": dict(self._state),
        }


class ClosedChestRestoreGame(RestoreGame):
    """模拟地图 checkpoint 原生恢复到尚未打开的宝箱。"""

    def __init__(
        self,
        target_state: dict[str, Any],
        audit: dict[str, Any],
    ) -> None:
        """保存目标地图和确定性的宝箱恢复阶段。

        Args:
            target_state (dict[str, Any]): 领取宝箱遗物后的地图状态。
            audit (dict[str, Any]): 地图入口隐藏审计。

        Returns:
            None: 此方法只初始化恢复阶段。
        """
        super().__init__(target_state, audit)
        self._stage = 0

    def execute_action(self, action: str, **parameters: Any) -> dict[str, Any]:
        """按原生顺序恢复宝箱、领取唯一遗物并进入地图。

        Args:
            action (str): 当前恢复阶段唯一允许的动作。
            parameters (Any): 当前状态 revision 与可选遗物索引。

        Returns:
            dict[str, Any]: 动作后的稳定宝箱或地图状态。
        """
        self.actions.append((action, dict(parameters)))
        states = (
            _closed_chest_state(),
            _opened_chest_state(),
            _claimed_chest_state(),
            dict(self._state),
        )
        state = states[self._stage]
        self._stage += 1
        return {"action": action, "stable": True, "state": state}


def test_capture_checkpoint_preserves_native_save_and_entry_audit(
    tmp_path: Path,
) -> None:
    """捕获必须保存原生文件、玩家观测和独立隐藏审计且拒绝覆盖。

    Args:
        tmp_path (Path): Pytest 提供的临时目录。

    Returns:
        None: checkpoint 可重新加载并保持入口语义。
    """
    checkpoint_module = importlib.import_module("play_sts2.checkpoint")
    state = _event_state()
    audit = {
        "screen": "EVENT",
        "run_id": "CHECKPOINT-SEED",
        "run_rng": {"Shuffle": {"counter": 3}},
    }
    game = CaptureGame(state, audit)
    home = tmp_path / "source-home"
    source_save = checkpoint_module.run_save_path(home)
    source_save.parent.mkdir(parents=True)
    source_save.write_bytes(b'{"native":"run-save"}')
    destination = tmp_path / "checkpoint"

    captured = checkpoint_module.capture_strategic_checkpoint(
        game,
        home=home,
        destination=destination,
    )
    loaded = checkpoint_module.load_strategic_checkpoint(destination)

    assert captured == loaded
    assert captured.save_path.read_bytes() == b'{"native":"run-save"}'
    assert captured.entry.screen == "EVENT"
    assert captured.entry.option_ids == ("event:0:NEOW_GIFT",)
    assert captured.entry.legal_actions == ("ACTION: choose_event_option 0",)
    assert captured.entry.audit == audit
    assert "=== 事件: 涅奥 ===" in captured.entry.policy_text
    assert game.actions == [("save_and_quit", {"expected_state_revision": 17})]
    with pytest.raises(FileExistsError):
        checkpoint_module.capture_strategic_checkpoint(
            game,
            home=home,
            destination=destination,
        )


def test_capture_checkpoint_rejects_card_selection_without_parent_anchor(
    tmp_path: Path,
) -> None:
    """CARD_SELECTION 不能直接保存，必须从可恢复父 REWARD 派生。

    Args:
        tmp_path (Path): Pytest 提供的临时目录。

    Returns:
        None: 直接捕获在修改游戏前被拒绝。
    """
    checkpoint_module = importlib.import_module("play_sts2.checkpoint")
    state = _card_selection_state()
    game = CaptureGame(state, {"screen": "CARD_SELECTION"})

    with pytest.raises(
        checkpoint_module.CheckpointError,
        match="CARD_SELECTION.*父 REWARD",
    ):
        checkpoint_module.capture_strategic_checkpoint(
            game,
            home=tmp_path / "source-home",
            destination=tmp_path / "checkpoint",
        )

    assert game.actions == []


def test_derive_checkpoint_reuses_parent_save_and_persists_resume_actions(
    tmp_path: Path,
) -> None:
    """原生不保存的子页面应复用父存档并显式记录恢复动作。

    Args:
        tmp_path (Path): Pytest 提供的临时目录。

    Returns:
        None: 卡牌奖励入口动作能随 checkpoint 往返持久化。
    """
    checkpoint_module = importlib.import_module("play_sts2.checkpoint")
    game = CaptureGame(
        _reward_state(),
        {"screen": "REWARD", "run_id": "CHECKPOINT-SEED"},
    )
    home = tmp_path / "source-home"
    save = checkpoint_module.run_save_path(home)
    save.parent.mkdir(parents=True)
    save.write_text('{"native":"save"}', encoding="utf-8")
    resume_actions = (HarnessAction("claim_reward", {"option_index": 2}),)

    base = checkpoint_module.capture_strategic_checkpoint(
        game,
        home=home,
        destination=tmp_path / "checkpoint-base",
    )
    target_game = CaptureGame(
        _card_selection_state(),
        {"screen": "CARD_SELECTION", "run_id": "CHECKPOINT-SEED"},
    )
    derived = checkpoint_module.derive_strategic_checkpoint(
        target_game,
        base=base,
        destination=tmp_path / "checkpoint-child",
        resume_actions=resume_actions,
    )
    loaded = checkpoint_module.load_strategic_checkpoint(derived.root)

    assert loaded.entry.resume_actions == resume_actions
    assert loaded.save_path.read_bytes() == base.save_path.read_bytes()
    assert game.actions == [("save_and_quit", {"expected_state_revision": 17})]
    assert target_game.actions == []


def test_restore_checkpoint_requires_exact_observation_options_and_audit(
    tmp_path: Path,
) -> None:
    """恢复入口必须同时匹配模型观测、合法选项与隐藏 RNG 审计。

    Args:
        tmp_path (Path): Pytest 提供的临时目录。

    Returns:
        None: 完整匹配时返回状态，任一隐藏 RNG 差异都会拒绝。
    """
    checkpoint_module = importlib.import_module("play_sts2.checkpoint")
    state = _event_state()
    audit = {
        "screen": "EVENT",
        "run_id": "CHECKPOINT-SEED",
        "run_rng": {"Shuffle": {"counter": 3}},
    }
    source_game = CaptureGame(state, audit)
    source_home = tmp_path / "source-home"
    source_save = checkpoint_module.run_save_path(source_home)
    source_save.parent.mkdir(parents=True)
    source_save.write_text('{"native":"save"}', encoding="utf-8")
    checkpoint = checkpoint_module.capture_strategic_checkpoint(
        source_game,
        home=source_home,
        destination=tmp_path / "checkpoint",
    )
    restored_game = RestoreGame(state, audit)

    restored = checkpoint_module.restore_strategic_checkpoint(
        restored_game,
        checkpoint,
    )

    assert restored == state
    assert restored_game.actions == [("continue_run", {"expected_state_revision": 30})]

    mismatched = RestoreGame(
        state,
        {
            **audit,
            "run_rng": {"Shuffle": {"counter": 4}},
        },
    )
    with pytest.raises(
        checkpoint_module.CheckpointMismatchError,
        match="audit",
    ):
        checkpoint_module.restore_strategic_checkpoint(mismatched, checkpoint)


def test_restore_checkpoint_advances_one_native_prefinished_event(
    tmp_path: Path,
) -> None:
    """原生 continue 恢复到唯一 Proceed 事件时应自动进入捕获的地图入口。

    Args:
        tmp_path (Path): Pytest 提供的临时目录。

    Returns:
        None: 自动动作只处理游戏明确标记的语义 Proceed。
    """
    checkpoint_module = importlib.import_module("play_sts2.checkpoint")
    state = _map_state()
    audit = {
        "screen": "MAP",
        "run_id": "CHECKPOINT-SEED",
        "run_rng": {"Shuffle": {"counter": 3}},
    }
    source_game = CaptureGame(state, audit)
    source_home = tmp_path / "source-home"
    save = checkpoint_module.run_save_path(source_home)
    save.parent.mkdir(parents=True)
    save.write_text('{"native":"save"}', encoding="utf-8")
    checkpoint = checkpoint_module.capture_strategic_checkpoint(
        source_game,
        home=source_home,
        destination=tmp_path / "checkpoint",
    )
    restored_game = PrefinishedEventRestoreGame(state, audit)

    restored = checkpoint_module.restore_strategic_checkpoint(
        restored_game,
        checkpoint,
    )

    assert restored["screen"] == "MAP"
    assert restored_game.actions == [
        ("continue_run", {"expected_state_revision": 30}),
        (
            "choose_event_option",
            {"expected_state_revision": 31, "option_index": 0},
        ),
    ]


def test_restore_map_checkpoint_replays_unique_native_chest(
    tmp_path: Path,
) -> None:
    """宝箱后的地图存档应重放唯一宝箱路径并通过完整入口核对。

    Args:
        tmp_path (Path): Pytest 提供的临时目录。

    Returns:
        None: 恢复器依次打开、领取唯一遗物并进入原地图。
    """
    checkpoint_module = importlib.import_module("play_sts2.checkpoint")
    state = _map_state()
    audit = {
        "screen": "MAP",
        "run_id": "CHECKPOINT-SEED",
        "run_rng": {"TreasureRoomRelics": {"counter": 1}},
    }
    source_game = CaptureGame(state, audit)
    source_home = tmp_path / "source-home"
    save = checkpoint_module.run_save_path(source_home)
    save.parent.mkdir(parents=True)
    save.write_text('{"native":"save"}', encoding="utf-8")
    checkpoint = checkpoint_module.capture_strategic_checkpoint(
        source_game,
        home=source_home,
        destination=tmp_path / "checkpoint",
    )
    restored_game = ClosedChestRestoreGame(state, audit)

    restored = checkpoint_module.restore_strategic_checkpoint(
        restored_game,
        checkpoint,
    )

    assert restored["screen"] == "MAP"
    assert restored_game.actions == [
        ("continue_run", {"expected_state_revision": 30}),
        ("open_chest", {"expected_state_revision": 31}),
        (
            "choose_treasure_relic",
            {"expected_state_revision": 32, "option_index": 0},
        ),
        ("proceed", {"expected_state_revision": 33}),
    ]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {
                "screen": "MAP",
                "map": {
                    "available_nodes": [
                        {"index": 0, "row": 2, "col": 3},
                        {"index": 1, "row": 2, "col": 5},
                    ]
                },
            },
            ("map:2:3", "map:2:5"),
        ),
        (
            {
                "screen": "REWARD",
                "reward": {
                    "rewards": [
                        {
                            "index": 0,
                            "reward_type": "Gold",
                            "name": "20金币",
                            "claimable": True,
                        }
                    ]
                },
            },
            ("reward:0:Gold:20金币",),
        ),
        (
            {
                "screen": "CARD_SELECTION",
                "selection": {
                    "cards": [
                        {"index": 0, "card_id": "ZAP"},
                        {"index": 1, "card_id": "DUALCAST"},
                    ]
                },
            },
            ("card:0:ZAP", "card:1:DUALCAST"),
        ),
        (
            {
                "screen": "SHOP",
                "shop": {
                    "cards": [{"index": 0, "card_id": "ZAP"}],
                    "relics": [{"index": 0, "relic_id": "ANCHOR"}],
                    "potions": [{"index": 0, "potion_id": "FIRE_POTION"}],
                    "card_removal": {"available": True},
                },
            },
            (
                "shop:card:0:ZAP",
                "shop:relic:0:ANCHOR",
                "shop:potion:0:FIRE_POTION",
                "shop:remove_card",
            ),
        ),
        (
            {
                "screen": "REST",
                "rest": {
                    "options": [
                        {"index": 0, "option_id": "HEAL"},
                        {"index": 1, "option_id": "SMITH"},
                    ]
                },
            },
            ("rest:0:HEAL", "rest:1:SMITH"),
        ),
        (
            {
                "screen": "EVENT",
                "event": {
                    "options": [
                        {"index": 0, "text_key": "EVENT.pages.INITIAL.options.A"},
                        {"index": 1, "text_key": "EVENT.pages.INITIAL.options.B"},
                    ]
                },
            },
            (
                "event:0:EVENT.pages.INITIAL.options.A",
                "event:1:EVENT.pages.INITIAL.options.B",
            ),
        ),
    ],
)
def test_checkpoint_option_ids_cover_all_strategic_branch_screens(
    state: dict[str, Any],
    expected: tuple[str, ...],
) -> None:
    """六类战略分支必须产生可读稳定选项身份而不是状态哈希。

    Args:
        state (dict[str, Any]): 当前页面的最小真实字段形状。
        expected (tuple[str, ...]): 手工确定的语义选项身份。

    Returns:
        None: 每种页面都保留索引与游戏稳定 ID。
    """
    checkpoint_module = importlib.import_module("play_sts2.checkpoint")

    assert checkpoint_module.checkpoint_option_ids(state) == expected


def _event_state() -> dict[str, Any]:
    """返回 checkpoint 捕获测试使用的完整事件决策状态。

    Returns:
        dict[str, Any]: 可由当前战略 Harness 与动作枚举器处理的状态。
    """
    return {
        "state_revision": 17,
        "screen": "EVENT",
        "in_combat": False,
        "available_actions": ["save_and_quit", "choose_event_option"],
        "run": {
            "act_id": "0",
            "boss_id": "KAISER_CRAB_BOSS",
            "ascension": 0,
            "floor": 1,
            "current_hp": 60,
            "max_hp": 75,
            "gold": 99,
            "potions": [],
            "relics": [],
            "deck": [],
        },
        "event": {
            "title": "涅奥",
            "options": [
                {
                    "index": 0,
                    "text_key": "NEOW_GIFT",
                    "title": "获得礼物",
                }
            ],
        },
    }


def _finished_event_state() -> dict[str, Any]:
    """返回原生保存恢复后常见的已完成事件 Proceed 页面。

    Returns:
        dict[str, Any]: 只有一个明确 ``is_proceed`` 选项的事件状态。
    """
    state = _event_state()
    state["state_revision"] = 31
    state["event"] = {
        "event_id": "NEOW",
        "title": "涅奥",
        "is_finished": True,
        "options": [
            {
                "index": 0,
                "text_key": "PROCEED",
                "title": "继续",
                "is_proceed": True,
                "is_locked": False,
            }
        ],
    }
    return state


def _closed_chest_state() -> dict[str, Any]:
    """返回原生 MAP checkpoint 恢复出的未打开宝箱。

    Returns:
        dict[str, Any]: 只有 ``open_chest`` 动作的稳定状态。
    """
    return {
        "state_revision": 31,
        "screen": "CHEST",
        "in_combat": False,
        "available_actions": ["save_and_quit", "open_chest"],
        "run": dict(_map_state()["run"]),
        "chest": {
            "is_opened": False,
            "has_relic_been_claimed": False,
            "relic_options": [],
        },
    }


def _opened_chest_state() -> dict[str, Any]:
    """返回已打开且只有一件遗物可领取的宝箱。

    Returns:
        dict[str, Any]: 只有 ``choose_treasure_relic`` 动作的稳定状态。
    """
    return {
        "state_revision": 32,
        "screen": "CHEST",
        "in_combat": False,
        "available_actions": ["save_and_quit", "choose_treasure_relic"],
        "run": dict(_map_state()["run"]),
        "chest": {
            "is_opened": True,
            "has_relic_been_claimed": False,
            "relic_options": [
                {
                    "index": 0,
                    "relic_id": "GORGET",
                    "name": "护喉甲",
                }
            ],
        },
    }


def _claimed_chest_state() -> dict[str, Any]:
    """返回领取遗物后只允许离开的宝箱。

    Returns:
        dict[str, Any]: 只有 ``proceed`` 动作的稳定状态。
    """
    return {
        "state_revision": 33,
        "screen": "CHEST",
        "in_combat": False,
        "available_actions": ["save_and_quit", "proceed"],
        "run": dict(_map_state()["run"]),
        "chest": {
            "is_opened": True,
            "has_relic_been_claimed": True,
            "relic_options": [],
        },
    }


def _reward_state() -> dict[str, Any]:
    """返回可作为卡牌选择父 checkpoint 的奖励页面。

    Returns:
        dict[str, Any]: 含唯一卡牌奖励入口的完整战略状态。
    """
    state = _event_state()
    state.update(
        {
            "screen": "REWARD",
            "event": None,
            "available_actions": ["save_and_quit", "claim_reward", "proceed"],
            "reward": {
                "rewards": [
                    {
                        "index": 2,
                        "reward_type": "Card",
                        "name": "将一张牌添加到你的牌组。",
                        "claimable": True,
                    }
                ]
            },
        }
    )
    return state


def _card_selection_state() -> dict[str, Any]:
    """返回从父奖励动作打开的卡牌选择页面。

    Returns:
        dict[str, Any]: 含一张候选牌的完整战略状态。
    """
    state = _event_state()
    state.update(
        {
            "screen": "CARD_SELECTION",
            "available_actions": ["save_and_quit", "choose_reward_card"],
            "event": None,
            "selection": {
                "kind": "deck_card_select",
                "cards": [
                    {
                        "index": 0,
                        "card_id": "ZAP",
                        "name": "电击",
                        "card_type": "Skill",
                        "energy_cost": 1,
                        "resolved_rules_text": "生成1个闪电充能球。",
                    }
                ],
            },
        }
    )
    return state


def _map_state() -> dict[str, Any]:
    """返回语义 Proceed 后的完整地图 checkpoint 状态。

    Returns:
        dict[str, Any]: 可由 Harness 和动作枚举器处理的地图决策。
    """
    state = _event_state()
    state.update(
        {
            "state_revision": 32,
            "screen": "MAP",
            "available_actions": ["save_and_quit", "choose_map_node"],
            "event": None,
            "map": {
                "current_node": {"row": 0, "col": 3},
                "available_nodes": [
                    {"index": 0, "row": 1, "col": 3, "node_type": "Monster"}
                ],
            },
        }
    )
    return state
