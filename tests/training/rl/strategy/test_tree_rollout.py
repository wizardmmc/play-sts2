"""验证宏 suffix runner 只训练战略 token 并冻结战斗 policy。"""

from collections import deque
from typing import Any

import httpx


class Strategy:
    """按 Event、Map 顺序返回两个战略决策。"""

    def __init__(self, steps: tuple[Any, ...]) -> None:
        """保存待返回决策。

        Args:
            steps (tuple[Any, ...]): 两个真实 DecisionStep 形状。

        Returns:
            None: 此方法只初始化队列。
        """
        self._steps = deque(steps)

    def step(self, _state: dict[str, Any]) -> Any:
        """返回下一条战略决策。

        Args:
            _state (dict[str, Any]): 当前状态，由预制步骤覆盖。

        Returns:
            Any: 下一条 DecisionStep。
        """
        return self._steps.popleft()


class ConflictThenStrategy(Strategy):
    """第一次遇到真实动作窗口冲突，随后返回合法步骤。"""

    def __init__(self, steps: tuple[Any, ...]) -> None:
        """保存合法步骤并初始化一次冲突。

        Args:
            steps (tuple[Any, ...]): 冲突恢复后的决策步骤。

        Returns:
            None: 此方法初始化调用计数。
        """
        super().__init__(steps)
        self.calls = 0

    def step(self, state: dict[str, Any]) -> Any:
        """首次抛动作窗口冲突，第二次执行真实策略。

        Args:
            state (dict[str, Any]): 当前稳定战略状态。

        Raises:
            httpx.HTTPStatusError: 首次调用模拟 Mod 的瞬时 409。

        Returns:
            Any: 后续合法 DecisionStep。
        """
        self.calls += 1
        if self.calls == 1:
            request = httpx.Request("POST", "http://game/action")
            response = httpx.Response(
                409,
                request=request,
                json={
                    "error": {
                        "code": "invalid_action",
                        "message": "Action is not available in the current state.",
                    }
                },
            )
            raise httpx.HTTPStatusError(
                "action window conflict",
                request=request,
                response=response,
            )
        return super().step(state)


class Battle:
    """返回一场由冻结 E6 policy 完成的战斗。"""

    def __init__(self, result: Any) -> None:
        """保存战斗结果。

        Args:
            result (Any): 真实 BattleResult 形状。

        Returns:
            None: 此方法只保存结果。
        """
        self._result = result

    def run(self, _state: dict[str, Any]) -> Any:
        """返回预制战斗结果。

        Args:
            _state (dict[str, Any]): 战斗入口状态。

        Returns:
            Any: BattleResult。
        """
        return self._result


class Game:
    """自动动作路径不应在本测试中调用游戏客户端。"""

    action_timeout = 30.0

    def execute_action(self, _action: str, **_parameters: Any) -> dict[str, Any]:
        """在 runner 错误执行自动动作时失败。

        Args:
            _action (str): 不应执行的动作。
            _parameters (Any): 不应使用的参数。

        Raises:
            AssertionError: 此方法被调用时始终失败。

        Returns:
            dict[str, Any]: 此方法不会返回。
        """
        raise AssertionError("本测试不应自动执行 UI 动作")


class TransientGame(Game):
    """依次返回过渡帧和稳定地图状态。"""

    def __init__(self, states: tuple[dict[str, Any], ...]) -> None:
        """保存等待端点依次返回的状态。

        Args:
            states (tuple[dict[str, Any], ...]): 过渡帧和最终稳定状态。

        Returns:
            None: 此方法只初始化队列。
        """
        self._states = deque(states)

    def wait_for_state(
        self,
        *,
        after_revision: int,
        timeout: float,
    ) -> dict[str, Any]:
        """返回下一 revision，并核对 runner 持续推进游标。

        Args:
            after_revision (int): 已处理的状态 revision。
            timeout (float): 剩余等待预算。

        Returns:
            dict[str, Any]: 下一份过渡或稳定状态。
        """
        assert timeout > 0
        state = self._states.popleft()
        assert state["state_revision"] > after_revision
        return state


class AutomaticProceedGame(Game):
    """记录 Tree 投影后自动执行的唯一离场动作。"""

    def __init__(self, final_state: dict[str, Any]) -> None:
        """保存自动 Proceed 的目标状态。

        Args:
            final_state (dict[str, Any]): 离开当前宏后的地图状态。

        Returns:
            None: 此方法只保存结果与调用记录。
        """
        self._final_state = final_state
        self.actions: list[str] = []

    def execute_action(self, action: str, **_parameters: Any) -> dict[str, Any]:
        """只允许 projected state 唯一留下的 Proceed。

        Args:
            action (str): 自动动作名称。
            **_parameters (Any): revision 等动作参数。

        Returns:
            dict[str, Any]: 稳定地图状态。
        """
        assert action == "proceed"
        self.actions.append(action)
        return {"stable": True, "state": self._final_state}


def test_suffix_runner_reaches_second_macro_checkpoint_through_battle() -> None:
    """Event→Map→战斗→Reward 应形成两步战略 arm 和短 horizon return。

    Returns:
        None: 战斗 token 不进入战略 steps，但冻结 battle policy 被核对。
    """
    from play_sts2.runtime import BattleOutcome, BattleResult
    from play_sts2.training.rl.strategy.rollout import (
        TreeSuffixRunner,
        build_tree_rollout_arm,
    )

    event = _event_state()
    map_state = _map_state()
    combat = _combat_state()
    reward = _reward_state()
    event_step = _decision_step(
        event,
        map_state,
        "choose_event_option",
        0,
        "ACTION: choose_event_option 0",
        101,
    )
    map_step = _decision_step(
        map_state,
        combat,
        "choose_map_node",
        0,
        "ACTION: choose_map_node 0",
        102,
    )
    battle_step = _decision_step(
        combat,
        reward,
        "end_turn",
        None,
        "ACTION: end_turn",
        103,
    )
    runner = TreeSuffixRunner(
        game=Game(),
        strategy=Strategy((event_step, map_step)),
        battle=Battle(
            BattleResult(
                outcome=BattleOutcome.CLEARED,
                steps=(battle_step,),
                final_state=reward,
            )
        ),
        strategy_policy_version="e6-feasibility",
        battle_policy_version="e6-feasibility",
        max_macro_checkpoints=2,
    )

    result = runner.run(event)

    assert result.plan_id == "ACTION: choose_event_option 0"
    assert [step.action for step in result.steps] == [
        "ACTION: choose_event_option 0",
        "ACTION: choose_map_node 0",
    ]
    assert result.final_state == reward
    assert result.horizon_reason == "next_macro_checkpoint"
    assert result.continuation_return.total > 0.01
    assert result.strategy_policy_version == "e6-feasibility"
    assert result.battle_policy_version == "e6-feasibility"
    arm = build_tree_rollout_arm(result, arm_index=3, worker_id="worker-1")
    assert arm.arm_index == 3
    assert arm.worker_id == "worker-1"
    assert arm.steps == result.steps
    assert arm.plan_id == result.plan_id


def test_suffix_runner_waits_through_multiple_transient_revisions() -> None:
    """战略动作后的连续过渡 revision 应等待到稳定宏节点。

    Returns:
        None: 真实 Mod 的异步事件动画不应被误判成 rollout 失败。
    """
    from play_sts2.training.rl.strategy.rollout import TreeSuffixRunner

    event = _event_state()
    transient_one = {
        "state_revision": 2,
        "screen": "UNKNOWN",
        "in_combat": False,
        "available_actions": [],
        "run": _run(1),
    }
    transient_two = {**transient_one, "state_revision": 3}
    map_state = {**_map_state(), "state_revision": 4}
    event_step = _decision_step(
        event,
        transient_one,
        "choose_event_option",
        0,
        "ACTION: choose_event_option 0",
        101,
    )
    event_step = type(event_step)(
        observation=event_step.observation,
        before_state=event_step.before_state,
        messages=event_step.messages,
        replies=event_step.replies,
        retry_errors=event_step.retry_errors,
        action=event_step.action,
        action_result={"stable": False, "state": transient_one},
        response_choices=event_step.response_choices,
        generation_profile=event_step.generation_profile,
    )
    runner = TreeSuffixRunner(
        game=TransientGame((transient_two, map_state)),
        strategy=Strategy((event_step,)),
        battle=Battle(None),
        strategy_policy_version="e6-feasibility",
        battle_policy_version="e6-feasibility",
        max_macro_checkpoints=1,
    )

    result = runner.run(event)

    assert result.final_state == map_state
    assert result.horizon_reason == "next_macro_checkpoint"


def test_suffix_runner_requeries_after_action_window_conflict() -> None:
    """未执行的瞬时冲突回复不进入 arm，等待新 revision 后重新查询。

    Returns:
        None: 最终只保留成功执行的一条战略步骤。
    """
    from play_sts2.training.rl.strategy.rollout import TreeSuffixRunner

    event = _event_state()
    refreshed = {**event, "state_revision": 2}
    map_state = {**_map_state(), "state_revision": 3}
    event_step = _decision_step(
        refreshed,
        map_state,
        "choose_event_option",
        0,
        "ACTION: choose_event_option 0",
        101,
    )
    strategy = ConflictThenStrategy((event_step,))
    runner = TreeSuffixRunner(
        game=TransientGame((refreshed,)),
        strategy=strategy,
        battle=Battle(None),
        strategy_policy_version="e6-feasibility",
        battle_policy_version="e6-feasibility",
        max_macro_checkpoints=1,
    )

    result = runner.run(event)

    assert strategy.calls == 2
    assert len(result.steps) == 1
    assert result.steps[0].action == "ACTION: choose_event_option 0"


def test_suffix_runner_folds_event_card_selection_into_one_macro_plan() -> None:
    """事件奖励打开的选牌页应属于同一宏计划而非后继 checkpoint。

    Returns:
        None: suffix 仍跨过地图和战斗到达统一奖励 horizon。
    """
    from play_sts2.runtime import BattleOutcome, BattleResult
    from play_sts2.training.rl.strategy.rollout import TreeSuffixRunner

    event = _event_state()
    selection = _selection_state()
    map_state = _map_state()
    combat = _combat_state()
    reward = _reward_state()
    event_step = _decision_step(
        event,
        selection,
        "choose_event_option",
        1,
        "ACTION: choose_event_option 1",
        101,
    )
    selection_step = _decision_step(
        selection,
        map_state,
        "select_deck_card",
        0,
        "ACTION: select_deck_card 0",
        102,
    )
    map_step = _decision_step(
        map_state,
        combat,
        "choose_map_node",
        0,
        "ACTION: choose_map_node 0",
        103,
    )
    battle_step = _decision_step(
        combat,
        reward,
        "end_turn",
        None,
        "ACTION: end_turn",
        104,
    )
    runner = TreeSuffixRunner(
        game=Game(),
        strategy=Strategy((event_step, selection_step, map_step)),
        battle=Battle(
            BattleResult(
                outcome=BattleOutcome.CLEARED,
                steps=(battle_step,),
                final_state=reward,
            )
        ),
        strategy_policy_version="e6-feasibility",
        battle_policy_version="e6-feasibility",
        max_macro_checkpoints=2,
    )

    result = runner.run(event)

    assert result.plan_id == (
        "ACTION: choose_event_option 1\nACTION: select_deck_card 0"
    )
    assert result.plan_step_count == 2
    assert result.final_state == reward


def test_suffix_runner_counts_boss_from_selected_map_node_before_act_changes() -> None:
    """Boss 奖励仍在旧幕时也应按战斗入口的可见地图语义计里程碑。

    Returns:
        None: CLEARED Boss 在 REWARD proceed 之前已计入一次。
    """
    from play_sts2.runtime import BattleOutcome, BattleResult
    from play_sts2.training.rl.strategy.rollout import TreeSuffixRunner

    boss_map = _boss_map_state()
    combat = {**_combat_state(), "run": _run(18)}
    reward = {**_reward_state(), "run": _run(18, hp=70)}
    map_step = _decision_step(
        boss_map,
        combat,
        "choose_map_node",
        0,
        "ACTION: choose_map_node 0",
        101,
    )
    battle_step = _decision_step(
        combat,
        reward,
        "end_turn",
        None,
        "ACTION: end_turn",
        102,
    )
    runner = TreeSuffixRunner(
        game=Game(),
        strategy=Strategy((map_step,)),
        battle=Battle(
            BattleResult(
                outcome=BattleOutcome.CLEARED,
                steps=(battle_step,),
                final_state=reward,
            )
        ),
        strategy_policy_version="e6-feasibility",
        battle_policy_version="e6-feasibility",
        max_macro_checkpoints=1,
    )

    result = runner.run(boss_map)

    components = {
        item.name: item.value for item in result.continuation_return.components
    }
    assert components["boss_milestones"] == 1.0
    assert result.final_state["run"]["act_id"] == "0"


def test_suffix_runner_does_not_reopen_closed_shop_inventory() -> None:
    """Tree 宏应复用整局 runner 的商店关闭记忆并自动离场。

    Returns:
        None: raw state 虽仍暴露 reopen，policy 不会再次看到它。
    """
    from play_sts2.training.rl.strategy.rollout import TreeSuffixRunner

    opened = _shop_state(inventory_open=True)
    closed = _shop_state(inventory_open=False)
    map_state = {**_map_state(), "run": _run(5)}
    close_step = _decision_step(
        opened,
        closed,
        "close_shop_inventory",
        None,
        "ACTION: close_shop_inventory",
        101,
    )
    game = AutomaticProceedGame(map_state)
    runner = TreeSuffixRunner(
        game=game,
        strategy=Strategy((close_step,)),
        battle=Battle(None),
        strategy_policy_version="e6-feasibility",
        battle_policy_version="e6-feasibility",
        max_macro_checkpoints=1,
    )

    result = runner.run(opened)

    assert result.plan_id == "ACTION: close_shop_inventory"
    assert game.actions == ["proceed"]
    assert result.final_state == map_state


def _decision_step(
    before: dict[str, Any],
    after: dict[str, Any],
    action_name: str,
    option_index: int | None,
    action_line: str,
    token: int,
) -> Any:
    """构造含真实回复元数据的 DecisionStep。

    Args:
        before (dict[str, Any]): 动作前状态。
        after (dict[str, Any]): 动作后状态。
        action_name (str): Harness 动作名。
        option_index (int | None): 可选索引。
        action_line (str): 规范动作行。
        token (int): 可区分 completion token。

    Returns:
        Any: Runtime DecisionStep。
    """
    from play_sts2.harness import HarnessAction, build_observation, legal_action_lines
    from play_sts2.inference import ChatMessage, ModelReply
    from play_sts2.runtime import DecisionGenerationProfile, DecisionStep

    observation = build_observation(before)
    parameters = {} if option_index is None else {"option_index": option_index}
    return DecisionStep(
        observation=observation,
        before_state=before,
        messages=(
            ChatMessage("system", "系统"),
            ChatMessage("user", observation.text),
        ),
        replies=(
            ModelReply(
                text=action_line,
                finish_reason="stop",
                model="e6-feasibility",
                token_ids=(token,),
                behavior_logprobs=(-0.2,),
            ),
        ),
        retry_errors=(),
        action=HarnessAction(action_name, parameters),
        action_result={"stable": True, "state": after},
        response_choices=legal_action_lines(before),
        generation_profile=DecisionGenerationProfile(128, 0.8, 0, False),
    )


def _run(floor: int, hp: int = 75) -> dict[str, Any]:
    """返回战略与战斗状态共享的最小整局字段。

    Args:
        floor (int): 当前总楼层。
        hp (int): 当前生命值。

    Returns:
        dict[str, Any]: 玩家可见 run 状态。
    """
    return {
        "act_id": "0",
        "ascension": 0,
        "floor": floor,
        "current_hp": hp,
        "max_hp": 75,
        "gold": 99,
        "potions": [],
        "relics": [],
        "deck": [],
    }


def _event_state() -> dict[str, Any]:
    """返回包含两个真实选项的事件入口。

    Returns:
        dict[str, Any]: EVENT 宏 checkpoint。
    """
    return {
        "state_revision": 1,
        "screen": "EVENT",
        "in_combat": False,
        "available_actions": ["choose_event_option"],
        "run": _run(1),
        "event": {
            "title": "涅奥",
            "options": [
                {"index": 0, "text_key": "A", "title": "A"},
                {"index": 1, "text_key": "B", "title": "B"},
            ],
        },
    }


def _map_state() -> dict[str, Any]:
    """返回两个目的地的地图宏 checkpoint。

    Returns:
        dict[str, Any]: MAP 状态。
    """
    return {
        "state_revision": 2,
        "screen": "MAP",
        "in_combat": False,
        "available_actions": ["choose_map_node"],
        "run": _run(1),
        "map": {
            "current_node": {"row": 0, "col": 3},
            "available_nodes": [
                {"index": 0, "row": 1, "col": 2, "node_type": "Monster"},
                {"index": 1, "row": 1, "col": 4, "node_type": "Monster"},
            ],
        },
    }


def _selection_state() -> dict[str, Any]:
    """返回事件内部的卡牌选择子页面。

    Returns:
        dict[str, Any]: 与事件处于同一 floor 的 CARD_SELECTION 状态。
    """
    return {
        "state_revision": 2,
        "screen": "CARD_SELECTION",
        "in_combat": False,
        "available_actions": ["select_deck_card"],
        "run": _run(1),
        "selection": {
            "prompt": "选择一张牌",
            "min_select": 1,
            "max_select": 1,
            "cards": [
                {
                    "index": 0,
                    "card_id": "STRIKE_DEFECT",
                    "name": "打击",
                    "cost": 1,
                    "type": "Attack",
                    "description": "造成6点伤害。",
                },
                {
                    "index": 1,
                    "card_id": "DEFEND_DEFECT",
                    "name": "防御",
                    "cost": 1,
                    "type": "Skill",
                    "description": "获得5点格挡。",
                },
            ],
        },
    }


def _boss_map_state() -> dict[str, Any]:
    """返回尚未切幕的 Boss 地图入口。

    Returns:
        dict[str, Any]: 唯一 Boss 目的地的 MAP 宏 checkpoint。
    """
    return {
        "state_revision": 2,
        "screen": "MAP",
        "in_combat": False,
        "available_actions": ["choose_map_node"],
        "run": _run(17),
        "map": {
            "current_node": {"row": 15, "col": 2},
            "available_nodes": [
                {
                    "index": 0,
                    "row": 16,
                    "col": 3,
                    "node_type": "Boss",
                    "is_boss": True,
                },
                {
                    "index": 1,
                    "row": 16,
                    "col": 5,
                    "node_type": "Monster",
                    "is_boss": False,
                },
            ],
        },
    }


def _shop_state(*, inventory_open: bool) -> dict[str, Any]:
    """返回有两个可买项的打开或关闭商店状态。

    Args:
        inventory_open (bool): 是否正在展示库存。

    Returns:
        dict[str, Any]: 同一商店宏 scope 的 SHOP 状态。
    """
    actions = (
        ["buy_card", "buy_relic", "close_shop_inventory"]
        if inventory_open
        else ["open_shop_inventory", "proceed"]
    )
    return {
        "state_revision": 2 if inventory_open else 3,
        "screen": "SHOP",
        "in_combat": False,
        "available_actions": actions,
        "run": _run(5),
        "shop": {
            "is_inventory_open": inventory_open,
            "cards": [
                {
                    "index": 0,
                    "card_id": "ZAP",
                    "name": "电击",
                    "price": 50,
                    "enough_gold": True,
                    "is_stocked": True,
                }
            ],
            "relics": [
                {
                    "index": 0,
                    "relic_id": "STRAWBERRY",
                    "name": "草莓",
                    "price": 80,
                    "enough_gold": True,
                    "is_stocked": True,
                }
            ],
            "potions": [],
            "card_removal": {"available": False},
        },
    }


def _combat_state() -> dict[str, Any]:
    """返回已经开放 end_turn 的最小战斗状态。

    Returns:
        dict[str, Any]: COMBAT 状态。
    """
    return {
        "state_revision": 3,
        "screen": "COMBAT",
        "in_combat": True,
        "turn": 1,
        "available_actions": ["end_turn"],
        "run": _run(2),
        "combat": {
            "player": {
                "current_hp": 75,
                "max_hp": 75,
                "block": 0,
                "energy": 3,
                "stars": 0,
            },
            "enemies": [
                {
                    "index": 0,
                    "name": "敌人",
                    "current_hp": 20,
                    "max_hp": 20,
                    "block": 0,
                }
            ],
            "hand": [],
            "draw_count": 0,
            "discard_count": 0,
        },
    }


def _reward_state() -> dict[str, Any]:
    """返回带卡牌入口的战斗奖励宏 checkpoint。

    Returns:
        dict[str, Any]: REWARD 状态。
    """
    return {
        "state_revision": 4,
        "screen": "REWARD",
        "in_combat": False,
        "available_actions": ["claim_reward", "proceed"],
        "run": _run(2, hp=70),
        "reward": {
            "rewards": [
                {
                    "index": 0,
                    "reward_type": "Card",
                    "name": "卡牌奖励",
                    "claimable": True,
                }
            ]
        },
    }
