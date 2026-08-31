"""执行冻结战略/战斗 policy 的宏 checkpoint suffix。"""

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol

import httpx

from ....client import GameClient
from ....harness import build_observation, format_action
from ....inference import InferenceGenerationTruncated
from ....runtime import (
    BattleOutcome,
    BattlePolicyFailure,
    BattleResult,
    DecisionGenerationProfile,
    DecisionRetriesExhausted,
    DecisionStep,
    RunRoute,
    classify_run_state,
    project_strategic_model_state,
)
from ....runtime.decision import (
    is_action_window_conflict,
    stale_state_from_conflict,
    state_revision,
)
from .contracts import StrategicReturn, TreeRolloutArm, TreeRolloutStep
from .macro import (
    MacroCheckpoint,
    automatic_strategic_action,
    classify_macro_checkpoint,
    same_macro_scope,
)
from .reward import StrategicReturnInput, score_engineering_milestone_return


class TreeRolloutError(RuntimeError):
    """表示 suffix 无法形成同策略、可训练的战略轨迹。"""


class StrategicStepRunner(Protocol):
    """声明 suffix runner 使用的单步战略 policy。"""

    def step(self, state: Mapping[str, Any]) -> DecisionStep:
        """生成并执行一个战略动作。

        Args:
            state (Mapping[str, Any]): 当前稳定战略状态。

        Returns:
            DecisionStep: 含真实 token 行为概率的动作。
        """
        ...


class BattleSuffixRunner(Protocol):
    """声明 suffix 中冻结战斗 policy 的运行接口。"""

    def run(self, initial_state: Mapping[str, Any] | None = None) -> BattleResult:
        """完成当前战斗。

        Args:
            initial_state (Mapping[str, Any] | None): 当前战斗入口。

        Returns:
            BattleResult: 战斗终局与全部冻结 policy 步骤。
        """
        ...


@dataclass(frozen=True, slots=True)
class TreeSuffixResult:
    """保存一条宏 suffix 的战略训练事实和环境结果。

    Args:
        strategy_policy_version (str): 冻结战略 policy。
        battle_policy_version (str): 冻结战斗 policy。
        generation_profile (DecisionGenerationProfile): 战略生成参数。
        max_macro_checkpoints (int): 后继宏节点 horizon 预算。
        plan_id (str): 首个完整宏计划的规范动作轨迹。
        plan_step_count (int): ``steps`` 中属于首个宏计划的动作数。
        macro_successor_text (str): 首个宏计划完成后的玩家可见语义状态。
        steps (tuple[TreeRolloutStep, ...]): 仅战略动作 token。
        final_state (dict[str, Any]): horizon 结束状态。
        horizon_reason (str): 下一宏节点、胜利或死亡。
        continuation_return (StrategicReturn): 工程 suffix return。
        elapsed_seconds (float): 当前 suffix 游戏时间。
        battle_candidates (tuple[Any, ...]): suffix 暴露的真实战斗入口。
        failure_reason (str | None): 明确策略失败诊断。
    """

    strategy_policy_version: str
    battle_policy_version: str
    generation_profile: DecisionGenerationProfile
    max_macro_checkpoints: int
    plan_id: str
    plan_step_count: int
    macro_successor_text: str
    steps: tuple[TreeRolloutStep, ...]
    final_state: dict[str, Any]
    horizon_reason: str
    continuation_return: StrategicReturn
    elapsed_seconds: float
    battle_candidates: tuple[Any, ...] = ()
    failure_reason: str | None = None


class TreeSuffixRunner:
    """从一个宏 checkpoint 续跑到固定数量的后继宏节点。"""

    def __init__(
        self,
        *,
        game: GameClient,
        strategy: StrategicStepRunner,
        battle: BattleSuffixRunner,
        strategy_policy_version: str,
        battle_policy_version: str,
        max_macro_checkpoints: int,
        continue_to_terminal: bool = False,
        max_model_steps: int = 100,
        seed: str = "",
    ) -> None:
        """保存冻结 policy、游戏客户端和短 horizon。

        Args:
            game (GameClient): 当前独占游戏实例。
            strategy (StrategicStepRunner): 冻结战略 policy。
            battle (BattleSuffixRunner): 冻结战斗 policy。
            strategy_policy_version (str): 期望战略模型身份。
            battle_policy_version (str): 期望战斗模型身份。
            max_macro_checkpoints (int): suffix 最多跨越的后继宏节点数。
            continue_to_terminal (bool): 是否忽略短 horizon 并续玩到整局终局。
            max_model_steps (int): 战略模型动作上限。
            seed (str): 当前原生 checkpoint 所属完整游戏种子。

        Raises:
            ValueError: policy 身份为空或预算不是正数。

        Returns:
            None: 此方法只初始化 runner。
        """
        if (
            not strategy_policy_version
            or not battle_policy_version
            or max_macro_checkpoints <= 0
            or max_model_steps <= 0
        ):
            raise ValueError("Tree suffix policy 与 horizon 必须有效")
        self._game = game
        self._strategy = strategy
        self._battle = battle
        self._strategy_policy_version = strategy_policy_version
        self._battle_policy_version = battle_policy_version
        self._max_macro_checkpoints = max_macro_checkpoints
        self._continue_to_terminal = continue_to_terminal
        self._max_model_steps = max_model_steps
        self._seed = seed

    def run(self, entry_state: Mapping[str, Any]) -> TreeSuffixResult:
        """执行一条宏 suffix 并只保留战略 policy token。

        Args:
            entry_state (Mapping[str, Any]): 已恢复并核对的宏 checkpoint 状态。

        Raises:
            TreeRolloutError: policy 漂移、token 缺失、未知页面或预算耗尽。

        Returns:
            TreeSuffixResult: 可构造 Tree arm 的完整结果。
        """
        started = time.monotonic()
        current = dict(entry_state)
        entry_floor = _run_integer(current, "floor")
        entry_checkpoint = classify_macro_checkpoint(current)
        if entry_checkpoint is None:
            raise TreeRolloutError("Tree suffix 入口不是宏 checkpoint")
        strategic_steps: list[TreeRolloutStep] = []
        generation_profile: DecisionGenerationProfile | None = None
        plan_open = True
        plan_step_count = 0
        macro_successor_text = ""
        continuation_macro: MacroCheckpoint | None = None
        macro_count = 0
        bosses_cleared = 0
        pending_battle_is_boss = False
        card_reward_skipped = False
        shop_inventory_closed = False
        conflict_retries = 0
        died = False
        horizon_reason = "next_macro_checkpoint"
        battle_candidates = []
        failure_reason = None

        while True:
            screen = str(current.get("screen") or "")
            if screen not in {"REWARD", "CARD_SELECTION"}:
                card_reward_skipped = False
            if screen != "SHOP":
                shop_inventory_closed = False
            model_state = project_strategic_model_state(
                current,
                card_reward_skipped=card_reward_skipped,
                shop_inventory_closed=shop_inventory_closed,
            )
            route = classify_run_state(model_state)
            if route is RunRoute.TERMINAL:
                if plan_open:
                    plan_open = False
                    macro_successor_text = _semantic_successor_text(model_state)
                game_over = current.get("game_over")
                victory = (
                    isinstance(game_over, Mapping)
                    and game_over.get("is_victory") is True
                )
                died = not victory
                horizon_reason = "victory" if victory else "died"
                break
            if route is RunRoute.UNKNOWN:
                raise TreeRolloutError(
                    f"Tree suffix 遇到未知页面: {current.get('screen')}"
                )
            if route is RunRoute.TRANSIENT:
                raise TreeRolloutError("Tree suffix 收到未稳定状态")
            if route is RunRoute.BATTLE:
                if plan_open:
                    plan_open = False
                    macro_successor_text = _semantic_successor_text(model_state)
                continuation_macro = None
                candidate = None
                if self._seed:
                    from ..gigpo.scenario import extract_battle_candidate

                    candidate = extract_battle_candidate(
                        current,
                        self._game.checkpoint_audit(),
                        seed=self._seed,
                        battle_index=len(battle_candidates),
                    )
                try:
                    result = self._battle.run(current)
                except BattlePolicyFailure as exc:
                    if candidate is not None:
                        battle_candidates.append(
                            _failed_tree_battle_candidate(candidate, exc.failure_state)
                        )
                    current = dict(exc.failure_state)
                    died = True
                    horizon_reason = "model_error"
                    failure_reason = str(exc)
                    break
                _validate_battle_policy(result, self._battle_policy_version)
                if candidate is not None:
                    battle_candidates.append(
                        _complete_tree_battle_candidate(candidate, result)
                    )
                current = dict(result.final_state)
                if pending_battle_is_boss and result.outcome is BattleOutcome.CLEARED:
                    bosses_cleared += 1
                pending_battle_is_boss = False
                if result.outcome is BattleOutcome.DIED:
                    died = True
                    horizon_reason = "died"
                    break
                continue

            checkpoint = classify_macro_checkpoint(model_state)
            if plan_open and checkpoint is not None:
                if not same_macro_scope(entry_checkpoint, checkpoint):
                    plan_open = False
                    macro_successor_text = _semantic_successor_text(model_state)
                    continuation_macro = checkpoint
                    macro_count += 1
                    if (
                        not self._continue_to_terminal
                        and macro_count >= self._max_macro_checkpoints
                    ):
                        break
            elif (
                not plan_open
                and checkpoint is not None
                and (
                    continuation_macro is None
                    or not same_macro_scope(continuation_macro, checkpoint)
                )
            ):
                continuation_macro = checkpoint
                macro_count += 1
                if (
                    not self._continue_to_terminal
                    and macro_count >= self._max_macro_checkpoints
                ):
                    break
            automatic = automatic_strategic_action(model_state)
            if automatic is not None:
                current = _execute_automatic(
                    self._game, model_state, automatic.name, automatic.parameters
                )
                continue
            if len(strategic_steps) >= self._max_model_steps:
                if plan_open:
                    raise TreeRolloutError("Tree 入口计划超过战略模型动作预算")
                died = True
                horizon_reason = "model_error"
                failure_reason = (
                    f"Tree suffix 超过战略模型动作预算: {self._max_model_steps}"
                )
                break
            try:
                decision = self._strategy.step(model_state)
            except (DecisionRetriesExhausted, InferenceGenerationTruncated) as exc:
                if plan_open:
                    raise TreeRolloutError(
                        "Tree 入口计划在首个可训练动作前策略失败"
                    ) from exc
                died = True
                horizon_reason = "model_error"
                failure_reason = str(exc)
                break
            except httpx.HTTPStatusError as exc:
                latest = stale_state_from_conflict(exc)
                if latest is None and not is_action_window_conflict(exc):
                    raise
                if conflict_retries >= 3:
                    raise TreeRolloutError("Tree 战略动作窗口冲突重试耗尽") from exc
                conflict_retries += 1
                current = _wait_after_action_conflict(
                    self._game,
                    model_state,
                    latest=latest,
                )
                continue
            conflict_retries = 0
            projected, profile = _project_tree_step(
                decision,
                expected_policy=self._strategy_policy_version,
                index=len(strategic_steps),
            )
            if generation_profile is None:
                generation_profile = profile
            elif generation_profile != profile:
                raise TreeRolloutError("Tree suffix 战略生成参数发生漂移")
            strategic_steps.append(projected)
            if plan_open:
                plan_step_count += 1
            pending_battle_is_boss = pending_battle_is_boss or _chosen_map_node_is_boss(
                decision
            )
            if decision.action.name == "skip_reward_cards":
                card_reward_skipped = True
            if decision.action.name == "close_shop_inventory":
                shop_inventory_closed = True
            current = _decision_state(self._game, decision)

        if (
            not strategic_steps
            or generation_profile is None
            or plan_step_count <= 0
            or not macro_successor_text
        ):
            raise TreeRolloutError("Tree suffix 没有可训练战略动作")
        final_run = current.get("run")
        if not isinstance(final_run, Mapping):
            raise TreeRolloutError("Tree suffix 终点缺少 run 状态")
        victory = horizon_reason == "victory"
        reward = score_engineering_milestone_return(
            StrategicReturnInput(
                entry_floor=entry_floor,
                final_floor=_run_integer(current, "floor"),
                final_hp=0 if died else _mapping_integer(final_run, "current_hp"),
                max_hp=_mapping_integer(final_run, "max_hp"),
                bosses_cleared=bosses_cleared,
                victory=victory,
                died=died,
            )
        )
        elapsed = time.monotonic() - started
        if not math.isfinite(elapsed):
            raise TreeRolloutError("Tree suffix 采样时间无效")
        return TreeSuffixResult(
            strategy_policy_version=self._strategy_policy_version,
            battle_policy_version=self._battle_policy_version,
            generation_profile=generation_profile,
            max_macro_checkpoints=self._max_macro_checkpoints,
            plan_id="\n".join(
                step.action for step in strategic_steps[:plan_step_count]
            ),
            plan_step_count=plan_step_count,
            macro_successor_text=macro_successor_text,
            steps=tuple(strategic_steps),
            final_state=current,
            horizon_reason=horizon_reason,
            continuation_return=reward,
            elapsed_seconds=elapsed,
            battle_candidates=tuple(battle_candidates),
            failure_reason=failure_reason,
        )


def build_tree_rollout_arm(
    result: TreeSuffixResult,
    *,
    arm_index: int,
    worker_id: str,
) -> TreeRolloutArm:
    """把一条 suffix 结果投影为 Tree group arm。

    Args:
        result (TreeSuffixResult): 冻结 policy 完成的 suffix。
        arm_index (int): 当前 K=8 组内序号。
        worker_id (str): 执行该 suffix 的本地 worker。

    Returns:
        TreeRolloutArm: 可进入 Tree group 准入的 arm。
    """
    return TreeRolloutArm(
        arm_index=arm_index,
        worker_id=worker_id,
        strategy_policy_version=result.strategy_policy_version,
        battle_policy_version=result.battle_policy_version,
        behavior_logprobs_mode="processed_logprobs",
        action_constraint_mode="vllm_structured_choice",
        generation_profile=result.generation_profile,
        max_macro_checkpoints=result.max_macro_checkpoints,
        plan_id=result.plan_id,
        plan_step_count=result.plan_step_count,
        macro_successor_text=result.macro_successor_text,
        steps=result.steps,
        final_state=result.final_state,
        horizon_reason=result.horizon_reason,
        continuation_return=result.continuation_return,
        elapsed_seconds=result.elapsed_seconds,
        battle_candidates=result.battle_candidates,
        failure_reason=result.failure_reason,
    )


def _complete_tree_battle_candidate(candidate: Any, result: BattleResult) -> Any:
    """用 terminal suffix 的真实离场结果补齐战斗候选。

    Args:
        candidate (Any): 战斗第一回合构造的候选。
        result (BattleResult): 冻结战斗 policy 的离场结果。

    Raises:
        ValueError: 离场生命值无法形成可见损失比例。

    Returns:
        Any: 带 outcome 与 HP 损失的候选数据类。
    """
    final_run = result.final_state.get("run")
    if not isinstance(final_run, Mapping):
        raise TypeError("Tree 战斗离场缺少 run 状态")
    final_hp = final_run.get("current_hp")
    entry_hp = candidate.scenario.current_hp
    if (
        isinstance(final_hp, bool)
        or not isinstance(final_hp, int)
        or entry_hp is None
        or entry_hp <= 0
    ):
        raise ValueError("Tree 战斗离场生命值无效")
    return replace(
        candidate,
        outcome=result.outcome.value,
        hp_loss_ratio=max(0, entry_hp - max(0, final_hp)) / entry_hp,
    )


def _failed_tree_battle_candidate(candidate: Any, state: Mapping[str, Any]) -> Any:
    """把战斗策略失败保留为场景选择中的困难入口。

    Args:
        candidate (Any): 战斗第一回合候选。
        state (Mapping[str, Any]): 失败时最后可靠游戏状态。

    Returns:
        Any: outcome 为 ``model_error`` 的候选数据类。
    """
    run = state.get("run")
    final_hp = run.get("current_hp") if isinstance(run, Mapping) else 0
    if isinstance(final_hp, bool) or not isinstance(final_hp, int):
        final_hp = 0
    entry_hp = candidate.scenario.current_hp or 1
    return replace(
        candidate,
        outcome="model_error",
        hp_loss_ratio=max(0, entry_hp - max(0, final_hp)) / entry_hp,
    )


def _semantic_successor_text(state: Mapping[str, Any]) -> str:
    """渲染首个宏计划结束后的玩家可见语义状态。

    Args:
        state (Mapping[str, Any]): 地图、战斗入口、后继宏节点或终局状态。

    Returns:
        str: 用于同组语义后继去重的无隐藏信息文本。
    """
    if classify_run_state(state) is RunRoute.TERMINAL:
        game_over = state.get("game_over")
        victory = isinstance(game_over, Mapping) and game_over.get("is_victory") is True
        return f"GAME_OVER|victory={victory}"
    return build_observation(state).text


def _chosen_map_node_is_boss(step: DecisionStep) -> bool:
    """判断规范地图动作是否选择玩家可见的 Boss 节点。

    Boss 战结束后的奖励页仍保留旧 ``act_id``，所以必须在进入战斗前记录地图
    语义，不能依赖战斗返回瞬间是否已经切幕。

    Args:
        step (DecisionStep): 已通过 Harness 校验并执行的战略步骤。

    Returns:
        bool: 选择 ``node_type=Boss`` 或 ``is_boss=true`` 的节点时为真。
    """
    if step.action.name != "choose_map_node":
        return False
    selected = step.action.parameters.get("option_index")
    map_state = step.before_state.get("map")
    nodes = map_state.get("available_nodes") if isinstance(map_state, Mapping) else None
    for fallback_index, node in enumerate(nodes or []):
        if not isinstance(node, Mapping):
            continue
        index = node.get("index", fallback_index)
        if index == selected:
            return node.get("is_boss") is True or node.get("node_type") == "Boss"
    return False


def _project_tree_step(
    step: DecisionStep,
    *,
    expected_policy: str,
    index: int,
) -> tuple[TreeRolloutStep, DecisionGenerationProfile]:
    """把成功的 Runtime 决策投影为战略训练步骤。

    Args:
        step (DecisionStep): 已执行的单步战略动作。
        expected_policy (str): 当前冻结战略模型身份。
        index (int): Tree arm 内步骤序号。

    Raises:
        TreeRolloutError: 重试、模型、token、候选或 profile 不符合契约。

    Returns:
        tuple[TreeRolloutStep, DecisionGenerationProfile]: 训练步骤和生成参数。
    """
    if step.retry_errors or len(step.replies) != 1:
        raise TreeRolloutError("Tree 训练 rollout 不允许动作内模型重试")
    reply = step.reply
    action = format_action(step.action)
    if (
        reply.model != expected_policy
        or reply.text != action
        or not reply.token_ids
        or len(reply.token_ids) != len(reply.behavior_logprobs)
        or any(
            not math.isfinite(value) or value > 0 for value in reply.behavior_logprobs
        )
        or not step.response_choices
        or action not in step.response_choices
        or not reply.finish_reason
        or step.generation_profile is None
    ):
        raise TreeRolloutError("Tree 战略步骤缺少真实行为策略 token 事实")
    return (
        TreeRolloutStep(
            index=index,
            messages=step.messages,
            reply_text=reply.text,
            action=action,
            token_ids=reply.token_ids,
            behavior_logprobs=reply.behavior_logprobs,
            response_choices=step.response_choices,
            finish_reason=reply.finish_reason,
        ),
        step.generation_profile,
    )


def _validate_battle_policy(result: BattleResult, expected_policy: str) -> None:
    """确认 suffix 中所有战斗动作来自同一冻结 policy。

    Args:
        result (BattleResult): 当前完整战斗结果。
        expected_policy (str): 期望战斗模型身份。

    Raises:
        TreeRolloutError: 战斗没有步骤、发生重试或模型身份漂移。

    Returns:
        None: 战斗 policy 全程冻结时返回。
    """
    if not result.steps or any(
        step.retry_errors
        or len(step.replies) != 1
        or step.reply.model != expected_policy
        for step in result.steps
    ):
        raise TreeRolloutError("Tree suffix 战斗 policy 缺失或发生漂移")


def _decision_state(game: GameClient, step: DecisionStep) -> dict[str, Any]:
    """读取战略动作后的稳定状态。

    Args:
        game (GameClient): 当前游戏客户端。
        step (DecisionStep): 已执行战略决策。

    Raises:
        TreeRolloutError: 动作结果没有状态或 revision。

    Returns:
        dict[str, Any]: 动作后的稳定状态。
    """
    state = step.action_result.get("state")
    if not isinstance(state, Mapping):
        raise TreeRolloutError("Tree 战略动作没有返回状态")
    if (
        step.action_result.get("stable") is True
        and classify_run_state(state) is not RunRoute.TRANSIENT
    ):
        return dict(state)
    return _wait_for_stable_route(game, state)


def _execute_automatic(
    game: GameClient,
    state: Mapping[str, Any],
    action: str,
    parameters: Mapping[str, int],
) -> dict[str, Any]:
    """执行一个唯一纯 UI 动作并取得稳定状态。

    Args:
        game (GameClient): 当前游戏客户端。
        state (Mapping[str, Any]): 动作前状态。
        action (str): 自动动作名。
        parameters (Mapping[str, int]): 动作整数参数。

    Raises:
        TreeRolloutError: revision 或响应状态缺失。

    Returns:
        dict[str, Any]: 自动动作后的稳定状态。
    """
    revision = state.get("state_revision")
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise TreeRolloutError("Tree 自动动作缺少 revision")
    result = game.execute_action(
        action,
        expected_state_revision=revision,
        **parameters,
    )
    next_state = result.get("state")
    if not isinstance(next_state, Mapping):
        raise TreeRolloutError("Tree 自动动作没有返回状态")
    if (
        result.get("stable") is True
        and classify_run_state(next_state) is not RunRoute.TRANSIENT
    ):
        return dict(next_state)
    return _wait_for_stable_route(game, next_state)


def _wait_for_stable_route(
    game: GameClient,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """跨过连续过渡 revision，等待可路由的下一状态。

    Args:
        game (GameClient): 当前游戏客户端。
        candidate (Mapping[str, Any]): 动作响应中最新候选状态。

    Raises:
        TreeRolloutError: 候选缺少 revision 或等待稳定状态超时。

    Returns:
        dict[str, Any]: 首份非过渡状态。
    """
    deadline = time.monotonic() + game.action_timeout
    state = dict(candidate)
    while classify_run_state(state) is RunRoute.TRANSIENT:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TreeRolloutError("Tree suffix 等待稳定状态超时")
        try:
            state = game.wait_for_state(
                after_revision=state_revision(state),
                timeout=remaining,
            )
        except TimeoutError as exc:
            raise TreeRolloutError("Tree suffix 等待稳定状态超时") from exc
    return state


def _wait_after_action_conflict(
    game: GameClient,
    state: Mapping[str, Any],
    *,
    latest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """等待瞬时动作窗口冲突后的新 revision。

    Args:
        game (GameClient): 当前游戏客户端。
        state (Mapping[str, Any]): 产生冲突的模型状态。
        latest (Mapping[str, Any] | None): stale-state 响应携带的最新状态。

    Raises:
        TreeRolloutError: 等待新状态超时。

    Returns:
        dict[str, Any]: 可重新查询冻结 policy 的稳定状态。
    """
    if latest is not None:
        candidate = dict(latest)
    else:
        try:
            candidate = game.wait_for_state(
                after_revision=state_revision(state),
                timeout=game.action_timeout,
            )
        except TimeoutError as exc:
            raise TreeRolloutError("Tree 动作窗口冲突后等待状态超时") from exc
    return _wait_for_stable_route(game, candidate)


def _run_integer(state: Mapping[str, Any], field: str) -> int:
    """读取状态 run 子对象中的非负整数。

    Args:
        state (Mapping[str, Any]): 当前游戏状态。
        field (str): run 字段名。

    Raises:
        TreeRolloutError: run 或整数字段无效。

    Returns:
        int: 校验后的值。
    """
    run = state.get("run")
    if not isinstance(run, Mapping):
        raise TreeRolloutError("Tree 状态缺少 run")
    return _mapping_integer(run, field)


def _mapping_integer(value: Mapping[str, Any], field: str) -> int:
    """读取映射中的非负整数。

    Args:
        value (Mapping[str, Any]): 当前映射。
        field (str): 字段名。

    Raises:
        TreeRolloutError: 字段不是非负整数。

    Returns:
        int: 校验后的值。
    """
    item = value.get(field)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise TreeRolloutError(f"Tree 字段必须是非负整数: {field}")
    return item
