"""组合隔离游戏、双远程 policy 与本地 checkpoint 捕获完整 episode。"""

import time
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

from ....checkpoint import (
    capture_strategic_checkpoint,
    restore_strategic_checkpoint,
)
from ....client import GameClient, Health
from ....game_launcher import launch_game
from ....harness import HarnessLayer, build_observation
from ....inference import OpenAICompatibleProvider
from ....run_start import start_run
from ....runtime import RunDecision, RunRoute, RunRunner, classify_run_state
from ..full_run import retryable_full_run_error
from ..strategy import classify_macro_checkpoint
from .rollout import (
    BackboneCheckpointCandidate,
    BackboneEpisodeDraft,
    build_gigpo_episode,
)
from .scenario import BackboneBattleCandidate, extract_battle_candidate


class _BackboneStateCapture:
    """在完整游戏中记录审计、原生宏 checkpoint 与战斗入口。"""

    def __init__(
        self,
        *,
        game: GameClient,
        home: Path,
        checkpoint_root: Path,
        seed: str,
        early_target_ordinal: int,
        late_target_ordinal: int,
    ) -> None:
        """保存当前游戏与本局候选输出位置。

        Args:
            game (GameClient): 当前独占游戏客户端。
            home (Path): 当前隔离 HOME。
            checkpoint_root (Path): 本局 checkpoint 输出目录。
            seed (str): 当前完整游戏种子。
            early_target_ordinal (int): 楼层十以前要物化的候选序号。
            late_target_ordinal (int): 楼层十起要物化的候选序号。
        """
        self._game = game
        self._home = home
        self._checkpoint_root = checkpoint_root
        self._seed = seed
        self._early_target_ordinal = early_target_ordinal
        self._late_target_ordinal = late_target_ordinal
        self._seen_macro_scopes: set[tuple[str, ...]] = set()
        self._early_candidates = 0
        self._late_candidates = 0
        self._pending_strategy_audit: Mapping[str, Any] | None = None
        self.strategy_audits: list[Mapping[str, Any]] = []
        self.checkpoints: list[BackboneCheckpointCandidate] = []
        self.battles: list[BackboneBattleCandidate] = []

    def __call__(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """观察一个稳定状态，并在宏 checkpoint 存退后恢复原页。

        Args:
            state (Mapping[str, Any]): 当前稳定游戏状态。

        Returns:
            Mapping[str, Any]: 原状态或原生 continue 后的等价入口。
        """
        route = classify_run_state(state)
        if route is RunRoute.BATTLE:
            audit = self._game.checkpoint_audit()
            self.battles.append(
                extract_battle_candidate(
                    state,
                    audit,
                    seed=self._seed,
                    battle_index=len(self.battles),
                )
            )
            return state
        if route is not RunRoute.STRATEGIC:
            return state
        audit = self._game.checkpoint_audit()
        self._pending_strategy_audit = audit
        checkpoint = classify_macro_checkpoint(state)
        screen = str(state.get("screen") or "")
        if checkpoint is None or screen not in {
            "MAP",
            "REWARD",
            "SHOP",
            "REST",
            "EVENT",
        }:
            return state
        scope = (checkpoint.kind, *checkpoint.scope_id)
        if scope in self._seen_macro_scopes:
            return state
        self._seen_macro_scopes.add(scope)
        run = state.get("run")
        floor = run.get("floor") if isinstance(run, Mapping) else None
        if isinstance(floor, bool) or not isinstance(floor, int) or floor < 0:
            raise ValueError("backbone checkpoint 缺少有效楼层")
        save_is_self_contained = _native_save_is_self_contained(screen, audit)
        if floor < 10:
            ordinal = self._early_candidates
            if save_is_self_contained:
                self._early_candidates += 1
            should_capture = (
                save_is_self_contained and ordinal == self._early_target_ordinal
            )
        else:
            ordinal = self._late_candidates
            if save_is_self_contained:
                self._late_candidates += 1
            should_capture = (
                save_is_self_contained and ordinal == self._late_target_ordinal
            )
        captured_path = None
        if should_capture:
            destination = self._checkpoint_root / (
                f"checkpoint-{len(self.checkpoints):03d}-{checkpoint.kind}"
            )
            captured = capture_strategic_checkpoint(
                self._game,
                home=self._home,
                destination=destination,
            )
            captured_path = captured.root
        self.checkpoints.append(
            BackboneCheckpointCandidate(
                index=len(self.checkpoints),
                kind=checkpoint.kind,
                floor=floor,
                option_ids=checkpoint.option_ids,
                policy_text=build_observation(state).text,
                path=captured_path,
            )
        )
        return (
            restore_strategic_checkpoint(self._game, captured)
            if captured_path is not None
            else state
        )

    def record_decision(self, decision: RunDecision) -> None:
        """在战略动作成功后提交与其对应的 pending audit。

        Args:
            decision (RunDecision): 已经进入整局时间线的成功决策。

        Raises:
            ValueError: 战略决策没有对应的稳定状态审计。
        """
        if decision.layer is not HarnessLayer.STRATEGIC:
            return
        if self._pending_strategy_audit is None:
            raise ValueError("成功战略决策缺少 pending checkpoint 审计")
        self.strategy_audits.append(self._pending_strategy_audit)
        self._pending_strategy_audit = None


class GameBackboneWorker:
    """每条 episode 启动一个新隔离 HOME 并使用双远程 policy 完成游戏。"""

    def __init__(
        self,
        *,
        worker_id: str,
        executable: Path,
        profile: Path,
        home_root: Path,
        checkpoint_root: Path,
        port: int,
        strategy_model_url: str,
        battle_model_url: str,
        strategy_policy_version: str,
        battle_policy_version: str,
        seed: str,
        character_id: str,
        ascension: int,
        max_tokens: int = 128,
        temperature: float = 0.8,
        infrastructure_attempts: int = 3,
    ) -> None:
        """保存本地游戏与 A100 双 residual 参数。

        Args:
            worker_id (str): 本地 worker 名称。
            executable (Path): v0.111.0 游戏可执行文件。
            profile (Path): 关闭 Steam 与共享存档的 profile。
            home_root (Path): 当前 worker 全部 episode HOME 父目录。
            checkpoint_root (Path): 当前 worker checkpoint 父目录。
            port (int): 当前 worker 独占 Mod 端口。
            strategy_model_url (str): A100 战略推理端点。
            battle_model_url (str): A100 战斗推理端点。
            strategy_policy_version (str): 冻结战略 residual 名。
            battle_policy_version (str): 冻结战斗 residual 名。
            seed (str): 当前组共享游戏种子。
            character_id (str): 角色稳定 ID。
            ascension (int): 进阶等级。
            max_tokens (int): 单次回复 token 预算。
            temperature (float): 两层冻结采样温度。
            infrastructure_attempts (int): 网络或超时故障时从干净 HOME 重采同一
                episode 的总尝试数。

        Raises:
            ValueError: 基础设施尝试数小于一。
        """
        self.worker_id = worker_id
        self._executable = executable
        self._profile = profile
        self._home_root = home_root
        self._checkpoint_root = checkpoint_root
        self._port = port
        self._strategy_model_url = strategy_model_url
        self._battle_model_url = battle_model_url
        self._strategy_policy_version = strategy_policy_version
        self._battle_policy_version = battle_policy_version
        self._seed = seed
        self._character_id = character_id
        self._ascension = ascension
        self._max_tokens = max_tokens
        self._temperature = temperature
        if infrastructure_attempts < 1:
            raise ValueError("backbone 基础设施总尝试数必须为正")
        self._infrastructure_attempts = infrastructure_attempts
        self.last_health: Health | None = None

    def collect_arm(self, *, arm_index: int) -> BackboneEpisodeDraft:
        """启动新局并采集一条完整双 policy episode。

        Args:
            arm_index (int): 同种子八局组内序号。

        Returns:
            BackboneEpisodeDraft: 训练 episode、精确审计和候选入口。
        """
        for attempt in range(self._infrastructure_attempts):
            try:
                return self._collect_once(arm_index=arm_index, attempt=attempt)
            except Exception as exc:
                if (
                    not retryable_full_run_error(exc)
                    or attempt + 1 == self._infrastructure_attempts
                ):
                    raise
        raise RuntimeError("backbone 基础设施重采循环意外结束")

    def _collect_once(
        self,
        *,
        arm_index: int,
        attempt: int,
    ) -> BackboneEpisodeDraft:
        """从一个全新 HOME 完成单次完整游戏尝试。

        Args:
            arm_index (int): 同种子八局组内序号。
            attempt (int): 当前基础设施尝试的零基序号。

        Returns:
            BackboneEpisodeDraft: 一次无基础设施故障的完整 episode。
        """
        suffix = f"arm-{arm_index:02d}"
        if attempt:
            suffix += f"-retry-{attempt}"
        home = self._home_root / suffix
        checkpoint_root = self._checkpoint_root / suffix
        started = time.monotonic()
        with ExitStack() as stack:
            running = stack.enter_context(
                launch_game(
                    self._executable,
                    port=self._port,
                    home=home,
                    profile=self._profile,
                    mode="headless",
                    enable_debug_actions=True,
                )
            )
            game = stack.enter_context(GameClient(running.base_url))
            strategy_provider = stack.enter_context(
                OpenAICompatibleProvider(
                    self._strategy_model_url,
                    model=self._strategy_policy_version,
                    enable_thinking=False,
                    capture_token_metadata=True,
                )
            )
            battle_provider = stack.enter_context(
                OpenAICompatibleProvider(
                    self._battle_model_url,
                    model=self._battle_policy_version,
                    enable_thinking=False,
                    capture_token_metadata=False,
                )
            )
            self.last_health = game.health()
            initial = start_run(
                game,
                self._character_id,
                seed=self._seed,
                ascension=self._ascension,
            )
            capture = _BackboneStateCapture(
                game=game,
                home=home,
                checkpoint_root=checkpoint_root,
                seed=self._seed,
                early_target_ordinal=arm_index % 4,
                late_target_ordinal=arm_index % 3,
            )
            result = RunRunner(
                game,
                strategy_provider=strategy_provider,
                battle_provider=battle_provider,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                max_retries=0,
                max_conflict_retries=3,
                constrain_actions=True,
                stable_state_hook=capture,
                decision_hook=capture.record_decision,
                capture_policy_failures=True,
            ).run(initial)
        elapsed = time.monotonic() - started
        if self.last_health is None:
            raise ValueError("backbone worker 缺少游戏运行时收据")
        episode = build_gigpo_episode(
            result,
            arm_index=arm_index,
            worker_id=self.worker_id,
            seed=self._seed,
            character_id=self._character_id,
            ascension=self._ascension,
            strategy_policy_version=self._strategy_policy_version,
            battle_policy_version=self._battle_policy_version,
            anchor_audits=capture.strategy_audits,
            elapsed_seconds=elapsed,
        )
        completed_battles = tuple(
            _complete_battle_candidate(candidate, battle)
            for candidate, battle in zip(
                capture.battles[: len(result.battle_results)],
                result.battle_results,
                strict=True,
            )
        )
        if len(capture.battles) == len(result.battle_results):
            battles = completed_battles
        elif (
            result.model_error
            and len(capture.battles) == len(result.battle_results) + 1
        ):
            battles = (
                *completed_battles,
                _complete_failed_battle_candidate(
                    capture.battles[-1],
                    result.final_state,
                ),
            )
        else:
            raise ValueError("backbone 战斗候选与 Runtime 战斗结果没有逐项对齐")
        return BackboneEpisodeDraft(
            episode=episode,
            anchor_audits=tuple(capture.strategy_audits),
            checkpoints=tuple(capture.checkpoints),
            battles=battles,
            health=self.last_health,
        )


def _native_save_is_self_contained(
    screen: str,
    audit: Mapping[str, Any],
) -> bool:
    """判断当前页面的原生保存局能否独立恢复到同一入口。

    Args:
        screen (str): 当前模型可见页面。
        audit (Mapping[str, Any]): 与页面同一时刻的隐藏房间审计。

    Returns:
        bool: 已知战斗奖励后的 MAP 返回 ``False``，其余页面保持可捕获。
    """
    if screen != "MAP":
        return True
    run = audit.get("run")
    room = run.get("current_room") if isinstance(run, Mapping) else None
    return not isinstance(room, Mapping) or room.get("room_type") != "Monster"


def _complete_battle_candidate(
    candidate: BackboneBattleCandidate,
    result: Any,
) -> BackboneBattleCandidate:
    """用 Runtime 离场结果补齐战斗候选的损失与结果。

    Args:
        candidate (BackboneBattleCandidate): 第一回合捕获的真实入口。
        result (Any): 对应 ``BattleResult``。

    Raises:
        ValueError: 离场状态缺少可见生命值。

    Returns:
        BackboneBattleCandidate: 带 outcome 与生命损失比例的候选。
    """
    final_run = result.final_state.get("run")
    if not isinstance(final_run, Mapping):
        raise TypeError("backbone 战斗离场缺少 run 状态")
    final_hp = final_run.get("current_hp")
    entry_hp = candidate.scenario.current_hp
    if (
        isinstance(final_hp, bool)
        or not isinstance(final_hp, int)
        or entry_hp is None
        or entry_hp <= 0
    ):
        raise ValueError("backbone 战斗离场生命值无效")
    loss = max(0, entry_hp - max(0, final_hp)) / entry_hp
    return replace(
        candidate,
        outcome=result.outcome.value,
        hp_loss_ratio=loss,
    )


def _complete_failed_battle_candidate(
    candidate: BackboneBattleCandidate,
    state: Mapping[str, Any],
) -> BackboneBattleCandidate:
    """把完整 run 中的战斗策略失败补成困难场景候选。

    Args:
        candidate (BackboneBattleCandidate): 已捕获的真实战斗入口。
        state (Mapping[str, Any]): 策略失败时最后可靠状态。

    Returns:
        BackboneBattleCandidate: outcome 为 ``model_error`` 的入口。
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
