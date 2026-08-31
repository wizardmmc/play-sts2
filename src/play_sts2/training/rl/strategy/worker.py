"""组合原生 checkpoint、远程 policy 与独立游戏进程采集 Tree arm。"""

from contextlib import ExitStack
from pathlib import Path

from ....checkpoint import StrategicCheckpoint, restore_strategic_checkpoint
from ....client import GameClient, Health
from ....game_launcher import launch_game
from ....inference import OpenAICompatibleProvider
from ....runtime import BattleRunner, DecisionEngine
from ..forcing import ForcedFirstChoiceProvider
from ..full_run import retryable_full_run_error
from .contracts import TreeRolloutArm
from .rollout import TreeSuffixRunner, build_tree_rollout_arm


class GameTreeRolloutWorker:
    """每条 arm 启动一个新 HOME，避免兄弟分支共享进程状态。"""

    def __init__(
        self,
        *,
        worker_id: str,
        executable: Path,
        profile: Path,
        home_root: Path,
        port: int,
        checkpoint: StrategicCheckpoint,
        strategy_model_url: str,
        battle_model_url: str,
        strategy_policy_version: str,
        battle_policy_version: str,
        max_tokens: int = 128,
        temperature: float = 0.8,
        max_macro_checkpoints: int = 2,
        continue_to_terminal: bool = False,
        max_model_steps: int = 100,
        infrastructure_attempts: int = 2,
    ) -> None:
        """保存两端 policy、游戏启动参数和短 horizon。

        Args:
            worker_id (str): 当前本地 worker 标识。
            executable (Path): v0.111.0 游戏可执行文件。
            profile (Path): 关闭 Steam 与共享存档的 profile。
            home_root (Path): 当前 worker 的 arm HOME 父目录。
            port (int): 当前 worker 独占 Mod 端口。
            checkpoint (StrategicCheckpoint): 所有 arms 共享的原生入口。
            strategy_model_url (str): A100 战略推理端点。
            battle_model_url (str): A100 战斗推理端点。
            strategy_policy_version (str): 冻结战略模型名。
            battle_policy_version (str): 冻结战斗模型名。
            max_tokens (int): 单次模型回复 token 预算。
            temperature (float): 两层冻结 policy 采样温度。
            max_macro_checkpoints (int): suffix 后继宏节点 horizon。
            continue_to_terminal (bool): 是否续玩到整局胜负而非短 horizon。
            max_model_steps (int): suffix 内战略模型动作总上限。
            infrastructure_attempts (int): 每条分支的基础设施总尝试数。

        Returns:
            None: 此方法只保存 worker 配置。
        """
        self.worker_id = worker_id
        self._executable = executable
        self._profile = profile
        self._home_root = home_root
        self._port = port
        self._checkpoint = checkpoint
        self._strategy_model_url = strategy_model_url
        self._battle_model_url = battle_model_url
        self._strategy_policy_version = strategy_policy_version
        self._battle_policy_version = battle_policy_version
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_macro_checkpoints = max_macro_checkpoints
        self._continue_to_terminal = continue_to_terminal
        self._max_model_steps = max_model_steps
        if infrastructure_attempts < 1:
            raise ValueError("Tree 基础设施总尝试数必须为正")
        self._infrastructure_attempts = infrastructure_attempts
        self.last_health: Health | None = None

    def collect_arm(
        self,
        *,
        arm_index: int,
        forced_action: str | None = None,
    ) -> TreeRolloutArm:
        """从共享 checkpoint 新启动游戏并采集一条冻结 policy suffix。

        Args:
            arm_index (int): 当前 K=8 group 内序号。
            forced_action (str | None): 分层 simple macro 的首个根动作。

        Returns:
            TreeRolloutArm: 含战略 token、环境结果与工程 return 的 arm。
        """
        for attempt in range(self._infrastructure_attempts):
            try:
                return self._collect_once(
                    arm_index=arm_index,
                    forced_action=forced_action,
                    attempt=attempt,
                )
            except Exception as exc:
                if (
                    not retryable_full_run_error(exc)
                    or attempt + 1 == self._infrastructure_attempts
                ):
                    raise
        raise RuntimeError("Tree 基础设施重采循环意外结束")

    def _collect_once(
        self,
        *,
        arm_index: int,
        forced_action: str | None,
        attempt: int,
    ) -> TreeRolloutArm:
        """从一个干净 HOME 执行单次 terminal branch 尝试。

        Args:
            arm_index (int): 当前分支序号。
            forced_action (str | None): 可选的分层根动作。
            attempt (int): 基础设施重采序号。

        Returns:
            TreeRolloutArm: 成功完成到终局的分支。
        """
        home = self._home_root / f"arm-{arm_index:02d}-attempt-{attempt}"
        with ExitStack() as stack:
            running = stack.enter_context(
                launch_game(
                    self._executable,
                    port=self._port,
                    home=home,
                    profile=self._profile,
                    mode="headless",
                    enable_debug_actions=True,
                    run_save=self._checkpoint.save_path,
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
            state = restore_strategic_checkpoint(game, self._checkpoint)
            routed_strategy_provider = (
                ForcedFirstChoiceProvider(strategy_provider, forced_action)
                if forced_action is not None
                else strategy_provider
            )
            strategy = DecisionEngine(
                game,
                routed_strategy_provider,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                constrain_actions=True,
            )
            battle = BattleRunner(
                game,
                battle_provider,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                max_retries=0,
                constrain_actions=True,
            )
            seed = self._checkpoint.entry.audit.get("run_id")
            if not isinstance(seed, str) or not seed:
                raise ValueError("terminal Tree checkpoint 缺少完整游戏 seed")
            result = TreeSuffixRunner(
                game=game,
                strategy=strategy,
                battle=battle,
                strategy_policy_version=self._strategy_policy_version,
                battle_policy_version=self._battle_policy_version,
                max_macro_checkpoints=self._max_macro_checkpoints,
                continue_to_terminal=self._continue_to_terminal,
                max_model_steps=self._max_model_steps,
                seed=seed,
            ).run(state)
        return build_tree_rollout_arm(
            result,
            arm_index=arm_index,
            worker_id=self.worker_id,
        )
