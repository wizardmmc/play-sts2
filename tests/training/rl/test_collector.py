"""验证多游戏 worker 的严格同入口 K=8 战斗收集。"""

import importlib
from typing import Any

import httpx
import pytest

from play_sts2 import RunStartError
from play_sts2.harness import HarnessAction, HarnessLayer, Observation
from play_sts2.inference import (
    ChatMessage,
    InferenceGenerationTruncated,
    ModelReply,
)
from play_sts2.runtime import (
    BattleOutcome,
    BattleResult,
    DecisionGenerationProfile,
    DecisionStep,
)
from play_sts2.scenario import (
    BattleScenario,
    BattleSnapshot,
    ModelInputSnapshot,
    ScenarioResetResult,
    ScenarioVerificationError,
)


class FakeBattleWorker:
    """返回确定性测试 arms，并记录每次入口基准。

    Args:
        rl (Any): 待验证的 RL 公共模块。
        worker_id (str): 当前 worker 标识。
        snapshot (BattleSnapshot): 所有 arms 应共享的入口快照。
    """

    def __init__(self, rl: Any, worker_id: str, snapshot: BattleSnapshot) -> None:
        """保存构造 rollout 所需的模块、标识和入口。

        Args:
            rl (Any): 待验证的 RL 公共模块。
            worker_id (str): 当前 worker 标识。
            snapshot (BattleSnapshot): 所有 arms 应共享的入口快照。

        Returns:
            None: 此方法只初始化测试 worker。
        """
        self._rl = rl
        self.worker_id = worker_id
        self._snapshot = snapshot
        self.calls: list[tuple[int, BattleSnapshot | None]] = []

    def collect_arm(
        self,
        scenario: BattleScenario,
        *,
        arm_index: int,
        expected_snapshot: BattleSnapshot | None,
    ) -> object:
        """返回由 arm 序号决定首动作和 reward 的完整 rollout。

        Args:
            scenario (BattleScenario): collector 传入的统一场景。
            arm_index (int): 当前 arm 在 group 内的编号。
            expected_snapshot (BattleSnapshot | None): 首条之后必须使用的入口基准。

        Raises:
            AssertionError: collector 传入了错误的场景。

        Returns:
            object: 可进入 group builder 的测试 rollout。
        """
        assert scenario.seed == "ABCDEF1234"
        self.calls.append((arm_index, expected_snapshot))
        action = "ACTION: end_turn" if arm_index % 2 == 0 else "ACTION: play_card 0"
        reward = 1.0 if arm_index % 2 == 0 else 3.0
        step = self._rl.BattleRolloutStep(
            index=0,
            before_state={"turn": 1},
            after_state={"turn": 2},
            messages=(),
            reply_text=action,
            action=action,
            token_ids=(101,),
            behavior_logprobs=(-0.1,),
            response_choices=(action,),
            finish_reason="stop",
        )
        return self._rl.BattleRollout(
            arm_index=arm_index,
            worker_id=self.worker_id,
            policy_version="policy-test",
            behavior_logprobs_mode="processed_logprobs",
            action_constraint_mode="vllm_structured_choice",
            generation_profile=_generation_profile(),
            entry_snapshot=self._snapshot,
            steps=(step,),
            outcome="cleared",
            final_state={"run": {"current_hp": 30}},
            reward=self._rl.BattleReward(
                scheme="test",
                total=reward,
                components=(self._rl.RewardComponent(name="terminal", value=reward),),
            ),
        )


def test_collector_builds_k8_group_across_two_workers() -> None:
    """首条建立入口基准，其余七条按 worker 分片并行收集。

    Raises:
        AssertionError: arm 数、worker 分配或 expected snapshot 传播错误。

    Returns:
        None: 此测试验证同入口多 worker collector 的主路径。
    """
    rl = importlib.import_module("play_sts2.training.rl")
    snapshot = _snapshot()
    workers = (
        FakeBattleWorker(rl, "worker-0", snapshot),
        FakeBattleWorker(rl, "worker-1", snapshot),
    )

    group = rl.BattleGroupCollector(workers, group_size=8).collect(
        _scenario(),
        group_id="battle-demo-001",
    )

    assert len(group.rollouts) == 8
    assert tuple(rollout.worker_id for rollout in group.rollouts) == (
        "worker-0",
        "worker-1",
        "worker-0",
        "worker-1",
        "worker-0",
        "worker-1",
        "worker-0",
        "worker-1",
    )
    assert workers[0].calls[0] == (0, None)
    assert all(
        expected == snapshot
        for worker in workers
        for arm_index, expected in worker.calls
        if arm_index != 0
    )


def test_game_worker_resets_scenario_and_projects_runtime_result() -> None:
    """真实 worker 组合 resetter、runner 与 rollout 投影边界。

    Raises:
        AssertionError: worker 没有透传入口基准或丢失 Runtime 行为事实。

    Returns:
        None: 此测试用轻量外部替身验证生产 worker 编排。
    """
    rl = importlib.import_module("play_sts2.training.rl")
    snapshot = _snapshot()
    entry_state = {
        "screen": "COMBAT",
        "in_combat": True,
        "available_actions": ["end_turn"],
        "turn": 1,
        "run": {"current_hp": 40, "max_hp": 70},
    }
    after_state = {"turn": 2, "run": {"current_hp": 35, "max_hp": 70}}
    step = DecisionStep(
        observation=Observation(
            layer=HarnessLayer.BATTLE,
            text="战斗状态",
            available_actions=("end_turn",),
        ),
        before_state=entry_state,
        messages=(ChatMessage(role="user", content="战斗状态"),),
        replies=(
            ModelReply(
                text="ACTION: end_turn",
                model="policy-test",
                finish_reason="stop",
                token_ids=(101,),
                behavior_logprobs=(-0.1,),
            ),
        ),
        retry_errors=(),
        action=HarnessAction(name="end_turn", parameters={}),
        action_result={"state": after_state, "stable": True},
        response_choices=("ACTION: end_turn",),
        generation_profile=_generation_profile(),
    )
    resetter = RecordingResetter(
        ScenarioResetResult(state=entry_state, snapshot=snapshot)
    )
    runner = RecordingRunner(
        BattleResult(
            outcome=BattleOutcome.CLEARED,
            steps=(step,),
            final_state={"run": {"current_hp": 30, "max_hp": 70}},
        )
    )
    worker = rl.GameBattleRolloutWorker(
        worker_id="worker-0",
        resetter=resetter,
        runner=runner,
        behavior_logprobs_mode="processed_logprobs",
    )

    rollout = worker.collect_arm(
        _scenario(),
        arm_index=3,
        expected_snapshot=snapshot,
    )

    assert resetter.expected_snapshots == [snapshot]
    assert runner.initial_states == [entry_state]
    assert rollout.arm_index == 3
    assert rollout.worker_id == "worker-0"
    assert rollout.policy_version == "policy-test"


def test_game_worker_classifies_network_failure_for_resampling() -> None:
    """游戏或推理网络错误转换为 collector 唯一会重采的故障类型。

    Returns:
        None: 此测试验证基础设施故障不会伪装成模型负样本。
    """
    rl = importlib.import_module("play_sts2.training.rl")
    request = httpx.Request("GET", "http://127.0.0.1:8080/state")
    resetter = FailingResetter(httpx.ConnectError("连接失败", request=request))
    worker = rl.GameBattleRolloutWorker(
        worker_id="worker-0",
        resetter=resetter,
        runner=object(),
        behavior_logprobs_mode="processed_logprobs",
    )

    with pytest.raises(rl.RolloutInfrastructureError, match="worker-0"):
        worker.collect_arm(
            _scenario(),
            arm_index=0,
            expected_snapshot=None,
        )


def test_game_worker_classifies_run_start_failure_for_resampling() -> None:
    """本地游戏没能开始新局属于可重建 arm 的环境故障。

    Returns:
        None: 此测试验证场景重置编排故障不会变成模型负样本。
    """
    rl = importlib.import_module("play_sts2.training.rl")
    resetter = FailingResetter(RunStartError("游戏不在主菜单"))
    worker = rl.GameBattleRolloutWorker(
        worker_id="worker-0",
        resetter=resetter,
        runner=object(),
        behavior_logprobs_mode="processed_logprobs",
    )

    with pytest.raises(rl.RolloutInfrastructureError, match="worker-0"):
        worker.collect_arm(
            _scenario(),
            arm_index=0,
            expected_snapshot=None,
        )


def test_game_worker_does_not_reclassify_truncated_model_output_as_infrastructure() -> (
    None
):
    """模型生成耗尽 token 属于 policy 失败，collector 不应自动重采美化结果。

    Returns:
        None: 此测试验证模型失败与基础设施失败的边界。
    """
    rl = importlib.import_module("play_sts2.training.rl")
    snapshot = _snapshot()
    entry_state = {"turn": 1, "run": {"current_hp": 40, "max_hp": 70}}
    resetter = RecordingResetter(
        ScenarioResetResult(state=entry_state, snapshot=snapshot)
    )
    runner = FailingRunner(
        InferenceGenerationTruncated(
            ModelReply(
                text="",
                reasoning="尚未完成",
                finish_reason="length",
                model="policy-test",
            )
        )
    )
    worker = rl.GameBattleRolloutWorker(
        worker_id="worker-0",
        resetter=resetter,
        runner=runner,
        behavior_logprobs_mode="processed_logprobs",
    )

    with pytest.raises(rl.RolloutModelError, match="worker-0"):
        worker.collect_arm(
            _scenario(),
            arm_index=0,
            expected_snapshot=None,
        )


def test_game_worker_does_not_resample_model_step_limit() -> None:
    """策略把战斗拖到动作上限属于模型失败，不能重采过滤退化轨迹。

    Returns:
        None: 此测试固定动作上限与环境故障的分类边界。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    rl = importlib.import_module("play_sts2.training.rl")
    snapshot = _snapshot()
    entry_state = {"turn": 1, "run": {"current_hp": 40, "max_hp": 70}}
    worker = rl.GameBattleRolloutWorker(
        worker_id="worker-0",
        resetter=RecordingResetter(
            ScenarioResetResult(state=entry_state, snapshot=snapshot)
        ),
        runner=FailingRunner(runtime.BattleStepLimitExceeded("动作数超限")),
        behavior_logprobs_mode="processed_logprobs",
    )

    with pytest.raises(rl.RolloutModelError, match="worker-0"):
        worker.collect_arm(
            _scenario(),
            arm_index=0,
            expected_snapshot=None,
        )


def test_game_worker_does_not_resample_nonretryable_http_status() -> None:
    """确定性的 4xx 请求错误立即暴露，不能重复发送同一错误请求。

    Returns:
        None: 此测试固定 HTTP 配置错误与瞬时网络故障的分类边界。
    """
    rl = importlib.import_module("play_sts2.training.rl")
    request = httpx.Request("POST", "http://127.0.0.1:8900/v1/chat/completions")
    response = httpx.Response(422, request=request)
    failure = httpx.HTTPStatusError(
        "无法处理请求",
        request=request,
        response=response,
    )
    worker = rl.GameBattleRolloutWorker(
        worker_id="worker-0",
        resetter=FailingResetter(failure),
        runner=object(),
        behavior_logprobs_mode="processed_logprobs",
    )

    with pytest.raises(httpx.HTTPStatusError) as captured:
        worker.collect_arm(
            _scenario(),
            arm_index=0,
            expected_snapshot=None,
        )

    assert captured.value is failure


def test_game_worker_classifies_retryable_http_status_for_resampling() -> None:
    """服务暂时不可用时允许 collector 重新构建当前 arm。

    Returns:
        None: 此测试验证 503 与确定性 4xx 使用不同故障分类。
    """
    rl = importlib.import_module("play_sts2.training.rl")
    request = httpx.Request("POST", "http://127.0.0.1:8900/v1/chat/completions")
    response = httpx.Response(503, request=request)
    worker = rl.GameBattleRolloutWorker(
        worker_id="worker-0",
        resetter=FailingResetter(
            httpx.HTTPStatusError(
                "服务暂时不可用",
                request=request,
                response=response,
            )
        ),
        runner=object(),
        behavior_logprobs_mode="processed_logprobs",
    )

    with pytest.raises(rl.RolloutInfrastructureError, match="暂时不可用"):
        worker.collect_arm(
            _scenario(),
            arm_index=0,
            expected_snapshot=None,
        )


def test_game_worker_rejects_group_when_entry_snapshot_differs() -> None:
    """跨 worker 入口不同必须立即拒绝整组，不能按网络故障重采。

    Returns:
        None: 此测试固定跨 worker 的严格同入口门槛。
    """
    rl = importlib.import_module("play_sts2.training.rl")
    resetter = FailingResetter(ScenarioVerificationError("初始战斗快照不一致"))
    worker = rl.GameBattleRolloutWorker(
        worker_id="worker-1",
        resetter=resetter,
        runner=object(),
        behavior_logprobs_mode="processed_logprobs",
    )

    with pytest.raises(rl.BattleGroupRejected, match="入口场景验证失败"):
        worker.collect_arm(
            _scenario(),
            arm_index=1,
            expected_snapshot=_snapshot(),
        )


def test_collector_retries_only_infrastructure_failure() -> None:
    """单条 arm 的瞬时基础设施故障按配置重试并保留原 arm 序号。

    Raises:
        AssertionError: collector 没有重试或改变了完成 group 的内容。

    Returns:
        None: 此测试固定 rollout collector 的故障重采规则。
    """
    rl = importlib.import_module("play_sts2.training.rl")
    snapshot = _snapshot()
    stable_worker = FakeBattleWorker(rl, "worker-0", snapshot)
    flaky_worker = FlakyBattleWorker(rl, "worker-1", snapshot)

    group = rl.BattleGroupCollector(
        (stable_worker, flaky_worker),
        group_size=2,
        infrastructure_attempts=2,
    ).collect(_scenario(), group_id="retry-demo")

    assert len(group.rollouts) == 2
    assert flaky_worker.attempts == [1, 1]


class FlakyBattleWorker(FakeBattleWorker):
    """首个调用失败、第二次返回正常 arm 的测试 worker。"""

    def __init__(self, rl: Any, worker_id: str, snapshot: BattleSnapshot) -> None:
        """保存依赖并初始化尝试记录。

        Args:
            rl (Any): 待验证的 RL 公共模块。
            worker_id (str): 当前 worker 标识。
            snapshot (BattleSnapshot): 所有 arms 应共享的入口快照。

        Returns:
            None: 此方法初始化一次性基础设施故障替身。
        """
        super().__init__(rl, worker_id, snapshot)
        self.attempts: list[int] = []

    def collect_arm(
        self,
        scenario: BattleScenario,
        *,
        arm_index: int,
        expected_snapshot: BattleSnapshot | None,
    ) -> object:
        """首次抛出基础设施故障，之后复用正常实现。

        Args:
            scenario (BattleScenario): collector 传入的统一场景。
            arm_index (int): 当前 arm 在 group 内的编号。
            expected_snapshot (BattleSnapshot | None): 入口快照基准。

        Raises:
            RolloutInfrastructureError: 当前 arm 的第一次采样。

        Returns:
            object: 第二次采样返回的正常 rollout。
        """
        self.attempts.append(arm_index)
        if len(self.attempts) == 1:
            raise self._rl.RolloutInfrastructureError("瞬时连接失败")
        return super().collect_arm(
            scenario,
            arm_index=arm_index,
            expected_snapshot=expected_snapshot,
        )


class RecordingResetter:
    """记录入口基准并返回预设 reset 结果。

    Args:
        result (ScenarioResetResult): 每次 reset 返回的场景结果。
    """

    def __init__(self, result: ScenarioResetResult) -> None:
        """保存结果并初始化入口基准记录。

        Args:
            result (ScenarioResetResult): 每次 reset 返回的场景结果。

        Returns:
            None: 此方法只初始化测试替身。
        """
        self._result = result
        self.expected_snapshots: list[BattleSnapshot | None] = []

    def reset(
        self,
        _scenario: BattleScenario,
        *,
        expected_snapshot: BattleSnapshot | None = None,
    ) -> ScenarioResetResult:
        """记录 collector 传入的入口基准并返回预设结果。

        Args:
            _scenario (BattleScenario): 当前测试不检查的场景。
            expected_snapshot (BattleSnapshot | None): collector 建立的入口基准。

        Returns:
            ScenarioResetResult: 预设的入口状态与快照。
        """
        self.expected_snapshots.append(expected_snapshot)
        return self._result


class FailingResetter:
    """在 reset 时抛出预设的外部故障。

    Args:
        error (Exception): 待抛出的网络或游戏异常。
    """

    def __init__(self, error: Exception) -> None:
        """保存下一次 reset 应抛出的异常。

        Args:
            error (Exception): 待抛出的异常。

        Returns:
            None: 此方法只初始化失败替身。
        """
        self._error = error

    def reset(
        self,
        _scenario: BattleScenario,
        *,
        expected_snapshot: BattleSnapshot | None = None,
    ) -> ScenarioResetResult:
        """抛出预设故障而不返回场景。

        Args:
            _scenario (BattleScenario): 当前测试不使用的场景。
            expected_snapshot (BattleSnapshot | None): 当前测试不使用的入口基准。

        Raises:
            Exception: 构造时传入的外部故障。

        Returns:
            ScenarioResetResult: 此测试路径不会返回。
        """
        del expected_snapshot
        raise self._error


class RecordingRunner:
    """记录初始状态并返回预设战斗结果。

    Args:
        result (BattleResult): 每次 run 返回的战斗结果。
    """

    def __init__(self, result: BattleResult) -> None:
        """保存结果并初始化状态记录。

        Args:
            result (BattleResult): 每次 run 返回的战斗结果。

        Returns:
            None: 此方法只初始化测试替身。
        """
        self._result = result
        self.initial_states: list[object] = []

    def run(self, initial_state: object) -> BattleResult:
        """记录传入状态并返回预设战斗结果。

        Args:
            initial_state (object): resetter 返回的入口状态。

        Returns:
            BattleResult: 预设的正常战斗结果。
        """
        self.initial_states.append(initial_state)
        return self._result


class FailingRunner:
    """在 run 时抛出预设的模型或外部故障。

    Args:
        error (Exception): 待抛出的异常。
    """

    def __init__(self, error: Exception) -> None:
        """保存下一次 run 应抛出的异常。

        Args:
            error (Exception): 待抛出的异常。

        Returns:
            None: 此方法只初始化失败替身。
        """
        self._error = error

    def run(self, _initial_state: object) -> BattleResult:
        """抛出预设异常而不返回战斗结果。

        Args:
            _initial_state (object): 当前测试不使用的入口状态。

        Raises:
            Exception: 构造时传入的故障。

        Returns:
            BattleResult: 此测试路径不会返回。
        """
        raise self._error


def _generation_profile() -> DecisionGenerationProfile:
    """返回 collector 测试共用的实际生成参数。

    Returns:
        DecisionGenerationProfile: 原提示 RL 采样配置。
    """
    return DecisionGenerationProfile(
        max_tokens=128,
        temperature=0.8,
        max_retries=0,
        thinking_enabled=False,
    )


def _scenario() -> BattleScenario:
    """创建 collector 使用的合法测试场景。

    Returns:
        BattleScenario: 固定 seed 的合成战斗配置。
    """
    return BattleScenario(
        character_id="DEFECT",
        seed="ABCDEF1234",
        floor=7,
        encounter_id="CULTISTS_NORMAL",
        deck=("ZAP",),
        relics=("CRACKED_CORE",),
        current_hp=40,
        max_hp=70,
    )


def _snapshot() -> BattleSnapshot:
    """创建跨 worker 共用的模型入口快照。

    Returns:
        BattleSnapshot: 最小确定性入口。
    """
    return BattleSnapshot(
        turn=1,
        enemies=(),
        hand=(),
        model_input=ModelInputSnapshot(
            system="战斗系统",
            user="战斗状态",
            available_actions=("play_card", "end_turn"),
        ),
    )
