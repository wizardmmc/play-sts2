"""从学生战斗 rollout 构造可审计的 CombatSolver DAgger 标签。"""

import json
import math
import random
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from ...client import GameClient, Health, SolverSuggestion
from ...harness import legal_action_lines
from ...inference import ChatMessage, ModelReply
from ...runtime import BattleRunner
from ...scenario import BattleScenario, ScenarioResetResult


class DaggerContractError(ValueError):
    """表示学生状态、教师动作或标签文件不满足 DAgger 契约。"""


class DaggerScenarioResetter(Protocol):
    """声明 DAgger 重放需要的最小场景重置接口。"""

    def reset(self, scenario: BattleScenario) -> ScenarioResetResult:
        """重置并返回一个可决策战斗入口。

        Args:
            scenario (BattleScenario): 来源 rollout 使用的确定性场景。

        Returns:
            ScenarioResetResult: 可交给 ``BattleRunner`` 的入口状态。
        """
        ...


@dataclass(frozen=True, slots=True)
class DaggerCandidate:
    """保存一条只在标注进程内使用的学生决策状态。

    Args:
        label_id (str): 由 group、arm 和 step 组成的稳定来源标识。
        group_id (str): 来源 rollout group。
        arm_index (int): 来源 arm 序号。
        step_index (int): 来源战斗步骤序号。
        outcome (str): 来源学生战斗的正常离场结果。
        student_policy_version (str): 产生该状态与动作的冻结学生 policy。
        student_action (str): 学生实际执行的规范动作。
        legal_actions (tuple[str, ...]): 学生请求实际使用的完整动作候选。
        messages (tuple[ChatMessage, ...]): 学生当时实际看到的 system/user 输入。
        before_state (Mapping[str, Any]): 仅用于受控重放和 revision 核对的完整状态；
            不允许写入最终标签。
        behavior_logprobs (tuple[float, ...]): 仅用于候选不确定性排序的学生概率。
        uncertainty (float): assistant token 平均负 log-prob。
        reason (str): ``death_tail``、``uncertainty`` 或 ``ordinary``。
    """

    label_id: str
    group_id: str
    arm_index: int
    step_index: int
    outcome: str
    student_policy_version: str
    student_action: str
    legal_actions: tuple[str, ...]
    messages: tuple[ChatMessage, ...]
    before_state: Mapping[str, Any]
    behavior_logprobs: tuple[float, ...]
    uncertainty: float
    reason: str = "unselected"


@dataclass(frozen=True, slots=True)
class DaggerRolloutSource:
    """保存一个 rollout group 的场景和全部可标注学生步骤。

    Args:
        group_id (str): 来源 group 标识。
        scenario (BattleScenario): 重放学生前缀所需的确定性战斗场景。
        policy_version (str): group 冻结学生 policy。
        candidates (tuple[DaggerCandidate, ...]): 按 arm、step 排列的全部状态。
    """

    group_id: str
    scenario: BattleScenario
    policy_version: str
    candidates: tuple[DaggerCandidate, ...]


@dataclass(frozen=True, slots=True)
class DaggerLabel:
    """保存一条不含 Solver 隐藏信息的监督标签。

    Args:
        label_id (str): 与学生来源一一对应的标签标识。
        training_role (str): 固定为 ``dagger_label``。
        group_id (str): 来源学生 rollout group。
        arm_index (int): 来源 arm 序号。
        step_index (int): 来源步骤序号。
        selection_reason (str): 该状态进入标注预算的原因。
        student_policy_version (str): 来源学生 policy。
        student_action (str): 学生实际动作。
        teacher_action (str): CombatSolver 建议的规范动作。
        agrees (bool): 学生与教师动作是否完全一致。
        legal_actions (tuple[str, ...]): 当时玩家可见状态的完整合法动作域。
        messages (tuple[dict[str, str], ...]): system/user 加教师 assistant 目标。
        game_version (str): 标注游戏版本。
        mod_version (str): 提供状态和端点的 Agent Mod 版本。
        protocol_version (str): Agent Mod HTTP 协议版本。
        solver_name (str): 固定教师名称。
        solver_version (str): 实际加载的 CombatSolver 版本。
        harness_version (str): 构造学生消息的 Harness 包版本。
        label_status (str): 固定为 ``labeled``。
    """

    label_id: str
    training_role: str
    group_id: str
    arm_index: int
    step_index: int
    selection_reason: str
    student_policy_version: str
    student_action: str
    teacher_action: str
    agrees: bool
    legal_actions: tuple[str, ...]
    messages: tuple[dict[str, str], ...]
    game_version: str
    mod_version: str
    protocol_version: str
    solver_name: str
    solver_version: str
    harness_version: str
    label_status: str


class DaggerReplayLabeler:
    """通过确定性前缀重放在 live 学生状态上请求 Solver 标签。"""

    def __init__(
        self,
        game: GameClient,
        *,
        resetter: DaggerScenarioResetter,
        search_timeout: float = 135.0,
        harness_version: str,
    ) -> None:
        """保存教师游戏、场景重置器与标注版本。

        Args:
            game (GameClient): 加载 Agent、RitsuLib 与 CombatSolver 的游戏客户端。
            resetter (DaggerScenarioResetter): 在教师游戏中恢复学生场景的重置器。
            search_timeout (float): 单次 Solver 搜索最长秒数。
            harness_version (str): 当前学生消息投影版本。

        Raises:
            ValueError: 搜索超时或 Harness 版本无效。
        """
        if search_timeout <= 0:
            raise ValueError("search_timeout 必须大于零")
        if not harness_version.strip():
            raise ValueError("harness_version 不能为空")
        self._game = game
        self._resetter = resetter
        self._search_timeout = search_timeout
        self._harness_version = harness_version

    def label(
        self,
        source: DaggerRolloutSource,
        candidates: Sequence[DaggerCandidate],
    ) -> tuple[DaggerLabel, ...]:
        """逐个重置、重放学生前缀并在目标动作前取得教师建议。

        Args:
            source (DaggerRolloutSource): 来源 group 与完整动作序列。
            candidates (Sequence[DaggerCandidate]): 已分配预算的目标状态。

        Raises:
            DaggerContractError: 候选不属于来源 group、重放消息或动作域漂移。
            BattleRunError: 学生前缀无法在教师游戏中重放。

        Returns:
            tuple[DaggerLabel, ...]: 与候选顺序一致的成功教师标签。
        """
        health = self._game.health()
        labels: list[DaggerLabel] = []
        source_ids = {candidate.label_id for candidate in source.candidates}
        for candidate in candidates:
            if candidate.label_id not in source_ids:
                raise DaggerContractError("DAgger 候选不属于当前 rollout group")
            prefix = tuple(
                step
                for step in source.candidates
                if step.arm_index == candidate.arm_index
                and step.step_index <= candidate.step_index
            )
            if not prefix or prefix[-1].step_index != candidate.step_index:
                raise DaggerContractError("DAgger 候选缺少完整学生动作前缀")
            replay = _DaggerReplayProvider(
                self._game,
                prefix=prefix,
                target=candidate,
                health=health,
                search_timeout=self._search_timeout,
                harness_version=self._harness_version,
            )
            reset = self._resetter.reset(source.scenario)
            try:
                BattleRunner(
                    self._game,
                    replay,
                    max_retries=0,
                    max_conflict_retries=0,
                ).run(reset.state)
            except _ReplayComplete:
                pass
            if replay.label is None:
                raise DaggerContractError("学生前缀结束前没有抵达 DAgger 目标状态")
            labels.append(replay.label)
        return tuple(labels)


class _ReplayComplete(RuntimeError):
    """表示目标状态已经标注，重放无需继续推进。"""


class _DaggerReplayProvider:
    """向 BattleRunner 回放学生动作，并在目标位置插入只读教师查询。"""

    def __init__(
        self,
        game: GameClient,
        *,
        prefix: Sequence[DaggerCandidate],
        target: DaggerCandidate,
        health: Health,
        search_timeout: float,
        harness_version: str,
    ) -> None:
        """保存重放动作和目标标注上下文。

        Args:
            game (GameClient): 教师游戏客户端。
            prefix (Sequence[DaggerCandidate]): 从入口到目标的学生动作。
            target (DaggerCandidate): 需要 Solver 建议的步骤。
            health (Health): 当前游戏和 Mod 版本。
            search_timeout (float): Solver 搜索超时。
            harness_version (str): Harness 版本。
        """
        self._game = game
        self._prefix = tuple(prefix)
        self._target = target
        self._health = health
        self._search_timeout = search_timeout
        self._harness_version = harness_version
        self._position = 0
        self.label: DaggerLabel | None = None

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        response_choices: Sequence[str] | None = None,
    ) -> ModelReply:
        """返回下一条学生动作，并在目标动作前同步请求 Solver。

        Args:
            messages (Sequence[ChatMessage]): 当前重放状态生成的真实消息。
            max_tokens (int): BattleRunner 的生成 token 参数；重放不使用。
            temperature (float): BattleRunner 的采样参数；重放不使用。
            response_choices (Sequence[str] | None): 可选动作约束；重放不使用。

        Raises:
            _ReplayComplete: 已完成目标标签且 Runner 再次请求动作。
            DaggerContractError: 可见消息、动作域或 live revision 发生漂移。

        Returns:
            ModelReply: 当前来源学生实际执行的动作。
        """
        del max_tokens, temperature, response_choices
        if self._position >= len(self._prefix):
            raise _ReplayComplete
        expected = self._prefix[self._position]
        actual_messages = tuple(messages)
        if actual_messages != expected.messages:
            raise DaggerContractError("DAgger 重放的学生可见消息发生漂移")
        if expected.label_id == self._target.label_id:
            live_state = self._game.state()
            live_actions = legal_action_lines(live_state)
            if live_actions != expected.legal_actions:
                raise DaggerContractError("DAgger 重放的学生合法动作域发生漂移")
            revision = live_state.get("state_revision")
            if isinstance(revision, bool) or not isinstance(revision, int):
                raise DaggerContractError("DAgger live 状态缺少 revision")
            suggestion = self._game.solver_suggestion(
                expected_state_revision=revision,
                timeout=self._search_timeout,
            )
            self.label = build_dagger_label(
                replace(
                    expected,
                    before_state=live_state,
                    reason=self._target.reason,
                ),
                suggestion,
                health=self._health,
                harness_version=self._harness_version,
            )
        self._position += 1
        return ModelReply(
            text=expected.student_action,
            model=expected.student_policy_version,
            finish_reason="replay",
        )


_DAGGER_LABEL_FIELDS = frozenset(
    {
        "label_id",
        "training_role",
        "group_id",
        "arm_index",
        "step_index",
        "selection_reason",
        "student_policy_version",
        "student_action",
        "teacher_action",
        "agrees",
        "legal_actions",
        "messages",
        "game_version",
        "mod_version",
        "protocol_version",
        "solver_name",
        "solver_version",
        "harness_version",
        "label_status",
    }
)


def write_dagger_labels(
    path: Path,
    labels: Sequence[DaggerLabel],
    *,
    selection_seed: int,
    selection_budget: int,
    unsupported_reasons: Mapping[str, int],
) -> Path:
    """以 JSONL 原子写出只含训练可见事实的 DAgger 标签。

    Args:
        path (Path): 目标标签文件。
        labels (Sequence[DaggerLabel]): 已通过 live Solver 校验的标签。
        selection_seed (int): 普通状态抽样使用的随机种子。
        selection_budget (int): 当前批请求的最大标签数。
        unsupported_reasons (Mapping[str, int]): 未进入候选池的状态原因计数。

    Raises:
        DaggerContractError: 标签为空或身份重复。
        OSError: 文件无法写入或替换。

    Returns:
        Path: 实际写入的标签路径。
    """
    ordered = tuple(labels)
    if not ordered:
        raise DaggerContractError("DAgger 标签文件不能为空")
    if len({label.label_id for label in ordered}) != len(ordered):
        raise DaggerContractError("DAgger 标签身份重复")
    policy_versions = {label.student_policy_version for label in ordered}
    if len(policy_versions) != 1:
        raise DaggerContractError("DAgger 标签批次混入不同 student policy")
    if selection_budget < len(ordered):
        raise DaggerContractError("DAgger 标签数超过选择预算")
    if any(
        not isinstance(reason, str)
        or not reason
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for reason, count in unsupported_reasons.items()
    ):
        raise DaggerContractError("DAgger unsupported reason 计数无效")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(asdict(label), ensure_ascii=False, allow_nan=False) + "\n"
            for label in ordered
        ),
        encoding="utf-8",
    )
    temporary.replace(output)
    manifest = {
        "format": "dagger_label_batch",
        "labels_file": output.name,
        "labels": len(ordered),
        "student_policy_version": next(iter(policy_versions)),
        "selection": {
            "budget": selection_budget,
            "seed": selection_seed,
            "unsupported_reasons": dict(sorted(unsupported_reasons.items())),
        },
        "selection_reasons": dict(
            sorted(Counter(label.selection_reason for label in ordered).items())
        ),
        "game_versions": dict(
            sorted(Counter(label.game_version for label in ordered).items())
        ),
        "mod_versions": dict(
            sorted(Counter(label.mod_version for label in ordered).items())
        ),
        "solver_versions": dict(
            sorted(Counter(label.solver_version for label in ordered).items())
        ),
        "harness_versions": dict(
            sorted(Counter(label.harness_version for label in ordered).items())
        ),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    manifest_temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_temporary.replace(manifest_path)
    return output


def load_dagger_sft_rows(
    path: Path,
    *,
    expected_policy_version: str,
) -> list[dict[str, Any]]:
    """读取旁路标签并转换为只进入训练分卷的行为监督行。

    Args:
        path (Path): 单个 JSONL 文件或递归包含标签文件的目录。
        expected_policy_version (str): 本次构建唯一允许的学生父 policy。

    Raises:
        DaggerContractError: 文件为空、含额外隐藏字段或标签语义不一致。
        OSError: 标签路径无法读取。

    Returns:
        list[dict[str, Any]]: 可由现有 SFT builder 聚合的 battle 行。
    """
    if not expected_policy_version.strip():
        raise DaggerContractError("DAgger expected student policy 不能为空")
    source = Path(path)
    files = sorted(source.rglob("*.jsonl")) if source.is_dir() else [source]
    if not files or any(not file.is_file() for file in files):
        raise DaggerContractError(f"DAgger 标签路径不可用: {source}")
    rows: list[dict[str, Any]] = []
    for file in files:
        for line_number, line in enumerate(
            file.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DaggerContractError(
                    f"DAgger 标签不是合法 JSON: {file}:{line_number}"
                ) from exc
            rows.append(
                _dagger_sft_row(
                    payload,
                    file,
                    line_number,
                    expected_policy_version,
                )
            )
    if not rows:
        raise DaggerContractError(f"DAgger 标签路径没有样本: {source}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise DaggerContractError("DAgger 标签存在重复 label_id")
    return rows


def _dagger_sft_row(
    value: object,
    path: Path,
    line_number: int,
    expected_policy_version: str,
) -> dict[str, Any]:
    """校验一行标签并投影为现有行为 SFT 结构。

    Args:
        value (object): JSON 解码后的标签。
        path (Path): 来源文件，用于错误定位。
        line_number (int): 来源行号。
        expected_policy_version (str): 本次构建允许的学生 policy。

    Raises:
        DaggerContractError: 字段集合、动作域、消息或状态无效。

    Returns:
        dict[str, Any]: 不含完整状态和 Solver 隐藏诊断的训练行。
    """
    location = f"{path}:{line_number}"
    if not isinstance(value, Mapping) or set(value) != _DAGGER_LABEL_FIELDS:
        raise DaggerContractError(f"DAgger 标签字段集合无效: {location}")
    if (
        value.get("training_role") != "dagger_label"
        or value.get("label_status") != "labeled"
    ):
        raise DaggerContractError(f"DAgger 标签角色或状态无效: {location}")
    if value.get("solver_name") != "CombatSolver":
        raise DaggerContractError(f"DAgger 标签教师身份无效: {location}")
    label_id = _text(value, "label_id")
    group_id = _text(value, "group_id")
    if any(token in group_id for token in ("/", "..")):
        raise DaggerContractError(f"DAgger group_id 不能用于安全路径: {location}")
    arm_index = _integer(value, "arm_index")
    _integer(value, "step_index")
    teacher_action = _text(value, "teacher_action")
    student_action = _text(value, "student_action")
    student_policy = _text(value, "student_policy_version")
    if student_policy != expected_policy_version:
        raise DaggerContractError(
            "DAgger student policy 与本次构建不一致: "
            f"expected={expected_policy_version}, actual={student_policy}"
        )
    selection_reason = _text(value, "selection_reason")
    if selection_reason not in {"death_tail", "uncertainty", "ordinary"}:
        raise DaggerContractError(f"DAgger selection_reason 无效: {location}")
    legal = value.get("legal_actions")
    if (
        not isinstance(legal, list)
        or not legal
        or any(not isinstance(action, str) or not action for action in legal)
    ):
        raise DaggerContractError(f"DAgger 合法动作域无效: {location}")
    if teacher_action not in legal or student_action not in legal:
        raise DaggerContractError(f"DAgger 动作不在合法域: {location}")
    agrees = value.get("agrees")
    if not isinstance(agrees, bool) or agrees != (teacher_action == student_action):
        raise DaggerContractError(f"DAgger agrees 与动作不一致: {location}")
    messages = value.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise DaggerContractError(
            f"DAgger messages 必须为 system/user/assistant: {location}"
        )
    parsed_messages = [asdict(_message(message)) for message in messages]
    if [message["role"] for message in parsed_messages] != [
        "system",
        "user",
        "assistant",
    ] or parsed_messages[-1]["content"] != teacher_action:
        raise DaggerContractError(f"DAgger messages 与教师动作不一致: {location}")
    tokens = teacher_action.split()
    if len(tokens) < 2 or tokens[0] != "ACTION:":
        raise DaggerContractError(f"DAgger 教师动作不是规范 ACTION: {location}")
    action = tokens[1]
    available_actions = list(dict.fromkeys(item.split()[1] for item in legal))
    return {
        "sample_id": f"dagger/{label_id}",
        "source": "human_play",
        "run_id": f"dagger-{group_id}",
        "battle_key": f"arm-{arm_index}",
        "layer": "battle",
        "screen": "COMBAT",
        "action": action,
        "available_actions": available_actions,
        "legal_actions": list(legal),
        "tags": sorted(
            {
                action,
                "dagger",
                f"dagger:{selection_reason}",
                "dagger:agree" if agrees else "dagger:disagree",
            }
        ),
        "messages": parsed_messages,
        "training_role": "dagger_label",
        "behavior_origin": "dagger",
        "action_source": "combat_solver",
        "student_policy_version": student_policy,
        "student_action": student_action,
        "teacher_action": teacher_action,
        "game_version": _text(value, "game_version"),
        "mod_version": _text(value, "mod_version"),
        "protocol_version": _text(value, "protocol_version"),
        "solver_version": _text(value, "solver_version"),
        "harness_version": _text(value, "harness_version"),
        "recording_complete": True,
        "recording_gaps": [],
        "run_victory": None,
    }


def load_dagger_rollout_source(path: Path) -> DaggerRolloutSource:
    """读取第一阶段 group 并投影为尚未选择的学生状态。

    Args:
        path (Path): 完整 battle rollout group JSON。

    Raises:
        DaggerContractError: group、场景、消息或行为概率字段无效。
        OSError: 文件无法读取。
        json.JSONDecodeError: 文件不是合法 JSON。

    Returns:
        DaggerRolloutSource: 可执行分层候选抽样的学生状态集合。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise DaggerContractError("DAgger rollout group 必须是 JSON 对象")
    group_id = _text(payload, "group_id")
    policy_version = _text(payload, "policy_version")
    scenario = _scenario(payload.get("scenario"))
    raw_rollouts = payload.get("rollouts")
    if not isinstance(raw_rollouts, list) or not raw_rollouts:
        raise DaggerContractError("DAgger rollout group 没有学生 arms")

    candidates: list[DaggerCandidate] = []
    for raw_rollout in raw_rollouts:
        if not isinstance(raw_rollout, Mapping):
            raise DaggerContractError("DAgger rollout arm 必须是对象")
        arm_index = _integer(raw_rollout, "arm_index")
        rollout_policy = _text(raw_rollout, "policy_version")
        if rollout_policy != policy_version:
            raise DaggerContractError("DAgger rollout group 混入不同学生 policy")
        outcome = _text(raw_rollout, "outcome")
        raw_steps = raw_rollout.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise DaggerContractError("DAgger rollout arm 没有学生步骤")
        for raw_step in raw_steps:
            candidates.append(
                _candidate(
                    group_id,
                    arm_index,
                    outcome,
                    policy_version,
                    raw_step,
                )
            )
    ordered = tuple(
        sorted(candidates, key=lambda item: (item.arm_index, item.step_index))
    )
    if len({candidate.label_id for candidate in ordered}) != len(ordered):
        raise DaggerContractError("DAgger rollout 存在重复 group/arm/step")
    return DaggerRolloutSource(
        group_id=group_id,
        scenario=scenario,
        policy_version=policy_version,
        candidates=ordered,
    )


def select_dagger_candidates(
    source: DaggerRolloutSource,
    *,
    max_labels: int,
    seed: int,
) -> tuple[DaggerCandidate, ...]:
    """在固定预算内平衡死亡尾部、不确定状态和普通状态。

    Args:
        source (DaggerRolloutSource): 一个完整学生 group 的全部决策步骤。
        max_labels (int): 本批最多请求的教师标签数。
        seed (int): 普通状态抽样使用的确定性随机种子。

    Raises:
        ValueError: 标签预算小于一。

    Returns:
        tuple[DaggerCandidate, ...]: 去重且带选择原因的候选状态。
    """
    if max_labels < 1:
        raise ValueError("max_labels 必须大于零")
    eligible = tuple(
        candidate
        for candidate in source.candidates
        if _dagger_unsupported_reason(candidate) is None
    )
    if not eligible:
        raise DaggerContractError("DAgger rollout 没有 Solver 支持的标准出牌状态")
    budget = min(max_labels, len(eligible))
    by_arm: dict[int, list[DaggerCandidate]] = {}
    for candidate in eligible:
        by_arm.setdefault(candidate.arm_index, []).append(candidate)

    death_pool = [
        arm[-1] for arm in by_arm.values() if arm and arm[-1].outcome == "died"
    ]
    death_count = min(len(death_pool), max(1, budget // 4)) if death_pool else 0
    selected = [
        replace(candidate, reason="death_tail")
        for candidate in sorted(
            death_pool,
            key=lambda item: (-item.uncertainty, item.arm_index),
        )[:death_count]
    ]
    selected_ids = {candidate.label_id for candidate in selected}

    remaining = [
        candidate for candidate in eligible if candidate.label_id not in selected_ids
    ]
    ordinary_count = 1 if remaining and len(selected) < budget else 0
    uncertainty_count = min(len(remaining), budget - len(selected) - ordinary_count)
    uncertainty = sorted(
        remaining,
        key=lambda item: (-item.uncertainty, item.arm_index, item.step_index),
    )[:uncertainty_count]
    selected.extend(
        replace(candidate, reason="uncertainty") for candidate in uncertainty
    )
    selected_ids.update(candidate.label_id for candidate in uncertainty)

    ordinary_pool = [
        candidate for candidate in eligible if candidate.label_id not in selected_ids
    ]
    random.Random(seed).shuffle(ordinary_pool)
    selected.extend(
        replace(candidate, reason="ordinary")
        for candidate in ordinary_pool[: budget - len(selected)]
    )
    return tuple(selected)


def summarize_dagger_unsupported(source: DaggerRolloutSource) -> dict[str, int]:
    """统计当前 Solver 建议端点不能处理的学生状态。

    Args:
        source (DaggerRolloutSource): 一个学生 rollout group 的全部步骤。

    Returns:
        dict[str, int]: 按稳定原因名称排序的跳过状态计数。
    """
    return dict(
        sorted(
            Counter(
                reason
                for candidate in source.candidates
                if (reason := _dagger_unsupported_reason(candidate)) is not None
            ).items()
        )
    )


def _dagger_unsupported_reason(candidate: DaggerCandidate) -> str | None:
    """判断一个学生状态是否属于 Solver 的标准出牌搜索域。

    Args:
        candidate (DaggerCandidate): 待检查的学生状态。

    Returns:
        str | None: 不支持原因；标准玩家出牌阶段返回 ``None``。
    """
    state = candidate.before_state
    if state.get("screen") != "COMBAT" or state.get("in_combat") is not True:
        return "unsupported_screen"
    actions = state.get("available_actions")
    if not isinstance(actions, list) or "end_turn" not in actions:
        return "unsupported_action_window"
    return None


def build_dagger_label(
    candidate: DaggerCandidate,
    suggestion: SolverSuggestion,
    *,
    health: Health,
    harness_version: str,
) -> DaggerLabel:
    """把一个教师建议投影为不含隐藏状态的监督标签。

    Args:
        candidate (DaggerCandidate): 已选中的学生决策状态。
        suggestion (SolverSuggestion): 同一 live revision 上的 Solver 动作。
        health (Health): 当前 Agent Mod 与游戏版本。
        harness_version (str): 生成学生消息的 Harness 包版本。

    Raises:
        DaggerContractError: revision、教师动作、消息或版本不满足契约。

    Returns:
        DaggerLabel: 可安全落盘并加入独立监督 loss 的标签。
    """
    revision = candidate.before_state.get("state_revision")
    if suggestion.state_revision != revision:
        raise DaggerContractError("Solver 建议与学生状态 revision 不一致")
    if suggestion.action not in candidate.legal_actions:
        raise DaggerContractError("Solver 建议不在学生同一合法动作域")
    if candidate.student_action not in candidate.legal_actions:
        raise DaggerContractError("学生动作不在其落盘合法动作域")
    if not harness_version.strip() or not suggestion.solver_version.strip():
        raise DaggerContractError("DAgger 标签缺少 Harness 或 Solver 版本")
    if tuple(message.role for message in candidate.messages) != ("system", "user"):
        raise DaggerContractError("DAgger 标签要求 stateless system/user 输入")
    messages = tuple(
        {"role": message.role, "content": message.content}
        for message in candidate.messages
    ) + ({"role": "assistant", "content": suggestion.action},)
    return DaggerLabel(
        label_id=candidate.label_id,
        training_role="dagger_label",
        group_id=candidate.group_id,
        arm_index=candidate.arm_index,
        step_index=candidate.step_index,
        selection_reason=candidate.reason,
        student_policy_version=candidate.student_policy_version,
        student_action=candidate.student_action,
        teacher_action=suggestion.action,
        agrees=suggestion.action == candidate.student_action,
        legal_actions=candidate.legal_actions,
        messages=messages,
        game_version=health.game_version,
        mod_version=health.mod_version,
        protocol_version=health.protocol_version,
        solver_name="CombatSolver",
        solver_version=suggestion.solver_version,
        harness_version=harness_version,
        label_status="labeled",
    )


def _candidate(
    group_id: str,
    arm_index: int,
    outcome: str,
    policy_version: str,
    value: object,
) -> DaggerCandidate:
    """解析 rollout JSON 中的一条学生步骤。

    Args:
        group_id (str): 来源 group。
        arm_index (int): 来源 arm。
        outcome (str): 来源战斗结果。
        policy_version (str): 来源学生 policy。
        value (object): 尚未校验的 step 对象。

    Raises:
        DaggerContractError: step 字段缺失、非有限或不自洽。

    Returns:
        DaggerCandidate: 尚未分配抽样原因的内部候选。
    """
    if not isinstance(value, Mapping):
        raise DaggerContractError("DAgger rollout step 必须是对象")
    step_index = _integer(value, "index")
    action = _text(value, "action")
    raw_state = value.get("before_state")
    if not isinstance(raw_state, Mapping):
        raise DaggerContractError("DAgger rollout step 缺少 before_state")
    raw_messages = value.get("messages")
    if not isinstance(raw_messages, list):
        raise DaggerContractError("DAgger rollout step 缺少 messages")
    messages = tuple(_message(message) for message in raw_messages)
    raw_actions = value.get("response_choices")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise DaggerContractError("DAgger rollout step 缺少合法动作域")
    legal_actions = tuple(str(item) for item in raw_actions)
    if any(not isinstance(item, str) or not item for item in raw_actions):
        raise DaggerContractError("DAgger rollout step 合法动作域无效")
    raw_logprobs = value.get("behavior_logprobs")
    if not isinstance(raw_logprobs, list) or not raw_logprobs:
        raise DaggerContractError("DAgger rollout step 缺少行为概率")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in raw_logprobs
    ):
        raise DaggerContractError("DAgger rollout step 行为概率无效")
    behavior_logprobs = tuple(float(item) for item in raw_logprobs)
    return DaggerCandidate(
        label_id=f"{group_id}:{arm_index}:{step_index}",
        group_id=group_id,
        arm_index=arm_index,
        step_index=step_index,
        outcome=outcome,
        student_policy_version=policy_version,
        student_action=action,
        legal_actions=legal_actions,
        messages=messages,
        before_state=dict(raw_state),
        behavior_logprobs=behavior_logprobs,
        uncertainty=-statistics.fmean(behavior_logprobs),
    )


def _scenario(value: object) -> BattleScenario:
    """把 group 中的场景对象转换为稳定模型。

    Args:
        value (object): 尚未校验的场景对象。

    Raises:
        DaggerContractError: 场景字段无法构成 ``BattleScenario``。

    Returns:
        BattleScenario: 用于重放的确定性场景。
    """
    if not isinstance(value, Mapping):
        raise DaggerContractError("DAgger rollout group 缺少 scenario")
    fields = dict(value)
    for name in ("deck", "relics", "potions"):
        raw = fields.get(name, [])
        if not isinstance(raw, list):
            raise DaggerContractError(f"DAgger scenario 字段必须是数组: {name}")
        fields[name] = tuple(raw)
    try:
        return BattleScenario(**fields)
    except (TypeError, ValueError) as exc:
        raise DaggerContractError("DAgger rollout group scenario 无效") from exc


def _message(value: object) -> ChatMessage:
    """解析一条学生实际消息。

    Args:
        value (object): 尚未校验的消息对象。

    Raises:
        DaggerContractError: 角色或正文无效。

    Returns:
        ChatMessage: 不可变学生消息。
    """
    if not isinstance(value, Mapping):
        raise DaggerContractError("DAgger rollout message 必须是对象")
    role = value.get("role")
    content = value.get("content")
    if role not in {"system", "user", "assistant"} or not isinstance(content, str):
        raise DaggerContractError("DAgger rollout message 无效")
    return ChatMessage(role=role, content=content)


def _text(value: Mapping[object, object], field: str) -> str:
    """读取 DAgger JSON 中的非空字符串。

    Args:
        value (Mapping[object, object]): 当前 JSON 对象。
        field (str): 字段名。

    Raises:
        DaggerContractError: 字段缺失或为空。

    Returns:
        str: 校验后的字符串。
    """
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise DaggerContractError(f"DAgger 字段必须是非空字符串: {field}")
    return item


def _integer(value: Mapping[object, object], field: str) -> int:
    """读取 DAgger JSON 中的非负整数。

    Args:
        value (Mapping[object, object]): 当前 JSON 对象。
        field (str): 字段名。

    Raises:
        DaggerContractError: 字段不是非负整数。

    Returns:
        int: 校验后的整数。
    """
    item = value.get(field)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise DaggerContractError(f"DAgger 字段必须是非负整数: {field}")
    return item
