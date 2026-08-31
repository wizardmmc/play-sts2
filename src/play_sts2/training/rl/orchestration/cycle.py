"""记录并持久化阶段七 policy pair 的固定采样、更新和验证顺序。"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any


class CyclePhase(str, Enum):
    """表示一轮完整训练当前已经完成的边界。"""

    CREATED = "created"
    BACKBONE_COLLECTED = "backbone_collected"
    TREE_COLLECTED = "tree_collected"
    STRATEGY_UPDATED = "strategy_updated"
    BATTLE_UPDATED = "battle_updated"
    PROVISIONAL = "provisional"
    VALIDATED = "validated"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"


_NEXT_PHASES = {
    CyclePhase.CREATED: {CyclePhase.BACKBONE_COLLECTED},
    CyclePhase.BACKBONE_COLLECTED: {CyclePhase.TREE_COLLECTED},
    CyclePhase.TREE_COLLECTED: {CyclePhase.STRATEGY_UPDATED},
    CyclePhase.STRATEGY_UPDATED: {CyclePhase.BATTLE_UPDATED},
    CyclePhase.BATTLE_UPDATED: {CyclePhase.VALIDATED, CyclePhase.PROVISIONAL},
    CyclePhase.VALIDATED: {
        CyclePhase.PROVISIONAL,
        CyclePhase.PROMOTED,
        CyclePhase.ROLLED_BACK,
    },
}


@dataclass(frozen=True, slots=True)
class CycleJournal:
    """保存一轮 Stage 7 的可读 policy pair 与当前阶段。

    Args:
        cycle_id (str): 短可读轮次名。
        phase (CyclePhase): 当前完成边界。
        parent_strategy_policy (str): 本轮采样使用的战略 residual。
        parent_battle_policy (str): 本轮采样使用的战斗 residual。
        strategy_policy (str): 当前候选战略 residual。
        battle_policy (str): 当前候选战斗 residual。
        train_seed (str): 本轮 active seed。
        last_promoted_strategy_policy (str): 当前血缘最后晋升战略 residual。
        last_promoted_battle_policy (str): 当前血缘最后晋升战斗 residual。
        promotion_receipt (dict[str, Any] | None): 原子 promotion 的硬门控收据。
    """

    cycle_id: str
    phase: CyclePhase
    parent_strategy_policy: str
    parent_battle_policy: str
    strategy_policy: str
    battle_policy: str
    train_seed: str
    last_promoted_strategy_policy: str
    last_promoted_battle_policy: str
    promotion_receipt: dict[str, Any] | None = None

    @classmethod
    def new(
        cls,
        *,
        cycle_id: str,
        strategy_policy: str,
        battle_policy: str,
        train_seed: str,
        last_promoted_strategy_policy: str,
        last_promoted_battle_policy: str,
    ) -> "CycleJournal":
        """创建尚未采样的轮次记录。

        Args:
            cycle_id (str): 短可读轮次名。
            strategy_policy (str): 冻结父战略 residual。
            battle_policy (str): 冻结父战斗 residual。
            train_seed (str): 本轮 active seed。
            last_promoted_strategy_policy (str): 血缘中最后正式战略 policy。
            last_promoted_battle_policy (str): 血缘中最后正式战斗 policy。

        Raises:
            ValueError: 任一可读身份为空。

        Returns:
            CycleJournal: 初始轮次记录。
        """
        if not all((cycle_id, strategy_policy, battle_policy, train_seed)):
            raise ValueError("cycle、policy 与 train seed 不能为空")
        if not last_promoted_strategy_policy or not last_promoted_battle_policy:
            raise ValueError("最后 promoted 双 policy 不能为空")
        return cls(
            cycle_id=cycle_id,
            phase=CyclePhase.CREATED,
            parent_strategy_policy=strategy_policy,
            parent_battle_policy=battle_policy,
            strategy_policy=strategy_policy,
            battle_policy=battle_policy,
            train_seed=train_seed,
            last_promoted_strategy_policy=last_promoted_strategy_policy,
            last_promoted_battle_policy=last_promoted_battle_policy,
        )

    def advance(
        self,
        phase: CyclePhase,
        *,
        strategy_policy: str | None = None,
        battle_policy: str | None = None,
        promotion_receipt: Mapping[str, Any] | None = None,
    ) -> "CycleJournal":
        """按冻结顺序推进一个完整 group 或 optimizer 边界。

        Args:
            phase (CyclePhase): 目标完成边界。
            strategy_policy (str | None): 战略更新后发布的新 residual 名。
            battle_policy (str | None): 战斗更新后发布的新 residual 名。
            promotion_receipt (Mapping[str, Any] | None): 仅 promotion 时使用的批准收据。

        Raises:
            ValueError: 阶段跳跃、缺少新 policy，或在错误边界替换 policy。

        Returns:
            CycleJournal: 推进后的不可变记录。
        """
        allowed = _NEXT_PHASES.get(self.phase, set())
        if phase not in allowed:
            if phase is CyclePhase.VALIDATED:
                raise ValueError("validation 前必须完成 battle refresh")
            raise ValueError(f"轮次阶段不能从 {self.phase.value} 跳到 {phase.value}")
        if phase is CyclePhase.STRATEGY_UPDATED:
            if not strategy_policy or strategy_policy == self.strategy_policy:
                raise ValueError("战略更新必须发布新的 strategy policy")
        elif strategy_policy is not None:
            raise ValueError("只能在战略更新边界替换 strategy policy")
        if phase is CyclePhase.BATTLE_UPDATED:
            if not battle_policy or battle_policy == self.battle_policy:
                raise ValueError("战斗更新必须发布新的 battle policy")
        elif battle_policy is not None:
            raise ValueError("只能在战斗更新边界替换 battle policy")
        receipt = dict(promotion_receipt) if promotion_receipt is not None else None
        if phase is CyclePhase.PROMOTED:
            if (
                receipt is None
                or receipt.get("strategy_policy") != self.strategy_policy
                or receipt.get("battle_policy") != self.battle_policy
                or receipt.get("parent_strategy_policy")
                != self.last_promoted_strategy_policy
                or receipt.get("parent_battle_policy")
                != self.last_promoted_battle_policy
                or not _valid_approved_receipt(receipt)
            ):
                raise ValueError("原子 promotion 缺少匹配双 policy 的硬门控收据")
        elif receipt is not None:
            raise ValueError("promotion 收据只能在 promoted 边界写入")
        return replace(
            self,
            phase=phase,
            strategy_policy=strategy_policy or self.strategy_policy,
            battle_policy=battle_policy or self.battle_policy,
            last_promoted_strategy_policy=(
                self.strategy_policy
                if phase is CyclePhase.PROMOTED
                else self.last_promoted_strategy_policy
            ),
            last_promoted_battle_policy=(
                self.battle_policy
                if phase is CyclePhase.PROMOTED
                else self.last_promoted_battle_policy
            ),
            promotion_receipt=receipt or self.promotion_receipt,
        )


def build_promotion_receipt(
    *,
    parent_report: Mapping[str, Any],
    rolling_candidate_reports: Sequence[Mapping[str, Any]],
    battle_regression_passed: bool,
    branch_regret_passed: bool,
    illegal_action_passed: bool,
    potion_guard_passed: bool,
) -> dict[str, Any]:
    """根据三次滚动验证与四项硬回归生成原子晋升收据。

    Args:
        parent_report (Mapping[str, Any]): 最后 promoted 父 pair 的 3+1 报告。
        rolling_candidate_reports (Sequence[Mapping[str, Any]]): 最近三次完整验证。
        battle_regression_passed (bool): 固定战斗套件是否无回归。
        branch_regret_passed (bool): held-out Tree regret 是否无回归。
        illegal_action_passed (bool): 非法动作率是否通过硬门槛。
        potion_guard_passed (bool): 药水行为 guard 是否通过。

    Raises:
        ValueError: 报告数量、policy 身份或核心指标无效。

    Returns:
        dict[str, Any]: 可供 ``CycleJournal`` 原子 promotion 的批准或拒绝收据。
    """
    reports = tuple(rolling_candidate_reports)
    if len(reports) != 3:
        raise ValueError("promotion 必须使用最近三次完整 3+1 验证")
    strategy = reports[-1].get("strategy_policy")
    battle = reports[-1].get("battle_policy")
    parent_strategy = parent_report.get("strategy_policy")
    parent_battle = parent_report.get("battle_policy")
    if (
        not isinstance(strategy, str)
        or not strategy
        or not isinstance(battle, str)
        or not battle
    ):
        raise ValueError("promotion 候选报告缺少双 policy 身份")
    if (
        not isinstance(parent_strategy, str)
        or not parent_strategy
        or not isinstance(parent_battle, str)
        or not parent_battle
    ):
        raise ValueError("promotion 父报告缺少双 policy 身份")
    parent_metrics = _evaluation_metrics(parent_report)
    candidate_metrics = tuple(_evaluation_metrics(report) for report in reports)
    parent_frozen = _frozen_metrics(parent_report)
    candidate_frozen = tuple(_frozen_metrics(report) for report in reports)
    frozen_seeds = tuple(item["seed"] for item in parent_frozen)
    if any(
        tuple(item["seed"] for item in metrics) != frozen_seeds
        for metrics in candidate_frozen
    ):
        raise ValueError("promotion 父子报告必须使用相同顺序的 frozen seeds")
    paired_no_regression = all(
        sum(item[field] for item in candidate_frozen[-1])
        >= sum(item[field] for item in parent_frozen)
        for field in ("victory", "bosses_cleared", "final_floor")
    )
    rolling_no_regression = all(
        sum(metrics[field] for metrics in candidate_metrics) / 3
        >= parent_metrics[field]
        for field in ("victories", "bosses_cleared", "mean_floor")
    )
    gates = {
        "paired_no_regression": paired_no_regression,
        "rolling_no_regression": rolling_no_regression,
        "battle_regression_passed": battle_regression_passed,
        "branch_regret_passed": branch_regret_passed,
        "illegal_action_passed": illegal_action_passed,
        "potion_guard_passed": potion_guard_passed,
    }
    return {
        "format": "stage7_promotion_receipt",
        "strategy_policy": strategy,
        "battle_policy": battle,
        "parent_strategy_policy": parent_strategy,
        "parent_battle_policy": parent_battle,
        "validations_used": 3,
        "paired_frozen_seeds": list(frozen_seeds),
        "gates": gates,
        "approved": all(gates.values()),
    }


def write_cycle_journal(journal: CycleJournal, path: Path) -> Path:
    """把当前完整 group/optimizer 边界原子写成可恢复 JSON。

    Args:
        journal (CycleJournal): 当前不可变轮次状态。
        path (Path): 固定 journal 文件。

    Returns:
        Path: 实际 journal 路径。
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "format": "stage7_cycle_journal",
        "cycle_id": journal.cycle_id,
        "phase": journal.phase.value,
        "parent_strategy_policy": journal.parent_strategy_policy,
        "parent_battle_policy": journal.parent_battle_policy,
        "strategy_policy": journal.strategy_policy,
        "battle_policy": journal.battle_policy,
        "train_seed": journal.train_seed,
        "last_promoted_strategy_policy": journal.last_promoted_strategy_policy,
        "last_promoted_battle_policy": journal.last_promoted_battle_policy,
        "promotion_receipt": journal.promotion_receipt,
    }
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_cycle_journal(path: Path) -> CycleJournal:
    """读取一个已持久化的阶段七轮次边界。

    Args:
        path (Path): journal JSON。

    Raises:
        ValueError: 格式、阶段或 policy 字段无效。

    Returns:
        CycleJournal: 可继续推进的轮次状态。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        journal = CycleJournal(
            cycle_id=str(payload["cycle_id"]),
            phase=CyclePhase(str(payload["phase"])),
            parent_strategy_policy=str(payload["parent_strategy_policy"]),
            parent_battle_policy=str(payload["parent_battle_policy"]),
            strategy_policy=str(payload["strategy_policy"]),
            battle_policy=str(payload["battle_policy"]),
            train_seed=str(payload["train_seed"]),
            last_promoted_strategy_policy=str(payload["last_promoted_strategy_policy"]),
            last_promoted_battle_policy=str(payload["last_promoted_battle_policy"]),
            promotion_receipt=(
                dict(payload["promotion_receipt"])
                if isinstance(payload.get("promotion_receipt"), Mapping)
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("阶段七 cycle journal 字段无效") from exc
    if payload.get("format") != "stage7_cycle_journal" or not all(
        (
            journal.cycle_id,
            journal.parent_strategy_policy,
            journal.parent_battle_policy,
            journal.strategy_policy,
            journal.battle_policy,
            journal.train_seed,
            journal.last_promoted_strategy_policy,
            journal.last_promoted_battle_policy,
        )
    ):
        raise ValueError("阶段七 cycle journal 格式无效")
    return journal


def _evaluation_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    """读取 promotion 使用的三项整局核心指标。

    Args:
        report (Mapping[str, Any]): 3+1 验证报告。

    Raises:
        ValueError: 报告格式或指标无效。

    Returns:
        dict[str, float]: 胜局、Boss 和平均楼层。
    """
    if report.get("format") != "stage7_policy_evaluation":
        raise ValueError("promotion 只接受阶段七完整验证报告")
    try:
        metrics = {
            "victories": float(report["victories"]),
            "bosses_cleared": float(report["bosses_cleared"]),
            "mean_floor": float(report["mean_floor"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("promotion 验证指标无效") from exc
    if any(value < 0 for value in metrics.values()):
        raise ValueError("promotion 验证指标无效")
    return metrics


def _frozen_metrics(report: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """读取三条固定 seed 的逐局 paired 指标。

    Args:
        report (Mapping[str, Any]): 阶段七 3+1 验证报告。

    Raises:
        ValueError: frozen 记录不是三个不同 seed 或字段无效。

    Returns:
        tuple[dict[str, Any], ...]: 保持配置顺序的 seed、胜利、Boss 与楼层。
    """
    raw = report.get("frozen")
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError("promotion 报告必须包含三条 frozen seed 结果")
    metrics = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise TypeError("promotion frozen 结果必须是对象")
        seed = item.get("seed")
        victory = item.get("victory")
        bosses = item.get("bosses_cleared")
        floor = item.get("final_floor")
        if (
            not isinstance(seed, str)
            or not seed
            or not isinstance(victory, bool)
            or isinstance(bosses, bool)
            or not isinstance(bosses, int)
            or bosses < 0
            or isinstance(floor, bool)
            or not isinstance(floor, int)
            or floor < 0
        ):
            raise ValueError("promotion frozen 结果字段无效")
        metrics.append(
            {
                "seed": seed,
                "victory": int(victory),
                "bosses_cleared": bosses,
                "final_floor": floor,
            }
        )
    if len({item["seed"] for item in metrics}) != 3:
        raise ValueError("promotion frozen seeds 必须彼此不同")
    return tuple(metrics)


def _valid_approved_receipt(receipt: Mapping[str, Any]) -> bool:
    """核对 journal 消费的 promotion 收据完整且全部硬门槛通过。

    Args:
        receipt (Mapping[str, Any]): ``decide-rl-promotion`` 生成的 JSON。

    Returns:
        bool: 格式、次数、paired seeds 和六项 gates 全部有效时返回真。
    """
    required_gates = {
        "paired_no_regression",
        "rolling_no_regression",
        "battle_regression_passed",
        "branch_regret_passed",
        "illegal_action_passed",
        "potion_guard_passed",
    }
    gates = receipt.get("gates")
    paired = receipt.get("paired_frozen_seeds")
    return (
        receipt.get("format") == "stage7_promotion_receipt"
        and receipt.get("validations_used") == 3
        and receipt.get("approved") is True
        and isinstance(gates, Mapping)
        and set(gates) == required_gates
        and all(gates.get(name) is True for name in required_gates)
        and isinstance(paired, list)
        and len(paired) == 3
        and len(set(paired)) == 3
        and all(isinstance(seed, str) and seed for seed in paired)
    )
