"""验证 SFT eval 树知识与组合题的加载和稳定评分口径。"""

import json
from pathlib import Path

from play_sts2.training.sft.knowledge_evaluation import (
    _report_payload,
    load_knowledge_probes,
    score_compositional_answer,
    score_recall_answer,
)


def test_load_knowledge_probes_reads_eval_tree_and_skips_behavior(
    tmp_path: Path,
) -> None:
    """知识评测器应读取类别文件并跳过 combat/strategy。

    Args:
        tmp_path (Path): Pytest 提供的隔离 eval 目录。

    Raises:
        AssertionError: 新目录格式没有转换成知识与算术题。

    Returns:
        None: 此测试不加载模型。
    """
    root = tmp_path / "eval"
    fixtures = {
        "cards/ZAP.jsonl": {
            "category": "cards",
            "object_id": "ZAP",
            "messages": [
                {"role": "user", "content": "Q: 电击是什么？\nA:"},
                {"role": "assistant", "content": "生成闪电。"},
            ],
        },
        "arithmetic/block_math.jsonl": {
            "category": "arithmetic",
            "object_id": "block_math",
            "messages": [
                {"role": "user", "content": "Q: 伤害结算？\nA:"},
                {"role": "assistant", "content": "剩余18点。"},
            ],
        },
        "combat/RUN/battle.jsonl": {
            "category": "behavior",
            "object_id": "ignored",
            "messages": [
                {"role": "user", "content": "状态"},
                {"role": "assistant", "content": "ACTION: end_turn"},
            ],
        },
    }
    for relative, row in fixtures.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(row, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    probes = load_knowledge_probes(root)

    assert [(probe.kind, probe.prompt) for probe in probes] == [
        ("form_holdout", "电击是什么？"),
        ("block_math", "伤害结算？"),
    ]


def test_score_recall_answer_requires_complete_reference() -> None:
    """知识召回只把完整参考答案判为自动通过。

    Raises:
        AssertionError: 缺少任一参考事实的回答被误判为通过。

    Returns:
        None: 此测试只检查无模型依赖的评分函数。
    """
    reference = "3费能力。获得1点敏捷和4点荆棘。"

    assert score_recall_answer("3费能力。获得1点敏捷和4点荆棘。", reference) is True
    assert score_recall_answer("答案是3费，敏捷1。", reference) is False


def test_score_recall_answer_rejects_numbers_without_factual_text() -> None:
    """含数字参考答案不能只凭数字集合命中。

    Raises:
        AssertionError: 无事实文本的数字串被误判为知识召回成功。

    Returns:
        None: 此测试只检查保守的可审计评分下限。
    """
    reference = "0费技能。失去1点生命，获得1点力量。"

    assert score_recall_answer("0 1", reference) is False
    assert score_recall_answer("0费技能。失去1点生命，获得1点力量。", reference)
    assert (
        score_recall_answer("0费技能，失去1点生命并获得1点力量，共2点。", reference)
        is False
    )


def test_score_recall_answer_ignores_whitespace_only() -> None:
    """无数字知识只忽略空白，不放宽事实文本。

    Raises:
        AssertionError: 仅空白差异没有被稳定归一化。

    Returns:
        None: 此测试只检查文本型知识评分。
    """
    assert score_recall_answer(
        "每场战斗开始时生成闪电充能球。",
        " 每场战斗开始时生成闪电充能球。",
    )


def test_score_compositional_answer_checks_choice_and_final_numbers() -> None:
    """组合题同时核对生死结论和构建期标出的最终数字。

    Raises:
        AssertionError: 相反结论或错误数字被接受。

    Returns:
        None: 此测试只检查组合算术评分。
    """
    probe = {"answer_choice": "不会死", "answer_numbers": [18]}

    assert score_compositional_answer("不会死，剩余18点生命。", probe) is True
    assert score_compositional_answer("会死，剩余18点生命。", probe) is False
    assert score_compositional_answer("不会死，剩余17点生命。", probe) is False
    assert score_compositional_answer("不会死，剩余18点，也可能17点。", probe) is False


def test_report_payload_keeps_every_scored_probe() -> None:
    """完整评测报告保留全部逐题输入、输出和判定。

    Raises:
        AssertionError: 报告只保留抽样或截断失败列表。

    Returns:
        None: 此测试不运行真实模型。
    """
    results = [{"prompt": str(index), "passed": index % 2 == 0} for index in range(100)]

    report = _report_payload({"questions": 100}, results)

    assert report["results"] == results
    assert len(report["results"]) == 100
