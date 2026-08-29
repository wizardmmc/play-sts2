"""验证 SFT 生成解码、评分与逐样本报告。"""

import json
from pathlib import Path

from play_sts2.training.sft import (
    GeneratedReply,
    decode_generation,
    evaluate_rows,
    score_generation,
)


def test_score_generation_reports_exact_match_and_action_shape() -> None:
    """生成评分区分全文精确匹配和 Harness 动作外形。

    Raises:
        AssertionError: 评分没有去除首尾空白或错误接受多行动作。

    Returns:
        None: 此测试只检查无模型依赖的评测口径。
    """
    exact = score_generation(
        expected="ACTION: end_turn",
        generated="  ACTION: end_turn\n",
        source="human_play",
    )
    invalid = score_generation(
        expected="ACTION: end_turn",
        generated="ACTION: end_turn\n额外说明",
        source="human_play",
    )

    assert exact.exact_match is True
    assert exact.action_shape_valid is True
    assert invalid.exact_match is False
    assert invalid.action_shape_valid is False


def test_decode_generation_marks_missing_eos_as_truncated() -> None:
    """区分正常 EOS 结束与生成预算耗尽。

    Raises:
        AssertionError: EOS 检测或解码 token 范围不符合约定。

    Returns:
        None: 此测试使用固定 token 解码器。
    """

    class FakeDecoder:
        """把 token ID 连接为便于断言的文本。"""

        def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
            """解码一段生成 token。

            Args:
                token_ids (list[int]): 模型生成的 token ID。
                skip_special_tokens (bool): 是否移除特殊 token。

            Returns:
                str: 用逗号连接的普通 token。
            """
            assert skip_special_tokens is True
            return ",".join(str(token) for token in token_ids if token != 99)

    complete = decode_generation(FakeDecoder(), [3, 4, 99], eos_token_id=99)
    truncated = decode_generation(FakeDecoder(), [3, 4], eos_token_id=99)

    assert complete == GeneratedReply("3,4", truncated=False)
    assert truncated == GeneratedReply("3,4", truncated=True)


def test_evaluate_rows_writes_per_sample_generations(tmp_path: Path) -> None:
    """生成式评测保存逐样本结果并汇总行为动作外形。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据与报告目录。

    Raises:
        AssertionError: 评测计数、指标或落盘内容不符合约定。

    Returns:
        None: 此测试使用确定性生成函数，不加载真实模型。
    """
    dataset = tmp_path / "validation"
    rows = [
        {
            "sample_id": "knowledge/ZAP",
            "source": "web_wiki",
            "messages": [
                {"role": "user", "content": "电击是什么？"},
                {"role": "assistant", "content": "生成闪电充能球。"},
            ],
        },
        {
            "sample_id": "human/RUN/1",
            "source": "human_play",
            "messages": [
                {"role": "user", "content": "可结束回合。"},
                {"role": "assistant", "content": "ACTION: end_turn"},
            ],
        },
    ]
    for relative, row in zip(("cards/ZAP.jsonl", "strategy/RUN.jsonl"), rows):
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(row, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    summary = evaluate_rows(
        dataset,
        lambda messages: GeneratedReply(
            "生成闪电充能球。"
            if messages[-1]["content"] == "电击是什么？"
            else "ACTION: play_card 0",
            truncated=messages[-1]["content"] != "电击是什么？",
        ),
        output_path=tmp_path / "eval.jsonl",
    )

    assert summary == {
        "samples": 2,
        "exact_matches": 1,
        "exact_match_rate": 0.5,
        "action_samples": 1,
        "valid_action_shapes": 1,
        "valid_action_shape_rate": 1.0,
        "truncated_samples": 1,
    }
    records = [
        json.loads(line)
        for line in (tmp_path / "eval.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[1]["expected"] == "ACTION: end_turn"
    assert records[1]["generated"] == "ACTION: play_card 0"
    assert records[1]["truncated"] is True
