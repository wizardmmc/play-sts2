"""验证 held-out Tree checkpoint 的分支 regret 计算与查询。"""

import json
from pathlib import Path
from typing import Self

from play_sts2.inference import ModelReply
from play_sts2.training.rl.strategy import evaluation


class _Provider:
    """返回固定结构化宏动作的测试 Provider。"""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        """接受与真实 Provider 相同的构造参数。

        Args:
            *_args (object): 未使用的位置参数。
            **_kwargs (object): 未使用的关键字参数。

        Returns:
            None: 测试替身无需保存参数。
        """

    def __enter__(self) -> Self:
        """返回上下文中的测试 Provider。

        Returns:
            _Provider: 当前实例。
        """
        return self

    def __exit__(self, *_args: object) -> None:
        """结束测试 Provider 上下文。

        Args:
            *_args (object): 上下文异常信息。

        Returns:
            None: 测试替身没有资源。
        """

    def chat(self, *_args: object, **_kwargs: object) -> ModelReply:
        """选择已采样但不是最优的宏动作。

        Args:
            *_args (object): 未使用的消息。
            **_kwargs (object): 未使用的生成参数。

        Returns:
            ModelReply: 固定为 ``ACTION: choose 1``。
        """
        return ModelReply(text="ACTION: choose 1", model="strategy-after")


def test_evaluate_tree_branch_regret_uses_sampled_plan_means(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    """regret 应比较 policy 所选计划与最佳已采兄弟的平均 return。

    Args:
        tmp_path (Path): Pytest 临时目录。
        monkeypatch (object): Pytest 属性替换工具。

    Returns:
        None: 断言报告保留 checkpoint、计数和非负 regret。
    """
    rollout = tmp_path / "heldout.json"
    output = tmp_path / "regret.json"
    arms = []
    for index, (plan, score) in enumerate(
        (
            ("ACTION: choose 0", 1.0),
            ("ACTION: choose 0", 3.0),
            ("ACTION: choose 1", 0.5),
            ("ACTION: choose 1", 1.5),
            ("ACTION: choose 2", 4.0),
            ("ACTION: choose 2", 4.0),
            ("ACTION: choose 2", 5.0),
            ("ACTION: choose 2", 3.0),
        )
    ):
        arms.append(
            {
                "arm_index": index,
                "plan_id": plan,
                "continuation_return": {"total": score},
                "steps": [
                    {
                        "action": plan,
                        "messages": [
                            {"role": "system", "content": "system"},
                            {"role": "user", "content": "state"},
                        ],
                        "response_choices": [
                            "ACTION: choose 0",
                            "ACTION: choose 1",
                            "ACTION: choose 2",
                        ],
                    }
                ],
            }
        )
    rollout.write_text(
        json.dumps(
            {
                "format": "tree_grpo_group",
                "group_id": "heldout-001",
                "checkpoint": {"kind": "event"},
                "arms": arms,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluation, "OpenAICompatibleProvider", _Provider)

    report = evaluation.evaluate_tree_branch_regret(
        rollout_path=rollout,
        model_url="http://127.0.0.1:8900",
        policy_model="strategy-after",
        output_path=output,
    )

    assert report["selected_plan"] == "ACTION: choose 1"
    assert report["selected_mean_return"] == 1.0
    assert report["best_sampled_plan"] == "ACTION: choose 2"
    assert report["best_sampled_mean_return"] == 4.0
    assert report["regret"] == 3.0
    assert report["sampled_plan_counts"] == {
        "ACTION: choose 0": 2,
        "ACTION: choose 1": 2,
        "ACTION: choose 2": 4,
    }
    assert json.loads(output.read_text(encoding="utf-8")) == report
