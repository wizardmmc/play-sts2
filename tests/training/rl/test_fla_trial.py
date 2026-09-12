"""验证独立 FLA 试验不会静默丢失内核的有效参数。"""

import pytest

from play_sts2.training.rl.orchestration.fla_trial import make_trial_kernel


def test_trial_preserves_supported_arguments_and_counts_calls():
    """支持的有效参数原样传递，未支持的关闭缓存选项单独记录。"""
    receipt = {"kernel_calls": 0, "dropped_inactive_kwargs": []}

    def target(q, *, beta):
        """返回参数以检查传递契约。"""
        return q, beta

    kernel = make_trial_kernel(target, receipt)
    assert kernel(3, beta=0.7, use_cache=False) == (3, 0.7)
    assert receipt == {"kernel_calls": 1, "dropped_inactive_kwargs": ["use_cache"]}


def test_trial_rejects_unsupported_active_argument():
    """防止未识别的门控参数被当作无关参数忽略。"""
    receipt = {"kernel_calls": 0, "dropped_inactive_kwargs": []}

    def target(q):
        """提供不支持额外参数的内核。"""
        return q

    kernel = make_trial_kernel(target, receipt)
    with pytest.raises(ValueError, match="beta"):
        kernel(3, beta=0.7)
    assert receipt["kernel_calls"] == 0


def test_trial_rejects_cross_backend_resume():
    """独立后端试验不能精确续跑没有后端记录的旧训练。"""
    from play_sts2.training.rl.orchestration.fla_trial import validate_trial_job

    with pytest.raises(ValueError, match="resume"):
        validate_trial_job(["battle-grpo", "--resume", "old-checkpoint"])
