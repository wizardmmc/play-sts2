"""验证服务别名不能掩盖实际加载的旧 residual。"""

import json
from pathlib import Path

import pytest


def test_serving_job_rejects_s20_alias_pointing_to_s3(tmp_path: Path) -> None:
    """复现真实事故：只改别名、仍加载 S3 时应在启动前拒绝。"""
    from play_sts2.training.rl.orchestration.serving import validate_serving_job

    adapter = tmp_path / "s3-serving"
    adapter.mkdir()
    (adapter / "serving_manifest.json").write_text(
        json.dumps(
            {"name": "qwen3.5-e7-s3", "residual_adapter": "adapters/qwen3.5-e7-s3"}
        )
    )
    with pytest.raises(ValueError, match="qwen3.5-e7-s20"):
        validate_serving_job(
            {"argv": ["--enable-lora", "--lora-modules", f"qwen3.5-e7-s20={adapter}"]},
            root=tmp_path,
        )


def test_online_receipt_rejects_same_name_with_old_root() -> None:
    """模型列表里名字正确、实际 root 错误时不得通过采集前检查。"""
    from play_sts2.training.rl.orchestration.serving import validate_serving_models

    with pytest.raises(ValueError, match="root"):
        validate_serving_models(
            {"data": [{"id": "s20", "root": "models/s3-serving"}]},
            {"s20": "models/s20-serving"},
        )


def test_online_receipt_accepts_exact_binding() -> None:
    """在线 root 与已验证的启动配置一致时返回实际绑定。"""
    from play_sts2.training.rl.orchestration.serving import validate_serving_models

    assert validate_serving_models(
        {"data": [{"id": "s3", "root": "models/s3-serving"}]},
        {"s3": "models/s3-serving"},
    ) == {"s3": "models/s3-serving"}


def test_serving_job_accepts_real_matching_manifest(tmp_path: Path) -> None:
    """正确映射必须能通过，并保留服务使用的原始相对路径。"""
    from play_sts2.training.rl.orchestration.serving import validate_serving_job

    adapter = tmp_path / "s3-serving"
    adapter.mkdir()
    (adapter / "serving_manifest.json").write_text(
        json.dumps({"name": "s3", "residual_adapter": "adapters/s3"})
    )
    (adapter / "adapter_model.safetensors").touch()
    assert validate_serving_job(
        {
            "argv": [
                "--enable-lora",
                "--lora-modules",
                "s3=s3-serving",
                "--port",
                "18000",
            ]
        },
        root=tmp_path,
    ) == {"s3": "s3-serving"}
