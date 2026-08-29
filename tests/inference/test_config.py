"""验证本地推理配置的模型身份和生成 profile 契约。"""

from pathlib import Path

import pytest


def test_local_inference_config_preserves_canonical_evaluation_profiles() -> None:
    """已配置的本地推理文件以 no-think 对齐 SFT，并保留 think 对照组。"""
    from play_sts2.inference.config import (
        DEFAULT_INFERENCE_CONFIG,
        load_inference_config,
    )

    if not DEFAULT_INFERENCE_CONFIG.is_file():
        pytest.skip("本地 inference.toml 不进入 Git")

    config = load_inference_config(DEFAULT_INFERENCE_CONFIG)

    assert config.default_profile == "no-think"
    assert config.profile("no-think").enable_thinking is False
    assert config.profile("think").enable_thinking is True


def test_load_inference_config_exposes_explicit_thinking_profiles(
    tmp_path: Path,
) -> None:
    """配置同时声明无思考与思考模式，且默认模式必须显式存在。"""
    from play_sts2.inference.config import load_inference_config

    config_path = tmp_path / "inference.toml"
    config_path.write_text(
        """
artifact_id = "demo-e3"
merged_model = "models/merged/demo-e3-merged"
serving_model = "models/serving/demo-e3-mlx-8bit"
default_profile = "no-think"

[server]
base_url = "http://127.0.0.1:9000"
port = 9000
prompt_cache_size = 6

[profiles.no-think]
enable_thinking = false
max_tokens = 128
temperature = 0.0

[profiles.think]
enable_thinking = true
max_tokens = 512
temperature = 0.2
""".strip(),
        encoding="utf-8",
    )

    config = load_inference_config(config_path)

    assert config.artifact_id == "demo-e3"
    assert config.merged_model == Path("models/merged/demo-e3-merged")
    assert config.serving_model == Path("models/serving/demo-e3-mlx-8bit")
    assert config.base_url == "http://127.0.0.1:9000"
    assert config.port == 9000
    assert config.prompt_cache_size == 6
    assert config.default_profile == "no-think"
    assert config.profile().enable_thinking is False
    assert config.profile().max_tokens == 128
    assert config.profile("think").enable_thinking is True
    assert config.profile("think").max_tokens == 512
    assert config.profile("think").temperature == 0.2


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ('default_profile = "missing"', "默认 profile 不存在"),
        (
            'serving_model = "models/serving/wrong-mlx-8bit"',
            "服务模型目录必须对应 artifact_id",
        ),
        (
            'base_url = "http://127.0.0.1:9001"',
            "base_url 端口与 server.port 不一致",
        ),
    ],
)
def test_load_inference_config_rejects_inconsistent_identity(
    tmp_path: Path,
    replacement: str,
    message: str,
) -> None:
    """配置内互相矛盾的模型身份或服务地址必须在启动前失败。"""
    from play_sts2.inference.config import InferenceConfigError, load_inference_config

    source = """
artifact_id = "demo-e3"
merged_model = "models/merged/demo-e3-merged"
serving_model = "models/serving/demo-e3-mlx-8bit"
default_profile = "no-think"

[server]
base_url = "http://127.0.0.1:9000"
port = 9000
prompt_cache_size = 6

[profiles.no-think]
enable_thinking = false
max_tokens = 128
temperature = 0.0

[profiles.think]
enable_thinking = true
max_tokens = 512
temperature = 0.0
""".strip()
    if replacement.startswith("default_profile"):
        source = source.replace('default_profile = "no-think"', replacement)
    elif replacement.startswith("serving_model"):
        source = source.replace(
            'serving_model = "models/serving/demo-e3-mlx-8bit"', replacement
        )
    else:
        source = source.replace('base_url = "http://127.0.0.1:9000"', replacement)
    config_path = tmp_path / "inference.toml"
    config_path.write_text(source, encoding="utf-8")

    with pytest.raises(InferenceConfigError, match=message):
        load_inference_config(config_path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            'default_profile = "no-think"',
            'default_profile = "think"',
            "默认 profile 必须是 no-think",
        ),
        (
            "\n[profiles.think]\nenable_thinking = true\nmax_tokens = 512\ntemperature = 0.0",
            "",
            "缺少 think profile",
        ),
        (
            "[profiles.no-think]\nenable_thinking = false",
            "[profiles.no-think]\nenable_thinking = true",
            "no-think profile 必须关闭 thinking",
        ),
        (
            "[profiles.think]\nenable_thinking = true",
            "[profiles.think]\nenable_thinking = false",
            "think profile 必须开启 thinking",
        ),
    ],
)
def test_load_inference_config_enforces_canonical_profile_semantics(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    """profile 名称不能与 thinking 语义或默认评测模式脱钩。"""
    from play_sts2.inference.config import InferenceConfigError, load_inference_config

    source = """
artifact_id = "demo-e3"
merged_model = "models/merged/demo-e3-merged"
serving_model = "models/serving/demo-e3-mlx-8bit"
default_profile = "no-think"

[server]
base_url = "http://127.0.0.1:9000"
port = 9000
prompt_cache_size = 6

[profiles.no-think]
enable_thinking = false
max_tokens = 128
temperature = 0.0

[profiles.think]
enable_thinking = true
max_tokens = 512
temperature = 0.0
""".strip()
    source = source.replace(old, new)
    config_path = tmp_path / "inference.toml"
    config_path.write_text(source, encoding="utf-8")

    with pytest.raises(InferenceConfigError, match=message):
        load_inference_config(config_path)
