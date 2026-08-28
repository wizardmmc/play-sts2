"""读取并校验本地推理模型、服务与生成 profile 配置。"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DEFAULT_INFERENCE_CONFIG = Path("configs/inference.toml")


class InferenceConfigError(ValueError):
    """表示推理配置缺少字段、类型错误或内部身份不一致。"""


@dataclass(frozen=True, slots=True)
class InferenceProfile:
    """表示一种显式的 chat template 与采样组合。"""

    enable_thinking: bool
    max_tokens: int
    temperature: float


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """表示一个不可变的本地模型服务配置。"""

    artifact_id: str
    merged_model: Path
    serving_model: Path
    base_url: str
    port: int
    prompt_cache_size: int
    default_profile: str
    profiles: Mapping[str, InferenceProfile]

    def profile(self, name: str | None = None) -> InferenceProfile:
        """返回指定 profile，省略名称时使用配置声明的默认项。"""
        selected = self.default_profile if name is None else name
        try:
            return self.profiles[selected]
        except KeyError as exc:
            raise InferenceConfigError(f"推理 profile 不存在: {selected}") from exc


def load_inference_config(path: Path) -> InferenceConfig:
    """从 TOML 读取推理配置，并在接触模型或网络前校验一致性。"""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    artifact_id = _string(data, "artifact_id")
    merged_model = Path(_string(data, "merged_model"))
    serving_model = Path(_string(data, "serving_model"))
    default_profile = _string(data, "default_profile")
    server = _mapping(data, "server")
    base_url = _string(server, "base_url")
    port = _integer(server, "port", minimum=1)
    prompt_cache_size = _integer(server, "prompt_cache_size", minimum=1)

    raw_profiles = _mapping(data, "profiles")
    profiles = {
        name: _profile(name, value)
        for name, value in raw_profiles.items()
        if isinstance(name, str)
    }
    if len(profiles) != len(raw_profiles) or not profiles:
        raise InferenceConfigError("profiles 必须是非空字符串键表")
    if default_profile not in profiles:
        raise InferenceConfigError(f"默认 profile 不存在: {default_profile}")
    if default_profile != "no-think":
        raise InferenceConfigError("默认 profile 必须是 no-think")
    if "no-think" not in profiles:
        raise InferenceConfigError("缺少 no-think profile")
    if "think" not in profiles:
        raise InferenceConfigError("缺少 think profile")
    if profiles["no-think"].enable_thinking:
        raise InferenceConfigError("no-think profile 必须关闭 thinking")
    if not profiles["think"].enable_thinking:
        raise InferenceConfigError("think profile 必须开启 thinking")
    if merged_model.name != f"{artifact_id}-merged":
        raise InferenceConfigError("合并模型目录必须对应 artifact_id")
    if serving_model.name != f"{artifact_id}-mlx-8bit":
        raise InferenceConfigError("服务模型目录必须对应 artifact_id")

    parsed_url = urlsplit(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise InferenceConfigError("server.base_url 必须是 HTTP(S) URL")
    try:
        url_port = parsed_url.port
    except ValueError as exc:
        raise InferenceConfigError("server.base_url 端口无效") from exc
    if url_port != port:
        raise InferenceConfigError("base_url 端口与 server.port 不一致")

    return InferenceConfig(
        artifact_id=artifact_id,
        merged_model=merged_model,
        serving_model=serving_model,
        base_url=base_url.rstrip("/"),
        port=port,
        prompt_cache_size=prompt_cache_size,
        default_profile=default_profile,
        profiles=profiles,
    )


def _profile(name: str, value: object) -> InferenceProfile:
    """校验并构造一个命名生成 profile。"""
    if not isinstance(value, Mapping):
        raise InferenceConfigError(f"profile 必须是表: {name}")
    enable_thinking = value.get("enable_thinking")
    if not isinstance(enable_thinking, bool):
        raise InferenceConfigError(f"profile.enable_thinking 必须是布尔值: {name}")
    max_tokens = _integer(value, "max_tokens", minimum=1)
    temperature = value.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise InferenceConfigError(f"profile.temperature 必须是数字: {name}")
    if not 0.0 <= float(temperature) <= 2.0:
        raise InferenceConfigError(f"profile.temperature 必须位于 [0, 2]: {name}")
    return InferenceProfile(
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
        temperature=float(temperature),
    )


def _mapping(data: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    """读取必需的 TOML 表。"""
    value = data.get(field)
    if not isinstance(value, Mapping):
        raise InferenceConfigError(f"{field} 必须是表")
    return value


def _string(data: Mapping[str, Any], field: str) -> str:
    """读取必需的非空字符串。"""
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise InferenceConfigError(f"{field} 必须是非空字符串")
    return value


def _integer(data: Mapping[str, Any], field: str, *, minimum: int) -> int:
    """读取具有下界的整数，同时拒绝 Python 的布尔整数别名。"""
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InferenceConfigError(f"{field} 必须是不小于 {minimum} 的整数")
    return value
