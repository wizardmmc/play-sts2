"""验证模型服务与游戏 Harness 之间的最小协议。"""

from collections.abc import Mapping
from pathlib import Path

import httpx

from ..harness import HarnessLayer, parse_action, system_prompt
from ..inference import (
    ChatMessage,
    InferenceGenerationTruncated,
    InferenceProtocolError,
    ModelReply,
    OpenAICompatibleProvider,
    ServingModelIdentityError,
    validate_serving_model,
)


def smoke_model(
    base_url: str,
    *,
    artifact_id: str,
    merged_model: Path,
    serving_model: Path,
    enable_thinking: bool,
    max_tokens: int,
    temperature: float,
    transport: httpx.BaseTransport | None = None,
) -> ModelReply:
    """检查模型服务健康状态并生成一个合法战斗动作。

    Args:
        base_url (str): OpenAI-compatible 服务根地址。
        artifact_id (str): 配置期望的不可变模型产物标识。
        merged_model (Path): 配置期望的合并模型来源。
        serving_model (Path): 配置期望的 MLX 模型目录。
        enable_thinking (bool): 本次冒烟显式选择的模板模式。
        max_tokens (int): profile 声明的最大生成 token 数。
        temperature (float): profile 声明的采样温度。
        transport (httpx.BaseTransport | None): 测试时可替换的 HTTP 传输。

    Raises:
        httpx.HTTPStatusError: 健康检查或生成请求失败。
        InferenceModelIdentityError: 服务响应的模型不是配置指定目录。
        ActionParseError: 模型回复不符合 Harness 动作协议。

    Returns:
        ModelReply: 已通过动作协议校验的模型原始回复。
    """
    validate_model_service(
        base_url,
        artifact_id=artifact_id,
        merged_model=merged_model,
        serving_model=serving_model,
        enable_thinking=enable_thinking,
        transport=transport,
    )

    messages = (
        ChatMessage("system", system_prompt(HarnessLayer.BATTLE)),
        ChatMessage(
            "user",
            "战斗状态:\n- 当前为玩家回合\n\n可执行动作:\n- end_turn",
        ),
    )
    with OpenAICompatibleProvider(
        base_url,
        model=str(Path(serving_model).resolve()),
        enable_thinking=enable_thinking,
        transport=transport,
    ) as provider:
        reply = provider.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    parse_action(reply.text, ("end_turn",))
    return reply


def validate_model_service(
    base_url: str,
    *,
    artifact_id: str,
    merged_model: Path,
    serving_model: Path,
    enable_thinking: bool,
    transport: httpx.BaseTransport | None = None,
) -> None:
    """在游戏发生任何状态变更前核对磁盘身份并完成一次真实生成。

    ``/v1/models`` 只证明服务声明了目录；随后的一 token 请求还会强制 MLX
    实际加载配置目录，并要求响应回报同一个绝对路径。
    """
    validate_serving_model(
        serving_model,
        artifact_id=artifact_id,
        merged_model=merged_model,
    )
    with httpx.Client(
        base_url=base_url.rstrip("/") + "/",
        transport=transport,
    ) as client:
        health = client.get("health")
        health.raise_for_status()
        models = client.get("v1/models")
        models.raise_for_status()
        if str(Path(serving_model).resolve()) not in _model_ids(models):
            raise ServingModelIdentityError("运行中的模型目录不一致")

    expected_model = str(Path(serving_model).resolve())
    with OpenAICompatibleProvider(
        base_url,
        model=expected_model,
        enable_thinking=enable_thinking,
        transport=transport,
    ) as provider:
        try:
            provider.chat(
                (ChatMessage("user", "Reply with OK."),),
                max_tokens=1,
                temperature=0.0,
            )
        except InferenceGenerationTruncated:
            # 一个 token 的探针常因长度停止；能获得绑定身份的生成响应即已就绪。
            pass


def _model_ids(response: httpx.Response) -> set[str]:
    """从 MLX ``/v1/models`` 响应读取当前可用模型标识。"""
    try:
        payload = response.json()
    except ValueError as exc:
        raise InferenceProtocolError("invalid models response") from exc
    if not isinstance(payload, Mapping):
        raise InferenceProtocolError("invalid models response")
    data = payload.get("data")
    if not isinstance(data, list):
        raise InferenceProtocolError("invalid models response")
    identifiers: set[str] = set()
    for item in data:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise InferenceProtocolError("invalid models response")
        identifiers.add(item["id"])
    return identifiers
