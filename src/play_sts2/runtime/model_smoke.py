"""验证模型服务与游戏 Harness 之间的最小协议。"""

import httpx

from ..harness import HarnessLayer, parse_action, system_prompt
from ..inference import ChatMessage, ModelReply, OpenAICompatibleProvider


def smoke_model(
    base_url: str,
    *,
    model: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> ModelReply:
    """检查模型服务健康状态并生成一个合法战斗动作。

    Args:
        base_url (str): OpenAI-compatible 服务根地址。
        model (str | None): 服务要求显式传入时使用的模型名称。
        transport (httpx.BaseTransport | None): 测试时可替换的 HTTP 传输。

    Raises:
        httpx.HTTPStatusError: 健康检查或生成请求失败。
        ActionParseError: 模型回复不符合 Harness 动作协议。

    Returns:
        ModelReply: 已通过动作协议校验的模型原始回复。
    """
    with httpx.Client(
        base_url=base_url.rstrip("/") + "/",
        transport=transport,
    ) as client:
        health = client.get("health")
        health.raise_for_status()

    messages = (
        ChatMessage("system", system_prompt(HarnessLayer.BATTLE)),
        ChatMessage(
            "user",
            "战斗状态:\n- 当前为玩家回合\n\n可执行动作:\n- end_turn",
        ),
    )
    with OpenAICompatibleProvider(
        base_url,
        model=model,
        transport=transport,
    ) as provider:
        reply = provider.chat(messages, max_tokens=48, temperature=0.0)
    parse_action(reply.text, ("end_turn",))
    return reply
