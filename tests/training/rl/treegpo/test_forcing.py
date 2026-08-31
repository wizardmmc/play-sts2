"""验证 terminal Tree 只强制第一条根动作。"""

from collections.abc import Sequence

from play_sts2.inference import ChatMessage, ModelReply


class Provider:
    """记录 wrapper 实际转发的 structured choices。"""

    def __init__(self) -> None:
        """初始化空调用记录。"""
        self.choices: list[tuple[str, ...] | None] = []

    def chat(
        self,
        _messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        response_choices: Sequence[str] | None = None,
    ) -> ModelReply:
        """返回转发候选的第一条动作。

        Args:
            _messages (Sequence[ChatMessage]): 当前 stateless 消息。
            max_tokens (int): token 预算。
            temperature (float): 采样温度。
            response_choices (Sequence[str] | None): wrapper 转发候选。

        Returns:
            ModelReply: 第一条候选动作。
        """
        assert max_tokens > 0 and temperature > 0
        choices = tuple(response_choices) if response_choices is not None else None
        self.choices.append(choices)
        assert choices
        return ModelReply(choices[0])


def test_forced_provider_only_restricts_first_root_request() -> None:
    """根动作之后的 continuation 应恢复冻结策略完整动作域。

    Returns:
        None: 首次转发 singleton，第二次转发原始完整 choices。
    """
    from play_sts2.training.rl.treegpo.forcing import ForcedFirstChoiceProvider

    provider = Provider()
    choices = ("ACTION: choose_event_option 0", "ACTION: choose_event_option 1")
    wrapper = ForcedFirstChoiceProvider(provider, choices[1])
    messages = (ChatMessage("system", "战略"), ChatMessage("user", "事件"))

    first = wrapper.chat(messages, temperature=0.8, response_choices=choices)
    second = wrapper.chat(messages, temperature=0.8, response_choices=choices)

    assert first.text == choices[1]
    assert second.text == choices[0]
    assert provider.choices == [(choices[1],), choices]


def test_forced_provider_forwards_thinking_receipt() -> None:
    """首动作分层包装不能把 thinking 收据从 ``False`` 变成未知。

    Returns:
        None: Tree 与 backbone 的生成 profile 可精确比较。
    """
    from play_sts2.training.rl.treegpo.forcing import ForcedFirstChoiceProvider

    provider = Provider()
    provider.thinking_enabled = False

    wrapper = ForcedFirstChoiceProvider(
        provider,
        "ACTION: choose_map_node 0",
    )

    assert wrapper.thinking_enabled is False
