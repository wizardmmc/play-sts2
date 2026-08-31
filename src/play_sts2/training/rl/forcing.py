"""提供只在分层 Tree 根节点强制一个动作的通用 provider。"""

from collections.abc import Sequence

from ...inference import ChatMessage, DecisionProvider, ModelReply


class ForcedFirstChoiceProvider:
    """首次请求使用 singleton proposal，之后恢复完整冻结策略。"""

    def __init__(self, provider: DecisionProvider, forced_choice: str) -> None:
        """保存底层远程 provider 和待强制根动作。

        Args:
            provider (DecisionProvider): 已连接到冻结战略 residual 的 provider。
            forced_choice (str): 必须属于首次完整 xgrammar 动作域的规范动作。

        Raises:
            ValueError: 强制动作为空。
        """
        if not forced_choice:
            raise ValueError("terminal Tree 强制根动作不能为空")
        self._provider = provider
        self._forced_choice = forced_choice
        self._used = False

    @property
    def thinking_enabled(self) -> bool | None:
        """返回底层冻结 provider 实际使用的 thinking 配置。

        Returns:
            bool | None: 与未包裹 provider 完全相同的生成参数收据。
        """
        return getattr(self._provider, "thinking_enabled", None)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        response_choices: Sequence[str] | None = None,
    ) -> ModelReply:
        """首次只转发被分层分配的动作，后续原样转发完整候选。

        Args:
            messages (Sequence[ChatMessage]): 当前 stateless 消息。
            max_tokens (int): 回复 token 预算。
            temperature (float): 冻结采样温度。
            response_choices (Sequence[str] | None): 当前完整合法动作行。

        Raises:
            ValueError: 首次请求没有完整候选或强制动作不合法。

        Returns:
            ModelReply: 底层 vLLM 生成结果。
        """
        choices = tuple(response_choices) if response_choices is not None else None
        forwarded = choices
        if not self._used:
            if not choices or self._forced_choice not in choices:
                raise ValueError("terminal Tree 强制根动作不属于入口完整动作域")
            forwarded = (self._forced_choice,)
            self._used = True
        return self._provider.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_choices=forwarded,
        )
