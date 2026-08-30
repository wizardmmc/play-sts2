"""验证 Runtime 把稳定状态闭环为一次真实游戏动作。"""

import importlib
import json
from typing import Any

import httpx
import pytest

from play_sts2.client import GameClient
from play_sts2.harness import ActionParseError
from play_sts2.inference import ChatMessage, ModelReply, OpenAICompatibleProvider


def test_decision_engine_executes_model_action() -> None:
    """将战斗状态依次转换为消息、模型动作和 Mod 请求。

    Raises:
        AssertionError: 闭环遗漏提示词、错误映射动作参数或丢失步骤产物。

    Returns:
        None: 此测试只验证一个成功决策步骤。
    """
    state = _combat_state()
    action_result = {
        "action": "play_card",
        "status": "completed",
        "stable": True,
        "state": {"screen": "COMBAT", "turn": 1},
    }

    def respond_model(request: httpx.Request) -> httpx.Response:
        """校验 Runtime 交给模型的完整消息并返回规范动作。

        Args:
            request (httpx.Request): Provider 发出的模型请求。

        Raises:
            AssertionError: 模型请求缺少正确层级的提示词或可读观测。

        Returns:
            httpx.Response: 要求打出第 0 张牌并选择敌人 1 的回复。
        """
        body = json.loads(request.content)
        assert body["max_tokens"] == 128
        assert body["temperature"] == 0.0
        assert body["structured_outputs"] == {
            "choice": ["ACTION: play_card 0 1", "ACTION: end_turn"]
        }
        assert body["messages"][0]["role"] == "system"
        assert "战斗决策模型" in body["messages"][0]["content"]
        assert (
            "- [0] 破损核心: 战斗开始时生成1个闪电充能球。"
            in (body["messages"][0]["content"])
        )
        assert body["messages"][1]["role"] == "user"
        assert (
            "[0]打击(1费)<目标:任一敌人> 造成6点伤害。"
            in (body["messages"][1]["content"])
        )
        assert body["messages"][1]["content"].endswith(
            "可执行动作:\n- play_card(card_index, target_index)\n- end_turn"
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ACTION: play_card 0 1"}}]},
        )

    def respond_game(request: httpx.Request) -> httpx.Response:
        """校验解析后的动作参数并返回 Mod 动作结果。

        Args:
            request (httpx.Request): GameClient 发出的动作请求。

        Raises:
            AssertionError: Runtime 使用了错误的端点或动作参数。

        Returns:
            httpx.Response: 与真实 Mod 外壳一致的成功响应。
        """
        assert request.method == "POST"
        assert request.url.path == "/action"
        assert json.loads(request.content) == {
            "action": "play_card",
            "card_index": 0,
            "expected_state_revision": 41,
            "target_index": 1,
        }
        return httpx.Response(200, json={"ok": True, "data": action_result})

    runtime = importlib.import_module("play_sts2.runtime")
    with (
        GameClient(
            "http://127.0.0.1:8080",
            transport=httpx.MockTransport(respond_game),
        ) as game,
        OpenAICompatibleProvider(
            "http://127.0.0.1:8900",
            enable_thinking=False,
            transport=httpx.MockTransport(respond_model),
        ) as provider,
    ):
        step = runtime.DecisionEngine(
            game,
            provider,
            constrain_actions=True,
        ).step(state)

    assert tuple(message.role for message in step.messages) == ("system", "user")
    assert step.observation.available_actions == ("play_card", "end_turn")
    assert step.reply.text == "ACTION: play_card 0 1"
    assert step.response_choices == (
        "ACTION: play_card 0 1",
        "ACTION: end_turn",
    )
    assert step.generation_profile == runtime.DecisionGenerationProfile(
        max_tokens=128,
        temperature=0.0,
        max_retries=0,
        thinking_enabled=False,
    )
    assert step.action.name == "play_card"
    assert step.action.parameters == {"card_index": 0, "target_index": 1}
    assert step.action_result == action_result
    state["combat"]["player"]["energy"] = 0
    assert step.before_state["combat"]["player"]["energy"] == 3


@pytest.mark.parametrize(
    "reply_text",
    ["我建议结束回合。", "ACTION: save_and_quit"],
)
def test_decision_engine_does_not_execute_invalid_model_output(
    reply_text: str,
) -> None:
    """模型输出不符合 ACTION 契约时不向游戏发送任何动作。

    Args:
        reply_text (str): 非规范文本或当前决策层不可见的规范动作。

    Returns:
        None: 此测试只验证动作解析失败发生在游戏写入之前。
    """

    def respond_model(_request: httpx.Request) -> httpx.Response:
        """返回结构合法但不符合 Harness 动作契约的文本。

        Args:
            _request (httpx.Request): Provider 发出的模型请求。

        Returns:
            httpx.Response: 含解释性自然语言的模型回复。
        """
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": reply_text}}]},
        )

    def reject_game_request(_request: httpx.Request) -> httpx.Response:
        """在错误路径触发游戏写入时立即让测试失败。

        Args:
            _request (httpx.Request): 不应出现的游戏 HTTP 请求。

        Raises:
            AssertionError: Runtime 在动作解析失败后仍然写入了游戏。

        Returns:
            httpx.Response: 此函数始终在返回前失败。
        """
        raise AssertionError("非法模型输出不得执行游戏动作")

    runtime = importlib.import_module("play_sts2.runtime")
    with (
        GameClient(
            "http://127.0.0.1:8080",
            transport=httpx.MockTransport(reject_game_request),
        ) as game,
        OpenAICompatibleProvider(
            "http://127.0.0.1:8900",
            transport=httpx.MockTransport(respond_model),
        ) as provider,
        pytest.raises(ActionParseError),
    ):
        runtime.DecisionEngine(game, provider).step(_combat_state())


class ReplyQueue:
    """按顺序返回测试预设的模型回复。

    Args:
        replies (list[str]): 每次 ``chat`` 调用依次返回的文本。
    """

    def __init__(self, replies: list[str]) -> None:
        """保存回复队列并初始化请求记录。

        Args:
            replies (list[str]): 每次 ``chat`` 调用依次返回的文本。

        Returns:
            None: 此方法只初始化测试替身。
        """
        self._replies = iter(replies)
        self.requests: list[tuple[ChatMessage, ...]] = []

    def chat(
        self,
        messages: tuple[ChatMessage, ...],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> ModelReply:
        """记录完整消息并返回下一条预设回复。

        Args:
            messages (tuple[ChatMessage, ...]): Runtime 本次发送的完整消息。
            max_tokens (int): 本次生成允许使用的最大输出 token 数。
            temperature (float): 本次生成使用的采样温度。

        Returns:
            ModelReply: 队列中的下一条回复。
        """
        self.requests.append(tuple(messages))
        return ModelReply(next(self._replies))


class RecordingGame:
    """记录 Runtime 实际提交的游戏动作。"""

    def __init__(self) -> None:
        """初始化空动作记录。

        Returns:
            None: 此方法只初始化测试替身。
        """
        self.actions: list[tuple[str, dict[str, int]]] = []

    def execute_action(self, action: str, **parameters: int) -> dict[str, Any]:
        """记录一个动作并返回已完成结果。

        Args:
            action (str): Runtime 提交的动作名称。
            parameters (int): Runtime 提交的动作参数。

        Returns:
            dict[str, Any]: 最小的 Mod 成功结果。
        """
        self.actions.append((action, parameters))
        return {"status": "completed", "stable": True, "state": {}}


class RejectingGame:
    """按指定次数拒绝模型动作，随后完成纠正动作。"""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_action",
        status_code: int = 409,
        rejection_count: int = 1,
    ) -> None:
        """保存 Mod 拒绝响应并初始化动作记录。

        Args:
            message (str): 动作返回的拒绝消息。
            code (str): Mod 错误代码。
            status_code (int): HTTP 状态码。
            rejection_count (int): 返回错误的动作次数。

        Returns:
            None: 此方法只初始化测试替身。
        """
        self._message = message
        self._code = code
        self._status_code = status_code
        self._rejection_count = rejection_count
        self.actions: list[tuple[str, dict[str, int]]] = []

    def execute_action(self, action: str, **parameters: int) -> dict[str, Any]:
        """在指定次数内抛出 HTTP 错误，随后返回成功结果。

        Args:
            action (str): Runtime 提交的动作名称。
            parameters (int): Runtime 提交的动作参数。

        Raises:
            httpx.HTTPStatusError: 当前动作仍在预设拒绝次数内。

        Returns:
            dict[str, Any]: 纠正动作的最小成功结果。
        """
        self.actions.append((action, parameters))
        if len(self.actions) <= self._rejection_count:
            raise _action_error(self._status_code, self._code, self._message)
        return {"status": "completed", "stable": True, "state": {}}


def test_decision_engine_retries_invalid_model_output() -> None:
    """非法输出携带错误说明重试，直到得到一个合法动作。

    Raises:
        AssertionError: 重试次数、消息历史或动作提交时机不符合约定。

    Returns:
        None: 此测试只验证单个决策内的有限重试。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    provider = ReplyQueue(["应该结束回合", "ACTION: end_turn"])
    game = RecordingGame()

    step = runtime.DecisionEngine(game, provider).step(
        _combat_state(),
        max_retries=1,
    )

    assert game.actions == [("end_turn", {"expected_state_revision": 41})]
    assert tuple(reply.text for reply in step.replies) == (
        "应该结束回合",
        "ACTION: end_turn",
    )
    assert len(step.retry_errors) == 1
    assert tuple(message.role for message in provider.requests[1]) == (
        "system",
        "user",
        "user",
    )
    assert "上次动作无效" in provider.requests[1][-1].content
    assert "模型输出必须以 ACTION: 开头" in provider.requests[1][-1].content


def test_decision_engine_rejects_dead_enemy_target_before_mod_request() -> None:
    """目标不在卡牌合法索引中时先反馈模型，不向 Mod 提交必败请求。"""
    runtime = importlib.import_module("play_sts2.runtime")
    provider = ReplyQueue(["ACTION: play_card 0 0", "ACTION: play_card 0 1"])
    game = RecordingGame()
    state = _combat_state()
    state["combat"]["hand"][0]["requires_target"] = True
    state["combat"]["enemies"].insert(
        0,
        {
            "index": 0,
            "name": "已死亡敌人",
            "current_hp": 0,
            "max_hp": 20,
            "is_alive": False,
        },
    )

    step = runtime.DecisionEngine(game, provider).step(state, max_retries=1)

    assert game.actions == [
        (
            "play_card",
            {
                "expected_state_revision": 41,
                "card_index": 0,
                "target_index": 1,
            },
        )
    ]
    assert step.retry_errors == ("target_index 0 不在合法目标 [1]",)
    assert "合法目标 [1]" in provider.requests[1][-1].content


def test_decision_engine_rejects_targeted_card_without_legal_targets() -> None:
    """需要目标但没有合法目标时先反馈模型，不向 Mod 提交必败请求。"""
    runtime = importlib.import_module("play_sts2.runtime")
    provider = ReplyQueue(["ACTION: play_card 0 0", "ACTION: end_turn"])
    game = RecordingGame()
    state = _combat_state()
    state["combat"]["hand"][0]["requires_target"] = True
    state["combat"]["hand"][0]["valid_target_indices"] = []

    step = runtime.DecisionEngine(game, provider).step(state, max_retries=1)

    assert game.actions == [
        ("end_turn", {"expected_state_revision": 41}),
    ]
    assert step.retry_errors == ("卡牌 [0] 当前没有合法目标",)
    assert "当前没有合法目标" in provider.requests[1][-1].content


def test_decision_engine_allows_card_that_does_not_require_target() -> None:
    """无需目标的卡牌不应被本地目标校验误拒绝。"""
    runtime = importlib.import_module("play_sts2.runtime")
    provider = ReplyQueue(["ACTION: play_card 0"])
    game = RecordingGame()
    state = _combat_state()
    state["combat"]["hand"][0]["requires_target"] = False
    state["combat"]["hand"][0]["valid_target_indices"] = []

    step = runtime.DecisionEngine(game, provider).step(state)

    assert game.actions == [
        (
            "play_card",
            {
                "expected_state_revision": 41,
                "card_index": 0,
            },
        )
    ]
    assert step.retry_errors == ()


@pytest.mark.parametrize(
    ("state", "replies", "code", "message", "expected_actions"),
    [
        (
            "combat",
            ["ACTION: play_card 0 1", "ACTION: end_turn"],
            "invalid_action",
            "Card cannot be played in the current state.",
            [
                (
                    "play_card",
                    {
                        "expected_state_revision": 41,
                        "card_index": 0,
                        "target_index": 1,
                    },
                ),
                ("end_turn", {"expected_state_revision": 41}),
            ],
        ),
        (
            "reward",
            ["ACTION: claim_reward 0", "ACTION: proceed"],
            "invalid_action",
            "The selected reward is not claimable in the current state.",
            [
                (
                    "claim_reward",
                    {"expected_state_revision": 42, "option_index": 0},
                ),
                ("proceed", {"expected_state_revision": 42}),
            ],
        ),
        (
            "combat",
            ["ACTION: play_card 0 1", "ACTION: end_turn"],
            "invalid_target",
            "Action is not available in the current state.",
            [
                (
                    "play_card",
                    {
                        "expected_state_revision": 41,
                        "card_index": 0,
                        "target_index": 1,
                    },
                ),
                ("end_turn", {"expected_state_revision": 41}),
            ],
        ),
    ],
    ids=("unplayable_card", "unclaimable_reward", "invalid_target"),
)
def test_decision_engine_lets_model_correct_rejected_action(
    state: str,
    replies: list[str],
    code: str,
    message: str,
    expected_actions: list[tuple[str, dict[str, int]]],
) -> None:
    """把 Mod 的业务拒绝反馈给模型，并执行模型自行选择的纠正动作。

    Args:
        state (str): 动作被拒绝时的模型观测类型。
        replies (list[str]): 模型首次动作和自行纠正动作。
        code (str): Mod 返回的语义错误代码。
        message (str): 真实运行中观测到的 Mod 业务拒绝消息。
        expected_actions (list[tuple[str, dict[str, int]]]): 模型依次选择的动作。

    Raises:
        AssertionError: Harness 代打、直接退出或没有反馈准确拒绝原因。

    Returns:
        None: 此测试只验证有限的业务动作纠错。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    provider = ReplyQueue(replies)
    game = RejectingGame(message, code=code)
    initial_state = _combat_state() if state == "combat" else _reward_state()

    step = runtime.DecisionEngine(game, provider).step(
        initial_state,
        max_retries=1,
    )

    assert game.actions == expected_actions
    assert step.action.name == expected_actions[-1][0]
    assert step.retry_errors == (f"Mod 拒绝动作: {message}",)
    assert "上次动作无效" in provider.requests[1][-1].content
    assert message in provider.requests[1][-1].content


def test_decision_engine_stops_after_rejected_action_retry_limit() -> None:
    """模型持续选择被 Mod 拒绝的动作时有限停止且不代打。

    Raises:
        AssertionError: Runtime 超出重试预算、吞掉原因或执行替代动作。

    Returns:
        None: 此测试验证业务拒绝与文本错误共享同一重试预算。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    message = "Card cannot be played in the current state."
    provider = ReplyQueue(["ACTION: play_card 0 1"] * 2)
    game = RejectingGame(message, rejection_count=2)

    with pytest.raises(runtime.DecisionRetriesExhausted) as raised:
        runtime.DecisionEngine(game, provider).step(
            _combat_state(),
            max_retries=1,
        )

    assert game.actions == [
        (
            "play_card",
            {
                "expected_state_revision": 41,
                "card_index": 0,
                "target_index": 1,
            },
        ),
        (
            "play_card",
            {
                "expected_state_revision": 41,
                "card_index": 0,
                "target_index": 1,
            },
        ),
    ]
    assert raised.value.errors == (
        f"Mod 拒绝动作: {message}",
        f"Mod 拒绝动作: {message}",
    )


@pytest.mark.parametrize(
    ("status_code", "code", "message"),
    [
        (409, "invalid_action", "Action is not available in the current state."),
        (409, "server_error", "Unexpected Mod failure."),
        (500, "invalid_action", "Card cannot be played in the current state."),
    ],
)
def test_decision_engine_propagates_non_semantic_http_errors(
    status_code: int,
    code: str,
    message: str,
) -> None:
    """瞬时窗口和非业务 HTTP 错误不在单步引擎中被吞掉。

    Args:
        status_code (int): Mod 返回的 HTTP 状态码。
        code (str): Mod 返回的错误代码。
        message (str): Mod 返回的错误消息。

    Raises:
        AssertionError: 单步引擎错误地把系统错误反馈给模型重试。

    Returns:
        None: 此测试只验证 HTTP 错误分类边界。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    provider = ReplyQueue(["ACTION: play_card 0 1"])
    game = RejectingGame(message, code=code, status_code=status_code)

    with pytest.raises(httpx.HTTPStatusError):
        runtime.DecisionEngine(game, provider).step(
            _combat_state(),
            max_retries=1,
        )

    assert len(provider.requests) == 1
    assert len(game.actions) == 1


def test_decision_engine_stops_after_retry_limit() -> None:
    """模型持续返回非法文本时在有限次数后停止且不写入游戏。

    Raises:
        AssertionError: Runtime 多试、少试或执行了非法动作。

    Returns:
        None: 此测试只验证重试上限。
    """
    runtime = importlib.import_module("play_sts2.runtime")
    provider = ReplyQueue(["无效一", "无效二"])
    game = RecordingGame()

    with pytest.raises(runtime.DecisionRetriesExhausted) as raised:
        runtime.DecisionEngine(game, provider).step(
            _combat_state(),
            max_retries=1,
        )

    assert tuple(reply.text for reply in raised.value.replies) == (
        "无效一",
        "无效二",
    )
    assert len(provider.requests) == 2
    assert game.actions == []


@pytest.mark.parametrize(
    ("status_code", "code", "message", "expected"),
    [
        (
            409,
            "invalid_action",
            "Action is not available in the current state.",
            True,
        ),
        (
            409,
            "invalid_target",
            "Action is not available in the current state.",
            False,
        ),
        (409, "invalid_action", "Action parameters are invalid.", False),
        (
            400,
            "invalid_action",
            "Action is not available in the current state.",
            False,
        ),
    ],
)
def test_action_window_conflict_matches_only_observed_mod_error(
    status_code: int,
    code: str,
    message: str,
    expected: bool,
) -> None:
    """只把实际观测到的 Mod 输入窗口错误视为可重试冲突。

    Args:
        status_code (int): Mod 返回的 HTTP 状态码。
        code (str): Mod 错误代码。
        message (str): Mod 错误消息。
        expected (bool): 该响应是否应进入瞬时重试。

    Raises:
        AssertionError: 错误匹配边界过宽或拒绝了真实冲突。

    Returns:
        None: 此测试只验证错误响应分类。
    """
    decision = importlib.import_module("play_sts2.runtime.decision")

    assert (
        decision.is_action_window_conflict(_action_error(status_code, code, message))
        is expected
    )


def _combat_state() -> dict[str, Any]:
    """构造与真实 Mod 字段一致的最小稳定战斗状态。

    Returns:
        dict[str, Any]: 可以打出第 0 张牌攻击敌人 1 的战斗状态。
    """
    return {
        "state_revision": 41,
        "screen": "COMBAT",
        "in_combat": True,
        "turn": 1,
        "available_actions": ["save_and_quit", "play_card", "end_turn"],
        "run": {
            "character_name": "故障机器人",
            "ascension": 0,
            "act_id": "0",
            "floor": 2,
            "current_hp": 70,
            "max_hp": 75,
            "gold": 99,
            "relics": [
                {
                    "index": 0,
                    "name": "破损核心",
                    "description": "战斗开始时生成1个闪电充能球。",
                }
            ],
            "potions": [],
        },
        "combat": {
            "player": {
                "current_hp": 70,
                "max_hp": 75,
                "block": 0,
                "energy": 3,
                "stars": 0,
                "focus": 0,
                "orb_capacity": 3,
                "orbs": [],
            },
            "enemies": [
                {
                    "index": 1,
                    "name": "邪教徒",
                    "current_hp": 48,
                    "max_hp": 48,
                    "block": 0,
                    "intents": [{"intent_type": "Buff", "label": None}],
                    "powers": [],
                }
            ],
            "hand": [
                {
                    "index": 0,
                    "name": "打击",
                    "upgraded": False,
                    "energy_cost": 1,
                    "star_cost": 0,
                    "resolved_rules_text": "造成6点伤害。",
                    "target_type": "AnyEnemy",
                    "playable": True,
                    "valid_target_indices": [1],
                }
            ],
            "draw_count": 5,
            "discard_count": 0,
        },
    }


def _reward_state() -> dict[str, Any]:
    """构造包含不可领取药水和真实出口的奖励状态。

    Returns:
        dict[str, Any]: 可以验证奖励业务拒绝纠错的战略状态。
    """
    return {
        "state_revision": 42,
        "screen": "REWARD",
        "in_combat": False,
        "available_actions": ["claim_reward", "proceed"],
        "run": {
            "character_name": "故障机器人",
            "ascension": 0,
            "act_id": "0",
            "floor": 8,
            "current_hp": 27,
            "max_hp": 75,
            "gold": 38,
            "potions": [
                {"index": 0, "name": "集中药水", "occupied": True},
                {"index": 1, "name": "力量药水", "occupied": True},
                {"index": 2, "name": "速度药水", "occupied": True},
            ],
        },
        "reward": {
            "rewards": [
                {
                    "index": 0,
                    "name": "流动铜液",
                    "reward_type": "Potion",
                    "claimable": False,
                },
                {
                    "index": 1,
                    "name": "卡牌奖励",
                    "reward_type": "Card",
                    "claimable": True,
                },
            ]
        },
    }


def _action_error(
    status_code: int,
    code: str,
    message: str,
) -> httpx.HTTPStatusError:
    """构造用于分类测试的 Mod HTTP 错误。

    Args:
        status_code (int): HTTP 状态码。
        code (str): Mod 错误代码。
        message (str): Mod 错误消息。

    Returns:
        httpx.HTTPStatusError: 包含指定错误外壳的异常。
    """
    request = httpx.Request("POST", "http://127.0.0.1:8080/action")
    response = httpx.Response(
        status_code,
        request=request,
        json={"error": {"code": code, "message": message}},
    )
    return httpx.HTTPStatusError(
        f"{status_code} response",
        request=request,
        response=response,
    )
