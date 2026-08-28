"""访问 STS2 Agent Mod 暴露的本地 HTTP API。"""

import json
import math
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from threading import Event
from types import TracebackType
from typing import Any, Self

import httpx

_READ_TIMEOUT_SECONDS = 5.0
_ACTION_TIMEOUT_SECONDS = 30.0


class ProtocolError(RuntimeError):
    """表示 Mod 响应违反预期的 HTTP 协议。"""


@dataclass(frozen=True, slots=True)
class Health:
    """表示 Mod 返回的版本与就绪信息。

    Args:
        service (str): Mod HTTP 服务的稳定名称。
        mod_version (str): 当前安装的 Mod 版本。
        protocol_version (str): HTTP 响应协议版本。
        game_version (str): 当前运行的《杀戮尖塔 2》版本。
        status (str): Mod 当前的就绪状态。
    """

    service: str
    mod_version: str
    protocol_version: str
    game_version: str
    status: str


@dataclass(frozen=True, slots=True)
class AvailableAction:
    """表示 Mod 当前允许执行的一个动作。

    Args:
        name (str): 动作的稳定名称。
        requires_target (bool): 动作是否需要指定目标索引。
        requires_index (bool): 动作是否需要指定选项或卡牌索引。
    """

    name: str
    requires_target: bool
    requires_index: bool


@dataclass(frozen=True, slots=True)
class AvailableActions:
    """表示当前屏幕及其全部合法动作。

    Args:
        screen (str): 动作所属的当前游戏屏幕。
        actions (tuple[AvailableAction, ...]): 当前屏幕允许执行的动作。
    """

    screen: str
    actions: tuple[AvailableAction, ...]


class GameClient:
    """持有与单个游戏实例通信的 HTTP 会话。"""

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.BaseTransport | None = None,
        action_timeout: float = _ACTION_TIMEOUT_SECONDS,
    ) -> None:
        """初始化连接单个 Mod HTTP 端点的客户端。

        Args:
            base_url (str): 正在运行的 Mod HTTP 服务根地址。
            transport (httpx.BaseTransport | None): 可选的 HTTP 传输实现，
                主要用于在测试中注入确定性响应。
            action_timeout (float): 动作请求允许等待的秒数。

        Returns:
            None: 此方法在当前实例上完成初始化。
        """
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            timeout=_READ_TIMEOUT_SECONDS,
            transport=transport,
        )
        self._action_timeout = action_timeout

    @property
    def action_timeout(self) -> float:
        """读取单次游戏动作允许占用的总秒数。

        Returns:
            float: 动作 HTTP 请求及后续状态等待共享的超时配置。
        """
        return self._action_timeout

    def __enter__(self) -> Self:
        """进入持有底层 HTTP 会话的上下文。

        Returns:
            Self: 当前客户端实例。
        """
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """离开上下文时关闭 HTTP 会话。

        Args:
            _exc_type (type[BaseException] | None): 上下文执行失败时的异常类型。
            _exc (BaseException | None): 上下文执行失败时的异常实例。
            _traceback (TracebackType | None): 上下文执行失败时的异常调用栈。

        Returns:
            None: 此方法释放资源，但不屏蔽上下文中的异常。
        """
        self.close()

    def close(self) -> None:
        """释放客户端持有的网络资源。

        Returns:
            None: 此方法关闭底层 HTTP 会话。
        """
        self._http.close()

    def health(self) -> Health:
        """读取并校验 Mod 的健康检查协议。

        Raises:
            httpx.HTTPStatusError: Mod 返回非成功 HTTP 状态码。
            ProtocolError: 响应体不是有效 JSON，或缺少必要的健康检查字段。

        Returns:
            Health: 校验通过的 Mod 与游戏版本信息。
        """
        data = self._request_data("/health")
        if not isinstance(data, Mapping):
            raise ProtocolError("invalid /health response")

        return Health(
            service=_required_text(data, "service", "/health"),
            mod_version=_required_text(data, "mod_version", "/health"),
            protocol_version=_required_text(data, "protocol_version", "/health"),
            game_version=_required_text(data, "game_version", "/health"),
            status=_required_text(data, "status", "/health"),
        )

    def state(self) -> dict[str, Any]:
        """读取 Mod 返回的完整游戏状态。

        此处只校验端点返回 JSON 对象，不解释战斗、地图等业务字段，
        以免客户端与频繁变化的游戏状态结构过度耦合。

        Raises:
            httpx.HTTPStatusError: Mod 返回非成功 HTTP 状态码。
            ProtocolError: 响应体违反通用协议，或状态数据不是 JSON 对象。

        Returns:
            dict[str, Any]: Mod 返回的完整原始状态对象。
        """
        data = self._request_data("/state")
        if not isinstance(data, Mapping):
            raise ProtocolError("invalid /state response")
        return dict(data)

    def data_collection(self, collection: str) -> list[dict[str, Any]]:
        """读取 Mod 从当前游戏实例导出的实体集合。

        Args:
            collection (str): ``cards``、``relics`` 等游戏数据集合名称。

        Raises:
            httpx.HTTPStatusError: Mod 不支持该集合或返回其他非成功状态码。
            ProtocolError: 响应体不是只包含 JSON 对象的数组。

        Returns:
            list[dict[str, Any]]: 保留 Mod 原始字段的游戏实体列表。
        """
        path = f"/data/{collection}"
        data = self._request_data(path)
        if not isinstance(data, list) or any(
            not isinstance(entity, Mapping) for entity in data
        ):
            raise ProtocolError(f"invalid {path} response")
        return [dict(entity) for entity in data]

    def available_actions(self) -> AvailableActions:
        """读取当前屏幕允许执行的动作。

        Raises:
            httpx.HTTPStatusError: Mod 返回非成功 HTTP 状态码。
            ProtocolError: 响应缺少屏幕、动作列表或动作描述字段。

        Returns:
            AvailableActions: 当前屏幕及其不可变动作集合。
        """
        path = "/actions/available"
        data = self._request_data(path)
        if not isinstance(data, Mapping):
            raise ProtocolError(f"invalid {path} response")

        raw_actions = data.get("actions")
        if not isinstance(raw_actions, list):
            raise ProtocolError(f"invalid {path} response")

        return AvailableActions(
            screen=_required_text(data, "screen", path),
            actions=tuple(_parse_action(action, path) for action in raw_actions),
        )

    def iter_events(self, stop_event: Event) -> Iterator[dict[str, Any]]:
        """持续读取 Mod 的只读 SSE 事件流。

        Args:
            stop_event (Event): 外部用于结束事件消费的线程事件。

        Raises:
            httpx.HTTPStatusError: Mod 拒绝事件流请求。
            httpx.HTTPError: 读取事件流时连接中断。
            ProtocolError: SSE data 不是有效的 JSON 对象。

        Yields:
            dict[str, Any]: Mod 发布的原始事件对象。
        """
        if stop_event.is_set():
            return

        path = "/events/stream"
        with self._http.stream(
            "GET",
            path,
            headers={"Accept": "text/event-stream"},
            timeout=None,
        ) as response:
            response.raise_for_status()
            data_lines: list[str] = []
            for line in response.iter_lines():
                if stop_event.is_set():
                    return
                if not line:
                    if not data_lines:
                        continue
                    raw_data = "\n".join(data_lines)
                    data_lines.clear()
                    try:
                        payload = json.loads(raw_data)
                    except json.JSONDecodeError as exc:
                        raise ProtocolError(f"invalid {path} response") from exc
                    if not isinstance(payload, Mapping):
                        raise ProtocolError(f"invalid {path} response")
                    yield dict(payload)
                    continue
                if line.startswith(":") or not line.startswith("data:"):
                    continue
                value = line[5:]
                data_lines.append(value.removeprefix(" "))

    def wait_for_state(
        self,
        *,
        after_revision: int,
        timeout: float,
    ) -> dict[str, Any]:
        """阻塞等待 SSE 交付一份更新的完整状态。

        该方法只建立一条事件流连接，不按固定时间间隔请求 ``/state``。
        ``stream_ready`` 携带的当前快照也参与比较，因此连接建立前已经发生的
        状态变化不会丢失。

        Args:
            after_revision (int): 调用方已经处理的最后状态 revision。
            timeout (float): 等待更新事件的最长秒数。

        Raises:
            ValueError: revision 或超时参数无效。
            TimeoutError: 事件流在新 revision 到达前结束或超时。
            httpx.HTTPStatusError: Mod 拒绝事件流请求。
            ProtocolError: SSE 或其中的状态 revision 违反协议。

        Returns:
            dict[str, Any]: revision 严格大于 ``after_revision`` 的完整状态。
        """
        if (
            isinstance(after_revision, bool)
            or not isinstance(after_revision, int)
            or after_revision < 0
        ):
            raise ValueError("after_revision must be a non-negative integer")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        path = "/events/stream"
        deadline = time.monotonic() + timeout
        timeout_ms = max(1, math.ceil(timeout * 1000))
        try:
            with self._http.stream(
                "GET",
                path,
                params={"timeout_ms": timeout_ms},
                headers={"Accept": "text/event-stream"},
                timeout=httpx.Timeout(timeout),
            ) as response:
                response.raise_for_status()
                data_lines: list[str] = []
                for line in response.iter_lines():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("timed out waiting for a newer game state")
                    if not line:
                        if not data_lines:
                            continue
                        raw_data = "\n".join(data_lines)
                        data_lines.clear()
                        try:
                            payload = json.loads(raw_data)
                        except json.JSONDecodeError as exc:
                            raise ProtocolError(f"invalid {path} response") from exc
                        if not isinstance(payload, Mapping):
                            raise ProtocolError(f"invalid {path} response")
                        data = payload.get("data")
                        state = data.get("state") if isinstance(data, Mapping) else None
                        if not isinstance(state, Mapping):
                            continue
                        revision = state.get("state_revision")
                        if isinstance(revision, bool) or not isinstance(revision, int):
                            raise ProtocolError(f"invalid {path} response")
                        if revision > after_revision:
                            return dict(state)
                        continue
                    if line.startswith(":") or not line.startswith("data:"):
                        continue
                    value = line[5:]
                    data_lines.append(value.removeprefix(" "))
        except httpx.TimeoutException as exc:
            raise TimeoutError("timed out waiting for a newer game state") from exc

        raise TimeoutError("event stream ended before a newer game state arrived")

    def execute_action(self, action: str, **parameters: Any) -> dict[str, Any]:
        """执行一个 Mod 当前允许的游戏动作。

        Args:
            action (str): 待执行动作的稳定名称。
            parameters (Any): 动作需要的额外协议参数，例如 ``option_index``。

        Raises:
            httpx.HTTPStatusError: Mod 拒绝动作或返回其他非成功状态码。
            ProtocolError: 响应体违反通用协议，或动作结果不是 JSON 对象。

        Returns:
            dict[str, Any]: Mod 返回的完整原始动作结果对象。
        """
        path = "/action"
        data = self._request_data(
            path,
            method="POST",
            body={"action": action, **parameters},
            timeout=self._action_timeout,
        )
        if not isinstance(data, Mapping):
            raise ProtocolError(f"invalid {path} response")
        return dict(data)

    def _request_data(
        self,
        path: str,
        *,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        timeout: float = _READ_TIMEOUT_SECONDS,
    ) -> object:
        """请求一个 Mod 端点并解析通用响应外壳。

        Args:
            path (str): 相对于 Mod 服务根地址的端点路径。
            method (str): 待使用的 HTTP 请求方法。
            body (dict[str, Any] | None): 可选的 JSON 请求体。
            timeout (float): 当前请求允许等待的秒数。

        Raises:
            httpx.HTTPStatusError: Mod 返回非成功 HTTP 状态码。
            ProtocolError: 响应不是有效 JSON，或不符合 ``{ok, data}`` 外壳。

        Returns:
            object: 尚未做端点专属校验的 ``data`` 字段。
        """
        response = self._http.request(method, path, json=body, timeout=timeout)
        response.raise_for_status()

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProtocolError(f"invalid {path} response") from exc

        if (
            not isinstance(payload, Mapping)
            or payload.get("ok") is not True
            or "data" not in payload
        ):
            raise ProtocolError(f"invalid {path} response")

        return payload["data"]


def _parse_action(data: object, path: str) -> AvailableAction:
    """解析一个动作描述对象。

    Args:
        data (object): 尚未校验的动作描述值。
        path (str): 动作描述所属的端点路径。

    Raises:
        ProtocolError: 动作描述不是对象或缺少必需字段。

    Returns:
        AvailableAction: 校验通过的不可变动作描述。
    """
    if not isinstance(data, Mapping):
        raise ProtocolError(f"invalid {path} response")

    return AvailableAction(
        name=_required_text(data, "name", path),
        requires_target=_required_bool(data, "requires_target", path),
        requires_index=_required_bool(data, "requires_index", path),
    )


def _required_text(
    data: Mapping[object, object],
    field: str,
    path: str,
) -> str:
    """从协议对象中读取一个必需的字符串字段。

    Args:
        data (Mapping[object, object]): 解码后的协议对象。
        field (str): 必需字段的名称。
        path (str): 协议对象所属的端点路径。

    Raises:
        ProtocolError: 字段缺失或字段值不是字符串。

    Returns:
        str: 校验通过的字段值。
    """
    value = data.get(field)
    if not isinstance(value, str):
        raise ProtocolError(f"invalid {path} response")
    return value


def _required_bool(
    data: Mapping[object, object],
    field: str,
    path: str,
) -> bool:
    """从协议对象中读取一个必需的布尔字段。

    Args:
        data (Mapping[object, object]): 解码后的协议对象。
        field (str): 必需字段的名称。
        path (str): 协议对象所属的端点路径。

    Raises:
        ProtocolError: 字段缺失或字段值不是布尔值。

    Returns:
        bool: 校验通过的字段值。
    """
    value = data.get(field)
    if not isinstance(value, bool):
        raise ProtocolError(f"invalid {path} response")
    return value
