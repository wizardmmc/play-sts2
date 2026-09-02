"""保存、装载并核对原生战略整局 checkpoint。"""

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import GameClient
from .harness import (
    HarnessAction,
    build_observation,
    legal_action_lines,
    model_actions,
)
from .run_start import resume_run

_RUN_SAVE_RELATIVE = Path(
    "Library/Application Support/SlayTheSpire2/"
    "default/1/modded/profile1/saves/current_run.save"
)
_NATIVE_CHECKPOINT_SCREENS = frozenset({"MAP", "REWARD", "SHOP", "REST", "EVENT"})
_DERIVED_CHECKPOINT_SCREENS = frozenset({"CARD_SELECTION"})


class CheckpointError(RuntimeError):
    """表示当前状态或文件不能形成可信的战略 checkpoint。"""


class CheckpointMismatchError(CheckpointError):
    """表示恢复入口与捕获时的可见或隐藏状态不一致。"""


@dataclass(frozen=True, slots=True)
class CheckpointEntry:
    """记录 checkpoint 入口处的玩家观测与隐藏环境审计。

    Args:
        screen (str): 原生保存前的战略页面。
        policy_layer (str): Harness 决策层名称。
        policy_text (str): 模型在该入口实际看到的玩家可见文本。
        legal_actions (tuple[str, ...]): 当前真实可执行的规范动作行。
        option_ids (tuple[str, ...]): 页面候选项的可读稳定身份。
        resume_actions (tuple[HarnessAction, ...]): 原生 continue 后重开子页面的动作。
        audit (dict[str, Any]): 不进入模型输入的 RNG、房间和牌堆审计。
    """

    screen: str
    policy_layer: str
    policy_text: str
    legal_actions: tuple[str, ...]
    option_ids: tuple[str, ...]
    resume_actions: tuple[HarnessAction, ...]
    audit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StrategicCheckpoint:
    """表示一个已经发布且不可覆盖的原生整局 checkpoint。

    Args:
        root (Path): checkpoint 目录。
        save_path (Path): 只读 ``current_run.save`` 副本。
        entry_path (Path): 玩家入口与隐藏审计元数据。
        game_version (str): 捕获时的游戏版本。
        mod_version (str): 捕获时的 Agent Mod 版本。
        entry (CheckpointEntry): 捕获时的规范入口。
    """

    root: Path
    save_path: Path
    entry_path: Path
    game_version: str
    mod_version: str
    entry: CheckpointEntry


def run_save_path(home: Path) -> Path:
    """返回隔离 HOME 中原生单人整局存档的精确路径。

    Args:
        home (Path): 当前游戏实例的独占 HOME。

    Returns:
        Path: 非 Steam、modded profile 1 的 ``current_run.save``。
    """
    return home / _RUN_SAVE_RELATIVE


def capture_strategic_checkpoint(
    client: GameClient,
    *,
    home: Path,
    destination: Path,
) -> StrategicCheckpoint:
    """在稳定战略页面原生存退，并发布不可覆盖的 checkpoint。

    Args:
        client (GameClient): 连接源游戏实例的客户端。
        home (Path): 源游戏实例的独占 HOME。
        destination (Path): 尚不存在的 checkpoint 目录。

    Raises:
        FileExistsError: 目标目录已经存在。
        CheckpointError: 游戏版本、入口、存退结果或原生文件不符合契约。
        OSError: 无法复制原生存档或写入入口元数据。

    Returns:
        StrategicCheckpoint: 可复制到其他隔离 HOME 的 checkpoint。
    """
    if destination.exists():
        raise FileExistsError(f"checkpoint 已存在: {destination}")

    state = client.state()
    screen = str(state.get("screen") or "")
    if screen in _DERIVED_CHECKPOINT_SCREENS:
        raise CheckpointError("CARD_SELECTION 必须从可恢复父 REWARD checkpoint 派生")
    if screen not in _NATIVE_CHECKPOINT_SCREENS or state.get("in_combat") is True:
        raise CheckpointError(f"当前不是可保存的战略 checkpoint: {screen}")

    health = client.health()
    if health.game_version != "v0.111.0":
        raise CheckpointError(
            f"checkpoint 游戏版本必须为 v0.111.0: {health.game_version}"
        )
    entry = observe_checkpoint_entry(client, state)
    revision = state.get("state_revision")
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise CheckpointError("checkpoint 入口缺少有效 state_revision")

    result = client.execute_action(
        "save_and_quit",
        expected_state_revision=revision,
    )
    result_state = result.get("state")
    if (
        result.get("stable") is not True
        or not isinstance(result_state, Mapping)
        or result_state.get("screen") != "MAIN_MENU"
        or "continue_run" not in (result_state.get("available_actions") or [])
    ):
        raise CheckpointError("原生存退没有稳定返回可续局主菜单")

    source_save = run_save_path(home)
    if not source_save.is_file() or source_save.stat().st_size == 0:
        raise CheckpointError(f"原生存档不存在或为空: {source_save}")

    return _publish_checkpoint(
        destination=destination,
        source_save=source_save,
        game_version=health.game_version,
        mod_version=health.mod_version,
        entry=entry,
    )


def derive_strategic_checkpoint(
    client: GameClient,
    *,
    base: StrategicCheckpoint,
    destination: Path,
    resume_actions: Sequence[HarnessAction],
) -> StrategicCheckpoint:
    """用可恢复父存档锚定原生存档不直接保留的战略子页面。

    Args:
        client (GameClient): 当前已经到达目标子页面的源游戏客户端。
        base (StrategicCheckpoint): 执行恢复动作前的原生父 checkpoint。
        destination (Path): 尚不存在的派生 checkpoint 目录。
        resume_actions (Sequence[HarnessAction]): 从父入口打开目标子页面的动作。

    Raises:
        FileExistsError: 目标目录已经存在。
        CheckpointError: 目标不是支持的战略页面，或运行时身份与父 checkpoint 不同。

    Returns:
        StrategicCheckpoint: 复用父原生存档、入口改为当前子页面的 checkpoint。
    """
    if destination.exists():
        raise FileExistsError(f"checkpoint 已存在: {destination}")
    if (
        base.entry.screen != "REWARD"
        or len(resume_actions) != 1
        or resume_actions[0].name != "claim_reward"
        or not isinstance(resume_actions[0].parameters.get("option_index"), int)
    ):
        raise CheckpointError(
            "CARD_SELECTION 派生必须来自父 REWARD 和一条 claim_reward 动作"
        )
    state = client.state()
    screen = str(state.get("screen") or "")
    if screen not in _DERIVED_CHECKPOINT_SCREENS or state.get("in_combat") is True:
        raise CheckpointError(f"当前不是可派生的战略 checkpoint: {screen}")
    health = client.health()
    if (
        health.game_version != base.game_version
        or health.mod_version != base.mod_version
    ):
        raise CheckpointError("派生 checkpoint 的游戏或 Mod 身份与父入口不同")
    entry = observe_checkpoint_entry(
        client,
        state,
        resume_actions=resume_actions,
    )
    return _publish_checkpoint(
        destination=destination,
        source_save=base.save_path,
        game_version=base.game_version,
        mod_version=base.mod_version,
        entry=entry,
    )


def load_strategic_checkpoint(root: Path) -> StrategicCheckpoint:
    """读取一个已发布的原生战略 checkpoint。

    Args:
        root (Path): 含原生存档和入口元数据的目录。

    Raises:
        CheckpointError: 文件缺失或元数据不是已知结构。
        OSError: 无法读取 checkpoint 文件。

    Returns:
        StrategicCheckpoint: 从磁盘恢复的不可变 checkpoint 描述。
    """
    save_path = root / "current_run.save"
    entry_path = root / "entry.json"
    if not save_path.is_file() or not entry_path.is_file():
        raise CheckpointError(f"checkpoint 文件不完整: {root}")
    payload = json.loads(entry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("entry"), Mapping
    ):
        raise CheckpointError(f"checkpoint 元数据无效: {entry_path}")
    entry = _entry_from_dict(payload["entry"])
    game_version = payload.get("game_version")
    mod_version = payload.get("mod_version")
    if not isinstance(game_version, str) or not isinstance(mod_version, str):
        raise CheckpointError(f"checkpoint 运行时身份无效: {entry_path}")
    return StrategicCheckpoint(
        root=root,
        save_path=save_path,
        entry_path=entry_path,
        game_version=game_version,
        mod_version=mod_version,
        entry=entry,
    )


def restore_strategic_checkpoint(
    client: GameClient,
    checkpoint: StrategicCheckpoint,
) -> dict[str, Any]:
    """通过游戏原生 continue 恢复并逐字段核对 checkpoint 入口。

    Args:
        client (GameClient): 连接已装入原生存档的新游戏实例。
        checkpoint (StrategicCheckpoint): 捕获时保存的入口与审计。

    Raises:
        CheckpointMismatchError: 游戏/Mod 身份、观测、选项或隐藏审计不一致。
        RunStartError: 主菜单没有可恢复的原生保存局。

    Returns:
        dict[str, Any]: 已通过完整入口核对的恢复状态。
    """
    health = client.health()
    if health.game_version != checkpoint.game_version:
        raise CheckpointMismatchError("恢复后的 game_version 不一致")
    if health.mod_version != checkpoint.mod_version:
        raise CheckpointMismatchError("恢复后的 mod_version 不一致")

    state = resume_run(client)
    state = _advance_native_chest(client, state, checkpoint.entry.screen)
    state = _advance_native_proceed(client, state, checkpoint.entry.screen)
    for action in checkpoint.entry.resume_actions:
        if state.get("screen") == checkpoint.entry.screen:
            break
        state = _execute_restore_action(client, state, action)
    state = _advance_native_proceed(client, state, checkpoint.entry.screen)
    restored = observe_checkpoint_entry(client, state)
    for field in (
        "screen",
        "policy_layer",
        "policy_text",
        "legal_actions",
        "option_ids",
        "audit",
    ):
        actual = getattr(restored, field)
        expected = getattr(checkpoint.entry, field)
        if actual != expected:
            if field == "screen":
                raise CheckpointMismatchError(
                    f"恢复后的 screen 不一致: expected={expected}, actual={actual}"
                )
            raise CheckpointMismatchError(f"恢复后的 {field} 不一致")
    return state


def _advance_native_chest(
    client: GameClient,
    state: dict[str, Any],
    target_screen: str,
) -> dict[str, Any]:
    """重放宝箱后地图存档中唯一确定的原生宝箱过渡。

    Args:
        client (GameClient): 连接恢复游戏的客户端。
        state (dict[str, Any]): 原生 continue 返回的稳定状态。
        target_screen (str): checkpoint 捕获时的目标页面。

    Returns:
        dict[str, Any]: 进入宝箱后续页面的状态；动作域不唯一时原样返回。
    """
    if target_screen != "MAP" or state.get("screen") != "CHEST":
        return state
    if model_actions(state) != ("open_chest",):
        return state
    state = _execute_restore_action(client, state, HarnessAction("open_chest", {}))
    chest = state.get("chest")
    options = chest.get("relic_options") if isinstance(chest, Mapping) else None
    if (
        state.get("screen") != "CHEST"
        or model_actions(state) != ("choose_treasure_relic",)
        or not isinstance(options, list)
        or len(options) != 1
        or not isinstance(options[0], Mapping)
        or isinstance(options[0].get("index"), bool)
        or not isinstance(options[0].get("index"), int)
    ):
        return state
    state = _execute_restore_action(
        client,
        state,
        HarnessAction(
            "choose_treasure_relic",
            {"option_index": options[0]["index"]},
        ),
    )
    if state.get("screen") == "CHEST" and model_actions(state) == ("proceed",):
        return _execute_restore_action(client, state, HarnessAction("proceed", {}))
    return state


def _advance_native_proceed(
    client: GameClient,
    state: dict[str, Any],
    target_screen: str,
) -> dict[str, Any]:
    """跨过原生存档恢复时出现的单一语义 Proceed 页面。

    Args:
        client (GameClient): 连接恢复游戏的客户端。
        state (dict[str, Any]): 原生 continue 返回的稳定状态。
        target_screen (str): checkpoint 捕获时的目标页面。

    Returns:
        dict[str, Any]: 进入下一页面后的状态；无需或不能自动继续时原样返回。
    """
    if state.get("screen") == target_screen:
        return state
    actions = model_actions(state)
    if actions == ("proceed",):
        return _execute_restore_action(client, state, HarnessAction("proceed", {}))
    event = state.get("event")
    options = event.get("options") if isinstance(event, Mapping) else None
    if (
        actions == ("choose_event_option",)
        and isinstance(options, list)
        and len(options) == 1
        and isinstance(options[0], Mapping)
        and options[0].get("is_proceed") is True
    ):
        return _execute_restore_action(
            client,
            state,
            HarnessAction(
                "choose_event_option",
                {"option_index": options[0].get("index", 0)},
            ),
        )
    return state


def _execute_restore_action(
    client: GameClient,
    state: Mapping[str, Any],
    action: HarnessAction,
) -> dict[str, Any]:
    """执行一条已知页面恢复动作并取得随后的稳定状态。

    Args:
        client (GameClient): 连接恢复游戏的客户端。
        state (Mapping[str, Any]): 动作前稳定状态。
        action (HarnessAction): 语义 Proceed 或调用方记录的页面打开动作。

    Raises:
        CheckpointMismatchError: 状态缺少 revision 或动作结果没有状态。

    Returns:
        dict[str, Any]: 动作完成后的稳定状态。
    """
    revision = state.get("state_revision")
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise CheckpointMismatchError("恢复动作缺少有效 state_revision")
    result = client.execute_action(
        action.name,
        expected_state_revision=revision,
        **action.parameters,
    )
    next_state = result.get("state")
    if not isinstance(next_state, Mapping):
        raise CheckpointMismatchError(f"恢复动作没有返回状态: {action.name}")
    if result.get("stable") is True:
        return dict(next_state)
    return client.wait_for_state(
        after_revision=revision,
        timeout=client.action_timeout,
    )


def observe_checkpoint_entry(
    client: GameClient,
    state: Mapping[str, Any],
    *,
    resume_actions: Sequence[HarnessAction] = (),
) -> CheckpointEntry:
    """同时采集玩家可见入口和独立隐藏环境审计。

    Args:
        client (GameClient): 连接当前游戏实例的客户端。
        state (Mapping[str, Any]): 与审计同一时刻的稳定游戏状态。
        resume_actions (Sequence[HarnessAction]): 恢复当前子页面所需的已知动作。

    Returns:
        CheckpointEntry: 不使用状态哈希的完整入口比较对象。
    """
    observation = build_observation(state)
    return CheckpointEntry(
        screen=str(state.get("screen") or ""),
        policy_layer=observation.layer.value,
        policy_text=observation.text,
        legal_actions=legal_action_lines(state),
        option_ids=checkpoint_option_ids(state),
        resume_actions=tuple(resume_actions),
        audit=client.checkpoint_audit(),
    )


def checkpoint_option_ids(state: Mapping[str, Any]) -> tuple[str, ...]:
    """提取六类战略页面的可读稳定候选身份。

    Args:
        state (Mapping[str, Any]): 当前完整游戏状态。

    Returns:
        tuple[str, ...]: 保留页面顺序且不依赖摘要哈希的候选身份。
    """
    screen = str(state.get("screen") or "")
    if screen == "MAP":
        map_state = state.get("map") or {}
        return tuple(
            f"map:{item.get('row')}:{item.get('col')}"
            for item in map_state.get("available_nodes") or []
        )
    if screen == "REWARD":
        reward = state.get("reward") or {}
        return tuple(
            f"reward:{item.get('index')}:{item.get('reward_type')}:{item.get('name')}"
            for item in reward.get("rewards") or []
            if item.get("claimable") is not False
        )
    if screen == "CARD_SELECTION":
        selection = state.get("selection") or {}
        values = [
            f"card:{item.get('index')}:{item.get('card_id')}"
            for item in selection.get("cards") or []
        ]
        actions = model_actions(state)
        if "skip_reward_cards" in actions or "skip_card_selection" in actions:
            values.append("card:skip")
        return tuple(values)
    if screen == "SHOP":
        values = list(_shop_option_ids(state.get("shop") or {}))
        actions = model_actions(state)
        if "close_shop_inventory" in actions or "proceed" in actions:
            values.append("shop:leave")
        return tuple(values)
    if screen == "REST":
        rest = state.get("rest") or {}
        values = [
            f"rest:{item.get('index')}:{item.get('option_id')}"
            for item in rest.get("options") or []
            if item.get("is_enabled") is not False
        ]
        if "proceed" in model_actions(state):
            values.append("rest:leave")
        return tuple(values)
    if screen == "EVENT":
        event = state.get("event") or {}
        return tuple(
            f"event:{item.get('index')}:{item.get('text_key')}"
            for item in event.get("options") or []
            if item.get("is_locked") is not True
        )
    return ()


def _publish_checkpoint(
    *,
    destination: Path,
    source_save: Path,
    game_version: str,
    mod_version: str,
    entry: CheckpointEntry,
) -> StrategicCheckpoint:
    """把原生存档和入口元数据发布为不可覆盖目录。

    Args:
        destination (Path): 尚不存在的 checkpoint 目录。
        source_save (Path): 待复制的原生整局存档。
        game_version (str): 捕获时游戏版本。
        mod_version (str): 捕获时 Agent Mod 版本。
        entry (CheckpointEntry): 目标战略入口快照。

    Returns:
        StrategicCheckpoint: 已发布且文件只读的 checkpoint。
    """
    destination.mkdir(parents=True)
    save_path = destination / "current_run.save"
    entry_path = destination / "entry.json"
    shutil.copy2(source_save, save_path)
    entry_path.write_text(
        json.dumps(
            {
                "game_version": game_version,
                "mod_version": mod_version,
                "entry": _entry_to_dict(entry),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    save_path.chmod(0o444)
    entry_path.chmod(0o444)
    return StrategicCheckpoint(
        root=destination,
        save_path=save_path,
        entry_path=entry_path,
        game_version=game_version,
        mod_version=mod_version,
        entry=entry,
    )


def _shop_option_ids(shop: Mapping[str, Any]) -> tuple[str, ...]:
    """提取当前商店仍可购买或可执行的库存身份。

    Args:
        shop (Mapping[str, Any]): Mod 返回的商店状态。

    Returns:
        tuple[str, ...]: 卡牌、遗物、药水和删牌身份。
    """
    values = []
    for kind, key in (
        ("card", "card_id"),
        ("relic", "relic_id"),
        ("potion", "potion_id"),
    ):
        for item in shop.get(f"{kind}s") or []:
            if item.get("is_stocked") is False or item.get("enough_gold") is False:
                continue
            values.append(f"shop:{kind}:{item.get('index')}:{item.get(key)}")
    removal = shop.get("card_removal")
    if (
        isinstance(removal, Mapping)
        and removal.get("available") is True
        and removal.get("enough_gold") is not False
    ):
        values.append("shop:remove_card")
    return tuple(values)


def _entry_to_dict(entry: CheckpointEntry) -> dict[str, Any]:
    """把入口值对象转换成 JSON 兼容字典。

    Args:
        entry (CheckpointEntry): 待持久化入口。

    Returns:
        dict[str, Any]: 保留全部可见和隐藏审计字段的字典。
    """
    return {
        "screen": entry.screen,
        "policy_layer": entry.policy_layer,
        "policy_text": entry.policy_text,
        "legal_actions": list(entry.legal_actions),
        "option_ids": list(entry.option_ids),
        "resume_actions": [
            {"name": action.name, "parameters": action.parameters}
            for action in entry.resume_actions
        ],
        "audit": entry.audit,
    }


def _entry_from_dict(payload: Mapping[str, Any]) -> CheckpointEntry:
    """从已发布元数据恢复入口值对象。

    Args:
        payload (Mapping[str, Any]): ``entry.json`` 中的入口对象。

    Raises:
        CheckpointError: 入口缺少必需字符串、列表或审计对象。

    Returns:
        CheckpointEntry: 恢复后的入口值对象。
    """
    screen = payload.get("screen")
    policy_layer = payload.get("policy_layer")
    policy_text = payload.get("policy_text")
    legal_actions = payload.get("legal_actions")
    option_ids = payload.get("option_ids")
    resume_actions = payload.get("resume_actions")
    audit = payload.get("audit")
    if (
        not isinstance(screen, str)
        or not isinstance(policy_layer, str)
        or not isinstance(policy_text, str)
        or not _is_string_sequence(legal_actions)
        or not _is_string_sequence(option_ids)
        or not isinstance(resume_actions, list)
        or not isinstance(audit, Mapping)
    ):
        raise CheckpointError("checkpoint 入口元数据无效")
    return CheckpointEntry(
        screen=screen,
        policy_layer=policy_layer,
        policy_text=policy_text,
        legal_actions=tuple(legal_actions),
        option_ids=tuple(option_ids),
        resume_actions=tuple(_parse_resume_action(action) for action in resume_actions),
        audit=dict(audit),
    )


def _parse_resume_action(payload: object) -> HarnessAction:
    """从 checkpoint 元数据恢复一个页面重开动作。

    Args:
        payload (object): 尚未校验的动作对象。

    Raises:
        CheckpointError: 动作名称或整数参数无效。

    Returns:
        HarnessAction: 可交给 Mod 的规范恢复动作。
    """
    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint 恢复动作无效")
    name = payload.get("name")
    parameters = payload.get("parameters")
    if (
        not isinstance(name, str)
        or not isinstance(parameters, Mapping)
        or any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            for key, value in parameters.items()
        )
    ):
        raise CheckpointError("checkpoint 恢复动作无效")
    return HarnessAction(name, dict(parameters))


def _is_string_sequence(value: object) -> bool:
    """判断 JSON 值是否为纯字符串数组。

    Args:
        value (object): 待检查的 JSON 值。

    Returns:
        bool: 值是非字符串序列且每项都是字符串时为真。
    """
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and all(isinstance(item, str) for item in value)
    )
