"""验证游戏知识的导入、实测导出与 Markdown 契约。"""

import json
from pathlib import Path

import pytest

from play_sts2 import game_knowledge
from play_sts2.client import Health
from play_sts2.game_knowledge import export_mod_knowledge, import_web_wiki


class FakeGameDataClient:
    """提供与真实 Mod 数据端点同形状的确定性测试替身。"""

    def health(self) -> Health:
        """返回用于知识快照目录的游戏版本。

        Returns:
            Health: 固定的 Mod 健康信息。
        """
        return Health(
            service="sts2-ai-agent",
            mod_version="0.8.0",
            protocol_version="2026-03-11-v1",
            game_version="0.107.1",
            status="ready",
        )

    def data_collection(self, collection: str) -> list[dict[str, object]]:
        """按集合名称返回一份真实协议形状的数据。

        Args:
            collection (str): Mod 游戏数据集合名称。

        Returns:
            list[dict[str, object]]: 卡牌集合含一个实体，其余集合为空。
        """
        if collection != "cards":
            return []
        return [
            {
                "id": "ZAP",
                "name": "电击",
                "description": "生成[blue]1[/blue]个闪电充能球。",
                "type": "Skill",
                "rarity": "Basic",
                "target": "Self",
                "cost": 1,
                "is_x_cost": False,
                "star_cost": None,
                "is_x_star_cost": False,
                "color": "defect",
                "vars": [],
                "upgrade": {
                    "description": "生成1个闪电充能球。",
                    "cost": 0,
                    "is_x_cost": False,
                    "star_cost": None,
                    "is_x_star_cost": False,
                    "vars": [],
                },
            },
            {
                "id": "ABRASIVE",
                "name": "磨蚀",
                "description": "获得4点荆棘。",
                "type": "Power",
                "rarity": "Rare",
                "target": "Self",
                "cost": 3,
                "is_x_cost": False,
                "star_cost": None,
                "is_x_star_cost": False,
                "color": "silent",
                "vars": [{"name": "Thorns", "current_value": 4}],
                "upgrade": {
                    "description": "获得6点荆棘。",
                    "cost": 3,
                    "is_x_cost": False,
                    "star_cost": None,
                    "is_x_star_cost": False,
                    "vars": [{"name": "Thorns", "current_value": 6}],
                },
            },
            {
                "id": "ADRENALINE",
                "name": "肾上腺素",
                "description": (
                    "获得[img]res://images/packed/sprite_fonts/"
                    "silent_energy_icon.png[/img]。 抽2张牌。"
                ),
                "description_raw": "获得{Energy}。抽{Cards}张牌。",
                "type": "Skill",
                "rarity": "Uncommon",
                "target": "Self",
                "cost": 0,
                "is_x_cost": False,
                "star_cost": None,
                "is_x_star_cost": False,
                "color": "silent",
                "vars": [],
                "upgrade": None,
            },
        ]


class EmptyGameDataClient(FakeGameDataClient):
    """表示同一游戏版本在下一次导出时不再含测试实体。"""

    def data_collection(self, _collection: str) -> list[dict[str, object]]:
        """为所有集合返回空数组。

        Args:
            _collection (str): 本次请求的集合名称。

        Returns:
            list[dict[str, object]]: 空实体集合。
        """
        return []


class ActGameDataClient(FakeGameDataClient):
    """提供地图及其遭遇池的固定版本导出替身。"""

    def data_collection(self, collection: str) -> list[dict[str, object]]:
        """返回一张密林地图及其弱、常规、精英和 Boss 池。

        Args:
            collection (str): Mod 游戏数据集合名称。

        Returns:
            list[dict[str, object]]: 地图集合或父类提供的卡牌集合。
        """
        if collection == "acts":
            return [
                {
                    "id": "OVERGROWTH",
                    "name": "密林",
                    "index": 1,
                    "is_default": True,
                    "weak_encounters": [{"id": "SLIMES_WEAK", "name": "史莱姆"}],
                    "regular_encounters": [{"id": "INKLETS_NORMAL", "name": "墨宝"}],
                    "elite_encounters": [
                        {"id": "BYRDONIS_ELITE", "name": "多尼斯异鸟"}
                    ],
                    "boss_encounters": [{"id": "THE_KIN_BOSS", "name": "同族"}],
                }
            ]
        return super().data_collection(collection)


class RichGameDataClient(FakeGameDataClient):
    """补充事件、遭遇和占位文本的真实协议边界。"""

    def data_collection(self, collection: str) -> list[dict[str, object]]:
        """返回用于验证可达对象与数据缺口的集合。

        Args:
            collection (str): Mod 游戏数据集合名称。

        Returns:
            list[dict[str, object]]: 遭遇和事件测试实体；其他集合沿用父类。
        """
        if collection == "encounters":
            return [
                {
                    "id": "FAKE_MERCHANT_EVENT_ENCOUNTER",
                    "name": "商人？？？",
                    "room_type": "Monster",
                    "is_weak": False,
                    "is_debug": False,
                    "should_give_rewards": False,
                    "monsters": [{"id": "FAKE_MERCHANT_MONSTER", "name": "商人？？？"}],
                }
            ]
        if collection == "events":
            return [
                {
                    "id": "VARIABLE_EVENT",
                    "name": "变量事件",
                    "type": "Event",
                    "act": "Shared",
                    "description": "你发现了一处泉水。",
                    "options": [
                        {
                            "id": "HEAL",
                            "title": "休息",
                            "description": "回复{Heal}点生命。",
                        },
                        {
                            "id": "LEAVE",
                            "title": "离开",
                            "description": "TODO",
                        },
                    ],
                }
            ]
        if collection == "enchantments":
            return [
                {
                    "id": "SWIFT",
                    "name": "迅速",
                    "description": "你第一次打出这张牌时，抽X张牌。",
                    "extra_card_text": "第一次打出时抽X张牌。",
                    "is_stackable": False,
                    "show_amount": False,
                }
            ]
        return super().data_collection(collection)


class WrongVersionGameDataClient(FakeGameDataClient):
    """返回不属于固定训练版本的游戏健康信息。"""

    def health(self) -> Health:
        """返回应被知识导出拒绝的游戏版本。

        Returns:
            Health: 非固定版本健康信息。
        """
        return Health(
            service="sts2-ai-agent",
            mod_version="0.8.0",
            protocol_version="2026-03-11-v1",
            game_version="0.108.0",
            status="ready",
        )


class PublicBetaGameDataClient(FakeGameDataClient):
    """返回项目明确支持的 ``v0.111.0`` 教师版本。"""

    def health(self) -> Health:
        """返回 public-beta 教师运行时的健康信息。

        Returns:
            Health: ``v0.111.0`` 健康信息。
        """
        return Health(
            service="sts2-ai-agent",
            mod_version="0.8.0",
            protocol_version="2026-03-11-v1",
            game_version="0.111.0",
            status="ready",
        )


def _required_act_rows() -> list[dict[str, object]]:
    """返回固定版本四张地图及其四类非空遭遇池。

    Returns:
        list[dict[str, object]]: 可供离线重建测试使用的完整地图集合。
    """
    names = {
        "OVERGROWTH": "密林",
        "UNDERDOCKS": "暗港",
        "HIVE": "巢穴",
        "GLORY": "荣耀",
    }
    return [
        {
            "id": act_id,
            "name": name,
            "index": index,
            "is_default": True,
            "weak_encounters": [{"id": f"{act_id}_WEAK", "name": "弱敌"}],
            "regular_encounters": [{"id": f"{act_id}_NORMAL", "name": "普通敌人"}],
            "elite_encounters": [{"id": f"{act_id}_ELITE", "name": "精英敌人"}],
            "boss_encounters": [{"id": f"{act_id}_BOSS", "name": "首领"}],
        }
        for index, (act_id, name) in enumerate(names.items(), start=1)
    ]


def _write_required_act_markdown(
    snapshot: Path,
    game_version: str = "v0.107.1",
) -> None:
    """给多问法单元测试写入固定四张地图的最小完整规范事实。

    Args:
        snapshot (Path): ``mod_export/v0.107.1`` 测试快照根目录。
        game_version (str): 写入 frontmatter 的游戏版本。

    Returns:
        None: 四个地图 Markdown 写入完成后返回。
    """
    acts = snapshot / "acts"
    acts.mkdir(parents=True, exist_ok=True)
    for row in _required_act_rows():
        object_id = str(row["id"])
        name = str(row["name"])
        sections = []
        for heading, field in (
            ("弱遭遇池", "weak_encounters"),
            ("常规遭遇池", "regular_encounters"),
            ("精英遭遇池", "elite_encounters"),
            ("Boss 遭遇池", "boss_encounters"),
        ):
            encounter = row[field][0]
            sections.append(f"## {heading}\n- {encounter['name']}（{encounter['id']}）")
        (acts / f"{object_id}.md").write_text(
            f"""---
id: {object_id}
name: {name}
type: act
source: mod_export
game_version: {game_version}
index: {row["index"]}
is_default: true
---
{"\n\n".join(sections)}
""",
            encoding="utf-8",
        )


def test_import_web_wiki_preserves_facts_and_rewrites_provenance(
    tmp_path: Path,
) -> None:
    """导入站点条目时保留事实，并移除其他工程的本地增补段。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 生成条目的来源、正文或清单不符合约定。

    Returns:
        None: 此测试只检查导入产物。
    """
    source = tmp_path / "source"
    (source / "cards").mkdir(parents=True)
    (source / "cards/ZAP.md").write_text(
        """---
id: ZAP
name: 电击
type: wiki/card
source: spire-codex 2026-08-16
cost: 1
---
## 效果
生成1个闪电充能球。

<!-- praxis:local -->
- 不应迁移的本地注记
""",
        encoding="utf-8",
    )

    result = import_web_wiki(source, tmp_path / "knowledge")

    output = tmp_path / "knowledge/web_wiki/cards/ZAP.md"
    text = output.read_text(encoding="utf-8")
    assert result.entry_count == 1
    assert "type: card" in text
    assert "source: web_wiki" in text
    assert "source_detail: spire-codex 2026-08-16" in text
    assert "生成1个闪电充能球。" in text
    assert "praxis:local" not in text
    assert "不应迁移" not in text
    assert json.loads(
        (tmp_path / "knowledge/web_wiki/manifest.json").read_text(encoding="utf-8")
    ) == {
        "source": "web_wiki",
        "entries": 1,
        "categories": {"cards": 1},
    }

    (source / "cards/ZAP.md").unlink()
    import_web_wiki(source, tmp_path / "knowledge")

    assert not output.exists()


def test_export_mod_knowledge_writes_raw_snapshot_and_markdown(
    tmp_path: Path,
) -> None:
    """把 Mod 实测集合同时保存为原始 JSON 与单实体 Markdown。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 快照目录、来源字段或卡牌正文不符合约定。

    Returns:
        None: 此测试只检查导出产物。
    """
    result = export_mod_knowledge(FakeGameDataClient(), tmp_path / "knowledge")

    root = tmp_path / "knowledge/mod_export/v0.107.1"
    raw_cards = json.loads((root / "raw/cards.json").read_text(encoding="utf-8"))
    card = (root / "cards/ZAP.md").read_text(encoding="utf-8")
    upgraded_card = (root / "cards/ABRASIVE.md").read_text(encoding="utf-8")
    assert result.output_root == root
    assert result.entry_count == 3
    assert raw_cards[0]["id"] == "ZAP"
    assert "source: mod_export" in card
    assert "game_version: v0.107.1" in card
    assert "## 效果\n生成1个闪电充能球。" in card
    assert "## 升级" in card
    assert "- 费用：1 → 0" in card
    assert "生成1个闪电充能球。" in card
    assert "## 升级\n- 效果：获得6点荆棘。" in upgraded_card
    adrenaline = (root / "cards/ADRENALINE.md").read_text(encoding="utf-8")
    assert "获得1点能量。 抽2张牌。" in adrenaline
    assert "res://" not in adrenaline

    export_mod_knowledge(EmptyGameDataClient(), tmp_path / "knowledge")

    assert not (root / "cards/ZAP.md").exists()
    assert not (root / "cards/ABRASIVE.md").exists()


def test_export_mod_knowledge_keeps_real_fake_merchant_encounter_and_gaps(
    tmp_path: Path,
) -> None:
    """保留真实假商人遭遇，同时不把未解析富文本写入知识。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 遭遇被前缀过滤或占位符进入 Markdown。

    Returns:
        None: 此测试只检查导出产物和审计清单。
    """
    result = export_mod_knowledge(RichGameDataClient(), tmp_path / "knowledge")

    encounter = (
        result.output_root / "encounters/FAKE_MERCHANT_EVENT_ENCOUNTER.md"
    ).read_text(encoding="utf-8")
    event = (result.output_root / "events/VARIABLE_EVENT.md").read_text(
        encoding="utf-8"
    )
    manifest = json.loads(
        (result.output_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert "商人？？？（FAKE_MERCHANT_MONSTER）" in encounter
    enchantment = (result.output_root / "enchantments/SWIFT.md").read_text(
        encoding="utf-8"
    )
    assert "你第一次打出这张牌时，抽X张牌。" in enchantment
    assert "## 卡面附加" in enchantment
    assert "{Heal}" not in event
    assert "TODO" not in event
    assert "缺少已解析选项数据" in event
    assert manifest["gaps"] == [
        "events:VARIABLE_EVENT:options.HEAL.description:unresolved_text",
        "events:VARIABLE_EVENT:options.LEAVE.description:unresolved_text",
    ]


def test_export_mod_knowledge_preserves_act_encounter_pools(tmp_path: Path) -> None:
    """固定版本导出应保存地图级普通、精英与 Boss 遭遇池。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 原始地图集合或 canonical Markdown 丢失任一池。

    Returns:
        None: 此测试只检查地图知识导出契约。
    """
    result = export_mod_knowledge(ActGameDataClient(), tmp_path / "knowledge")

    acts = json.loads((result.output_root / "raw/acts.json").read_text())
    act = (result.output_root / "acts/OVERGROWTH.md").read_text()
    assert acts[0]["name"] == "密林"
    assert "## 弱遭遇池\n- 史莱姆（SLIMES_WEAK）" in act
    assert "## 常规遭遇池\n- 墨宝（INKLETS_NORMAL）" in act
    assert "## 精英遭遇池\n- 多尼斯异鸟（BYRDONIS_ELITE）" in act
    assert "## Boss 遭遇池\n- 同族（THE_KIN_BOSS）" in act


def test_export_mod_knowledge_rejects_unsupported_version(
    tmp_path: Path,
) -> None:
    """知识导出拒绝意外升级后的 Steam 游戏。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 非固定版本仍被写入知识目录。

    Returns:
        None: 此测试只检查固定版本边界。
    """
    with pytest.raises(ValueError, match="不在受支持版本中"):
        export_mod_knowledge(WrongVersionGameDataClient(), tmp_path / "knowledge")


def test_export_mod_knowledge_keeps_v01110_separate_from_v01071(
    tmp_path: Path,
) -> None:
    """public-beta 导出必须写入独立目录且不覆盖旧基线。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 新旧版本知识没有并存或版本元数据错误。

    Returns:
        None: 此测试只验证受支持版本的物理隔离。
    """
    knowledge_root = tmp_path / "knowledge"
    old_result = export_mod_knowledge(FakeGameDataClient(), knowledge_root)
    new_result = export_mod_knowledge(PublicBetaGameDataClient(), knowledge_root)

    assert old_result.output_root.name == "v0.107.1"
    assert new_result.output_root.name == "v0.111.0"
    assert (old_result.output_root / "cards/ZAP.md").is_file()
    new_card = (new_result.output_root / "cards/ZAP.md").read_text(encoding="utf-8")
    assert "game_version: v0.111.0" in new_card


def test_rebuild_mod_knowledge_uses_version_from_raw_parent(tmp_path: Path) -> None:
    """离线重建应从 raw 父目录选择同版本 curated 与输出目录。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: ``v0.111.0`` 被拒绝或写回旧版目录。

    Returns:
        None: 此测试只验证版本化离线重建边界。
    """
    knowledge_root = tmp_path / "knowledge"
    raw_root = knowledge_root / "mod_export/v0.111.0/raw"
    raw_root.mkdir(parents=True)
    (raw_root / "acts.json").write_text(
        json.dumps(_required_act_rows(), ensure_ascii=False),
        encoding="utf-8",
    )

    result = game_knowledge.rebuild_mod_knowledge(raw_root, knowledge_root)

    assert result.output_root == knowledge_root / "mod_export/v0.111.0"
    assert "game_version: v0.111.0" in (
        result.output_root / "acts/OVERGROWTH.md"
    ).read_text(encoding="utf-8")


def test_generate_question_variants_preserves_v01110_version(tmp_path: Path) -> None:
    """多问法生成应沿用 ``v0.111.0`` 快照版本。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 新版本快照被拒绝或 manifest 错写为旧版本。

    Returns:
        None: 此测试只验证多问法产物的版本血缘。
    """
    snapshot = tmp_path / "mod_export/v0.111.0"
    output = tmp_path / "generated-v0.111.0"
    _write_required_act_markdown(snapshot, "v0.111.0")

    result = game_knowledge.generate_question_variants(snapshot, output)

    manifest = json.loads(
        (result.output_root / "_knowledge_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["game_version"] == "v0.111.0"
    assert manifest["facts"] == 16


def test_generate_question_variants_uses_curated_facts_without_web_wiki(
    tmp_path: Path,
) -> None:
    """同一 canonical 实体生成多种问法，并应用版本化事实修正。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 多问法生成缺失、重复或仍含错误 FIFO 文本。

    Returns:
        None: 此测试只检查知识问法产物。
    """
    snapshot = tmp_path / "knowledge/mod_export/v0.107.1"
    _write_required_act_markdown(snapshot)
    (snapshot / "cards").mkdir(parents=True)
    (snapshot / "cards/DUALCAST.md").write_text(
        """---
id: DUALCAST
name: 双重释放
type: card
source: mod_export
game_version: v0.107.1
character: defect
card_type: Skill
rarity: Basic
target: Self
cost: 1
---
## 效果
激发你最右侧的充能球两次。
## 升级
- 费用：1 → 0
- 效果：激发你最右侧的充能球两次。
""",
        encoding="utf-8",
    )
    (snapshot / "enchantments").mkdir()
    (snapshot / "enchantments/DEPRECATED_ENCHANTMENT.md").write_text(
        """---
id: DEPRECATED_ENCHANTMENT
name: 弃用
type: enchantment
source: mod_export
game_version: v0.107.1
stackable: false
---
## 效果
这个附魔已经从游戏中被移除。
""",
        encoding="utf-8",
    )
    (snapshot / "powers").mkdir()
    (snapshot / "powers/KNOCKDOWN_POWER.md").write_text(
        """---
id: KNOCKDOWN_POWER
name: 击倒
type: power
source: mod_export
game_version: v0.107.1
power_type: Debuff
stack_type: Counter
sample_amount: 1
uses_amount: true
---
## 效果
该敌人在本回合受到的来自其他玩家的伤害变为1倍。
""",
        encoding="utf-8",
    )
    (snapshot / "monsters").mkdir()
    (snapshot / "monsters/UNKNOWN_CYCLE.md").write_text(
        """---
id: UNKNOWN_CYCLE
name: 未知循环怪
type: monster
source: mod_export
game_version: v0.107.1
room: Normal
min_hp: 10
max_hp: 10
supplement_source: web_wiki:monster_moves_cycles;human_rl_cycle_lab:pending_observation
---
## 招式
- BUFF（Buff）｜→ [STRENGTH](../powers/STRENGTH.md) 2 (self)
## 循环
（循环未知）
""",
        encoding="utf-8",
    )

    output = tmp_path / "knowledge/generated-v0.107.1"
    result = game_knowledge.generate_question_variants(snapshot, output)

    rows = [
        json.loads(line)
        for line in (output / "cards/DUALCAST.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    power_rows = [
        json.loads(line)
        for line in (output / "powers/KNOCKDOWN_POWER.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    monster_rows = [
        json.loads(line)
        for line in (output / "monsters/UNKNOWN_CYCLE.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert result.entry_count == len(rows) + len(power_rows) + len(monster_rows) + 64
    assert result.categories["encounters"] == 64
    assert len(rows) >= 5
    assert len({row["prompt"] for row in rows}) == len(rows)
    assert all(row["fact_id"].startswith("cards/DUALCAST/") for row in rows)
    facts: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        facts.setdefault(str(row["fact_id"]), []).append(row)
    assert all(
        {str(row["question_role"]) for row in fact_rows}
        == {"train", "validation", "eval"}
        for fact_rows in facts.values()
    )
    assert all(
        sum(row["question_role"] == "validation" for row in fact_rows) == 1
        and sum(row["question_role"] == "eval" for row in fact_rows) == 1
        for fact_rows in facts.values()
    )
    assert all(row["source"] != "web_wiki" for row in rows)
    assert all("最右侧" not in row["completion"] for row in rows)
    assert any("最旧（队首）" in row["completion"] for row in rows)
    assert any("费用1→0" in row["completion"] for row in rows)
    assert not (output / "enchantments/DEPRECATED_ENCHANTMENT.jsonl").exists()
    manifest = json.loads(
        (output / "_knowledge_manifest.json").read_text(encoding="utf-8")
    )
    assert (
        "enchantments:DEPRECATED_ENCHANTMENT:curated_exclusion" in manifest["skipped"]
    )
    assert manifest["question_roles"] == {
        "eval": manifest["facts"],
        "train": len(rows + power_rows + monster_rows) + 64 - 2 * manifest["facts"],
        "validation": manifest["facts"],
    }
    assert all("1" in row["prompt"] for row in power_rows)
    assert not any("行动循环" in row["prompt"] for row in monster_rows)
    assert all("../powers/" not in row["completion"] for row in monster_rows)
    for row in monster_rows:
        if row["fact_id"].endswith("/moves"):
            assert row["supplement_source"] == "web_wiki:monster_moves_cycles"
        else:
            assert "supplement_source" not in row


def test_generate_review_report_lists_suspicious_objects_and_missing_fields(
    tmp_path: Path,
) -> None:
    """审计报告区分可保留对象、明确弃用对象和待确认对象。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 报告粗暴过滤 FAKE 前缀或遗漏补录项。

    Returns:
        None: 此测试只检查人类可读报告。
    """
    snapshot = tmp_path / "mod_export/v0.107.1"
    raw = snapshot / "raw"
    raw.mkdir(parents=True)
    fixtures = {
        "relics": [
            {
                "id": "FAKE_ANCHOR",
                "name": "锚？？？",
                "description": "战斗开始时获得4点格挡。",
            },
            {
                "id": "DEPRECATED_RELIC",
                "name": "弃用遗物",
                "description": "这件遗物已经从游戏中被移除。",
            },
            {
                "id": "BYRDPIP",
                "name": "异鸟宝宝",
                "description": "每场战斗打出第一张能力牌时触发。",
            },
        ],
        "monsters": [
            {
                "id": "TEST_SUBJECT",
                "name": "实验体 #C8",
                "show_in_compendium": True,
                "type": "Boss",
                "moves": [{"id": "RESPAWN_MOVE", "name": "重生"}],
            }
        ],
        "encounters": [
            {
                "id": "FAKE_MERCHANT_EVENT_ENCOUNTER",
                "name": "商人？？？",
                "is_debug": False,
                "monsters": [{"id": "FAKE_MERCHANT_MONSTER", "name": "商人？？？"}],
            },
            {
                "id": "TEST_SUBJECT_BOSS",
                "name": "实验体",
                "room_type": "Boss",
                "is_debug": False,
                "should_give_rewards": True,
                "monsters": [{"id": "TEST_SUBJECT", "name": "实验体 #C8"}],
            },
        ],
        "events": [
            {
                "id": "ABYSSAL_BATHS",
                "name": "深渊浴场",
                "description": "你发现了一间僻静的密室。",
                "type": "Event",
                "act": "Shared",
                "options": [
                    {
                        "id": "IMMERSE",
                        "title": "投身其中",
                        "description": "获得{MaxHp}点最大生命值。受到{Damage}点伤害。",
                    }
                ],
            }
        ],
    }
    for category, rows in fixtures.items():
        (raw / f"{category}.json").write_text(
            json.dumps(rows, ensure_ascii=False),
            encoding="utf-8",
        )

    report_path = tmp_path / "review.md"
    game_knowledge.generate_review_report(snapshot, report_path)

    report = report_path.read_text(encoding="utf-8")
    assert "FAKE_ANCHOR" in report
    assert "保留：有完整游戏文本；不能按 FAKE_ 前缀删除" in report
    assert "DEPRECATED_RELIC" in report
    assert "明确标注已移除" in report
    assert "真实遗物模型，与同名辅助 MonsterModel 区分；保留" in report
    assert "TEST_SUBJECT" in report
    assert (
        "当前版本非调试、可奖励 Boss 遭遇；raw 保留，不生成现场可见的组合题" in report
    )
    assert "当前版本图鉴可见 Boss 且招式完整；保留并进入监督问法" in report
    assert "FAKE_MERCHANT_EVENT_ENCOUNTER" in report
    assert "用户确认真实存在，保留" in report


def test_rebuild_mod_knowledge_applies_curated_facts_and_monster_supplements(
    tmp_path: Path,
) -> None:
    """离线重建把 FIFO 修正、Wiki 招式和实跳观察写入 canonical 知识。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 补充来源覆盖基础事实、丢失遭遇或未留下来源说明。

    Returns:
        None: 此测试只检查固定快照的离线重建结果。
    """
    knowledge_root = tmp_path / "knowledge"
    snapshot = knowledge_root / "mod_export/v0.107.1"
    raw = snapshot / "raw"
    raw.mkdir(parents=True)
    payloads = {
        "acts": _required_act_rows(),
        "cards": [
            {
                "id": "DUALCAST",
                "name": "双重释放",
                "description": "激发你最右侧的充能球两次。",
                "type": "Skill",
                "rarity": "Basic",
                "target": "Self",
                "cost": 1,
                "is_x_cost": False,
                "star_cost": None,
                "is_x_star_cost": False,
                "color": "defect",
                "upgrade": {
                    "description": "激发你最右侧的充能球两次。",
                    "cost": 0,
                    "is_x_cost": False,
                    "star_cost": None,
                    "is_x_star_cost": False,
                },
            }
        ],
        "monsters": [
            {
                "id": "SKULKING_COLONY",
                "name": "鬼祟珊瑚群",
                "type": "Elite",
                "min_hp": 75,
                "max_hp": 75,
                "moves": [],
                "acts": [{"id": "OVERGROWTH", "index": 1, "name": "蔓生之地"}],
                "encounters": ["SKULKING_COLONY_ELITE"],
            }
        ],
        "encounters": [
            {
                "id": "FAKE_MERCHANT_EVENT_ENCOUNTER",
                "name": "商人？？？",
                "room_type": "Monster",
                "is_weak": False,
                "is_debug": False,
                "should_give_rewards": False,
                "monsters": [{"id": "FAKE_MERCHANT_MONSTER", "name": "商人？？？"}],
            }
        ],
        "events": [
            {
                "id": "ABYSSAL_BATHS",
                "name": "深渊浴场",
                "description": "你发现了一间僻静的密室。",
                "type": "Event",
                "act": "Shared",
                "options": [
                    {
                        "id": "IMMERSE",
                        "title": "投身其中",
                        "description": "获得{MaxHp}点最大生命值。受到{Damage}点伤害。",
                    }
                ],
            }
        ],
        "powers": [
            {
                "id": "ASLEEP_POWER",
                "name": "沉睡",
                "description": "TODO",
                "type": "Buff",
                "stack_type": "Counter",
            }
        ],
    }
    for category, rows in payloads.items():
        (raw / f"{category}.json").write_text(
            json.dumps(rows, ensure_ascii=False),
            encoding="utf-8",
        )

    wiki = knowledge_root / "web_wiki"
    (wiki / "monsters").mkdir(parents=True)
    (wiki / "monsters/SKULKING_COLONY.md").write_text(
        """---
id: SKULKING_COLONY
name: 鬼祟珊瑚群
type: monster
source: web_wiki
act: 0
room: elite
hp: 75
hp_ascension: 80
innate: [HARDENED_SHELL]
encounters: [SKULKING_COLONY_ELITE]
---
## 招式
- ZOOM（Attack）14 (A+:16)
- PIERCING_STABS（Attack）7×2 (A+:8×2)
## 循环
猛冲 → 穿刺戳击 → repeat
""",
        encoding="utf-8",
    )
    cycles = tmp_path / "cycles"
    cycles.mkdir()
    (cycles / "SKULKING_COLONY_ELITE.json").write_text(
        json.dumps(
            {
                "encounter_id": "SKULKING_COLONY_ELITE",
                "status": "pending_online_verification",
                "monsters": [
                    {
                        "name": "鬼祟珊瑚群",
                        "bands": {
                            "high": {
                                "moves": [
                                    {
                                        "move": "ZOOM_MOVE",
                                        "intent_type": "Attack",
                                        "damage": 14,
                                        "hits": 1,
                                    }
                                ],
                                "turns_observed": 1,
                                "closed": False,
                            }
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    event_entries = tmp_path / "event_entries"
    event_entries.mkdir()
    (event_entries / "ABYSSAL_BATHS.json").write_text(
        json.dumps(
            {
                "state": {
                    "event": {
                        "event_id": "ABYSSAL_BATHS",
                        "title": "深渊浴场",
                        "description": "你发现了一间僻静的密室。",
                        "options": [
                            {
                                "index": 0,
                                "text_key": (
                                    "ABYSSAL_BATHS.pages.INITIAL.options.IMMERSE"
                                ),
                                "title": "投身其中",
                                "description": "获得2点最大生命值。受到3点伤害。",
                            }
                        ],
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = game_knowledge.rebuild_mod_knowledge(
        raw,
        knowledge_root,
        wiki_root=wiki,
        cycles_root=cycles,
        event_entries_root=event_entries,
    )

    card = (result.output_root / "cards/DUALCAST.md").read_text(encoding="utf-8")
    monster = (result.output_root / "monsters/SKULKING_COLONY.md").read_text(
        encoding="utf-8"
    )
    encounter = (
        result.output_root / "encounters/FAKE_MERCHANT_EVENT_ENCOUNTER.md"
    ).read_text(encoding="utf-8")
    event = (result.output_root / "events/ABYSSAL_BATHS.md").read_text(encoding="utf-8")
    asleep = (result.output_root / "powers/ASLEEP_POWER.md").read_text(encoding="utf-8")
    assert "最旧（队首）的充能球" in card
    assert "最右侧" not in card
    assert "费用：1 → 0" in card
    assert "hp_ascension: 80" not in monster
    assert "## 招式\n- ZOOM" in monster
    assert "## 循环\n猛冲 → 穿刺戳击 → repeat" in monster
    assert "## 固有能力" not in monster
    assert "## 所在 Act\n- Act 1：蔓生之地（OVERGROWTH）" in monster
    assert "## 可能关联的遭遇\n- SKULKING_COLONY_ELITE" in monster
    assert "## 实跳观察（待复核）" in monster
    assert "ZOOM_MOVE：14×1" in monster
    assert "FAKE_MERCHANT_MONSTER" in encounter
    assert "获得2点最大生命值。受到3点伤害。" in event
    assert "{MaxHp}" not in event
    assert "snapshot_scope: single_state" in event
    assert "受到未被格挡的伤害时苏醒" in asleep
    assert "TODO" not in asleep
    manifest = json.loads((result.output_root / "manifest.json").read_text())
    assert "game_ui_snapshot:resolved_options" in manifest["supplements"]

    generated = tmp_path / "generated-v0.107.1"
    game_knowledge.generate_question_variants(result.output_root, generated)
    assert not (generated / "events/ABYSSAL_BATHS.jsonl").exists()
    generated_manifest = json.loads(
        (generated / "_knowledge_manifest.json").read_text(encoding="utf-8")
    )
    assert (
        "events:ABYSSAL_BATHS:state_dependent_snapshot" in generated_manifest["skipped"]
    )
    review_path = tmp_path / "review.md"
    game_knowledge.generate_review_report(result.output_root, review_path)
    review = review_path.read_text(encoding="utf-8")
    assert "状态依赖事件快照（不进入通用监督问法）" in review
    assert "ABYSSAL_BATHS" in review


def test_review_report_uses_post_rebuild_manifest_gaps(tmp_path: Path) -> None:
    """审计报告应展示 curated/实机补充后仍存在的缺口，而非原始模板噪声。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 报告未使用重建 manifest 或仍列出已补全条目。

    Returns:
        None: 此测试只检查审计报告的缺口来源。
    """
    snapshot = tmp_path / "mod_export/v0.107.1"
    raw = snapshot / "raw"
    raw.mkdir(parents=True)
    (raw / "events.json").write_text(
        json.dumps(
            [
                {
                    "id": "ALREADY_RESOLVED",
                    "name": "已补全",
                    "description": "正文",
                    "options": [{"id": "A", "title": "选项", "description": "{Value}"}],
                },
                {
                    "id": "STILL_MISSING",
                    "name": "仍缺失",
                    "description": "正文",
                    "options": [{"id": "B", "title": "选项", "description": "{Value}"}],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "gaps": ["events:STILL_MISSING:options.B.description:unresolved_text"],
                "supplements": ["game_ui_snapshot:resolved_options"],
            }
        ),
        encoding="utf-8",
    )

    report_path = tmp_path / "review.md"
    game_knowledge.generate_review_report(snapshot, report_path)

    report = report_path.read_text(encoding="utf-8")
    assert "STILL_MISSING" in report
    assert "ALREADY_RESOLVED" not in report
    assert "game_ui_snapshot:resolved_options" in report


def test_generate_question_variants_disambiguates_same_named_cards(
    tmp_path: Path,
) -> None:
    """同名不同角色卡牌不会生成相同 prompt 的冲突监督。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 防御等同名卡牌仍产生相同问题。

    Returns:
        None: 此测试只检查生成问法的全局唯一性。
    """
    snapshot = tmp_path / "mod_export/v0.107.1"
    _write_required_act_markdown(snapshot)
    cards = snapshot / "cards"
    cards.mkdir(parents=True)
    for object_id, character, effect in (
        ("DEFEND_IRONCLAD", "ironclad", "获得5点格挡。"),
        ("DEFEND_NECROBINDER", "necrobinder", "获得6点格挡。"),
    ):
        (cards / f"{object_id}.md").write_text(
            f"""---
id: {object_id}
name: 防御
type: card
source: mod_export
game_version: v0.107.1
character: {character}
card_type: Skill
rarity: Basic
target: Self
cost: 1
---
## 效果
{effect}
""",
            encoding="utf-8",
        )

    output = tmp_path / "generated-v0.107.1"
    game_knowledge.generate_question_variants(snapshot, output)

    rows = [
        json.loads(line)
        for path in sorted((output / "cards").glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prompts = [row["prompt"] for row in rows]
    assert len(prompts) == len(set(prompts))
    assert any("铁甲战士" in prompt for prompt in prompts)
    assert any("亡灵契约师" in prompt for prompt in prompts)


def test_question_generation_rejects_sentinels_and_keeps_real_test_subject(
    tmp_path: Path,
) -> None:
    """枚举哨兵/谜面不进入答案，真实 TEST_SUBJECT 不能按前缀误删。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 哨兵值泄漏到答案或真实实验体被前缀误删。

    Returns:
        None: 此测试只检查问法的清洗与保留规则。
    """
    snapshot = tmp_path / "mod_export/v0.107.1"
    fixtures = {
        "cards/MAD_SCIENCE.md": """---
id: MAD_SCIENCE
name: 疯狂科学
type: card
source: mod_export
game_version: v0.107.1
character: colorless
card_type: None
rarity: Event
target: Self
cost: 1
---
## 效果
？？？？？？
""",
        "cards/BURN.md": """---
id: BURN
name: 灼伤
type: card
source: mod_export
game_version: v0.107.1
character: status
card_type: Status
rarity: Status
target: None
cost: -1
---
## 效果
不可打出。回合结束时受到2点伤害。
""",
        "cards/ALIGNMENT.md": """---
id: ALIGNMENT
name: 星位序列
type: card
source: mod_export
game_version: v0.107.1
character: regent
card_type: Skill
rarity: Common
target: Self
cost: 0
star_cost: 3
---
## 效果
获得7点格挡。
""",
        "cards/STARDUST.md": """---
id: STARDUST
name: 星尘
type: card
source: mod_export
game_version: v0.107.1
character: regent
card_type: Skill
rarity: Rare
target: RandomEnemy
cost: 0
star_cost: X
---
## 效果
获得X点星能对应的效果。
""",
        "relics/CIRCLET.md": """---
id: CIRCLET
name: 头环
type: relic
source: mod_export
game_version: v0.107.1
rarity: None
---
## 效果
这是一个头环。
""",
        "powers/DAMPEN_POWER.md": """---
id: DAMPEN_POWER
name: 抑制
type: power
source: mod_export
game_version: v0.107.1
power_type: Debuff
stack_type: None
---
## 效果
魔法骑士存活时，你的所有卡牌被降级。
""",
        "monsters/TEST_SUBJECT.md": """---
id: TEST_SUBJECT
name: 实验体 #C8
type: monster
source: mod_export
game_version: v0.107.1
room: Boss
min_hp: 100
max_hp: 100
---
## 招式
- RESPAWN_MOVE：重生
""",
        "encounters/TEST_SUBJECT_BOSS.md": """---
id: TEST_SUBJECT_BOSS
name: 实验体
type: encounter
source: mod_export
game_version: v0.107.1
room: Boss
is_debug: false
should_give_rewards: true
---
## 可能出现的敌人类型
- 实验体 #C8（TEST_SUBJECT）
""",
        "encounters/OBSERVED_COMPOSITIONS.md": """---
id: OBSERVED_COMPOSITIONS
name: 实机敌人组合目录
type: encounter
source: mod_export
game_version: v0.107.1
supplement_source: human_rl:data/entries/encounters/*.json
---
## 覆盖概览
共有2种不同敌人组合：单个敌人1种、2名敌人1种。

## 单个敌人组合
下水道蚌。

## 2名敌人组合
啃咬机×2。
""",
        "acts/OVERGROWTH.md": """---
id: OVERGROWTH
name: 密林
type: act
source: mod_export
game_version: v0.107.1
index: 1
is_default: true
---
## 弱遭遇池
- 史莱姆（SLIMES_WEAK）

## 常规遭遇池
- 墨宝（INKLETS_NORMAL）

## 精英遭遇池
- 多尼斯异鸟（BYRDONIS_ELITE）

## Boss 遭遇池
- 同族（THE_KIN_BOSS）
""",
    }
    for object_id, name, index in (
        ("UNDERDOCKS", "暗港", 2),
        ("HIVE", "巢穴", 3),
        ("GLORY", "荣耀", 4),
    ):
        fixtures[f"acts/{object_id}.md"] = f"""---
id: {object_id}
name: {name}
type: act
source: mod_export
game_version: v0.107.1
index: {index}
is_default: true
---
## 弱遭遇池
- 弱敌（{object_id}_WEAK）

## 常规遭遇池
- 普通敌人（{object_id}_NORMAL）

## 精英遭遇池
- 精英敌人（{object_id}_ELITE）

## Boss 遭遇池
- 首领（{object_id}_BOSS）
"""
    for relative, content in fixtures.items():
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    output = tmp_path / "generated-v0.107.1"
    game_knowledge.generate_question_variants(snapshot, output)

    assert not (output / "cards/MAD_SCIENCE.jsonl").exists()
    burn = (output / "cards/BURN.jsonl").read_text(encoding="utf-8")
    assert "-1费" not in burn
    assert "不能直接打出" in burn
    assert "status" not in burn
    assert "Status" not in burn
    alignment = (output / "cards/ALIGNMENT.jsonl").read_text(encoding="utf-8")
    assert "0点能量 + 3点星能" in alignment
    stardust = (output / "cards/STARDUST.jsonl").read_text(encoding="utf-8")
    assert "0点能量 + X点星能" in stardust
    assert "随机敌人" in stardust
    assert "RandomEnemy" not in stardust
    for relative in ("relics/CIRCLET.jsonl", "powers/DAMPEN_POWER.jsonl"):
        text = (output / relative).read_text(encoding="utf-8")
        assert "None" not in text
    dampen = (output / "powers/DAMPEN_POWER.jsonl").read_text(encoding="utf-8")
    assert "减益" in dampen
    assert "Debuff" not in dampen
    assert (output / "monsters/TEST_SUBJECT.jsonl").is_file()
    test_subject = (output / "monsters/TEST_SUBJECT.jsonl").read_text(encoding="utf-8")
    assert "首领" in test_subject
    assert "Boss。" not in test_subject
    monster_rows = [json.loads(line) for line in test_subject.splitlines()]
    hp_answers = [
        row["completion"] for row in monster_rows if row["completion"] == " 100。"
    ]
    assert len(hp_answers) == 4
    assert not (output / "encounters/TEST_SUBJECT_BOSS.jsonl").exists()
    assert not (output / "encounters/OBSERVED_COMPOSITIONS.jsonl").exists()
    assert not (output / "acts/OVERGROWTH.jsonl").exists()
    encounter_rows = [
        json.loads(line)
        for line in (output / "encounters/OVERGROWTH.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(encounter_rows) == 16
    assert {row["category"] for row in encounter_rows} == {"encounters"}
    assert all(
        sum(
            row["completion"] == candidate["completion"] for candidate in encounter_rows
        )
        == 4
        for row in encounter_rows
    )
    assert any("全部怪池" in row["prompt"] for row in encounter_rows)
    assert any("普通怪池" in row["prompt"] for row in encounter_rows)
    assert any("精英怪池" in row["prompt"] for row in encounter_rows)
    assert any("Boss池" in row["prompt"] for row in encounter_rows)


def test_question_generation_rejects_incomplete_map_catalog_before_cleanup(
    tmp_path: Path,
) -> None:
    """地图规范事实不完整时不得发布部分怪池或清理旧产物。

    Args:
        tmp_path (Path): Pytest 提供的隔离知识目录。

    Raises:
        AssertionError: 单张地图仍被生成，或失败前删除了已有候选。

    Returns:
        None: 此测试只验证多问法生成器的发布前完整性门槛。
    """
    snapshot = tmp_path / "mod_export/v0.107.1"
    acts = snapshot / "acts"
    acts.mkdir(parents=True)
    (acts / "OVERGROWTH.md").write_text(
        """---
id: OVERGROWTH
name: 密林
type: act
source: mod_export
game_version: v0.107.1
index: 1
is_default: true
---
## 弱遭遇池
- 弱敌（WEAK）
## 常规遭遇池
- 普通敌人（NORMAL）
## 精英遭遇池
- 精英敌人（ELITE）
## Boss 遭遇池
- 首领（BOSS）
""",
        encoding="utf-8",
    )
    output = tmp_path / "generated-v0.107.1"
    existing = output / "encounters/GLORY.jsonl"
    existing.parent.mkdir(parents=True)
    existing.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="固定四张地图"):
        game_knowledge.generate_question_variants(snapshot, output)

    assert existing.read_text(encoding="utf-8") == "existing\n"


def test_question_generation_rejects_missing_map_catalog_before_cleanup(
    tmp_path: Path,
) -> None:
    """地图目录完全缺失时也不得清理已经存在的怪池候选。

    Args:
        tmp_path (Path): Pytest 提供的隔离知识目录。

    Raises:
        AssertionError: 无地图输入仍被发布，或既有地图候选被删除。

    Returns:
        None: 此测试覆盖完整地图目录缺失的发布边界。
    """
    snapshot = tmp_path / "mod_export/v0.107.1"
    snapshot.mkdir(parents=True)
    output = tmp_path / "generated-v0.107.1"
    existing = output / "encounters/GLORY.jsonl"
    existing.parent.mkdir(parents=True)
    existing.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="固定四张地图"):
        game_knowledge.generate_question_variants(snapshot, output)

    assert existing.read_text(encoding="utf-8") == "existing\n"


def test_rebuild_rejects_mismatched_monster_wiki_identity(tmp_path: Path) -> None:
    """受控 Wiki 文件名不能掩盖 frontmatter 中的另一只怪物。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: Wiki frontmatter 身份不一致时未拒绝重建。

    Returns:
        None: 此测试只检查怪物 Wiki 身份边界。
    """
    knowledge = tmp_path / "knowledge"
    raw = knowledge / "mod_export/v0.107.1/raw"
    raw.mkdir(parents=True)
    (raw / "acts.json").write_text(
        json.dumps(_required_act_rows(), ensure_ascii=False),
        encoding="utf-8",
    )
    (raw / "monsters.json").write_text(
        json.dumps(
            [
                {
                    "id": "EXPECTED",
                    "name": "预期怪物",
                    "type": "Normal",
                    "min_hp": 10,
                    "max_hp": 10,
                    "moves": [],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    wiki = knowledge / "web_wiki/monsters"
    wiki.mkdir(parents=True)
    (wiki / "EXPECTED.md").write_text(
        """---
id: DIFFERENT
name: 另一只怪物
type: monster
source: web_wiki
---
## 招式
- HIT：攻击
""",
        encoding="utf-8",
    )

    with pytest.raises(game_knowledge.KnowledgeFormatError, match="身份或来源不匹配"):
        game_knowledge.rebuild_mod_knowledge(
            raw,
            knowledge,
            wiki_root=knowledge / "web_wiki",
        )


def test_rebuild_requires_all_four_complete_act_pools(tmp_path: Path) -> None:
    """离线重建缺少固定四张地图或任一怪池时必须失败。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Raises:
        AssertionError: 缺失地图集合或空怪池仍被发布。

    Returns:
        None: 此测试只检查地图知识的 fail-closed 边界。
    """
    raw = tmp_path / "knowledge/mod_export/v0.107.1/raw"
    raw.mkdir(parents=True)

    with pytest.raises(game_knowledge.KnowledgeFormatError, match="acts.json"):
        game_knowledge.rebuild_mod_knowledge(raw, tmp_path / "knowledge")

    incomplete = _required_act_rows()
    incomplete[0]["boss_encounters"] = []
    (raw / "acts.json").write_text(
        json.dumps(incomplete, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(game_knowledge.KnowledgeFormatError, match="boss_encounters"):
        game_knowledge.rebuild_mod_knowledge(raw, tmp_path / "knowledge")


def test_rebuild_generates_complete_defect_orb_knowledge(tmp_path: Path) -> None:
    """故障机器人应包含初始配置和五种充能球的完整机制知识。

    Args:
        tmp_path (Path): Pytest 提供的隔离知识目录。

    Raises:
        AssertionError: 名称解析、充能球机制或生成问答未达到固定契约。

    Returns:
        None: 此测试覆盖受控充能球补录到多问法生成的完整链路。
    """
    knowledge = tmp_path / "knowledge"
    raw = knowledge / "mod_export/v0.107.1/raw"
    raw.mkdir(parents=True)
    payloads = {
        "acts": _required_act_rows(),
        "cards": [
            {"id": "STRIKE_DEFECT", "name": "打击", "description": "造成6点伤害。"},
            {"id": "DEFEND_DEFECT", "name": "防御", "description": "获得5点格挡。"},
            {"id": "ZAP", "name": "电击", "description": "生成1个闪电充能球。"},
            {
                "id": "DUALCAST",
                "name": "双重释放",
                "description": "激发最旧的充能球两次。",
            },
        ],
        "relics": [
            {
                "id": "CRACKED_CORE",
                "name": "破损核心",
                "description": "战斗开始时生成1个闪电。",
            }
        ],
        "characters": [
            {
                "id": "DEFECT",
                "name": "故障机器人",
                "description": "故障机器人使用充能球持续产生效果，并可激发它们取得更强效果。",
                "starting_hp": 75,
                "starting_gold": 99,
                "max_energy": 3,
                "orb_slots": 3,
                "starting_deck": [
                    "STRIKE_DEFECT",
                    "STRIKE_DEFECT",
                    "STRIKE_DEFECT",
                    "STRIKE_DEFECT",
                    "DEFEND_DEFECT",
                    "DEFEND_DEFECT",
                    "DEFEND_DEFECT",
                    "DEFEND_DEFECT",
                    "ZAP",
                    "DUALCAST",
                ],
                "starting_relics": ["CRACKED_CORE"],
                "starting_potions": [],
            }
        ],
    }
    for category, rows in payloads.items():
        (raw / f"{category}.json").write_text(
            json.dumps(rows, ensure_ascii=False),
            encoding="utf-8",
        )
    supplement = knowledge / "supplements/v0.107.1/characters/defect_orbs.json"
    supplement.parent.mkdir(parents=True)
    supplement.write_text(
        json.dumps(
            {
                "captured_at_utc": "2026-08-27T00:00:00Z",
                "game_version": "0.107.1",
                "mod_version": "0.8.0",
                "provenance": "fixed-version-mod-snapshot",
                "orbs": [
                    {
                        "id": "LIGHTNING_ORB",
                        "name": "闪电",
                        "description": "充能球：对随机敌人造成伤害。",
                        "passive_base": 3,
                        "evoke_base": 8,
                        "trigger": "turn_end",
                        "focus_effect": "数值为基值,集中每+1被动与激发各+1",
                    },
                    {
                        "id": "FROST_ORB",
                        "name": "冰霜",
                        "description": "充能球：获得格挡。",
                        "passive_base": 2,
                        "evoke_base": 5,
                        "trigger": "turn_end",
                        "focus_effect": "数值为基值,集中每+1被动与激发各+1",
                    },
                    {
                        "id": "DARK_ORB",
                        "name": "黑暗",
                        "description": "充能球：激发时对生命最低的敌人造成伤害，每回合提高伤害。",
                        "passive_base": 6,
                        "evoke_base": None,
                        "trigger": "turn_end",
                        "focus_effect": "数值为基值,集中每+1仅被动+1,激发不变",
                    },
                    {
                        "id": "GLASS_ORB",
                        "name": "玻璃",
                        "description": "充能球：对所有敌人造成伤害，伤害随回合降低。",
                        "passive_base": 4,
                        "evoke_base": 8,
                        "trigger": "turn_end",
                        "focus_effect": "集中每+1使被动伤害+1,激发伤害+2",
                    },
                    {
                        "id": "PLASMA_ORB",
                        "name": "等离子",
                        "description": "充能球：获得能量。",
                        "passive_base": 1,
                        "evoke_base": 2,
                        "trigger": "turn_start",
                        "focus_effect": "不受集中影响",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = game_knowledge.rebuild_mod_knowledge(raw, knowledge)
    canonical = (result.output_root / "characters/DEFECT.md").read_text(
        encoding="utf-8"
    )
    generated = tmp_path / "generated-v0.107.1"
    game_knowledge.generate_question_variants(result.output_root, generated)
    rows = [
        json.loads(line)
        for line in (generated / "characters/DEFECT.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert "- 打击（STRIKE_DEFECT）" in canonical
    assert "- 破损核心（CRACKED_CORE）" in canonical
    assert (
        "supplement_source: supplements/v0.107.1/characters/defect_orbs.json"
        in canonical
    )
    assert (
        "supplement_origin: human_rl:data/mod_information/snapshots/"
        "v0.107.1/orbs.json" in canonical
    )
    assert "## 充能球：闪电" in canonical
    assert "## 充能球：等离子" in canonical
    assert len(rows) == 59
    lightning_mechanics = [
        row
        for row in rows
        if "闪电" in row["prompt"] and "被动（回合结束）3，激发8" in row["completion"]
    ]
    assert len(lightning_mechanics) == 5
    answers = [row["completion"].strip() for row in rows]
    assert (
        "初始HP75，金币99，每回合能量3，充能球槽3。"
        "初始遗物：破损核心。初始牌组（10张）："
        "打击×4、防御×4、电击、双重释放。"
    ) in answers
    assert "闪电、冰霜、黑暗、玻璃、等离子。" in answers
    assert any("被动（回合结束）3，激发8" in answer for answer in answers)
    assert any("仅被动+1，激发不变" in answer for answer in answers)
    assert all("激发0" not in answer for answer in answers)
    assert any("被动（回合开始）1，激发2" in answer for answer in answers)
    assert any("不受集中影响" in answer for answer in answers)

    original_supplement = json.loads(supplement.read_text(encoding="utf-8"))
    invalid_values = (("DARK_ORB", 0), ("LIGHTNING_ORB", None))
    for orb_id, evoke_base in invalid_values:
        invalid_supplement = json.loads(json.dumps(original_supplement))
        next(orb for orb in invalid_supplement["orbs"] if orb["id"] == orb_id)[
            "evoke_base"
        ] = evoke_base
        supplement.write_text(
            json.dumps(invalid_supplement, ensure_ascii=False),
            encoding="utf-8",
        )
        with pytest.raises(game_knowledge.KnowledgeFormatError, match="evoke_base"):
            game_knowledge.rebuild_mod_knowledge(raw, knowledge)
    supplement.write_text(
        json.dumps(original_supplement, ensure_ascii=False),
        encoding="utf-8",
    )

    (raw / "relics.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(game_knowledge.KnowledgeFormatError, match="无法解析显示名"):
        game_knowledge.rebuild_mod_knowledge(raw, knowledge)
    (raw / "relics.json").write_text(
        json.dumps(payloads["relics"], ensure_ascii=False),
        encoding="utf-8",
    )

    supplement.unlink()
    with pytest.raises(
        game_knowledge.KnowledgeFormatError, match="缺少故障机器人充能球"
    ):
        game_knowledge.rebuild_mod_knowledge(raw, knowledge)
