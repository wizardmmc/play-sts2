"""验证游戏知识的导入、实测导出与 Markdown 契约。"""

import json
from pathlib import Path

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
                "color": "defect",
                "upgrade": {"description": "生成1个闪电充能球。"},
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
                "color": "silent",
                "upgrade": {"description": "获得6点荆棘。"},
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

    root = tmp_path / "knowledge/mod_export/0.107.1"
    raw_cards = json.loads((root / "raw/cards.json").read_text(encoding="utf-8"))
    card = (root / "cards/ZAP.md").read_text(encoding="utf-8")
    upgraded_card = (root / "cards/ABRASIVE.md").read_text(encoding="utf-8")
    assert result.output_root == root
    assert result.entry_count == 2
    assert raw_cards[0]["id"] == "ZAP"
    assert "source: mod_export" in card
    assert "game_version: 0.107.1" in card
    assert "## 效果\n生成1个闪电充能球。" in card
    assert "## 升级" not in card
    assert "## 升级\n获得6点荆棘。" in upgraded_card

    export_mod_knowledge(EmptyGameDataClient(), tmp_path / "knowledge")

    assert not (root / "cards/ZAP.md").exists()
    assert not (root / "cards/ABRASIVE.md").exists()
