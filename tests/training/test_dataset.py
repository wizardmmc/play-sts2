"""验证 Markdown 知识和精确人类决策到 SFT messages 的转换。"""

import json
from pathlib import Path

from play_sts2.training import build_sft_dataset


def test_build_sft_dataset_keeps_sources_and_uses_current_harness(
    tmp_path: Path,
) -> None:
    """构建可读 SFT 行，并使用当前 Harness 渲染行为样本。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 消息、来源、动作或清单不符合数据集契约。

    Returns:
        None: 此测试只检查数据集构建产物。
    """
    knowledge = tmp_path / "knowledge/web_wiki"
    (knowledge / "cards").mkdir(parents=True)
    (knowledge / "cards/ZAP.md").write_text(
        """---
id: ZAP
name: 电击
type: card
source: web_wiki
source_detail: spire-codex 2026-08-16
cost: 1
---
## 效果
生成1个闪电充能球。
""",
        encoding="utf-8",
    )
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    state = {
        "screen": "MAP",
        "available_actions": ["choose_map_node", "save_and_quit"],
        "run": {
            "character_name": "故障机器人",
            "ascension": 0,
            "act_id": 0,
            "floor": 1,
            "current_hp": 75,
            "max_hp": 75,
            "gold": 99,
            "relics": [],
            "potions": [],
            "deck": [],
        },
        "map": {
            "available_nodes": [
                {"index": 0, "row": 1, "col": 2, "node_type": "Monster"}
            ]
        },
    }
    decisions = [
        {
            "run_id": "RUN-001",
            "source_sequence": 3,
            "event_id": 8,
            "observed_at": "2026-08-27T02:00:02Z",
            "recorded_layer": "strategic",
            "before_state": {"screen": "CHARACTER_SELECT"},
            "action": "set_seed",
            "parameters": {"game_seed": "TEST-SEED"},
        },
        {
            "run_id": "RUN-001",
            "source_sequence": 4,
            "event_id": 9,
            "observed_at": "2026-08-27T02:00:03Z",
            "recorded_layer": "strategic",
            "before_state": state,
            "action": "choose_map_node",
            "parameters": {"option_index": 0},
        },
    ]
    (transcripts / "RUN-001.jsonl").write_text(
        "".join(
            json.dumps(
                decision,
                ensure_ascii=False,
            )
            + "\n"
            for decision in decisions
        ),
        encoding="utf-8",
    )

    result = build_sft_dataset(
        knowledge_root=knowledge,
        transcripts_root=transcripts,
        output_root=tmp_path / "dataset",
        dev_run_ids={"RUN-001"},
    )

    rows = [
        json.loads(line)
        for output_file in (result.train_path, result.dev_path)
        for line in output_file.read_text(encoding="utf-8").splitlines()
    ]
    assert result.train_count == 1
    assert result.dev_count == 1
    assert {row["source"] for row in rows} == {"web_wiki", "human_play"}
    knowledge_row = next(row for row in rows if row["source"] == "web_wiki")
    assert knowledge_row["messages"][0] == {
        "role": "user",
        "content": "请说明《杀戮尖塔 2》中的卡牌“电击”（ZAP）。",
    }
    assert "费用：1" in knowledge_row["messages"][1]["content"]
    assert "生成1个闪电充能球。" in knowledge_row["messages"][1]["content"]
    behavior_row = next(row for row in rows if row["source"] == "human_play")
    assert behavior_row["messages"][1]["role"] == "user"
    assert "[0] 第 1 行，第 2 列 | 普通敌人" in behavior_row["messages"][1]["content"]
    assert behavior_row["messages"][2] == {
        "role": "assistant",
        "content": "ACTION: choose_map_node 0",
    }
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["splits"] == {"train": 1, "dev": 1, "test": 0}
    assert manifest["sources"] == {"human_play": 1, "web_wiki": 1}
    assert manifest["skipped_environment_actions"] == {"set_seed": 1}


def test_build_sft_dataset_preserves_mod_game_version(tmp_path: Path) -> None:
    """Mod 知识行和 sample ID 保留具体游戏版本。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 派生数据或清单丢失 Mod 游戏版本。

    Returns:
        None: 此测试只检查来源追溯字段。
    """
    knowledge = tmp_path / "knowledge/mod_export/v0.107.1"
    (knowledge / "cards").mkdir(parents=True)
    (knowledge / "cards/ZAP.md").write_text(
        """---
id: ZAP
name: 电击
type: card
source: mod_export
game_version: v0.107.1
cost: 1
---
## 效果
生成1个闪电充能球。
""",
        encoding="utf-8",
    )
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()

    result = build_sft_dataset(
        knowledge_root=knowledge,
        transcripts_root=transcripts,
        output_root=tmp_path / "dataset",
    )
    row = json.loads(result.train_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert row["sample_id"] == "mod_export/v0.107.1/cards/ZAP"
    assert row["game_version"] == "v0.107.1"
    assert manifest["knowledge"]["game_versions"] == ["v0.107.1"]
