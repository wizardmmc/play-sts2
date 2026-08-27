"""验证 Markdown 知识和精确人类决策到 SFT messages 的转换。"""

import json
from pathlib import Path

import pytest

from play_sts2.training import (
    DatasetBuildError,
    build_sft_dataset,
    validate_sft_dataset,
)


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
    human = tmp_path / "raw/human"
    run_dir = human / "RUN-001"
    (run_dir / "strategy").mkdir(parents=True)
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
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "RUN-001",
                "source": "human",
                "termination_reason": "game_over",
                "training_eligible": True,
                "integrity": {"verified": True, "ineligibility_reasons": []},
                "battle_count": 0,
                "battle_sample_count": 0,
                "strategic_sample_count": 1,
            }
        ),
        encoding="utf-8",
    )
    decision = {
        "run_id": "RUN-001",
        "source_sequence": 4,
        "event_id": "human-confirmed:map-floor1-1",
        "observed_at": "2026-08-27T02:00:03Z",
        "layer": "strategic",
        "before_state": state,
        "action": "choose_map_node",
        "parameters": {"option_index": 0},
    }
    (run_dir / "strategy/decisions.jsonl").write_text(
        json.dumps(decision, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (human / "splits.json").write_text(
        json.dumps({"train": [], "dev": ["RUN-001"], "test": []}),
        encoding="utf-8",
    )

    result = build_sft_dataset(
        knowledge_root=knowledge,
        human_root=human,
        output_root=tmp_path / "dataset",
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
    assert behavior_row["sample_id"] == (
        "human_play/RUN-001/human-confirmed:map-floor1-1"
    )
    assert behavior_row["messages"][1]["role"] == "user"
    assert "[0] 第 1 行，第 2 列 | 普通敌人" in behavior_row["messages"][1]["content"]
    assert behavior_row["messages"][2] == {
        "role": "assistant",
        "content": "ACTION: choose_map_node 0",
    }
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["splits"] == {"train": 1, "dev": 1, "test": 0}
    assert manifest["sources"] == {"human_play": 1, "web_wiki": 1}
    assert manifest["human"]["dev_runs"] == ["RUN-001"]


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
    human = tmp_path / "raw/human"
    human.mkdir(parents=True)

    result = build_sft_dataset(
        knowledge_root=knowledge,
        human_root=human,
        output_root=tmp_path / "dataset",
    )
    row = json.loads(result.train_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert row["sample_id"] == "mod_export/v0.107.1/cards/ZAP"
    assert row["game_version"] == "v0.107.1"
    assert manifest["knowledge"]["game_versions"] == ["v0.107.1"]


def test_validate_sft_dataset_rejects_split_changed_after_manifest(
    tmp_path: Path,
) -> None:
    """任一分卷在 manifest 发布后变化时训练入口拒绝混合代数据。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 分卷哈希不一致仍通过完整性校验。

    Returns:
        None: 此测试只检查发布后的 fail-closed 读取。
    """
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    human = tmp_path / "raw/human"
    human.mkdir(parents=True)
    result = build_sft_dataset(
        knowledge_root=knowledge,
        human_root=human,
        output_root=tmp_path / "dataset",
    )
    result.train_path.write_text('{"partial":true}\n', encoding="utf-8")

    with pytest.raises(DatasetBuildError, match="train.jsonl.*SHA-256"):
        validate_sft_dataset(result.output_root)


def test_build_sft_dataset_reads_curated_questions_and_keeps_battle_steps_independent(
    tmp_path: Path,
) -> None:
    """保留知识变化问法，并把同一战斗的动作输出为独立单步样本。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 知识问法或独立战斗步骤没有进入训练集。

    Returns:
        None: 此测试只检查新的统一数据入口。
    """
    knowledge = tmp_path / "game_knowledge/curated-v0.107.1"
    (knowledge / "characters").mkdir(parents=True)
    (knowledge / "characters/DEFECT.jsonl").write_text(
        json.dumps(
            {
                "category": "characters",
                "object_id": "DEFECT",
                "source": "mod_snapshot/v0.107.1",
                "prompt": "Q: 故障机器人的初始配置？\nA:",
                "completion": " 初始HP75，能量3。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    human = tmp_path / "raw/human"
    run = human / "RUN-A"
    (run / "combat").mkdir(parents=True)
    (run / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "RUN-A",
                "source": "human",
                "termination_reason": "game_over",
                "training_eligible": True,
                "integrity": {"verified": True, "ineligibility_reasons": []},
                "battle_count": 1,
                "battle_sample_count": 2,
                "strategic_sample_count": 0,
            }
        ),
        encoding="utf-8",
    )
    state = {
        "screen": "COMBAT",
        "in_combat": True,
        "available_actions": ["end_turn"],
        "run": {
            "character_name": "故障机器人",
            "ascension": 1,
            "act_id": 0,
            "floor": 2,
            "current_hp": 70,
            "max_hp": 75,
            "gold": 99,
            "relics": [],
            "potions": [],
            "deck": [],
        },
        "combat": {
            "player": {
                "current_hp": 70,
                "max_hp": 75,
                "block": 0,
                "energy": 3,
                "stars": 0,
                "focus": 0,
                "powers": [],
                "orbs": [],
            },
            "enemies": [],
            "hand": [],
            "draw_count": 0,
            "discard_count": 0,
        },
    }
    rows = [
        {
            "event_id": event_id,
            "observed_at": f"2026-08-27T08:00:0{event_id}Z",
            "before_state": state,
            "action": "end_turn",
            "parameters": {},
        }
        for event_id in (1, 2)
    ]
    (run / "combat/battle-f002-01.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = build_sft_dataset(
        knowledge_root=knowledge,
        human_root=human,
        output_root=tmp_path / "datasets/sft",
    )

    output = [
        json.loads(line)
        for line in result.train_path.read_text(encoding="utf-8").splitlines()
    ]
    assert output[0]["messages"] == [
        {"role": "user", "content": "Q: 故障机器人的初始配置？"},
        {"role": "assistant", "content": "初始HP75，能量3。"},
    ]
    battles = [row for row in output if row.get("layer") == "battle"]
    assert len(battles) == 2
    assert all(
        [message["role"] for message in battle["messages"]]
        == ["system", "user", "assistant"]
        for battle in battles
    )
    assert [battle["action"] for battle in battles] == ["end_turn", "end_turn"]
    assert all(battle["screen"] == "COMBAT" for battle in battles)


def test_build_sft_dataset_excludes_training_ineligible_runs(tmp_path: Path) -> None:
    """Raw meta 明确标记不可训练时跳过整局。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 完整性审计未通过的局进入派生数据。

    Returns:
        None: 此测试只检查 raw meta 的准入边界。
    """
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    human = tmp_path / "raw/human"
    run = human / "REJECTED"
    (run / "strategy").mkdir(parents=True)
    (run / "meta.json").write_text(
        json.dumps(
            {
                "run_id": "REJECTED",
                "termination_reason": "legacy_import",
                "training_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    (run / "strategy/decisions.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "human_play/REJECTED/1",
                "messages": [
                    {"role": "user", "content": "状态"},
                    {"role": "assistant", "content": "ACTION: end_turn"},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    result = build_sft_dataset(
        knowledge_root=knowledge,
        human_root=human,
        output_root=tmp_path / "dataset",
    )

    assert result.train_count == 0


def test_build_sft_dataset_rejects_published_run_with_truncated_samples(
    tmp_path: Path,
) -> None:
    """可训练局的 meta 计数与实际分片不一致时拒绝构建。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 截断的已发布 raw 被静默当作完整训练数据。

    Returns:
        None: 此测试只检查 fail-closed 准入。
    """
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    human = tmp_path / "raw/human"
    run = human / "TRUNCATED"
    (run / "strategy").mkdir(parents=True)
    (run / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "TRUNCATED",
                "termination_reason": "game_over",
                "training_eligible": True,
                "integrity": {"verified": True, "ineligibility_reasons": []},
                "battle_count": 0,
                "battle_sample_count": 0,
                "strategic_sample_count": 2,
            }
        ),
        encoding="utf-8",
    )
    (run / "strategy/decisions.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(DatasetBuildError, match="strategic_sample_count"):
        build_sft_dataset(
            knowledge_root=knowledge,
            human_root=human,
            output_root=tmp_path / "dataset",
        )


def test_build_sft_dataset_ignores_unpublished_recordings(tmp_path: Path) -> None:
    """构建器不读取隐藏临时局或缺少发布标记的目录。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 录制中的数据进入派生训练集。

    Returns:
        None: 此测试只检查发布边界。
    """
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    human = tmp_path / "raw/human"
    for name in (".recording-IN-PROGRESS", "VISIBLE-BUT-UNPUBLISHED"):
        run = human / name
        (run / "strategy").mkdir(parents=True)
        (run / "meta.json").write_text(
            json.dumps({"run_id": name, "termination_reason": None}),
            encoding="utf-8",
        )
        (run / "strategy/decisions.jsonl").write_text(
            "this is intentionally incomplete\n",
            encoding="utf-8",
        )

    result = build_sft_dataset(
        knowledge_root=knowledge,
        human_root=human,
        output_root=tmp_path / "dataset",
    )

    assert result.train_count == 0
