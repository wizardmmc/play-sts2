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
                "schema_version": 2,
                "run_id": "RUN-001",
                "source": "human",
                "termination_reason": "stream_interrupted",
                "training_eligible": True,
                "recording_complete": False,
                "integrity": {
                    "samples_verified": True,
                    "ineligibility_reasons": [],
                },
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

    assert result.train_path == tmp_path / "dataset/train.jsonl"
    assert result.dev_path == tmp_path / "dataset/validation/dev.jsonl"
    assert result.test_path == tmp_path / "dataset/eval/test.jsonl"
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
    assert manifest["sources_by_split"] == {
        "train": {"web_wiki": 1},
        "dev": {"human_play": 1},
        "test": {},
    }
    assert manifest["human"]["dev_runs"] == ["RUN-001"]
    assert set(manifest["files"]) == {
        "train.jsonl",
        "validation/dev.jsonl",
        "eval/test.jsonl",
    }


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
                "schema_version": 2,
                "run_id": "RUN-A",
                "source": "human",
                "termination_reason": "game_over",
                "training_eligible": True,
                "recording_complete": True,
                "integrity": {
                    "samples_verified": True,
                    "ineligibility_reasons": [],
                },
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
        "turn": 1,
        "available_actions": ["end_turn"],
        "run": {
            "character_name": "故障机器人",
            "ascension": 1,
            "act_id": 0,
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
    (human / "splits.json").write_text(
        json.dumps({"train": ["RUN-A"], "dev": [], "test": []}),
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
    assert all(
        "- [0] 破损核心: 战斗开始时生成1个闪电充能球。"
        in battle["messages"][0]["content"]
        for battle in battles
    )
    assert all(
        "【当前回合】" not in battle["messages"][0]["content"] for battle in battles
    )
    assert all(
        battle["messages"][1]["content"].startswith("玩家:")
        and "角色:" not in battle["messages"][1]["content"]
        and "牌组 " not in battle["messages"][1]["content"]
        for battle in battles
    )


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


def test_build_sft_dataset_holds_out_one_question_form_for_validation(
    tmp_path: Path,
) -> None:
    """同一知识对象有多种问法时，固定留出一种进入验证集。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 问法没有按对象留出，或相同问题同时出现在训练和验证集。

    Returns:
        None: 此测试只检查知识问法分卷契约。
    """
    knowledge = tmp_path / "generated-v0.107.1/cards"
    knowledge.mkdir(parents=True)
    rows = [
        {
            "category": "cards",
            "object_id": "ZAP",
            "source": "mod_export+curated_override",
            "prompt": "Q: 电击的费用是多少？\nA:",
            "completion": " 1点能量。",
        },
        {
            "category": "cards",
            "object_id": "ZAP",
            "source": "mod_export+curated_override",
            "prompt": "Q: 打出电击需要几点能量？\nA:",
            "completion": " 1点能量。",
        },
    ]
    (knowledge / "ZAP.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    human = tmp_path / "raw/human"
    human.mkdir(parents=True)

    result = build_sft_dataset(
        knowledge_root=knowledge.parent,
        human_root=human,
        output_root=tmp_path / "dataset",
    )

    train = [json.loads(line) for line in result.train_path.read_text().splitlines()]
    validation = [json.loads(line) for line in result.dev_path.read_text().splitlines()]
    assert result.train_count == 1
    assert result.dev_count == 1
    assert train[0]["object_id"] == validation[0]["object_id"] == "ZAP"
    assert train[0]["messages"][0] != validation[0]["messages"][0]


def test_build_sft_dataset_honors_generated_arithmetic_validation_split(
    tmp_path: Path,
) -> None:
    """独立 seed 生成的算术验证题应进入验证目录而不是训练集。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 算术行的显式 train/dev 用途被自动问法留出覆盖。

    Returns:
        None: 此测试只检查算术候选的分卷提示。
    """
    knowledge = tmp_path / "generated-v0.107.1/arithmetic"
    (knowledge / "train").mkdir(parents=True)
    (knowledge / "validation").mkdir()
    common = {
        "category": "arithmetic",
        "object_id": "block_math",
        "source": "synthetic_arithmetic",
        "completion": " 所以HP是18。",
    }
    (knowledge / "train/cot.jsonl").write_text(
        json.dumps(
            {
                **common,
                "split": "train",
                "prompt": "Q: 训练算术题。请写出计算过程。\nA:",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (knowledge / "validation/cot.jsonl").write_text(
        json.dumps(
            {
                **common,
                "split": "dev",
                "prompt": "Q: 验证算术题。请写出计算过程。\nA:",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    human = tmp_path / "raw/human"
    human.mkdir(parents=True)

    result = build_sft_dataset(
        knowledge_root=knowledge.parent,
        human_root=human,
        output_root=tmp_path / "dataset",
    )

    train = json.loads(result.train_path.read_text(encoding="utf-8"))
    validation = json.loads(result.dev_path.read_text(encoding="utf-8"))
    assert train["messages"][0]["content"].startswith("Q: 训练")
    assert validation["messages"][0]["content"].startswith("Q: 验证")
    assert train["source"] == validation["source"] == "synthetic_arithmetic"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["sources_by_split"] == {
        "train": {"synthetic_arithmetic": 1},
        "dev": {"synthetic_arithmetic": 1},
        "test": {},
    }


def test_build_sft_dataset_applies_explicit_mix_recipe(tmp_path: Path) -> None:
    """正式构建应在分卷后应用 E3 类别上限并记录实际配方。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Raises:
        AssertionError: 远古者进入数据集、卡牌上限失效或 manifest 未记录配方。

    Returns:
        None: 此测试只构建最小知识数据集。
    """
    knowledge = tmp_path / "generated-v0.107.1"
    (knowledge / "cards").mkdir(parents=True)
    (knowledge / "ancients").mkdir()
    for category, count in (("cards", 2), ("ancients", 1)):
        rows = [
            {
                "category": category,
                "object_id": f"{category}-{index}",
                "source": "knowledge",
                "prompt": f"Q: {category}-{index}？\nA:",
                "completion": f" {category}-{index}。",
            }
            for index in range(count)
        ]
        (knowledge / category / "rows.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    human = tmp_path / "human"
    human.mkdir()
    mix = tmp_path / "mix.toml"
    mix.write_text(
        """seed = 3

[knowledge.train]
cards = 1
ancients = 0

[knowledge.dev]
cards = 0
ancients = 0

[human.train_max_per_action]
""",
        encoding="utf-8",
    )

    try:
        result = build_sft_dataset(
            knowledge_root=knowledge,
            human_root=human,
            output_root=tmp_path / "dataset",
            mix_config_path=mix,
        )
    except TypeError as exc:
        pytest.fail(f"尚未接入 E3 混合配方: {exc}")

    rows = [json.loads(line) for line in result.train_path.read_text().splitlines()]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["category"] == "cards"
    assert manifest["mix"] == {
        "seed": 3,
        "knowledge_limits": {
            "train": {"cards": 1, "ancients": 0},
            "dev": {"cards": 0, "ancients": 0},
        },
        "human_train_action_limits": {},
    }


def test_build_sft_dataset_rejects_unassigned_eligible_run(tmp_path: Path) -> None:
    """可训练的人类局不在归属名册中时不得静默进入训练集。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 未分配整局没有触发构建失败。

    Returns:
        None: 此测试只检查整局 fail-closed 分卷。
    """
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    human = tmp_path / "raw/human"
    run = human / "UNASSIGNED"
    (run / "combat").mkdir(parents=True)
    state = {
        "screen": "COMBAT",
        "in_combat": True,
        "turn": 1,
        "available_actions": ["end_turn"],
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
        "combat": {
            "player": {
                "current_hp": 75,
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
    (run / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "UNASSIGNED",
                "termination_reason": "game_over",
                "training_eligible": True,
                "recording_complete": True,
                "integrity": {
                    "samples_verified": True,
                    "ineligibility_reasons": [],
                },
                "battle_count": 1,
                "battle_sample_count": 1,
                "strategic_sample_count": 0,
            }
        ),
        encoding="utf-8",
    )
    (run / "combat/battle-f001-01.jsonl").write_text(
        json.dumps(
            {
                "event_id": 1,
                "observed_at": "2026-08-28T00:00:00Z",
                "before_state": state,
                "action": "end_turn",
                "parameters": {},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (human / "splits.json").write_text(
        json.dumps({"train": [], "dev": [], "test": []}),
        encoding="utf-8",
    )

    with pytest.raises(DatasetBuildError, match="未分配.*UNASSIGNED"):
        build_sft_dataset(
            knowledge_root=knowledge,
            human_root=human,
            output_root=tmp_path / "dataset",
        )


def test_build_sft_dataset_rejects_probe_prompt_in_training(tmp_path: Path) -> None:
    """训练问题与知识考试卷完全相同时必须拒绝发布数据集。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 被训练集污染的 probe 没有触发失败。

    Returns:
        None: 此测试只检查跨集合问题泄漏。
    """
    knowledge = tmp_path / "generated-v0.107.1/cards"
    knowledge.mkdir(parents=True)
    knowledge_rows = [
        {
            "category": "cards",
            "object_id": "ZAP",
            "source": "mod_export+curated_override",
            "prompt": "Q: 电击的费用是多少？\nA:",
            "completion": " 1点能量。",
        },
        {
            "category": "cards",
            "object_id": "DEFEND",
            "source": "mod_export+curated_override",
            "prompt": "Q: 防御能提供多少格挡？\nA:",
            "completion": " 5点格挡。",
        },
    ]
    (knowledge / "cards.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in knowledge_rows),
        encoding="utf-8",
    )
    probes = tmp_path / "eval/knowledge"
    probes.mkdir(parents=True)
    probe_rows = [
        {
            "kind": "form_holdout",
            "category": "cards",
            "object_id": "ZAP",
            "prompt": "电击的费用是多少？",
            "reference": "1点能量。",
        },
        {
            "kind": "form_holdout",
            "category": "cards",
            "object_id": "DEFEND",
            "prompt": "防御能提供多少格挡？",
            "reference": "5点格挡。",
        },
    ]
    (probes / "probes_recall.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in probe_rows),
        encoding="utf-8",
    )
    human = tmp_path / "raw/human"
    human.mkdir(parents=True)

    with pytest.raises(
        DatasetBuildError,
        match="2 个 probe.*电击的费用.*防御能提供多少格挡",
    ):
        build_sft_dataset(
            knowledge_root=knowledge.parent,
            human_root=human,
            output_root=tmp_path / "dataset",
            knowledge_probe_root=probes,
        )


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
                "schema_version": 2,
                "run_id": "TRUNCATED",
                "termination_reason": "game_over",
                "training_eligible": True,
                "recording_complete": True,
                "integrity": {
                    "samples_verified": True,
                    "ineligibility_reasons": [],
                },
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
