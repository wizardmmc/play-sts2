"""验证 Markdown 知识和精确人类决策到 SFT messages 的转换。"""

import hashlib
import json
from pathlib import Path

import pytest

import play_sts2.training.sft.dataset as dataset_module
from play_sts2.training import (
    DatasetBuildError,
    build_sft_dataset,
    validate_sft_dataset,
)


def _directory_rows(root: Path) -> list[dict[str, object]]:
    """递归读取一个目录分卷中的全部 JSONL 行。

    Args:
        root (Path): train、validation 或 eval 目录。

    Returns:
        list[dict[str, object]]: 按文件路径和行顺序排列的样本。
    """
    return [
        json.loads(line)
        for path in sorted(root.rglob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_formal_coverage_requires_every_arithmetic_and_behavior_bucket() -> None:
    """正式分卷缺少任一事实、算术题型或行为层都应拒绝发布。

    Raises:
        AssertionError: 缺失正式桶没有触发构建错误。

    Returns:
        None: 此测试直接检查发布前覆盖断言。
    """
    roles = {"train": "train", "validation": "dev", "eval": "test"}
    categories = {
        "cards",
        "characters",
        "enchantments",
        "encounters",
        "events",
        "keywords",
        "monsters",
        "potions",
        "powers",
        "relics",
    }
    objects_by_category = {
        category: ("OBJECT", "SECOND") if category == "cards" else ("OBJECT",)
        for category in categories
    }
    knowledge_rows = []
    for category in categories:
        for object_id in objects_by_category[category]:
            for role, split in roles.items():
                epochs: tuple[int | None, ...] = (
                    (1, 2, 3, 4, 5) if role == "train" else (None,)
                )
                for training_epoch in epochs:
                    epoch_suffix = training_epoch or role
                    row = {
                        "sample_id": (
                            f"{category}/{object_id}/fact/{role}/{epoch_suffix}"
                        ),
                        "source": "knowledge",
                        "category": category,
                        "object_id": object_id,
                        "fact_id": f"{category}/{object_id}/fact",
                        "question_role": role,
                        "dataset_split": split,
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    f"问题-{category}-{object_id}-{role}-{epoch_suffix}"
                                ),
                            },
                            {
                                "role": "assistant",
                                "content": f"答案-{category}-{object_id}",
                            },
                        ],
                    }
                    if training_epoch is not None:
                        row["training_epoch"] = training_epoch
                    knowledge_rows.append(row)
    buckets = {
        "block_math",
        "energy_math",
        "lethal",
        "multihit",
        "orb_focus",
        "status_math",
    }
    splits: dict[str, list[dict[str, object]]] = {}
    for role, split in roles.items():
        arithmetic = [
            {
                "source": "synthetic_arithmetic",
                "category": "arithmetic",
                "object_id": bucket,
                "case_id": f"{bucket}/{role}",
                "question_role": role,
            }
            for bucket in buckets
        ]
        behavior = [
            {"source": "human_play", "layer": "battle"},
            {"source": "human_play", "layer": "strategic"},
        ]
        splits[split] = (
            [row for row in knowledge_rows if row["question_role"] == role]
            + arithmetic
            + behavior
        )

    dataset_module._validate_formal_coverage(knowledge_rows, splits)
    invalid_epoch_rows = [dict(row) for row in knowledge_rows]
    invalid_epoch_splits = {
        split: [dict(row) for row in rows] for split, rows in splits.items()
    }
    for rows in (invalid_epoch_rows, invalid_epoch_splits["train"]):
        next(
            row
            for row in rows
            if row.get("fact_id") == "cards/OBJECT/fact"
            and row.get("training_epoch") == 5
        )["training_epoch"] = 4
    with pytest.raises(
        DatasetBuildError,
        match=r"cards/OBJECT/fact.*训练轮次.*\[1, 2, 3, 4, 5\]",
    ):
        dataset_module._validate_formal_coverage(
            invalid_epoch_rows,
            invalid_epoch_splits,
        )
    leaking_case_splits = {split: list(rows) for split, rows in splits.items()}
    leaking_case_splits["dev"] = [dict(row) for row in leaking_case_splits["dev"]]
    next(
        row
        for row in leaking_case_splits["dev"]
        if row.get("object_id") == "block_math"
    )["case_id"] = "block_math/train"
    with pytest.raises(DatasetBuildError, match="算术案例跨分卷重复.*block_math/train"):
        dataset_module._validate_formal_coverage(knowledge_rows, leaking_case_splits)

    missing_fact_splits = {split: list(rows) for split, rows in splits.items()}
    missing_fact_splits["dev"] = [
        row
        for row in missing_fact_splits["dev"]
        if row.get("fact_id") != "cards/OBJECT/fact"
    ]
    with pytest.raises(
        DatasetBuildError,
        match="validation 知识事实覆盖不完整.*cards/OBJECT/fact",
    ):
        dataset_module._validate_formal_coverage(knowledge_rows, missing_fact_splits)

    missing_bucket_splits = {split: list(rows) for split, rows in splits.items()}
    missing_bucket_splits["test"] = [
        row
        for row in missing_bucket_splits["test"]
        if row.get("object_id") != "orb_focus"
    ]
    with pytest.raises(DatasetBuildError, match="eval 算术题型覆盖不完整.*orb_focus"):
        dataset_module._validate_formal_coverage(knowledge_rows, missing_bucket_splits)


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

    assert result.train_path == tmp_path / "dataset/train"
    assert result.dev_path == tmp_path / "dataset/validation"
    assert result.test_path == tmp_path / "dataset/eval"
    rows = _directory_rows(result.train_path) + _directory_rows(result.dev_path)
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
    assert behavior_row["available_actions"] == ["choose_map_node"]
    assert behavior_row["legal_actions"] == ["ACTION: choose_map_node 0"]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["splits"] == {"train": 1, "validation": 1, "eval": 0}
    assert manifest["sources"] == {"human_play": 1, "web_wiki": 1}
    assert manifest["sources_by_split"] == {
        "train": {"web_wiki": 1},
        "validation": {"human_play": 1},
        "eval": {},
    }
    assert manifest["human"]["validation_runs"] == ["RUN-001"]
    behavior_audit = json.loads(
        (result.output_root / "behavior-audit.json").read_text(encoding="utf-8")
    )
    assert behavior_audit == manifest["behavior_audit"]
    assert behavior_audit["splits"]["validation"]["actions"] == {"choose_map_node": 1}
    assert set(manifest["files"]) == {
        "train/cards/ZAP.jsonl",
        "validation/strategy/RUN-001.jsonl",
    }


def test_behavior_audit_captures_explicit_decision_contrasts() -> None:
    """行为标签和审计应区分有真实选择条件的关键战略对照。

    Raises:
        AssertionError: 药水保留、火堆、商店或卡牌奖励对照没有被独立统计。

    Returns:
        None: 此测试只验证行为审计的决策语义。
    """
    potion_state = {
        "screen": "COMBAT",
        "in_combat": True,
        "available_actions": ["play_card", "use_potion", "end_turn"],
    }
    shop_state = {
        "screen": "SHOP",
        "available_actions": ["buy_card", "proceed"],
        "shop": {
            "cards": [{"index": 0, "is_stocked": True, "enough_gold": True}],
            "relics": [],
            "potions": [],
        },
    }
    reward_state = {
        "screen": "CARD_SELECTION",
        "available_actions": ["choose_reward_card", "skip_reward_cards"],
    }
    assert "potion:hold" in dataset_module._behavior_tags(potion_state, "end_turn", {})
    assert "shop:purchase" in dataset_module._behavior_tags(
        shop_state, "buy_card", {"option_index": 0}
    )
    assert "shop:leave" in dataset_module._behavior_tags(shop_state, "proceed", {})
    assert "card_reward:choose" in dataset_module._behavior_tags(
        reward_state, "choose_reward_card", {"option_index": 0}
    )
    assert "card_reward:skip" in dataset_module._behavior_tags(
        reward_state, "skip_reward_cards", {}
    )

    training_tags = [
        ["potion:use"],
        ["potion:hold"],
        ["rest:heal"],
        ["rest:smith"],
        ["shop:purchase"],
        ["shop:leave"],
        ["card_reward:choose"],
        ["card_reward:skip"],
    ]
    audit = dataset_module._behavior_audit(
        {
            "train": [
                {
                    "source": "human_play",
                    "action": f"action-{index}",
                    "screen": "TEST",
                    "tags": tags,
                }
                for index, tags in enumerate(training_tags)
            ],
            "dev": [],
            "test": [],
        }
    )

    assert audit["training_contrasts"] == {
        "card_reward_choose_vs_skip": {
            "available": True,
            "counts": {"choose": 1, "skip": 1},
        },
        "potion_use_vs_hold": {
            "available": True,
            "counts": {"hold": 1, "use": 1},
        },
        "rest_heal_vs_smith": {
            "available": True,
            "counts": {"heal": 1, "smith": 1},
        },
        "shop_purchase_vs_leave": {
            "available": True,
            "counts": {"leave": 1, "purchase": 1},
        },
    }
    assert audit["missing_training_contrasts"] == []


def test_behavior_row_rejects_reference_outside_exact_legal_actions() -> None:
    """参考动作参数越过当前状态索引时不得发布为行为监督。

    Raises:
        AssertionError: 非法参考动作没有触发数据构建失败。

    Returns:
        None: 此测试钉住行为行的合法集合自证。
    """
    decision = {
        "run_id": "RUN-ILLEGAL",
        "event_id": 1,
        "before_state": {
            "screen": "MAP",
            "available_actions": ["choose_map_node"],
            "map": {
                "available_nodes": [
                    {"index": 0, "row": 1, "col": 0, "node_type": "Monster"}
                ]
            },
        },
        "action": "choose_map_node",
        "parameters": {"option_index": 9},
    }

    with pytest.raises(DatasetBuildError, match="动作参数不在当前合法边界"):
        dataset_module._behavior_row(decision)


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
    row = _directory_rows(result.train_path)[0]
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
    (knowledge / "cards").mkdir(parents=True)
    (knowledge / "cards/ZAP.md").write_text(
        """---
id: ZAP
name: 电击
type: card
source: mod_export
game_version: v0.107.1
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
    changed = result.train_path / "cards/ZAP.jsonl"
    changed.write_text('{"partial":true}\n', encoding="utf-8")

    with pytest.raises(DatasetBuildError, match="train/cards/ZAP.jsonl.*SHA-256"):
        validate_sft_dataset(result.output_root)

    build_sft_dataset(
        knowledge_root=knowledge,
        human_root=human,
        output_root=result.output_root,
    )
    (result.output_root / "behavior-audit.json").write_text(
        '{"format":"tampered"}\n', encoding="utf-8"
    )
    with pytest.raises(DatasetBuildError, match="行为审计与 manifest 不一致"):
        validate_sft_dataset(result.output_root)

    build_sft_dataset(
        knowledge_root=knowledge,
        human_root=human,
        output_root=result.output_root,
    )
    (result.output_root / "train.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(DatasetBuildError, match="旧聚合.*train.jsonl"):
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

    output = _directory_rows(result.train_path)
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


def test_build_sft_dataset_preserves_five_training_epochs_and_two_holdouts(
    tmp_path: Path,
) -> None:
    """五轮训练问法与两套留出问法应保留各自用途。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 训练轮次或 question_role 没有保留到正式数据。

    Returns:
        None: 此测试只检查知识问法分卷契约。
    """
    knowledge = tmp_path / "generated-v0.107.1/cards"
    knowledge.mkdir(parents=True)
    rows = [
        {
            "category": "cards",
            "object_id": "ZAP",
            "fact_id": "cards/ZAP/cost",
            "question_role": "train",
            "training_epoch": 1,
            "source": "mod_export+curated_override",
            "prompt": "Q: 电击的费用是多少？\nA:",
            "completion": " 1点能量。",
        },
        {
            "category": "cards",
            "object_id": "ZAP",
            "fact_id": "cards/ZAP/cost",
            "question_role": "train",
            "training_epoch": 2,
            "source": "mod_export+curated_override",
            "prompt": "Q: 打出电击需要几点能量？\nA:",
            "completion": " 1点能量。",
        },
        {
            "category": "cards",
            "object_id": "ZAP",
            "fact_id": "cards/ZAP/cost",
            "question_role": "train",
            "training_epoch": 3,
            "source": "mod_export+curated_override",
            "prompt": "Q: 未受修正时电击消耗多少资源？\nA:",
            "completion": " 1点能量。",
        },
        {
            "category": "cards",
            "object_id": "ZAP",
            "fact_id": "cards/ZAP/cost",
            "question_role": "train",
            "training_epoch": 4,
            "source": "mod_export+curated_override",
            "prompt": "Q: 电击的基础资源消耗是什么？\nA:",
            "completion": " 1点能量。",
        },
        {
            "category": "cards",
            "object_id": "ZAP",
            "fact_id": "cards/ZAP/cost",
            "question_role": "train",
            "training_epoch": 5,
            "source": "mod_export+curated_override",
            "prompt": "Q: 请只回答电击的原始费用。\nA:",
            "completion": " 1点能量。",
        },
        {
            "category": "cards",
            "object_id": "ZAP",
            "fact_id": "cards/ZAP/cost",
            "question_role": "validation",
            "source": "mod_export+curated_override",
            "prompt": "Q: 电击的基础能量消耗是什么？\nA:",
            "completion": " 1点能量。",
        },
        {
            "category": "cards",
            "object_id": "ZAP",
            "fact_id": "cards/ZAP/cost",
            "question_role": "eval",
            "source": "mod_export+curated_override",
            "prompt": "Q: 未受修正时，电击要支付多少能量？\nA:",
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

    train = _directory_rows(result.train_path)
    validation = _directory_rows(result.dev_path)
    evaluation = _directory_rows(result.test_path)
    assert result.train_count == 5
    assert result.dev_count == 1
    assert result.test_count == 1
    assert {row["object_id"] for row in train + validation + evaluation} == {"ZAP"}
    assert sorted(row["training_epoch"] for row in train) == [1, 2, 3, 4, 5]
    assert train[0]["messages"][0] != validation[0]["messages"][0]
    assert (result.train_path / "cards/ZAP.jsonl").is_file()
    assert (result.dev_path / "cards/ZAP.jsonl").is_file()
    assert (result.test_path / "cards/ZAP.jsonl").is_file()


def test_build_sft_dataset_honors_generated_arithmetic_question_roles(
    tmp_path: Path,
) -> None:
    """独立 seed 生成的三套算术题应进入各自目录。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 算术行的显式用途没有映射到三个目录。

    Returns:
        None: 此测试只检查算术候选的分卷提示。
    """
    knowledge = tmp_path / "generated-v0.107.1/arithmetic"
    (knowledge / "train").mkdir(parents=True)
    (knowledge / "validation").mkdir()
    (knowledge / "eval").mkdir()
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
                "fact_id": "arithmetic/block_math/train-0001",
                "question_role": "train",
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
                "fact_id": "arithmetic/block_math/validation-0001",
                "question_role": "validation",
                "prompt": "Q: 验证算术题。请写出计算过程。\nA:",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (knowledge / "eval/cot.jsonl").write_text(
        json.dumps(
            {
                **common,
                "fact_id": "arithmetic/block_math/eval-0001",
                "question_role": "eval",
                "prompt": "Q: 最终算术题。请写出计算过程。\nA:",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (knowledge / "_manifest.json").write_text(
        json.dumps(
            {
                "format": "prompt_completion_candidates",
                "source": "data/raw/human/*/combat/*.jsonl",
                "human_root": "data/raw/human",
                "training_run_ids": ["RUN-TRAIN"],
                "seed": 20260824,
            }
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

    train = _directory_rows(result.train_path)[0]
    validation = _directory_rows(result.dev_path)[0]
    evaluation = _directory_rows(result.test_path)[0]
    assert train["messages"][0]["content"].startswith("Q: 训练")
    assert validation["messages"][0]["content"].startswith("Q: 验证")
    assert evaluation["messages"][0]["content"].startswith("Q: 最终")
    assert {train["source"], validation["source"], evaluation["source"]} == {
        "synthetic_arithmetic"
    }
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["sources_by_split"] == {
        "train": {"synthetic_arithmetic": 1},
        "validation": {"synthetic_arithmetic": 1},
        "eval": {"synthetic_arithmetic": 1},
    }
    assert manifest["arithmetic"] == {
        "candidate_manifest": "arithmetic/_manifest.json",
        "candidate_manifest_sha256": hashlib.sha256(
            (knowledge / "_manifest.json").read_bytes()
        ).hexdigest(),
        "human_root": "data/raw/human",
        "source": "data/raw/human/*/combat/*.jsonl",
        "training_run_ids": ["RUN-TRAIN"],
        "seed": 20260824,
    }


def test_build_sft_dataset_applies_explicit_mix_recipe(tmp_path: Path) -> None:
    """正式构建应记录只限制人类动作的混合配方。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Raises:
        AssertionError: 知识被混合器删除或 manifest 未记录配方。

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

    rows = _directory_rows(result.train_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert len(rows) == 3
    assert {row["category"] for row in rows} == {"cards", "ancients"}
    assert manifest["mix"] == {
        "seed": 3,
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


def test_build_sft_dataset_rejects_prompt_reused_across_roles(tmp_path: Path) -> None:
    """相同知识问题不能同时进入训练和最终评测。

    Args:
        tmp_path (Path): Pytest 提供的隔离数据目录。

    Raises:
        AssertionError: 跨用途重复问题没有触发失败。

    Returns:
        None: 此测试只检查三棵目录之间的问题泄漏。
    """
    knowledge = tmp_path / "generated-v0.107.1/cards"
    knowledge.mkdir(parents=True)
    rows = [
        {
            "category": "cards",
            "object_id": "ZAP",
            "fact_id": "cards/ZAP/cost",
            "question_role": role,
            "source": "mod_export+curated_override",
            "prompt": "Q: 电击的费用是多少？\nA:",
            "completion": " 1点能量。",
        }
        for role in ("train", "eval")
    ]
    (knowledge / "ZAP.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    human = tmp_path / "raw/human"
    human.mkdir(parents=True)

    with pytest.raises(DatasetBuildError, match="知识问题跨分卷重复.*电击的费用"):
        build_sft_dataset(
            knowledge_root=knowledge.parent,
            human_root=human,
            output_root=tmp_path / "dataset",
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
