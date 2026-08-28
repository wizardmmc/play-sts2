"""验证 E3 数据混合只压缩高频项并保留稀有行为。"""

from pathlib import Path

from play_sts2.training import sft


def _knowledge(
    sample_id: str,
    category: str,
    *,
    fact: str | None = None,
) -> dict[str, object]:
    """构造一个无需 Harness 的最小知识行。

    Args:
        sample_id (str): 样本身份。
        category (str): 知识类别。
        fact (str | None): 同一事实共享的实体与答案标识。

    Returns:
        dict[str, object]: 可供混合器分类的知识样本。
    """
    fact = fact or category
    return {
        "sample_id": sample_id,
        "source": "knowledge",
        "category": category,
        "object_id": fact,
        "messages": [{"role": "assistant", "content": fact}],
    }


def _human(sample_id: str, action: str) -> dict[str, object]:
    """构造一个最小人类行为行。

    Args:
        sample_id (str): 样本身份。
        action (str): 精确动作名。

    Returns:
        dict[str, object]: 可供混合器分类的人类行为样本。
    """
    return {
        "sample_id": sample_id,
        "source": "human_play",
        "action": action,
    }


def test_apply_sft_mix_excludes_unselected_knowledge_and_keeps_rare_actions() -> None:
    """E3 混合应按类别限量知识，只裁剪训练集中的高频行为。

    Raises:
        AssertionError: 远古者未排除、稀有动作被裁剪或验证行为被改写。

    Returns:
        None: 此测试只检查内存中的确定性混合。
    """
    apply_sft_mix = getattr(sft, "apply_sft_mix", None)
    assert apply_sft_mix is not None, "尚未实现 E3 数据混合"
    splits = {
        "train": [
            _knowledge("card-1", "cards"),
            _knowledge("card-2", "cards"),
            _knowledge("character-1", "characters"),
            _knowledge("ancient-1", "ancients"),
            _human("play-1", "play_card"),
            _human("play-2", "play_card"),
            _human("potion-1", "use_potion"),
            _human("heal-1", "choose_rest_option"),
        ],
        "dev": [
            _knowledge("dev-card-1", "cards"),
            _knowledge("dev-character-1", "characters"),
            _knowledge("dev-ancient-1", "ancients"),
            _human("dev-play-1", "play_card"),
        ],
        "test": [_human("test-play-1", "play_card")],
    }

    mixed = apply_sft_mix(
        splits,
        seed=7,
        knowledge_limits={
            "train": {"cards": 1, "characters": -1, "ancients": 0},
            "dev": {"cards": 0, "characters": -1, "ancients": 0},
        },
        human_train_action_limits={"play_card": 1},
    )

    train_ids = {row["sample_id"] for row in mixed["train"]}
    assert len(train_ids & {"card-1", "card-2"}) == 1
    assert {"character-1", "potion-1", "heal-1"} <= train_ids
    assert "ancient-1" not in train_ids
    assert len(train_ids & {"play-1", "play-2"}) == 1
    assert [row["sample_id"] for row in mixed["dev"]] == [
        "dev-character-1",
        "dev-play-1",
    ]
    assert mixed["test"] == splits["test"]


def test_apply_sft_mix_keeps_dev_only_for_retained_train_facts() -> None:
    """自动验证问法必须对应混合后仍在训练集里的同一事实。

    Raises:
        AssertionError: 验证集保留了训练集中已经抽掉的事实。

    Returns:
        None: 此测试只检查事实对齐和重复执行结果。
    """
    splits = {
        "train": [
            _knowledge("train-a", "cards", fact="a"),
            _knowledge("train-b", "cards", fact="b"),
        ],
        "dev": [
            _knowledge("dev-a", "cards", fact="a"),
            _knowledge("dev-b", "cards", fact="b"),
        ],
        "test": [],
    }
    arguments = {
        "seed": 7,
        "knowledge_limits": {"train": {"cards": 1}, "dev": {"cards": -1}},
        "human_train_action_limits": {},
    }

    mixed = sft.apply_sft_mix(splits, **arguments)
    retained_fact = mixed["train"][0]["object_id"]

    assert [row["object_id"] for row in mixed["dev"]] == [retained_fact]
    assert sft.apply_sft_mix(splits, **arguments) == mixed


def test_load_sft_mix_reads_small_toml_recipe(tmp_path: Path) -> None:
    """混合配方只需种子、知识上限和高频动作上限。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Raises:
        AssertionError: TOML 配方没有转换成明确的内存配置。

    Returns:
        None: 此测试不构建真实数据集。
    """
    load_sft_mix = getattr(sft, "load_sft_mix", None)
    assert load_sft_mix is not None, "尚未实现 E3 混合配方读取"
    path = tmp_path / "mix.toml"
    path.write_text(
        """seed = 20260828

[knowledge.train]
characters = -1
ancients = 0

[knowledge.dev]
characters = -1
ancients = 0

[human.train_max_per_action]
play_card = 600
""",
        encoding="utf-8",
    )

    config = load_sft_mix(path)

    assert config.seed == 20260828
    assert config.knowledge_limits == {
        "train": {"characters": -1, "ancients": 0},
        "dev": {"characters": -1, "ancients": 0},
    }
    assert config.human_train_action_limits == {"play_card": 600}
