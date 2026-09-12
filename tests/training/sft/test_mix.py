"""验证 SFT 混合保留全部知识并只压缩高频人类行为。"""

from pathlib import Path

import pytest

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


def _human(
    sample_id: str,
    action: str,
    *,
    origin: str = "human",
) -> dict[str, object]:
    """构造一个最小人类行为行。

    Args:
        sample_id (str): 样本身份。
        action (str): 精确动作名。
        origin (str): 行为来自旧人类 raw 或人机协作 raw。

    Returns:
        dict[str, object]: 可供混合器分类的人类行为样本。
    """
    return {
        "sample_id": sample_id,
        "source": "human_play",
        "action": action,
        "behavior_origin": origin,
    }


def _arithmetic(sample_id: str, kind: str) -> dict[str, object]:
    """构造一个最小算术训练行。

    Args:
        sample_id (str): 样本身份。
        kind (str): 算术题型。

    Returns:
        dict[str, object]: 可供分层抽样的算术行。
    """
    return {
        "sample_id": sample_id,
        "source": "synthetic_arithmetic",
        "category": "arithmetic",
        "object_id": kind,
    }


def test_apply_sft_mix_keeps_all_knowledge_and_rare_actions() -> None:
    """混合器应保留全部知识，只裁剪训练集中的高频行为。

    Raises:
        AssertionError: 知识、稀有动作或留出行为被改写。

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
        human_train_action_limits={"play_card": 1},
    )

    train_ids = {row["sample_id"] for row in mixed["train"]}
    assert {
        "card-1",
        "card-2",
        "character-1",
        "ancient-1",
        "potion-1",
        "heal-1",
    } <= train_ids
    assert len(train_ids & {"play-1", "play-2"}) == 1
    assert mixed["dev"] == splits["dev"]
    assert mixed["test"] == splits["test"]


def test_apply_sft_mix_oversamples_selected_action_with_suffixed_ids() -> None:
    """按动作超采样应复制行为行并保留唯一样本 ID。

    Raises:
        AssertionError: 行为副本数量、样本后缀、身份唯一性或留出分卷被破坏。

    Returns:
        None: 此测试只检查内存中的确定性复制。
    """
    splits = {
        "train": [
            _human("claim-1", "claim_reward"),
            _human(
                "solver-claim-1",
                "claim_reward",
                origin="human_combat_solver",
            ),
            _human("dagger-claim-1", "claim_reward", origin="dagger"),
            _human("play-1", "play_card"),
        ],
        "dev": [_human("dev-claim-1", "claim_reward")],
        "test": [],
    }

    mixed = sft.apply_sft_mix(
        splits,
        seed=7,
        human_train_action_limits={},
        human_oversample_per_action={"claim_reward": 3},
    )

    train_ids = [row["sample_id"] for row in mixed["train"]]
    assert train_ids.count("claim-1") == 1
    assert train_ids.count("claim-1#o2") == 1
    assert train_ids.count("claim-1#o3") == 1
    assert train_ids.count("solver-claim-1#o2") == 1
    assert "dagger-claim-1#o2" not in train_ids
    assert "play-1#o2" not in train_ids
    assert len(set(train_ids)) == len(train_ids)
    assert mixed["dev"] == splits["dev"]


def test_apply_sft_mix_rejects_invalid_oversample_factor() -> None:
    """超采样倍数必须是不小于一的整数。

    Raises:
        AssertionError: 零或负倍数未被拒绝。

    Returns:
        None: 此测试只检查配方边界。
    """
    with pytest.raises(sft.DatasetBuildError):
        sft.apply_sft_mix(
            {"train": [], "dev": [], "test": []},
            seed=7,
            human_train_action_limits={},
            human_oversample_per_action={"claim_reward": 0},
        )


def test_apply_sft_mix_is_deterministic_without_knowledge_limits() -> None:
    """不限制行为时三分卷必须原样保留并可重复执行。

    Raises:
        AssertionError: 知识被抽掉或重复执行结果变化。

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
        "human_train_action_limits": {},
    }

    mixed = sft.apply_sft_mix(splits, **arguments)

    assert mixed == splits
    assert sft.apply_sft_mix(splits, **arguments) == mixed


def test_apply_sft_mix_keeps_teacher_and_dagger_data() -> None:
    """高频旧行为上限不应裁剪教师或学生状态 DAgger 标签。

    Raises:
        AssertionError: 教师、DAgger 行为被旧上限裁剪或算术数量不符合配方。

    Returns:
        None: 此测试只检查内存中的来源分层混合。
    """
    splits = {
        "train": [
            _human("old-play-1", "play_card"),
            _human("old-play-2", "play_card"),
            _human(
                "solver-play-1",
                "play_card",
                origin="human_combat_solver",
            ),
            _human(
                "solver-play-2",
                "play_card",
                origin="human_combat_solver",
            ),
            _human("dagger-play-1", "play_card", origin="dagger"),
            _human("dagger-play-2", "play_card", origin="dagger"),
            _arithmetic("orb-1", "orb_focus"),
            _arithmetic("orb-2", "orb_focus"),
            _arithmetic("status-1", "status_math"),
            _arithmetic("status-2", "status_math"),
            _arithmetic("status-3", "status_math"),
        ],
        "dev": [],
        "test": [],
    }

    mixed = sft.apply_sft_mix(
        splits,
        seed=7,
        human_train_action_limits={"play_card": 1},
        arithmetic_train_per_kind={"orb_focus": 1, "status_math": 2},
    )

    ids = {row["sample_id"] for row in mixed["train"]}
    assert {"solver-play-1", "solver-play-2"} <= ids
    assert {"dagger-play-1", "dagger-play-2"} <= ids
    assert len(ids & {"old-play-1", "old-play-2"}) == 1
    assert len(ids & {"orb-1", "orb-2"}) == 1
    assert len(ids & {"status-1", "status-2", "status-3"}) == 2


def test_load_sft_mix_reads_small_toml_recipe(tmp_path: Path) -> None:
    """混合配方只需种子和高频动作上限。

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

[human]
game_version = "v0.107.1"

    [human.train_max_per_action]
play_card = 600

[arithmetic.train_per_kind]
orb_focus = 100
status_math = 120
""",
        encoding="utf-8",
    )

    config = load_sft_mix(path)

    assert config.seed == 20260828
    assert config.human_game_version == "v0.107.1"
    assert config.human_train_action_limits == {"play_card": 600}
    assert config.human_oversample_per_action == {}
    assert config.arithmetic_train_per_kind == {
        "orb_focus": 100,
        "status_math": 120,
    }


def test_load_sft_mix_reads_oversample_recipe(tmp_path: Path) -> None:
    """配方可按动作声明超采样倍数并保留其余字段。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Raises:
        AssertionError: 倍数未读入或非法倍数被接受。

    Returns:
        None: 此测试只检查配置解析。
    """
    path = tmp_path / "mix.toml"
    path.write_text(
        """seed = 20260902

[human]
game_version = "v0.107.1"

[human.train_max_per_action]
claim_reward = 600

[human.oversample_per_action]
claim_reward = 6
""",
        encoding="utf-8",
    )

    config = sft.load_sft_mix(path)

    assert config.human_train_action_limits == {"claim_reward": 600}
    assert config.human_oversample_per_action == {"claim_reward": 6}

    broken = tmp_path / "broken.toml"
    broken.write_text(
        """seed = 20260902

[human.train_max_per_action]
claim_reward = 600

[human.oversample_per_action]
claim_reward = 0
""",
        encoding="utf-8",
    )
    with pytest.raises(sft.DatasetBuildError, match="无效 SFT 混合配方"):
        sft.load_sft_mix(broken)


def test_load_sft_mix_rejects_knowledge_limits(tmp_path: Path) -> None:
    """旧知识上限不能重新引入事实覆盖缺口。

    Args:
        tmp_path (Path): Pytest 提供的隔离目录。

    Raises:
        AssertionError: 配方仍允许按类别截断知识。

    Returns:
        None: 此测试只检查配置边界。
    """
    path = tmp_path / "mix.toml"
    path.write_text(
        """seed = 1

[knowledge.train]
cards = 1

[human.train_max_per_action]
""",
        encoding="utf-8",
    )

    with pytest.raises(sft.DatasetBuildError, match="不能再设置知识类别上限"):
        sft.load_sft_mix(path)
