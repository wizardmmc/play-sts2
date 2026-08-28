"""覆盖游戏知识离线构建命令的命令行契约。"""

from pathlib import Path

from play_sts2.game_knowledge import (
    ArithmeticBuildResult,
    KnowledgeBuildResult,
    cli,
)


def test_rebuild_command_passes_controlled_supplement_sources(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """离线重建应显式传入原始快照、Wiki 和实跳记录。

    Args:
        tmp_path (Path): Pytest 提供的隔离临时目录。
        monkeypatch (pytest.MonkeyPatch): 用于替换真实重建实现。
        capsys (pytest.CaptureFixture[str]): 用于读取命令标准输出。

    Raises:
        AssertionError: CLI 未传递受控补充来源或未输出条目数。

    Returns:
        None: 此测试只验证 rebuild 命令契约。
    """
    calls: list[tuple[Path, Path, Path | None, Path | None, Path | None]] = []

    def fake_rebuild(
        raw_root: Path,
        output_root: Path,
        *,
        wiki_root: Path | None = None,
        cycles_root: Path | None = None,
        event_entries_root: Path | None = None,
    ) -> KnowledgeBuildResult:
        """记录 CLI 传入的受控补充来源并返回构建替身。

        Args:
            raw_root (Path): 原始 Mod 快照目录。
            output_root (Path): 知识产物根目录。
            wiki_root (Path | None): 受控 Wiki 补充目录。
            cycles_root (Path | None): 实跳循环观察目录。
            event_entries_root (Path | None): 事件 UI 快照目录。

        Returns:
            KnowledgeBuildResult: 不执行磁盘重建的测试结果。
        """
        calls.append(
            (
                raw_root,
                output_root,
                wiki_root,
                cycles_root,
                event_entries_root,
            )
        )
        return KnowledgeBuildResult(
            output_root / "mod_export/v0.107.1", 3, {"cards": 3}
        )

    monkeypatch.setattr(cli, "rebuild_mod_knowledge", fake_rebuild)
    raw = tmp_path / "mod_export/v0.107.1/raw"
    wiki = tmp_path / "web_wiki"
    cycles = tmp_path / "cycles"
    event_entries = tmp_path / "event_entries"

    assert (
        cli.main(
            [
                "rebuild",
                str(raw),
                "--output-root",
                str(tmp_path),
                "--wiki-root",
                str(wiki),
                "--cycles-root",
                str(cycles),
                "--event-entries-root",
                str(event_entries),
            ]
        )
        == 0
    )

    assert calls == [(raw, tmp_path, wiki, cycles, event_entries)]
    assert '"entries": 3' in capsys.readouterr().out


def test_generate_questions_command_does_not_build_e3_splits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """多问法命令只生成知识样本，不暴露 train/dev/test 参数。

    Args:
        tmp_path (Path): Pytest 提供的隔离临时目录。
        monkeypatch (pytest.MonkeyPatch): 用于替换真实问法生成实现。

    Raises:
        AssertionError: CLI 越界构建分割或未传递输入输出目录。

    Returns:
        None: 此测试只验证 generate-questions 命令契约。
    """
    calls: list[tuple[Path, Path]] = []

    def fake_generate(snapshot: Path, output: Path) -> KnowledgeBuildResult:
        """记录多问法输入输出目录并返回构建替身。

        Args:
            snapshot (Path): Canonical 快照目录。
            output (Path): 问法产物目录。

        Returns:
            KnowledgeBuildResult: 不写入问法的测试结果。
        """
        calls.append((snapshot, output))
        return KnowledgeBuildResult(output, 5, {"cards": 5})

    monkeypatch.setattr(cli, "generate_question_variants", fake_generate)
    snapshot = tmp_path / "mod_export/v0.107.1"
    output = tmp_path / "generated-v0.107.1"

    assert (
        cli.main(
            [
                "generate-questions",
                str(snapshot),
                "--output-root",
                str(output),
            ]
        )
        == 0
    )

    assert calls == [(snapshot, output)]


def test_generate_arithmetic_command_passes_independent_sources(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """算术命令应显式传入人类战斗帧、候选目录和最终 probe。

    Args:
        tmp_path (Path): Pytest 提供的隔离临时目录。
        monkeypatch (pytest.MonkeyPatch): 用于替换真实算术生成实现。
        capsys (pytest.CaptureFixture[str]): 用于读取命令标准输出。

    Raises:
        AssertionError: CLI 未保持训练候选和最终考试题的来源边界。

    Returns:
        None: 此测试只验证命令行契约。
    """
    calls: list[tuple[Path, Path, Path]] = []

    def fake_generate(
        *,
        human_root: Path,
        output_root: Path,
        probe_root: Path,
    ) -> ArithmeticBuildResult:
        """记录算术生成参数并返回测试结果。

        Args:
            human_root (Path): 当前项目的人类精确战斗目录。
            output_root (Path): 算术候选输出目录。
            probe_root (Path): 不得泄漏的最终知识考试目录。

        Returns:
            ArithmeticBuildResult: 不执行真实生成的测试结果。
        """
        calls.append((human_root, output_root, probe_root))
        return ArithmeticBuildResult(output_root, 12, 3, {"block_math": 15})

    monkeypatch.setattr(cli, "generate_arithmetic_candidates", fake_generate)
    human = tmp_path / "raw/human"
    output = tmp_path / "generated-v0.107.1"
    probes = tmp_path / "eval/knowledge"

    assert (
        cli.main(
            [
                "generate-arithmetic",
                "--human-root",
                str(human),
                "--output-root",
                str(output),
                "--probes-root",
                str(probes),
            ]
        )
        == 0
    )

    assert calls == [(human, output, probes)]
    assert '"train": 12' in capsys.readouterr().out


def test_review_command_writes_human_readable_report(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """审计命令应输出报告路径，而不是伪装成知识构建结果。

    Args:
        tmp_path (Path): Pytest 提供的隔离临时目录。
        monkeypatch (pytest.MonkeyPatch): 用于替换真实报告生成实现。
        capsys (pytest.CaptureFixture[str]): 用于读取命令标准输出。

    Raises:
        AssertionError: CLI 未传递报告路径或未打印最终位置。

    Returns:
        None: 此测试只验证 review 命令契约。
    """
    calls: list[tuple[Path, Path]] = []

    def fake_review(snapshot: Path, output: Path) -> Path:
        """记录审计输入输出目录并返回报告路径。

        Args:
            snapshot (Path): Canonical 快照目录。
            output (Path): 审计报告路径。

        Returns:
            Path: 测试替身声明已生成的报告路径。
        """
        calls.append((snapshot, output))
        return output

    monkeypatch.setattr(cli, "generate_review_report", fake_review)
    snapshot = tmp_path / "mod_export/v0.107.1"
    output = tmp_path / "reports/v0.107.1.md"

    assert cli.main(["review", str(snapshot), "--output", str(output)]) == 0

    assert calls == [(snapshot, output)]
    assert str(output) in capsys.readouterr().out
