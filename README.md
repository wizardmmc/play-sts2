# Play STS2

用最简优雅实现方式，逐步完成 Slay the Spire 2 的游戏控制、轨迹录制、数据转录、
后训练与评测。

## 开发命令

```bash
uv sync
uv run ruff format .
uv run ruff check .
uv run pytest
```

真实游戏 E2E 默认使用无头模式和隔离测试存档：

```bash
uv run pytest e2e --run-e2e
uv run pytest e2e --run-e2e --sts2-mode=headed
```

## 目录边界

- `src/play_sts2/`：Python 客户端、录制、转录、推理与训练代码。
- `mod/`：C# Mod 源码，不由 uv 管理。
- `e2e/fixtures/`：可提交的合成测试数据，不允许包含个人存档。
- `data/`：人类与 Agent 轨迹及派生数据，不进入 Git。
- `models/`：基座、Adapter 和合并模型，不进入 Git。
- `runs/`：训练日志、临时 checkpoint 和评测结果，不进入 Git。

录制数据格式见 [docs/data-format.md](docs/data-format.md)。
