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

## 录制人类轨迹

先启动已经加载 Agent Mod 的有头 STS2，再运行：

```bash
uv run play-sts2-record
```

命令默认连接 `http://127.0.0.1:8080`，等待一局开始，并将轨迹保存到
`data/raw/human/<run_id>/`。服务地址或数据根目录不同时可以显式指定：

```bash
uv run play-sts2-record \
  --base-url http://127.0.0.1:8082 \
  --output-root data/raw
```

## 转录精确决策

将一局原始人类轨迹转换为无损的结构化决策：

```bash
uv run play-sts2-transcribe data/raw/human/<run_id>
```

命令默认写入 `data/transcripts/<run_id>.jsonl`。这一层只提取动作前状态、
原生 UI 动作和非空参数；面向模型的状态文本与 SFT messages 将由后续 Harness
共享组件生成。

## Harness 契约

Harness 使用严格的单行 `ACTION:` 协议，并根据完整状态区分战斗、战略与过渡
屏幕。模型系统提示词作为独立文本资源存放在
`src/play_sts2/harness/prompts/`，在线推理和离线 SFT 数据构建将读取同一份内容。

## 目录边界

- `src/play_sts2/`：Python 客户端、录制、转录、推理与训练代码。
- `mod/`：C# Mod 源码，不由 uv 管理。
- `e2e/fixtures/`：可提交的合成测试数据，不允许包含个人存档。
- `data/`：人类与 Agent 轨迹及派生数据，不进入 Git。
- `models/`：基座、Adapter 和合并模型，不进入 Git。
- `runs/`：训练日志、临时 checkpoint 和评测结果，不进入 Git。

录制数据格式见 [docs/data-format.md](docs/data-format.md)。
