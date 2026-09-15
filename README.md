# Play STS2

让语言模型游玩《Slay the Spire 2》的实验项目，围绕 Qwen3.5-4B 构建游戏控制、
人类轨迹录制、监督微调（SFT）、强化学习（RL）与评测流程。

## 功能

- **自动游玩**：通过 C# Mod 读取游戏状态、执行动作，支持单场战斗和整局运行。
- **轨迹与知识数据**：录制人类操作、生成可读战报，从游戏导出知识并构建训练数据。
- **模型训练与评测**：支持 LoRA 微调、模型合并，以及知识、算术和动作生成评测。
- **强化学习**：支持可重复构造的战斗场景、战斗 GRPO 和整局战略训练。

模型每一步接收当前游戏状态的中文观测，返回一行 `ACTION:` 指令，由 Python
校验后交给 Mod 执行。在线决策与 SFT 数据构建共用观测格式和系统提示词。

## 运行前准备

- Python 3.12+ 和 [uv](https://docs.astral.sh/uv/)。
- 自行安装的 STS2 与已加载的 [Agent Mod](mod/STS2AIAgent/README.md)；
  构建 Mod 需要 .NET 9 SDK 和游戏程序集。
- 当前隔离游戏启动器面向 macOS；下文的本地推理流程使用 Apple Silicon 和 MLX。
  CUDA 训练入口见[训练文档](docs/training.md)。
- 游戏文件、模型权重、轨迹和训练产物需自行准备，仓库提供代码与配置模板。

```bash
uv sync
```

按需将 `configs-template/` 中的文件复制到 `configs/` 的同名路径，并填写本机
模型、数据和服务配置。具体结构见[配置说明](configs-template/README.md)。

## 启动 Agent

以下流程使用经过本项目 `merge-sft` 导出的合并模型；训练与合并步骤见
[训练文档](docs/training.md#合并模型)。

### 1. 准备本地推理模型

```bash
mkdir -p configs/inference
cp -n configs-template/inference/inference.toml configs/inference/inference.toml
```

编辑 `configs/inference/inference.toml`，填写 `artifact_id`、`merged_model` 和
`serving_model`，然后转换为 MLX 8-bit：

```bash
uv sync --group inference
uv run --group inference play-sts2-model prepare
```

### 2. 启动模型服务与游戏

在三个终端中分别执行以下命令，等待前两个服务就绪后再启动 Agent。

**终端 1：模型服务**

```bash
uv run --group inference play-sts2-model serve
```

**终端 2：游戏**

```bash
uv run play-sts2-game --app-path /path/to/SlayTheSpire2.app --mode headed
```

去掉 `--mode headed` 即以无头模式运行。游戏服务默认位于
`http://127.0.0.1:8080`。省略 `--app-path` 时，启动器读取 `STS2_APP_PATH`，
或使用项目内的 `.runtime/SlayTheSpire2-v0.107.1/SlayTheSpire2.app`。
请按所用游戏版本设置路径。

启动器使用独立的临时存档和全解锁模板；退出启动器时会结束游戏并清理本次存档。

**终端 3：检查推理并开始一局**

```bash
uv run play-sts2-model smoke
uv run play-sts2-run --character DEFECT --ascension 0
```

### 常用选项

| 用途 | 命令或参数 |
| --- | --- |
| 固定新局种子 | `play-sts2-run --seed <seed>` |
| 接管当前游戏中的局或恢复该存档中的局 | `play-sts2-run --resume` |
| 接管当前战斗 | `play-sts2-battle` |
| 启用模型思考模式 | 为 `play-sts2-run` 或 `play-sts2-battle` 添加 `--profile think` |
| 使用其他推理配置 | `--inference-config <路径>` |
| 连接其他游戏端口 | `--game-url http://127.0.0.1:8082`，对应游戏启动参数 `--port 8082` |

以上命令均通过 `uv run` 执行。单场战斗命令需要游戏已进入战斗；默认推理配置
使用 `no-think` 模式。

## 录制与转录

启动加载 Agent Mod 的可见游戏窗口，在开始游玩前运行：

```bash
uv run play-sts2-record
```

录制器默认连接 `http://127.0.0.1:8080`，等待一局开始，将数据保存到
`data/raw/human/YYYYMMDD-aN-fN-SEED/`。每场战斗对应一个 `combat/*.jsonl`，
战略动作保存在 `strategy/decisions.jsonl`，`meta.json` 记录局信息与录制完整性。

可用 `--base-url` 指定游戏地址，`--output-root` 指定数据根目录。
录制后生成可读战报：

```bash
uv run play-sts2-transcribe data/raw/human/<局目录名>
```

输出位于 `data/transcripts/human/<局目录名>/`。完整字段和其他录制来源见
[数据格式](docs/data-format.md)。

## 数据、训练与评测

训练数据由游戏知识、实战算术题和人类行为样本组成：

```text
固定版本 Mod 导出与受控补充 → 应用 curated 修正 → 规范事实 → 知识候选
人类轨迹 → 算术候选与行为样本
                          ↓
                按配置混合并划分训练、验证、测试集
                          ↓
                   SFT → 合并模型 → 评测与游戏运行
```

知识修正与候选生成命令见[数据说明](data/README.md#重建命令)。
调整数据时修改来源、curated 规则或生成代码，再重新构建数据集。

准备好候选数据、人类局分卷和 `configs/sft/` 配置后，可以构建数据并在 CUDA 上训练：

```bash
uv run play-sts2-train build-sft \
  --knowledge-root data/game_knowledge/generated-v0.107.1 \
  --mix configs/sft/mix.toml

uv sync --group training
uv run --group training play-sts2-train sft-cuda \
  --config configs/sft/sft-cuda.toml \
  --name 20260915-sft-example
```

示例使用 v0.107.1 候选目录；使用其他版本时指定对应目录。训练配置模板位于
[`configs-template/sft/`](configs-template/sft/)。训练日志与 checkpoint 写入
`runs/sft/<name>/`，最终 LoRA 权重写入 `models/adapters/<name>/`。

- [SFT 训练与评测](docs/training.md#sft-训练与评测)：数据构建、训练、续训、评测和模型合并。
- [RL 训练](docs/training.md#rl-训练)：战斗与战略训练、课程推进和验证流程。
- [RL 配置模板](configs-template/rl/README.md)与[战斗场景模板](configs-template/scenarios/)：采样和实验配置。

## 开发

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

真实游戏 E2E 使用隔离测试存档，默认无头运行：

```bash
uv run pytest e2e --run-e2e
uv run pytest e2e --run-e2e --sts2-mode=headed
```

| 目录 | 内容 |
| --- | --- |
| `src/play_sts2/` | Python 客户端、决策运行时、录制、数据处理与训练 |
| `mod/STS2AIAgent/` | C# 游戏 Mod |
| `configs-template/` | 可提交的配置示例 |
| `tests/`、`e2e/` | 单元测试、真实游戏测试与测试存档模板 |
| `docs/` | [观测设计](docs/design-harness.md)、[数据格式](docs/data-format.md)与[训练说明](docs/training.md) |
| `configs/`、`data/`、`models/`、`runs/` | 本机配置、数据与实验产物，Git 忽略（保留 `data/README.md`） |

## 许可证

Python 代码采用 [MIT](LICENSE) 许可证。
C# Mod 源自 [CharTyr/STS2-Agent](https://github.com/CharTyr/STS2-Agent)，沿用
[AGPL-3.0](mod/STS2AIAgent/LICENSE)；来源记录见
[UPSTREAM.md](mod/STS2AIAgent/UPSTREAM.md)。
