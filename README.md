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

正式运行模型时可用同一套隔离启动基础拉起游戏：

```bash
uv run play-sts2-game
uv run play-sts2-game --mode headed
```

命令默认监听 `http://127.0.0.1:8080`，使用
`e2e/fixtures/profile/` 中不含个人信息的全解锁模板，并在临时 HOME 中关闭
Steam 与 `UnifiedSavePath`。命令保持前台运行，退出时会结束游戏进程并删除本次
隔离存档；模板还会启用游戏原生 `fast` 模式，加快战斗动画和转场。端口、游戏
路径或模板不同时可传入 `--port`、`--app-path` 和 `--profile`。

## 录制人类轨迹

先启动已经加载 Agent Mod 的有头 STS2，再运行：

```bash
uv run play-sts2-record
```

命令默认连接 `http://127.0.0.1:8080`，等待一局开始，并将轨迹保存到
`data/raw/human/YYYYMMDD-aN-fN-SEED/`。录制器只消费 Mod 的 SSE，不会再
额外轮询 `/state`；每场战斗写入一个 `combat/*.jsonl`，战略动作写入
`strategy/decisions.jsonl`。服务地址或数据根目录不同时可以
显式指定：

```bash
uv run play-sts2-record \
  --base-url http://127.0.0.1:8082 \
  --output-root data/raw
```

`meta.json` 会持续记录进阶难度、战斗数量、战斗样本数、战略样本数和完整性。
即使进程异常退出也能检查已经落盘的部分；Mod 报告原生 UI 采集缺口时，整局会
保留以供审计，但自动关闭训练准入。

## 人类可读轨迹

可读文本不放在 raw，而是随时可从动作事实重建：

```bash
uv run play-sts2-transcribe data/raw/human/<规范局目录名>
```

输出位于 `data/transcripts/<同名局目录>/`，不含内部 sample/event ID。
旧数据迁移已完成并删除一次性代码；raw 不再保存 `events.jsonl`。

## 游戏知识与 SFT 数据集

结构化 Wiki 可以迁移为带来源字段的单实体 Markdown：

```bash
uv run play-sts2-knowledge import-wiki /path/to/wiki
```

当游戏和 Agent Mod 已启动时，也可以直接从当前版本的 `/data/*` 端点实测
导出。命令会同时保存原始 JSON 快照和可读 Markdown：

```bash
uv run play-sts2-knowledge export
```

Web Wiki、Mod 实测与继承的核验问法分别保留来源字段，不会互相冒充。默认把
`data/game_knowledge/` 和 `data/raw/human/` 构建为统一的可读 messages 数据集：

```bash
uv run play-sts2-train build-sft
```

产物直接写入 `data/datasets/sft/{train,dev,test}.jsonl`。知识行保留实体来源，
行为行通过当前 Harness 重新生成观测和规范 `ACTION:`；战斗与战略每个动作
都是独立 `system/user/assistant` 样本。整局 split 来自
`data/raw/human/splits.json`，`training_eligible=false` 的局不会进入训练。

安装 PyTorch、Transformers 与 PEFT 后，可以用本地 Qwen3.5-4B 训练 LoRA：

```bash
uv sync --group training
uv run --group training play-sts2-train sft \
  --config configs/sft.toml \
  --name sft-clean-20260827-native-r16-e3
```

训练过程写入 `runs/sft/<name>/`，最终 adapter 写入
`models/adapters/<name>/`。当前配置从 e2 adapter 近似续训；Qwen3.5-4B 的 32 层
都会按层类型覆盖对应的线性注意力或全注意力投影。训练器用模型原生 chat
template 的 assistant mask 监督每个动作，并用 2,048-token 分块交叉熵控制
logits 峰值。checkpoint 每 2,000 个优化
步覆盖同一个 `checkpoint-last`，不会沿用旧实现的 15 步频率。用
`--max-steps 1` 可以执行一次真实模型冒烟。

为完整人类局分配 dev/test 后，可以独立加载 adapter 做确定性生成验证：

```bash
uv run --group training play-sts2-train eval-sft \
  --adapter models/adapters/sft-clean-20260827-native-r16-e3 \
  --split dev
```

评测会保存目标与实际生成全文，并报告精确匹配率；人类行为还会单独报告单行
`ACTION:` 外形通过率。它不会把动作外形合法等同于策略选择正确。

训练完成后，一条命令即可把 adapter 安全合并为独立 Hugging Face 模型：

```bash
uv run --group training play-sts2-train merge-sft \
  --adapter models/adapters/sft-clean-20260827-native-r16-e3
```

基座模型从 `configs/sft.toml` 读取，默认输出为
`models/merged/sft-clean-20260827-native-r16-e3-merged/`。命令拒绝覆盖已有目录，
通过同盘暂存目录排他原子发布；发布前会核对 adapter 的基座血缘并确认输入没有
变化。`merge_manifest.json` 记录全部基座权重、adapter 与实际 tokenizer 来源的
SHA-256。需要非默认位置时增加 `--output <目录>`。

正式轮次比较优先使用只计算 assistant token 的 teacher-forced 指标：

```bash
uv run --group training play-sts2-train eval-sft-loss \
  --adapter models/adapters/sft-clean-20260827-native-r16-e3 \
  --split test
uv run --group training play-sts2-train eval-sft-knowledge \
  --model models/merged/sft-clean-20260827-native-r16-e3-merged
```

## Harness 契约

Harness 使用严格的单行 `ACTION:` 协议，并根据完整状态区分战斗、战略与过渡
屏幕。模型系统提示词作为独立文本资源存放在
`src/play_sts2/harness/prompts/`，在线推理和离线 SFT 数据构建将读取同一份内容。
可读观测覆盖战斗、地图、事件、奖励、选牌、休息处、商店、宝箱、卡牌包、
水晶球、弹窗及其他 Mod 已开放决策的页面，并统一过滤模型不应执行的存档、
退出及底层重复动作。

## 推理适配

`src/play_sts2/inference/` 定义与游戏无关的 `DecisionProvider` 协议。
`OpenAICompatibleProvider` 可以连接本地 MLX 等兼容服务；原版 Qwen 与合并后的
SFT 模型复用同一份 Python 接入代码，只在启动推理服务时选择不同模型目录。
LoRA adapter 是训练权重，不是另一套 Provider 实现。

## 单步决策闭环

`src/play_sts2/runtime/DecisionEngine` 接收一个稳定游戏状态，依次构建 Harness
观测与系统提示词、调用 `DecisionProvider`、校验 `ACTION:` 并通过 `GameClient`
执行动作。`BattleRunner` 在每个成功动作后丢弃历史，下一次只提供当前完整
状态；同一状态的格式错误重试仍由 `DecisionEngine` 在有限次数内处理。

## Qwen 战斗闭环

在 Apple Silicon Mac 上先安装独立的本地推理依赖，并把合并后的 Transformers
BF16 模型一次性转换为 MLX 8-bit：

```bash
uv sync --group inference
uv run --group inference play-sts2-model prepare \
  --source models/merged/sft-clean-20260827-native-r16-e3-merged \
  --output models/serving/sft-clean-20260827-native-r16-e3-mlx-8bit
```

转换不会修改 `models/merged/` 中的原模型，默认产物写入
`models/serving/sft-clean-20260827-native-r16-e3-mlx-8bit/`。随后在一个终端启动
带十个前缀缓存槽的模型服务，并在另一个终端验证模型能够生成合法 Harness 动作：

```bash
uv run --group inference play-sts2-model serve
uv run play-sts2-model smoke
```

`play-sts2-battle` 和 `play-sts2-run` 只连接这个模型服务，不直接加载模型。

让已经加载 Agent Mod 的 STS2 停在一个稳定战斗决策画面，再运行：

```bash
uv run play-sts2-battle
```

模型服务需要名称时可加 `--model Qwen/Qwen3.5-4B`；游戏或模型使用其他端口时，
分别传入 `--game-url` 和 `--model-url`。`BattleRunner` 每次只提交当前完整状态，
非法模型输出最多重试三次，等待异步动作重新进入可决策状态，并在胜利或角色
死亡时返回。该命令只负责当前战斗，适合战斗调试和后续场景采样。

## Qwen 整局闭环

分别启动 SFT 模型服务、隔离游戏和整局 Agent：

```bash
uv run --group inference play-sts2-model serve
uv run play-sts2-game
uv run play-sts2-run --character DEFECT --ascension 0
```

需要固定新局时可增加 `--seed <seed>`。若游戏已经处于一局之中，或要从主菜单
恢复保存局，则使用：

```bash
uv run play-sts2-run --resume
```

`RunRunner` 会在战略与战斗页面执行无跨动作历史的单步决策，并在过渡动画结束
后继续路由，直到 `GAME_OVER`。遇到未枚举页面时会明确停止，不根据相似动作
猜测游戏语义。

## 确定性战斗场景

`src/play_sts2/scenario/` 可以从任意残局重置出一场已核对的战斗。场景显式保存
局种子、原总层数、遭遇、进阶、牌组、遗物、药水和生命；`BattleResetter` 会清理
旧局、结算初始事件、装载配置，等待开场效果结束后设置生命，并返回可直接传给
`BattleRunner.run()` 的状态。首次结果中的 `snapshot` 可作为后续 GRPO rollout 的
入口核对基准。

真实游戏的复现测试使用：

```bash
uv run pytest e2e/test_battle_scenario.py --run-e2e
```

该测试在隔离 HOME 中重复创建同一场景，比较敌人组成与 HP、初始手牌和意图，
并比较固定空过一回合后的第二回合手牌与意图。不同模型动作轨迹无需保持后续
手牌顺序一致。场景使用的 `scenariofight` 和 `loadout` 都受
`STS2_ENABLE_DEBUG_ACTIONS=1` 保护；这种调试局不得录入人类轨迹、SFT 数据或
正式评测结果。

## 目录边界

- `src/play_sts2/`：Python 客户端、录制、转录、推理与训练代码。
- `mod/`：C# Mod 源码，不由 uv 管理。
- `e2e/fixtures/`：可提交的合成测试数据，不允许包含个人存档。
- `data/`：人类与 Agent 轨迹及派生数据，不进入 Git。
- `models/`：基座、Adapter 和合并模型，不进入 Git。
- `runs/`：训练日志、临时 checkpoint 和评测结果，不进入 Git。

录制数据格式见 [docs/data-format.md](docs/data-format.md)，训练配方见
[docs/training.md](docs/training.md)。
