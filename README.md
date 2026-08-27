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

Web Wiki 条目标记为 `source: web_wiki`，Mod 实测条目标记为
`source: mod_export`；两类数据不会互相冒充。默认用 Web Wiki 和当前精确人类
转录构建可读的 chat messages 数据集：

```bash
uv run play-sts2-train build-sft
```

产物写入 `data/datasets/sft/baseline-v1/`。知识行保留实体来源，行为行通过
当前 Harness 重新生成观测和规范 `ACTION:`，不会复用旧版本的 token ID。

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
执行动作。它会返回本次完整决策产物，并允许上层传入同一场战斗的消息历史。

## Qwen 战斗闭环

在 Apple Silicon Mac 上先安装独立的本地推理依赖，并把合并后的 Transformers
BF16 模型一次性转换为 MLX 8-bit：

```bash
uv sync --group inference
uv run --group inference play-sts2-model prepare
```

转换不会修改 `models/merged/` 中的原模型，默认产物写入
`models/serving/sft-clean-20260827-native-r16-e1-mlx-8bit/`。随后在一个终端启动
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
分别传入 `--game-url` 和 `--model-url`。`BattleRunner` 会在一场战斗内保留对话
历史，非法模型输出最多重试三次，等待异步动作重新进入可决策状态，并在胜利或
角色死亡时返回。该命令只负责当前战斗，适合战斗调试和后续场景采样。

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

`RunRunner` 会在战略页面执行无跨页面历史的单步决策，在战斗开始后交给
`BattleRunner` 保留当前战斗的短期历史，并在过渡动画结束后继续路由，直到
`GAME_OVER`。遇到未枚举页面时会明确停止，不根据相似动作猜测游戏语义。

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

录制数据格式见 [docs/data-format.md](docs/data-format.md)。
