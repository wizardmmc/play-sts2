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

需要在 `8082` 端口打开可见游戏窗口时，直接运行：

```bash
uv run play-sts2-game --port 8082 --mode headed
```

`play-sts2-model serve` 只启动模型推理服务，不会启动游戏；模型服务、游戏进程和
自动决策闭环是三个独立入口，便于分别重启和排查。

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

结构化 Wiki 可以迁移为带来源字段的单实体 Markdown，但它不再直接进入后续
知识监督数据：

```bash
uv run play-sts2-knowledge import-wiki /path/to/wiki
```

默认游戏路径是项目内忽略版本控制的
`.runtime/SlayTheSpire2-v0.107.1/SlayTheSpire2.app`，不会跟随 Steam 自动更新。
游戏和 Agent Mod 启动后，从固定版本的 `/data/*` 端点保存原始 JSON 与可读
Markdown：

```bash
uv run play-sts2-knowledge export
```

随后离线应用版本化人工修正、怪物补充、故障机器人充能球实测和已经真正进入
事件页后的界面快照，再生成按实体保存的多问法候选与人工审计报告。人工修正在
`rebuild` 阶段生效，不是多问法生成后的独立数据目录：

```bash
uv run play-sts2-knowledge rebuild \
  data/game_knowledge/mod_export/v0.107.1/raw \
  --wiki-root data/game_knowledge/web_wiki \
  --cycles-root /path/to/human-rl/data/cycles \
  --event-entries-root data/game_knowledge/event_entries/v0.107.1/events
uv run play-sts2-knowledge generate-questions \
  data/game_knowledge/mod_export/v0.107.1
uv run play-sts2-knowledge generate-arithmetic
uv run play-sts2-knowledge review \
  data/game_knowledge/mod_export/v0.107.1
```

多问法产物位于 `data/game_knowledge/generated-v0.107.1/`，仍是知识候选，不会
自动改写正式 SFT 分卷。算术命令只从人类整局训练名册中通过 raw 审计的战斗帧
读取真实攻击意图，用三个独立随机种子确定性生成 train、validation 与 eval
算术候选；验证、测试、不合格和未分配局不会影响算术输入。Wiki 只作为
怪物招式和循环的明确补充来源，事件变量优先使用同版本实机界面快照。

`mod_export` 是“规范事实层”：同一张卡牌、怪物或地图只保留一份确定身份和字段
的事实。它不是模板。Markdown 的标题和列表外形由渲染模板统一生成，但模板只是
排版规则，真正作为后续问答依据的是其中的规范事实。地图事实还保存密林、暗港、
巢穴和荣耀各自的前期弱遭遇池、常规遭遇池、精英池与 Boss 池。逐个遭遇的
“进场后有哪几名敌人”属于现场已知信息，不生成训练题；四张地图的怪池候选统一
写入 `generated-v0.107.1/encounters/`。故障机器人的初始配置和五种充能球机制
则由固定版本 Mod 原始数据与受控实测快照合并后生成。

完整链路是：

```text
固定版本 Mod 原始导出与受控补充
→ 离线重建并应用版本化人工修正
→ 固定版本规范事实层（mod_export）
→ 多问法知识候选（generated-v0.107.1）与实战算术候选
→ 人工确定知识、算术和行为的配比与分卷
→ 可训练数据（data/datasets/sft）
```

历史目录
`data/game_knowledge/curated-v0.107.1/` 不在该链路中，也不是后续 SFT 的输入。

安装 PyTorch、Transformers 与 PEFT 后，可以用本地 Qwen3.5-4B 训练 LoRA：

当前 E3 数据使用 `configs/sft-e3-mix.toml` 定向混合：远古者不进入训练或验证，
其余正式知识实体和事实全部保留；高频常规动作按上限抽样，未列出的低频动作全部
保留。
重新构建命令为：

```bash
uv run play-sts2-train build-sft --mix configs/sft-e3-mix.toml
```

```bash
uv sync --group training
uv run --group training play-sts2-train sft \
  --config configs/sft.toml \
  --name 20260828-sft-clean-native-r16-e3
```

训练过程写入 `runs/sft/<name>/`，最终 adapter 写入
`models/adapters/<name>/`。名称必须以 `YYYYMMDD-` 开头，目录按名称排序就是时间
顺序。Qwen3.5-4B 的 32 层
都会按层类型覆盖对应的线性注意力或全注意力投影。训练器用模型原生 chat
template 的 assistant mask 监督每个动作，并用 2,048-token 分块交叉熵控制
logits 峰值。checkpoint 每 2,000 个优化
步覆盖同一个 `checkpoint-last`，同时保存 LoRA、AdamW、数据游标、洗牌状态和
PyTorch 随机状态，约占 180～200 MB。基座在 MPS 上使用 BF16，所有 LoRA
可训练参数由训练器强制保持 FP32；任一可训练参数不是 FP32 都会在创建优化器前
失败。中断后使用同一名称并增加 `--resume` 可
精确续训；当前 r16 实测净文件约 165 MB，180～200 MB 是预留文件系统余量后的
预算。中断若发生在两个 checkpoint 之间，恢复会先原子回滚没有对应权重的末尾
指标，再从 checkpoint 的下一步重算并继续追加。父 adapter 只锁定实际加载的 LoRA 权重与 PEFT 配置
哈希，不把可独立调整的聊天模板作为硬锁。数据分卷与父 adapter 在加载前后都会
复核文件身份；精确恢复还分别锁定解析后的设备、基座精度和 adapter 精度。同名训练从加载到发布全程
持有排他锁，不能并发写坏指标或 checkpoint。用 `--max-steps 1` 可以执行一次
真实模型冒烟。

```bash
uv run --group training play-sts2-train sft \
  --config configs/sft.toml \
  --name 20260828-sft-clean-native-r16-e3 \
  --resume
```

数据目录固定为同构的 `train/`、`validation/` 和 `eval/` 三棵树；知识按类别和
实体拆成 JSONL，人类行为按 `combat/<run-id>/<battle-key>.jsonl` 与
`strategy/<run-id>.jsonl` 保存。同一知识事实显式提供 train、validation、eval
三种自然问法，后两种分别用于调参与冻结后的最终验收。三种问题原文必须互不
重复；算术使用三个独立随机种子，并以 `case_id` 隔离相同运算语义和操作数。整局人类游戏必须先在
`data/raw/human/splits.json` 明确归属，否则构建失败。

当前分卷共有 11,485 条训练样本、6,509 条验证样本和 6,736 条最终评测样本；
train 覆盖 1,347 个知识实体和全部 5,856 个显式事实。validation/eval 都覆盖相同
事实、全部知识类别、六类算术题以及按整局留出的战斗与战略行为。

为完整人类局分配 validation/eval 后，可以独立加载 adapter 做确定性生成验证：

```bash
uv run --group training play-sts2-train eval-sft \
  --adapter models/adapters/20260828-sft-clean-native-r16-e3 \
  --split validation
```

评测会保存目标与实际生成全文，并报告精确匹配率；人类行为还会单独报告单行
`ACTION:` 外形通过率。它不会把动作外形合法等同于策略选择正确。

训练完成后，一条命令即可把 adapter 安全合并为独立 Hugging Face 模型：

```bash
uv run --group training play-sts2-train merge-sft \
  --adapter models/adapters/20260828-sft-clean-native-r16-e3
```

基座模型从 `configs/sft.toml` 读取，默认输出为
`models/merged/20260828-sft-clean-native-r16-e3-merged/`。命令拒绝覆盖已有目录，
通过同盘暂存目录排他原子发布；发布前会核对 adapter 的基座血缘并确认输入没有
变化。`merge_manifest.json` 记录全部基座权重、adapter 与实际 tokenizer 来源的
SHA-256。需要非默认位置时增加 `--output <目录>`。

正式轮次比较优先使用只计算 assistant token 的 teacher-forced 指标：

```bash
uv run --group training play-sts2-train eval-sft-loss \
  --adapter models/adapters/20260828-sft-clean-native-r16-e3 \
  --split eval
uv run --group training play-sts2-train eval-sft-knowledge \
  --model models/merged/20260828-sft-clean-native-r16-e3-merged
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
BF16 模型一次性转换为 MLX 8-bit。模型身份、目录、服务端口和生成 profile
统一由 `configs/inference.toml` 管理；完成新一轮训练时先更新其中的
`artifact_id`、`merged_model` 与 `serving_model`，再执行：

```bash
uv sync --group inference
uv run --group inference play-sts2-model prepare
```

转换不会修改 `models/merged/` 中的原模型，默认产物写入
配置指定的不可变服务目录。它会解析合并清单中的 adapter/输出血缘，只对小型
`merge_manifest.json` 取摘要，并随 MLX 产物写入 `serving_manifest.json`，不会
重复哈希模型权重。服务启动还会核对实际量化/EOS，以及固定对话在 thinking
开关两侧的渲染文本与 token IDs。随后在一个终端启动模型服务，并在另一个终端
验证磁盘身份、端口实际加载目录和 Harness 动作协议：

```bash
uv run --group inference play-sts2-model serve
uv run play-sts2-model smoke
uv run play-sts2-model smoke --profile think
```

默认 `no-think` profile 与 e3 SFT 编码一致；`think` profile 通过每次请求的
`chat_template_kwargs.enable_thinking=true` 显式启用动态模板分支并使用独立 token
预算。思考耗尽预算会报告生成截断并保留 reasoning、finish reason 和模型标识，
不会伪装成响应协议错误或自动降级为 no-think。

`play-sts2-battle` 和 `play-sts2-run` 从同一配置读取服务与默认 profile；两者会
在连接游戏、改变任何游戏状态之前执行相同的磁盘与 `/v1/models` 身份预检，
再用配置指定的绝对模型路径完成一次真实的一 token 生成，避免把“端点存活”误当
成“权重可加载”。此后每次决策请求都内部绑定该路径，并要求 MLX 响应回报完全
相同的模型标识；服务中途切换模型会立即失败。

让已经加载 Agent Mod 的 STS2 停在一个稳定战斗决策画面，再运行：

```bash
uv run play-sts2-battle
uv run play-sts2-battle --profile think
```

非默认配置可传 `--inference-config <路径>`，临时覆盖地址仍可用 `--model-url`。
本地 CLI 不接受请求级 `--model`，避免绕过配置中的服务目录身份；底层通用
Provider 仍可供其他 OpenAI-compatible 集成显式传入并核对模型名。`BattleRunner` 每次
只提交当前完整状态，
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

`src/play_sts2/scenario/` 可以在同一个长期存活的游戏进程中，从任意残局重置并按
显式配置构造一个可重复的战斗起点。场景保存局种子、用于遭遇 RNG 的总层数参数、
遭遇、进阶、牌组（含升级与附魔）、遗物、带空槽位置的药水栏和当前/最大生命。
`BattleResetter` 会清理旧局，跳过 Neow 奖励直接装载并进入战斗，等待开场效果结束
后设置生命，再返回可直接传给 `BattleRunner.run()` 的状态。首次结果中的
`snapshot` 同时保存模型实际收到的中文 system/user 输入与合法动作域，可作为同一
场景后续 GRPO rollout 的统一入口基准。

这里的确定性契约是“同一场景配置的所有 rollout 从相同起点出发”，不是重放某次
人类历史局已经消耗过的 RNG 状态：它不承诺初始手牌、同种敌人的随机 HP 分配或
其他随机结果与某条人类轨迹逐项相同，也不会重放人类后续动作。GRPO 只把统一的
初始中文观测作为场景入口契约；模型开始行动后，各 rollout 独立演化。

真实游戏的场景确定性测试使用：

```bash
uv run pytest e2e/scenario --run-e2e
```

这些测试在隔离 HOME 中重复构造同一场景，比较实际中文模型输入、敌人组成、
初始手牌和意图，并覆盖附魔数值与药水空槽位置；固定空过一回合的测试还比较
第二回合状态。不同模型动作轨迹无需保持后续手牌顺序一致。场景使用的
`scenariofight` 和 `loadout` 都受
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
