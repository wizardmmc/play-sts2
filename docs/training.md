# RL 训练

本节是长期 RL 的正式运行规范，覆盖课程、预算、准入门槛与单轮执行顺序。
各轮实验的执行状态与证据保存在 `runs/rl/` 对应 run 目录的实验文档里。

## 资源与单轮顺序

### 多步 loss 归一化

新训练配置显式使用 `loss_normalization="group_tokens"`。原来的 `per_arm` 会按每条
完整轨迹各自的 completion token 数平均，长轨迹每个 token 的信用更弱，并破坏
共享前缀的正负优势抵消；这与整局目标不一致。GLM 复核与实际 autograd 测试已确认。

共同分母取 `sum(importance_i * tokens_i) / sum(importance_i)`；普通八臂组退化为
平均轨迹 token 数，Tree 仍保留原始 `pi_old/q/K` 系数与来源0.75/0.25配比。loss 与
post-step 指标使用同一权重。该组常数缩放不是严格无偏梯度的保证，仍须游戏评估。
历史缺少此字段的配置按 `per_arm` 解释，不能在 exact-resume 中切换公式。

当前修正先用相同 S3/B3、相同冻结数据做独立分支对照；没有把旧 rollout 改名，
也没有从被拒绝的 B1 初始化。所有当前对照保持原训练后端。

### 执行拓扑

- 当前进阶记为 `A`，初始为 A0；同一轮所有 rollout 必须使用同一个 `A`。
- Mac 固定启动两个隔离无头 STS2 v0.111.0 实例，服务器默认只使用一张 GPU；本地
  不做模型推理或训练。
- 一轮开始后冻结 `(S_n, B_n, generation profile, environment)`，依次执行：

```text
同一 seed 的 K=8 完整 backbone
→ 全局候选池选择 1～2 个 terminal Tree checkpoint
→ GiGPO + Tree 更新战略 residual 为 S_(n+1)
→ 从 backbone + Tree 入口选择战斗场景
→ 每场景 K=8 战斗 GRPO，更新战斗 residual 为 B_(n+1)
→ 仅在 controller 标记 block 结束时跑 3 frozen + 1 fresh 完整验证
→ 记录 TensorBoard、cycle journal 与长期 curriculum 状态
```

backbone 保存全部宏候选元数据，每条 episode 最多物化一个早期和一个较晚 checkpoint；
Tree 在全局池中按节点类型和早/晚分层随机抽取，固定随机 seed，并受墙钟预算约束，
不能在八局中每局各展开几个节点。每轮默认最多两个 Tree checkpoint：二至四岔 MAP
按实际根选项数采分支，事件、奖励、火堆和商店等组合宏采 4 条 iid 完整计划。Tree
游戏墙钟不得超过训练游戏总时间的 40%。

当已经能多次通过第一幕后，后续新训练采样使用
`collect-rl-gigpo --late-checkpoint-min-floor 18`，把晚期物化预算保留给第二幕；
仍每局最多一早一晚。未达到18层则不消耗晚期预算，候选元数据照常记录。
默认值10保留历史复现；当前同数据归一化对照不改变采样配置。

冷启动战略回报为：

```text
R_terminal = 4 × 完整胜利
R_progress = 1 × 击败 Boss 数 + 0.01 × 终局楼层
             + 0.001 × 胜局终点 HP / 最大 HP
```

GiGPO 分别形成 episode 与 milestone advantage；Tree 使用同一完整终局回报，不另造
卡牌、遗物或路线强度表。没有胜局时 Boss 和楼层提供密集课程信号；出现胜局后完整
胜利自然成为主要信号。

战斗场景优先选择死亡、模型失败、高战损和精英/Boss，同时在最近 20 个已选场景中
累计保留至少 20% 普通战斗。每场景仍是严格 8 条同入口学生 rollout，恢复训练默认
`reward_scheme="core_no_turn"`：

```text
R_battle = 3 × 战胜 + 6 × clip((终局HP-入场HP)/入场HP, -1, 0.25)
           - 3 × 死亡
```

旧 `core` 另含 `-0.05 × 回合数`，只保留为历史对照；当前不让它在全灭组中奖励
更快死亡。采集配额与 learner 必须使用相同奖励口径判断是否有方差。

第一轮正式长训不启用 CombatSolver DAgger：`dagger_weight=0` 且不提供
`dagger_root`。零 reward 方差的战斗 group 直接跳过；只有未来明确重新启用独立
DAgger loss 时，才允许保留零优势 group 做纯监督更新。既有 DAgger 代码和 smoke
只作为可选回退与对照，不进入当前主路径。

## 课程状态机

### 当前进阶尚未首次通关

使用一个固定 target seed，严格执行完整的 `3 target + 1 fresh` 周期：初轮 target
加再重复两轮 target，然后训练一轮从未见过的 fresh seed。直到 target seed 的某轮
K=8 中至少出现一局完整胜利，进入迁移阶段。

### 首次通关后的迁移阶段

继续执行 `3 target + 1 fresh`。fresh seed 即使失败也已经属于“见过”，以后不能再次
冒充 fresh。至少两个不同 fresh seed 都能在各自第一次 K=8 训练中出现一局完整胜利，
且 frozen validation 没有硬退化，才进入正式循环；单个幸运新图不够。

### 正式循环

按五个 seed block 固定调度：

```text
20% 已通关 seed
40% 困难 seed
40% 从未训练的 fresh seed
```

一个 seed 最多连续训练三轮；任一轮出现完整胜利即可结束该 block 并进入 cleared 池。
三轮仍未通关，或表现低于最近三次 3+1 validation 的滚动基线，则进入 hard 池。池中
分类优先比较 Boss/幕进度，再解释平均楼层；controller 当前以胜局和滚动平均楼层完成
最小自动分类，Boss 指标同时落盘供 GPT 监工判断。

当连续三个不同、此前未训练的 fresh seed 都在各自某轮 K=8 中至少 4 局完整胜利，
并且 frozen validation、固定战斗回归和 held-out Tree 回归同时通过时，把当前进阶
提高到 `A+1`。新难度以最后一条合格 seed 作为 target，清空当前难度 seed 池，重新
从“尚未首次通关”开始；旧难度仍保留便宜回归，不参加新难度 reward normalization。

## 验证、TensorBoard 与 checkpoint

cold-start 与 transfer 严格按一个 `3 target + 1 training-fresh` 训练块只做一次完整
验证：前三个 target cycle 跳过 3+1，training-fresh cycle 更新完双 residual 后，再用
三个永不训练的 frozen seed 和一条独立 validation-fresh seed 各玩一局。跳过完整
验证的 cycle 仍需通过 K=8 合同、KL/ratio/梯度有限、battle/Tree group 准入等便宜
硬门槛，policy pair 只能标为 provisional。formal 阶段默认每轮完整验证；若验证墙钟
超过总轮次的 20%，改为每两轮一次。任何阶段都不能让大部分时间花在验证上。

`battle-grpo` 与 `strategy-grpo` 在各自 run 目录下写 `tensorboard/`；本地
`eval-rl-policy` 和长期 curriculum CLI 通过 `--tensorboard-dir` 写入本轮公共 event
目录。服务器上的 run 产物同步回本地后，TensorBoard 递归读取 `runs/rl/`，分别展示 loss、
KL、ratio、梯度、胜率、楼层、Boss、战斗战损、frozen/fresh 验证、阶段与进阶：

```bash
uv run --group training tensorboard --logdir runs/rl
```

TensorBoard event 只追加小型标量，不触发模型 checkpoint。完整 LoRA/AdamW/RNG
checkpoint 默认每 50 个战斗 group 覆盖一次，formal 配置不得小于 20；主动暂停强制
保存，正常完成直接发布最终 adapter，不重复写 `checkpoint-last`。

长期 controller 只负责 seed、阶段、进阶和池状态，现有 cycle CLI 继续负责实际采样、
两次 adapter 更新与验证：

```bash
uv run play-sts2-train init-rl-curriculum \
  --state runs/rl/curriculum.json \
  --target-seed <TARGET> --ascension 0

uv run play-sts2-train plan-rl-curriculum \
  --state runs/rl/curriculum.json --fresh-seed <NEW_SEED>

uv run --group training play-sts2-train update-rl-curriculum \
  --state runs/rl/curriculum.json \
  --seed <ACTUAL_SEED> --source <target|cleared|hard|fresh> \
  --ascension <A> --cycle-index <N> --wins <0..8> --attempts 8 \
  --mean-floor <F> --bosses-cleared <N> \
  --rolling-validation-mean-floor <F> \
  --validation-stable --battle-regression-passed --tree-regression-passed \
  [--validation-over-budget] \
  --tensorboard-dir runs/rl/tensorboard
```

# SFT 训练与评测

## 目录边界

- `configs/`：每台机器的真实配置，整体忽略；可提交示例位于同构的
  `configs-template/`，运行时不会回退读取模板。
- `data/game_knowledge/`：Web Wiki、Mod 实测导出与已核验问法。
- `data/raw/human/`：按局、战斗和战略分片的精确人类动作事实。
- `data/transcripts/{agent,human,human_combat_solver}/`：按录制来源隔离的可覆盖
  可读投影。
- `data/datasets/sft/`：同构的 `train/`、`validation/`、`eval/` 目录树与
  `manifest.json`；知识按实体、行为按战斗或整局战略拆分。
- `runs/sft/`：配置副本、逐步指标和固定名称的 `checkpoint-last`。
- `models/base/`、`models/adapters/`、`models/merged/`、`models/serving/`：
  分别保存基座、LoRA、合并 HF 权重和 MLX 服务件。

## 数据链

```text
Web Wiki / Mod 导出 / 核验问法 ──────────────┐
                                              ├─→ data/datasets/sft/{train,validation,eval}/
SSE 精确人类动作 ─→ combat/strategy ─→ Harness ┘
```

构建命令：

```bash
uv run play-sts2-train build-sft \
  --knowledge-root data/game_knowledge/generated-v0.111.0 \
  --mix configs/sft/sft-e4-mix.toml
```

CLI 的兼容默认值仍是 `generated-v0.107.1`；E4 必须显式选择
`generated-v0.111.0`。地图怪池由固定版本 Mod 的四张地图模型
导出，并统一写成 `encounters/{地图ID}.jsonl` 候选；进场后已经可见的敌人数和
敌人组合不生成监督题。算术只从 `data/raw/human/splits.json` 的训练
名册读取通过 raw 审计的真实攻击意图后确定性生成，不读取验证、测试、不合格或
未分配局，也不复用归档 JSONL。行为分卷读取同一名册；新增
可训练局如果没有明确归属，构建会失败。审计失败的
`A7L5LAXFYJ` 在 raw meta 中标记 `training_eligible=false`，保留追溯但不训练。

当前 E4 数据共有 34,123 条训练候选、6,622 条验证样本和 6,849 条最终评测样本；
两轮训练各自实际选择 10,247 条，其中知识问法不重复，算术和行为锚点复用。
E4 的远古者不进入正式数据；全部 1,552 个知识实体和 5,969 个显式事实进入 train。
每个事实提供 `training_epoch=1..5` 五条训练问法，训练器第 N 轮只读取第 N 条；
validation/eval 再使用同一事实的两条独立自然改写。六类算术题分别使用三个随机种子，
高频常规动作只在 train 按配置上限抽样，未列出的低频动作全部保留。

知识、算术和人类行为都直接位于三棵目录中，不再维护独立 probe 文件。任何知识
问题跨用途精确重合、事实缺少五轮训练或任一留出问法、类别或算术桶为空，都会令
构建失败。算术与行为不声明 `training_epoch`，因此每轮作为回放锚点复用。
当前训练继承冻结的 E3，且真实
战斗、商店和奖励状态本来就会出现实体名称与效果，无法在不删除有价值行为数据
的前提下证明实体从未进入模型。组合算术仍作为独立、可精确评分的泛化指标。

## 代码结构

SFT 实现集中在 `src/play_sts2/training/sft/`：

- `dataset.py`：合并知识与 raw，按整局分卷。
- `encoding.py`：配置、chat template 与 assistant-only mask。
- `trainer.py`：LoRA、分块 loss、优化、checkpoint 与 adapter 发布。
- `merge.py`：安全合并 LoRA、记录来源哈希并原子发布 HF 模型。
- `evaluation.py`：teacher-forced 与生成式行为评测。
- `knowledge_evaluation.py`：从 eval 树读取知识召回和组合算术题。
- `game_knowledge/arithmetic.py`：从真实攻击意图生成互斥的三用途算术候选。

顶层 `play_sts2.training` 只保留稳定公共入口；CLI 直接依赖 SFT 包。

CUDA 入口位于 `src/play_sts2/training/sft_cuda/`，使用独立配置和
`play-sts2-train sft-cuda` 子命令。它与 Mac 入口共享数据编码、LoRA 装配和
checkpoint 格式，但 CUDA 基座固定使用 BF16，并单独保存 CUDA RNG；LoRA
可训练参数仍必须为 FP32。单次训练只使用一张卡，需要并行实验时最多启动两个
独立单卡任务，不使用五卡 DDP。

## 当前 LoRA 配方

E4 CUDA 候选从冻结 E3 adapter 初始化：

```toml
init_adapter = "models/adapters/20260828-sft-clean-native-r16-e3"
epochs = 2
learning_rate = 0.00005  # 对照组为 0.0001
lora_rank = 16
lora_alpha = 32
gradient_accumulation_steps = 8
warmup_steps = 100
max_length = 12288
logits_chunk_size = 2048
checkpoint_steps = 2000
```

E5 从冻结 E4 开始一轮单变量容量 A/B。两路使用同一数据、seed、`1e-4` 和第 3
套知识问法；r32 使用函数等价扩容，不随机重训父 adapter：

```toml
init_adapter = "models/adapters/20260829-sft-e4-knowledge-r16-lr1e4"
epochs = 1
knowledge_epoch_start = 3
learning_rate = 0.0001
lora_rank = 16       # 对照组为 32
lora_alpha = 32      # 对照组为 64
expand_init_adapter = false  # r32 对照组为 true
```

`knowledge_epoch_start` 只改变本次运行首轮选择的训练问法；没有轮次的算术和行为
仍作为通用锚点加入。`expand_init_adapter=true` 要求目标 rank 更高、
`alpha/rank` 不变且父 adapter 未启用 rsLoRA；扩容复制旧通道、保留新增 A 的标准
初始化并把新增 B 置零。

`1e-4` 是 Qwen LoRA 的常见量级，也是前两轮实际使用的设置；它不是裸全参数微调
的通用学习率。“全层 LoRA”指 adapter 覆盖 Qwen3.5-4B 的全部 32 个 decoder
layer，并不是解冻 4B 基座：24 个 linear-attention layer 覆盖
`in_proj_a/in_proj_b/in_proj_qkv/in_proj_z/out_proj`，8 个 full-attention layer
覆盖 `q_proj/k_proj/v_proj/o_proj`。

训练基座在 MPS 上使用 BF16，LoRA 可训练参数由 PEFT 自动提升并由训练器硬校验
为 FP32；不是 FP32 会在创建 AdamW 前立即失败。训练同时使用梯度检查点、梯度
裁剪和非有限梯度检查。词表投影与交叉熵按 2,048 token 分块，避免物化
`sequence × 151k vocabulary` 的完整 logits。
模板通过 `{% generation %}` 标记取得精确 assistant loss mask，并先验证标记模板
与服务模板逐 token 等价。

`checkpoint_steps = 2000` 表示每 2,000 个优化步覆盖同一个
`runs/sft/<name>/checkpoint-last`，不会按步数生成无限多个 adapter 目录。
checkpoint 同时保存约 55 MB LoRA 权重、约 110 MB AdamW 状态、样本游标、当前
洗牌顺序和 CPU 及所选 MPS/CUDA 设备随机状态；r16 实测净文件约 165 MB，按 180～200 MB 预算
可以给文件系统元数据留出余量。权重与训练状态在同一
暂存目录写完后再原子切换，避免恢复到不同 step。最终 adapter 仍通过暂存目录
原子发布。若中断发生在 checkpoint 之后，恢复会原子截掉领先于 checkpoint 的
指标行，再从对应下一步重算；不会把没有对应权重的日志误当作已完成进度。
此外，每个完整 epoch 结束后都会更新 `checkpoint-last`，并原子发布不可覆盖的
`checkpoint-epoch-N` 实体目录。它保存完全相同的 adapter、AdamW、游标和 RNG
状态，用于逐轮 validation；中途 step checkpoint 不再冒充轮末模型。
若进程恰好在轮末指针更新后、不可变目录发布前中断，精确恢复会识别
`order=[]` 的轮末状态，从 `checkpoint-last` 原子补齐对应实体目录再继续。旧运行的
`config.json` 没有 `knowledge_epoch_start`/`expand_init_adapter` 时按默认
`1`/`false` 归一化，非默认变化仍会被拒绝。

父 adapter 的血缘记录实际被 PEFT 加载的 `adapter_config.json` 与 LoRA 权重
SHA-256；聊天模板和 tokenizer 由基础模型目录加载，不作为父 adapter 的硬锁。
基础模型、三个数据分卷与 manifest 仍会取摘要；数据和父 adapter 在实际加载前后
还会复核文件身份。训练入口从加载到最终发布全程持有同名运行排他锁，第二个同名
进程会立即失败。精确恢复还分别锁定解析后的设备、基座精度和 adapter 精度，
不能把一次 MPS 运行在 CPU 上伪装成精确续训；任一训练输入变化都会被拒绝。

## 训练

当前落盘数据已按 `configs/sft/sft-e4-mix.toml` 从 `generated-v0.111.0` 构建。修改
候选数据或配比后，应重新生成知识与算术候选并运行同一混合命令。构建器会逐事实
检查五轮训练与两套留出问法，并拒绝知识题面或算术案例跨用途泄漏。

E5 同时读取旧人类 raw 与人机协作 raw，附加根使用自己的 `splits.json` 选择固定
局集合；未列入该附加名册的后续录制不会静默进入已经冻结的 E5：

```bash
uv run play-sts2-train build-sft \
  --knowledge-root data/game_knowledge/generated-v0.111.0 \
  --human-root data/raw/human \
  --additional-human-root data/raw/human_combat_solver \
  --output-root data/datasets/e5/sft \
  --mix configs/sft/sft-e5-mix.toml
```

后续实验不能为了换分卷而覆盖冻结 E5 的 raw 名册。E6 使用一个跨所有 human root
的独立权威名册：

```bash
uv run play-sts2-train build-sft \
  --knowledge-root data/game_knowledge/generated-v0.111.0 \
  --human-root data/raw/human \
  --additional-human-root data/raw/human_combat_solver \
  --run-splits data/raw/human_combat_solver/splits-e6.json \
  --output-root data/datasets/e6/sft \
  --mix configs/sft/sft-e6-mix.toml
```

显式 `--run-splits` 必须存在、非空，并且其中每个 run 都能产生合格样本；未列出的
未来录制会被忽略。该参数不能与 `--train-run`、`--validation-run` 或 `--eval-run`
混用，避免一套输入被另一套静默覆盖。

行为 JSONL 保留 `behavior_origin`、`action_source`、胜负与录制 gap，但这些字段不
进入模型 messages。混合器的旧行为上限不裁剪 `human_combat_solver` 来源；算术可
通过 `[arithmetic.train_per_kind]` 按六类确定性抽样。
manifest 的 `human.roots` 保存每个 raw 根实际采用的整局分卷，`human.runs` 逐局保存
root、split、胜负、录制完整性、gap、样本数和动作来源计数；仅用于构建审计的 root
字段不会写入训练 JSONL。

```bash
uv sync --group training
uv run --group training play-sts2-train sft-cuda \
  --config configs/sft/sft-e4-cuda-lr5e5.toml \
  --name 20260829-sft-e4-knowledge-r16-lr5e5
```

第二张空闲 CUDA 卡使用同数据、同起点的 `1e-4` 对照配置：

```bash
uv run --group training play-sts2-train sft-cuda \
  --config configs/sft/sft-e4-cuda-lr1e4.toml \
  --name 20260829-sft-e4-knowledge-r16-lr1e4
```

训练名必须以 `YYYYMMDD-` 开头。`init_adapter` 表示“用 E3 LoRA 初始化一个新的
E4 运行”，不是恢复中断的同一运行，所以 manifest 仍记录
`approximate_resume=true`。E4 运行中断后使用同名 `--resume`，会恢复优化器、
数据游标和随机状态，并把 `metrics.jsonl` 从 checkpoint 的下一步继续追加：

```bash
uv run --group training play-sts2-train sft-cuda \
  --config configs/sft/sft-e4-cuda-lr5e5.toml \
  --name 20260829-sft-e4-knowledge-r16-lr5e5 \
  --resume
```

CUDA 续训使用同一名称和独立子命令：

```bash
uv run --group training play-sts2-train sft-cuda \
  --config configs/sft/sft-e4-cuda-lr5e5.toml \
  --name 20260829-sft-e4-knowledge-r16-lr5e5 \
  --resume
```

先用独立的日期前缀运行名和 `--max-steps 1` 做真实模型冒烟，避免与正式目录
冲突；一步冒烟只证明训练链路可运行，不代表模型取得有效进度。

## 行为评测

轮次比较优先使用 assistant-only teacher-forced loss、perplexity 和 token
accuracy。当前目标是为 GRPO 打好基础，不必为了防御性横向比较重跑 e1/e2；
需要诊断 e3 时再运行：

```bash
uv run --group training play-sts2-train eval-sft-loss \
  --adapter models/adapters/20260828-sft-clean-native-r16-e3 \
  --split validation
uv run --group training play-sts2-train eval-sft-loss \
  --adapter models/adapters/20260828-sft-clean-native-r16-e3 \
  --split eval
```

生成式验证用于检查完整输出、截断与 `ACTION:` 外形，但不能把动作外形合法等同
于策略正确。行为行会使用数据中保存的可见动作和精确合法动作集合，分别统计空输出、
格式错误、不可用动作与越界参数；不从 reasoning 恢复，也不重试。报告同时写明
`enable_thinking=false` 和 `max_retries=0`：

```bash
uv run --group training play-sts2-train eval-sft \
  --adapter models/adapters/20260828-sft-clean-native-r16-e3 \
  --split validation \
  --temperature 0
uv run --group training play-sts2-train eval-sft \
  --adapter models/adapters/20260828-sft-clean-native-r16-e3 \
  --split validation \
  --temperature 0.8
```

## 合并模型

训练成功后从配置读取基座模型并合并指定 adapter：

```bash
uv run --group training play-sts2-train merge-sft \
  --adapter models/adapters/20260828-sft-clean-native-r16-e3
```

默认输出为 `models/merged/20260828-sft-clean-native-r16-e3-merged/`。合并固定在
CPU bfloat16 上执行，调用 `merge_and_unload(safe_merge=True)`，保存失败不会发布
半成品；排他 rename 保证发布竞态也不会覆盖已有目标。命令会先核对训练清单或
旧 adapter 声明的本地基座，且输入在合并期间变化时拒绝发布。
`merge_manifest.json` 保存全部基座权重、adapter、实际 tokenizer 来源的
SHA-256、血缘校验方式以及依赖版本。

## 知识与算术评测

最终评测直接递归读取 `data/datasets/sft/eval/`。其中每个显式知识事实有一条未在
train/validation 出现的 `eval` 问法；六类算术题除了题面互斥，还通过 `case_id`
保证完整计算语义和操作数不跨分卷复用。`combat/` 与 `strategy/` 由行为评测入口
处理，因此知识评测器会跳过这两个目录。知识题按忽略空白后的完整参考答案评分，
算术题使用结构化结论与数字字段评分。报告保存逐题 prompt/reference/answer/pass，
便于人工复核。完整评测使用合并 HF 模型：

```bash
uv run --group training play-sts2-train eval-sft-knowledge \
  --model models/merged/20260828-sft-clean-native-r16-e3-merged \
  --eval-root data/datasets/sft/eval
```

## MLX 服务件

先把 `configs/inference/inference.toml` 的 `artifact_id`、`merged_model` 与
`serving_model` 更新为同一轮不可变产物，再把新合并模型转换到独立目录：

```bash
uv sync --group inference
uv run --group inference play-sts2-model prepare
```

转换会解析合并清单的 adapter/输出血缘，把小型 `merge_manifest.json` 的摘要、
artifact ID、源目录、量化/EOS 和 thinking 模板功能指纹写入
`serving_manifest.json`；服务启动前会用实际 MLX config 和固定对话的渲染文本
及 token IDs 重新校验，不扫描模型权重。默认 `no-think` profile 与 SFT 编码的
`enable_thinking=false` 保持一致；需要比较动态模板的思考分支时显式使用
`--profile think`。冒烟与游戏 Runtime 还会在进入游戏前用配置中的绝对服务目录
执行一次真实生成，并在每次后续请求中固定且核对该模型标识；因此端口重启到旧
服务或权重无法加载都不会被健康端点掩盖。
