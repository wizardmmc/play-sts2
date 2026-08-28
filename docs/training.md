# SFT 训练与评测

## 目录边界

- `data/game_knowledge/`：Web Wiki、Mod 实测导出与已核验问法。
- `data/raw/human/`：按局、战斗和战略分片的精确人类动作事实。
- `data/transcripts/`：raw 的可覆盖人类可读投影。
- `data/datasets/sft/`：`train.jsonl`、`validation/dev.jsonl`、
  `eval/test.jsonl`、manifest 与知识考试卷。
- `runs/sft/`：配置副本、逐步指标和固定名称的 `checkpoint-last`。
- `models/base/`、`models/adapters/`、`models/merged/`、`models/serving/`：
  分别保存基座、LoRA、合并 HF 权重和 MLX 服务件。

## 数据链

```text
Web Wiki / Mod 导出 / 核验问法 ──────────────┐
                                              ├─→ data/datasets/sft/*.jsonl
SSE 精确人类动作 ─→ combat/strategy ─→ Harness ┘
```

构建命令：

```bash
uv run play-sts2-train build-sft --mix configs/sft-e3-mix.toml
```

默认知识入口是 `generated-v0.107.1`。地图怪池由固定版本 Mod 的四张地图模型
导出，并统一写成 `encounters/{地图ID}.jsonl` 候选；进场后已经可见的敌人数和
敌人组合不生成监督题。算术只从 `data/raw/human/splits.json` 的训练
名册读取通过 raw 审计的真实攻击意图后确定性生成，不读取验证、测试、不合格或
未分配局，也不复用归档 JSONL。行为分卷读取同一名册；新增
可训练局如果没有明确归属，构建会失败。审计失败的
`A7L5LAXFYJ` 在 raw meta 中标记 `training_eligible=false`，保留追溯但不训练。

当前 E3 数据共有 1,974 条训练样本、528 条验证样本和 600 条测试样本。验证集由
373 条人类行为、123 条知识问答和 32 条独立数字的算术题组成；远古者不进入
训练或验证。角色机制和地图怪池重点保留，高频常规动作按配置上限抽样，未列出的
低频动作全部保留。

知识验证集按“同一事实留出一种未见问法”构建，可用于学习率和训练轮数选择，
不声称实体从未出现。算术验证题改用独立随机种子和新数字生成，不从训练题中
抽取；两者都会主动排除最终算术考试题。独立知识考试卷
位于 `eval/knowledge`；任何考试问题与训练或验证问题完全相同都会令构建失败。
当前训练继承 e2，且真实
战斗、商店和奖励状态本来就会出现实体名称与效果，无法在不删除有价值行为数据
的前提下证明实体从未进入模型。组合算术仍作为独立、可精确评分的泛化指标。

## 代码结构

SFT 实现集中在 `src/play_sts2/training/sft/`：

- `dataset.py`：合并知识与 raw，按整局分卷。
- `encoding.py`：配置、chat template 与 assistant-only mask。
- `trainer.py`：LoRA、分块 loss、优化、checkpoint 与 adapter 发布。
- `merge.py`：安全合并 LoRA、记录来源哈希并原子发布 HF 模型。
- `evaluation.py`：teacher-forced 与生成式行为评测。
- `knowledge_evaluation.py`：知识召回和组合算术探针。
- `game_knowledge/arithmetic.py`：从真实攻击意图生成互斥的算术训练/验证候选。

顶层 `play_sts2.training` 只保留稳定公共入口；CLI 直接依赖 SFT 包。

CUDA 入口位于 `src/play_sts2/training/sft_cuda/`，使用独立配置和
`play-sts2-train sft-cuda` 子命令。它与 Mac 入口共享数据编码、LoRA 装配和
checkpoint 格式，但 CUDA 基座固定使用 BF16，并单独保存 CUDA RNG；LoRA
可训练参数仍必须为 FP32。单次训练只使用一张卡，需要并行实验时最多启动两个
独立单卡任务，不使用五卡 DDP。

## 当前 LoRA 配方

`configs/sft.toml` 继承第二轮 adapter：

```toml
init_adapter = "models/adapters/sft-clean-20260827-native-r16-e2"
epochs = 1
learning_rate = 0.0001
lora_rank = 16
lora_alpha = 32
warmup_steps = 4
max_length = 12288
logits_chunk_size = 2048
checkpoint_steps = 2000
```

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

父 adapter 的血缘记录实际被 PEFT 加载的 `adapter_config.json` 与 LoRA 权重
SHA-256；聊天模板和 tokenizer 由基础模型目录加载，不作为父 adapter 的硬锁。
基础模型、三个数据分卷与 manifest 仍会取摘要；数据和父 adapter 在实际加载前后
还会复核文件身份。训练入口从加载到最终发布全程持有同名运行排他锁，第二个同名
进程会立即失败。精确恢复还分别锁定解析后的设备、基座精度和 adapter 精度，
不能把一次 MPS 运行在 CPU 上伪装成精确续训；任一训练输入变化都会被拒绝。

## 训练

当前落盘数据已按 `configs/sft-e3-mix.toml` 构建为 E3 定向混合。修改候选数据或
配比后，应重新生成召回考试卷并运行同一混合命令，确认无考试题精确泄漏后再训练。

```bash
uv sync --group training
uv run --group training play-sts2-train sft \
  --config configs/sft.toml \
  --name 20260828-sft-clean-native-r16-e3
```

在 CUDA 机器上使用对应独立入口：

```bash
uv run --group training play-sts2-train sft-cuda \
  --config configs/sft-cuda.toml \
  --name 20260828-sft-clean-native-r16-e3
```

训练名必须以 `YYYYMMDD-` 开头。`init_adapter` 表示“用 E2 LoRA 初始化一个新的
E3 运行”，不是恢复中断的同一运行，所以 manifest 仍记录
`approximate_resume=true`。E3 运行中断后使用同名 `--resume`，会恢复优化器、
数据游标和随机状态，并把 `metrics.jsonl` 从 checkpoint 的下一步继续追加：

```bash
uv run --group training play-sts2-train sft \
  --config configs/sft.toml \
  --name 20260828-sft-clean-native-r16-e3 \
  --resume
```

CUDA 续训使用同一名称和独立子命令：

```bash
uv run --group training play-sts2-train sft-cuda \
  --config configs/sft-cuda.toml \
  --name 20260828-sft-clean-native-r16-e3 \
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
  --split dev
uv run --group training play-sts2-train eval-sft-loss \
  --adapter models/adapters/20260828-sft-clean-native-r16-e3 \
  --split test
```

生成式验证用于检查完整输出、截断与 `ACTION:` 外形，但不能把动作外形合法等同
于策略正确：

```bash
uv run --group training play-sts2-train eval-sft \
  --adapter models/adapters/20260828-sft-clean-native-r16-e3 \
  --split dev
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

## 知识探针

当前旧考试卷含 1,352 道召回题和 100 道组合算术。以新的知识候选构建 E3 时，
泄漏断言已经识别出其中 98 道召回题与训练/验证候选题面完全相同，因此这 1,352
道题不能继续被称为 E3 的“未见问法”；确定最终知识配比后必须重新生成召回
考试卷。100 道组合算术已作为算术生成排除项保留。知识题的自动通过口径是忽略空白后的
完整参考答案匹配，宁可产生假阴性也不把“只命中几个数字”误报为正确；组合题
严格检查结论和末尾最终数字。报告保存全部逐题 prompt/reference/answer/pass，
便于人工复核。完整评测使用合并 HF 模型：

```bash
uv run --group training play-sts2-train eval-sft-knowledge \
  --model models/merged/20260828-sft-clean-native-r16-e3-merged
```

## MLX 服务件

先把 `configs/inference.toml` 的 `artifact_id`、`merged_model` 与
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
