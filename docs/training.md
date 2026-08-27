# SFT 训练与评测

## 目录边界

- `data/game_knowledge/`：Web Wiki、Mod 实测导出与已核验问法。
- `data/raw/human/`：按局、战斗和战略分片的精确人类动作事实。
- `data/transcripts/`：raw 的可覆盖人类可读投影。
- `data/datasets/sft/`：`train/dev/test.jsonl`、manifest 与评测集。
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
uv run play-sts2-train build-sft
```

默认输入是 `data/game_knowledge/` 与 `data/raw/human/`。知识包含
`ancients/arithmetic/catalog/characters/encounters` 等核验资产及其变化问法；
行为 split 读取 `data/raw/human/splits.json`。审计失败的
`A7L5LAXFYJ` 在 raw meta 中标记 `training_eligible=false`，保留追溯但不训练。

当前构建结果为 17,297/373/600 条 train/dev/test。有效人类动作共 4,091 条，
其中战斗 2,866、战略 1,225。两层行为都使用独立
`system/user/assistant`，不再把整场战斗拼成增长的多轮样本。

知识探针只报告“未见问法”而不声称“未见实体”：当前训练继承 e2，且真实
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

顶层 `play_sts2.training` 只保留稳定公共入口；CLI 直接依赖 SFT 包。

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

训练使用 MPS bfloat16、梯度检查点、梯度裁剪和非有限梯度检查。词表投影与交叉
熵按 2,048 token 分块，避免物化 `sequence × 151k vocabulary` 的完整 logits。
模板通过 `{% generation %}` 标记取得精确 assistant loss mask，并先验证标记模板
与服务模板逐 token 等价。

`checkpoint_steps = 2000` 表示每 2,000 个优化步覆盖同一个
`runs/sft/<name>/checkpoint-last`，不会按步数生成无限多个 adapter 目录。最终
adapter 仍通过暂存目录原子发布。

## 训练

```bash
uv sync --group training
uv run --group training play-sts2-train sft \
  --config configs/sft.toml \
  --name sft-clean-20260827-native-r16-e3
```

从 `init_adapter` 续训会保留 LoRA 权重，但不恢复优化器动量或数据游标，因此
manifest 会记录 `approximate_resume=true`。先用独立运行名和 `--max-steps 1`
做真实模型冒烟，避免与正式目录冲突。

## 行为评测

轮次比较优先使用 assistant-only teacher-forced loss、perplexity 和 token
accuracy。当前目标是为 GRPO 打好基础，不必为了防御性横向比较重跑 e1/e2；
需要诊断 e3 时再运行：

```bash
uv run --group training play-sts2-train eval-sft-loss \
  --adapter models/adapters/sft-clean-20260827-native-r16-e3 \
  --split dev
uv run --group training play-sts2-train eval-sft-loss \
  --adapter models/adapters/sft-clean-20260827-native-r16-e3 \
  --split test
```

生成式验证用于检查完整输出、截断与 `ACTION:` 外形，但不能把动作外形合法等同
于策略正确：

```bash
uv run --group training play-sts2-train eval-sft \
  --adapter models/adapters/sft-clean-20260827-native-r16-e3 \
  --split dev
```

## 合并模型

训练成功后从配置读取基座模型并合并指定 adapter：

```bash
uv run --group training play-sts2-train merge-sft \
  --adapter models/adapters/sft-clean-20260827-native-r16-e3
```

默认输出为 `models/merged/sft-clean-20260827-native-r16-e3-merged/`。合并固定在
CPU bfloat16 上执行，调用 `merge_and_unload(safe_merge=True)`，保存失败不会发布
半成品；排他 rename 保证发布竞态也不会覆盖已有目标。命令会先核对训练清单或
旧 adapter 声明的本地基座，且输入在合并期间变化时拒绝发布。
`merge_manifest.json` 保存全部基座权重、adapter、实际 tokenizer 来源的
SHA-256、血缘校验方式以及依赖版本。

## 知识探针

共 1,352 道未见问法和 100 道组合算术。知识题的自动通过口径是忽略空白后的
完整参考答案匹配，宁可产生假阴性也不把“只命中几个数字”误报为正确；组合题
严格检查结论和末尾最终数字。报告保存全部逐题 prompt/reference/answer/pass，
便于人工复核。完整评测使用合并 HF 模型：

```bash
uv run --group training play-sts2-train eval-sft-knowledge \
  --model models/merged/sft-clean-20260827-native-r16-e3-merged
```

## MLX 服务件

把新合并模型转换到独立目录：

```bash
uv sync --group inference
uv run --group inference play-sts2-model prepare \
  --source models/merged/sft-clean-20260827-native-r16-e3-merged \
  --output models/serving/sft-clean-20260827-native-r16-e3-mlx-8bit
```
