# 后训练边界

- `data/game_knowledge/` 保存按来源隔离的游戏事实与原始快照。
- `data/datasets/sft/` 保存可读、未 token 化的 SFT messages。
- `data/datasets/rl/` 保存可复用的战斗场景和后续 RL 输入。
- `runs/` 保存每次实验的配置副本、日志、指标和中间 checkpoint。
- `models/base/` 保存未经项目训练的基座模型。
- `models/adapters/`、`models/merged/` 和 `models/serving/` 只保存选定产物。
- `configs/` 保存能够人工阅读和修改的训练参数。

## 当前数据链

```text
Web Wiki / Mod 实测导出 ──→ 单实体 Markdown ──┐
                                              ├─→ SFT messages
人类原始轨迹 ──→ 精确转录 ──→ 当前 Harness ──┘
```

使用方式：

```bash
uv run play-sts2-knowledge import-wiki /path/to/wiki
uv run play-sts2-knowledge export
uv run play-sts2-train build-sft
```

数据集构建阶段不保存模型相关 token ID。训练器以后在启动时读取目标模型的
chat template，并只对 assistant 内容计算损失。这样 Harness 变化后可以从原始
知识和精确决策重新生成数据，而不是继续训练旧提示词的缓存结果。

训练器与生成式评测尚未实现。下一阶段再加入 PyTorch、Transformers、PEFT 和
MPS LoRA；正式训练前还需要明确独立的 dev/test 人类局。
