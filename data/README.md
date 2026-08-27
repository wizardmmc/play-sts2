# 数据目录

这里把“事实源”和“可重建产物”分开，避免在多个目录手工维护同一份信息。

```text
game_knowledge/ ───────────────┐
                               ├─→ datasets/sft/
raw/human/ ──→ transcripts/ ──┘
```

## 可以手工修改的来源

- `game_knowledge/web_wiki/`：保留网页来源的单实体 Markdown。
- `game_knowledge/mod_export/`：按游戏版本保存 Mod 实测导出；事实更新应重新导出。
- `game_knowledge/curated-v0.107.1/`：已核验问法、回答变体和训练反馈补强。新增
  问法直接进入相应类别，不复制 Wiki/Mod 已经能确定的事实。
- `raw/human/`：人类实际执行的精确动作事实。录制器写入，通常不手工编辑。
- `raw/human/splits.json`：按 seed 隔离完整局的 train/dev/test 分卷。

## 可覆盖重建的派生产物

- `transcripts/`：raw 的人类可读投影，不参与训练事实判定。
- `datasets/sft/`：知识与人类动作经当前 Harness 渲染后的训练、开发和测试集。

不要直接修补派生产物。状态渲染错误应修 Harness；知识事实错误应修来源；新问法
应加到 curated；人类行为准入由对应局 `meta.json` 的 `training_eligible` 决定。

## 人类 raw 契约

一局目录名固定为 `YYYYMMDD-a<进阶>-f<最高层>-<seed>`，日期使用上海时区：

```text
raw/human/20260827-a0-f17-VX7C7FLRRS/
├── meta.json
├── combat/
│   └── battle-f002-01.jsonl
└── strategy/
    └── decisions.jsonl
```

每个战斗文件保存一场战斗，每行保存一个动作前状态、动作、参数、观察时间和稳定
事件 ID。Raw 不保存 `messages`、`readable` 或总事件流 `events.jsonl`。字符串事件
ID 只用于带明确 provenance 的人工确认修正。

`meta.json` 持续记录进阶、最高层、战斗数、战斗/战略动作数、终止原因、完整性
结论与训练资格。录制中的隐藏目录以及尚无终止原因的目录不会被 dataset builder
读取；Mod 报告任何 `native_ui_capture_gap` 时，录制仍保留，但整局自动标为
`training_eligible=false`。`A7L5LAXFYJ` 也因旧录制缺失动作而明确排除。

## 重建命令

重建某一局 transcript：

```bash
uv run play-sts2-transcribe \
  data/raw/human/20260827-a0-f17-VX7C7FLRRS
```

输出根必须与 raw 分离；渲染器会拒绝可能覆盖 raw 的重叠路径。

重建完整 SFT 数据集：

```bash
uv run play-sts2-train build-sft
```

当前行为样本的战斗与战略动作都使用独立的 `system/user/assistant` 三消息结构；
同一局不会跨 train/dev/test，且 raw 不通过物理复制来增加稀有动作权重。
