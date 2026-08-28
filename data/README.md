# 数据目录

这里把“事实源”和“可重建产物”分开，避免在多个目录手工维护同一份信息。

```text
game_knowledge/ ───────────────┐
                               ├─→ datasets/sft/
raw/human/ ──→ transcripts/ ──┘
```

## 受控来源与重建入口

- `game_knowledge/web_wiki/`：保留网页来源的单实体 Markdown；不直接进入当前
  多问法候选，只允许在离线重建时补充怪物招式与循环。
- `game_knowledge/mod_export/v0.107.1/`：固定版本的 Mod 原始快照与规范事实
  Markdown；包含卡牌完整升级、附魔、怪物招式、四张地图的分级怪池和事件战
  遭遇。统一 Markdown 外形来自渲染模板，但规范事实本身不是模板；这里由导出和
  离线重建命令维护，不直接手改。
- `game_knowledge/supplements/v0.107.1/`：受控补充事实；当前包含从固定版本
  human-rl 实测快照迁移的故障机器人充能球描述、基础数值与集中关系。
- `game_knowledge/event_entries/v0.107.1/`：固定版本事件 UI 入场快照；`events/`
  与 `ancients/` 保存真正进入事件后的已解析文本，状态依赖快照不会被泛化为
  通用监督问法。
- `game_knowledge/reports/v0.107.1.md`：特殊对象、补充来源和待补录缺口汇总。
- `src/play_sts2/game_knowledge/curated/v0.107.1.json`：真正随 Git 和 Python
  包分发的固定版本事实修正及监督排除清单。
- `raw/human/`：人类实际执行的精确动作事实。录制器写入，通常不手工编辑。
- `raw/human/splits.json`：按 seed 将一局的全部样本整体放入训练、验证或测试集；
  任何未列出的可训练局都会令构建失败。

## 可覆盖重建的派生产物

- `game_knowledge/generated-v0.107.1/`：从规范事实可重复生成的按实体 JSONL
  候选；`arithmetic/train` 与 `arithmetic/validation` 只从训练名册内、通过 raw
  审计的实战意图和不同随机种子生成。四张地图的普通、精英和 Boss 怪池写在
  `encounters/`；进场时已知的具体敌人组合不生成问答。这里不是 E3 正式分卷，
  禁止直接手改。
- `transcripts/`：raw 的人类可读投影，不参与训练事实判定。
- `datasets/sft/`：知识与人类动作经当前 Harness 渲染后的训练、验证和测试集；
  路径分别为 `train.jsonl`、`validation/dev.jsonl`、`eval/test.jsonl`。

当前 `datasets/sft/` 已按 `configs/sft-e3-mix.toml` 构建为 E3 定向混合，共
1,974 条训练样本、528 条验证样本和 600 条测试样本。修改知识候选、算术候选或
人类行为后，使用 `play-sts2-train build-sft --mix configs/sft-e3-mix.toml`
整体重建。
构建器会为每个具有多种问法的同一知识事实留出一种问法到验证集，并拒绝训练/
验证问题与 `eval/knowledge` 考试卷完全重合。

不要直接修补派生产物。状态渲染错误应修 Harness；知识事实错误应修 Mod 导出器
或 `src/play_sts2/game_knowledge/curated/v0.107.1.json`；人类行为准入由对应局
`meta.json` 的 `training_eligible` 决定。

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

`meta.json` 持续记录进阶、最高层、战斗数、战斗/战略动作数、终止原因、单步样本
训练资格和整局录制完整性。`training_eligible` 与
`integrity.samples_verified` 表示已有样本能否用于 SFT；`recording_complete`
表示是否从开局录到 `game_over` 且没有已知采集缺口。流中断只把后者设为
`false`，不会否定断流前已经验证的样本。Mod 报告任何
`native_ui_capture_gap` 时，两项都会变为 `false`。录制中的隐藏目录以及尚无终止
原因的目录不会被 dataset builder 读取；`A7L5LAXFYJ` 也因旧录制缺失动作而明确
排除。

## 重建命令

游戏知识应按 `export → rebuild（应用版本化人工修正）→ generate-questions →
generate-arithmetic → review` 顺序重建；两个 generate 命令都不会创建 E3 正式
分卷。完整参数见项目 README。

重建某一局 transcript：

```bash
uv run play-sts2-transcribe \
  data/raw/human/20260827-a0-f17-VX7C7FLRRS
```

输出根必须与 raw 分离；渲染器会拒绝可能覆盖 raw 的重叠路径。

重建完整 SFT 数据集：

```bash
uv run play-sts2-train build-sft --mix configs/sft-e3-mix.toml
```

当前行为样本的战斗与战略动作都使用独立的 `system/user/assistant` 三消息结构；
同一局不会跨训练、验证和测试集，且 raw 不通过物理复制来增加稀有动作权重。
