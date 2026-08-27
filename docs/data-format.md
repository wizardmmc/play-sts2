# 数据格式

## 人类原始数据

每局人类数据使用本地日期、进阶、最高层和 seed 命名：

```text
data/raw/human/20260827-a0-f17-VX7C7FLRRS/
├── meta.json
├── combat/
│   ├── battle-f002-01.jsonl
│   └── battle-f003-02.jsonl
└── strategy/
    └── decisions.jsonl
```

录制期间先写入同一父目录的隐藏临时目录；正常结束或流中断时，根据已经落盘的
最高层原子改为最终名称。录制器只消费 Mod SSE 中的精确 `action_executed`，不
轮询 `/state`，也不保存高频通用事件流。Mod 若报告
`native_ui_capture_gap`，录制器保留已收到的事实，并把整局标记为不可训练。

一个战斗文件严格对应一场战斗。战斗与战略 JSONL 的每一行都是动作事实：

```json
{
  "event_id": 10009,
  "observed_at": "2026-08-27T02:05:13.625Z",
  "before_state": {"screen": "EVENT"},
  "action": "choose_event_option",
  "parameters": {"option_index": 2}
}
```

`run_id`、层级、`sample_id`、Harness 文本和 messages 都可以由路径、meta 与
`before_state` 重建，所以不在每行重复保存。人工确认修正可以使用稳定字符串
`event_id`，并额外保存最小 `provenance`。

`meta.json` 的核心字段如下：

```json
{
  "schema_version": 1,
  "run_id": "KESBVUW71U",
  "source": "human",
  "started_at": "2026-08-27T13:26:57.647Z",
  "completed_at": "2026-08-27T13:39:59.755Z",
  "played_on": "2026-08-27",
  "character_id": "DEFECT",
  "seed": "KESBVUW71U",
  "ascension": 0,
  "max_floor_reached": 17,
  "battle_count": 9,
  "battle_sample_count": 179,
  "strategic_sample_count": 77,
  "termination_reason": "game_over",
  "training_eligible": true,
  "integrity": {"verified": true, "ineligibility_reasons": []}
}
```

`data/raw/human/splits.json` 使用 seed 按完整局划分 train/dev/test。构建器只读取
非隐藏、已有非空 `termination_reason` 的正式局，并跳过
`training_eligible=false` 的整局；当前 `A7L5LAXFYJ` 保留用于审计但不训练。

## 人类可读 transcript

Transcript 位于与 raw 同名的独立目录：

```text
data/transcripts/20260827-a0-f17-VX7C7FLRRS/
├── combat/
│   └── battle-f002-01.txt
└── strategy/
    └── decisions.txt
```

每个 TXT 只在开头展示一次 system 规则，再按“决策 N / 状态 / 动作”展开。它不
显示 `sample_id`、`event_id` 或 `human_play/...` 等内部定位信息，也不作为训练
事实源。Harness 改动后可以直接覆盖重建：

```bash
uv run play-sts2-transcribe data/raw/human/<规范局目录名>
```

输出目录不得与 raw 源目录重叠；渲染器会在写文件前拒绝危险路径。

## 历史人类数据

历史录制已经从带完整 `before_state` 的精确动作重新物化，而不是从旧 messages
反推 raw。迁移已核对动作数、人工确认修正、旧审计、分卷、进阶和最高层；一次性
迁移代码及其测试已经删除。

当前 raw 共 11 局、4,887 个动作。A7 的 796 个动作不进入训练，其余 10 局得到
4,091 个独立行为监督样本：战斗 2,866 个，战略 1,225 个。

## 游戏知识

单实体事实使用 Markdown frontmatter：

```markdown
---
id: ZAP
name: 电击
type: card
source: mod_export
game_version: v0.107.1
cost: 1
---
## 效果
生成1个闪电充能球。
```

- `web_wiki` 保存结构化网页资料及 `source_detail`。
- `mod_export` 保存当前游戏版本实测数据和 raw JSON。
- `curated-v0.107.1` 保存核验问法与回答变体，包括远古者、算术、目录、角色和
  遭遇知识。

事实错误回到事实源修正；新问法和训练反馈补强进入 curated。

## 可读 SFT 数据集

```text
data/datasets/sft/
├── train.jsonl
├── dev.jsonl
├── test.jsonl
├── manifest.json
└── eval/knowledge/
```

知识问答和每个人类动作各占一行。行为行统一为独立的
`system/user/assistant`，通过当前 Harness 从 raw 重新生成观测和规范
`ACTION:`。Dataset 不缓存 token ID，也不把一场战斗拼成增长的多轮历史。

重建命令为：

```bash
uv run play-sts2-train build-sft
```

更简洁的写入边界和常用命令见 `data/README.md`。
