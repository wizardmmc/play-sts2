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
轮询 `/state`，也不保存高频通用事件流。流中断前已经验证的单步样本仍可用于
SFT，但 `recording_complete=false`；Mod 若报告 `native_ui_capture_gap`，录制器
保留已收到的事实，并把整局标记为不可训练且不完整。

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
  "schema_version": 2,
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
  "recording_complete": true,
  "integrity": {"samples_verified": true, "ineligibility_reasons": []}
}
```

`training_eligible` 只回答已经落盘的单步样本能否进入 SFT，并与
`integrity.samples_verified` 保持一致。`recording_complete` 单独回答该局是否从
开局录到 `game_over` 且没有已知采集缺口；它为 `false` 不会自动排除已经验证的
单步样本。

`data/raw/human/splits.json` 使用 seed 按局整体划分训练、验证和测试集。构建器
拒绝任何未在名册中声明的可训练局，并只读取
非隐藏、已有非空 `termination_reason` 的正式局，并跳过
`training_eligible=false` 的整局；当前 `A7L5LAXFYJ` 保留用于审计但不训练。

## 可读 transcript

Transcript 镜像 raw 的来源层级，再使用与 raw 局同名的独立目录：

```text
data/transcripts/
├── agent/
├── human/
│   └── 20260827-a0-f17-VX7C7FLRRS/
│       ├── combat/
│       │   └── battle-f002-01.txt
│       └── strategy/
│           └── decisions.txt
└── human_combat_solver/
```

每个 TXT 只在开头展示一次 system 规则，再按“决策 N / 状态 / 动作”展开。它不
显示 `sample_id`、`event_id` 或 `human_play/...` 等内部定位信息，也不作为训练
事实源。Harness 改动后可以直接覆盖重建：

```bash
uv run play-sts2-transcribe data/raw/human/<规范局目录名>
```

渲染器只接受 `agent`、`human` 和 `human_combat_solver` 三种
`meta.json.source`，并自动选择对应来源目录。输出根不得与 raw 源目录重叠；
渲染器会在写文件前拒绝危险路径。

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
- `mod_export/v0.107.1` 保存经离线修正后的规范事实；
  `generated-v0.107.1` 保存多问法候选，以及由实战攻击意图确定性生成的算术
  训练/验证候选。

Mod 原始 `acts` 负责四张地图的分级怪池，生成后统一归入
`generated-v0.107.1/encounters/{地图ID}.jsonl`。原始 `encounters` 中进场即可
观察到的敌人数和具体组合不生成监督题。故障机器人充能球由固定版本受控补录先
合并进 `characters/DEFECT.md`，再生成角色问答。

事实错误回到 Mod 导出、受控补充或版本化 curated 事实修正规则；新问法修改
`src/play_sts2/game_knowledge/generation.py` 中的生成规则，不能直接修补
`generated-v0.107.1` JSONL。

## 可读 SFT 数据集

```text
data/datasets/sft/
├── manifest.json
├── train/
│   ├── arithmetic/*.jsonl
│   ├── cards/*.jsonl
│   ├── ...其余知识类别/*.jsonl
│   ├── combat/<run-id>/<battle-key>.jsonl
│   └── strategy/<run-id>.jsonl
├── validation/
│   └── 与 train 相同的目录骨架
└── eval/
    └── 与 train 相同的目录骨架
```

知识问答和每个人类动作各占一行。行为行统一为独立的
`system/user/assistant`，通过当前 Harness 从 raw 重新生成观测和规范
`ACTION:`。Dataset 不缓存 token ID，也不把一场战斗拼成增长的多轮历史。
知识候选使用可读 `fact_id` 和显式 `question_role`；每个正式事实至少有一条 train
问法，并恰好有一条 validation 和 eval 问法。三种问题原文发生精确重合时拒绝
发布。算术同样显式标记三种用途，并使用互不相同的随机种子；`case_id` 由题型和
完整计算语义/操作数生成，任何运算案例跨分卷重合也会被拒绝。

重建命令为：

```bash
uv run play-sts2-train build-sft --mix configs/sft-e3-mix.toml
```

更简洁的写入边界和常用命令见 `data/README.md`。
