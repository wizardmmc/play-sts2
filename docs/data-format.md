# 原始轨迹格式

每局原始轨迹位于：

```text
data/raw/<human|agent>/<run_id>/
├── meta.json
└── events.jsonl
```

`meta.json` 保存局 ID、来源、开始时间、角色和游戏种子。`events.jsonl` 每行
保存一条按顺序编号的原始事件：

```json
{
  "sequence": 1,
  "observed_at": "2026-08-27T08:00:01Z",
  "type": "state",
  "payload": {"screen": "EVENT"}
}
```

录制层只保留 Mod 状态和动作，不生成模型提示词。转录器负责把原始轨迹转换
成决策记录，数据集构建器再生成 SFT 或 RL 输入。
