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

当前有三种原始事件：

- `state`：轮询发现变化后的完整 `/state` 数据。
- `mod_event`：`/events/stream` 发布的原始 SSE 对象，其中
  `action_executed` 包含动作前状态、动作请求和动作后状态。
- `recording_ended`：本次录制的结束原因。

录制层不推断动作，也不生成模型提示词。转录器负责从原始轨迹中选择
`source=human_ui` 的精确动作并转换成决策记录，数据集构建器再生成 SFT 或 RL
输入。角色选择和 `embark` 等环境初始化动作虽然保留在原始数据中，但不应成为
策略训练标签。

## 精确决策转录

每局转录结果位于 `data/transcripts/<run_id>.jsonl`。每行对应一条成功执行的
`human_ui` 动作：

```json
{
  "run_id": "RUN-ID",
  "source_sequence": 17,
  "event_id": 10009,
  "observed_at": "2026-08-27T02:05:13.625Z",
  "recorded_layer": "strategic",
  "before_state": {"screen": "EVENT"},
  "action": "choose_event_option",
  "parameters": {"option_index": 2}
}
```

`recorded_layer` 是 Mod 捕获的原始层提示。后续构建 SFT 数据时，Harness 的
屏幕归属逻辑仍需根据 `before_state` 重新判断并校验层级。转录结果本身不包含
系统提示、可读状态文本或模型消息。
