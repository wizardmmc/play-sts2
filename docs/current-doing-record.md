# 人机协作录制链路

最后更新：2026-08-29

- 状态：CombatPile 捕获与胜负持久化已经修复。修复后首盘正式 A3 已通关并通过
  日志、raw 与 Harness 三方验收，736 条样本连续完整。
- 下一步：沿用当前协议继续下一难度录制；E5 构建前先修 Harness 样式问题，再从 raw
  统一重建 transcript 和训练分卷。

## 当前目标

由人类负责路线、选牌、商店、营火等战略决策，CombatSolver 负责战斗动作；录制器把两类动作连同各自动作发生前的可见游戏状态保存下来，之后转录为 Harness 可读轨迹。

这条数据线必须满足以下约束：

- 训练侧只使用玩家当时可见的信息，不向 4B 暴露隐藏 RNG、搜索树或 Solver 评分。
- raw 的 `action_source=combat_solver` 只表示 CombatSolver 实际提交的战斗动作，
  `action_source=human_ui` 表示人类通过原生界面提交的动作。
- 已收到样本通过合法性与可见信息审计后即可进入 stateless SFT；日志与 raw 完全对账
  是声明整局 `recording_complete=true` 的额外条件。
- 不用日志推测并伪造缺失样本；无法恢复完整 `before_state` 的动作只能标为缺失。

## 已确认正常：A0

目录：`data/raw/human_combat_solver/20260829-a0-f48-UG7W38SMZZWJ`

- 故障机器人 A0 通关至 48 层。
- 共 24 场战斗、601 个 Solver 战斗动作、236 个人类战略动作。
- 原始录制、游戏日志与转录数量一致，样本动作均属于各自状态的合法动作。
- 9 次 `worse_recalculation` 都发生在动作提交前，不构成轨迹缺口。

## A1 发现的完整性缺口

目录：`data/raw/human_combat_solver/20260829-a1-f48-9X3DLG53FW2P`

A1 已通关，但不能按“完整教师轨迹”验收：游戏日志记录了 38 次战斗选牌提交，原始录制中只有 32 个 `select_deck_card`。缺少的 6 个动作全部来自全息影像的弃牌堆选择。

| 战斗 | 回合 | 实际选择 | 所属部署 | 界面状态 |
| --- | ---: | --- | --- | --- |
| `battle-f028-14` THE_OBSCURA_NORMAL | 6 | `ZAP+1` | `deployment:6:1:HOLOGRAM` | 隐式、不可见 |
| `battle-f030-16` INFESTED_PRISMS_ELITE | 6 | `FTL+0` | `deployment:6:1:Hologram` | 隐式、不可见 |
| `battle-f035-18` DEVOTED_SCULPTOR_WEAK | 1 | `FTL+1` | `deployment:1:4:Hologram` | 隐式、不可见 |
| `battle-f045-23` SCROLLS_OF_BITING_NORMAL | 1 | `FTL+1` | `deployment:1:2:Hologram` | 隐式、不可见 |
| `battle-f048-25` TEST_SUBJECT_BOSS | 11 | `FOCUSED_STRIKE+1` | `deployment:11:1:Hologram` | 隐式、不可见 |
| `battle-f048-25` TEST_SUBJECT_BOSS | 13 | `FTL+1` | `deployment:13:2:Hologram` | 可见，2 个候选 |

这些都不是被撤销的候选动作：每次选择后对应部署都继续执行，选中的卡随后实际被打出。其余已写入的 A1 样本本身均合法：721 个 Solver 动作与 240 个人类动作可以无损保留，但整盘轨迹不完整。

A1 的 `training_eligible=true` 没有问题：961 条已收到动作都有各自完整当前状态，
缺失动作不会让后续样本依赖一条不存在的历史。错误的是
`recording_complete=true`；现已改为 `false`，并把 6 次缺失写入
`integrity.recording_gaps`。A1 可以进入按局隔离的 stateless SFT，但不能冒充无缺口
连续轨迹。

## 根因定位

问题位于 Mod 的战斗选牌捕获边界，而不是转录器：缺失动作在 raw 中已经不存在，转录无法补回。

当前 `NativeUiActionRecorder` 主要依赖卡牌网格界面的点击和确认回调：

1. 点击时调用 `BeginSelectedCard`，从当前界面的候选卡中定位所选卡并暂存捕获结果。
2. `ConfirmSelection` 或 `CompleteSelection` 时提交暂存结果。
3. 界面退出时直接清除剩余暂存结果。

五次单候选全息影像选择确定走了 CombatSolver 的隐式提交路径。CombatSolver 在 `CardSelectCmd.FromCombatPile` 入口计算候选：当不要求手动确认且候选数不大于最少选择数时，`RequiresSurface=false`，只校验原版将全部候选隐式选中，不创建或操作卡牌网格。因此这五个动作一定会绕过当前依赖 UI 点击的 recorder。对应实现见 [NativeChoiceRuntime.cs](https://github.com/Torch1230/CombatSolver/blob/main/src/Runtime/NativeChoiceRuntime.cs)。

另一次可见选择缺失暴露出第二个问题：`QueueGridSelection` 已经拿到了实际的 `screen`，但 `BeginSelectedCard` 又从 `ActiveScreenContext.Instance.GetCurrentScreen()` 重新寻找候选。CombatSolver 自己则通过 `NOverlayStack.Instance.Peek()` 定位可见选牌页。现有证据不能区分“点击补丁没有进入”和“进入后 current screen 暂时不同”这两个分支；后者是代码中已经确认存在的静默失败窗口。A2 的五次可见全息影像全部成功，说明可见路径的问题是间歇性的，不是每次必现。

代码在 `BeginSelectedCard` 定位失败、没有进入提交补丁、退出时仍有暂存结果等情况下没有统一报告 capture gap，所以最终元数据仍错误地显示完整。

## 是否能无损恢复 A1

结论分两部分：

- 已录到的 961 个动作可以原样、无损保留，当前转录也可以作为人工复核材料。
- 缺失的 6 个动作不能仅凭现有 raw 和日志无损重建为训练样本。日志能证明所选卡牌、回合、部署归属以及部分界面信息，但没有保存动作发生前完整的 Harness 状态、事件标识、观测时间和所有候选项。尤其可见的二选一缺口，不能安全地从结果反推完整输入。

因此不应把日志推导出的 6 行混入 raw。除非另有同一时刻的完整状态快照或可精确复现的存档，否则只能保留 A1 的有效样本并将整盘标记为“不完整、不可作为连续教师轨迹”。

A0、A1、A2 的胜负本身可以从各自游戏日志中的明确 `WON/LOST` 记录无损回填；这与无法恢复 A1 缺失的六个动作前状态是两件事。

## 已确认正常：A2

目录：`data/raw/human_combat_solver/20260829-a2-f48-K6RFGM1UZF91`

- 故障机器人 A2 到达 48 层，在第三幕 Boss `AEONGLASS` 第 5 回合死亡。
- 共 24 场战斗、494 个 Solver 战斗动作、249 个人类战略动作。
- 游戏日志有 400 次部署动作、89 次结束回合和 5 次战斗选牌，分别精确对应 raw 的 `390 play_card + 10 use_potion`、`89 end_turn` 和 `5 select_deck_card`。
- 24 场战斗逐动作比对后，动作类型、卡牌或药水身份、选择结果和先后顺序全部一致。
- 5 次全息影像均为可见的 `CombatPile` 选择，候选数分别为 4、14、5、8、2，全部被正确录制。A2 没有发生单候选隐式选择，因此不能推翻 A1 的隐式路径缺陷。
- 743 个 event ID 全部唯一；所有样本均能由当前 Harness 重建，并且生成的动作属于对应状态的合法动作。
- Harness 文本未出现隐藏 RNG、搜索树、Solver 评分、内部事件 ID 或完整抽牌序列。
- A2 的技术录制完整。494 个 CombatSolver 战斗动作可以进入战斗教师 SFT；249 个人类战略动作可以保留为真实行为，但由于整局战败，不能未经筛选就视为战略正例。

## 新增 A2 通关：DDHQ5TB3UCY7

目录：`data/raw/human_combat_solver/20260829-a2-f48-DDHQ5TB3UCY7`

这盘由修复前已经启动的旧 Mod 录制，故障机器人 A2 在 48 层击败 `AEONGLASS`，
以 9 HP 进入 Boss 奖励和 `THE_ARCHITECT` 终局事件。用户确认通关；raw 的终局路径
也与胜利一致。

- 24 场战斗、546 条 Solver 战斗动作、225 条人工战略动作，共 771 条；
- 771 个 event ID 全部唯一，meta 计数、物理行数和动作来源计数一致；
- 战斗分片全部为 `combat_solver`，战略分片全部为 `human_ui`；
- 771 条样本全部能由当前 Harness 重建，参考动作都位于对应合法动作域；
- 训练消息未出现 `draw_cards`、结构化牌堆内部字段、state revision、event ID、
  CombatSolver 身份、RNG 状态或搜索树字段；
- 13 次“打出全息影像且弃牌堆非空”中，10 次多候选选择均紧随一条候选数完全一致的
  `select_deck_card`，3 次单候选隐式选择缺失：

| 战斗 | 回合 | 唯一弃牌 | 全息影像 event ID | 后续动作 |
| --- | ---: | --- | ---: | --- |
| `battle-f019-11` | 1 | `COOLHEADED` | 18985 | 立即打出 `COOLHEADED` |
| `battle-f035-19` | 1 | `ULTIMATE_STRIKE` | 21527 | 立即打出 `ULTIMATE_STRIKE` |
| `battle-f038-20` | 1 | `BEAM_CELL` | 21727 | 立即打出 `BEAM_CELL` |

因此该局的 `training_eligible=true` 正确，771 条已有样本都可以进入 stateless SFT；
`recording_complete=true` 则不准确，应在受控元数据修正时改为 `false` 并记录上述
3 个 gap。由于旧游戏退出时启动器删除了临时 headed 日志，本盘不能再补做
`DEPLOY_ACTION`、`DEPLOY_END` 与 `NATIVE_CHOICE_SELECTED` 的日志级全量对账；不能
据此声称除 raw 可证明的三处以外绝对没有其他未报告动作。

## 已确认完整：A3

目录：`data/raw/human_combat_solver/20260829-a3-f48-PHRNGCP5S2QS`

故障机器人 A3 在 48 层击败 `QUEEN_BOSS` 通关。该局是 CombatPile 修复后第一盘
正式完整轨迹：

- 24 场战斗、493 条 Solver 战斗动作、243 条人工战略动作，共 736 条；
- 日志中的 367 次出牌和 3 次药水，与 raw 的 370 条部署动作身份及顺序逐项一致；
- 日志 99 次 `end_turn=true` 与 raw 的 99 条 `end_turn` 一致；
- 日志 24 次 `NATIVE_CHOICE_SELECTED` 与 raw 的 24 条 Solver
  `select_deck_card` 身份及顺序逐项一致，其中 13 次为全息影像 CombatPile、11 次为
  `SCAVENGE` 手牌选择；
- 13 次非空弃牌堆全息影像全部录到并保存相同候选数；其中 2 次是单候选隐式路径，
  均保存为 `CARD_SELECTION`、候选数 1、来源 `combat_solver`；
- 736 个 event ID 唯一，战斗/战略来源没有混淆；全部样本通过当前 Harness 合法动作
  校验，训练消息没有内部牌堆结构、revision、event ID、Solver、RNG 或搜索树字段；
- 日志没有 capture gap、非零 deployment drift 或实际 unexpected replan；
- 最终元数据为 `victory=true`、`training_eligible=true`、
  `recording_complete=true`，`recording_gaps=[]`。

可读投影位于
`data/transcripts/human_combat_solver/20260829-a3-f48-PHRNGCP5S2QS/`，包含 24 个
战斗文件和 1 个战略文件，493/243 条决策与 raw 完全一致。

## Harness 训练前样式审计

审计范围为两盘 A2 与完整 A3，共 75 个 transcript 文件、2,250 条独立决策。

已经通过的契约：

- 2,250 条决策都有一组 system/user/assistant 和恰好一行合法 `ACTION:`；
- 训练消息没有 `draw_cards`、`discard_cards`、`exhaust_cards`、revision、event ID、
  CombatSolver、RNG 状态或搜索树字段，也没有残留 `res://` 路径；
- 正式 Qwen3.5 tokenizer 下长度为 470～2,076 token，P99 为 1,873，没有样本超过
  8,192 或训练配置的 12,288 token 上限；
- 战斗、战略、地图、商店、奖励和选牌页面的动作索引与 raw 合法域一致；A3 的隐式
  与可见 CombatPile 选择都呈现完整战斗状态和明确候选。

E5 构建前必须修正的表现层问题：

1. 能量图标资源路径被 `_clean_text` 直接删除，出现 1,859 行“获得。”、“在下个回合
   获得。”等缺失语义，影响卡牌、药水和遗物；raw 仍保留 18,180 个
   `defect_energy_icon.png`，可以按连续图标数重建能量值。
2. Mod 的升级牌名称已经带 `+`，Harness 又根据 `upgraded=true` 追加一次，产生
   4,272 行 `++`。
3. 卡牌目标和不可用原因仍暴露内部英文枚举：2,804 个 `<AnyEnemy>`、
   `<AllEnemies>`、`<RandomEnemy>` 或 `<None>`，以及 1,226 个
   `not_enough_energy`、`unplayable`、`blocked_by_hook`。
4. `act_id` 是零基索引，但战略文本直接写成“第0幕/第1幕/第2幕”；717 条战略决策
   都应转成人类口径的第一至第三幕。
5. Neow、古代人等事件有 18 行 `*.pages.*.description` 本地化键；旧 raw 无法恢复
   文本时至少应隐藏键，新 Mod 应优先输出已解析描述。
6. 商店已售出的空槽出现 17 行“未知卡牌(0费) | 0金币 | 已售出”，应简化为明确的
   已售出槽位。

Transcript 中每条决策重复 system 是刻意镜像实际 SFT 的独立三消息结构，不是训练
重复错误；`docs/data-format.md` 里“每个 TXT 只在开头展示一次 system”的旧描述需要
随修复一起更新。

以上问题不要求重录。先继续保存 raw；修正 Harness/Mod 后覆盖重建 transcript，再
构建 E5，禁止直接编辑当前 TXT 或生成后的 SFT JSONL。

## 已修复：胜负标签持久化

A2 原始 `meta.json` 只有 `termination_reason=game_over`，没有 `victory` 或
`run_outcome`。A0 通关和 A2 战败在旧 recorder 中都会被写成同一种 `game_over`。

Mod 的 `run_ended` 事件实际上已经携带 `victory`，但 Python recorder 的 `_termination_from_event` 只取出 `reason`，`RawRunWriter.finalize` 也只保存终止原因和时间。因此问题不是游戏端不知道胜负，而是 recorder 在落盘时丢掉了已有字段。

这不影响当前单步动作 SFT，却会直接影响整局筛选、战略样本质量判断以及后续
Tree-GRPO/GiGPO reward。Recorder 现已保存 `victory: true/false/null`；旧的
A0、A1、A2 已根据仍保留的明确终局日志受控回填为 `true/true/false`。

## 后续每盘的验收清单

录制进程自然结束后执行以下只读核验：

1. 确认临时目录已原子发布，并检查结束原因、楼层、角色、难度和种子。
2. 对账游戏日志中的 `DEPLOY_ACTION` 与 raw 的 `play_card + use_potion`。
3. 对账 `DEPLOY_END` 与 raw 的 `end_turn`。
4. 对账 `NATIVE_CHOICE_SELECTED` 与 raw 的 `select_deck_card`，按战斗、回合和部署顺序定位每个差异。
5. 检查 `capture_gap`、意外 replan、动作重复、非法动作、来源混淆和转录禁用字段。
6. 若仍缺失，判断是否继续集中在 `CombatPile/Hologram`，以及可见与隐式路径各自的比例。
7. 核对 raw 中持久化的胜负与日志终局一致。
8. 已收到样本合法即可转录和训练；只有三方数量和顺序一致后，才设置
   `recording_complete=true`。

## 修复实现

修复围绕 `CardSelectCmd.FromCombatPile` 的真实返回结果建立边界，而不是只监听可见
UI 点击：

- 入口保存过滤后的候选、选择约束、生命周期和触发本次选择的动作来源；包装异步
  返回值，在调用方应用结果之前，以原版实际返回卡牌作为最终标签；
- 可见路径在 `QueueGridSelection` 已收到的真实 screen 上保存候选顺序和点击时状态，
  不再二次读取可能滞后的 `ActiveScreenContext`；隐式路径在原版返回、调用方尚未移动
  卡牌时构造同构的 `CARD_SELECTION` 状态；
- CombatPile 的 UI 提交不再独立发布动作，由底层结果统一发布，避免可见选择重复；
- 来源优先采用可见点击调用栈，否则继承最近一次同生命周期的已提交卡牌或药水来源；
- 无法构造完整 `before_state` 时发布 `capture_gap`。gap 关闭连续录制完整性，但不
  否定其他已经验证的独立样本；
- `run_ended.victory` 已传给 writer；新 recorder 只有收到布尔胜负且没有 gap 时，
  才把完整 `game_over` 局标记为完整。

以上修复已经实现：Agent 包装原版 `CardSelectCmd.FromCombatPile` 的异步返回值，以
原版实际返回卡牌作为标签；可见网格保留点击时的真实候选顺序，隐式路径构造同构的
`CARD_SELECTION` 状态，触发动作来源从已提交卡牌或药水传播到异步选择上下文。
现有 CombatPile UI 提交不再独立发布，避免与底层结果重复。

0.111.0 隔离 headless 回归结果：

- 单候选弃牌堆：Solver `visible=0, selected=1`，raw 为
  `play_card → select_deck_card → play_card → end_turn`，选牌候选数 1；
- 双候选弃牌堆：Solver `visible=1, selected=1`，raw 同样恰好 1 条选牌动作，候选数
  2，没有重复 event ID；
- 两次选择均为 `action_source=combat_solver`，`before_state.screen=CARD_SELECTION`，
  无 `recording_gaps`。

恢复正式录制的实现门槛已经通过。任何构造失败都会产生
`native_ui_capture_gap`、关闭 `recording_complete`，但保留其他合法样本的训练资格；
完整新 `game_over` 局的 `meta.json.victory` 必须为布尔值。A1 缺失的 6 条仍不回写或
伪造，已有 961 条样本直接保留。
