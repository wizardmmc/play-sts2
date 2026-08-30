# 人机协作录制链路

最后更新：2026-08-29

- 状态：A9 首盘在第 35 层知识恶魔 Boss 战败；404 条动作通过日志、raw 与 Harness
  三方验收，技术录制完整。
- 下一步：沿用 High 档继续下一难度录制；每局继续逐项对账 Solver 日志和 raw。

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

## 已确认完整：A4

目录：`data/raw/human_combat_solver/20260829-a4-f48-DKM6RXF43VH4`

故障机器人 A4 在 48 层击败 `TEST_SUBJECT_BOSS` 通关：

- 27 场战斗、564 条 Solver 战斗动作、262 条人工战略动作，共 826 条；
- 日志 431 次出牌和 9 次药水，与 raw 的 440 条部署动作身份及顺序逐项一致；
- 日志 120 次 `end_turn=true` 与 raw 的 120 条 `end_turn` 一致；
- 日志 4 次 `NATIVE_CHOICE_REQUEST/SELECTED` 与 raw 的 4 条 Solver 选牌身份及顺序
  一致，依次为 `THUNDER`、`MIND_ROT`、`SLOTH`、`DISINTEGRATION`；本盘四次均为
  ChooseCard，没有触发全息影像 CombatPile；
- 826 个 event ID 唯一，战斗/战略来源没有混淆；全部样本通过当前 Harness 合法动作
  与样式审计，能量空句、重复升级符号、英文枚举、零基幕号、本地化键、假已售卡和
  `res://` 路径计数均为零；
- 日志没有 capture gap、非零 deployment drift 或实际 unexpected replan；
- 最终元数据为 `victory=true`、`training_eligible=true`、
  `recording_complete=true`，`recording_gaps=[]`。

可读投影位于
`data/transcripts/human_combat_solver/20260829-a4-f48-DKM6RXF43VH4/`，包含 27 个
战斗文件和 1 个战略文件，564/262 条决策与 raw 完全一致。正式 Qwen3.5 tokenizer
下 P99 1,985、最大 2,064 token，没有样本超过 12,288 上限。

## A5 通关、一个 Hand 隐式选择缺口

目录：`data/raw/human_combat_solver/20260829-a5-f50-9JG499AMX7JS`

故障机器人 A5 在 50 层击败 `AEONGLASS_BOSS` 通关：

- 21 场战斗、717 条战斗样本、235 条战略样本，共 952 条；其中
  `combat_solver=714`、`human_ui=238`。13 层 `GAS_BOMB.EXPLODE_MOVE` 令 Solver
  连续 7 次在 `combat_root_snapshot` 初始化失败，用户手动执行 2 次出牌和 1 次药水；
  三条动作均以 `human_ui` 保存，后续战斗恢复 Solver，数据来源没有混淆；
- 日志 537 次部署动作与 raw 的 `532 play_card + 5 use_potion` 逐项一致，148 次
  `end_turn=true` 与 raw 逐项一致；952 个 event ID 全部唯一；
- 日志有 30 次 `NATIVE_CHOICE_SELECTED`，raw 只有 29 条 Solver
  `select_deck_card`。唯一缺口位于 `battle-f031-11` 第 1 回合：`SCAVENGE` 在 Hand
  仅有一个合格候选时隐式选择 `WHITE_NOISE+1@TAINTED:2`，日志显示
  `surface=Hand visible=False options=1`，但没有动作事件或 `capture_gap`；
- 缺失行没有完整动作前状态，故不从日志伪造。元数据已受控改为
  `training_eligible=true`、`recording_complete=false`，并在
  `integrity.recording_gaps` 记录该缺口；已有 952 条独立样本均能由 Harness 重建为
  合法动作；
- transcript 位于
  `data/transcripts/human_combat_solver/20260829-a5-f50-9JG499AMX7JS/`，包含 21 个
  战斗文件和 1 个战略文件。脏文本与隐藏字段计数均为零，system/user/assistant/
  `ACTION:` 都是 952 条；Qwen3.5 tokenizer P99 1,949、最大 2,047，没有样本超过
  12,288。

这盘也覆盖了两类动态遗物和特殊地图：

- 精致折扇的 system 说明明确写出“同一回合第 3 张攻击牌获得 4 格挡”；user 状态从
  `计数 0 → 1 → 2；已高亮` 推进。实际第三张攻击牌前为计数 2，打出后格挡从 22
  增至 26，和说明一致；
- 艳丽围巾明确写出“从手牌打出的第 5 张牌免费”；计数 4 时显示“已高亮”，候选
  手牌实时显示为 0 费，打出后计数清空。训练输入同时含触发规则与当前触发进度；
- 18 层事件选择了黄金罗盘，选项和遗物说明都写明“将第 2 阶段地图替换为特殊直道”。
  下一张 MAP 状态为 18 个节点、每行仅 `(row,3)`、每个节点只有一个后继；Harness
  全图忠实输出从古代节点到 Boss 的 17 段单链，而非误还原成普通分叉图。

缺口根因是上一次只包装了 `CardSelectCmd.FromCombatPile` 的异步结果。原版
`CardSelectCmd.FromHand` 具有相同的隐式分支：无需手动确认且候选数不大于最少选择
数时直接返回，不调用 `NPlayerHand.SelectCardInSimpleMode`，所以既绕过点击捕获，也
不会触发现有 gap。修复只包装这个隐式 Hand 分支；可见 Hand 路径保持原样，避免重复。

隔离 0.111.0 实机回归使用 `SCAVENGE + WITHER` 强制单候选 Hand 选择：修复前 Solver
日志有 `NATIVE_CHOICE_SELECTED`，事件流只有 `play_card`；修复后无人测试通过，事件
流新增且仅新增 1 条 `select_deck_card`，状态为 `CARD_SELECTION / combat_hand_select`、
候选 `[WITHER]`、索引 0、来源 `combat_solver`，并且没有 `capture_gap`。

## 已确认完整：A6 High

目录：`data/raw/human_combat_solver/20260829-a6-f48-11MPY1N5GNCF`

故障机器人 A6 在 48 层击败 `QUEEN_BOSS` 通关：

- 23 场战斗、571 条 Solver 战斗动作、230 条人工战略动作，共 801 条；801 个
  event ID 全部唯一，来源没有混淆，元数据为 `victory=true`、
  `training_eligible=true`、`recording_complete=true`、`recording_gaps=[]`；
- 日志 465 次部署动作与 raw 的 `460 play_card + 5 use_potion` 身份及顺序逐项一致，
  90 次 `end_turn=true` 与 raw 的 90 条结束回合逐项一致；
- 日志 16 次 `NATIVE_CHOICE_SELECTED` 与 raw 的 16 条 Solver
  `select_deck_card` 身份及顺序逐项一致，包括 13 次全息影像 CombatPile 和 3 次
  ChooseCard。42 层全息影像只有 `BEAM_CELL` 一个候选，日志为
  `surface=CombatPile visible=False options=1`，raw 正确保存同一候选、索引和来源；
- 日志没有搜索初始化失败、capture gap、人工战斗动作或重复动作。机甲骑士战第 5
  回合有一次 `DEPLOY_REPLAN`：计划中的适应打击在执行前变为能量不足，Solver 没有
  提交非法动作，而是从当时真实状态重新搜索并继续执行；重搜前后的全部实际动作仍与
  raw 对齐，因此这是一条有效的安全重搜，不是录制缺口；
- 游戏启动默认仍为 Medium；首战第一次自动搜索使用 5 秒/60 秒预算。用户在任何实际
  部署前切换到 High 并主动重算；此后 36 次搜索全部为 8 秒/120 秒、18/45 beam、
  8 GB no-GC 预算，实际部署没有使用 Medium 路线。Recorder 的
  `--solver-preset high` 与实际教师动作一致；
- transcript 位于
  `data/transcripts/human_combat_solver/20260829-a6-f48-11MPY1N5GNCF/`，包含 23 个
  战斗文件和 1 个战略文件。801 条参考动作全部位于对应完整合法动作域；已知脏文本、
  隐藏字段与资源路径计数均为零，system/user/assistant/`ACTION:` 都是 801 条；正式
  Qwen3.5 tokenizer P99 2,034、最大 2,204，没有样本超过 12,288。

## 已确认完整：A7 High

目录：`data/raw/human_combat_solver/20260829-a7-f48-KTFGDDRB868S`

故障机器人 A7 在 48 层击败 `TEST_SUBJECT_BOSS` 通关：

- 20 场战斗、583 条战斗样本、228 条战略样本，共 811 条；来源计数为
  `combat_solver=581`、`human_ui=230`。两条额外人工战斗动作分别是 8 层第 1 回合的
  敏捷药水和 33 层第 4 回合的易伤药水，均明确保存为 `human_ui`；
- 日志 462 次 Solver 部署与 raw 的 `459 play_card + 3 use_potion` 身份及顺序逐项
  一致；99 次 `end_turn=true` 与 raw 逐项一致；811 个 event ID 全部唯一；
- 日志 20 次 `NATIVE_CHOICE_SELECTED` 与 raw 的 20 条 Solver
  `select_deck_card` 身份及顺序逐项一致，其中 16 次为 `SCAVENGE` 的 Hand 选择，
  4 次为 ChooseCard；
- 28 层第 3 回合的 `SCAVENGE` 只有 `GREED` 一个合格候选。日志明确为
  `surface=Hand visible=False options=1`，raw 正确保存
  `CARD_SELECTION / combat_hand_select`、候选数 1、索引 0、`GREED` 和
  `combat_solver` 来源。这是 A5 缺口修复后第一次正式生产覆盖同一路径；
- 日志没有搜索初始化失败、capture gap、unexpected replan、deployment drift、
  continuation missing 或动作重复；本局 25 次搜索全部使用 High 的 8 秒/120 秒预算，
  与 recorder 元数据一致；
- 元数据为 `victory=true`、`training_eligible=true`、
  `recording_complete=true`、`recording_gaps=[]`；
- transcript 位于
  `data/transcripts/human_combat_solver/20260829-a7-f48-KTFGDDRB868S/`，包含 20 个
  战斗文件和 1 个战略文件。811 条参考动作全部位于对应完整合法动作域；已知脏文本、
  隐藏字段与资源路径计数均为零，system/user/assistant/`ACTION:` 都是 811 条；正式
  Qwen3.5 tokenizer P99 1,993、最大 2,056，没有样本超过 12,288。

## 已确认完整：A8 High 战败

目录：`data/raw/human_combat_solver/20260829-a8-f48-YL1PEYDY72PF`

故障机器人 A8 在 48 层 `AEONGLASS_BOSS` 第 5 回合战败；最后一条动作前为 5 HP、
11 格挡，Boss 仍有 409/535 HP：

- 23 场战斗、579 条 Solver 战斗动作、214 条人工战略动作，共 793 条；793 个
  event ID 全部唯一，来源没有混淆；
- 日志 446 次 Solver 部署与 raw 的 `441 play_card + 5 use_potion` 身份及顺序逐项
  一致；107 次 `end_turn=true` 与 raw 的 107 条结束回合逐项一致；
- 日志 26 次 `NATIVE_CHOICE_SELECTED` 与 raw 的 26 条 Solver
  `select_deck_card` 身份及顺序逐项一致，其中 17 次 ChooseCard、9 次全息影像
  CombatPile。46 层全息影像只有 `MOMENTUM_STRIKE` 一个候选，raw 正确保存候选数 1、
  索引和来源；
- 日志没有搜索初始化失败、capture gap、unexpected replan、deployment drift、
  continuation missing、人工战斗动作或动作重复；本局 31 次搜索全部使用 High 的
  8 秒/120 秒预算；
- 元数据正确保存 `victory=false`，同时保持 `training_eligible=true`、
  `recording_complete=true`、`recording_gaps=[]`。579 条战斗单步状态与标签可以直接
  保留；214 条战略样本属于真实战败路线，不能未经筛选当作战略正例；
- transcript 位于
  `data/transcripts/human_combat_solver/20260829-a8-f48-YL1PEYDY72PF/`，包含 23 个
  战斗文件和 1 个战略文件。793 条参考动作全部位于对应完整合法动作域；已知脏文本、
  隐藏字段与资源路径计数均为零，system/user/assistant/`ACTION:` 都是 793 条；正式
  Qwen3.5 tokenizer P99 1,850、最大 1,931，没有样本超过 12,288。

本盘也验证了微型帐篷的多选休息处。raw 中该遗物在 `act_id=2` 的 38 层商店购入；
Harness 使用玩家幕号，因此 transcript 显示为第 3 幕：

- 遗物上下文明确显示“你可以在休息处选择任意数量的选项”；
- 40 层完整序列为“锻造 → 选择压缩升级 → 仅剩休息且同时可 proceed → 休息 →
  proceed”；
- 47 层完整序列为“休息 → 仅剩锻造且同时可 proceed → 锻造 → 选择富足升级 →
  proceed”；
- 每一步都按 raw 当前剩余的真实选项重新编号并生成合法动作，没有假定休息处只能选择
  一次，也没有丢失中间的选牌状态。因此现有通用休息处投影已经适配，无需特殊代码。

## 已确认完整：A8 High 早夭重试

目录：`data/raw/human_combat_solver/20260829-a8-f7-X48CA7QL0QXL`

故障机器人第二盘 A8 在第 7 层 `PHROG_PARASITE_ELITE` 第 7 回合战败；最后一条动作
前为 8 HP、0 格挡，场上仍有 4 只 `WRIGGLER`：

- 5 场战斗、98 条 Solver 战斗动作、28 条人工战略动作，共 126 条；126 个 event ID
  全部唯一，来源没有混淆；
- 日志 74 次 Solver 部署与 raw 的 `71 play_card + 3 use_potion` 身份及顺序逐项
  一致；23 次 `end_turn=true` 与 raw 逐项一致；唯一一次 Hand 选牌也与 raw 的
  `STRIKE_DEFECT`、候选数 4 和来源一致；
- 日志没有搜索初始化失败、capture gap、unexpected replan、deployment drift、
  continuation missing、人工战斗动作或动作重复；20 次搜索全部使用 High。最终精英战
  从首轮即为 `only_death_routes=true`，用户多次主动重算只产生新搜索，没有重复动作；
- 元数据正确保存 `victory=false`、`training_eligible=true`、
  `recording_complete=true`、`recording_gaps=[]`。98 条战斗样本可以保留，28 条战略
  样本保留早期战败标签；
- transcript 位于
  `data/transcripts/human_combat_solver/20260829-a8-f7-X48CA7QL0QXL/`，包含 5 个战斗
  文件和 1 个战略文件。126 条参考动作全部位于对应完整合法动作域；已知脏文本、隐藏
  字段与资源路径计数均为零；正式 Qwen3.5 tokenizer P99 1,697、最大 1,791。

## 已确认完整：A8 High 第三盘战败

目录：`data/raw/human_combat_solver/20260829-a8-f33-6JKFJZRTVD4Q`

故障机器人第三盘 A8 在第 33 层 `KAISER_CRAB_BOSS` 第 12 回合战败；最后一条动作前
为 16 HP、0 格挡，`CRUSHER` 与 `ROCKET` 分别仍有 124/219、38/209 HP：

- 16 场战斗、444 条战斗样本、151 条战略样本，共 595 条；来源计数为
  `combat_solver=439`、`human_ui=156`。5 条额外人工战斗动作均为药水，分别在 15、
  22、27 和 33 层执行，来源标记正确；
- 日志 332 次 Solver 部署与 raw 的 332 条 Solver `play_card` 身份及顺序逐项一致；
  90 次 `end_turn=true` 与 raw 逐项一致；17 次 `NATIVE_CHOICE_SELECTED` 与 raw
  逐项一致，其中包含 13 次 CombatPile、4 次 Hand 选择；
- 3 次全息影像只有一个候选，分别选择 `DEFEND_DEFECT`、`DARKNESS`、
  `DUALCAST`，均正确保存候选数 1、索引和来源；
- 日志没有搜索初始化失败、capture gap、unexpected replan、deployment drift、
  continuation missing 或动作重复；30 次搜索全部使用 High。Boss 战从首轮即为
  `only_death_routes=true`，后续重搜和人工用药没有产生录制缺口；
- 元数据正确保存 `victory=false`、`training_eligible=true`、
  `recording_complete=true`、`recording_gaps=[]`。444 条战斗样本可保留；151 条战略
  样本保留中期 Boss 战败标签；
- transcript 位于
  `data/transcripts/human_combat_solver/20260829-a8-f33-6JKFJZRTVD4Q/`，包含 16 个
  战斗文件和 1 个战略文件。595 条参考动作全部位于对应完整合法动作域；已知脏文本、
  隐藏字段与资源路径计数均为零；正式 Qwen3.5 tokenizer P99 1,798、最大 1,921。

## 已确认完整：A8 High 第四盘通关

目录：`data/raw/human_combat_solver/20260829-a8-f48-WB3S4HHN9DCG`

故障机器人第四盘 A8 在 48 层击败 `AEONGLASS_BOSS`，随后进入
`THE_ARCHITECT` 终局事件并通关：

- 23 场战斗、435 条战斗样本、234 条战略样本，共 669 条；来源计数为
  `combat_solver=432`、`human_ui=237`。3 条额外人工战斗动作分别是 37 层的果汁和
  48 层首回合的集中、虚弱药水，均明确保存为 `human_ui`；
- 日志 331 次 Solver 部署与 raw 的 `324 play_card + 7 use_potion` 身份及顺序逐项
  一致；83 次 `end_turn=true` 与 raw 逐项一致；18 次原生选牌与 raw 逐项一致，
  包含 13 次 CombatPile、3 次 ChooseCard 和 2 次 Hand 选择；
- 4 次全息影像只有一个候选，分别选择 `COOLHEADED`、`DUALCAST`、
  `ULTIMATE_STRIKE`、`SQUASH`，均正确保存候选数 1、索引和来源；
- 日志没有搜索初始化失败、capture gap、unexpected replan、deployment drift、
  continuation missing 或动作重复；33 次搜索全部使用 High；
- 元数据正确保存 `victory=true`、`training_eligible=true`、
  `recording_complete=true`、`recording_gaps=[]`；
- transcript 位于
  `data/transcripts/human_combat_solver/20260829-a8-f48-WB3S4HHN9DCG/`，包含 23 个
  战斗文件和 1 个战略文件。669 条参考动作全部位于对应完整合法动作域；已知脏文本、
  隐藏字段与资源路径计数均为零；正式 Qwen3.5 tokenizer P99 2,084、最大 2,191。

## 已确认完整：A9 High 首盘战败

目录：`data/raw/human_combat_solver/20260829-a9-f35-QAFQ6J4JFUBB`

故障机器人首盘 A9 在第 35 层 `KNOWLEDGE_DEMON_BOSS` 第 3 回合战败；最后一条动作
前为 18 HP、0 格挡，Boss 仍有 270/399 HP：

- 14 场战斗、252 条战斗样本、152 条战略样本，共 404 条；来源计数为
  `combat_solver=250`、`human_ui=154`。两条额外人工战斗动作分别是 23 层的爆裂安瓿
  和 35 层首回合的集中药水，均明确保存为 `human_ui`；
- 日志 186 次 Solver 部署与 raw 的 `185 play_card + 1 use_potion` 身份及顺序逐项
  一致；57 次 `end_turn=true` 与 raw 逐项一致；7 次原生选牌与 raw 逐项一致，包含
  2 次 ChooseCard 和 5 次 Hand 选择；
- 日志没有搜索初始化失败、capture gap、unexpected replan、deployment drift、
  continuation missing 或动作重复；20 次搜索全部使用 High。知识恶魔战从首轮即为
  `only_death_routes=true`，人工集中药水和主动重算没有产生录制缺口；
- 元数据正确保存 `victory=false`、`training_eligible=true`、
  `recording_complete=true`、`recording_gaps=[]`。252 条战斗样本可以保留；152 条战略
  样本保留第二幕 Boss 战败标签；
- transcript 位于
  `data/transcripts/human_combat_solver/20260829-a9-f35-QAFQ6J4JFUBB/`，包含 14 个
  战斗文件和 1 个战略文件。404 条参考动作全部位于对应完整合法动作域；已知脏文本、
  隐藏字段与资源路径计数均为零；正式 Qwen3.5 tokenizer P99 1,646、最大 1,729。

## 已修复：Harness 训练前样式

初次审计两盘 A2 与完整 A3 的 2,250 条独立决策时，动作合法性、隐藏信息和长度已经
通过，但发现六类高频表现问题：能量图标被删为空句、升级牌重复 `+`、内部目标/不可用
枚举、零基幕号、事件本地化键和已售空槽假卡牌。

这些问题不是第一次出现。前辈项目 `human-rl` 在 2026-08-23～25 已经实现并测试：

- 连续能量/星能图标按数量转为“1点能量/2点能量”；
- 游戏名称已经带 `+` 时不再追加；
- 目标与不可用原因转换成玩家措辞；
- 零基 `act_id` 加一显示；
- 已售空槽不伪造零费未知卡牌。

Git 历史确认当前 `Play_sts2` 的 Harness 于 2026-08-27～28 重新实现时没有移植这些
规则，且旧测试反而固定了 `<AnyEnemy>`、`unplayable` 和“第0幕”。本次修复复用了
前辈已经实证的语义，但按当前 stateless Harness 和 pytest 结构重新实现：

- 新增共享文本清洗，保留连续图标数量，未知图标保持可发现；
- 战斗、选牌、商店与战略牌组统一去除重复升级符号；
- 只为真实需要目标的卡牌显示中文目标，三种已出现的不可用原因全部中文化；
- 战略幕号转成玩家的一基编号；
- `*.pages.*` 未解析事件键不再输出，已售空槽显示为“已售出”。

随后从 raw 覆盖重建 A0、A1、两盘 A2、A3 和 A4：共 6 局、148 个战斗文件、6 个
战略文件、4,874 条决策。能量空句、`++`、英文目标/不可用原因、零基幕号、本地化键、
假已售卡牌和 `res://` 路径计数均为零；system/user/assistant 与 `ACTION:` 数量均为
4,874。正式 Qwen3.5 tokenizer 的既有总集最大值仍为 2,085，
没有样本超过 8,192 或训练上限 12,288。

Transcript 每条决策重复 system 是刻意镜像实际 SFT 的独立三消息结构，不是训练
重复错误；`docs/data-format.md` 已同步更正。全部问题都通过 raw 重建解决，没有直接
手工修补 TXT、生成 JSONL 或重录游戏。

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
6. 若仍缺失，分别检查 `CombatPile/Hologram` 与 `Hand/SCAVENGE`，以及可见与隐式
   路径各自的比例。
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
