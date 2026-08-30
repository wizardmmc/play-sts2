# CombatSolver 教师数据主线

- 状态：前三阶段与首轮正式 DAgger 已完成。A0～A9 十四盘教师 raw 已通过可见
  信息、动作来源和 Harness 审计，并分别进入 E5/E6 的整局隔离分卷；E6 `5e-5`
  仍是第一代正式 RL 基座。DAgger-R1 的 `1e-6`、`3e-6` 与追加 `1e-5` 候选均未在
  固定 A0 整局和未训练高温战斗回归上取得稳定净收益，因此全部不晋级。
- 最后更新：2026-08-30
- 当前目标：保留 E6 为冻结父 policy，使用 DAgger-R1 已落盘的 8 个 K=8 同入口
  group 进入小预算战斗 GRPO；DAgger 标签可作为诊断或受控辅助 loss，不把本轮
  未晋级 adapter 继续滚入下一批 on-policy 数据。

## 目标

用 CombatSolver 产生高质量战斗示范，先补强 4B 的战斗行为监督，再通过真实游戏
环境中的强化学习提高未见局面的稳健性。最终部署仍然只有一个 4B 模型，不依赖
CombatSolver。

本项目采用公平主线：4B 只能使用普通玩家在当前界面能够获得的信息。不得把精确
抽牌顺序、游戏 RNG 状态、怪物未来 RNG、CombatSolver 搜索树、节点评分或未来
路线直接放进模型输入。

CombatSolver 本身会读取精确牌序和怪物 RNG 进行搜索，因此它是有特权信息的教师，
不是部署时的策略。教师可以决定实际执行动作，但训练样本的 `user` 内容必须始终由
现有 Harness 的可见信息投影重新生成。相同可见状态可能因隐藏状态不同得到不同教师
动作，这是部分可观测问题，不应通过向 4B 泄露隐藏状态来消除。

## 当前状态

截至 2026-08-29，第一阶段的环境与知识准备已经完成：

- `.runtime/SlayTheSpire2-v0.107.1/` 与独立
  `.runtime/SlayTheSpire2-v0.111.0/` 并存；本机 Steam 已恢复正式 `public`
  分支的 `v0.107.1`。
- 教师运行时使用官方 RitsuLib `0.5.18` 与 CombatSolver `0.17.0`，实际加载了
  RitsuLib 的 `0.111.0` 兼容变体。
- `play-sts2-game` 接受两个精确白名单：仅 Agent 的基线 profile，以及
  Agent + RitsuLib + CombatSolver 的教师 profile；两者都继续拒绝额外启用的 Mod。
- STS2AIAgent 已针对 `0.111.0` 重新构建，同时仍能针对 `0.107.1` 零警告构建。
- 三 Mod headed 与 headless 实测均通过 `/health`、`/state`；A0 故障机器人首场
  战斗中 Solver 完成搜索、全自动执行第 2、3 回合并进入奖励页。
- `v0.111.0` 知识链已经从 Mod raw 独立 rebuild，生成 1,552 个规范实体、
  5,969 个事实和 41,783 条七问法候选；版本差异见
  `data/game_knowledge/reports/v0.111.0-diff.md`。
- STS2AIAgent 现在会在 CombatSolver 实际提交卡牌、药水、结束回合和战斗内选择时
  记录动作前状态，并以 `combat_solver` 发布；人工原生 UI 仍标为 `human_ui`，HTTP
  Agent 动作继续被 suppression 排除。`play-sts2-record --source
  human_combat_solver` 可以把两种来源写入同一局并对账。
- 正式局已经录到 A2：A0 通关且完整；A1 通关，但 Solver 日志中的 38 次战斗选牌
  只有 32 次进入 raw，缺少的 6 次都是全息影像的弃牌堆选择；A2 在第三幕 Boss
  战败，但 5 次同类可见选牌全部进入 raw，整局技术录制完整。逐局证据和修复边界
  见 `docs/current-doing-record.md`。
- A1 的 `training_eligible=true` 是正确的：每条 SFT 样本都包含独立完整的当前状态，
  缺少 6 个标签不会污染其余 961 条动作。错误的是 `recording_complete=true`；现已
  改为 `false` 并记录 gap，A1 可以按整局隔离进入 stateless SFT，但不能声称是无缺口
  连续轨迹，也不能根据日志补写缺少完整动作前状态的 6 行。
- Recorder 已把动作缺口和样本非法分开：gap 只关闭 `recording_complete`，已经收到
  且通过 Harness 的样本继续保持 `training_eligible=true`。
- `run_ended.victory` 已写入新 raw；A0/A1/A2 也已依据同一 headed 日志受控回填为
  `true/true/false`。
- 修复前旧进程中的新增 A2 位于
  `data/raw/human_combat_solver/20260829-a2-f48-DDHQ5TB3UCY7/`。该局通关，raw
  包含 24 场战斗、546 条 Solver 动作和 225 条人工动作；771 个 event ID 唯一，
  全部样本通过当前 Harness 合法性与可见信息审计。13 次非空弃牌堆全息影像中，
  10 次多候选选择完整，3 次单候选隐式选择缺失，因此样本可训练但不能声明连续
  录制完整。退出旧游戏时临时 headed 日志已删除，无法再做 Solver 日志逐动作对账。
- 修复后首盘 A3 位于
  `data/raw/human_combat_solver/20260829-a3-f48-PHRNGCP5S2QS/`，击败
  `QUEEN_BOSS` 通关。24 场战斗共 493 条 Solver 动作，人工战略 243 条；日志中的
  367 次出牌、3 次药水、99 次结束回合和 24 次原生选牌与 raw 数量、身份和顺序
  逐项一致。13 次全息影像弃牌堆选择全部录到，其中 2 次是修复目标的单候选隐式
  路径；另有 11 次手牌选择也全部对齐。736 条样本通过 Harness 与隐藏信息审计，
  没有 capture gap、部署漂移或实际 unexpected replan，最终
  `training_eligible=true`、`recording_complete=true`、`victory=true`。
- 完整 A4 位于 `data/raw/human_combat_solver/20260829-a4-f48-DKM6RXF43VH4/`，
  击败 `TEST_SUBJECT_BOSS` 通关。27 场战斗共 564 条 Solver 动作，人工战略 262 条；
  日志 431 次出牌、9 次药水、120 次结束回合和 4 次原生 ChooseCard 与 raw 身份、
  数量和顺序逐项一致。826 条样本 event ID 唯一、来源正确、Harness 合法且没有脏
  文本；日志没有 gap、部署漂移或实际 unexpected replan，最终三个准入/胜负字段
  全为正确布尔值。
- A5 位于 `data/raw/human_combat_solver/20260829-a5-f50-9JG499AMX7JS/`，在 50 层
  击败 `AEONGLASS_BOSS` 通关。21 场战斗共 717 条战斗样本，人工战略 235 条；
  Solver 的 537 次部署动作和 148 次结束回合与 raw 逐项一致。13 层
  `GAS_BOMB.EXPLODE_MOVE` 搜索初始化失败后，3 次人工战斗动作被正确标为
  `human_ui`。唯一缺口是 31 层 `SCAVENGE` 对单个 `WHITE_NOISE+1` 的隐式 Hand
  选择：日志 30 次选牌、raw 29 次。已有 952 条样本合法且可训练，元数据按事实标为
  `recording_complete=false`；对应 `CardSelectCmd.FromHand` 隐式分支已修复并通过
  隔离实机红绿回归。
- A5 同时验证精致折扇、艳丽围巾的触发规则、UI 计数与高亮都进入训练消息；黄金罗盘
  选项及遗物说明进入战略上下文，第二幕 18 个节点的单链地图也被 Harness 原样输出。
  A5 transcript 为 952 条，Qwen3.5 tokenizer P99 1,949、最大 2,047；全量现有数据的
  最大值仍为 2,085。
- A6 位于 `data/raw/human_combat_solver/20260829-a6-f48-11MPY1N5GNCF/`，在 48 层
  击败 `QUEEN_BOSS` 通关。23 场战斗共 571 条 Solver 动作，人工战略 230 条；日志
  465 次部署、90 次结束回合和 16 次战斗选牌与 raw 逐项一致。13 次全息影像包括
  一次修复目标的单候选隐式 CombatPile，完整保存且无 gap；另有 3 次 ChooseCard。
  首战在实际部署前从 Medium 重算为 High，之后所有搜索均使用 8 秒/120 秒和 8 GB
  预算。801 条 transcript 全部合法且无脏文本，tokenizer P99 2,034、最大 2,204。
  机甲骑士战中一张适应打击临场能量不足触发安全重搜，非法动作未被提交，日志与 raw
  仍完全对齐，不影响录制完整性。
- A7 位于 `data/raw/human_combat_solver/20260829-a7-f48-KTFGDDRB868S/`，在 48 层
  击败 `TEST_SUBJECT_BOSS` 通关。20 场战斗共 583 条战斗样本，人工战略 228 条；
  两次人工喝药正确标为 `human_ui`。Solver 的 462 次部署、99 次结束回合和 20 次
  战斗选牌与 raw 逐项一致；16 次 Hand 选择包含一次 `SCAVENGE` 单候选隐式选择，
  正确保存 `GREED` 的完整 `combat_hand_select` 状态，证明 A5 修复在正式局生效。
  本局 25 次搜索全部为 High，且无 gap、异常重搜或部署漂移。811 条 transcript 全部
  合法且无脏文本，tokenizer P99 1,993、最大 2,056。
- A8 位于 `data/raw/human_combat_solver/20260829-a8-f48-YL1PEYDY72PF/`，在 48 层
  `AEONGLASS_BOSS` 第 5 回合战败。23 场战斗共 579 条 Solver 动作，人工战略 214 条；
  446 次部署、107 次结束回合和 26 次战斗选牌与 raw 逐项一致，31 次搜索全部为 High，
  且无 gap、异常重搜或部署漂移。793 条 transcript 全部合法且无脏文本，tokenizer
  P99 1,850、最大 1,931。战斗样本可保留，战略样本按真实战败标签筛选。
- A8 还验证了微型帐篷“在休息处选择任意数量选项”的分支：raw 的 `act_id=2` 对应
  Harness 第 3 幕，40 层和 47 层均正确保留连续的休息/锻造、升级选牌、剩余选项与
  `proceed`。当前通用 REST 投影已经适配，不需要遗物特判。
- 第二盘 A8 位于 `data/raw/human_combat_solver/20260829-a8-f7-X48CA7QL0QXL/`，在
  第 7 层精英战第 7 回合战败。5 场战斗共 98 条 Solver 动作，人工战略 28 条；日志
  74 次部署、23 次结束回合和 1 次 Hand 选牌与 raw 逐项一致，20 次搜索全部为 High，
  且无 gap、异常重搜或部署漂移。126 条 transcript 全部合法且无脏文本，tokenizer
  P99 1,697、最大 1,791；战斗样本可保留，战略样本保留早期战败标签。
- 第三盘 A8 位于 `data/raw/human_combat_solver/20260829-a8-f33-6JKFJZRTVD4Q/`，在
  第 33 层帝皇蟹 Boss 第 12 回合战败。16 场战斗共 444 条战斗样本，人工战略 151 条；
  5 次人工喝药正确标为 `human_ui`。332 次 Solver 部署、90 次结束回合和 17 次战斗
  选牌与 raw 逐项一致，30 次搜索全部为 High，且无 gap、异常重搜或部署漂移。595 条
  transcript 全部合法且无脏文本，tokenizer P99 1,798、最大 1,921；战斗样本可保留，
  战略样本保留中期 Boss 战败标签。
- 第四盘 A8 位于 `data/raw/human_combat_solver/20260829-a8-f48-WB3S4HHN9DCG/`，在
  48 层击败永世沙漏通关。23 场战斗共 435 条战斗样本，人工战略 234 条；3 次人工
  喝药正确标为 `human_ui`。331 次 Solver 部署、83 次结束回合和 18 次战斗选牌与
  raw 逐项一致，33 次搜索全部为 High，且无 gap、异常重搜或部署漂移。669 条
  transcript 全部合法且无脏文本，tokenizer P99 2,084、最大 2,191。
- 首盘 A9 位于 `data/raw/human_combat_solver/20260829-a9-f35-QAFQ6J4JFUBB/`，在
  第 35 层知识恶魔 Boss 第 3 回合战败。14 场战斗共 252 条战斗样本，人工战略 152 条；
  两次人工喝药正确标为 `human_ui`。186 次 Solver 部署、57 次结束回合和 7 次战斗
  选牌与 raw 逐项一致，20 次搜索全部为 High，且无 gap、异常重搜或部署漂移。404 条
  transcript 全部合法且无脏文本，tokenizer P99 1,646、最大 1,729；战斗样本可保留，
  战略样本保留第二幕 Boss 战败标签。
- 两盘 A2 和 A3 的 2,250 条 transcript 训练前样式审计曾发现：1,859 行能量语义
  被资源清洗删除、4,272 行升级牌重复 `+`、2,804 个英文目标枚举、1,226 个英文
  不可用原因、717 个零基幕编号、18 个事件本地化键和 17 个假“未知卡牌”已售槽。
  对照前辈 `human-rl` 在 2026-08-23～25 已验证的渲染与测试后，确认这是当前 Harness
  2026-08-27 重写时的迁移遗漏；现已补齐共享图标数量化、卡名去重、玩家措辞、幕号、
  占位键隐藏与已售槽渲染。十四盘 raw 共 10,025 条 transcript 已全部重建，以上脏文本
  计数均为零；正式 Qwen3.5 tokenizer 最大 2,085 token、P99 1,841，仍远低于
  12,288 上限。

不带教师参数运行：

```bash
uv run play-sts2-game --port 8082 --mode headed
```

仍只会启动 `0.107.1 + STS2AIAgent`。教师环境必须显式传入 `v0.111.0` app 与
`combat-solver-profile`。教师环境已经能稳定运行，但每局仍须用 Solver 日志、raw
与 Harness 三方验收。当前磁盘上的教师 Mod 已更新；已经运行中的游戏不会热加载，
必须重启后才使用新捕获逻辑。

## 运行时安装边界

官方来源：

- CombatSolver Workshop：`3790899961`
- RitsuLib Workshop：`3747602295`
- CombatSolver 源码说明：<https://github.com/Torch1230/CombatSolver>
- RitsuLib 发布页：<https://github.com/BAKAOLC/STS2-RitsuLib/releases>

保留现有 `.runtime/SlayTheSpire2-v0.107.1/`，它仍然服务当前知识、SFT 和 E2E
基线，不能为了教师 Mod 就地升级。

教师运行时应独立放置：

```text
.runtime/SlayTheSpire2-v0.111.0/
├── runtime-receipt.json
└── SlayTheSpire2.app/
    └── Contents/MacOS/mods/
        ├── STS2AIAgent/
        ├── STS2-RitsuLib/
        └── CombatSolver/
```

只有以下条件全部成立才把安装标记为完成：

1. 从用户拥有的 Steam `public-beta` 分支取得 `0.111.0` macOS 游戏文件，并复制到
   独立运行时；不得覆盖 `v0.107.1`。
2. 从官方发布渠道取得 RitsuLib 和 CombatSolver 二进制，不把 GitHub 的公开源码
   当作可自由修改、再发布的许可证。
3. 为 `0.111.0` 重新构建 STS2AIAgent，不能复用针对 `0.107.1` 编译的 DLL。
4. 使用独立教师 profile，只启用 `STS2AIAgent`、`STS2-RitsuLib` 和
   `CombatSolver`，继续关闭 `UnifiedSavePath` 与 Steam 云存档。
5. headed 冒烟日志明确显示三个 Mod 都初始化成功，Agent `/health` 与 `/state`
   可用，CombatSolver 能在单人战斗中搜索并执行一回合。

## 原始数据目录

人机协作教师数据写入：

```text
data/raw/human_combat_solver/
└── YYYYMMDD-aN-fN-SEED/
    ├── meta.json
    ├── combat/
    │   └── battle-fNNN-NN.jsonl
    └── strategy/
        └── decisions.jsonl
```

第一盘先固定为 A0 故障机器人。CombatSolver 只负责战斗；卡牌奖励、路线、商店、
事件和火堆仍由人操作，因此同一局要在元数据和每条动作来源中区分
`combat_solver` 与 `human_ui`，不能把整局笼统标成纯教师轨迹。

`meta.json` 至少要补充以下来源信息：

- 游戏版本与分支；
- CombatSolver、RitsuLib、STS2AIAgent 版本；
- Solver 搜索档位及时间、节点和内存预算；
- 角色、进阶、seed、最终楼层和终局原因；
- 每种动作来源的数量；
- 是否完整录制、是否出现重搜、部署漂移、未支持效果或采集缺口；
- `student_observation_policy: visible_only`；
- `teacher_uses_hidden_rng: true`。

当前 Agent 不读取 CombatSolver 私有设置对象；`solver.settings_source` 固定为
`recorder_argument`，表示档位和预算由录制命令显式声明。正式录制时
`--solver-preset` 必须与游戏内当前档位一致，人工 review 同时核对 Solver 日志。

raw 可以保留审计与重建所需的完整实机状态，但训练构建器必须只调用 Harness 产生
可见观测。发布数据前要断言训练文本中不含 `draw_cards`、精确抽牌顺序、RNG 状态、
Solver 路线或评分。普通玩家能查看的剩余牌堆组成、弃牌堆、消耗牌堆、当前意图、
手牌、费用和遗物计数仍然属于可见信息。

## 录制器需要补齐的契约

不能通过轮询前后状态猜动作。STS2AIAgent 已在自己的观察层捕获 CombatSolver
实际提交给游戏的动作，并发布与现有 `action_executed` 等价的事件：

- 卡牌：卡牌实例、提交前手牌索引和目标；
- 药水：提交前槽位和目标；
- 结束回合；
- 战斗内选择牌及确认；
- CombatSolver 中止、重搜、部署漂移和未支持边界。

动作事件应标记 `client_context.source = "combat_solver"`。HTTP Agent 执行动作仍由
现有 suppression 排除，人工 UI 动作继续标成 `human_ui`，避免一条动作被重复记录。
Recorder 将整局写入 `data/raw/human_combat_solver/`，并按动作来源区分人工战略动作
与 Solver 战斗动作。

这部分优先在 STS2AIAgent 的通用“外部自动执行观察”层实现，不修改 CombatSolver
源码。这样既不依赖其内部 API，也不需要复制其实现。

## 第一盘 A0 故障机器人验收

完成运行时和捕获适配后，操作顺序应为：

```bash
# 终端 1：教师运行时。最终参数名以实现后的 --help 为准。
uv run play-sts2-game \
  --app-path .runtime/SlayTheSpire2-v0.111.0/SlayTheSpire2.app \
  --profile e2e/fixtures/combat-solver-profile \
  --port 8082 \
  --mode headed

# 终端 2：必须在点击“出发”前启动。
uv run play-sts2-record \
  --base-url http://127.0.0.1:8082 \
  --source human_combat_solver \
  --output-root data/raw
```

上面的命令已经可用。自动化实现与冒烟全部使用 headless，避免占用桌面；正式混合
轨迹包含人的战略输入，因此由用户自行启动 headed 游戏并操作。第一局需满足：

- A0、故障机器人，从 `run_started` 开始录到 `run_ended`；
- 至少一场战斗由 CombatSolver “全自动”完整接管；
- 卡牌、药水、结束回合和战斗内选牌的实际动作数与 Solver 部署日志一致；
- 人工战略动作被记录且来源正确；
- 每个样本都能用可见 Harness 观测重新生成合法 `ACTION:`；
- 已收到样本通过 Harness 校验时设置 `training_eligible=true`；只有没有动作缺口时
  才设置 `recording_complete=true`；
- 先人工 review 一整局，再逐级录制并逐局验收；任何提交动作与 raw 不一致都先暂停
  后续难度录制。

## 实施顺序

下面是一条连续主线。第一、第二阶段准备版本化运行时与可信教师数据；若只看模型
训练，第三至第六阶段分别对应 CombatSolver 专家轨迹 SFT、DAgger/Expert
Iteration、真实游戏 GRPO 和战略 Tree-GRPO/GiGPO。前一阶段通过验收后再进入下一
阶段，避免用有动作缺口或隐藏信息泄漏的数据训练模型。

### 第一阶段：建立 `0.111.0` 教师运行时与知识基线

目标是在不破坏现有 `0.107.1` 基线的前提下，得到可重复启动、可单独重建知识的
`0.111.0` 环境。

- [x] 保留 `.runtime/SlayTheSpire2-v0.107.1/` 及现有
  `generated-v0.107.1`，不就地覆盖、重命名或修补旧版本产物；
- [x] 将本机 Steam 临时切到 `public-beta`，把 STS2 `0.111.0` 复制到独立
  `.runtime/SlayTheSpire2-v0.111.0/` 后再切回正式分支；
- [x] 从官方渠道安装 RitsuLib 与 CombatSolver，并建立只供教师采集使用的独立
  profile；
- [x] 针对 `0.111.0` 重新编译 STS2AIAgent，验证 `/health`、`/state` 和三 Mod
  headed 启动；
- [x] 重新导出 `0.111.0` game knowledge，生成独立的
  `data/game_knowledge/mod_export/v0.111.0/` 与
  `data/game_knowledge/generated-v0.111.0/`；
- [x] 结构化比较 `0.107.1` 与 `0.111.0`，只把真实版本变化写入
  `src/play_sts2/game_knowledge/curated/v0.111.0.json` 或上游受控补录，再通过
  rebuild 生成结果，不直接修改 JSONL。

验收条件：两个版本的运行时和知识树能够并存；教师 profile 中三个 Mod 都成功
初始化；`0.111.0` 知识链可从上游输入独立重建，并有版本差异清单。游戏升级本身
不等于知识必然变化，未在导出和对比中确认的内容不做猜测性修正。

第一阶段验收记录：

- 运行时 receipt：`.runtime/SlayTheSpire2-v0.111.0/runtime-receipt.json`；
- headed 日志确认 `Loaded 3 mods`，RitsuLib 选择 `0.111.0` 变体，Agent 健康检查
  返回 `game_version=v0.111.0` 与 `status=ready`；
- Solver 搜索首场 A0 故障机器人战斗后，日志出现 `FULL_AUTO enabled=true`、连续
  `DEPLOY_ACTION`、`DEPLOY_END`，最终 `/state` 为 `REWARD`；
- 新知识审计：`data/game_knowledge/reports/v0.111.0.md`；结构化版本差异：
  `data/game_knowledge/reports/v0.111.0-diff.md`；
- 没有复用 `v0.107.1` 的事件 UI 快照冒充新版本事实。新版本仍有 49 个事件动态
  文本和 1 个能力动态描述待实机补录，raw 保留，生成器只输出可验证部分。

### 第二阶段：支持人机协作录制并完成 A0 故障机器人

目标是在同一局中记录人类的战斗外决策和 CombatSolver **实际提交给游戏的战斗
动作**，而不是转译或复制 Solver 搜索树；随后用 Harness 将动作前状态投影成 4B
可见的输入。

- [x] 在 STS2AIAgent 的外部自动执行观察层捕获卡牌、药水、结束回合和战斗内选牌，
  不修改 CombatSolver 源码；
- [x] 让动作事件携带 `combat_solver` 或 `human_ui` 来源，Recorder 支持同一局混合
  教师战斗动作与人工战略动作；
- [x] 将动作前状态与实际动作写入 `data/raw/human_combat_solver/`，保留审计元数据和
  完整的版本、预算、seed、终局信息；
- [x] 构建训练样本时只调用现有 Harness 的可见信息投影，断言输入不含精确抽牌顺序、
  RNG 状态、Solver 搜索树、节点评分和未来路线；
- [x] 完成两次 headless 单战斗冒烟：常规部署实际捕获 1 次药水、2 次出牌和 1 次
  结束回合；原生战斗内选择另捕获 1 次选牌和 1 次出牌。六条动作均唯一且来源为
  `combat_solver`；
- [x] 从 `run_started` 到 `run_ended` 录制一盘 A0 故障机器人；
- [x] 人工 review 整局，核对 Solver 部署日志、动作事件和 JSONL 行数；已有样本只要
  都能重建为合法 `ACTION:` 就保留训练资格，完整轨迹则额外要求没有动作缺口。

验收条件：第一盘 A0 故障机器人数据来源可追溯，至少一场战斗被 Solver 完整接管，
人工战略动作没有被误标成教师动作，Harness 重建样本通过可见信息审计。A0 已满足
该门槛；A1 的生产复验补上了“游戏提交动作与 raw 全量对账”的闭环。缺口修复已经
通过单候选隐式和多候选可见全息影像回归，下一次正式录制从重启后的教师进程恢复。

第二阶段实现与无头烟测记录：

- 卡牌和药水在游戏实际提交方法处捕获；结束回合在
  `EndPlayerTurnAction(Player, int)` 构造提交点捕获，避免上层回调造成一回合重复
  两条事件；
- CombatSolver 来源通过实际同步调用栈中的程序集归属识别，不引用或修改其源码；
- 合成烟测使用 CombatSolver 自带无人测试协议，在 `0.111.0` 三 Mod headless
  运行时分别验证常规两回合部署和 Toolbox 原生战斗内选牌；raw 审计、来源计数、
  六条 Harness 重建与可见文本检查均通过；
- 两个烟测局都是为了验证捕获链路构造的铁甲战士局，验证后已从正式 raw 目录移除，
  不能冒充 A0 故障机器人专家数据；
- 首盘正式数据位于
  `data/raw/human_combat_solver/20260829-a0-f48-UG7W38SMZZWJ/`。A0 故障机器人在
  48 层击败 `QUEEN_BOSS` 通关，终局 HP 为 83/116；共记录 24 场战斗、837 条动作，
  其中 CombatSolver 战斗动作 601 条、人工战略动作 236 条；
- 部署日志与 raw 逐项对齐：479 次出牌、4 次药水、114 次结束回合和 4 次战斗内
  选牌，共 601 条 Solver 动作；没有 `native_ui_capture_gap`、事件 ID 重复或来源错标；
- 837 条样本全部能够用可见 Harness 重建，目标动作均在对应 `legal_actions` 内，训练
  文本不含精确抽牌顺序、RNG 状态、Solver 搜索树、节点评分或未来路线，最终
  `training_eligible=true`；
- 可读投影已生成到
  `data/transcripts/human_combat_solver/20260829-a0-f48-UG7W38SMZZWJ/`，包含 24 个
  战斗文件和 1 个战略文件；`data/transcripts/` 同时已按 `agent`、`human` 与
  `human_combat_solver` 三种录制来源分层；
- Solver 在整局中因 `stop_on_worse_recalculation=true` 安全暂停 9 次，均发生在提交
  新动作前，由用户确认后继续；没有 `UNEXPECTED_REPLAN` 或部署漂移，因此不构成动作
  缺口。这个设置适合人工监督录制，但若以后做无人批量采集，需要单独决定是否关闭。

批量生产复验记录：

- A1 位于 `data/raw/human_combat_solver/20260829-a1-f48-9X3DLG53FW2P/`，通关至
  48 层；已录到的 721 条 Solver 动作和 240 条人工动作本身合法，但缺少 6 次
  `select_deck_card`。961 条已有样本可以训练，整局只是不具备连续完整轨迹声明；
- A2 位于 `data/raw/human_combat_solver/20260829-a2-f48-K6RFGM1UZF91/`，在
  `AEONGLASS` 第 5 回合战败；494 条 Solver 动作和 249 条人工动作完成逐动作对账，
  其中 5 次多候选 CombatPile 选择全部录到，技术录制完整；
- A2 证明当前可见路径通常有效，但没有覆盖 A1 的单候选隐式路径，也不能排除可见
  路径的间歇性静默失败。现已在隔离 0.111.0 headless 实例中强制重放两种路径：
  单候选日志 `visible=0, selected=1`，双候选日志 `visible=1, selected=1`；两次 raw
  都恰好只有 1 条 `select_deck_card`，来源、候选、索引和动作前状态正确且没有重复。
  详细修复记录见 `docs/current-doing-record.md`。

### 第三阶段：CombatSolver 专家轨迹 SFT

目标是让 4B 快速学会强战斗先验，包括合法动作、基础出牌顺序、目标选择以及能量、
药水和充能球等资源管理。这一阶段是行为克隆起点，不把它误当作最终战斗策略。

- [x] 只纳入 `training_eligible=true` 且通过可见信息审计的战斗轨迹；
- [x] 按整局或完整战斗划分 train、validation、eval，避免同一战斗的相邻状态跨
  分卷泄漏；
- [x] 从 A0 起步，确认数据和训练链路后再增加不同 seed、敌人、牌组与进阶；
- [x] 从 `docs/current-doing-sft.md` 冻结交付的 no-thinking SFT policy 分支独立
  adapter，不覆盖通用 SFT 基模；
- [x] 同时报告最终 `ACTION:` 格式、动作合法率、真实战斗结果和按动作类型的表现，
  不要求与教师逐字一致作为唯一指标。

验收条件：4B 在未见 seed 和未见战斗上相对通用 SFT 基线稳定提高合法率、胜率或
战后 HP；提升不能依赖隐藏 RNG 字段，也不能只来自重复记忆同一盘轨迹。

第三阶段验收记录：

- E5 从冻结 E4-r16 出发，使用第 3 套知识问法、A0～A5 教师池、旧人类回放和 500
  条分层算术；只用 validation 选择 r16，Solver 战斗精确率相对 E4 提高 8.32 个
  百分点，满药水奖励死点从 2/9 合法提升到 9/9；
- E6 从冻结 E5-r16 出发，使用第 4 套知识问法、E5 回放和 A6～A9 高进阶池；A6、
  A7、三盘 A8 进入 train，另一盘 A8 进入 validation，A9 进入冻结 eval，同一 run
  没有跨分卷；
- E6 两路 r16 只比较 `5e-5`/`1e-4`。`5e-5` 把未见问法知识严格率从 80.21% 提高
  到 89.09%、算术提高到 98.21%、行为合法率保持 100%；`1e-4` 的知识退化到
  74.42%，因此淘汰；
- 胜出 E6 在 temperature 0.8 的 1,902 条行为 validation 上，格式通过率 100%、
  动作合法率 99.947%；两局 A0 与三局 A8 真机共 783 次模型决策、0 次模型重试、
  13 次喝药，其中一局 A0 推进到第 35 层；
- 最终 adapter 为 `20260830-sft-e6-teacher-r16-lr5e5`。当前暂时认同它为第一代
  正式 RL 父 policy；后续 RL 若产生新的监督缺口，另开独立 SFT/DAgger 分支，不
  原地覆盖 E6，也不机械启动 E7。

### 第四阶段：DAgger / Expert Iteration

目标是修复纯专家轨迹 SFT 的分布偏移：由 4B 自己访问状态，CombatSolver 再对这些
学生真实遇到的局面给出教师动作，继续进行小轮次 SFT。

工程链已经用 E5 完成只读 Solver 建议、学生状态选择、隐藏信息隔离、标签聚合、
独立监督更新和固定场景对照；该 smoke adapter 没有取得净收益，不是正式 policy。
正式第四阶段必须改用 E6 重新采 on-policy rollout 和标签，E5 的 rollout、标签及
单步 adapter 只保留为工程证据，不能混入 E6 batch。

#### E6 正式 DAgger-R1 配方

第一轮采用课程式训练，先把 A0 做稳，同时保留一个 A8 canary：

- 固定 live validation 为 4 个 A0 完整 run；父模型和候选使用相同 seed、角色、
  temperature、xgrammar 合法动作域与运行时，分别报告最大/平均/最小/中位楼层、
  第一幕 Boss 到达和通过、第二幕到达、战斗数、相对损血、喝药、非法输出和
  基础设施失败；
- 正式训练批包含 8 个不同战斗入口，其中 7 个 A0、1 个 A8。每个入口继续按现有
  GRPO/DAgger 合同采 K=8 条同入口学生 arms，因此一轮共有 64 条冻结 E6 学生战斗，
  不是把 8 个不同入口伪装成同一个 group；
- 每个入口最多请求 16 个 Solver 标签，总预算上限 128。选择配比固定为总标签预算的
  25% 死亡/高战损尾部、50% 高不确定状态、25% 普通随机状态；尾部池不足时把空余
  名额补给另外两类，不从尾部池内部再机械抽 25%。高战损定义为相对该合成战斗入口
  损失至少 20% 最大 HP，死亡优先，再按战损和不确定度排序；
- 当前不确定度使用学生已采动作 assistant token 的平均负 processed log-prob；教师
  分歧只能在标注后统计，不能在查询 Solver 前假装已知；死亡尾部不声称是唯一致死
  因果节点，DAgger 只负责给学生真实访问分布补教师动作；
- 8 个入口全部使用同一个冻结 `20260830-sft-e6-teacher-r16-lr5e5`。批内可以由 2～4
  个本地 Agent-only worker 并行采样，教师状态重放也可由独立实例并行；只有整批
  rollout、标签和 manifest 全部冻结后才执行一次参数更新，不按每场战斗在线更新；
- 两张 A100 只用于同批标签的高价值学习率对照：r16 `1e-6` 与 `3e-6`。两路父权、
  replay、标签、顺序和评测完全相同；`1e-5` 已在工程 smoke 中出现 HP 退化，不再
  测试。选出 R1 后，下一轮必须由 R1 重新采 on-policy 状态；
- 服务器正式目录使用无项目隐私的通用路径，例如
  `~/qwen3.5-formal/`，内部模型名使用 `policy-v1`、
  `dagger-r1-lr1e6` 等通用标识。本地 raw、transcript、个人 HOME、存档和完整教师
  日志不上传；服务器只接收基座/adapter、玩家可见 observation、动作、行为
  log-prob 和最小 DAgger 标签。

第一轮的 128 个标签是正式 pilot，不预设一次就结束。候选在固定 4 个 A0 live
validation 上取得净收益后才晋级；随后用候选在 8 个新入口上既记录滚动实战指标、
又生成下一轮 on-policy 标签。固定 live validation 负责可比晋级，新的训练入口负责
滚动闭环，不能只用每轮不同 seed 的最大楼层决定胜负。xgrammar 保证 A100 采样和
晋级比较使用同一动作支持集；候选冻结后仍须合并到本机部署格式，另跑无约束动作
协议 smoke，不能把约束解码误报成模型已经学会格式。

#### E6 DAgger-R1 执行结果

正式批次已按上述配方完成，完整摘要见 `runs/rl/dagger-r1/summary.md`：

- 7 个 A0 与 1 个 A8 入口各采 K=8，共 64 条学生战斗、1,115 个动作；62 胜、
  2 死，所有 group 都有非零 reward 方差，模型重试为零；
- 首次标注发现 selector 只识别死亡、遗漏高战损通关，相关 death-only adapter
  降级为诊断；修正后以“相对入口损失至少 20% 最大 HP”为高战损阈值，正式 128
  标签为 death-tail 2、high-loss-tail 21、uncertainty 73、ordinary 32，教师同意
  32、分歧 96，并包含 4 个教师喝药动作；
- 同一批修正版标签分别训练 r16 `1e-6`、`3e-6`，每路一轮 16 步；追加 `1e-5`
  只用于判断步长。无泄漏 label-fit 只把 system/user 交给模型：父 E6 为 55/128，
  `1e-6`/`3e-6` 为 56/128，`1e-5` 为 55/128；三路仍全部错过 4 个喝药目标；
- 固定四局 A0 的父模型楼层为 13/12/15/7，均值 11.75；修正版 `1e-6` 为
  13/9/17/9，`3e-6` 与 `1e-5` 都为 13/12/12/7。所有有效局均为零重试，只有
  `1e-6` 的均值略高，但它同时包含两个退化 seed，不能据此晋级；
- 追加的未训练 `cultists-validation`/`fresh-seed` temperature 0.8 回归各 K=8：
  父 E6 通关率 100%、终局 HP 64.46%、平均 5.6875 回合；修正版最佳 `1e-6`
  降至 93.75%、50.18%、6.0625 回合，因此三路全部不晋级。

第一版 label-fit 曾错误把教师 assistant 一并送入模型，旧 108/128 结论已经作废；
修正后确认父 E6 仍有 73 个贪心 hard correction，尤其喝药为 0/4，但当前 16 步
监督更新只增加 0～1 个 teacher-match，真实回归反而退化。若未来重开 DAgger，
应按完整 group 留出新 validation，针对贪心分歧和药水缺口设计采样/加权，而不是
继续无差别增加同类标签或学习率。服务器 adapter metadata 与日志中的个人 HOME、
旧工作目录也已改写为通用路径。

- [x] 由 4B 在真实游戏中推进战斗并保存其实际到达的可见状态；
- [x] 对非法动作、明显失误、低置信度和最终失败附近的状态优先请求 Solver 标注，
  预算允许时也保留一定比例的普通状态；
- [x] Solver 标注仍只作为目标动作，学生输入继续由同一 Harness 可见投影生成；
- [x] 每轮聚合新状态、训练独立 checkpoint，并在固定未见 seed 上比较，避免无限
  回灌相似样本；
- [x] 当新增一轮标注不再稳定改善战斗结果时停止，而不是为了扩大数据集机械迭代。

验收条件：4B 自己造成的偏离状态明显减少，未见 seed 上的胜率、战后 HP 和动作
合法率优于仅做专家轨迹 SFT 的版本。

### 第五阶段：真实游戏战斗 GRPO

目标是在关闭 CombatSolver 的学生 rollout 中，只优化 4B 仍然做不好的残余局面。
本阶段训练的是可见信息下的稳健策略，不追逐教师搜索器的内部评分。

- [ ] 本机启动游戏实例并采样，A100 服务器负责推理批处理、log-prob 与参数更新；
- [ ] rollout 使用 4B 自己的动作分布，非法、截断或缺少最终 `ACTION:` 的样本保留
  并获得负反馈，不能通过重试只留下成功动作；
- [ ] reward 以战斗胜负、死亡、战后 HP 和药水消耗为主，只给很小的回合数平局项；
- [ ] 不把 CombatSolver 的 Beam 权重、搜索分数或未来信息复制进奖励函数；
- [ ] 先从第三、第四阶段暴露出的困难战斗和残余状态开始，再用完整随机战斗验证
  是否泛化。

验收条件：相同可见输入与固定评测预算下，GRPO 模型在未见 seed 上稳定改善战斗
胜率或战后 HP，并且药水消耗、非法动作率和平均回合数没有出现明显退化。

### 第六阶段：战略 Tree-GRPO / GiGPO

目标是在战斗能力已经相对稳定后，训练卡牌奖励、路线、商店、事件、营火和整局
构筑。CombatSolver 不负责这些标签，它的作用是降低战斗执行噪声，让战略长期回报
更可信。

- [ ] Tree-GRPO 从同一个战略决策状态分叉有限预算的候选动作或路线，比较后续真实
  回报，用于选牌、跳牌、购物、营火与路线决策；
- [ ] GiGPO 管理跨节点和整局的信用分配，把 Boss、幕终与最终通关结果传回前面的
  构筑决策；
- [ ] 战略 reward 以生存、Boss/幕推进和整局结果为主，金币、HP、卡组与遗物等只
  作为阶段性辅助信号，不能替代真实终局目标；
- [ ] 对第三幕等长期目标保留足够 rollout 深度，不能只按当前节点的即时收益评价
  卡组强度；
- [ ] 战斗 policy 在战略对照实验中保持固定或版本一致，避免把战斗模型变化误判为
  战略决策提升。

验收条件：在固定战斗 policy、相同角色/进阶/seed 预算下，战略模型提高幕推进率、
Boss 胜率或整局通关率，并分别报告选牌、路线、商店和营火行为，不能只给总 reward。

### 第七阶段：冻结部署

最终部署关闭 CombatSolver、RitsuLib 教师链和任何隐藏信息通道，只运行一个 4B。
部署输入与训练时 Harness 的可见信息契约完全一致；战斗和战略能力可以来自同一个
最终 adapter/合并模型，也可以先在实验阶段分 adapter 对照，但交付时必须冻结明确
的单模型血缘。

- [ ] 冻结最终 4B、Harness 版本、数据 manifest、训练配置和评测报告；
- [ ] 在不加载 CombatSolver 的普通运行时完成整局 headed smoke；
- [ ] 确认部署进程没有读取精确抽牌顺序、RNG 状态、Solver 文件或教师日志；
- [ ] 以“单个 4B 独立完成战斗与战略决策”为最终验收，不把 Solver 当线上兜底。

## 关键原则

CombatSolver 数据的正确角色是教师示范，不是最终奖励函数。教师使用隐藏 RNG 会
造成同一可见观测对应多个合理动作；初期直接保留实际动作，不为追求标签唯一性泄露
隐藏状态。数据量增加后，可以按可见状态聚类，比较不同隐藏实例下动作的胜率与战损，
优先训练跨 RNG 更稳健的动作。

当前下一步是冻结 E6，直接使用 DAgger-R1 已采的 8 个合法 K=8 group 做小预算战斗
GRPO smoke；先不混入本轮 128 个标签，单独判断真实战斗 reward 能否改善未训练入口。
若纯 GRPO 有收益，再把 73 个“父模型贪心与 Solver 分歧”按 group 划分并作为小权重
辅助消融，而不是把 96 个 temperature 0.8 采样分歧都直接解释成贪心错误。Mac
继续只运行隔离的 v0.111.0 游戏实例，A100 负责推理和参数更新，服务器通用正式目录
保持 `~/qwen3.5-formal/`。
