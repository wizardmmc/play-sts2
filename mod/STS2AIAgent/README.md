# STS2AIAgent Mod

这里存放 STS2-Agent 的 C# 游戏 Mod。它通过 HTTP 暴露游戏状态和动作，供
Python 客户端调用。

来源与许可证信息见
[`UPSTREAM.md`](UPSTREAM.md) 和 [`LICENSE`](LICENSE)。

## 构建

工程需要引用游戏安装目录中的 `sts2.dll`、`0Harmony.dll` 和
`GodotSharp.dll`。可以通过环境变量指定目录：

```shell
STS2_DATA_DIR="/path/to/game/data" dotnet build -c Release
```

也可以复制 `local.props.example` 为 `local.props` 并填写本机路径。
`local.props`、`bin/` 和 `obj/` 不进入 Git。

## 代码检查

使用 .NET SDK 自带的格式化器检查 C# 代码：

```shell
dotnet format STS2AIAgent.csproj --verify-no-changes
```

提交前以警告即错误的方式构建：

```shell
STS2_DATA_DIR="/path/to/game/data" dotnet build -c Release -warnaserror
```

## 状态同步契约

`/state` 返回单调的 `state_revision`。依赖当前状态的 `/action` 请求必须回传
`expected_state_revision`；Mod 会在游戏线程执行动作前原子比较，过期请求以
`409 stale_state` 返回当前完整状态。开发控制台和退出清理动作保留无 revision
调用能力。

加载 CombatSolver 教师 Mod 时，`POST /solver/suggest` 接受同一个
`expected_state_revision`，在当前真实战斗状态上搜索并只返回第一条规范
`ACTION:`。该端点不会执行 Solver 路线，也不会返回搜索树、评分、牌序或 RNG；
未加载 CombatSolver 时返回 `503 solver_unavailable`。标注必须使用独占的教师游戏
并关闭 Solver full auto；开关已启用或搜索期间被打开时请求会被拒绝。

Runtime 通过 `/events/stream` 的 `stream_ready` 与 `state_changed` 等待新状态，
不按固定间隔请求 `/state`。Mod 也只在游戏事件发生时合并采集一次状态；战斗动作
窗口以引擎的 `CombatManager.TurnStarted` 为准确边界，不再使用毫秒静默窗口猜测。
