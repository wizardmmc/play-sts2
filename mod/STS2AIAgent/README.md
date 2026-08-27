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
