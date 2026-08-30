# RL 配置

`battle-grpo.toml` 由 `play-sts2-train battle-grpo` 直接读取；`tree-grpo.toml`
由 `play-sts2-train tree-grpo` 读取。复制到 `configs/rl/` 后填写 A100 上的模型、
rollout 与标签路径。工程验证保持 `run_role="engineering_smoke"`；最终父 policy
冻结并重新采样后，战斗正式完整训练才显式改为 `run_role="formal"`，且不使用
`--max-groups`。当前 Tree 入口只允许单个 group 的工程 smoke。

战斗 rollout collector 仍通过 `collect-rl-battle` 显式接收两个本地游戏端点和冻结
模型服务端点，并要求声明服务端 xgrammar backend 与精确版本；训练配置不负责启动
游戏或推理服务。

Tree rollout 由 Mac 上的 `collect-rl-tree` 启动两个无头游戏 worker；模型推理和
训练仍在 A100 上完成。模板里的路径是服务器工作目录 `qwen3.5` 下的相对路径。
