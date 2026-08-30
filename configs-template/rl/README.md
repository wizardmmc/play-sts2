# RL 配置

`battle-grpo.toml` 由 `play-sts2-train battle-grpo` 直接读取。复制到
`configs/rl/battle-grpo.toml` 后填写本机或 A100 上的模型、rollout 与标签路径。
E5 工程验证保持 `run_role="engineering_smoke"`；最终父 policy 冻结并重新采样后，
正式完整训练显式改为 `run_role="formal"`，且不使用 `--max-groups`。

战斗 rollout collector 仍通过 `collect-rl-battle` 显式接收两个本地游戏端点和冻结
模型服务端点，并要求声明服务端 xgrammar backend 与精确版本；训练配置不负责启动
游戏或推理服务。
