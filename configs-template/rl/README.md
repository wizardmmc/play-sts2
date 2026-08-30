# RL 配置

当前战斗 rollout collector 通过 `play-sts2-train collect-rl-battle` 显式接收场景、
本地游戏端点、远程模型端点和采样参数。等稳定的 RL TOML 加载器落地后，再在这里
提供由生产代码验证的模板；暂不提交不会被任何入口读取的配置字段。
