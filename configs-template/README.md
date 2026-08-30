# 配置模板

`configs-template/` 是可提交的配置示例，`configs/` 是每台机器自己的真实配置并被
Git 完整忽略。首次配置时复制需要的示例到同构路径，再填写本机模型、数据、端口和
训练参数；运行时不会回退读取模板。

```text
configs/
├── inference/
│   └── inference.toml
├── rl/
├── scenarios/
│   ├── battle.json
│   ├── battle-validation.json
│   ├── battle-fresh-seed.json
│   └── suite.json
└── sft/
    ├── sft.toml
    ├── sft-cuda.toml
    └── mix.toml
```

推理和 SFT 命令默认读取上述 `configs/inference/` 与 `configs/sft/` 路径。RL 战斗
collector 当前仍通过 CLI 显式接收场景、模型服务和采样参数；`rl/` 目录暂不提供
尚未被代码读取的伪配置。
