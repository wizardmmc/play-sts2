# 训练配置

`sft.toml` 保存 Qwen LoRA SFT 的默认数据、模型和超参数。配置文件进入 Git；
训练日志写入 `runs/sft/<name>/`，最终 adapter 写入
`models/adapters/<name>/`。当前配置从 round-2 e2 adapter 近似续训，使用
`1e-4`、r16/alpha32、2,048-token 分块 CE，并每 2,000 个优化
步覆盖一次 `checkpoint-last`。dev/test 由 `data/raw/human/splits.json` 按完整局
隔离；改动配置后先运行 token 化与单步冒烟，禁止静默截断或高频 checkpoint。
