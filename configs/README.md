# 配置

`sft.toml` 保存 Qwen LoRA SFT 的默认数据、模型和超参数。配置文件进入 Git；
训练日志写入 `runs/sft/<name>/`，最终 adapter 写入
`models/adapters/<name>/`，名称必须以 `YYYYMMDD-` 开头。当前配置用 round-2
e2 adapter 初始化新的 E3 运行，使用
`1e-4`、r16/alpha32、2,048-token 分块 CE，并每 2,000 个优化
步覆盖一次带 AdamW、游标和随机状态的精确 `checkpoint-last`；同名运行增加
`--resume` 可以精确恢复。验证/测试由 `data/raw/human/splits.json` 按完整局
隔离；改动配置后先运行 token 化与单步冒烟，禁止静默截断或高频 checkpoint。

`inference.toml` 是本地模型转换、服务、冒烟和游戏 Runner 共用的唯一模型选择。
`artifact_id`、合并目录和 MLX 目录必须对应；默认预先指向下一轮 e3，因此 e3
尚未合并和转换时，无参数命令会明确失败，不会回退到 e2。`no-think` 与 `think`
profile 都通过每次 MLX 请求的 `chat_template_kwargs.enable_thinking` 显式选择动态
模板分支；e3 验收默认使用与 SFT 编码一致的 `no-think`。

转换会解析 `merge_manifest.json` 的 adapter 与输出目录血缘，只对该小清单取
SHA-256，并把摘要写入 `serving_manifest.json`；不会扫描或重复哈希模型权重。
服务启动会核对实际量化/EOS，并用固定对话的渲染文本与 token IDs 证明
`enable_thinking` 的两个分支不同且转换前后一致。冒烟和游戏 Runner 还会在连接
游戏之前通过 MLX `/v1/models` 核对当前端口声明的精确目录，再以该目录的绝对
路径执行一次真实生成；后续每个请求也固定该路径并核对响应模型标识。配置必须
保留 `no-think=false`、`think=true`，且默认 profile 必须是 `no-think`；可以另加
实验 profile，但不能改写这组验收语义。
