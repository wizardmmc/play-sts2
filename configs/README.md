# 配置

本目录下的 TOML 是按机器维护的本地配置，不进入 Git；已有工作区会继续保留这些
文件，新环境首次运行前需要按下面的契约创建对应配置。

`sft.toml` 保存 Qwen LoRA SFT 的本机默认数据、模型和超参数。
训练日志写入 `runs/sft/<name>/`，最终 adapter 写入
`models/adapters/<name>/`，名称必须以 `YYYYMMDD-` 开头。E4 CUDA 配置用冻结的
E3 adapter 初始化新运行，使用 `5e-5`/`1e-4` 两组对照、r16/alpha32、
梯度累积 8、2,048-token 分块 CE，并每 250 个优化
步覆盖一次带 AdamW、游标和随机状态的精确 `checkpoint-last`；同名运行增加
`--resume` 可以精确恢复。验证/测试由 `data/raw/human/splits.json` 按完整局
隔离；改动配置后先运行 token 化与单步冒烟，禁止静默截断或高频 checkpoint。

`sft-e4-mix.toml` 只保存 train 中高频人类动作的上限及旧行为数据的真实游戏版本：
知识实体与事实必须全部保留，远古者由上游知识生成范围排除；未列出的低频动作全部
保留。构建时显式指定 `generated-v0.111.0`。

`sft-e5-mix.toml` 在相同旧行为上限之外，按六类固定 500 条算术；
`human_combat_solver` 来源不受旧行为上限裁剪。E5 CUDA 的 r16/r32 配置都从选定
E4 开始，使用 `knowledge_epoch_start=3` 和一轮 `1e-4`；r32/alpha64 额外设置
`expand_init_adapter=true`，保持与 r16/alpha32 相同的缩放。

`sft-cuda.toml` 使用同一数据和 LoRA 配方，但只允许 `cuda` 或 `cuda:N`，基座以
BF16 加载，LoRA 可训练参数仍强制为 FP32。通过独立的 `sft-cuda` 子命令运行，
不改变 Mac 默认配置和 `sft` 命令。E4 每轮实际使用 10,247 条样本，梯度累积 8 后
约 1,281 个优化步；CUDA 配置每 250 步覆盖一次 `checkpoint-last`。
每个完整 epoch 还会保存不可覆盖的 `checkpoint-epoch-N`，供逐轮验证和精确恢复。

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
