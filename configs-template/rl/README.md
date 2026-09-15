# RL 配置

`battle-grpo.toml` 由 `play-sts2-train battle-grpo` 直接读取；`tree-grpo.toml`
保留第六阶段短 horizon smoke；`strategy-grpo.toml` 由阶段七组合战略更新读取。
复制到 `configs/rl/` 后填写服务器上的模型、
rollout 与标签路径。工程验证保持 `run_role="engineering_smoke"`；最终父 policy
冻结并重新采样后，战斗正式完整训练才显式改为 `run_role="formal"`，且不使用
`--max-groups`。第七阶段先用 `init-rl-policies` 从最终 SFT adapter 的 LoRA 结构
创建两个函数为零的 residual，再用 `collect-rl-gigpo`、
`collect-rl-terminal-tree` 和 `strategy-grpo` 完成战略更新；战斗刷新继续使用
既有 battle GRPO。当前正式模板固定 `dagger_weight=0` 且不提供 `dagger_root`；
DAgger 代码只保留为人工启用的平台期回退和消融。

阶段七的 `collect-rl-gigpo`、`collect-rl-terminal-tree` 与 `eval-rl-policy` 最多接收
两个本地端口；服务器可以同时服务更多请求，但本机四个游戏会放大 Godot 转场故障。
terminal Tree 的完整战略动作上限用 `--max-model-steps` 配置，默认 400。战斗场景选择
可重复传入 `--candidates` 合并 backbone 与 Tree 文件，并必须用 `--history` 指向跨轮
固定历史文件，使 20% 普通场景欠额不会在小预算下丢失。

每个完整 group 或 optimizer 边界使用 `record-rl-cycle --journal ... --phase ...` 推进
同一份可读 journal。跳过 3+1 或不足三次滚动完整验证时落为 `provisional`；没有匹配
双 policy 的 `decide-rl-promotion` 收据时，CLI 状态机不会写成 `promoted`。组合战略 learner 在
optimizer step 后固定重算父策略 KL/ratio，超门槛会恢复父 residual，不另增配置旋钮。
新建 journal 时必须显式同时传入当前 sampling parent 和
`--last-promoted-strategy-policy/--last-promoted-battle-policy`；即使第一轮二者相同
也不能省略，避免下一轮把 provisional parent 静默当成已晋升模型。

TensorBoard/JSONL 标量可以每个 optimizer step 记录，因为单条事件很小；完整 LoRA、
AdamW 与 RNG checkpoint 不跟随这个频率。战斗 RL 默认每 50 个完整 group 覆盖一次
`checkpoint-last`，formal 配置拒绝小于 20；主动暂停无论是否整除都强制保存。正常
结束直接发布最终 adapter，不再额外重写一份相同边界的 `checkpoint-last`。
`checkpoint_groups=1` 只允许在少量工程 smoke 中验证精确恢复，不能用于长训。

战斗 rollout collector 仍通过 `collect-rl-battle` 显式接收两个本地游戏端点和冻结
模型服务端点，并要求声明服务端 xgrammar backend 与精确版本；训练配置不负责启动
游戏或推理服务。

Tree rollout 由 Mac 上的 `collect-rl-tree` 启动两个无头游戏 worker；模型推理和
训练在服务器上完成。模板里的路径按服务器工作目录的相对路径书写。

长期课程由 `init-rl-curriculum`、`plan-rl-curriculum` 和
`update-rl-curriculum` 维护一个固定 JSON；它只选择 seed、阶段和进阶，不启动服务器
或更改算法参数。完整运行顺序、TensorBoard tags 和 3+1/20% 验证预算见
`docs/training.md`。
