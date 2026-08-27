# 后训练边界

- `data/` 保存原始轨迹、转录结果和训练数据集。
- `models/base/` 保存未经项目训练的基座模型。
- `models/adapters/` 和 `models/merged/` 保存选定的训练产物。
- `runs/` 保存每次实验的日志、指标和中间 checkpoint。
- `configs/` 保存能够人工阅读和修改的训练参数。

当前阶段先完成真实轨迹录制与转录，再接入 PyTorch、MPS、SFT 和 RL。
