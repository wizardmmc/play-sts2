# E2E 测试存档

这里保存可公开提交的专用测试档，只用于隔离 HOME 下的真实游戏 E2E。

- 五个角色均解锁至 A10。
- 卡牌、遗物、药水、事件、章节和时代节点均已发现。
- 教程已关闭，不包含进行中的游戏。
- `unique_id`、解锁时间与游玩统计均使用固定测试值。
- 只启用 `STS2AIAgent`，并关闭 `UnifiedSavePath`。

不要用个人 `progress.save` 或 `settings.save` 直接覆盖这些文件。更新存档后必须
运行 `uv run pytest tests/test_e2e_profile.py e2e/test_launcher.py`。
