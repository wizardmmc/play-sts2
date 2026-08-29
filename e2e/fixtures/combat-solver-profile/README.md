# CombatSolver 隔离存档模板

此模板只供 `v0.111.0` 教师运行时使用，同时启用 `STS2AIAgent`、
`STS2-RitsuLib` 与 `CombatSolver`，并显式关闭 `UnifiedSavePath`。

`progress.save` 和 `prefs.save` 复用相邻 `profile/` 的全解锁进度与无上传偏好；
启动器仍会把三份文件复制到每次运行独占的临时 HOME，不会读写个人 Steam 存档。
