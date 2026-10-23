# Runs

每次运行的场景、状态、报告、日志和失败诊断存放于此；实际运行产物不提交 Git。

统一入口将新任务写入 `<场景名>_<北京时间>/`，包含 `config.json`、`manifest.json`、三个阶段的 `attempt_001/` 及最终 `delivery/`。失败后使用 `real2sim resume --run <运行目录>`，会保留失败尝试并重新执行该阶段。

将整个 `delivery/` 复制到其他位置后，执行 `real2sim verify --run <交付目录>` 检查资产哈希和 MuJoCo 加载。整个运行目录中的来源绝对路径用于本机续跑；独立搬移的交付单元是 `delivery/`。
