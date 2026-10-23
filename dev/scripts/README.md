# Real-to-Sim 辅助脚本

完整重建使用 `real2sim run --config configs/scenes/demo.yaml`。本目录只提供重建相关的批处理、实验、维护和可视化，不参与主流程调度。命令从仓库根目录运行，先完成 editable 安装。

| 目录 | 文件 | 用途 |
| --- | --- | --- |
| `batch/` | `run_stage2_batch.py` | 按历史数据布局批量运行或续跑 Blender 重建 |
| `experiments/` | `run_stage2_uncalibrated.py`、`prepare_uncalibrated_blend.py` | 无初始标定消融与 Blender 工作脚本 |
| `experiments/` | `export_stage2_experiment.py`、`export_stage2_ssim.py` | 导出实验表格、证据与四视角 SSIM |
| `maintenance/` | `organize_stage2_results.py` | 索引历史交付；显式 `--prune` 时清理已完成轮次的冗余预览 |
| `maintenance/` | `promote_stage2_deliveries.py`、`archive_stage3_runs.py` | 整理、迁移历史结果；可先执行 `--dry-run` |
| `maintenance/` | `repair_piper_wrist_blends.py` | 修复旧 Blend 的腕部相机附件并保留备份 |
| `visualization/` | `visualize_stage1_arm_cloud.py`、`visualize_stage3_grasp.py` | 点云和简化物理探针可视化 |

例如查看参数：

```bash
python dev/scripts/batch/run_stage2_batch.py --help
python dev/scripts/experiments/export_stage2_ssim.py --help
```

单阶段模块统一通过 `real2sim tools --help` 查找。Stage 3 的导出和物理检查属于 Real-to-Sim 交付；策略开发脚本已移除。

实验工具需要对应的历史输入；可视化额外使用 `matplotlib`。Blender 依次读取 `--blender`、`BLENDER_BIN` 和 PATH，macOS 另有应用目录回退。Agent 调用需要身份验证和模型额度。完整输入与运行说明见 [工作流](../../docs/WORKFLOW.md)。
