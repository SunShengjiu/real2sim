# Stage 1：感知、机器人配准与米制交付

新任务由 `real2sim run --config configs/scenes/demo.yaml` 调度，显式传入本次场景目录，不依赖历史任务名称或已有实验缓存。

1. 均匀采样 32 帧，缩放补边到 512×288，运行 VGGT-Ω 相机/深度和 SAM3 机器人掩码。
2. 将掩码内深度反投影，与固定关节姿态的机器人 URDF 表面点云进行 Sim(3) 配准。Franka 使用 48 个旋转初值 Top-1；Piper 使用 128 个初值 Top-3，各分支独立优化后按最终 32 帧 loss 选优。
3. 多尺度 ICP 与局部旋转搜索求出尺度和位姿。以 URDF 米制几何锚定尺度，同步缩放深度、相机外参平移和机器人平移。
4. 将世界坐标重基到机器人基座，以 0、10、21、31 帧建立四台参考相机，将 RGB 打包进 `initial.blend`，保存并重新打开验证。

| 模块 | 职责 |
| --- | --- |
| `inference.py`、`inference_batch.py` | 视频采样、后端选择、重建数据校验与感知缓存 |
| `robot_surface.py`、`geometry.py`、`piper_wrist.py` | URDF 几何、固定姿态、Piper 腕部相机与支架 |
| `initial_pose_batch.py`、`sim3_registration.py` | 多初值、反投影、投影与评分 |
| `refinement.py`、`topk_refinement.py`、`rotation_search.py` | 共享的多尺度 Sim(3) 和旋转优化 |
| `piper_top3.py` | Piper 三分支结果校验与选优 |
| `metric_calibration.py` | 米制化及机器人基座坐标变换 |
| `delivery.py`、`delivery_worker.py` | 两种机器人的标准交付与 Blender 校验 |

统一工作流中的标准交付位于 `data/runs/<运行目录>/stage1/attempt_001/scene/`，含 `stage1_delivery.json`、`registration.json`、`camera_metric.json`、`depth_metric/`、四视角参考和 `initial.blend`。格式见 [数据契约](../../../docs/DATA_CONTRACT.md)。

## 阶段诊断

```bash
real2sim models check
real2sim tools stage1-init-batch --help
real2sim tools stage1-refine --help
real2sim tools stage1-deliver --robot piper --help
```

`stage1-deliver --robot franka|piper` 共用同一个导出器。独立指定历史缓存时，提供 `--processed`、`--results`、`--initializations` 和可选的 `--only`；目录中 `<任务>/initialization.json` 的哈希必须与配准记录一致。

SAM3 掩码复查、提示词对比、物体伪参考指标及历史配准实验均收在 `real2sim tools` 下，不自动覆盖主流程结果。物体指标以 SAM3 和重建深度为伪参考，不能当作实测真值。`legacy.py` 的外部 `--metric-scale` 路径只用于旧版诊断，不用于本次主流程。

## 验证边界与历史证据

生产配准以同一组 32 帧拟合、选优和评分，没有独立验收帧；交付保持 `candidate_unverified`、`accepted=false`。URDF 米制化依赖真实机器人姿态与模板一致，不能单凭低 loss 宣称已验证尺度或深度精度。主流程保留这一状态供后续视觉建模和人工审阅。

历史算法、具体场景与旧目录的记录见 [Franka 交付](../../../docs/STAGE1_FRANKA_DELIVERY.md)、[Piper 交付](../../../docs/STAGE1_PIPER_DELIVERY.md)、[CPU 验证](../../../docs/STAGE1_CPU_VALIDATION.md)。它们不代表当前真实视频完整复现已通过。
