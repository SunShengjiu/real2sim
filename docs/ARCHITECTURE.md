# Real-to-Sim 架构

本目录只实现真实视频到物理仿真场景的构建，终点为 Stage 3 的场景包及检查报告。命令见 [WORKFLOW.md](WORKFLOW.md)，格式见 [DATA_CONTRACT.md](DATA_CONTRACT.md)。

```text
视频 + 任务描述 + 固定机器人模型
  │
  ├─ perception：VGGT-Ω 深度/相机；SAM3 机器人 mask
  ▼
Stage 1：反投影 → 机器人模型配准 → 米制尺度与机器人基座坐标
  │  depth_metric/ + camera_metric.json + registration.json + initial.blend
  ▼
Stage 2：规划 → Blender 建模/渲染 → 四视角深度/颜色/覆盖率反馈
  │  冻结机器人和相机；交付 accepted=true 且哈希匹配的 final.blend
  ▼
Stage 3：物体分类 → MJCF/网格导出 → 物理探针与候选复验
     scene.xml + assets/ + manifests / reports
```

| 模块 | 职责 |
| --- | --- |
| `src/real2sim/perception/` | 默认本地、显式远端的模型后端和统一数据契约 |
| `src/real2sim/stage1/` | 32 帧采样、机器人点云、配准、坐标恢复与初始场景 |
| `src/real2sim/stage2/` | Agent 建模、冻结资产检查、渲染评价与交付 |
| `src/real2sim/stage3/` | 语义、碰撞/关节、MuJoCo 导出和物理合理性检查 |
| `external/robot_control/` | 上游机器人 URDF、网格、固定姿态和 MuJoCo 装配依赖 |
| `dev/scripts/` | 重建批处理、评价、消融、结果整理与可视化 |

所有阶段均不依赖已剥离的任务策略运行模块。保留的上游机器人子模块含其他功能，但本项目使用其几何和装配能力，详见 [ROBOT_CONTROL_INTEGRATION.md](ROBOT_CONTROL_INTEGRATION.md)。

生产配准使用全部 32 帧拟合与评分，交付仍为 `candidate_unverified`。Stage 2 可探索性消费该输入，不改变其独立验收状态；深度参考来自重建。Stage 3 仿真检查无实测对照时最多为 `provisional`。当前代码是论文中的历史 Codex 路径，不能据此保证复现论文全部结果。
