# Real-to-Sim

本目录包含从真实视频到可加载 MuJoCo 场景的三个阶段：

1. [Stage 1](stage1/README.md)：视频采样、VGGT-Ω/SAM3、反投影、机器人 Sim(3) 配准，交付米制相机/深度和 `initial.blend`。
2. [Stage 2](stage2/README.md)：冻结机器人与四参考相机，由 Codex 修改 Blender 场景，依据深度、颜色与覆盖率反馈迭代，交付已验收 `final.blend`。
3. [Stage 3](stage3/README.md)：语义分类、MuJoCo MJCF/资产导出、静置与简化夹取等物理检查，交付最终场景及报告。

感知适配器位于 `perception/`，机器人模型来自固定版本 `external/robot_control`。每个阶段保留来源、哈希、坐标和验收状态；检查通过与真实物理标定应分别表述。

本工作流在 Stage 3 场景交付处结束。入口、顺序及依赖见 [WORKFLOW.md](../../docs/WORKFLOW.md)。

统一入口为 `cli.py`；`pipeline.py` 调用三个阶段，`config.py` 校验场景与机器人参数，`contracts.py` 管理来源哈希和相对路径交付。
