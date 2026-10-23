# Robot Configurations

`piper.yaml` 与 `franka.yaml` 绑定固定 URDF、基座、关节姿态和配准设置。关节旋转使用弧度，夹爪开口使用米。视频里的机器人必须保持该姿态。

当前 Piper 使用 128 个旋转初始化和 Top-3 精化，Franka 使用 48 个旋转和 Top-1；任意更改这些值会被配置校验拒绝。表面模板与实际 URDF 哈希在执行前核对，真实资产来自固定版本 `external/robot_control`。
