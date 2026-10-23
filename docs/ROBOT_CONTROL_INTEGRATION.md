# Real-to-Sim 使用的机器人依赖

`external/robot_control` 是固定提交的上游子模块。虽然它自身包含仿真、控制和硬件后端，Real-to-Sim 仍需保留它：

| 使用位置 | 所需能力 |
| --- | --- |
| `stage1/geometry.py`、`robot_surface.py` | Piper/FR3 URDF、visual 网格和固定关节姿态；构造已知米制表面 |
| `stage1/piper_wrist.py` | 腕部相机与支架几何 |
| `stage3/mujoco_export.py` | Piper Xacro/场景装配、FR3 MJCF、机器人基座及初始姿态 |
| `stage3/physics_probe.py` | 需要时为旧场景装配 Piper，使物理探针可加载场景 |

```bash
git submodule update --init --recursive
```

配准模板、Blender 机器人和 MuJoCo 机器人应绑定同一资产版本。修改模型或姿态后，需重新生成并核对表面模板与 provenance，不能手改来源哈希。

`sim` extra 只包含 MuJoCo、Pillow 和 robot-control 基础包，供 Stage 3 场景检查与机器人装配使用；本项目不建立机器人控制会话，因此不需要 Pinocchio/IK。主项目不再提供任务策略生成、随机化策略评测或真机执行入口。上游子模块保持原样，保留其版权与资产说明；来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。
