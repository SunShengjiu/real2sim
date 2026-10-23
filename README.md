# Agentic RSR — Real-to-Sim

本目录从 Agentic RSR 中独立保留 Real-to-Sim：输入真实工作空间视频、任务描述和已知机器人模型，输出具有米制尺度、固定参考相机、任务物体与碰撞表示的 MuJoCo 场景及检查报告。

```text
真实视频 + 任务描述 + 机器人 URDF / 模型
  → VGGT-Ω 深度与相机 + SAM3 机器人分割
  → 反投影点云 + Sim(3) 配准与米制坐标恢复
  → Blender 场景建模、四视角渲染与数值反馈
  → MuJoCo 导出、碰撞/关节配置与物理合理性检查
  → scene.xml + assets/ + manifests / reports（Real-to-Sim 终点）
```

统一入口位于 `src/real2sim/cli.py`，调度位于 `pipeline.py`，配置和产物约定分别位于 `config.py`、`contracts.py`。

## 仓库布局

```text
configs/   场景与机器人配置
src/       Real-to-Sim 源码
data/      视频、机器人资产、模型权重、运行结果
external/  固定版本的第三方依赖
docs/      使用与方法文档
dev/       辅助脚本和测试
```

输入视频放在 `data/raw/`，两份权重放在 `data/models/`，新结果写入 `data/runs/`。详细目录及迁移对应关系见 [仓库结构](docs/PROJECT_STRUCTURE.md)。

## 代码从哪里读

| 环节 | 主要代码 | 作用 |
| --- | --- | --- |
| 模型接入 | `src/real2sim/perception/` | 本地/远端 VGGT-Ω、SAM3 和统一输出格式 |
| 视频与初始几何 | `src/real2sim/stage1/inference_batch.py`、`geometry.py` | 32 帧采样、推理、反投影 |
| 机器人配准 | `stage1/robot_surface.py`、`initial_pose_batch.py`、`sim3_registration.py` | 已知机器人模型、多候选初始化、尺度和位姿求解 |
| Stage 1 交付 | `stage1/metric_calibration.py`、`delivery.py`、`delivery_worker.py` | 米制深度、机器人基座坐标相机与 `initial.blend` |
| 场景细化 | `stage2/phase_runner.py`、`blender_worker.py`、`evaluator.py` | 冻结机器人/相机，修改场景、渲染、计算深度/颜色/覆盖率反馈 |
| 物理场景 | `stage3/mujoco_export.py`、`semantics_agent.py` | 任务物体分类、网格与 MJCF 导出、碰撞和关节 |
| 物理检查 | `stage3/physics_probe.py`、`agent_runner.py`、`batch.py` | 静置、扰动、简化夹取与参数候选复验 |

表中 `stage1/`、`stage2/`、`stage3/` 均位于 `src/real2sim/`。完整命令见 [WORKFLOW.md](docs/WORKFLOW.md)。

## 安装与使用

项目使用 Python 3.11。Windows CPU 环境已验证离线测试、Blender 渲染和 MuJoCo 检查；真实视频与模型权重的完整流程仍待验证。Linux x86_64 上可使用 NVIDIA CUDA 加速感知推理。Blender 需另行安装。克隆本仓库及固定版本依赖后安装：

```bash
git clone --recurse-submodules https://github.com/SunShengjiu/real2sim.git
cd real2sim
uv sync --locked --python 3.11 --extra local --extra sim --extra agent --extra dev
```

`sim` extra 在本目录中用于 Stage 3 的 MuJoCo 物理检查和机器人模型装配。`external/robot_control` 保留为上游依赖：Stage 1 读取其机器人几何，Stage 3 复用其模型装配；主项目的策略开发、随机化评测和真机部署代码已剥离。

按 [LOCAL_INFERENCE.md](docs/LOCAL_INFERENCE.md) 准备权重、Blender 与 Codex 身份验证，编辑 [场景配置](configs/scenes/demo.yaml) 中的视频路径和任务，然后运行：

```bash
uv run --no-sync real2sim check --config configs/scenes/demo.yaml
uv run --no-sync real2sim run --config configs/scenes/demo.yaml
uv run --no-sync real2sim resume --run data/runs/<运行目录>
uv run --no-sync real2sim verify --run data/runs/<运行目录>/delivery
```

统一入口依次执行三个阶段、阻止失败阶段继续流转，并交付可整体搬移的 `delivery/scene.xml`、资产和报告。配置、输入/源码/权重哈希及阶段状态保存在同一运行目录。示例 YAML 需要自行提供视频；当前没有附带已验收的完整视频示例。详细配置、续跑和交付说明见 [工作流](docs/WORKFLOW.md)。

## 验证与边界

```bash
uv sync --locked --python 3.11 --extra sim --extra dev
uv run --no-sync python -m pytest dev/tests/unit dev/tests/integration -m 'not models' -ra
```

- 生产 Stage 1 用全部 32 帧拟合和评分，目前交付仍为 `candidate_unverified`、`accepted=false`。
- Stage 2 的参考深度来自重建，指标表示与该参考的一致性；这份代码保留的是历史 Codex 实现路径。
- Stage 3 的仿真探针检查物理合理性；无实测对照时，结果最多为 `provisional`。
- 模型权重、视频、第三方子模块、Blender 和模型服务身份验证需要单独准备；离线测试不代表真实视频全流程已通过。

[English](docs/README.en.md) · [架构](docs/ARCHITECTURE.md) · [数据契约](docs/DATA_CONTRACT.md) · [测试](dev/tests/README.md) · [第三方来源](docs/THIRD_PARTY.md)

原项目与论文署名见 [CITATION.cff](CITATION.cff)。原创代码采用 [Apache-2.0](LICENSE)；第三方源码、权重和机器人资产保留各自条款，见 [NOTICE](NOTICE)。
