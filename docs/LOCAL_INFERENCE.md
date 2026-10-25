# 本地感知模型安装与最小复现

首个完整推理验收目标是 **Ubuntu x86_64、NVIDIA RTX 5080、Python 3.11、PyTorch 2.8.0 / CUDA 12.8**。源码以固定提交的子模块放在本仓库，checkpoint 放入 `data/models/`，也可以挂载到对应目录。默认使用本地 VGGT-Ω 和 SAM3；仅显式选择 `remote` 时连接远端服务。

当前本机为 Windows CPU 开发环境，已经完成离线代码、机器人资产、Blender 和 MuJoCo 检查，尚未完成真实模型推理。安装成功、模型检查成功与真实视频复现通过是三个独立的验收步骤。CPU/MPS 可显式配置，真实推理的性能与内存需求仍需实测。

## 1. 源码与环境

```bash
git submodule update --init --recursive
uv sync --locked --python 3.11 --extra local --extra sim --extra dev
```

当前按源码 checkout + editable 安装使用，不承诺独立 wheel 的机器人资产可用性。`uv.lock` 固定依赖解析结果；`uv sync --locked` 遇到过期锁文件会报错，不更新版本。Linux x86_64 的本地推理使用 PyTorch 2.8.0 / torchvision 0.23.0 的 CUDA 12.8 wheel；Windows 继续选择 PyTorch 2.6.0 / torchvision 0.21.0 CPU wheel，其他平台保留原有版本。安装依据见 [PyTorch 官方版本矩阵](https://pytorch.org/get-started/previous-versions/#v280)。RTX 5080 属于 Blackwell，不能沿用原 PyTorch 2.6 / CUDA 12.4 环境。

Ubuntu 机器先确认 `nvidia-smi` 能识别 RTX 5080，驱动支持该卡及 CUDA 12.8。PyTorch wheel 和锁定依赖提供 CUDA 用户态运行库，不需要为本流程单独安装另一套 CUDA Toolkit。旧环境建议在拉取后重新执行上述 `uv sync --locked`，不要用系统 Python 代替项目环境。新拉取时默认设备为 `cuda`；若自行复制了旧 `.env`，需检查两个 `*_DEVICE` 都为 `cuda`。

精简 Linux 系统还需 OpenCV 的 GL/GLib 动态库，以及无窗口 MuJoCo 渲染使用的 EGL/Mesa。Ubuntu 24.04/26.04 和 Debian 13 使用以下包名；Ubuntu 22.04 将 `libglib2.0-0t64` 替换为 `libglib2.0-0`，见 [Ubuntu GLib 包列表](https://packages.ubuntu.com/glib2.0)。这些库不提供 NVIDIA 驱动或 CUDA 验收。

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0t64 libegl1 libgl1-mesa-dri
export MUJOCO_GL=egl
```

Blender 另行安装，用于 Stage 1 场景交付、Stage 2 建模/渲染和 Stage 3 导出；感知最小示例本身不调用 Blender。各阶段和相关实验脚本统一按以下优先级选择可执行文件：显式 `--blender` 参数、`BLENDER_BIN` 环境变量、PATH 中的 `blender`；仅在 macOS 上最后尝试 `/Applications/Blender.app/Contents/MacOS/Blender`。参数和环境变量既可使用路径（支持 `~` 和空格），也可使用 PATH 中的命令名。显式配置缺失或不可执行时直接报错，不改用另一个安装。

```bash
blender --version  # Blender already on PATH
# For a separate installation, replace this path:
export BLENDER_BIN=/path/to/blender
```

| 源码 | 位置 | 固定提交 |
| --- | --- | --- |
| [VGGT-Ω](https://github.com/facebookresearch/vggt-omega) | `external/vggt_omega/` | `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6` |
| [Ultralytics v8.4.160](https://github.com/ultralytics/ultralytics) | `external/ultralytics/` | `f171c4b7aea56062d77061577e6b878b62e520fb` |
| [机器人控制包](https://github.com/HY-LiYihan/robot_control) | `external/robot_control/` | 以主仓库 gitlink 为准 |

`local` extra 包含 `numpy>=1.26,<2`、`opencv-python<4.12`：VGGT-Ω 要求 NumPy 1.x，而 OpenCV 4.12 开始的相关 wheel 要求 NumPy 2.x。支持的项目 Python 范围为 3.10 至 3.12，完整推理统一选 3.11。

SAM3 文本编码器需要 Ultralytics CLIP 包及其内置 BPE 词表；锁文件通过官方源码 ZIP 及 SHA256 固定 [CLIP 提交](https://github.com/ultralytics/CLIP/tree/7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40)。不要安装 PyPI 上同名但不兼容的 `clip` 包。安装阶段获取依赖；本地推理检查依赖，不在运行中安装包或下载模型。

`sim` extra 只安装 Stage 3 场景导出和物理检查需要的 MuJoCo、Pillow 与 robot-control 基础包。配准和模型装配不建立运动控制会话，不需要 Pinocchio/IK 及其 cmeel 依赖。上游 robot-control 的控制扩展保持原样，本项目不安装这些扩展。

仅运行无权重测试时不安装 `local`：

```bash
uv sync --locked --python 3.11 --extra sim --extra dev
uv run --no-sync python -m pytest dev/tests/unit dev/tests/integration -m 'not models' -ra
```

## 2. 准备权重

阅读[第三方来源与许可](THIRD_PARTY.md)后，可显式选择公开镜像，不需要 Hugging Face 账号或 Token：

```bash
uv run --no-sync real2sim models download --source mirror
```

镜像固定为 [1kaiser/vggt-omega-jax](https://huggingface.co/1kaiser/vggt-omega-jax/blob/main/vggt_omega_1b_512.pt) 和 [cubert-gmbh/sam3](https://huggingface.co/cubert-gmbh/sam3)。下载器对元数据及实际文件都核对项目记录的官方大小和 SHA256，镜像请求明确不携带 Hugging Face 凭据；镜像不可用或校验失败时明确报错，不隐式切换来源。镜像只是分发渠道，原模型许可仍适用。

也可使用默认的官方来源。分别在官方 Hugging Face 页面申请访问并接受条款：

- [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega)：选择 `vggt_omega_1b_512.pt`。416 reproduction checkpoint 不用于当前 512 输入契约。
- [facebook/sam3](https://huggingface.co/facebook/sam3)：选择 `sam3.pt`，不混用 SAM3.1 checkpoint。

批准后在本机登录 Hugging Face，下载到忽略目录：

```bash
uv run --no-sync hf auth login
uv run --no-sync real2sim models download
uv run --no-sync real2sim models check
```

当前安装的 HF CLI 支持浏览器设备授权：打开终端显示的链接，输入短码并确认授权即可。模型访问申请需要在两个模型页面分别完成；登录成功不等于访问申请已获批。凭据由 HF CLI 保存，不发送到聊天或提交 Git。

Windows 可直接执行 `& .venv/Scripts/hf.exe auth login`，再执行 `& .venv/Scripts/real2sim.exe models download`。下载器使用既有授权，固定官方仓库提交，核对文件大小与 SHA256；已存在且一致的文件会复用，内容不同的已有文件会报错。具体版本和哈希见 `src/real2sim/perception/config.py` 中的 `MODEL_WEIGHTS`。两个文件合计约 8.03 GB。下载成功不代表模型推理已通过。

也可将已有且哈希一致的权重目录挂载到 `data/models/vggt_omega/`、`data/models/sam3/`。默认路径：

```text
data/models/vggt_omega/vggt_omega_1b_512.pt
data/models/sam3/sam3.pt
```

权重、Hugging Face 下载缓存与凭据不提交 Git。没有 checkpoint 时明确报错，不隐式下载，不自动改用远端。

## 3. 后端与设备

从 `.env.example` 复制本机 `.env`，或设置环境变量。相对模型目录始终相对于仓库根目录解析；绝对路径可用于外部挂载。

```dotenv
VGGT_OMEGA_BACKEND=local
VGGT_OMEGA_MODEL_DIR=data/models/vggt_omega
VGGT_OMEGA_DEVICE=cuda
SAM3_BACKEND=local
SAM3_MODEL_DIR=data/models/sam3
SAM3_DEVICE=cuda
```

设备可指定 `cuda:0`；配置请求的 CUDA 设备不存在时会失败。两模型可独立选择后端；服务地址仅在该模型的 `remote` 模式读取。例如：

```dotenv
VGGT_OMEGA_BACKEND=remote
VGGT_OMEGA_HOST=your-vggt-host
VGGT_OMEGA_PORT=9001
VGGT_OMEGA_PROTOCOL=tcp
SAM3_BACKEND=local
```

SAM3 远端对应 `SAM3_BACKEND=remote`、`SAM3_HOST`、`SAM3_PORT`、`SAM3_PROTOCOL=tcp`。旧 CLI 显式传入的 SAM3 host/port 继续表示远端调用。Stage 1、物体评估和提示词试验共享配置。

## 4. 检查与最小示例

检查配置、文件、源码版本和设备。CUDA 检查报告 GPU 名称、计算能力、总显存及当前可用显存，并实际执行一个小型 CUDA 运算，避免把“能发现显卡”误当成 PyTorch 已支持该架构；`--load` 进一步串行加载并释放两个本地模型，但不代替真实视频推理：

```bash
uv run --no-sync real2sim models check
uv run --no-sync real2sim models check --load
```

使用自己的至少 32 帧任务视频，选择一个尚不存在的输出目录：

```bash
uv run --no-sync real2sim perceive \
  --video /path/to/task.mp4 --robot piper \
  --output data/runs/local-example
```

示例输出保持 Stage 1 的帧、深度、相机、mask 和 manifest 结构，并增加 `previews/` 和 `example_report.json`。报告包含后端、源码提交、checkpoint SHA256、预处理版本、设备、Torch/CUDA 版本、耗时和峰值显存。目录已存在即拒绝执行；失败状态保留诊断，不能当作复现通过。

输入为 32 帧、512×288 RGB；VGGT-Ω 直接使用此像素网格，不调用会裁剪宽图的默认加载器。深度、置信度与内参需与该网格对齐。SAM3 实例 mask 恢复到原图网格后，Stage 1 合并实例，物体评估按分数与面积规则选择实例。VGGT 与 SAM3 串行执行，阶段切换释放模型；显存不足会报错，不自动减少帧数或分辨率。

这一步只验收感知输出，不完成 URDF 配准、米制尺度锚定或 Blender 场景交付。后续步骤参考 Stage 1 文档。

## 5. 验证边界

当前仓库没有 GitHub Actions CI（已按作者要求移除）。无 GPU/无 checkpoint 的离线测试覆盖配置、后端契约、实例合并/选择、mask 网格以及相机投影，同时保留远端协议回归；提交修改前需手动运行。Linux 无 GPU 的 MuJoCo 渲染可使用 OSMesa，真实 Linux GPU 观测使用 EGL。真实模型测试标记为 `models`，需要显式提供权重和测试视频，在 Linux CUDA 环境运行：

```bash
export RSR_MODEL_TEST_VIDEO=/path/to/task.mp4
uv run --no-sync python -m pytest dev/tests/integration -m models
```

没有提供相应的显式输入或缺少 CUDA 时真实模型测试会跳过；跳过不代表推理通过。保留的真实模型测试读取上述视频，验证视频感知输出。

必须保存真实运行报告及依赖版本后，才能声明本地完整推理复现通过。推理性能和峰值显存受视频、GPU 与模型设备设置影响，不引用上游不同输入尺寸的显存表作为本流程保证。

## Windows 本机开发环境

Windows x64 的锁定依赖选择 PyTorch 2.6.0 CPU wheel，Linux x86_64 使用 PyTorch 2.8.0 CUDA 12.8 wheel。本项目的场景构建/物理探针不需要运动控制 IK，因此 `sim` extra 不安装 Pinocchio。

本机隔离环境位于仓库 `.venv/`；便携 Blender 位于 `.venv/tools/`，路径保存在忽略的 `.env` 中。可以直接调用，不需要修改系统 PATH 或 PowerShell 执行策略：

```powershell
& .venv/Scripts/real2sim.exe --help
& .venv/Scripts/python.exe -m pytest dev/tests/unit dev/tests/integration -m 'not models' -ra
& .venv/Scripts/real2sim.exe models check
```

离线测试覆盖代码、机器人资产、Blender 场景和 MuJoCo 物理检查。模型检查因没有 checkpoint 而失败属于输入未准备，不能把安装成功写成真实模型推理通过。`.env` 的 CPU 默认值用于感知开发工具；场景 YAML 中显式的 `perception.device` 会覆盖它。

Ryzen AI 9 365 / Radeon 880M 在 AMD 官方 Windows ROCm 支持矩阵中，但其 PyTorch 2.9.1 / Python 3.12 / 指定驱动方案与本仓库的 PyTorch 2.6 锁定栈不同，需要独立适配和验收。本轮不修改显卡驱动，也不声称 32 帧 VGGT-Ω/SAM3 已在该核显通过。

官方依据：[AMD 支持矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/windows/windows_compatibility.html)、[AMD Windows 安装要求](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/windows/install-pytorch.html)、[PyTorch 2.6 安装矩阵](https://docs.pytorch.org/get-started/previous-versions/)。

### 2026-10-23 Windows 本机安装验证

硬件：Ryzen AI 9 H 365（10 核 20 线程）、约 32 GB 内存、Radeon 880M 核显，Windows 11 x64。隔离环境为 Python 3.11、PyTorch 2.6.0+cpu、torchvision 0.21.0+cpu、NumPy 1.26.4、OpenCV 4.11.0、MuJoCo 3.14.0、Blender 4.5.15 LTS；具体 Python 依赖以 `uv.lock` 为准。

安装检查通过：依赖一致性、锁文件一致性、三个固定子模块及嵌套机器人资产、CLIP 分词器、CLI、Blender CPU/Eevee 渲染、MuJoCo 物理步进与离屏渲染。完整离线命令：

```powershell
$env:BLENDER_BIN = (Resolve-Path '.venv/tools/blender-4.5.15-windows-x64/blender.exe').Path
& .venv/Scripts/python.exe -B -m pytest -p no:cacheprovider dev/tests/unit dev/tests/integration dev/tests/end_to_end -m 'not models' -q -ra
```

2026-10-24 顶层布局归并后重新完整回归：291 passed、3 skipped、1 deselected。测试由 `pyproject.toml` 从 `dev/tests/` 自动发现，资产、模型和输出分别位于 `data/assets/`、`data/models/`、`data/runs/`。跳过的是两个 POSIX 权限测试和未提供 `RSR_E2E_CONFIG` 的真实端到端测试；真实 checkpoint 视频测试未选中。编码与日志处理已显式使用 UTF-8。唯一安装命令为 `real2sim`，24 个命令帮助检查通过；直接从虚拟环境启动时，子工具也使用同一 Python 环境。

上述 2026-10-24 检查时两个 checkpoint 尚未准备。2026-10-25 已从公开镜像补齐 8.03 GB 权重，两个文件的大小和 SHA256 均匹配项目固定的官方版本，CPU `models check` 通过。额外的元数据检查确认 VGGT 的 1,411 个参数条目和 SAM3 图像模型所需的 1,118 个条目尺寸匹配；现有 Piper 视频完整解码 706 帧，可采样 32 张 512×288 RGB。上述结构检查未执行真实权重前向推理，不能代替 NVIDIA 实机验收。

本机为 AMD 核显且当时可用提交内存不足，未运行普通权重加载、Radeon 加速、真实视频全流程或论文完整结果。Ubuntu/RTX 5080 仍需依次运行 `models check --load`、`perceive` 及上面的 `-m models` 测试，以实际推理报告和峰值显存为准。

2026-10-25 NVIDIA 适配后的 Windows CPU 离线回归：303 passed、3 skipped、1 deselected。`uv lock --check` 和 Linux x86_64 / Python 3.11 的安装预演通过，确认选择 `torch==2.8.0+cu128`、`torchvision==0.23.0+cu128`；两个公开镜像的固定提交元数据也重新通过官方大小/SHA256 比对。上述结果不包含 RTX 5080 实机安装或 GPU 视频推理。
