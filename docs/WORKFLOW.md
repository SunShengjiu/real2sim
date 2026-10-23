# Real-to-Sim 完整运行流程

入口统一为 `real2sim`，输入一个场景 YAML，输出一个独立运行目录。`models` 检查感知环境，`perceive` 单独运行感知，`tools` 提供阶段诊断和历史实验；新任务使用下面这条主线。

## 1. 准备环境与输入

完整本地感知目标环境：Linux x86_64、NVIDIA CUDA、Python 3.11。Windows 可以运行配置与部分离线检查，完整模型/渲染链需另行验收。

```bash
git submodule update --init --recursive
uv sync --locked --python 3.11 --extra local --extra sim --extra agent --extra dev
```

另行安装 Blender，按 [LOCAL_INFERENCE.md](LOCAL_INFERENCE.md) 准备两个官方 checkpoint，并完成 Codex CLI 身份验证。`sim` extra 为 Stage 3 提供 MuJoCo 和机器人装配依赖。模型/代理调用需要相应访问权限，Agent 阶段会消耗模型额度。

示例配置为 `configs/scenes/demo.yaml`，它是输入模板，未随仓库附带已验收视频。请提供配置所指视频，或修改 `video` 指向自己的文件；至少包含 32 个可解码帧。采集时机器人保持 `configs/robots/piper.yaml` 中的固定姿态，相机移动。Franka 使用 `configs/robots/franka.yaml`。

YAML 中的文件路径相对于该 YAML 文件解析，模型目录同理；`.env` 可用于远端 TCP 地址和本机凭据。环境变量中的凭据不写入配置快照。不要随意修改机器人固定姿态：现有模板和配准路径只支持已声明姿态。

直接运行虚拟环境内的 `real2sim` 也会为子工具设置同一 Python 环境，无须先激活虚拟环境。Windows 本机可使用 `& .venv/Scripts/real2sim.exe ...`。当前本机示例已指向用户提供的 Piper 视频，并设置 `perception.device: cpu`；任务暂按采样画面中的“将急停按钮放入绿色收纳盒”填写。Linux NVIDIA 推理可显式改为 `cuda`。CPU 模式的 32 帧真实模型推理尚未验收。

## 2. 检查与运行

```bash
uv run --no-sync real2sim check --config configs/scenes/demo.yaml
uv run --no-sync real2sim run --config configs/scenes/demo.yaml
```

`check` 不创建输出、不下载权重、不发起 Agent 模型调用。它检查 Python/依赖、子模块提交及工作区、视频、固定机器人模板、Blender 和本地模型前置条件；Codex CLI 存在不等于已完成身份验证，认证由实际 Agent 调用确认。

`run` 先检查前置条件，再顺序执行：

1. Stage 1：VGGT-Ω/SAM3 → 多视角反投影 → 固定机器人表面 Sim(3) 配准 → 米制相机/深度和 `initial.blend`。默认 Piper 为 128 旋转、Top-3；Franka 为 48 旋转、Top-1。
2. Stage 2：Agent 规划、冻结机器人和四相机、Blender 建模、渲染及深度/颜色/覆盖率评价。未通过联合指标就停止，禁止继续导出。
3. Stage 3：语义分类、MuJoCo 导出和物理检查。`physics: agent` 执行原有物理审查/修复及独立复验；`physics: probes` 仅做静置和扰动响应检查，两者的证据强度不同，不等同真实参数标定。
4. Delivery：拷贝最终 XML 的全部引用资产，改为相对路径，并重新用 MuJoCo 加载，生成交付清单与哈希。

配置中的 `agent.model` 和 `agent.effort` 用于建模、语义分类和物理 Agent。`max_rounds` 控制 Stage 2 预算。数值评价阈值与提示词版本由源码及输入来源记录绑定；模型结果可能波动，不能保证不同运行生成逐字节相同的场景。

## 3. 运行目录与续跑

```text
data/runs/<场景名>_<北京时间>/
  config.json                   # 解析后的配置快照
  manifest.json                 # 输入、源码、模型、依赖与阶段状态
  stage1/attempt_001/            # scene/ 中是 Stage 1 标准交付
  stage2/attempt_001/            # plan/ 与 geometry/
  stage3/attempt_001/            # export/、physics/、views/
  delivery/                     # scene.xml、assets/、reports/、manifest.json
```

```bash
uv run --no-sync real2sim resume --run data/runs/<运行目录>
uv run --no-sync real2sim verify --run data/runs/<运行目录>
uv run --no-sync real2sim verify --run data/runs/<运行目录>/delivery
```

续跑检查配置、视频/权重、源码、机器人模板、依赖版本和子模块来源，以及已完成阶段的全部产物哈希。已完成阶段保持不动；失败阶段从新的 `attempt_002` 重新执行，保留失败诊断。当前不会接着某个失败阶段内部的优化轮次继续；Stage 2 的细粒度轮次恢复仍可通过原有阶段命令使用。

配置、模型或源码变化后应使用新运行目录。并发操作同一运行目录会被锁阻止；异常终止留下 `.running.lock` 时，先确认原进程已停止再人工清理。运行目录保存原始绝对来源用于续跑；可独立搬移的是最终 `delivery/`。

## 4. 交付与验收

整体复制 `delivery/` 到其他位置后，再运行 `real2sim verify --run <新的交付目录>`。清单记录文件哈希，XML 只使用交付包内相对资源路径。`reports/` 保存历史检查证据，其中的来源路径只是记录，不用于加载场景。

Stage 1 当前仍为 `candidate_unverified`：全部 32 帧参与拟合和评分，没有独立精度验收。Stage 2 深度参考来自重建。Stage 3 无真实测量时最多为 `provisional`。运行完成、数值指标通过、真实参数独立验证是不同结论，统一入口不会改写这些边界。

离线测试见 [dev/tests/README.md](../dev/tests/README.md)。真实全流程测试只有显式设置 `RSR_E2E_CONFIG` 才会运行，并会使用真实权重、Blender 及付费 Agent 调用；跳过不能记作复现成功。
