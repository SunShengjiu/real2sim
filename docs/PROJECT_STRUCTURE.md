# Real-to-Sim 仓库布局

顶层按使用职责归为六个目录：日常运行主要修改 `configs/` 和准备 `data/`，代码在 `src/`，开发辅助内容集中到 `dev/`。

```text
agentic_rsr/
├── README.md                     # 中文使用入口
├── pyproject.toml / uv.lock       # 包定义、命令和依赖锁
├── configs/
│   ├── robots/{piper,franka}.yaml # 固定机器人、姿态及配准设置
│   └── scenes/demo.yaml          # 视频、任务、模型与迭代预算
├── src/real2sim/
│   ├── cli.py                    # check / run / resume / verify / models / perceive / tools
│   ├── config.py                 # 配置校验、路径解析、快照
│   ├── contracts.py              # 清单、哈希、可搬移交付包
│   ├── pipeline.py               # 三阶段调度、失败阻断与恢复
│   ├── blender.py                # Blender 可执行文件发现
│   ├── perception/               # VGGT-Ω / SAM3 本地与远端适配器
│   ├── stage1/                   # 感知、Sim(3) 配准、米制化与初版 Blender 交付
│   ├── stage2/                   # Blender 建模、渲染和视觉评价
│   └── stage3/                   # 语义、MuJoCo 导出和物理检查
├── data/
│   ├── raw/                      # 原始视频
│   ├── assets/robot_surfaces/    # 固定姿态机器人表面模板与来源
│   ├── models/                   # VGGT-Ω / SAM3 权重与缓存
│   ├── processed/                # 历史单阶段数据
│   └── runs/                     # 新运行的尝试、日志、清单和 delivery/
├── external/                     # 固定第三方子模块及嵌套机器人资产
├── docs/                         # 工作流、方法、数据契约和第三方说明
│   ├── README.en.md              # 英文使用说明
│   └── CONTRIBUTING.md           # 开发约定
└── dev/
    ├── scripts/
    │   ├── batch/                # 历史多场景批处理
    │   ├── experiments/          # 消融、SSIM 与实验导出
    │   ├── maintenance/          # 历史结果整理、归档与附件修复
    │   └── visualization/        # 点云和物理探针可视化
    └── tests/                    # unit / integration / end_to_end / fixtures
```

`.venv/` 保留在根目录，维持本机已安装的 Python 和 Blender 环境。`.env`、Git 元数据、许可证、署名及依赖文件按各自工具要求留在根目录。

## 原目录去向

| 原位置 | 当前位置 |
| --- | --- |
| `asset/` | `data/assets/` |
| `model/` | `data/models/` |
| `runs/` | `data/runs/` |
| `scripts/` | `dev/scripts/` |
| `tests/` | `dev/tests/` |
| `README.zh-CN.md` | `README.md` |
| 原英文 `README.md` | `docs/README.en.md` |
| `CONTRIBUTING.md` | `docs/CONTRIBUTING.md` |

源码默认路径、场景 YAML、测试发现规则、辅助脚本、`.env` 中的已知模型目录及忽略规则均使用新位置。迁移保留已有文件内容，机器人表面模板与第三方依赖的来源哈希不变。

`pyproject.toml` 只安装 `real2sim` 一个命令。主流程使用 `check/run/resume/verify`；`models` 检查感知环境，`perceive` 单独运行感知，`tools` 提供阶段诊断。辅助脚本见 [开发脚本说明](../dev/scripts/README.md)，测试见 [测试说明](../dev/tests/README.md)。

新运行写入 `data/runs/<场景名>_<时间>/`。其中 `delivery/` 是独立可搬移的最终场景、资产和报告；完整运行目录保留本机来源路径用于续跑。Stage 1 的共享配准和交付分别位于 `refinement.py`、`delivery.py`、`delivery_worker.py`；旧版诊断与历史实验不由主流程自动执行。

源码按 checkout + editable 安装使用。策略开发、随机化策略评测和真机部署已从主项目移除；保留的原项目媒体仅用于来源记录。
