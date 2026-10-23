# Perception

VGGT-Ω 和 SAM3 的本地/远端适配器位于对应子目录。`config.py` 为 Stage 1、物体评估和提示词试验创建统一后端；默认本地 CUDA 推理，远端需显式选择。权重不自动下载，加载失败不切换后端。

运行 `real2sim models check` 检查源码、依赖、权重和设备；`--load` 逐个加载并释放模型。`real2sim perceive` 对一段用户提供的视频生成独立的深度、相机、mask、可视化及资源使用报告。详见 `docs/LOCAL_INFERENCE.md`。

获得官方模型访问权限并用 `hf auth login` 登录后，显式执行 `real2sim models download` 下载两份固定版本权重。下载与检查均校验官方 SHA256；推理本身不自动下载。源码与权重分别管理在 `external/` 和 `data/models/`。
