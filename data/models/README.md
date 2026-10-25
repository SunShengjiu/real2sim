# Local Models

本目录用于存放本地推理需要的模型权重、配置、缓存和编译产物。模型文件通常较大或受许可证限制，**不提交到 Git**。

```text
data/models/
├── vggt_omega/    # VGGT-Ω checkpoint/config/cache
├── sam3/          # SAM3 checkpoint/config/cache
└── cache/         # 可选的通用下载和编译缓存
```

本地推理适配代码分别位于：

- `src/real2sim/perception/vggt_omega/local_inference.py`
- `src/real2sim/perception/sam3/local_inference.py`

默认后端为 `local`。固定的 VGGT-Ω 与 Ultralytics 源码分别放在 `external/vggt_omega/` 和 `external/ultralytics/`，通过 checkout + editable 安装；源码和权重目录分开。`VGGT_OMEGA_MODEL_DIR`、`SAM3_MODEL_DIR` 可指向外部挂载。

默认 checkpoint 为 `vggt_omega/vggt_omega_1b_512.pt` 和 `sam3/sam3.pt`，合计约 8.03 GB。可显式选择公开镜像，无需 Hugging Face 登录：

```powershell
uv run --no-sync real2sim models download --source mirror
uv run --no-sync real2sim models check --load
```

下载器固定来源提交并校验官方大小和 SHA256。若选默认官方来源，先在两个官方模型页面申请访问，再通过 `hf auth login` 完成本机登录，执行 `models download`。公开镜像不携带 HF 凭据，原模型许可仍适用。源码继续以 `external/` 中的 Git 子模块管理；多 GB 权重留在本目录并由 Git 忽略。推理不隐式下载或切换远端，真实推理另行验收。

详见[本地安装与最小示例](../../docs/LOCAL_INFERENCE.md)和[第三方来源与许可](../../docs/THIRD_PARTY.md)。
