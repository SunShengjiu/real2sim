# Agentic RSR — Real-to-Sim

This checkout extracts the Real-to-Sim portion of Agentic RSR. Given a workspace video, a task description and a known robot model, it produces a metric MuJoCo scene with task geometry, collision and joint definitions, assets and physical-check reports.

```text
Video + task + robot model
  → VGGT-Ω cameras/depth + SAM3 robot masks
  → backprojection + Sim(3) robot registration + metric coordinates
  → Blender reconstruction + four-view rendering and evaluation
  → MuJoCo export + physical plausibility checks
  → scene.xml + assets/ + manifests/reports
```

| Component | Location |
| --- | --- |
| Local/remote perception | `src/real2sim/perception/` |
| Geometry, registration and initial Blender delivery | `src/real2sim/stage1/` |
| Feedback-guided scene refinement | `src/real2sim/stage2/` |
| MJCF export, semantics and physics probes | `src/real2sim/stage3/` |
| Reconstruction batch, ablation and inspection tools | `dev/scripts/` |

Policy development, randomized policy evaluation and real-robot deployment have been removed. The pinned `external/robot_control` dependency remains necessary for robot geometry and MuJoCo model assembly.

## Layout

`configs/` holds scene settings, `src/` the reconstruction code, `data/` all inputs/assets/weights/results, `external/` pinned dependencies, `docs/` documentation, and `dev/` scripts and tests. Commands below run from the repository root. See [the layout and migration map](PROJECT_STRUCTURE.md).

## Setup

Use Python 3.11. Offline tests, Blender rendering and MuJoCo checks have been verified on Windows CPU; the complete video-and-checkpoint workflow remains unverified. Linux x86_64 supports NVIDIA CUDA acceleration for perception. Install Blender separately, then clone this repository with its pinned dependencies:

```bash
git clone --recurse-submodules https://github.com/SunShengjiu/real2sim.git
cd real2sim
uv sync --locked --python 3.11 --extra local --extra sim --extra agent --extra dev
```

The `sim` extra supplies Stage 3 physics and robot-model dependencies. Prepare official checkpoints, Blender and Codex authentication using [LOCAL_INFERENCE.md](LOCAL_INFERENCE.md). Edit `configs/scenes/demo.yaml` to supply your captured video and task, then use the single workflow entry point:

```bash
uv run --no-sync real2sim check --config configs/scenes/demo.yaml
uv run --no-sync real2sim run --config configs/scenes/demo.yaml
uv run --no-sync real2sim resume --run data/runs/<run-id>
uv run --no-sync real2sim verify --run data/runs/<run-id>/delivery
```

The workflow records configuration, input/source/checkpoint fingerprints and stage status. Completed stages are verified before reuse; failed stages retry in a new attempt directory. The final delivery rewrites runtime resources to relative paths and is load-tested in MuJoCo. The demo configuration is a template: a verified video and checkpoint end-to-end run are not included. See [WORKFLOW.md](WORKFLOW.md) and [the repository structure](PROJECT_STRUCTURE.md).

## Verification

```bash
uv sync --locked --python 3.11 --extra sim --extra dev
uv run --no-sync python -m pytest dev/tests/unit dev/tests/integration -m 'not models' -ra
```

Production Stage 1 fits and scores the same 32 frames; its delivery remains `candidate_unverified`, `accepted=false`. Stage 2 compares against reconstructed pseudo-depth and retains the historical Codex path. Stage 3 physics is provisional without real measurements. Offline tests do not establish checkpoint inference or full workflow reproduction.

[简体中文与代码阅读入口](../README.md) · [Architecture](ARCHITECTURE.md) · [Data contracts](DATA_CONTRACT.md) · [Tests](../dev/tests/README.md) · [Third-party terms](THIRD_PARTY.md)

Original project attribution is preserved in [CITATION.cff](../CITATION.cff). Original code uses [Apache-2.0](../LICENSE); third-party sources, weights and robot assets retain their own terms. See [NOTICE](../NOTICE).
