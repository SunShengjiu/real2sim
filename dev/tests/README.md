# Real-to-Sim Tests

Run from the repository root. Pytest adds `src` and the checkout root to its import path.

Tests live under `dev/tests/`; `pyproject.toml` sets this as the default discovery directory, so a root-level `python -m pytest` also finds this suite.

```bash
git submodule update --init --recursive
uv sync --locked --python 3.11 --extra sim --extra dev
uv run --no-sync python -m pytest dev/tests/unit dev/tests/integration -m 'not models' -ra
```

The retained suite covers perception contracts, backprojection/registration, metric coordinates, Blender evaluation/delivery, MJCF export, semantics and physical probes. Policy, randomized-policy and real-robot runtime tests have been removed.

`sim` supplies Stage 3 MuJoCo/Pillow and compatible robot-model dependencies. Tests loading robot meshes need initialized nested assets. Blender tests use `BLENDER_BIN` or PATH; a missing Blender produces an explained skip. Some historical physics tests require generated scenes under ignored `data/runs/` or `data/processed/` paths. Missing scenes and skips do not establish a successful reconstruction.

## Real checkpoint inference

Follow [LOCAL_INFERENCE.md](../../docs/LOCAL_INFERENCE.md) to install local model dependencies and official weights on Linux CUDA, then supply a video with at least 32 frames:

```bash
export RSR_MODEL_TEST_VIDEO=/path/to/task.mp4
uv run --no-sync python -m pytest dev/tests/integration -m models -ra
```

This opt-in test records model provenance and memory. It never downloads weights or silently selects remote inference. A missing video or CUDA produces a skip, not inference acceptance.

| Directory | Scope |
| --- | --- |
| `unit/` | Geometry, scoring, evaluation, manifests, semantics and physics checks. |
| `integration/` | Video inference with test doubles, optional real inference and Blender delivery. |
| `end_to_end/` | Opt-in real video-to-delivery workflow using checkpoints and paid agent calls. |
| `fixtures/` | Small retained Stage 3 physics fixtures. |
| `contract/` | Documentation; current backend contracts are exercised in unit tests. |

No hosted test workflow is included. Headless Linux rendering can use EGL/OSMesa; macOS rendering can require `mjpython`. Offline checks do not validate full checkpoint inference or all paper results.

## Unified pipeline and full workflow

`test_config.py` and `test_pipeline.py` exercise configuration validation, unified command dispatch, child-process Python selection, stage ordering, failure gates, input/artifact changes, resume and portable asset bundling using stage doubles. These tests do not execute neural models or establish reconstruction quality.

The real video-to-scene test is opt-in and invokes paid agent turns:

```bash
export RSR_E2E_CONFIG=/absolute/path/to/reviewed_scene.yaml
uv run --no-sync python -m pytest dev/tests/end_to_end -m end_to_end -ra
```

It runs all real stages, checks the completed run, copies the final delivery and reloads it at another location. An absent config produces an explicit skip. The input video, checkpoints, initialized robot assets, Blender and an authenticated model account are required.
