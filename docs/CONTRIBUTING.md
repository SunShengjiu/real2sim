# Contributing

This checkout is scoped to Real-to-Sim scene reconstruction. Keep changes within perception, robot registration, Blender scene refinement, MuJoCo export and physical checks. Read [the workflow](WORKFLOW.md), [data contracts](DATA_CONTRACT.md) and [third-party terms](THIRD_PARTY.md).

Use a source checkout with pinned recursive submodules and Python 3.11. Linux CUDA is the complete local perception target; offline checks do not need weights.

```bash
uv sync --locked --python 3.11 --extra sim --extra dev
uv run --no-sync python -m pytest dev/tests/unit dev/tests/integration -m 'not models' -ra
git diff --check
```

Report passed/skipped checks and missing assets. Blender, CUDA, checkpoints and task scenes have separate validation requirements. Preserve input fingerprints, coordinate conventions, model versions, frozen robot/camera constraints and honest acceptance states.

Write new experiment outputs into independent directories. Do not commit credentials, weights, user recordings or generated caches. Keep third-party source, asset licenses and notices with redistributed material. Original code uses [Apache-2.0](../LICENSE).
