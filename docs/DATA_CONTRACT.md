# Unified Run Contract

The main CLI stores one resolved `config.json`, one run `manifest.json`, and independent stage attempt directories under `data/runs/<run-id>/`. Completed stages have relative attempt paths and SHA256 inventories. Resume rejects changed sources, inputs, weights, dependency versions or completed artifacts. Failed attempts are preserved and retried as new attempts.

Stage 1 delivery declares `robot`, `scene_id` and `task_description`; Stage 2 delivery declares the robot and task. Readers retain historical layout-based fallback. `delivery/manifest.json` inventories a portable scene bundle with relative runtime asset paths. Historical evidence under `reports/` may retain absolute source paths, which are not used to load the scene.

The detailed stage formats below also document the older readers and outputs.

# Data Contracts

This document describes the current public implementation, including the
historical formats still supported by the readers. Paths in JSON may be absolute
and tied to the checkout that produced the run. Moving an artifact requires
preserving its associated files and resolving recorded paths; copying a single
manifest or XML does not make it self-contained.

## Capture And Pixel Grid

The input is a video containing at least 32 decodable frames and a task
description. Frame indices are `linspace(0, frame_count - 1, 32).astype(int)`;
`source_frame_indices` records the actual video indices. Each frame is resized
with preserved aspect ratio and centered black padding onto a 512 by 288 grid,
then saved as JPEG. Thus the model pixel grid is the saved frame grid, not
necessarily the original video's grid. RGB, depth, confidence, masks and camera
intrinsics must agree on this grid.

References use sampled frame indices `[0, 10, 21, 31]`, saved as
`reference/frame_00.jpg` through `frame_03.jpg`. Reference ordinals 0..3 and sampled
frame indices 0/10/21/31 are different identifiers. In the production 48/128
registration paths the references are also included in fitting/scoring; they are
not held-out measurements.

## Stage 1 Perception Cache

```text
<processed>/<robot>/stage1/<task>/
  manifest.json                    # schema: stage1-inference-v1
  frames/00.jpg ... 31.jpg
  depth/depth_00.npy ... depth_31.npy
  sam3_masks/mask_00.npy ... mask_31.npy
  depth_conf.npz                  # depth_conf key
  camera.json
  reference/frame_00.jpg ... frame_03.jpg
```

| Field/array | Shape and semantics |
| --- | --- |
| Depth (per frame) | `(288, 512)`, normally float32, camera-axis Z depth in unverified VGGT units |
| `depth_conf` | `(32, 288, 512)`, finite model confidence values; not calibrated probabilities |
| Robot mask (per frame) | `(288, 512)`, boolean union of SAM3 instances |
| `camera.json: intrinsics` | `(32, 3, 3)`, pixel intrinsics `K` on the saved frame grid |
| `camera.json: extrinsics` | `(32, 3, 4)`, `[R|t]`, world-to-camera in VGGT units |
| Camera axes | OpenCV: +X right, +Y down, +Z forward |

`manifest.json` records `robot`, `source_video`, `video_sha256`, actual frame
indices, robot prompt and `models` metadata. Backend metadata identifies local or
explicit remote use; local runs record source revision, checkpoint SHA256,
device and preprocessing version. Failure writes `status=failed` and diagnostics;
`inference_complete` only confirms perception output, not metric scale or robot
registration. Empty masks are recorded in `empty_mask_frames` and can block
subsequent registration even when inference completed.

Current batch cache eligibility also checks the robot prompt. A historical cache
with `robotic arm` can contain valid arrays while being ineligible for reuse under
the current `robotic arm with base and camera` prompt. Preserve its real prompt
and source provenance; do not relabel historical outputs as fresh inference.

SAM3's internal instance interface returns `(K,H,W)` boolean masks and `(K,)`
scores on the requested image grid. No detection is `K=0`. Stage 1 discards
instance identity by unioning the masks; old union-only caches cannot reconstruct
individual scores. Object evaluation can select a single instance before depth backprojection.

## Coordinates And Metric Anchoring

For VGGT-world point `p_G`, the camera point is `p_C = R p_G + t`. A valid depth
pixel `(u,v,d)` backprojects as:

```text
p_C = [(u - cx) * d / fx, (v - cy) * d / fy, d]
p_G = R.T @ (p_C - t)
```

Robot registration fits `p_G = s R_BG p_B + t_BG`, where `p_B` is a robot-base
point in meters and `s` is `scale_vggt_units_per_meter`. Its inverse is:

```text
p_B = R_BG.T @ (p_G - t_BG) / s
depth_m = depth_vggt / s
meters_per_vggt_unit = 1 / s
```

The official delivered camera grid keeps the original intrinsics. Camera
translations are scaled and the world is rebased to the robot root; rotations
change with that rebase. Do not scale the robot mesh, change focal lengths, or
stretch depth to hide a grid mismatch. Intermediate registration camera caches
can be in the VGGT-oriented metric world; only the delivered
`camera_metric.json` declares the robot-base delivery frame. Inspect
`metric_calibration` and its coordinate version before comparing poses.

## Stage 1 Registration And Delivery

Initialization outputs contain `initialization.json`, ranked SO(3) candidates,
surface/video/mask provenance and visual checks. Refinement outputs contain
`result.json`, selected seed, multiscale and rotation history, metric depth and
camera caches. Generated surface templates depend on the fixed joint pose and
the robot-control assets; changing either invalidates their provenance.

Delivery adds these files to the perception cache:

| Artifact | Meaning |
| --- | --- |
| `depth_metric/depth_XX.npy` | All 32 depths in meters, unchanged pixel grid |
| `camera_metric.json` | All 32 cameras in the meter-scale robot-base world |
| `registration.json` | Compact fitting scores, scale, source hashes and coordinate provenance |
| `stage1_delivery.json` | Cross-stage input manifest and delivery status |
| `initial.blend` | Fixed robot visuals and four reference cameras; reference images packed |
| `reference/overlay_*.png`, `quality_check.png` | Robot-mask and projected-robot review images |

The root robot has identity world transform in official delivered scenes.
`stage1_delivery.json` and registration remain `candidate_unverified`,
`accepted=false`, because production fitting and scoring share all 32 frames.
Do not infer acceptance from `initial.blend` existing.

The legacy single-video `real2sim tools stage1-legacy` path uses `schema=stage1-v1`,
`reconstruction.npz`, `robot_masks.npz`, `view_XX.*` and `scene_spec.json` in a
unique run directory. It has separate metric-scale and held-out checks. Its
`manifest.json` is not interchangeable with the production
`stage1_delivery.json` layout consumed by phased Stage 2.

## Stage 2 Blender Delivery

One run contains `plan/plan.md`, phase/session logs, a frozen robot/camera
signature and `geometry/round_NNN/` candidate/edit/render/evaluation artifacts.
`run_manifest.json` indexes the run. The delivery reader for Stage 3 consumes
`geometry/stage2_manifest.json` (or its enclosing task directory), requiring
`accepted=true`, `evaluation.passed=true` and a matching `scene_sha256` for
`geometry/final.blend`.

The final manifest references the scene, plan/input provenance and four evaluated
RGB views. New deliveries declare `robot` and `task_description` explicitly. Historical
deliveries without these fields derive the robot from the directory layout `<root>/<robot>/stage2/<task>/geometry/stage2_manifest.json`; preserve this
layout even inside a new `data/runs/` root. Each evaluation records depth MAE, Lab Delta E76 and coverage per
reference view, with the exact thresholds. A mean-depth threshold alone does not
prove joint acceptance. The reference depth is a reconstructed pseudo-reference,
not independently measured ground truth. SSIM and object point-cloud metrics
are additional analysis and do not change the delivery gate.

`geometry/input_provenance.json` binds a resumable run to the Stage 1 Blend and
manifest, four reference RGB/depth files, plan, evaluation settings and
renderer/evaluator source hashes. It also records the prepared scene, inspection
and baseline render hashes. Each completed `round_report.json` records the
candidate/render hashes in `artifact_sha256`. Resume verifies those hashes and
recomputes metrics; stale report flags cannot establish acceptance. Missing or
changed provenance requires a new run directory. Existing delivery readers
still accept historical manifests; this requirement applies to continuation.

## Stage 3 Scene Bundle: Real-to-Sim Output

Stage 3 preserves the accepted Blend source and exports `scene.xml`,
`assets/meshes/task/`, robot assets, extracted image resources,
`object_manifest.json`, `camera_manifest.json` and `export_manifest.json`.
`semantics.json` ties visual/static/dynamic/operable hypotheses to the source
Blend hash. Four cameras become `stage3_camera_0` through `stage3_camera_3`.
MJCF lengths are meters; action/state quaternions use `[w,x,y,z]`.

An agent physics run preserves candidate XMLs, probe reports, task context and
session evidence separately. A final scene can be a repaired candidate rather
than the first export; read `final_manifest.json` or the batch's `final_scene`
instead of guessing from a `mujoco_vN/` directory. A physical status of
`provisional` is a simulation conclusion, not measured calibration.

The Real-to-Sim workflow ends with this scene bundle and its physical-check
evidence. Keep the XML together with its referenced meshes, textures and
manifests when moving it. Policy generation, policy evaluation and real-robot
execution are outside this checkout's scope.

Weights, credentials, cached model files, raw task recordings and local run
artifacts are ignored and must not be added merely to make recorded absolute
paths resolve for another user. See [WORKFLOW.md](WORKFLOW.md) for preparation
commands and [THIRD_PARTY.md](THIRD_PARTY.md) for redistribution requirements.
