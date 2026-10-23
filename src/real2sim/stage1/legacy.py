"""Historical externally-scaled Piper diagnostic; the main workflow uses URDF Sim(3)."""

import argparse
import json
import subprocess
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from real2sim.blender import resolve_blender
from real2sim.perception.config import backend_manifest, release_backend
from .geometry import ASSET_ROOT, visual_meshes
from .inference import ROOT, backends, extract, pointcloud, save_json, sha256, validate
from .registration import REFERENCE, register

THRESHOLDS = Path(__file__).with_name("thresholds.json")


def run(video: Path, task: str, output: Path, *, metric_scale: float | None = None,
        vggt=None, sam3=None, blender: str | None = None) -> tuple[Path, bool]:
    if not task.strip():
        raise ValueError("task description is required")
    if metric_scale is not None and (not np.isfinite(metric_scale) or metric_scale <= 0):
        raise ValueError("metric scale must be positive and finite")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    folder = output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
    folder.mkdir()
    manifest = {"schema": "stage1-v1", "status": "running", "task": task,
                "video": str(video.resolve()), "metric_scale_m_per_unit": metric_scale,
                "scale_source": "user-supplied external metric calibration" if metric_scale is not None else "unknown; candidate only",
                "run_id": folder.name, "thresholds": json.loads(THRESHOLDS.read_text(encoding="utf-8")),
                "artifacts": {}, "robot": "piper", "coordinates": "VGGT world, OpenCV camera (+X right, +Y down, +Z forward)"}
    save_json(folder / "manifest.json", manifest)
    try:
        manifest["video_sha256"] = sha256(video)
        manifest["robot_asset_revision"] = subprocess.run(
            ["git", "-C", str(ROOT / "external/robot_control"), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.strip()
        manifest["robot_description_sha256"] = sha256(ASSET_ROOT / "piper/urdf/piper_with_gripper_description.xacro")
        frames, indices = extract(video, folder / "frames")
        manifest["source_frame_indices"] = indices
        if vggt is None or sam3 is None:
            configured_vggt, configured_sam3, selection = backends()
            vggt = vggt or configured_vggt
            sam3 = sam3 or configured_sam3
            manifest["models"] = selection
        else:
            manifest["models"] = {"vggt": {"backend": "injected"}, "sam3": {"backend": "injected"}}
        arrays = vggt.infer(frames)
        manifest["models"]["vggt"].update(backend_manifest(vggt, "vggt"))
        manifest["models"]["vggt"]["response"] = getattr(vggt, "metadata", {})
        release_backend(vggt)
        np.savez_compressed(folder / "raw_vggt.npz", **arrays)
        depth, intrinsics, extrinsics = validate(arrays, metric_scale if metric_scale is not None else 1.0,
                                                metric_range=metric_scale is not None)
        np.savez_compressed(folder / "reconstruction.npz", depth=depth, intrinsics=intrinsics,
                            extrinsics=extrinsics, depth_conf=arrays["depth_conf"])
        masks = [np.asarray(sam3.segment(frame), dtype=bool) for frame in frames]
        manifest["models"]["sam3"].update(backend_manifest(sam3, "sam3"))
        manifest["models"]["sam3"]["response"] = getattr(sam3, "metadata", {})
        release_backend(sam3)
        if any(mask.shape != (288, 512) for mask in masks):
            raise ValueError("SAM3 mask does not align with RGB/depth")
        np.savez_compressed(folder / "robot_masks.npz", masks=np.stack(masks))
        mesh_rows, model_points = visual_meshes(folder / "meshes")
        pose, report, projections = register(model_points, depth, intrinsics, extrinsics,
                                               masks, {key: value for key, value in manifest["thresholds"].items() if key != "version"})
        report["threshold_version"] = manifest["thresholds"]["version"]
        report["metric_scale_verified"] = metric_scale is not None
        report["depth_unit"] = "meters" if metric_scale is not None else "unverified VGGT units"
        report["accepted"] = report["accepted"] and metric_scale is not None
        report["status"] = "passed" if report["accepted"] else "failed"
        if metric_scale is None:
            report["rejection_reason"] = "VGGT monocular reconstruction has no independently calibrated metric scale"
        save_json(folder / "registration.json", report)
        for slot, index in enumerate(REFERENCE):
            prefix = folder / f"view_{slot:02d}"
            (prefix.with_suffix(".jpg")).write_bytes(frames[index].read_bytes())
            np.save(prefix.with_suffix(".depth.npy"), depth[index])
            np.save(prefix.with_suffix(".cloud.npy"), pointcloud(depth[index], intrinsics[index], extrinsics[index]))
            overlay = cv2.imread(str(frames[index]))
            overlay[projections[slot]] = (0.5 * overlay[projections[slot]] + np.array([0, 0, 127])).astype(np.uint8)
            cv2.imwrite(str(prefix.with_suffix(".projection.png")), overlay)
        spec = {"width": 512, "height": 288, "intrinsics": intrinsics[list(REFERENCE)].tolist(),
                "extrinsics": extrinsics[list(REFERENCE)].tolist(), "world_from_base": pose.tolist(), "meshes": mesh_rows}
        save_json(folder / "scene_spec.json", spec)
        executable = resolve_blender(blender)
        process = subprocess.run([str(executable), "-b", "-noaudio", "--python", str(Path(__file__).with_name("legacy_blender_scene.py")),
                                  "--", str(folder / "scene_spec.json"), str(folder / "initial.blend")],
                                 capture_output=True, text=True, timeout=300, encoding="utf-8", errors="replace")
        (folder / "blender.log").write_text(process.stdout + "\n" + process.stderr, encoding="utf-8")
        if (process.returncode or "Traceback (most recent call last)" in process.stderr or
                not (folder / "initial.blend").is_file() or
                any(not image.is_file() for image in [folder / "initial.smoke.png"] +
                    [folder / f"view_{index:02d}.smoke.png" for index in range(1, 4)])):
            raise RuntimeError(f"Blender scene validation failed: exit code {process.returncode}; see blender.log")
        manifest["status"] = report["status"]
        manifest["artifacts"] = {"scene": "initial.blend", "registration": "registration.json",
                                  "reconstruction": "reconstruction.npz", "masks": "robot_masks.npz",
                                  "scene_spec": "scene_spec.json", "held_out_views": list(REFERENCE),
                                  "blender_smoke": ["initial.smoke.png"] +
                                                   [f"view_{index:02d}.smoke.png" for index in range(1, 4)]}
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if not (folder / "registration.json").exists():
            save_json(folder / "registration.json", {"status": "failed", "accepted": False,
                                                     "reason": str(exc), "thresholds": manifest["thresholds"]})
        (folder / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        release_backend(vggt)
        release_backend(sam3)
        save_json(folder / "manifest.json", manifest)
    return folder, manifest["status"] == "passed"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build and validate a Piper Real2Sim Stage 1 candidate")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blender", help="Blender executable path or command; defaults to BLENDER_BIN or PATH")
    parser.add_argument("--metric-scale", type=float, default=None,
                        help="Externally calibrated meters per VGGT unit; omit to force rejection")
    args = parser.parse_args(argv)
    folder, passed = run(args.video, args.task, args.output,
                         metric_scale=args.metric_scale, blender=args.blender)
    print(f"Stage 1 {'passed' if passed else 'failed'}: {folder} (see manifest.json)")
    if not passed:
        parser.exit(1)


if __name__ == "__main__":
    main()
