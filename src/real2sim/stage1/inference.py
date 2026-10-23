"""Aligned video sampling, perception backends and reconstruction validation."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from real2sim.perception.config import backend_manifest, create_sam3, create_vggt

from real2sim.stage1.registration import camera_world, unproject


ROOT = Path(__file__).resolve().parents[3]


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(video: Path, folder: Path) -> tuple[list[Path], list[int]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"unable to read video: {video}")
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 32:
            raise ValueError(f"video needs at least 32 frames, found {count}")
        indices = np.linspace(0, count - 1, 32).astype(int).tolist()
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for order, index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"video frame {index} cannot be decoded")
            height, width = frame.shape[:2]
            scale = min(512 / width, 288 / height)
            scaled = cv2.resize(frame, (round(width * scale), round(height * scale)))
            canvas = np.zeros((288, 512, 3), dtype=np.uint8)
            top = (288 - scaled.shape[0]) // 2
            left = (512 - scaled.shape[1]) // 2
            canvas[top:top + scaled.shape[0], left:left + scaled.shape[1]] = scaled
            destination = folder / f"{order:02d}.jpg"
            if not cv2.imwrite(str(destination), canvas):
                raise OSError(f"unable to save frame {index}")
            paths.append(destination)
        return paths, indices
    finally:
        capture.release()


def backends() -> tuple[object, object, dict]:
    vggt, sam3 = create_vggt(), create_sam3()
    return vggt, sam3, {"vggt": backend_manifest(vggt, "vggt"),
                        "sam3": backend_manifest(sam3, "sam3")}


def validate(arrays: dict, scale: float, *, metric_range: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = arrays["depth"].astype(np.float64) * scale
    confidence = arrays["depth_conf"]
    intrinsics, extrinsics = arrays["intrinsics"], arrays["extrinsics"].astype(np.float64).copy()
    extrinsics[:, :, 3] *= scale
    if (depth.shape != (32, 288, 512) or confidence.shape != depth.shape or
            intrinsics.shape != (32, 3, 3) or extrinsics.shape != (32, 3, 4)):
        raise ValueError("reconstruction shapes do not match the 32 aligned 512x288 RGB images")
    if (not np.isfinite(depth).all() or not np.isfinite(confidence).all() or
            not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all()):
        raise ValueError("nonfinite reconstruction")
    if metric_range:
        if np.count_nonzero((depth > 0.05) & (depth < 5)) / depth.size < 0.3:
            raise ValueError("insufficient plausible positive depth at the declared metric scale")
    elif np.count_nonzero(depth > 0) / depth.size < 0.3:
        raise ValueError("insufficient positive depth in VGGT reconstruction")
    for intrinsic, extrinsic in zip(intrinsics, extrinsics):
        rotation = extrinsic[:, :3]
        if (intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or
                not 0.5 < intrinsic[0, 0] / intrinsic[1, 1] < 2 or
                not 0 <= intrinsic[0, 2] < 512 or not 0 <= intrinsic[1, 2] < 288 or
                not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-3) or
                not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-2) or
                not np.isclose(np.linalg.det(rotation), 1, atol=1e-2)):
            raise ValueError("invalid camera intrinsics or world-to-camera extrinsics")
    return depth.astype(np.float32), intrinsics, extrinsics


def pointcloud(depth: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    points = unproject(depth, intrinsic, np.ones_like(depth, dtype=bool), stride=3)
    matrix = camera_world(extrinsic)
    return (points @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)
