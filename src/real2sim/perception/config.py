"""Shared backend configuration; model inference is local unless selected otherwise."""

from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib import metadata as packages
import os
from pathlib import Path
import subprocess

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
MODEL_INFO = {
    "vggt": ("VGGT_OMEGA", "vggt_omega", "vggt_omega_1b_512.pt",
             "vggt-omega", "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6"),
    "sam3": ("SAM3", "sam3", "sam3.pt", "ultralytics",
             "f171c4b7aea56062d77061577e6b878b62e520fb"),
}

# Official Hugging Face revisions and LFS hashes, verified on 2026-10-24.
# These identify weights independently of the pinned inference-source commits.
MODEL_WEIGHTS = {
    "vggt": {
        "repo": "facebook/VGGT-Omega",
        "revision": "1041e80fc0e911235d3426b0a3d9a81075111a53",
        "size": 4576706117,
        "sha256": "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934",
    },
    "sam3": {
        "repo": "facebook/sam3",
        "revision": "3c879f39826c281e95690f02c7821c4de09afae7",
        "size": 3450062241,
        "sha256": "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e",
    },
}

# Public copies verified against MODEL_WEIGHTS on 2026-10-25. A mirror is
# selected explicitly; its contents must still match the official size/hash.
MODEL_MIRRORS = {
    "vggt": {
        "repo": "1kaiser/vggt-omega-jax",
        "revision": "a8c3a718e0cf78e9e4c6847229efea793d37f060",
    },
    "sam3": {
        "repo": "cubert-gmbh/sam3",
        "revision": "6d25af14a085ff9d3e1342c35bae7c87de4811f4",
    },
}


def backend_settings(name: str) -> dict:
    load_dotenv(ROOT / ".env", override=False)
    prefix, folder, filename, _, _ = MODEL_INFO[name]
    kind = os.getenv(f"{prefix}_BACKEND", "local")
    if kind not in ("local", "remote"):
        raise ValueError(f"unsupported {prefix}_BACKEND: {kind}")
    directory = Path(os.getenv(f"{prefix}_MODEL_DIR", f"data/models/{folder}")).expanduser()
    if not directory.is_absolute():
        directory = ROOT / directory
    settings = {"backend": kind, "model_dir": directory.resolve(),
                "checkpoint": directory.resolve() / filename,
                "device": os.getenv(f"{prefix}_DEVICE", "cuda"),
                "host": None, "port": None}
    if kind == "remote":
        host, port = os.getenv(f"{prefix}_HOST"), os.getenv(f"{prefix}_PORT")
        if not host or not port:
            raise ValueError(f"{prefix}_HOST and {prefix}_PORT must be configured for remote mode")
        if os.getenv(f"{prefix}_PROTOCOL", "tcp") != "tcp":
            raise ValueError(f"{prefix}_PROTOCOL must be tcp")
        try:
            port_number = int(port)
        except ValueError as exc:
            raise ValueError(f"{prefix}_PORT must be an integer") from exc
        if not 1 <= port_number <= 65535:
            raise ValueError(f"{prefix}_PORT must be within 1..65535")
        settings.update(host=host, port=port_number)
    return settings


def require_device(torch, device: str):
    selected = torch.device(device)
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is unavailable; install the CUDA environment")
        if selected.index is not None and selected.index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested CUDA device does not exist: {device}")
    elif selected.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested mps, but MPS is unavailable")
    elif selected.type != "cpu":
        raise ValueError(f"Unsupported model device: {device}")
    return selected


@lru_cache(maxsize=16)
def _file_digest(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_digest(checkpoint: Path) -> str:
    stat = checkpoint.stat()
    return _file_digest(str(checkpoint.resolve()), stat.st_size, stat.st_mtime_ns)


def dependency_versions() -> dict:
    versions = {}
    for name in ("numpy", "opencv-python", "torch", "torchvision", "vggt-omega",
                 "ultralytics", "timm", "clip", "mujoco", "pin", "robot-control"):
        try:
            versions[name] = packages.version(name)
        except packages.PackageNotFoundError:
            versions[name] = None
    return versions


def local_metadata(name: str, checkpoint: Path, device: str) -> dict:
    _, folder, _, distribution, pinned_revision = MODEL_INFO[name]
    source = ROOT / "external" / ("ultralytics" if name == "sam3" else folder)
    source_revision = None
    if (source / ".git").exists():
        result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False, encoding="utf-8", errors="replace")
        if result.returncode == 0:
            source_revision = result.stdout.strip()
    try:
        version = packages.version(distribution)
    except packages.PackageNotFoundError:
        version = None
    checksum = checkpoint_digest(checkpoint)
    weight = MODEL_WEIGHTS[name]
    matches = checksum == weight["sha256"]
    return {"backend": "local", "model": name, "model_version": version,
            "source_revision": source_revision, "pinned_revision": pinned_revision,
            "checkpoint_sha256": checksum,
            "checkpoint_repository": weight["repo"] if matches else None,
            "checkpoint_revision": weight["revision"] if matches else None,
            "checkpoint_matches_official": matches, "device": device,
            "preprocessing_version": "aligned-rgb-grid-v1"}


def backend_manifest(service, name: str) -> dict:
    details = dict(getattr(service, "metadata", {}) or {})
    details.setdefault("backend", "remote" if type(service).__name__.startswith("Remote") else "injected")
    details.setdefault("model", name)
    return details


def release_backend(service) -> None:
    close = getattr(service, "close", None)
    if close is not None:
        close()


def create_vggt():
    settings = backend_settings("vggt")
    if settings["backend"] == "local":
        from .vggt_omega.local_inference import LocalVGGT
        return LocalVGGT(settings["model_dir"], device=settings["device"])
    from .vggt_omega.remote_client import RemoteVGGT
    return RemoteVGGT(settings["host"], settings["port"])


def create_sam3():
    settings = backend_settings("sam3")
    if settings["backend"] == "local":
        from .sam3.local_inference import LocalSAM3
        return LocalSAM3(settings["model_dir"], device=settings["device"])
    from .sam3.remote_client import RemoteSAM3
    return RemoteSAM3(settings["host"], settings["port"])


def sam3_endpoint() -> tuple[str | None, int | None]:
    settings = backend_settings("sam3")
    return settings["host"], settings["port"]


def create_instance_segmenter(host=None, port=None):
    from .sam3.instances import InstanceSegmenter
    if host:
        if port is None or not 1 <= int(port) <= 65535:
            raise ValueError("An explicit SAM3 host requires a port within 1..65535")
        from .sam3.remote_client import RemoteSAM3
        backend = RemoteSAM3(host, int(port), timeout=60)
    elif port is not None:
        raise ValueError("An explicit SAM3 port requires a host")
    else:
        backend = create_sam3()
    return InstanceSegmenter(backend)
