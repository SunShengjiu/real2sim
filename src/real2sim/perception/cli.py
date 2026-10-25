"""Check local models or explicitly download and verify the pinned official weights."""

from __future__ import annotations

import argparse
import importlib.util
import json

from .config import (MODEL_INFO, MODEL_MIRRORS, MODEL_WEIGHTS, ROOT, backend_settings, checkpoint_digest,
                     create_sam3, create_vggt, local_metadata, release_backend, require_device)


def check_models(names, *, load=False) -> dict:
    records = {}
    for name in names:
        errors = []
        service = None
        record = {"errors": errors}
        records[name] = record
        try:
            settings = backend_settings(name)
            record.update(backend=settings["backend"])
            if settings["backend"] == "remote":
                record["status"] = "configured_not_contacted"
                if load:
                    raise ValueError("--load checks local weights; select the local backend")
                continue
            checkpoint = settings["checkpoint"]
            record.update(device=settings["device"], checkpoint=str(checkpoint))
            if not checkpoint.is_file():
                errors.append(f"Checkpoint missing: {checkpoint}")
            distribution = MODEL_INFO[name][3]
            import_name = "vggt_omega" if name == "vggt" else distribution
            dependencies = ("torch", "torchvision", import_name)
            if name == "sam3":
                dependencies += ("clip", "timm")
            for dependency in dependencies:
                if importlib.util.find_spec(dependency) is None:
                    errors.append(f"Dependency missing: {dependency}; follow docs/LOCAL_INFERENCE.md")
            source = ROOT / "external" / ("vggt_omega" if name == "vggt" else "ultralytics")
            if not (source / ".git").exists():
                errors.append(f"Pinned source is not initialized: {source}")
            installed = importlib.util.find_spec(import_name)
            if installed is not None and installed.origin:
                from pathlib import Path
                installed_source = Path(installed.origin).resolve()
                record["installed_source"] = str(installed_source)
                if not installed_source.is_relative_to(source.resolve()):
                    errors.append(f"{import_name} is not installed from the pinned editable checkout")
            if importlib.util.find_spec("torch") is not None:
                import torch
                record.update(torch_version=torch.__version__, cuda_version=torch.version.cuda)
                selected = require_device(torch, settings["device"])
                if selected.type == "cuda":
                    properties = torch.cuda.get_device_properties(selected)
                    free, total = torch.cuda.mem_get_info(selected)
                    record["gpu"] = {"name": properties.name,
                                     "compute_capability": [properties.major, properties.minor],
                                     "free_bytes": free, "total_bytes": total}
                    # CUDA availability alone does not prove that this wheel
                    # contains kernels for the selected GPU (e.g. RTX 5080).
                    torch.ones(1, device=selected).add_(1)
                    torch.cuda.synchronize(selected)
                    record["cuda_kernel_verified"] = True
            if checkpoint.is_file():
                record["model"] = local_metadata(name, checkpoint, settings["device"])
                if not record["model"]["checkpoint_matches_official"]:
                    errors.append("Checkpoint SHA256 differs from the pinned official weight")
                if record["model"]["source_revision"] != record["model"]["pinned_revision"]:
                    errors.append("Source revision differs from the pinned model revision")
            if load and not errors:
                service = (create_vggt if name == "vggt" else create_sam3)()
                service.load()
                record["model"] = service.metadata
                record["loaded"] = True
            record["status"] = "failed" if errors else "ready"
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            record["status"] = "failed"
        finally:
            release_backend(service)
    return {"passed": all(not record["errors"] for record in records.values()),
            "models": records, "inference_verified": False}


def download_models(names, *, source="official") -> dict:
    """Download pinned weights; public mirrors never receive HF credentials."""
    if source not in ("official", "mirror"):
        raise ValueError(f"Unsupported weight source: {source}")
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    records = {}
    for name in names:
        weight = MODEL_WEIGHTS[name]
        origin = weight if source == "official" else MODEL_MIRRORS[name]
        request_options = {"token": False} if source == "mirror" else {}
        record = records[name] = {
            "source": source, "repository": origin["repo"], "revision": origin["revision"],
            "official_repository": weight["repo"], "official_revision": weight["revision"],
            "expected_bytes": weight["size"], "expected_sha256": weight["sha256"],
            "status": "failed", "errors": [],
        }
        try:
            settings = backend_settings(name)
            if settings["backend"] != "local":
                raise ValueError(f"Select the local backend before downloading {name}")
            checkpoint = settings["checkpoint"]
            record["checkpoint"] = str(checkpoint)
            existed = checkpoint.exists()
            if existed:
                if (not checkpoint.is_file() or checkpoint.stat().st_size != weight["size"] or
                        checkpoint_digest(checkpoint) != weight["sha256"]):
                    raise FileExistsError(f"Existing checkpoint has different contents: {checkpoint}")
            else:
                filename = MODEL_INFO[name][2]
                url = hf_hub_url(origin["repo"], filename, revision=origin["revision"])
                metadata = get_hf_file_metadata(url, **request_options)
                if metadata.etag != weight["sha256"] or metadata.size != weight["size"]:
                    raise ValueError(f"Download metadata differs from the pinned official {name} checkpoint")
                # Check access and metadata before creating output/cache files.
                hf_hub_download(repo_id=origin["repo"], filename=filename,
                                revision=origin["revision"], local_dir=str(checkpoint.parent),
                                **request_options)
                if (checkpoint.stat().st_size != weight["size"] or
                        checkpoint_digest(checkpoint) != weight["sha256"]):
                    raise ValueError(f"Downloaded checkpoint failed size/SHA256 verification: {checkpoint}")
            record.update(status="present" if existed else "downloaded",
                          bytes=checkpoint.stat().st_size, sha256=weight["sha256"])
        except GatedRepoError:
            record["status"] = "access_required"
            record["errors"].append(
                f"Obtain access at https://huggingface.co/{weight['repo']} and run hf auth login")
        except HfHubHTTPError as exc:
            record["errors"].append(
                f"Hugging Face request failed: HTTP {getattr(exc.response, 'status_code', 'unknown')}")
        except (OSError, RuntimeError, ValueError) as exc:
            record["errors"].append(f"{type(exc).__name__}: {exc}")
    return {"passed": all(not record["errors"] for record in records.values()),
            "models": records, "inference_verified": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="check dependencies, weights, source revision and device")
    check.add_argument("--models", nargs="+", choices=("vggt", "sam3"), default=["vggt", "sam3"])
    check.add_argument("--load", action="store_true", help="also load each local model, then release it")
    download = subparsers.add_parser("download", help="download pinned weights from official access or public mirrors")
    download.add_argument("--models", nargs="+", choices=("vggt", "sam3"), default=["vggt", "sam3"])
    download.add_argument("--source", choices=("official", "mirror"), default="official",
                          help="select official access or public mirrors; always verify official SHA256")
    args = parser.parse_args(argv)
    report = (download_models(args.models, source=args.source) if args.command == "download" else
              check_models(args.models, load=args.load))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
