import os
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from real2sim.perception import config


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    for key in tuple(os.environ):
        if key.startswith(("VGGT_OMEGA_", "SAM3_")):
            monkeypatch.delenv(key)


def test_defaults_are_local_without_endpoints():
    vggt, sam = config.create_vggt(), config.create_sam3()
    assert vggt.device == sam.device == "cuda"
    assert config.backend_settings("vggt")["backend"] == "local"
    assert config.sam3_endpoint() == (None, None)
    assert config.backend_manifest(sam, "sam3")["backend"] == "local"


def test_independent_selection_and_environment_precedence(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("VGGT_OMEGA_BACKEND=remote\nSAM3_BACKEND=remote\n")
    monkeypatch.setenv("SAM3_BACKEND", "local")
    monkeypatch.setenv("VGGT_OMEGA_HOST", "test-service")
    monkeypatch.setenv("VGGT_OMEGA_PORT", "8317")
    assert config.backend_settings("vggt")["host"] == "test-service"
    assert config.backend_settings("sam3")["backend"] == "local"
    assert type(config.create_vggt()).__name__ == "RemoteVGGT"


def test_relative_and_absolute_model_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("SAM3_MODEL_DIR", "weights")
    assert config.backend_settings("sam3")["checkpoint"] == tmp_path / "weights/sam3.pt"
    monkeypatch.setenv("SAM3_MODEL_DIR", str(tmp_path / "mounted"))
    assert config.backend_settings("sam3")["model_dir"] == tmp_path / "mounted"


@pytest.mark.parametrize("port", ["", "abc", "0", "65536"])
def test_remote_requires_valid_endpoint(monkeypatch, port):
    monkeypatch.setenv("SAM3_BACKEND", "remote")
    monkeypatch.setenv("SAM3_HOST", "test-host")
    monkeypatch.setenv("SAM3_PORT", port)
    with pytest.raises(ValueError):
        config.create_sam3()


def test_invalid_backend_fails_without_fallback(monkeypatch):
    monkeypatch.setenv("SAM3_BACKEND", "automatic")
    with pytest.raises(ValueError, match="unsupported"):
        config.create_sam3()


def test_explicit_remote_instance_endpoint_is_backward_compatible():
    segmenter = config.create_instance_segmenter("test-host", 8317)
    assert segmenter.backend.host == "test-host"
    with pytest.raises(ValueError, match="requires a host"):
        config.create_instance_segmenter(port=8317)


def test_device_requires_requested_hardware():
    module = SimpleNamespace(device=lambda name: SimpleNamespace(type=name, index=None),
                             cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        config.require_device(module, "cuda")
    assert config.require_device(module, "cpu").type == "cpu"


def test_metadata_records_weight_hash_without_inventing_installed_revision(tmp_path):
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"model")
    details = config.local_metadata("sam3", checkpoint, "cuda")
    assert details["backend"] == "local"
    assert len(details["checkpoint_sha256"]) == 64
    assert details["source_revision"] is None
    assert details["preprocessing_version"] == "aligned-rgb-grid-v1"


def test_model_check_missing_dependencies_is_read_only(monkeypatch, tmp_path):
    from real2sim.perception import cli

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)
    report = cli.check_models(["vggt", "sam3"])
    assert report["passed"] is False
    assert report["inference_verified"] is False
    assert not list(tmp_path.iterdir())
    assert all(record["errors"] for record in report["models"].values())


@pytest.fixture
def official_download(monkeypatch, tmp_path):
    hub = pytest.importorskip("huggingface_hub")
    from real2sim.perception import cli

    contents = b"official test checkpoint"
    weight = {**config.MODEL_WEIGHTS["vggt"], "size": len(contents),
              "sha256": hashlib.sha256(contents).hexdigest()}
    monkeypatch.setitem(config.MODEL_WEIGHTS, "vggt", weight)
    calls = []
    def metadata(url, **kwargs):
        calls.append(("metadata", url))
        return SimpleNamespace(etag=weight["sha256"], size=weight["size"])
    def download(**kwargs):
        calls.append(("download", kwargs))
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True)
        path.write_bytes(contents)
        return str(path)
    monkeypatch.setattr(hub, "get_hf_file_metadata", metadata)
    monkeypatch.setattr(hub, "hf_hub_download", download)
    return SimpleNamespace(hub=hub, cli=cli, contents=contents, weight=weight, calls=calls,
                           checkpoint=tmp_path / "data/models/vggt_omega/vggt_omega_1b_512.pt")


def test_download_pins_revision_verifies_content_and_reuses_valid_file(official_download):
    fixture = official_download
    report = fixture.cli.download_models(["vggt"])
    assert report["passed"] and not report["inference_verified"]
    assert fixture.checkpoint.read_bytes() == fixture.contents
    assert fixture.calls[1][1]["revision"] == fixture.weight["revision"]
    assert fixture.calls[1][1]["repo_id"] == "facebook/VGGT-Omega"
    original_time = fixture.checkpoint.stat().st_mtime_ns
    again = fixture.cli.download_models(["vggt"])
    assert again["models"]["vggt"]["status"] == "present"
    assert len(fixture.calls) == 2
    assert fixture.checkpoint.stat().st_mtime_ns == original_time


def test_download_access_denial_creates_no_output(official_download, monkeypatch, tmp_path):
    from huggingface_hub.errors import GatedRepoError
    from httpx import Request, Response

    def denied(url):
        raise GatedRepoError("access required", response=Response(401, request=Request("HEAD", url)))
    monkeypatch.setattr(official_download.hub, "get_hf_file_metadata", denied)
    report = official_download.cli.download_models(["vggt"])
    assert not report["passed"]
    assert report["models"]["vggt"]["status"] == "access_required"
    assert not list(tmp_path.iterdir())


def test_download_preserves_a_conflicting_existing_checkpoint(official_download):
    fixture = official_download
    fixture.checkpoint.parent.mkdir(parents=True)
    fixture.checkpoint.write_bytes(b"user checkpoint")
    report = fixture.cli.download_models(["vggt"])
    assert not report["passed"]
    assert fixture.checkpoint.read_bytes() == b"user checkpoint"
    assert fixture.calls == []


@pytest.mark.parametrize("mismatch", ["size", "sha256"])
def test_download_rejects_unexpected_official_metadata(official_download, monkeypatch, mismatch):
    fixture = official_download
    metadata = SimpleNamespace(etag=fixture.weight["sha256"], size=fixture.weight["size"])
    if mismatch == "size":
        metadata.size += 1
    else:
        metadata.etag = "0" * 64
    monkeypatch.setattr(fixture.hub, "get_hf_file_metadata", lambda _: metadata)
    report = fixture.cli.download_models(["vggt"])
    assert not report["passed"]
    assert not fixture.checkpoint.parent.exists()
    assert fixture.calls == []


def test_download_rejects_corrupted_response_body(official_download, monkeypatch):
    fixture = official_download
    def corrupted(**kwargs):
        fixture.checkpoint.parent.mkdir(parents=True)
        fixture.checkpoint.write_bytes(b"X" * len(fixture.contents))
        return str(fixture.checkpoint)
    monkeypatch.setattr(fixture.hub, "hf_hub_download", corrupted)
    report = fixture.cli.download_models(["vggt"])
    assert not report["passed"]
    assert "SHA256" in report["models"]["vggt"]["errors"][0]


def test_mirror_download_pins_source_and_never_sends_credentials(official_download, monkeypatch):
    fixture = official_download
    requests = []

    def metadata(url, **kwargs):
        requests.append((url, kwargs))
        return SimpleNamespace(etag=fixture.weight["sha256"], size=fixture.weight["size"])

    monkeypatch.setattr(fixture.hub, "get_hf_file_metadata", metadata)
    report = fixture.cli.download_models(["vggt"], source="mirror")
    mirror = config.MODEL_MIRRORS["vggt"]
    assert report["passed"]
    assert report["models"]["vggt"]["source"] == "mirror"
    assert report["models"]["vggt"]["official_repository"] == "facebook/VGGT-Omega"
    assert mirror["repo"] in requests[0][0] and mirror["revision"] in requests[0][0]
    assert requests[0][1] == {"token": False}
    download = fixture.calls[0][1]
    assert download["repo_id"] == mirror["repo"]
    assert download["revision"] == mirror["revision"]
    assert download["token"] is False
    assert fixture.checkpoint.read_bytes() == fixture.contents


@pytest.mark.parametrize("mismatch", ["size", "sha256"])
def test_mirror_rejects_metadata_before_creating_files(official_download, monkeypatch, mismatch):
    fixture = official_download
    metadata = SimpleNamespace(etag=fixture.weight["sha256"], size=fixture.weight["size"])
    if mismatch == "size":
        metadata.size += 1
    else:
        metadata.etag = "0" * 64
    monkeypatch.setattr(fixture.hub, "get_hf_file_metadata", lambda *args, **kwargs: metadata)
    report = fixture.cli.download_models(["vggt"], source="mirror")
    assert not report["passed"]
    assert not fixture.checkpoint.parent.exists()
    assert fixture.calls == []


def test_mirror_rejects_corrupted_body(official_download, monkeypatch):
    fixture = official_download

    def corrupted(**kwargs):
        fixture.checkpoint.parent.mkdir(parents=True)
        fixture.checkpoint.write_bytes(b"X" * len(fixture.contents))
        return str(fixture.checkpoint)

    monkeypatch.setattr(fixture.hub, "hf_hub_download", corrupted)
    report = fixture.cli.download_models(["vggt"], source="mirror")
    assert not report["passed"]
    assert "SHA256" in report["models"]["vggt"]["errors"][0]


@pytest.mark.parametrize("kernel_supported", [True, False])
def test_cuda_check_executes_a_kernel_before_reporting_ready(monkeypatch, tmp_path, kernel_supported):
    import sys
    from real2sim.perception import cli

    checkpoint = tmp_path / "data/models/vggt_omega/vggt_omega_1b_512.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    source = tmp_path / "external/vggt_omega"
    source.mkdir(parents=True)
    (source / ".git").write_text("test checkout")
    revision = config.MODEL_INFO["vggt"][4]
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "local_metadata", lambda *args: {
        "checkpoint_matches_official": True, "source_revision": revision, "pinned_revision": revision})
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=None))

    def ones(*args, **kwargs):
        if not kernel_supported:
            raise RuntimeError("no kernel image is available for execution on the device")
        return SimpleNamespace(add_=lambda value: None)

    torch = SimpleNamespace(
        __version__="2.8.0+cu128", version=SimpleNamespace(cuda="12.8"),
        device=lambda name: SimpleNamespace(type="cuda", index=None), ones=ones,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda device: SimpleNamespace(name="RTX 5080", major=12, minor=0),
            mem_get_info=lambda device: (10 * 1024**3, 16 * 1024**3),
            synchronize=lambda device: None))
    monkeypatch.setitem(sys.modules, "torch", torch)
    report = cli.check_models(["vggt"])
    record = report["models"]["vggt"]
    assert report["passed"] is kernel_supported
    assert record["gpu"]["name"] == "RTX 5080"
    assert record["cuda_version"] == "12.8"
    if kernel_supported:
        assert record["cuda_kernel_verified"] is True
        assert record["status"] == "ready"
    else:
        assert record["status"] == "failed"
        assert "no kernel image" in record["errors"][0]
        assert "cuda_kernel_verified" not in record
