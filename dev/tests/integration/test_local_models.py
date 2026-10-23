"""Opt-in real model checks; never fetch weights or silently use remote services."""

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.models


def cuda_model_environment():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is required for real model verification")
    if os.getenv("VGGT_OMEGA_BACKEND", "local") != "local" or os.getenv("SAM3_BACKEND", "local") != "local":
        pytest.fail("Model tests require explicitly local backends")
    return torch


def test_real_video_inference_records_models_and_memory(tmp_path):
    video = os.getenv("RSR_MODEL_TEST_VIDEO")
    if not video:
        pytest.skip("Set RSR_MODEL_TEST_VIDEO to a supplied video with a visible robot and 32+ frames")
    cuda_model_environment()
    from real2sim.perception.example import run_example

    report = run_example(Path(video), tmp_path / "example")
    assert report["status"] == "inference_complete"
    assert report["cuda_memory"]
    assert all(model["checkpoint_sha256"] for model in report["models"].values())
    manifest = json.loads((tmp_path / "example/manifest.json").read_text())
    assert len(manifest["empty_mask_frames"]) < 32
    assert len(list((tmp_path / "example/previews").glob("*.png"))) == 8
