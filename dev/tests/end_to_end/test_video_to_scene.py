"""Explicit opt-in: runs real checkpoints, Blender and paid agent turns."""

from dataclasses import replace
import os
from pathlib import Path
import shutil

import pytest

from real2sim.config import load_config
from real2sim.pipeline import run, verify


@pytest.mark.end_to_end
def test_real_video_to_relocatable_scene(tmp_path):
    supplied = os.getenv("RSR_E2E_CONFIG")
    if not supplied:
        pytest.skip("Set RSR_E2E_CONFIG explicitly to authorize the full model/agent workflow")
    config = replace(load_config(Path(supplied)), output_root=str(tmp_path / "runs"))
    result = run(config)
    assert verify(result)["passed"]
    relocated = tmp_path / "relocated_delivery"
    shutil.copytree(result / "delivery", relocated)
    assert verify(relocated)["passed"]
