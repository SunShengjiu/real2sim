import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import yaml

from real2sim.config import ROOT, load_config, runtime_environment


def make_config(tmp_path, **changes):
    robot = yaml.safe_load((ROOT / "configs/robots/piper.yaml").read_text())
    (tmp_path / "robot.yaml").write_text(yaml.safe_dump(robot))
    scene = {"name": "example", "video": "input.mp4", "task": "place the cube",
             "robot": "robot.yaml", "output_root": "runs", **changes}
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(scene))
    return path


def test_paths_are_config_relative_and_demo_loads(tmp_path, monkeypatch):
    path = make_config(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    config = load_config(path)
    assert Path(config.video) == tmp_path / "input.mp4"
    assert Path(config.output_root) == tmp_path / "runs"
    assert config.robot["registration"]["top_k"] == 3
    assert load_config(ROOT / "configs/scenes/demo.yaml").robot["name"] == "piper"


@pytest.mark.parametrize("change", [
    {"name": "../escape"}, {"task": ""}, {"unknown": 1},
    {"stage2": {"max_rounds": -1}}, {"stage2": {"max_rounds": True}},
    {"stage3": {"physics": "none"}}, {"stage3": {"render_views": "false"}},
    {"perception": {"vggt_backend": "automatic"}},
    {"perception": {"device": "gpu"}}, {"stage2": [1, 2]},
    {"schema_version": True}, {"blender": 123},
])
def test_invalid_settings_are_rejected(tmp_path, change):
    with pytest.raises(ValueError):
        load_config(make_config(tmp_path, **change))


def test_runtime_environment_is_restored_on_failure(tmp_path, monkeypatch):
    config = load_config(make_config(tmp_path))
    monkeypatch.setenv("SAM3_BACKEND", "remote")
    with pytest.raises(RuntimeError), runtime_environment(config):
        assert os.environ["SAM3_BACKEND"] == "local"
        raise RuntimeError("stop")
    assert os.environ["SAM3_BACKEND"] == "remote"


def test_direct_launch_children_use_current_python_without_activation(tmp_path, monkeypatch):
    config = load_config(make_config(tmp_path))
    monkeypatch.setenv("PATH", "")
    with runtime_environment(config):
        arguments = [Path(sys.executable).name, "-c", "import sys; print(sys.prefix)"]
        command = subprocess.list2cmdline(arguments) if os.name == "nt" else shlex.join(arguments)
        child = subprocess.run(command, shell=True, capture_output=True,
                               text=True, encoding="utf-8", check=True)
        assert Path(child.stdout.strip()).resolve() == Path(sys.prefix).resolve()
    assert os.environ["PATH"] == ""
