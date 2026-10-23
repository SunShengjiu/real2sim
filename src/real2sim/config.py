"""Validated scene configuration and reproducible configuration snapshots."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]


def read_mapping(path: Path) -> dict[str, Any]:
    import yaml

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def _keys(value: dict, allowed: set[str], label: str) -> None:
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"unknown {label} settings: {sorted(extra)}")


def _path(value: str, base: Path) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("file and directory paths must be nonempty strings")
    path = Path(value).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def _settings(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


@dataclass(frozen=True)
class SceneConfig:
    name: str
    video: str
    task: str
    robot: dict[str, Any]
    output_root: str
    blender: str | None
    perception: dict[str, Any]
    agent: dict[str, Any]
    stage2: dict[str, Any]
    stage3: dict[str, Any]
    schema_version: int = 1

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_snapshot(cls, path: Path) -> "SceneConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def load_config(path: Path) -> SceneConfig:
    path = path.expanduser().resolve(strict=True)
    data = read_mapping(path)
    _keys(data, {"schema_version", "name", "video", "task", "robot", "output_root",
                 "blender", "perception", "agent", "stage2", "stage3"}, "scene")
    if type(data.get("schema_version", 1)) is not int or data.get("schema_version", 1) != 1:
        raise ValueError("unsupported scene configuration schema_version")
    name = data.get("name", "")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("name must contain 1–80 letters, digits, underscores or hyphens")
    if not isinstance(data.get("task"), str) or not data["task"].strip():
        raise ValueError("task must be a nonempty task description")
    for key in ("video", "robot"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"{key} must be a file path")
    robot_path = Path(_path(data["robot"], path.parent))
    robot = read_mapping(robot_path)
    _keys(robot, {"name", "urdf", "base_frame", "joint_positions", "registration"}, "robot")
    if robot.get("name") not in {"piper", "franka"}:
        raise ValueError("robot name must be piper or franka")
    if not isinstance(robot.get("joint_positions"), dict) or not robot["joint_positions"]:
        raise ValueError("robot needs explicit fixed joint_positions")
    if any(type(value) not in {int, float} or not math.isfinite(value)
           for value in robot["joint_positions"].values()):
        raise ValueError("joint_positions must contain finite numeric values")
    if not isinstance(robot.get("base_frame"), str) or not robot["base_frame"]:
        raise ValueError("robot.base_frame must be nonempty")
    robot["urdf"] = _path(robot["urdf"], robot_path.parent)
    registration = robot["registration"] = _settings(robot.get("registration"), "registration")
    _keys(registration, {"rotations", "top_k", "local_passes"}, "registration")
    expected = (128, 3) if robot["name"] == "piper" else (48, 1)
    registration.setdefault("rotations", expected[0])
    registration.setdefault("top_k", expected[1])
    registration.setdefault("local_passes", 2)
    if (registration["rotations"], registration["top_k"]) != expected:
        raise ValueError(f"current {robot['name']} delivery requires rotations/top_k={expected}")
    if type(registration["local_passes"]) is not int or registration["local_passes"] < 0:
        raise ValueError("local_passes must be a nonnegative integer")
    perception = _settings(data.get("perception"), "perception")
    _keys(perception, {"vggt_backend", "sam3_backend", "device", "vggt_model_dir", "sam3_model_dir"}, "perception")
    for model in ("vggt", "sam3"):
        perception.setdefault(f"{model}_backend", "local")
        if perception[f"{model}_backend"] not in {"local", "remote"}:
            raise ValueError(f"invalid {model} backend")
        value = perception.get(f"{model}_model_dir")
        perception[f"{model}_model_dir"] = (_path(value, path.parent) if value else
            str(ROOT / "data/models" / ("vggt_omega" if model == "vggt" else "sam3")))
    perception.setdefault("device", "cuda")
    if not isinstance(perception["device"], str) or not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", perception["device"]):
        raise ValueError("perception.device must be cpu, mps or cuda[:index]")
    agent = {"model": "gpt-6-astra", "effort": "high", **_settings(data.get("agent"), "agent")}
    _keys(agent, {"model", "effort"}, "agent")
    if not isinstance(agent["model"], str) or not agent["model"].strip():
        raise ValueError("agent.model must be nonempty")
    if agent["effort"] not in {"low", "medium", "high", "xhigh"}:
        raise ValueError("unsupported agent effort")
    stage2 = {"max_rounds": 10, **_settings(data.get("stage2"), "stage2")}
    _keys(stage2, {"max_rounds"}, "stage2")
    if type(stage2["max_rounds"]) is not int or stage2["max_rounds"] < 0:
        raise ValueError("stage2.max_rounds must be a nonnegative integer")
    stage3 = {"physics": "agent", "semantics": None, "render_views": True,
              **_settings(data.get("stage3"), "stage3")}
    _keys(stage3, {"physics", "semantics", "render_views"}, "stage3")
    if stage3["physics"] not in {"agent", "probes"}:
        raise ValueError("stage3.physics must be agent or probes")
    if type(stage3["render_views"]) is not bool:
        raise ValueError("stage3.render_views must be a boolean")
    if stage3["semantics"]:
        stage3["semantics"] = _path(stage3["semantics"], path.parent)
    blender = data.get("blender")
    if blender is not None:
        if not isinstance(blender, str) or not blender:
            raise ValueError("blender must be an executable path, command or null")
        if "/" in blender or "\\" in blender:
            blender = _path(blender, path.parent)
    return SceneConfig(name, _path(data["video"], path.parent), data["task"].strip(), robot,
        _path(data.get("output_root", "../../data/runs"), path.parent), blender,
        perception, agent, stage2, stage3)


@contextmanager
def runtime_environment(config: SceneConfig) -> Iterator[None]:
    """Use one configuration consistently without persisting machine settings."""
    from dotenv import dotenv_values

    values = {key: value for key, value in dotenv_values(ROOT / ".env").items()
              if value is not None and key not in os.environ}
    values.update({"REAL2SIM_CODEX_MODEL": config.agent["model"],
                   "REAL2SIM_CODEX_EFFORT": config.agent["effort"]})
    # Direct .venv/Scripts/real2sim.exe launches must give child tools the same
    # Python environment as activated-shell and uv launches.
    values["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    for model, prefix in (("vggt", "VGGT_OMEGA"), ("sam3", "SAM3")):
        values.update({f"{prefix}_BACKEND": config.perception[f"{model}_backend"],
                       f"{prefix}_DEVICE": config.perception["device"],
                       f"{prefix}_MODEL_DIR": config.perception[f"{model}_model_dir"]})
    if config.blender:
        values["BLENDER_BIN"] = config.blender
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
