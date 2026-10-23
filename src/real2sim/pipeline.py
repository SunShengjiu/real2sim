"""One video-to-MuJoCo workflow with resumable, fingerprinted stage attempts."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from importlib import metadata, util
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
from typing import Any, Iterator

from .config import ROOT, SceneConfig, runtime_environment
from .contracts import (STAGES, bundle_scene, digest, inventory, read_json,
                        validate_bundle_paths, verify_inventory, write_json)


def check_environment(config: SceneConfig) -> dict[str, Any]:
    """Check prerequisites without writing outputs or starting paid model turns."""
    errors: list[str] = []
    report: dict[str, Any] = {"errors": errors, "inference_verified": False,
                              "agent_authentication": "not_verified"}
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        errors.append("Python 3.10–3.12 is required; use 3.11 for full local inference")
    for module in ("numpy", "cv2", "trimesh", "collada", "dotenv", "yaml", "mujoco", "PIL", "openai_codex"):
        if util.find_spec(module) is None:
            errors.append(f"missing dependency: {module}")
    for name in ("robot_control", "ultralytics", "vggt_omega"):
        # Remote perception does not need the corresponding local model checkout.
        if name == "ultralytics" and config.perception["sam3_backend"] == "remote":
            continue
        if name == "vggt_omega" and config.perception["vggt_backend"] == "remote":
            continue
        path = ROOT / "external" / name
        if not (path / ".git").exists():
            errors.append(f"uninitialized submodule: {path}")
        else:
            try:
                current = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, encoding="utf-8", errors="replace").strip()
                expected = subprocess.check_output(["git", "-C", str(ROOT), "ls-tree", "HEAD", f"external/{name}"], text=True, encoding="utf-8", errors="replace").split()[2]
                if current != expected:
                    errors.append(f"submodule revision differs from gitlink: {name}")
                dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True, encoding="utf-8", errors="replace").strip()
                if dirty:
                    errors.append(f"submodule has uncommitted changes: {name}")
            except (OSError, subprocess.SubprocessError, IndexError) as exc:
                errors.append(f"submodule provenance {name}: {exc}")
    for label, path in (("video", Path(config.video)), ("robot URDF", Path(config.robot["urdf"]))):
        if not path.is_file():
            errors.append(f"missing {label}: {path}")
    for name in ("metadata.json", "surface_5000.npz", "surface_10000.npz", "surface_40000.npz"):
        path = ROOT / "data/assets/robot_surfaces" / config.robot["name"] / name
        if not path.is_file():
            errors.append(f"missing robot template: {path}")
    if config.stage3["semantics"] and not Path(config.stage3["semantics"]).is_file():
        errors.append("reviewed semantics file does not exist")
    with runtime_environment(config):
        try:
            from .blender import resolve_blender
            executable = resolve_blender(config.blender)
            version = subprocess.run([str(executable), "--version"], capture_output=True,
                                     text=True, timeout=20, check=True, encoding="utf-8", errors="replace")
            report["blender"] = {"executable": str(executable), "version": version.stdout.splitlines()[0]}
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            errors.append(f"Blender: {exc}")
        try:
            from .perception.cli import check_models
            report["models"] = check_models(("vggt", "sam3"))
            if not report["models"]["passed"]:
                errors.append("perception model prerequisites failed; see models")
        except (ImportError, OSError, ValueError) as exc:
            errors.append(f"perception: {exc}")
    if util.find_spec("cv2") is not None and Path(config.video).is_file():
        import cv2
        capture = cv2.VideoCapture(config.video)
        count = 0
        try:
            while count < 32 and capture.read()[0]:
                count += 1
        finally:
            capture.release()
        report["video_decoded_frames_checked"] = count
        if count < 32:
            errors.append("video must contain at least 32 decodable frames")
    if not shutil.which("codex"):
        errors.append("Codex CLI is missing; authenticate it before agent stages")
    # Validate the pose and template before committing resources to inference.
    if util.find_spec("trimesh") and Path(config.robot["urdf"]).is_file():
        try:
            from .stage1.robot_surface import POSES, BASE_LINKS
            from .stage1.geometry import MODEL
            from .stage1.robot_surface import FRANKA_URDF
            robot = config.robot["name"]
            expected_urdf = MODEL if robot == "piper" else FRANKA_URDF
            if Path(config.robot["urdf"]).resolve() != expected_urdf.resolve():
                errors.append("configured URDF must match the supported robot model")
            import numpy as np
            actual = config.robot["joint_positions"]
            if set(actual) != set(POSES[robot]) or any(
                    not np.isclose(actual[key], value, rtol=0, atol=1e-10)
                    for key, value in POSES[robot].items()):
                errors.append("configured pose differs from the implemented fixed robot/template pose")
            if config.robot["base_frame"] != BASE_LINKS[robot]:
                errors.append("configured base_frame differs from the robot model")
            template = read_json(ROOT / "data/assets/robot_surfaces" / robot / "metadata.json")
            if template["urdf_sha256"] != digest(Path(config.robot["urdf"])):
                errors.append("robot template URDF fingerprint is stale; regenerate templates")
        except (ImportError, OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"robot template: {exc}")
    report["passed"] = not errors
    return report


def _provenance(config: SceneConfig) -> dict[str, Any]:
    paths = [*sorted((ROOT / "src/real2sim").rglob("*.py")),
             *sorted((ROOT / "src/real2sim").rglob("*.md")),
             ROOT / "src/real2sim/stage1/thresholds.json", ROOT / "pyproject.toml", ROOT / "uv.lock"]
    paths += sorted((ROOT / "data/assets/robot_surfaces" / config.robot["name"]).glob("*"))
    sources = {str(path.relative_to(ROOT)): digest(path) for path in paths if path.is_file()}
    versions = {}
    for name in ("numpy", "opencv-python", "trimesh", "mujoco", "torch", "torchvision", "openai-codex", "PyYAML"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    inputs = {config.video: digest(Path(config.video)), config.robot["urdf"]: digest(Path(config.robot["urdf"]))}
    if config.stage3["semantics"]:
        inputs[config.stage3["semantics"]] = digest(Path(config.stage3["semantics"]))
    with runtime_environment(config):
        from .perception.config import backend_settings, checkpoint_digest
        for name in ("vggt", "sam3"):
            settings = backend_settings(name)
            if settings["backend"] == "local":
                checkpoint = settings["checkpoint"]
                inputs[str(checkpoint)] = checkpoint_digest(checkpoint)
    revisions = {}
    for name in ("robot_control", "vggt_omega", "ultralytics"):
        path = ROOT / "external" / name
        if (path / ".git").exists():
            revisions[name] = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, encoding="utf-8", errors="replace").strip()
    return {"source_sha256": sources, "input_sha256": inputs, "versions": versions,
            "submodules": revisions, "python": platform.python_version(), "platform": platform.platform()}


@contextmanager
def _run_lock(run: Path) -> Iterator[None]:
    path = run / ".running.lock"
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"run is locked: {path}; ensure its owner is stopped before removing a stale lock") from exc
    try:
        with stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def _stage1(config: SceneConfig, output: Path, _: Path) -> dict[str, Any]:
    import numpy as np
    from .perception.config import create_sam3, create_vggt
    from .stage1.inference_batch import infer_video
    from .stage1.initial_pose_batch import visualize_scene
    from .stage1.refinement import refine_scene
    from .stage1.delivery import publish
    from .stage1.piper_top3 import publish_winner

    source = output / "scene"
    manifest = infer_video(Path(config.video), source, config.robot["name"],
        digest(Path(config.video)), create_vggt(), create_sam3(), {"vggt": {}, "sam3": {}})
    if manifest["status"] != "inference_complete" or manifest["empty_mask_frames"]:
        raise ValueError("perception failed or contains empty robot masks; inspect stage log and manifest")
    manifest.update(task_description=config.task, scene_id=config.name)
    write_json(source / "manifest.json", manifest)
    initialization = output / "initialization"
    visualize_scene(source, initialization, config.robot["registration"]["rotations"])
    model_root = ROOT / "data/assets/robot_surfaces" / config.robot["name"]
    models = {}
    for size in (5000, 10000, 40000):
        with np.load(model_root / f"surface_{size}.npz", allow_pickle=False) as data:
            models[size] = data["points"].astype(float)
    full = models.pop(40000)
    models[20000] = full[np.random.default_rng(2027).permutation(len(full))[:20000]]
    branches = []
    for rank in range(1, config.robot["registration"]["top_k"] + 1):
        branch = output / "registration" / f"rank_{rank}"
        result = refine_scene(config.name, output, initialization, branch, models, full,
            config.robot["registration"]["local_passes"], robot=config.robot["name"], seed_rank=rank,
            source=source, initialization_file=initialization / "initialization.json")
        branches.append((result, branch))
    if config.robot["name"] == "piper":
        selected = output / "registration" / "selected"
        publish_winner(config.name, branches, selected, initialization / "initialization.json", manifest)
    else:
        selected = branches[0][1]
    delivery = publish(source, selected / "result.json", config.blender, models[20000],
                       initialization_file=initialization / "initialization.json")
    return {"source": str(source), "registration_status": delivery["status"],
            "independently_validated": False}


def _stage2(config: SceneConfig, output: Path, source: Path) -> dict[str, Any]:
    from .stage2 import agent_runner, phase_runner
    for module in (agent_runner, phase_runner):
        module.CODEX_MODEL = config.agent["model"]
        module.CODEX_REASONING_EFFORT = config.agent["effort"]
    result = phase_runner.run_phases(source, output, config.task,
                                    config.stage2["max_rounds"], config.blender)
    if result["status"] != "passed":
        raise ValueError(f"Stage 2 did not pass joint visual evaluation: {result['status']}")
    manifest_path = output / "geometry/stage2_manifest.json"
    manifest = read_json(manifest_path)
    manifest.update(robot=config.robot["name"], task_description=config.task, scene_id=config.name)
    write_json(manifest_path, manifest)
    return {"source": str(output), "visual_metrics_passed": True}


def _stage3(config: SceneConfig, output: Path, source: Path) -> dict[str, Any]:
    from .stage3 import semantics_agent
    from .stage3.mujoco_export import ExportConfig, export_blend
    from .stage3.physics_probe import check_static, auto_probe
    semantics_agent.MODEL = config.agent["model"]
    semantics_agent.EFFORT = config.agent["effort"]
    exported = export_blend(ExportConfig(None, output / "export", blender=config.blender,
        stage2_delivery=source, semantics_path=Path(config.stage3["semantics"]) if config.stage3["semantics"] else None))
    scene = Path(exported["scene"])
    if config.stage3["physics"] == "agent":
        from .stage3.agent_runner import run_agent
        hypotheses = output / "hypotheses.json"
        write_json(hypotheses, {"real_world_measurements": [], "hypotheses": [{
            "origin": "visual reconstruction; physical properties are unmeasured",
            "expectation": "task objects remain supported and allow task-relevant interaction"}]})
        plan = (source / "plan/plan.md").read_text(encoding="utf-8")
        result = run_agent(scene, hypotheses, output / "physics", f"Task: {config.task}\n\n{plan}")
        checked = result.get("verification") or {}
        candidate = checked.get("candidate_recheck") or {}
        passed = checked.get("passed") and ((candidate.get("static_passed") and candidate.get("grasp_passed")) or
            (checked.get("sim_static_passed") and (checked.get("sim_grasp_passed") or checked.get("sim_articulation_passed"))))
        if not passed:
            raise ValueError("Stage 3 physical audit did not pass; inspect physics/session.json")
        candidate_path = result.get("agent_result", {}).get("candidate_scene")
        if candidate_path:
            scene = Path(candidate_path)
            if not scene.is_absolute():
                scene = ROOT / scene
    else:
        static = check_static(scene, output / "physics/static")
        sweep = auto_probe(scene)
        write_json(output / "physics/response/report.json", sweep)
        if not static["passed"] or not sweep["passed"]:
            raise ValueError("Stage 3 static/response probes failed")
    if not scene.resolve().is_relative_to(output.resolve()):
        raise ValueError("final physical scene must belong to this stage attempt")
    if config.stage3["render_views"]:
        from .stage3.render_overview import render_reference_views
        render_reference_views(scene, output / "views")
    write_json(output / "final_manifest.json", {"scene": str(scene), "scene_sha256": digest(scene),
        "physics_status": "provisional", "real_world_verified": False, "mode": config.stage3["physics"]})
    return {"scene": str(scene), "physics_status": "provisional", "real_world_verified": False}


def _delivery(config: SceneConfig, output: Path, source: Path) -> dict[str, Any]:
    final = read_json(source / "final_manifest.json")
    scene = Path(final["scene"])
    bundle_scene(scene, output)
    # Include evidence separately so its historic paths do not become runtime dependencies.
    for name in ("export", "physics", "views"):
        origin = source / name
        if origin.is_dir():
            for path in origin.rglob("*"):
                if path.is_file() and (path.suffix == ".json" or name == "views"):
                    target = output / "reports" / name / path.relative_to(origin)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
    validate_bundle_paths(output)
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    report = {"schema": "real2sim-delivery-v1", "scene": "scene.xml", "robot": config.robot["name"],
        "task": config.task, "loadable": True, "bodies": int(model.nbody),
        "registration_status": "candidate_unverified", "physics_status": "provisional",
        "real_world_verified": False, "files": inventory(output)}
    write_json(output / "manifest.json", report)
    return {"scene": str(output / "scene.xml"), "loadable": True}


def run(config: SceneConfig, *, existing: Path | None = None) -> Path:
    report = check_environment(config)
    if not report["passed"]:
        raise ValueError("prerequisite check failed: " + "; ".join(report["errors"]))
    if existing is None:
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
        folder = Path(config.output_root) / f"{config.name}_{stamp}"
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / "config.json", config.snapshot())
        write_json(folder / "manifest.json", {"schema": "real2sim-run-v1", "status": "pending",
            "config_sha256": digest(folder / "config.json"), "provenance": _provenance(config),
            "environment_check": report, "stages": {}})
    else:
        folder = existing.expanduser().resolve(strict=True)
    with _run_lock(folder), runtime_environment(config):
        manifest = read_json(folder / "manifest.json")
        if digest(folder / "config.json") != manifest["config_sha256"] or config.snapshot() != read_json(folder / "config.json"):
            raise ValueError("run configuration changed; create a new run")
        if _provenance(config) != manifest["provenance"]:
            raise ValueError("source, input, model or environment changed; create a new run")
        stages = manifest["stages"]
        for name in STAGES:
            previous = stages.get(name, {})
            if previous.get("status") == "completed":
                verify_inventory(folder / previous["folder"], previous["artifacts"])
                continue
            attempt = int(previous.get("attempt", 0)) + 1
            target = folder / (f"stage3/delivery_attempt_{attempt:03d}" if name == "delivery"
                               else f"{name}/attempt_{attempt:03d}")
            if target.exists():
                raise FileExistsError(f"stage output already exists: {target}")
            if name != "delivery":
                target.mkdir(parents=True)
            source = folder
            if name == "stage2":
                source = Path(stages["stage1"]["result"]["source"])
            elif name == "stage3":
                source = Path(stages["stage2"]["result"]["source"])
            elif name == "delivery":
                source = folder / stages["stage3"]["folder"]
            record = {"status": "running", "attempt": attempt, "folder": target.relative_to(folder).as_posix()}
            stages[name] = record
            manifest["status"] = "running"
            write_json(folder / "manifest.json", manifest)
            print(f"{name}: attempt {attempt}", flush=True)
            try:
                log_path = folder / (f"{name}.log" if name == "delivery" else f"{record['folder']}/stage.log")
                with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                    result = {"stage1": _stage1, "stage2": _stage2, "stage3": _stage3, "delivery": _delivery}[name](config, target, source)
                if name == "delivery":
                    destination = folder / "delivery"
                    if destination.exists():
                        raise FileExistsError(f"delivery already exists: {destination}")
                    target.rename(destination)
                    target = destination
                    record["folder"] = "delivery"
                    result["scene"] = str(destination / "scene.xml")
                record.update(status="completed", result=result, artifacts=inventory(target))
            except Exception as exc:
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(traceback.format_exc())
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                manifest["status"] = "failed"
                write_json(folder / "manifest.json", manifest)
                raise RuntimeError(f"{name} failed in {folder}: {exc}") from exc
            write_json(folder / "manifest.json", manifest)
        if _provenance(config) != manifest["provenance"]:
            manifest["status"] = "inputs_changed"
            write_json(folder / "manifest.json", manifest)
            raise ValueError("inputs or source changed during execution; results require review")
        manifest["status"] = "completed"
        write_json(folder / "manifest.json", manifest)
    return folder


def verify(folder: Path) -> dict[str, Any]:
    folder = folder.expanduser().resolve(strict=True)
    errors = []
    manifest = read_json(folder / "manifest.json")
    if manifest.get("schema") == "real2sim-delivery-v1":
        for name, expected in manifest["files"].items():
            path = (folder / name).resolve()
            if not path.is_relative_to(folder) or not path.is_file() or digest(path) != expected:
                errors.append(f"delivery artifact changed: {name}")
    elif manifest.get("schema") == "real2sim-run-v1":
        if manifest.get("status") != "completed":
            errors.append("run did not complete successfully")
        if digest(folder / "config.json") != manifest["config_sha256"]:
            errors.append("configuration snapshot changed")
        for name in STAGES:
            stage = manifest["stages"].get(name, {})
            if stage.get("status") != "completed":
                errors.append(f"{name} is not completed")
                continue
            try:
                verify_inventory(folder / stage["folder"], stage["artifacts"])
            except ValueError as exc:
                errors.append(str(exc))
        folder = folder / "delivery"
    else:
        errors.append("unknown manifest schema")
    if not errors:
        try:
            validate_bundle_paths(folder)
            import mujoco
            model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
            del model
        except (OSError, ValueError, ImportError) as exc:
            errors.append(str(exc))
    return {"passed": not errors, "errors": errors, "real_world_verified": False}
