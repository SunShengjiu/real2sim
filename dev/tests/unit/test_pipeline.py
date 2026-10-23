import shutil
from pathlib import Path

import pytest

from real2sim import pipeline
from real2sim.config import SceneConfig
from real2sim.contracts import (bundle_scene, digest, read_json, write_json,
                               validate_bundle_paths)


def fixture_config(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video")
    return SceneConfig("example", str(video), "place cube", {"name": "piper"},
        str(tmp_path / "runs"), None, {}, {}, {}, {})


def stub_stages(monkeypatch, calls, *, fail_stage2=False):
    monkeypatch.setattr(pipeline, "check_environment", lambda _: {"passed": True})
    monkeypatch.setattr(pipeline, "_provenance", lambda config: {"video": digest(Path(config.video))})
    def first(config, output, source):
        calls.append("stage1")
        (output / "result").write_text("registered")
        return {"source": str(output)}
    def second(config, output, source):
        calls.append("stage2")
        if fail_stage2:
            raise ValueError("visual gate failed")
        assert (source / "result").is_file()
        (output / "result").write_text("modeled")
        return {"source": str(output)}
    def third(config, output, source):
        calls.append("stage3")
        assert (source / "result").read_text() == "modeled"
        (output / "scene.xml").write_text('<mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>')
        write_json(output / "final_manifest.json", {"scene": str(output / "scene.xml")})
        return {"physics_status": "provisional"}
    def delivery(config, output, source):
        calls.append("delivery")
        bundle_scene(source / "scene.xml", output)
        return {"loadable": True}
    for name, function in (("_stage1", first), ("_stage2", second), ("_stage3", third), ("_delivery", delivery)):
        monkeypatch.setattr(pipeline, name, function)
    from contextlib import nullcontext
    monkeypatch.setattr(pipeline, "runtime_environment", lambda _: nullcontext())


def test_complete_run_and_resume_skip_completed_stages(tmp_path, monkeypatch):
    calls = []
    stub_stages(monkeypatch, calls)
    config = fixture_config(tmp_path)
    run = pipeline.run(config)
    assert calls == ["stage1", "stage2", "stage3", "delivery"]
    assert read_json(run / "manifest.json")["status"] == "completed"
    calls.clear()
    pipeline.run(config, existing=run)
    assert calls == []


def test_failed_visual_gate_stops_downstream_and_resume_retries_failed_stage(tmp_path, monkeypatch):
    calls = []
    stub_stages(monkeypatch, calls, fail_stage2=True)
    config = fixture_config(tmp_path)
    with pytest.raises(RuntimeError, match="visual gate"):
        pipeline.run(config)
    run = next((tmp_path / "runs").iterdir())
    assert calls == ["stage1", "stage2"]
    assert not (run / "stage3").exists()
    assert not (run / ".running.lock").exists()
    calls.clear()
    stub_stages(monkeypatch, calls)
    pipeline.run(config, existing=run)
    assert calls == ["stage2", "stage3", "delivery"]
    assert (run / "stage2/attempt_001").is_dir()
    assert (run / "stage2/attempt_002").is_dir()


@pytest.mark.parametrize("target", ["input", "artifact", "config"])
def test_resume_refuses_changed_inputs_artifacts_or_config(tmp_path, monkeypatch, target):
    stub_stages(monkeypatch, [])
    config = fixture_config(tmp_path)
    run = pipeline.run(config)
    if target == "input":
        Path(config.video).write_bytes(b"changed")
    elif target == "artifact":
        (run / "stage1/attempt_001/result").write_text("changed")
    else:
        (run / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        pipeline.run(config, existing=run)


def test_bundle_copies_absolute_assets_and_nested_include(tmp_path):
    source = tmp_path / "source"
    (source / "meshes").mkdir(parents=True)
    mesh = source / "meshes/object.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    include = source / "part.xml"
    include.write_text(f'<mujocoinclude><asset><mesh name="a" file="{mesh.as_posix()}"/></asset></mujocoinclude>')
    scene = source / "scene.xml"
    scene.write_text('<mujoco><include file="part.xml"/><worldbody/></mujoco>')
    output = tmp_path / "delivery"
    bundle_scene(scene, output)
    moved = tmp_path / "relocated"
    shutil.move(str(output), str(moved))
    shutil.rmtree(source)
    validate_bundle_paths(moved)
    assert len(list((moved / "assets").iterdir())) == 1
    assert not any(str(tmp_path) in xml.read_text() for xml in moved.glob("*.xml"))


def test_run_lock_rejects_concurrent_resume(tmp_path, monkeypatch):
    stub_stages(monkeypatch, [])
    config = fixture_config(tmp_path)
    run = pipeline.run(config)
    (run / ".running.lock").write_text("123")
    with pytest.raises(RuntimeError, match="locked"):
        pipeline.run(config, existing=run)


def test_failed_delivery_can_be_retried_without_repeating_stages(tmp_path, monkeypatch):
    calls = []
    stub_stages(monkeypatch, calls)
    config = fixture_config(tmp_path)
    deliver = pipeline._delivery
    def fail(config, output, source):
        output.mkdir()
        (output / "partial.xml").write_text("incomplete")
        raise ValueError("asset copy failed")
    monkeypatch.setattr(pipeline, "_delivery", fail)
    with pytest.raises(RuntimeError, match="asset copy failed"):
        pipeline.run(config)
    run = next((tmp_path / "runs").iterdir())
    assert not (run / "delivery").exists()
    calls.clear()
    monkeypatch.setattr(pipeline, "_delivery", deliver)
    pipeline.run(config, existing=run)
    assert calls == ["delivery"]
    assert (run / "delivery/scene.xml").is_file()
    assert (run / "stage3/delivery_attempt_001/partial.xml").is_file()


def test_actual_delivery_reloads_after_relocation_without_original(tmp_path):
    pytest.importorskip("mujoco")
    source = tmp_path / "stage3"
    source.mkdir()
    scene = source / "scene.xml"
    scene.write_text('<mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>')
    write_json(source / "final_manifest.json", {"scene": str(scene)})
    delivery = tmp_path / "delivery"
    pipeline._delivery(fixture_config(tmp_path), delivery, source)
    relocated = tmp_path / "moved"
    shutil.move(str(delivery), str(relocated))
    shutil.rmtree(source)
    assert pipeline.verify(relocated)["passed"]
    (relocated / "scene.xml").write_text("corrupt")
    assert not pipeline.verify(relocated)["passed"]


def test_bundle_rejects_cyclic_includes(tmp_path):
    source = tmp_path / "scene.xml"
    source.write_text('<mujoco><include file="scene.xml"/></mujoco>')
    with pytest.raises(ValueError, match="cyclic"):
        bundle_scene(source, tmp_path / "delivery")


@pytest.mark.parametrize("prefix,arguments", [
    (["models"], ["check", "--models", "sam3"]),
    (["perceive"], ["--video", "D:/真实视频/task clip.mp4", "--output", "D:/输出"]),
    (["tools", "stage2"], ["--phase", "geometry", "--input", "D:/场景", "--resume"]),
    (["tools", "stage3-physics"], ["grasp", "--scene", "scene.xml", "--normal-forces", "1", "3"]),
])
def test_unified_cli_preserves_tool_arguments_and_exit_status(monkeypatch, prefix, arguments):
    from types import SimpleNamespace
    from real2sim import cli

    received = []
    def child(argv):
        received.extend(argv)
        return 2
    monkeypatch.setattr(cli, "import_module", lambda _: SimpleNamespace(main=child))
    assert cli.main(prefix + arguments) == 2
    assert received == arguments


def test_unified_cli_forwards_tool_help_without_running_checks(monkeypatch, capsys):
    from real2sim import cli
    from real2sim.perception import cli as models

    def forbidden(*args, **kwargs):
        raise AssertionError("help must not check or load models")
    monkeypatch.setattr(models, "check_models", forbidden)
    with pytest.raises(SystemExit) as stopped:
        cli.main(["models", "check", "--help"])
    assert stopped.value.code == 0
    assert "--load" in capsys.readouterr().out


def test_unified_cli_refuses_unknown_workflow_options():
    from real2sim import cli

    with pytest.raises(SystemExit) as stopped:
        cli.main(["run", "--config", "scene.yaml", "--max-round", "3"])
    assert stopped.value.code == 2
