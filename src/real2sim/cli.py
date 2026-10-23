"""User-facing entry point for the complete Real-to-Sim workflow."""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import sys

from .config import SceneConfig, load_config


# Import each optional tool only when requested; basic help needs no model/GPU stack.
TOOLS = {
    "robot-surfaces": "stage1.robot_surface",
    "stage1-infer-batch": "stage1.inference_batch",
    "stage1-init-batch": "stage1.initial_pose_batch",
    "stage1-register": "stage1.sim3_registration",
    "stage1-refine": "stage1.refinement",
    "stage1-deliver": "stage1.delivery",
    "stage1-piper-final": "stage1.piper_final",
    "stage1-piper-top3": "stage1.piper_top3",
    "stage1-topk": "stage1.topk_refinement",
    "stage1-franka-rotation": "stage1.franka_rotation_refine",
    "stage1-promote-sam3": "stage1.promote_sam3_masks",
    "stage1-sam3-review": "stage1.sam3_review",
    "stage1-sam3-prompt-trial": "stage1.sam3_prompt_trial",
    "stage1-object-metrics": "stage1.object_metrics",
    "stage1-legacy": "stage1.legacy",
    "stage2": "stage2.agent_runner",
    "stage3-export": "stage3.mujoco_export",
    "stage3-batch": "stage3.batch",
    "stage3-physics": "stage3.physics_probe",
    "stage3-agent": "stage3.agent_runner",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    for name in ("resume", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True,
                             help="run directory; verify also accepts a delivery directory")
    commands.add_parser("models", add_help=False, help="check or download perception weights")
    commands.add_parser("perceive", add_help=False, help="run perception alone on a supplied video")
    tools = commands.add_parser("tools", help="stage diagnostics and historical reconstruction experiments")
    choices = tools.add_subparsers(dest="tool", required=True)
    for name in TOOLS:
        choices.add_parser(name, add_help=False)
    args, remaining = parser.parse_known_args(argv)
    if args.command in ("check", "run", "resume", "verify") and remaining:
        parser.error("unrecognized arguments: " + " ".join(remaining))
    try:
        if args.command in ("models", "perceive", "tools"):
            module = {"models": "perception.cli", "perceive": "perception.example"}.get(args.command)
            module = module or TOOLS[args.tool]
            result = import_module(f"real2sim.{module}").main(remaining)
            return int(result or 0)
        from . import pipeline
        if args.command == "check":
            report = pipeline.check_environment(load_config(args.config))
        elif args.command == "verify":
            report = pipeline.verify(args.run)
        else:
            if args.command == "resume":
                config = SceneConfig.from_snapshot(args.run / "config.json")
                folder = pipeline.run(config, existing=args.run)
            else:
                folder = pipeline.run(load_config(args.config))
            report = {"passed": True, "run": str(folder), "delivery": str(folder / "delivery")}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["passed"] else 1
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
