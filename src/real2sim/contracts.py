"""Run manifests, integrity checks and portable MuJoCo delivery bundles."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
import xml.etree.ElementTree as ET


STAGES = ("stage1", "stage2", "stage3", "delivery")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def inventory(folder: Path) -> dict[str, str]:
    return {path.relative_to(folder).as_posix(): digest(path)
            for path in sorted(folder.rglob("*")) if path.is_file()}


def verify_inventory(folder: Path, expected: dict[str, str]) -> None:
    if not expected:
        raise ValueError(f"missing artifact fingerprints: {folder}")
    if inventory(folder) != expected:
        raise ValueError(f"artifacts changed or missing: {folder}; use a new run")


def bundle_scene(scene: Path, destination: Path) -> None:
    """Copy resources by content and rewrite all runtime paths as relative paths."""
    if destination.exists():
        raise FileExistsError(f"delivery already exists: {destination}")
    destination.mkdir(parents=True)
    active: set[Path] = set()

    def copy_xml(source: Path, target: Path) -> None:
        source = source.resolve(strict=True)
        if source in active:
            raise ValueError(f"cyclic MJCF include: {source}")
        active.add(source)
        tree = ET.parse(source)
        root = tree.getroot()
        compiler = root.find("compiler")
        directories = {}
        if compiler is not None:
            assetdir = compiler.attrib.pop("assetdir", "")
            for tag, attribute in (("mesh", "meshdir"), ("texture", "texturedir")):
                directories[tag] = compiler.attrib.pop(attribute, assetdir)
        for element in root.iter():
            filename = element.get("file")
            if not filename:
                continue
            path = Path(filename)
            if not path.is_absolute():
                path = source.parent / directories.get(element.tag, "") / path
            path = path.resolve(strict=True)
            suffix = path.suffix.lower()
            name = digest(path) + suffix
            relative = Path("assets") / name
            output = destination / relative
            if element.tag == "include":
                relative = Path(name)
                output = destination / relative
                if not output.exists():
                    copy_xml(path, output)
            elif not output.exists():
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, output)
            element.set("file", relative.as_posix())
        target.parent.mkdir(parents=True, exist_ok=True)
        tree.write(target, encoding="utf-8", xml_declaration=True)
        active.remove(source)

    copy_xml(scene, destination / "scene.xml")


def validate_bundle_paths(folder: Path) -> None:
    for xml in folder.glob("*.xml"):
        root = ET.parse(xml).getroot()
        for element in root.iter():
            filename = element.get("file")
            if filename:
                path = Path(filename)
                resolved = (xml.parent / path).resolve()
                if path.is_absolute() or not resolved.is_relative_to(folder.resolve()) or not resolved.is_file():
                    raise ValueError(f"nonportable or missing resource: {filename}")
