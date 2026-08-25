# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Start a new scene by composing it from one that already works.

A template scene supplies the room, the robot, the ground plane and the
version block; the caller chooses which of its editable props to carry over.
Dropped props leave both halves of the document (via
``scene_io.remove_objects``), the template's embedded ``scene_file`` snapshot
is discarded, and every serialized velocity — ``lin_vel``, ``ang_vel`` and
``joint_vel`` — is zeroed so nothing drifts or slides when physics starts.
"""

import copy
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from scene_io import (
    DEFAULT_DATASET_DIR,
    DEFAULT_ROBOT_ASSET_DIR,
    SceneEditError,
    atomic_write_text,
    background_id,
    editable_object_names,
    iter_objects,
    remove_objects,
)

#: Structure a composed scene inherits wholesale, and cannot be a scene
#: without. Ground plane is deliberately absent: a mesh room's triangles are
#: colliders, so requiring one would refuse templates that work.
REQUIRED_STRUCTURE = ("versions", "robot", "background")

#: Scene names become directory names, filename stems and `--scene-name`
#: arguments, so they are held to what all three accept without quoting.
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,62}[a-z0-9]$")

MAX_NAME_LENGTH = 64


def validate_scene_name(name):
    """Check a proposed scene name, or say precisely what is wrong with it.

    Raises:
        SceneEditError: With a message meant to be shown in the wizard.
    """
    if not isinstance(name, str) or not name.strip():
        raise SceneEditError("a scene name is required")
    name = name.strip()
    if len(name) > MAX_NAME_LENGTH:
        raise SceneEditError(
            f"scene name is {len(name)} characters; keep it under {MAX_NAME_LENGTH}"
        )
    if not NAME_RE.match(name):
        raise SceneEditError(
            f"{name!r} is not a usable scene name — use lower-case letters, digits "
            "and underscores, starting with a letter (e.g. droid_desk_sort_tools)"
        )
    return name


def template_summary(template_json, *, dataset_dir=None, robot_asset_dir=None):
    """What a new composition would inherit from *template_json*.

    Args:
        template_json (str or Path): Scene to inherit from.
        dataset_dir (str or Path or None): OmniGibson's ``gm.DATA_PATH``.
            Without it a ``DatasetObject``'s asset cannot be looked for, and
            the row says so rather than claiming the object is resolvable.
        robot_asset_dir (str or Path or None): Root of
            ``omnigibson-robot-assets``, for the same reason.

    Returns:
        dict: ``scene``, ``name``, ``background``, ``robot``, ``objects`` — one
        row per object with ``name``, ``category``, ``kind``, ``editable``,
        ``resolvable`` (its asset is on this machine), ``asset_status``
        (``none``/``ok``/``missing``/``unchecked``) and ``asset_detail`` — plus
        ``has_versions``, ``has_ground_plane`` and ``missing_structure``.

    Raises:
        SceneEditError: If the template cannot be read or carries no room.
    """
    path = Path(template_json)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SceneEditError(f"could not read template {path.name}: {e}") from None
    if not isinstance(document, dict):
        raise SceneEditError(f"{path.name} is not a scene document")

    records = list(iter_objects(document, path, robot_asset_dir=robot_asset_dir,
                                dataset_dir=dataset_dir, usd_facts=False))
    objects = [
        {
            "name": record["name"],
            "category": record["category"],
            "kind": record["kind"],
            "editable": record["editable"],
            # Only "missing" blocks a carry-over; "unchecked" means no root was
            # given to look in, which is not the template's fault.
            "resolvable": record["usd_status"] != "missing",
            "asset_status": record["usd_status"],
            "asset_detail": record["usd_detail"],
        }
        for record in records
    ]
    return {
        "scene": str(path),
        "name": path.stem.split("_scene_state_")[0],
        "background": background_id(document, path),
        "robot": next((r["category"] for r in records if r["kind"] == "robot"), None),
        "objects": objects,
        "has_versions": bool(document.get("versions")),
        "has_ground_plane": bool(document.get("ground_plane_info")),
        # What `compose_scene` would refuse this template for.
        "missing_structure": _missing_structure(document, records),
    }


def _missing_structure(document, records):
    """Which of :data:`REQUIRED_STRUCTURE` this template does not supply."""
    present = {
        "versions": bool(document.get("versions")),
        "robot": any(r["kind"] == "robot" for r in records),
        "background": any(r["kind"] == "background" for r in records),
    }
    return [name for name in REQUIRED_STRUCTURE if not present[name]]


def _zero_velocities(scene):
    """Clear every serialized velocity in the document.

    ``joint_vel`` hangs off the registry entry itself, not ``root_link``, so
    an object whose base is at rest can still carry per-joint motion.

    Returns:
        int: How many objects carried a live velocity and so were changed.
    """
    registry = scene.get("state", {}).get("registry", {}).get("object_registry", {})
    touched = 0
    for entry in registry.values():
        changed = False
        link = entry.get("root_link")
        if isinstance(link, dict):
            for field in ("lin_vel", "ang_vel"):
                if any(abs(float(v)) > 0 for v in link.get(field, []) or []):
                    changed = True
                if field in link:
                    link[field] = [0.0, 0.0, 0.0]
        joints = entry.get("joint_vel")
        if isinstance(joints, list):
            if any(abs(float(v)) > 0 for v in joints):
                changed = True
            entry["joint_vel"] = [0.0] * len(joints)
        touched += 1 if changed else 0
    return touched


def _drop_nested_scene_file(scene):
    """Drop the template's recursive scene snapshot from ``init_info.args``.

    ``og.sim.save()`` embeds the whole document it loaded under
    ``init_info.args.scene_file``; for a composition that snapshot describes a
    different scene, naming dropped props at another author's absolute paths.
    The key is set to ``None`` rather than deleted: ``None`` is the
    constructor's own default and no consumer reads the value back.

    Args:
        scene (dict): Scene document. Mutated in place.

    Returns:
        bool: True if a snapshot was dropped.
    """
    args = scene.get("init_info", {}).get("args")
    if isinstance(args, dict) and args.get("scene_file") is not None:
        args["scene_file"] = None
        return True
    return False


def _relative_asset_paths(scene, template_json):
    """Scene-relative ``usd_path`` values that live *inside* the scene directory.

    A background referenced as ``../../mesh_backgrounds/<room>.usd`` is a
    shared library asset that resolves unchanged from a sibling directory, so
    only paths that stay inside the scene folder are copied.

    Returns:
        tuple[set[Path], list[str]]: (relative directories to copy, paths that
        escape the scene directory and are therefore left where they are).
    """
    scene_dir = Path(template_json).parent
    inside, outside = set(), []
    for info in scene.get("objects_info", {}).get("init_info", {}).values():
        raw = (info.get("args") or {}).get("usd_path")
        if not raw:
            continue
        candidate = Path(raw)
        if candidate.is_absolute() or ".." in candidate.parts:
            outside.append(raw)
            continue
        # Copy the bundle, not the file: a .usd alone leaves ../material/*.png
        # dangling.
        parts = candidate.parts
        if len(parts) >= 2 and parts[-2] == "usd":
            inside.add(Path(*parts[:-2]))
        else:
            inside.add(candidate.parent if len(parts) > 1 else candidate)
    # A parent already being copied makes its children redundant.
    minimal = {p for p in inside if not any(q != p and q in p.parents for q in inside)}
    return minimal, outside


def _verify_resolvable(scene, scene_json, *, dataset_dir=None, robot_asset_dir=None):
    """Every asset a composed scene names must be on this machine.

    Checks every object, not only those with a ``usd_path``: a
    ``DatasetObject`` and a robot name their geometry by class. ``unchecked``
    is not refused — an asset with no root to look in is not an asset known to
    be absent.

    Returns:
        list[str]: Objects whose asset could not be checked, for the caller to
        report. Empty when everything resolved.

    Raises:
        SceneEditError: Naming every object whose asset is missing.
    """
    records = list(iter_objects(scene, scene_json, robot_asset_dir=robot_asset_dir,
                                dataset_dir=dataset_dir, usd_facts=False))
    missing = [f"{r['name']} ({r['usd_detail']})"
               for r in records if r["usd_status"] == "missing"]
    if missing:
        raise SceneEditError(
            "the composed scene references assets that are not on disk: "
            + "; ".join(sorted(missing))
        )
    return sorted(r["name"] for r in records if r["usd_status"] == "unchecked")


def compose_scene(template_json, name, keep, dest_root=None, *,
                  dataset_dir=None, robot_asset_dir=None):
    """Write a new scene composed from *template_json*.

    Args:
        template_json (str or Path): Scene to inherit room, robot, ground plane
            and version block from.
        name (str): New scene name; becomes the directory and the filename stem.
        keep (iterable[str]): Editable objects from the template to carry over.
            Everything else editable is dropped. The robot and the background
            are always kept — a scene without them cannot be evaluated or laid
            out against.
        dest_root (str or Path or None): Where the scene directory is created.
            Defaults to the template's own parent, which is what keeps a
            background referenced as ``../../mesh_backgrounds/...`` resolving.
        dataset_dir (str or Path or None): OmniGibson's ``gm.DATA_PATH``, so an
            implicitly-named ``DatasetObject`` asset can be checked for.
        robot_asset_dir (str or Path or None): Root of
            ``omnigibson-robot-assets``, likewise.

    Returns:
        dict: ``path``, ``dir``, ``name``, ``kept``, ``dropped``, ``copied``,
        ``shared``, ``zeroed`` and ``unchecked`` (objects whose asset could not
        be looked for, because no root was given for it).

    Raises:
        SceneEditError: On a bad name, an existing directory, an unknown object
            to keep, a template that does not supply
            :data:`REQUIRED_STRUCTURE`, or a result that would not load.
    """
    template = Path(template_json).resolve()
    name = validate_scene_name(name)
    root = Path(dest_root).resolve() if dest_root else template.parent.parent
    destination = root / name
    if destination.exists():
        raise SceneEditError(f"{destination} already exists; choose another name")
    if not root.is_dir():
        raise SceneEditError(f"scene root does not exist: {root}")

    try:
        scene = json.loads(template.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SceneEditError(f"could not read template {template.name}: {e}") from None
    if not isinstance(scene, dict):
        raise SceneEditError(f"{template.name} is not a scene document")

    # Refused before anything is copied or written.
    records = list(iter_objects(scene, template, robot_asset_dir=robot_asset_dir,
                                dataset_dir=dataset_dir, usd_facts=False))
    absent = _missing_structure(scene, records)
    if absent:
        raise SceneEditError(
            f"{template.name} cannot be a template: it has no "
            + ", ".join(absent)
            + ". A new scene inherits its room, its robot and its version block "
              "from the template, so a template without them produces a document "
              "that looks like a scene and is not one."
        )

    editable = editable_object_names(scene, template)
    keep = {str(n) for n in keep}
    unknown = keep - editable
    if unknown:
        raise SceneEditError(
            "cannot keep object(s) the template does not have, or that are not "
            f"editable: {', '.join(sorted(unknown))}"
        )
    dropped = sorted(editable - keep)
    remove_objects(scene, dropped, removable_names=editable)

    # The template's snapshot describes the template, not this scene.
    _drop_nested_scene_file(scene)
    zeroed = _zero_velocities(scene)

    metadata = scene.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = scene["metadata"] = {}
    # Additive provenance record: where the composition came from.
    metadata["composed_from"] = {
        "template": str(template),
        "template_scene": template.stem.split("_scene_state_")[0],
        "kept": sorted(keep),
        "dropped": dropped,
        "created": datetime.now().isoformat(timespec="seconds"),
        "tool": "light_editor/compose.py",
    }

    bundles, shared = _relative_asset_paths(scene, template)

    # Built aside and renamed into place, so a failure halfway through leaves
    # no half-populated scene directory.
    staging = root / f".{name}.partial-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    try:
        staging.mkdir(parents=True)
        copied = []
        for relative in sorted(bundles):
            source = template.parent / relative
            if not source.exists():
                # Nothing kept references it, or the template is incomplete;
                # the resolvability check below decides.
                continue
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
            copied.append(str(relative))

        scene_path = staging / f"{name}_scene_state_latest.json"
        atomic_write_text(scene_path, json.dumps(scene, indent=4, allow_nan=False))
        # Verified against the staged layout; staging is renamed into place
        # only if this passes.
        unchecked = _verify_resolvable(
            scene, scene_path, dataset_dir=dataset_dir, robot_asset_dir=robot_asset_dir)
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "path": str(destination / f"{name}_scene_state_latest.json"),
        "dir": str(destination),
        "name": name,
        "kept": sorted(keep),
        "dropped": dropped,
        "copied": copied,
        "shared": shared,
        "zeroed": zeroed,
        "unchecked": unchecked,
        "template": str(template),
    }
