# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical properties for articulated objects, sourced from the articulation
pipeline.

The articulation pipeline (deps/articulate-anything) is the single source of
dynamics for articulated objects. Its physics-estimation step writes, next to
each object's mobility.urdf:

  - ``physics_properties.json`` — pipeline estimates: a ``parts`` list of
    ``{"name", "mass_kg", "friction", "joint_damping"}`` (the schema stage 11
    consumes as parts_properties) and a ``joints`` mapping of joint name to
    ``{"damping", "friction"}``;
  - ``<dynamics damping friction>`` elements inside mobility.urdf itself;

and the interactive refinement UI writes user edits to
``physics_overrides.json`` (same shapes, ``parts`` keyed by link name).

Stage 11 resolves physics as: pipeline estimates, then user overrides on top.
Its own VLM estimation remains only as a fallback for results produced before
the pipeline's physics step existed. This module is stdlib-only.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

PHYSICS_FILENAME = "physics_properties.json"
OVERRIDES_FILENAME = "physics_overrides.json"

PART_KEYS = ("mass_kg", "friction", "joint_damping")
JOINT_KEYS = ("damping", "friction")

DEFAULT_JOINT_DAMPING = 0.5


def _load_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Ignoring unreadable %s: %s", path, exc)
        return None


def _clean_joint_entries(raw) -> dict:
    joints = {}
    if not isinstance(raw, dict):
        return joints
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        try:
            clean = {k: float(v) for k, v in entry.items() if k in JOINT_KEYS and v is not None}
        except (TypeError, ValueError) as exc:
            logger.warning("Ignoring joint physics entry %r: %s", name, exc)
            continue
        if clean:
            joints[name] = clean
    return joints


def load_pipeline_physics(results_dir: str) -> dict | None:
    """Load the articulation pipeline's physics_properties.json.

    Returns {"parts": [...], "joints": {...}} or None when the file is absent
    or unusable (callers then fall back to legacy estimation).
    """
    path = os.path.join(results_dir, PHYSICS_FILENAME)
    if not os.path.exists(path):
        return None
    data = _load_json(path)
    if not isinstance(data, dict):
        return None
    parts = []
    for entry in data.get("parts") or []:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        clean = {"name": str(entry["name"])}
        try:
            for key in PART_KEYS:
                if entry.get(key) is not None:
                    clean[key] = float(entry[key])
        except (TypeError, ValueError) as exc:
            logger.warning("Ignoring part physics entry %r in %s: %s", entry.get("name"), path, exc)
            continue
        parts.append(clean)
    if not parts:
        logger.warning("%s has no usable parts entries; ignoring it", path)
        return None
    return {"parts": parts, "joints": _clean_joint_entries(data.get("joints"))}


def load_physics_overrides(results_dir: str) -> dict:
    """User overrides from the refinement UI; always returns both sections."""
    overrides = {"parts": {}, "joints": {}}
    path = os.path.join(results_dir, OVERRIDES_FILENAME)
    if not os.path.exists(path):
        return overrides
    data = _load_json(path)
    if not isinstance(data, dict):
        return overrides
    raw_parts = data.get("parts")
    if isinstance(raw_parts, dict):
        for name, entry in raw_parts.items():
            if not isinstance(entry, dict):
                continue
            try:
                clean = {k: float(v) for k, v in entry.items() if k in PART_KEYS and v is not None}
            except (TypeError, ValueError) as exc:
                logger.warning("Ignoring part override %r: %s", name, exc)
                continue
            if clean:
                overrides["parts"][name] = clean
    overrides["joints"] = _clean_joint_entries(data.get("joints"))
    return overrides


def merge_parts_properties(parts_properties: list, part_overrides: dict) -> list:
    """Apply per-link user overrides to a parts_properties list (in a copy).

    Overrides for links not present in the list are appended so a user-added
    property still reaches the importer.
    """
    merged = [dict(entry) for entry in parts_properties]
    seen = set()
    for entry in merged:
        name = entry.get("name")
        seen.add(name)
        if name in part_overrides:
            entry.update(part_overrides[name])
    for name, entry in part_overrides.items():
        if name not in seen:
            merged.append({"name": name, **entry})
    return merged


def _child_to_joint_map(urdf_path: str | None) -> dict:
    """child link name -> joint name for every joint in the URDF."""
    if not urdf_path or not os.path.exists(urdf_path):
        return {}
    import xml.etree.ElementTree as ET

    mapping = {}
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as exc:
        logger.warning("Could not parse %s for joint mapping: %s", urdf_path, exc)
        return {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None and joint.attrib.get("name"):
            mapping[child.attrib.get("link", "")] = joint.attrib["name"]
    return mapping


def resolve_articulation_physics(results_dir: str, urdf_path: str | None = None,
                                 fallback_parts_fn=None):
    """Resolve an articulated object's physical properties.

    Precedence for the final per-joint dynamics (applied by
    ``resolve_joint_dynamics`` in the importer):
    user overrides (physics_overrides.json, incl. the UI parts table's
    joint_damping lifted onto the child's joint) > values already in the
    URDF's <dynamics> (pipeline-authored or hand-edited) > pipeline estimates
    (physics_properties.json, ``joint_defaults``) > legacy per-part/constant
    defaults. ``fallback_parts_fn`` (legacy stage-11 VLM estimation) is
    invoked only when the pipeline file is missing/unusable.

    Returns (parts_properties, joint_overrides, joint_defaults, source) where
    source is "articulation_pipeline" or "legacy_fallback".
    """
    physics = load_pipeline_physics(results_dir)
    if physics is not None:
        parts = physics["parts"]
        joint_defaults = {name: dict(entry) for name, entry in physics["joints"].items()}
        source = "articulation_pipeline"
    else:
        parts = list(fallback_parts_fn()) if fallback_parts_fn is not None else []
        joint_defaults = {}
        source = "legacy_fallback"

    overrides = load_physics_overrides(results_dir)
    parts = merge_parts_properties(parts, overrides["parts"])

    joint_overrides = {name: dict(entry) for name, entry in overrides["joints"].items()}
    # The refinement UI's parts table exposes joint damping per child link;
    # lift those onto the child's joint so they actually win (an explicit
    # joints-section damping override still takes precedence).
    child_to_joint = _child_to_joint_map(urdf_path)
    for link_name, entry in overrides["parts"].items():
        if "joint_damping" in entry:
            joint_name = child_to_joint.get(link_name)
            if joint_name:
                joint_overrides.setdefault(joint_name, {}).setdefault(
                    "damping", entry["joint_damping"])
    return parts, joint_overrides, joint_defaults, source


def resolve_joint_dynamics(joint_type: str, existing_attrib: dict | None,
                           child_props: dict, override: dict | None,
                           revolute_friction: float, prismatic_friction: float,
                           defaults_entry: dict | None = None):
    """Final (damping, friction) for one movable joint in the sim-ready URDF.

    Precedence: user override > value already in the URDF's <dynamics>
    (pipeline-authored or hand-edited — preserved, never overwritten by
    estimates) > pipeline estimate (``defaults_entry``) > legacy child-link
    joint_damping / joint-type friction constants.
    """
    existing = existing_attrib or {}
    defaults_entry = defaults_entry or {}

    def existing_float(key):
        try:
            return float(existing[key]) if key in existing else None
        except (TypeError, ValueError):
            return None

    damping = existing_float("damping")
    if damping is None:
        damping = defaults_entry.get("damping")
    if damping is None:
        damping = child_props.get("joint_damping", DEFAULT_JOINT_DAMPING)
    friction = existing_float("friction")
    if friction is None:
        friction = defaults_entry.get("friction")
    if friction is None:
        friction = revolute_friction if joint_type == "revolute" else prismatic_friction
    override = override or {}
    damping = override.get("damping", damping)
    friction = override.get("friction", friction)
    return float(damping), float(friction)
