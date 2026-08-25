# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare an authored scene against what a simulator actually loaded.

Has no OmniGibson import so the verdict logic is unit-testable offline;
``parity_check.py`` supplies the observations. Handles quaternion double cover
(``q`` and ``-q`` are the same rotation), checks orientation and scale as well
as position, and reports objects missing from either side rather than skipping
them.
"""

import math

# Positions are metres; 1 mm is far above serialization noise and far below a
# visible mismatch.
DEFAULT_POSITION_TOL = 1e-3
DEFAULT_ANGLE_TOL_DEG = 0.5
DEFAULT_SCALE_TOL = 1e-4


def quaternion_angle_deg(a, b):
    """Angle between two XYZW quaternions, in degrees, ignoring sign.

    Uses ``abs(dot)`` so ``q`` and ``-q`` -- the same rotation -- compare equal.
    Both inputs are normalised first: serialized quaternions are only
    unit-length to about 1e-5, enough to fail a half-degree gate otherwise.
    """
    def unit(q):
        norm = math.sqrt(sum(v * v for v in q))
        if norm <= 1e-12:
            raise ValueError("cannot compare a zero-length quaternion")
        return [v / norm for v in q]

    dot = sum(x * y for x, y in zip(unit(a), unit(b)))
    # Clamp so accumulated float error cannot push acos out of its domain.
    dot = max(-1.0, min(1.0, abs(dot)))
    return math.degrees(2.0 * math.acos(dot))


def compare_object(authored, observed, *, position_tol=DEFAULT_POSITION_TOL,
                   angle_tol_deg=DEFAULT_ANGLE_TOL_DEG, scale_tol=DEFAULT_SCALE_TOL):
    """Compare one object's authored transform against what loaded.

    Args:
        authored (dict): ``{"position": [3], "orientation": [4], "scale": [3]}``.
            Any key may be omitted to skip that comparison.
        observed (dict): Same shape, as read back from the simulator.

    Returns:
        dict: ``{"ok": bool, "failures": [str], "position_delta": float,
        "angle_deg": float, "scale_delta": float}``. Metrics are None when the
        corresponding field was not supplied.
    """
    failures = []
    result = {"position_delta": None, "angle_deg": None, "scale_delta": None}

    if "position" in authored and "position" in observed:
        delta = math.dist(authored["position"], observed["position"])
        result["position_delta"] = delta
        if delta > position_tol:
            failures.append(f"position off by {delta:.6f} m (tolerance {position_tol})")

    if "orientation" in authored and "orientation" in observed:
        angle = quaternion_angle_deg(authored["orientation"], observed["orientation"])
        result["angle_deg"] = angle
        if angle > angle_tol_deg:
            failures.append(f"orientation off by {angle:.4f} deg (tolerance {angle_tol_deg})")

    if "scale" in authored and "scale" in observed:
        delta = max(abs(a - b) for a, b in zip(authored["scale"], observed["scale"]))
        result["scale_delta"] = delta
        if delta > scale_tol:
            failures.append(f"scale off by {delta:.6g} per-axis (tolerance {scale_tol})")

    result["ok"] = not failures
    result["failures"] = failures
    return result


def compare_scene(authored, observed, **tolerances):
    """Compare whole authored and observed scenes, object sets included.

    Args:
        authored (dict): ``{name: transform}`` as authored.
        observed (dict): ``{name: transform}`` as loaded.

    Returns:
        dict: ``ok``, ``missing`` (authored but absent), ``unexpected``
        (present but not authored), and ``objects`` keyed by name.
    """
    missing = sorted(set(authored) - set(observed))
    unexpected = sorted(set(observed) - set(authored))

    objects = {}
    for name in sorted(set(authored) & set(observed)):
        objects[name] = compare_object(authored[name], observed[name], **tolerances)

    return {
        # The set comparison gates the verdict alongside the per-object checks.
        "ok": not missing and not unexpected and all(o["ok"] for o in objects.values()),
        "missing": missing,
        "unexpected": unexpected,
        "objects": objects,
    }


def authored_transforms(scene):
    """Extract ``{name: transform}`` from a scene-state document.

    Scale lives in ``objects_info.init_info`` and pose in the state registry, so
    an object present in one and not the other is reported with whatever it has
    rather than being skipped.
    """
    registry = scene.get("state", {}).get("registry", {}).get("object_registry", {})
    init_info = scene.get("objects_info", {}).get("init_info", {})

    transforms = {}
    for name in set(registry) | set(init_info):
        entry = {}
        root = registry.get(name, {}).get("root_link", {})
        if "pos" in root:
            entry["position"] = [float(v) for v in root["pos"]]
        if "ori" in root:
            entry["orientation"] = [float(v) for v in root["ori"]]
        scale = init_info.get(name, {}).get("args", {}).get("scale")
        if isinstance(scale, (int, float)):
            entry["scale"] = [float(scale)] * 3
        elif isinstance(scale, (list, tuple)) and len(scale) == 3:
            entry["scale"] = [float(v) for v in scale]
        if entry:
            transforms[name] = entry
    return transforms


def format_report(comparison):
    """Render a comparison as readable lines."""
    lines = []
    for name in comparison["missing"]:
        lines.append(f"{name:28s} MISSING -- authored but did not load")
    for name in comparison["unexpected"]:
        lines.append(f"{name:28s} UNEXPECTED -- loaded but not authored")
    for name, result in comparison["objects"].items():
        pos = result["position_delta"]
        ang = result["angle_deg"]
        scl = result["scale_delta"]
        status = "ok" if result["ok"] else "MISMATCH"
        lines.append(
            f"{name:28s} {'-' if pos is None else format(pos, '.6f'):>10s} m "
            f"{'-' if ang is None else format(ang, '.4f'):>9s} deg "
            f"{'-' if scl is None else format(scl, '.2e'):>10s}  {status}"
        )
        for failure in result["failures"]:
            lines.append(f"{'':28s}   {failure}")
    return "\n".join(lines)
