# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Read and write OmniGibson ``external_sensors`` camera configs
(``scripts/cfg/external_sensors/*.yaml``), a property of the robot rig, not of
a scene.

Poses in these files are **relative to the parent prim**, not world::

    relative_prim_path: /controllable__frankapanda__robot0/panda_link0/external_cam0_left
    position: [-0.067, 0.6119, 0.4202]
    orientation: [-0.3010, 0.5097, 0.6943, -0.4094]   # (x, y, z, w)
    pose_frame: parent

The browser parents each camera to the robot, so a camera's local transform is
written back here with no frame conversion. Only ``position`` and
``orientation`` are ever rewritten; optics, modalities, prim paths and
resolution pass through untouched.
"""

import math
from datetime import datetime
from pathlib import Path

import yaml

from scene_io import SceneEditError, _validated_vector, atomic_write_text

# Where the pipeline looks up configs by bare name
# (``s15_eval.external_sensors_cfg=<name>``).
CFG_SUBDIR = "scripts/cfg/external_sensors"


def resolve_camera_config(name_or_path, repo_root):
    """Resolve a camera config given either a bare config name or a path.

    Args:
        name_or_path (str): e.g. ``nv_franka_droid`` or an explicit path.
        repo_root (Path): Repository root.

    Returns:
        Path: Existing config file.

    Raises:
        SceneEditError: If nothing matches.
    """
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml") and candidate.exists():
        return candidate.resolve()

    stem = candidate.stem if candidate.suffix else str(name_or_path)
    resolved = repo_root / CFG_SUBDIR / f"{stem}.yaml"
    if resolved.exists():
        return resolved.resolve()
    raise SceneEditError(
        f"camera config not found: {name_or_path} (looked in {repo_root / CFG_SUBDIR})"
    )


def camera_config_paths(repo_root, template, *, background=None, scene_name="",
                        explicit_out=None):
    """Decide where a placement is saved, and what to resume from.

    Placements are keyed by background id: poses are relative to the robot
    base, so re-opening any scene built in the same background resumes the
    authored poses rather than the rig template.

    Args:
        repo_root (Path): Repository root; configs live under :data:`CFG_SUBDIR`.
        template (str or Path): Config ``--cameras`` named, used as a starting
            point only when nothing has been authored yet.
        background (str or None): Background id from
            :func:`scene_io.background_id`.
        scene_name (str): Scene name, used when the background is unidentifiable.
        explicit_out (str or None): ``--cameras-out``, which overrides the key.

    Returns:
        tuple[Path, Path]: ``(out_path, source_path)`` — where a save lands, and
        the config the editor should load now.
    """
    cfg_dir = Path(repo_root) / CFG_SUBDIR
    template = Path(template)

    if explicit_out:
        out = cfg_dir / f"{Path(explicit_out).stem}.yaml"
        candidates = [out]
    else:
        key = background or scene_name or template.stem
        out = cfg_dir / f"{key}_cameras.yaml"
        candidates = [out]
        # Older saves were keyed by scene name; still resume from them.
        legacy = cfg_dir / f"{scene_name}_cameras.yaml"
        if scene_name and legacy != out:
            candidates.append(legacy)

    for candidate in candidates:
        if candidate.exists():
            return out, candidate.resolve()
    return out, template.resolve()


def camera_export_path(repo_root, name):
    """Where an explicitly named export lands.

    The name comes from a text field, so it is reduced to one safe filename
    component — which is also a name ``--external_sensors_cfg`` can take.

    Raises:
        SceneEditError: If nothing usable is left of the name.
    """
    stem = Path(str(name)).stem
    cleaned = "".join(c if (c.isalnum() or c in "_-") else "_" for c in stem).strip("_-")
    if not cleaned:
        raise SceneEditError(f"{name!r} is not a usable config name")
    return Path(repo_root) / CFG_SUBDIR / f"{cleaned}.yaml"


def _fov_degrees(kwargs):
    """Horizontal/vertical field of view from USD-style optics.

    Returns:
        tuple[float, float] or tuple[None, None]: (horizontal, vertical) degrees.
    """
    focal = kwargs.get("focal_length")
    aperture = kwargs.get("horizontal_aperture")
    width = kwargs.get("image_width")
    height = kwargs.get("image_height")
    if not focal or not aperture:
        return None, None

    h_fov = 2.0 * math.degrees(math.atan(float(aperture) / (2.0 * float(focal))))
    if not width or not height:
        return h_fov, None
    aspect = float(width) / float(height)
    v_fov = 2.0 * math.degrees(
        math.atan(math.tan(math.radians(h_fov) / 2.0) / aspect)
    )
    return h_fov, v_fov


def load_cameras(config_path):
    """Read a camera config.

    Args:
        config_path (str or Path): Path to an external_sensors YAML.

    Returns:
        tuple[list[dict], dict]: (camera records for the browser, raw document)

    Raises:
        SceneEditError: If the document is not a usable external_sensors config.
    """
    path = Path(config_path)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise SceneEditError(f"could not read {path}: {e}") from e

    if not isinstance(document, dict) or "external_sensors" not in document:
        raise SceneEditError(f"{path} has no top-level 'external_sensors' key")

    sensors = document.get("external_sensors")
    if not isinstance(sensors, list):
        raise SceneEditError(f"{path}: 'external_sensors' must be a list")

    cameras = []
    seen_names = set()
    parents = set()
    for index, sensor in enumerate(sensors):
        if not isinstance(sensor, dict):
            raise SceneEditError(f"{path}: sensor #{index} is not a mapping")
        name = str(sensor.get("name") or f"sensor_{index}")
        kwargs = sensor.get("sensor_kwargs") or {}
        h_fov, v_fov = _fov_degrees(kwargs)

        # The browser writes a camera's *local* transform straight back, so
        # anything this editor cannot represent in that frame is refused
        # rather than silently serialized in the wrong frame.
        sensor_type = sensor.get("sensor_type", "VisionSensor")
        if sensor_type != "VisionSensor":
            raise SceneEditError(
                f"{path}: {name} has sensor_type={sensor_type!r}; this editor only "
                "places VisionSensor entries"
            )
        if name in seen_names:
            raise SceneEditError(f"{path}: duplicate sensor name {name!r}")
        seen_names.add(name)

        pose_frame = sensor.get("pose_frame", "parent")
        if pose_frame != "parent":
            raise SceneEditError(
                f"{path}: {name} has pose_frame={pose_frame!r}; this editor only handles "
                "'parent', because that is the frame it writes back unconverted"
            )

        prim_path = str(sensor.get("relative_prim_path") or "")
        segments = [s for s in prim_path.split("/") if s]
        if not prim_path.startswith("/") or len(segments) < 2:
            raise SceneEditError(
                f"{path}: {name} needs an absolute relative_prim_path of at least "
                f"/<parent>/<link>/<sensor>; got {prim_path!r}"
            )
        parents.add((segments[0], segments[1]))
        if len(parents) > 1:
            joined = ", ".join(f"/{a}/{b}" for a, b in sorted(parents))
            raise SceneEditError(
                f"{path}: cameras hang off more than one parent link ({joined}); this "
                "editor parents every camera to one object and would place the rest wrong"
            )

        raw_pos = sensor.get("position", [0.0, 0.0, 0.0])
        raw_ori = sensor.get("orientation", [0.0, 0.0, 0.0, 1.0])
        cameras.append({
            "index": index,
            "name": name,
            "position": _validated_vector(name, "position", list(raw_pos), 3),
            "orientation": _validated_vector(
                name, "orientation", list(raw_ori), 4, normalize=True
            ),
            "relative_prim_path": sensor.get("relative_prim_path", ""),
            # 'parent': the pose is relative to relative_prim_path's parent.
            "pose_frame": sensor.get("pose_frame", "parent"),
            "image_width": kwargs.get("image_width"),
            "image_height": kwargs.get("image_height"),
            "h_fov_deg": h_fov,
            "v_fov_deg": v_fov,
            # Shown in the preview tooltip.
            "modalities": list(sensor["modalities"])
            if isinstance(sensor.get("modalities"), (list, tuple)) else [],
            # Passed through so the browser clips where the sensor does; when
            # absent, the viewer picks a room-scale range of its own.
            "clipping_range": list(kwargs["clipping_range"])
            if isinstance(kwargs.get("clipping_range"), (list, tuple))
            and len(kwargs["clipping_range"]) == 2 else None,
        })
    return cameras, document


# --- which policy observation key each camera fills --------------------------
# The evaluation stage reads two camera *names* out of the task config
# (base_camera_1_name / base_camera_2_name) and falls back to config order
# when a name is absent, so the mapping is worked out from the task configs
# rather than assumed from order.

#: DROID observation keys filled from external sensors, in the order the
#: evaluation stage falls back to when a task config names nothing.
OBSERVATION_KEYS = ("exterior_image_1_left", "exterior_image_2_left")

#: Task configs live here, one per task, alongside the pipeline's own settings.
TASK_CFG_SUBDIR = "scripts/cfg"

_TASK_CAMERA_FIELDS = ("base_camera_1_name", "base_camera_2_name")


def read_task_camera_bindings(repo_root):
    """Read every task config's external-camera binding.

    Args:
        repo_root (str or Path): Repository root; configs live under
            :data:`TASK_CFG_SUBDIR`.

    Returns:
        list[dict]: One ``{"name", "sensors_cfg", "cameras"}`` per task config
        that names at least one external camera, sorted by file name.
    """
    cfg_dir = Path(repo_root) / TASK_CFG_SUBDIR
    bindings = []
    for path in sorted(cfg_dir.glob("*.yaml")):
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            # An unparseable task config contributes no binding.
            continue
        if not isinstance(document, dict):
            continue
        section = document.get("s15_eval")
        if not isinstance(section, dict):
            continue
        cameras = [section.get(field) for field in _TASK_CAMERA_FIELDS]
        if not any(cameras):
            continue
        bindings.append({
            "name": path.stem,
            "sensors_cfg": section.get("external_sensors_cfg"),
            "cameras": [str(c) if c else None for c in cameras],
        })
    return bindings


def _assign_observation_keys(camera_names, binding):
    """Emulate the evaluation stage's own lookup for one task config.

    Args:
        camera_names (list[str]): Sensor names, in config order.
        binding (dict or None): One entry from
            :func:`read_task_camera_bindings`, or None for the bare
            config-order fallback.

    Returns:
        dict: ``{observation_key: camera_name}``, omitting keys nothing fills.
    """
    wanted = binding["cameras"] if binding else []
    assignment = {}
    for index, key in enumerate(OBSERVATION_KEYS):
        want = wanted[index] if index < len(wanted) else None
        if want and want in camera_names:
            assignment[key] = want
        elif len(camera_names) > index:
            assignment[key] = camera_names[index]
    return assignment


def _named(configs, limit=3):
    """Name a few task configs and count the rest — a tooltip, not a report."""
    if len(configs) <= limit:
        return ", ".join(configs)
    return f"{', '.join(configs[:limit])} and {len(configs) - limit} more"


def _short_key(keys):
    """``exterior_image_1_left`` -> ``ext 1``; several -> ``ext 1+2``."""
    numbers = [
        str(OBSERVATION_KEYS.index(k) + 1) for k in OBSERVATION_KEYS if k in keys
    ]
    return f"ext {'+'.join(numbers)}" if numbers else ""


def observation_key_map(camera_names, repo_root, *, sensors_cfg=None, task_cfg=None):
    """Work out which policy observation key each camera fills.

    Args:
        camera_names (list[str]): Sensor names in config order — the order the
            evaluation stage falls back to.
        repo_root (str or Path): Repository root.
        sensors_cfg (str or None): Stem of the rig config these cameras came
            from (``nv_franka_droid``). Task configs that select a different rig
            are ignored when any config selects this one.
        task_cfg (str or None): Pin the answer to one task config, which is the
            way out when several disagree.

    Returns:
        dict: ``{"cameras": {name: {"key", "keys", "short", "certain",
        "detail"}}, "note": str or None, "task_configs": [...],
        "pinned": str or None}``.

    Raises:
        SceneEditError: If ``task_cfg`` names a config that does not exist or
            binds no cameras.
    """
    bindings = read_task_camera_bindings(repo_root)

    pinned = None
    if task_cfg:
        pinned = Path(task_cfg).stem
        candidates = [b for b in bindings if b["name"] == pinned]
        if not candidates:
            known = ", ".join(b["name"] for b in bindings) or "none"
            raise SceneEditError(
                f"task config {pinned!r} names no external cameras "
                f"(configs that do: {known})"
            )
    else:
        selecting = [b for b in bindings if sensors_cfg and b["sensors_cfg"] == sensors_cfg]
        # Fall back to every config: a saved placement is named
        # <background>_cameras, which no task config selects, and the bindings
        # are by camera *name* anyway.
        candidates = selecting or bindings

    # When no task config binds cameras, config order is the whole answer.
    per_config = [(b["name"] if b else "config order", _assign_observation_keys(camera_names, b))
                  for b in (candidates or [None])]

    # Group by key set so the detail line names who says what; the same key
    # set reached by several configs is agreement, not a conflict.
    cameras = {}
    conflicts = []
    for name in camera_names:
        by_keys = {}
        for config_name, assignment in per_config:
            keys = tuple(k for k in OBSERVATION_KEYS if assignment.get(k) == name)
            by_keys.setdefault(keys, []).append(config_name)
        # Most-supported key set wins; ties break on config name.
        ranked = sorted(by_keys.items(), key=lambda kv: (-len(kv[1]), kv[1][0]))
        keys, supporters = ranked[0]
        certain = len(ranked) == 1
        if not certain:
            conflicts.append(name)
        detail = "; ".join(
            f"{_short_key(ks) or 'unused'} per {_named(who)}" for ks, who in ranked
        )
        cameras[name] = {
            "key": keys[0] if keys else None,
            "keys": list(keys),
            "short": _short_key(keys),
            "certain": certain,
            "detail": detail,
            "supporters": supporters,
        }

    note = None
    if pinned:
        note = f"observation keys pinned to task config {pinned}"
    elif conflicts:
        note = (
            f"task configs disagree on {', '.join(conflicts)} — "
            "pass --task-cfg <name> to pin one"
        )
    # One camera filling both keys means the policy sees the same picture twice.
    doubled = [n for n, info in cameras.items() if len(info["keys"]) > 1]
    if doubled and not conflicts:
        note = f"{', '.join(doubled)} fills both exterior keys — the policy sees it twice"

    return {
        "cameras": cameras,
        "note": note,
        "task_configs": [name for name, _ in per_config],
        "pinned": pinned,
    }


def validate_camera_edits(cameras, edits):
    """Validate a browser camera payload against the loaded config.

    Args:
        cameras (list[dict]): Records from :func:`load_cameras`.
        edits (dict): ``{name: {"position": [3], "orientation": [4]}}``.

    Returns:
        dict: Normalized edits.

    Raises:
        SceneEditError: On unknown names or malformed vectors.
    """
    if not isinstance(edits, dict):
        raise SceneEditError("camera edits must be an object")

    known = {camera["name"] for camera in cameras}
    unknown = set(edits) - known
    if unknown:
        raise SceneEditError(f"unknown camera(s): {', '.join(sorted(unknown))}")

    normalized = {}
    for name, edit in edits.items():
        if not isinstance(edit, dict):
            raise SceneEditError(f"{name}: edit must be an object")
        normalized[name] = {
            "position": _validated_vector(name, "position", edit.get("position"), 3),
            "orientation": _validated_vector(
                name, "orientation", edit.get("orientation"), 4, normalize=True
            ),
        }
    return normalized


# Below this a pose counts as unchanged, so quaternion normalization
# (hand-authored configs are unit-length only to ~1e-5) is not reported as an edit.
POSE_EPSILON = 1e-4

# Written precision: 1e-6 m, far finer than any camera mount tolerance.
POSE_DECIMALS = 6


def _rounded(values):
    return [round(float(v), POSE_DECIMALS) for v in values]


def _pose_differs(before, after):
    if len(before) != len(after):
        return True
    return any(abs(float(a) - float(b)) > POSE_EPSILON for a, b in zip(before, after))


def apply_camera_edits(document, edits):
    """Rewrite poses in a loaded config, in place.

    Everything other than ``position`` and ``orientation`` is left untouched, so
    optics, modalities, prim paths and resolution survive verbatim.

    Returns:
        list[str]: Names whose pose moved by more than :data:`POSE_EPSILON`.
    """
    changed = []
    for sensor in document.get("external_sensors", []):
        name = str(sensor.get("name", ""))
        edit = edits.get(name)
        if edit is None:
            continue
        position = _rounded(edit["position"])
        orientation = _rounded(edit["orientation"])
        moved = (
            _pose_differs(list(sensor.get("position", [])), position)
            or _pose_differs(list(sensor.get("orientation", [])), orientation)
        )
        sensor["position"] = position
        sensor["orientation"] = orientation
        if moved:
            changed.append(name)
    return changed


def save_cameras(document, out_path, source_path, scene_name, background=None):
    """Write a camera config, with a provenance header.

    PyYAML does not preserve the hand-written source's comments, so the header
    records where the file came from.

    Args:
        document (dict): Config to write.
        out_path (str or Path): Destination YAML.
        source_path (str or Path): Config this was derived from.
        scene_name (str): Scene the poses were authored against.
        background (str or None): Background id the placement belongs to,
            recorded in the header.

    Returns:
        Path: The file written.
    """
    out = Path(out_path)
    atomic_write_text(out, camera_config_text(
        document, source_path, scene_name, background=background, cfg_name=out.stem))
    return out


def camera_config_text(document, source_path, scene_name, *, background=None,
                       cfg_name=None):
    """The exact bytes :func:`save_cameras` would write, without writing them.

    Split out so an export can stage and digest every artifact before
    publishing any of them.

    Args:
        document (dict): Config to serialize.
        source_path (str or Path): Config this was derived from.
        scene_name (str): Scene the poses were authored against.
        background (str or None): Background id the placement belongs to.
        cfg_name (str or None): Stem the header tells the reader to use;
            None leaves the line off.

    Returns:
        str: Header plus body.
    """
    lines = [
        "# Generated by the SimFoundry light scene editor.",
        f"# Derived from: {Path(source_path).name}",
        f"# Authored against scene: {scene_name}",
    ]
    if background:
        lines.append(f"# Background: {background}")
    lines += [
        f"# Written: {datetime.now().isoformat(timespec='seconds')}",
        "#",
        "# Only camera position/orientation were edited; optics, modalities and",
        "# prim paths are unchanged from the source. Comments in the source are",
        "# not preserved by this round-trip.",
    ]
    if cfg_name:
        lines += ["#", f"# Use with:  s15_eval.external_sensors_cfg={cfg_name}"]
    lines += ["", ""]
    return "\n".join(lines) + yaml.safe_dump(
        document, sort_keys=False, default_flow_style=False)
