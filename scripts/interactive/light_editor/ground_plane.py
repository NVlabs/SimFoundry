# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Read and write a scene's ``ground_plane_info``.

A Gaussian-splat background is visual-only, with no collision geometry, so
props fall through it when physics starts; OmniGibson's floor plane is what
they rest on, and this block records where to put it::

    "ground_plane_info": {
        "position":    [0.0, 0.0, 0.03],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "visible":     false
    }

``PickPlaceTask`` applies it on every task load. The plane is infinite, so only
``position[2]`` -- its height -- matters unless it is tilted.
``og.sim.restore()`` does not read it: anything restoring a scene must apply it
itself, via ``simfoundry.utils.ground_plane_utils.apply_ground_plane_info``.
``visible`` is optional; when absent, visibility comes from the run config's
``floor_plane_visible``. Whether a plane exists at all is decided by
``use_floor_plane`` in the task YAML -- this block positions a plane that is
already there and cannot create one.
"""

import math
from numbers import Real

from scene_io import SceneEditError

#: Top-level key in the scene document.
GROUND_PLANE_KEY = "ground_plane_info"

#: Identity quaternion (XYZW): a level plane.
LEVEL = [0.0, 0.0, 0.0, 1.0]

#: Default plane height. z=0 is where a scanned room's support surface is
#: registered and where a room-less scene's props are laid out.
DEFAULT_HEIGHT = 0.0


def read_ground_plane(scene):
    """The ground plane a scene carries, or None.

    Args:
        scene (dict): Parsed scene document.

    Returns:
        dict or None: ``position`` (3 floats), ``orientation`` (4, XYZW) and
        ``visible`` (bool, or None meaning the document says nothing and the
        run config decides).
    """
    block = scene.get(GROUND_PLANE_KEY)
    if not isinstance(block, dict):
        return None

    position = block.get("position")
    orientation = block.get("orientation")
    if not _is_vector(position, 3):
        return None
    if not _is_vector(orientation, 4):
        orientation = list(LEVEL)

    visible = block.get("visible")
    return {
        "position": [float(v) for v in position],
        "orientation": [float(v) for v in orientation],
        "visible": None if not isinstance(visible, bool) else visible,
    }


def _is_vector(value, length):
    return (
        isinstance(value, (list, tuple))
        and len(value) == length
        and all(not isinstance(v, bool) and isinstance(v, Real) and math.isfinite(v)
                for v in value)
    )


def validate_ground_plane(spec):
    """Check a browser's ground-plane request and return a normalized copy.

    Args:
        spec (dict or None): ``{"position": [3], "orientation": [4], "visible":
            bool}``, or None meaning "this scene has no ground plane". Any of
            the three may be omitted; the defaults are level, at
            :data:`DEFAULT_HEIGHT`, and visibility left to the run config.

    Returns:
        dict or None: Normalized, or None for "no ground plane".

    Raises:
        SceneEditError: On anything that would write a plane nobody can stand
            on -- a non-finite height, a zero quaternion, the wrong arity.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise SceneEditError("ground_plane must be an object or null")

    unknown = set(spec) - {"position", "orientation", "visible"}
    if unknown:
        raise SceneEditError(
            f"unsupported ground_plane field(s): {', '.join(sorted(unknown))}")

    position = spec.get("position", [0.0, 0.0, DEFAULT_HEIGHT])
    if not _is_vector(position, 3):
        raise SceneEditError("ground_plane.position must be three finite numbers")

    orientation = spec.get("orientation", list(LEVEL))
    if not _is_vector(orientation, 4):
        raise SceneEditError("ground_plane.orientation must be four finite numbers")
    norm = math.sqrt(sum(float(v) * float(v) for v in orientation))
    if norm <= 1e-12:
        raise SceneEditError("ground_plane.orientation must be a non-zero quaternion")
    # An already-unit quaternion passes through byte-for-byte, so
    # re-normalising does not show up as an edit nobody made.
    if abs(norm - 1.0) > 1e-6:
        orientation = [float(v) / norm for v in orientation]

    visible = spec.get("visible")
    if visible is not None and not isinstance(visible, bool):
        raise SceneEditError("ground_plane.visible must be true, false or null")

    return {
        "position": [float(v) for v in position],
        "orientation": [float(v) for v in orientation],
        "visible": visible,
    }


def apply_ground_plane(scene, spec):
    """Write a validated ground plane into a scene document, in place.

    Args:
        scene (dict): Parsed scene document. Mutated.
        spec (dict or None): From :func:`validate_ground_plane`. None removes
            the block, which returns the scene to "the run config decides".

    Returns:
        str: What happened -- ``added``, ``changed``, ``removed`` or
        ``unchanged``.
    """
    before = read_ground_plane(scene)

    if spec is None:
        if GROUND_PLANE_KEY in scene:
            del scene[GROUND_PLANE_KEY]
            return "removed" if before else "unchanged"
        return "unchanged"

    block = {
        # Rounded to six decimals (a micron), as camera poses are.
        "position": [round(v, 6) for v in spec["position"]],
        "orientation": [round(v, 6) for v in spec["orientation"]],
    }
    # Omitted when unset, so the scene does not gain an opinion it never had.
    if spec["visible"] is not None:
        block["visible"] = spec["visible"]

    scene[GROUND_PLANE_KEY] = block
    if before is None:
        return "added"
    same = (
        before["position"] == block["position"]
        and before["orientation"] == block["orientation"]
        and before["visible"] == block.get("visible")
    )
    return "unchanged" if same else "changed"


def describe(spec):
    """One line naming a ground plane, for a status message or a log."""
    if spec is None:
        return "no ground plane"
    height = spec["position"][2]
    if spec["visible"] is None:
        shown = "visibility left to the run config"
    else:
        shown = "visible" if spec["visible"] else "hidden"
    return f"ground plane at z={height:+.3f} m, {shown}"
