# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Put the simulator's floor plane where a saved scene says it is.

``ground_plane_info`` is the one part of the saved-scene contract that decides
where a prop comes to rest. Under a Gaussian-splat room it is the *only* thing
anything can rest on: the splat is loaded ``visual_only=True`` and has no
colliders, so the desk in the picture is a picture of a desk and every prop
placed on it falls through.

The block is read on the evaluation path by
:meth:`simfoundry.tasks.pick_place_task.PickPlaceTask._apply_ground_plane_from_scene`,
which is what makes an exported layout hold together in a run. The two physics
gates in the light editor -- ``settle.py`` and ``parity_check.py`` -- restore a
scene through ``og.sim.restore()``, which does **not** read it, and so used to
simulate against a floor at z=0 whatever the scene said. Six props fell 30-46 mm
and both gates failed; the failure was real but it was the gate's, not the
scene's, and a gate that does not model the contract cannot prove anything about
a run that does.

So the application lives here, once, and all three call it. A gate that models
the scene differently from the consumer is worse than no gate: it fails runs
that would have worked and, on the day the offset happens to cancel out, passes
ones that would not.
"""

import torch as th

#: Where the plane sits when a scene says nothing. OmniGibson's own default, and
#: what every scene written before ``ground_plane_info`` existed means.
LEVEL_POSITION = (0.0, 0.0, 0.0)
LEVEL_ORIENTATION = (0.0, 0.0, 0.0, 1.0)


def read_ground_plane_info(scene_info):
    """The ground plane a scene document carries, normalized.

    Args:
        scene_info (dict or None): A parsed scene-state document.

    Returns:
        dict: ``position`` (3 floats), ``orientation`` (4, XYZW), ``visible``
        (bool or None -- None meaning the document states no opinion and the run
        config decides) and ``authored`` (whether the block was there at all).
    """
    block = (scene_info or {}).get("ground_plane_info") if isinstance(scene_info, dict) else None
    if not isinstance(block, dict):
        return {"position": list(LEVEL_POSITION), "orientation": list(LEVEL_ORIENTATION),
                "visible": None, "authored": False}

    position = block.get("position")
    orientation = block.get("orientation")
    visible = block.get("visible")
    return {
        "position": ([float(v) for v in position] if _is_vector(position, 3)
                     else list(LEVEL_POSITION)),
        "orientation": ([float(v) for v in orientation] if _is_vector(orientation, 4)
                        else list(LEVEL_ORIENTATION)),
        "visible": visible if isinstance(visible, bool) else None,
        "authored": True,
    }


def _is_vector(value, length):
    return (isinstance(value, (list, tuple)) and len(value) == length
            and all(not isinstance(v, bool) and isinstance(v, (int, float)) for v in value))


def apply_ground_plane_info(scene_info, floor_plane, *, z_offset=None):
    """Move *floor_plane* to where *scene_info* puts it.

    The plane is infinite, so only ``position[2]`` has any effect unless the
    scene tilts it -- but the whole vector is applied rather than just the
    height, because a scene that did tilt it meant to.

    Args:
        scene_info (dict or None): A parsed scene-state document.
        floor_plane: ``og.sim.floor_plane``, or None when the run was configured
            without one. None is not an error here: whether a floor plane exists
            at all is decided by ``use_floor_plane`` in the scene config, not by
            the scene document, and this cannot conjure one.
        z_offset (float or None): A run-config override of the height, applied
            after the scene's own value. This is ``PickPlaceTask``'s
            ``ground_plane_z_offset``.

    Returns:
        dict or None: What was applied -- ``position``, ``orientation``,
        ``visible`` and ``authored`` -- or None when there was no plane to move.
    """
    if floor_plane is None:
        return None

    plane = read_ground_plane_info(scene_info)
    position = list(plane["position"])
    if z_offset is not None:
        position[2] = float(z_offset)

    floor_plane.set_position_orientation(
        position=th.tensor(position, dtype=th.float32),
        orientation=th.tensor(plane["orientation"], dtype=th.float32),
    )
    if plane["visible"] is not None:
        floor_plane.visible = plane["visible"]
    return {**plane, "position": position}


def describe(applied):
    """One line naming what was applied, for a gate's report or a log."""
    if applied is None:
        return ("no floor plane in this run (use_floor_plane is off), so nothing "
                "is holding the props up")
    source = "from the scene" if applied["authored"] else "the default (the scene states none)"
    visible = {True: "visible", False: "hidden"}.get(
        applied["visible"], "visibility from the run config")
    return f"floor plane at z={applied['position'][2]:+.4f} m, {source}, {visible}"
