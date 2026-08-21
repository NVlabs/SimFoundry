# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover a mesh's upright orientation from stage 8's fitted pose.

Pixel-aligned mesh backends (pixal3d) generate in the conditioning image's camera
frame, so the conditioning viewpoint's elevation and roll are baked into the exported
GLB — the mesh is not upright in its own frame (hunyuan/trellis emit a learned
canonical frame instead). Stage 8's full-SO(3) fit against the scene's partial point
cloud measures exactly that error: the direction in mesh coordinates that the fit
sends onto the scene's gravity axis is the object's true up. The helpers here compute
the minimal rotation that moves that direction onto the mesh convention's +Y so the
tilt can be baked into ``canonical_mesh`` (which stages 9/10/11 and cousin generation
consume) while the stored pose keeps only gravity yaw + translation + scale.
"""

from __future__ import annotations

import numpy as np

# glTF convention, and what front_canonicalization's yaw renders assume.
MESH_UP = np.array([0.0, 1.0, 0.0])

# Backends that generate in the conditioning camera's frame; only their meshes carry a
# generation tilt worth baking. Canonical-frame backends (hunyuan, trellis2, direct3d)
# are upright by construction, so a fitted tilt there is a fit error or the object's
# physical placement — baking either corrupts the mesh.
PIXEL_ALIGNED_SHAPE_MODELS = ("pixal3d",)


def resolve_bake_fitted_tilt(mode, shape_model):
    """Bool passthrough; "auto" bakes only for pixel-aligned shape backends."""
    if isinstance(mode, bool):
        return mode
    return str(shape_model) in PIXEL_ALIGNED_SHAPE_MODELS


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-length direction vector")
    return v / n


def rotation_between(v_from, v_to):
    """Minimal 3x3 rotation taking direction `v_from` onto direction `v_to`."""
    a, b = _unit(v_from), _unit(v_to)
    c = float(np.clip(a @ b, -1.0, 1.0))
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3)
        # Antiparallel: rotate 180 degrees about any axis perpendicular to `a`.
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(a, helper))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = axis / s
    angle = np.arctan2(s, c)
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def up_in_mesh_frame(rot_fit, gravity_up):
    """Direction in mesh coordinates that `rot_fit` sends onto the scene's gravity axis.

    `rot_fit` maps mesh points into the frame of the fitted point cloud;
    `gravity_up` is the scene's up axis expressed in that same frame.
    """
    return _unit(np.asarray(rot_fit, dtype=float).T @ _unit(gravity_up))


def tilt_from_fit(rot_fit, gravity_up, mesh_up=MESH_UP):
    """(tilt_deg, rot_tilt) implied by a fitted pose.

    `rot_tilt @ mesh` is upright (its `mesh_up` is the object's true up), and the
    adjusted pose `rot_fit @ rot_tilt.T` sends `mesh_up` exactly onto `gravity_up`,
    so the residual rotation in the stored pose is pure gravity yaw.
    """
    v = up_in_mesh_frame(rot_fit, gravity_up)
    mesh_up = _unit(mesh_up)
    tilt_deg = float(np.degrees(np.arccos(np.clip(v @ mesh_up, -1.0, 1.0))))
    return tilt_deg, rotation_between(v, mesh_up)


def up_axis_spread_deg(rots, gravity_up):
    """Largest pairwise angle (deg) between the up axes implied by candidate fits.

    A consensus gate: independent CPD restarts that agree on where the object's up
    lies can be trusted; wildly disagreeing fits mean the registration is unreliable
    and no tilt should be baked from it.
    """
    ups = [up_in_mesh_frame(r, gravity_up) for r in rots]
    worst = 0.0
    for i in range(len(ups)):
        for j in range(i + 1, len(ups)):
            ang = float(np.degrees(np.arccos(np.clip(ups[i] @ ups[j], -1.0, 1.0))))
            worst = max(worst, ang)
    return worst


def decide_tilt(info_sorted, gravity_up, *, min_tilt_deg, max_tilt_deg, consensus_deg,
                consensus_top_k=3, mesh_up=MESH_UP):
    """Gatekeeping for tilt baking. Returns (rot_tilt or None, info dict).

    `info_sorted` is stage 8's fit list, best first; each entry's ``tf_z_up.rot`` maps
    mesh points into the target-cloud frame. None = leave the mesh as-is.
    """
    best_rot = info_sorted[0]["tf_z_up"].rot
    tilt_deg, rot_tilt = tilt_from_fit(best_rot, gravity_up, mesh_up=mesh_up)
    spread = up_axis_spread_deg(
        [e["tf_z_up"].rot for e in info_sorted[:consensus_top_k]], gravity_up,
    ) if len(info_sorted) > 1 else 0.0
    info = {
        "tilt_deg": tilt_deg,
        "consensus_spread_deg": spread,
        "applied_tilt_deg": 0.0,
        "status": "unknown",
    }
    if tilt_deg < min_tilt_deg:
        info["status"] = "below_threshold"
        return None, info
    if tilt_deg > max_tilt_deg:
        info["status"] = "implausible_tilt"
        return None, info
    if spread > consensus_deg:
        info["status"] = "inconsistent_fits"
        return None, info
    info["applied_tilt_deg"] = tilt_deg
    info["status"] = "baked"
    return rot_tilt, info
