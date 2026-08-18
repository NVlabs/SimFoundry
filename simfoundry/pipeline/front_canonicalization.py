# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLM front-pick yaw canonicalization for generated object meshes.

Image-to-3D models snap each object's semantic front to an arbitrary horizontal
axis. The helpers here render yaw views of a mesh, ask a VLM which view shows
the front, and return the yaw rotation that puts the front on the target axis.
"""

from __future__ import annotations

import json
import logging
import os
import re

import numpy as np

logger = logging.getLogger(__name__)

# Azimuth (deg from +X toward +Z; mesh frame, +Y up) the front is rotated to.
# 90 = +Z: the glTF front, which reaches the articulation "frontview" camera.
FRONT_TARGET_AZIMUTH_DEG = 90
VIEW_AZIMUTHS_DEG = tuple(range(0, 360, 45))
_VIEW_LABELS = "ABCDEFGH"

# Second, fine-grained pass around the coarse pick: the 45-degree grid leaves up to
# +/-22.5 degrees of residual yaw, visible as a side face in the "front" view.
FRONT_REFINE_SPAN_DEG = 22.5
FRONT_REFINE_STEP_DEG = 7.5


def render_yaw_views(mesh, out_dir, azimuths_deg=VIEW_AZIMUTHS_DEG, elevation_deg=15.0):
    """Software-render labeled yaw views (no GL). Returns [(label, azimuth_deg, path)]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    try:
        colors = np.asarray(mesh.visual.to_color().vertex_colors)[:, :3] / 255.0
    except Exception:
        colors = np.full((len(mesh.vertices), 3), 0.7)
    v = np.asarray(mesh.vertices, dtype=float)
    f = np.asarray(mesh.faces)
    v = v - v.mean(axis=0)
    face_color = colors[f].mean(axis=1)
    fn = np.asarray(mesh.face_normals)

    os.makedirs(out_dir, exist_ok=True)
    el = np.deg2rad(elevation_deg)
    views = []
    for label, az_deg in zip(_VIEW_LABELS, azimuths_deg):
        az = np.deg2rad(az_deg)
        cam = np.array([np.cos(az) * np.cos(el), np.sin(el), np.sin(az) * np.cos(el)])
        right = np.cross([0.0, 1.0, 0.0], cam)
        right /= np.linalg.norm(right)
        up = np.cross(cam, right)
        sx, sy, depth = v @ right, v @ up, v @ cam
        order = np.argsort(depth[f].mean(axis=1))
        # Off-axis light and lifted albedo so dark objects still show geometry, not a silhouette.
        light = cam + 0.5 * up + 0.25 * right
        light /= np.linalg.norm(light)
        shade = np.clip(fn @ light, 0.0, 1.0)[:, None]
        albedo = 0.25 + 0.75 * face_color
        cols = np.clip(albedo * (0.3 + 0.7 * shade), 0, 1)
        polys = np.stack([sx[f], sy[f]], axis=-1)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=110)
        ax.add_collection(PolyCollection(polys[order], facecolors=cols[order], edgecolors="none"))
        lim = np.abs(np.concatenate([sx, sy])).max() * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"view {label}", fontsize=12)
        fpath = os.path.join(out_dir, f"view_{label}_az{az_deg}.png")
        fig.savefig(fpath, bbox_inches="tight")
        plt.close(fig)
        views.append((label, az_deg, fpath))
    return views


def front_pick_prompt(labels, category=None, has_photo=False):
    obj = f" of a {category}" if category else ""
    photo_note = " The final image is a reference photo of the real object." if has_photo else ""
    return (
        f"These images show the same generated 3D object{obj} from {len(labels)} camera "
        f"directions, 45 degrees apart, labeled {', '.join(labels)}.{photo_note} "
        "Which labeled view shows the object's semantic front facing the camera "
        "(labeled face, door or opening, screen, nozzle or business end)? "
        "If the object has no meaningful front (ball, plain cup, bowl, plate, fruit), "
        f"answer NONE. Answer with exactly one token: one of {', '.join(labels)} or NONE."
    )


def parse_front_choice(text, labels):
    """First label token (or NONE) in the response; None when ambiguous/absent."""
    tokens = re.findall(r"\b(" + "|".join(list(labels) + ["NONE"]) + r")\b", text.strip().upper())
    if not tokens or tokens[0] == "NONE":
        return None
    return tokens[0]


def refine_front_prompt(labels):
    return (
        f"These images show the same generated 3D object from {len(labels)} nearby camera "
        "directions, a few degrees of yaw apart, all roughly facing the object's front. "
        "Which view faces the front most directly — the front face as symmetric and "
        "perpendicular to the camera as possible, with the least of the side faces "
        f"visible? Answer with exactly one token: one of {', '.join(labels)}."
    )


def _refine_front_azimuth(mesh, render_dir, vlm, coarse_az_deg, span_deg, step_deg):
    """Fine-grained VLM pass around the coarse front pick. Never raises.

    Returns (refined_azimuth_deg, info): the azimuth of the fine view the VLM judged
    most directly front-on, or the coarse azimuth when the pass fails or is ambiguous.
    """
    refine_info = {"refine_status": "skipped", "refine_delta_deg": 0.0}
    try:
        deltas = np.arange(-span_deg, span_deg + 1e-9, step_deg)
        azimuths = [float(coarse_az_deg + d) for d in deltas]
        views = render_yaw_views(mesh, os.path.join(render_dir, "refine"), azimuths_deg=azimuths)
        labels = [v[0] for v in views]
        result = vlm(
            prompt=refine_front_prompt(labels),
            image_paths=[v[2] for v in views],
            temperature=0,
            top_p=0,
            seed=0,
        )
        choice = parse_front_choice(vlm.get_result_text(result), labels)
    except Exception as exc:
        logger.warning("Front refinement failed (%s); keeping the coarse pick.", exc)
        refine_info["refine_status"] = f"error: {exc}"
        return coarse_az_deg, refine_info
    if choice is None:
        refine_info["refine_status"] = "ambiguous"
        return coarse_az_deg, refine_info
    refined_az = dict((v[0], v[1]) for v in views)[choice]
    refine_info["refine_status"] = "refined"
    refine_info["refine_delta_deg"] = float(refined_az - coarse_az_deg)
    return refined_az, refine_info


def yaw_to_target_rotation(front_azimuth_deg, target_azimuth_deg=FRONT_TARGET_AZIMUTH_DEG):
    """3x3 rotation about +Y mapping the front azimuth onto the target azimuth."""
    alpha = np.deg2rad(front_azimuth_deg - target_azimuth_deg)
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def applied_yaw_deg(front_azimuth_deg, target_azimuth_deg=FRONT_TARGET_AZIMUTH_DEG):
    return float((front_azimuth_deg - target_azimuth_deg + 180) % 360 - 180)


def canonicalize_front(mesh, render_dir, *, vlm=None, photo_path=None, category=None,
                       gcloud_project=None, model="gemini-2.5-flash", refine=False,
                       refine_span_deg=FRONT_REFINE_SPAN_DEG,
                       refine_step_deg=FRONT_REFINE_STEP_DEG):
    """Decide the yaw rotation that puts `mesh`'s semantic front on the target axis.

    Two-stage: a coarse 45-degree pick of the semantic front, then (when `refine`) a
    fine pass over +/-refine_span_deg around it in refine_step_deg increments, asking
    which view is most directly front-on. Returns (rotation_3x3 or None, info dict).
    None = leave the mesh as-is (no semantic front, TEST_MODE, or VLM failure).
    Never raises.
    """
    info = {
        "target_azimuth_deg": FRONT_TARGET_AZIMUTH_DEG,
        "front_azimuth_deg": None,
        "front_azimuth_coarse_deg": None,
        "applied_yaw_deg": 0.0,
        "status": "unknown",
    }
    try:
        from simfoundry.models.remote_cache import RemoteModelCache
        if RemoteModelCache.from_env().test_enabled:
            info["status"] = "skipped_test_mode"
            return None, info

        views = render_yaw_views(mesh, render_dir)
        labels = [v[0] for v in views]
        image_paths = [v[2] for v in views]
        has_photo = bool(photo_path) and os.path.isfile(str(photo_path))
        if has_photo:
            image_paths.append(str(photo_path))

        if vlm is None:
            from simfoundry.models.vlm import Gemini
            vlm = Gemini(project=gcloud_project, location="global", model=model)
        result = vlm(
            prompt=front_pick_prompt(labels, category=category, has_photo=has_photo),
            image_paths=image_paths,
            temperature=0,
            top_p=0,
            seed=0,
        )
        choice = parse_front_choice(vlm.get_result_text(result), labels)
    except Exception as exc:
        logger.warning("Front canonicalization failed (%s); leaving mesh orientation as-is.", exc)
        info["status"] = f"error: {exc}"
        return None, info

    if choice is None:
        info["status"] = "ambiguous"
        return None, info

    front_az = dict((v[0], v[1]) for v in views)[choice]
    info["front_azimuth_coarse_deg"] = front_az
    if refine:
        front_az, refine_info = _refine_front_azimuth(
            mesh, render_dir, vlm, front_az,
            span_deg=refine_span_deg, step_deg=refine_step_deg,
        )
        info.update(refine_info)
    info["front_azimuth_deg"] = front_az
    info["applied_yaw_deg"] = applied_yaw_deg(front_az)
    info["status"] = "rotated"
    return yaw_to_target_rotation(front_az), info


def yaw_rotation(yaw_deg):
    """3x3 rotation about +Y by `yaw_deg` (same convention as yaw_to_target_rotation)."""
    a = np.deg2rad(yaw_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def yaw_snap_to_axes(vertices, min_rectangularity=0.80):
    """Snap residual yaw so the mesh footprint's min-area rectangle aligns with x/z.

    The VLM passes decide WHICH face is the front (a semantic question); the precise
    angle is geometry's job — VLM picks plateau at their render granularity, leaving
    10-20 degrees of visible yaw on pixel-aligned meshes. The correction is bounded to
    (-45, 45] degrees so the semantic front choice is preserved, and it is skipped for
    footprints that are not box-like (rectangularity = hull area / min-rect area; a
    circle scores pi/4 ~ 0.785, where the rectangle's angle is meaningless).

    Expects an already-upright mesh (+Y up). Returns (rotation_3x3 or None, info).
    Never raises.
    """
    info = {"snap_yaw_deg": 0.0, "rectangularity": None, "status": "unknown"}
    try:
        from scipy.spatial import ConvexHull

        pts = np.asarray(vertices, dtype=float)[:, [0, 2]]
        hull = ConvexHull(pts)
        hp = pts[hull.vertices]
        edges = np.diff(np.vstack([hp, hp[:1]]), axis=0)
        # The min-area rectangle has a side collinear with a hull edge (rotating calipers).
        angles = np.unique(np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2))
        best_area, best_angle = np.inf, 0.0
        for ang in angles:
            c, s = np.cos(ang), np.sin(ang)
            q = hp @ np.array([[c, -s], [s, c]])
            area = float(np.ptp(q[:, 0]) * np.ptp(q[:, 1]))
            if area < best_area:
                best_area, best_angle = area, float(ang)
        info["rectangularity"] = float(hull.volume / best_area)  # 2D hull.volume == area
    except Exception as exc:
        logger.warning("Yaw snap failed (%s); leaving yaw as-is.", exc)
        info["status"] = f"error: {exc}"
        return None, info

    if info["rectangularity"] < min_rectangularity:
        info["status"] = "not_boxy"
        return None, info

    # Smallest correction within the 90-degree family of the rectangle's orientation.
    theta = (best_angle + np.pi / 4) % (np.pi / 2) - np.pi / 4
    snap_deg = float(np.degrees(theta))
    info["snap_yaw_deg"] = snap_deg
    info["status"] = "snapped"
    return yaw_rotation(snap_deg), info


def read_orientation_yaw(fpath):
    """applied_yaw_deg recorded at `fpath`; 0.0 when absent/unreadable (legacy data)."""
    return read_orientation_stamp(fpath)[0]


def read_orientation_stamp(fpath):
    """(applied_yaw_deg, applied_tilt_deg) recorded at `fpath`; zeros when absent/unreadable.

    The tilt component exists only for meshes whose pitch/roll was baked from the
    stage-8 fit (pixel-aligned backends); legacy stamps read as tilt 0.0, so
    already-articulated objects are not invalidated by the stamp format change alone.
    """
    if not os.path.isfile(fpath):
        return 0.0, 0.0
    try:
        with open(fpath) as f:
            data = json.load(f)
        return (
            float(data.get("applied_yaw_deg", 0.0)),
            float(data.get("applied_tilt_deg", 0.0)),
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return 0.0, 0.0


def orientation_stamp_changed(recorded, current, yaw_tol_deg=10.0, tilt_tol_deg=2.0):
    """Whether a mesh's orientation stamp differs enough to invalidate articulation.

    Both components get a tolerance because both derive from the stochastic fit:
    re-running stage 8 jitters the baked tilt by fractions of a degree, and the fine
    yaw-refinement pass picks among 7.5-degree bins from renders of that baked mesh,
    so a rerun can flip one bin without the orientation meaningfully changing. A
    coarse front change (the 45-degree grid) still invalidates. Yaw compares on the
    circle so -180 and +180 are the same angle.
    """
    yaw_diff = abs((recorded[0] - current[0] + 180.0) % 360.0 - 180.0)
    return yaw_diff > yaw_tol_deg or abs(recorded[1] - current[1]) > tilt_tol_deg
