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


def yaw_to_target_rotation(front_azimuth_deg, target_azimuth_deg=FRONT_TARGET_AZIMUTH_DEG):
    """3x3 rotation about +Y mapping the front azimuth onto the target azimuth."""
    alpha = np.deg2rad(front_azimuth_deg - target_azimuth_deg)
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def applied_yaw_deg(front_azimuth_deg, target_azimuth_deg=FRONT_TARGET_AZIMUTH_DEG):
    return float((front_azimuth_deg - target_azimuth_deg + 180) % 360 - 180)


def canonicalize_front(mesh, render_dir, *, vlm=None, photo_path=None, category=None,
                       gcloud_project=None, model="gemini-2.5-flash"):
    """Decide the yaw rotation that puts `mesh`'s semantic front on the target axis.

    Returns (rotation_3x3 or None, info dict). None = leave the mesh as-is
    (no semantic front, TEST_MODE, or VLM failure). Never raises.
    """
    info = {
        "target_azimuth_deg": FRONT_TARGET_AZIMUTH_DEG,
        "front_azimuth_deg": None,
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
    info["front_azimuth_deg"] = front_az
    info["applied_yaw_deg"] = applied_yaw_deg(front_az)
    info["status"] = "rotated"
    return yaw_to_target_rotation(front_az), info


def read_orientation_yaw(fpath):
    """applied_yaw_deg recorded at `fpath`; 0.0 when absent/unreadable (legacy data)."""
    if not os.path.isfile(fpath):
        return 0.0
    try:
        with open(fpath) as f:
            return float(json.load(f).get("applied_yaw_deg", 0.0))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return 0.0
