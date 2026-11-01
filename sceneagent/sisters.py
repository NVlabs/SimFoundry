"""Model-free digital sisters (SceneAgent section 3.8, first method).

The paper's first sister generator does not call a model. Bounded deformations
change the object's own mesh. Each result is checked for extent, volume, and
whether the object can still be grasped. A procedural texture of noise, hue
rotation, mottling, and speckle is baked into vertex colors. Phase 1 records
that texture as parameters. It does not rasterize an image.

The paper does not publish the deformation bounds or the numeric grasp test.
The default length change of ±5% is the geometry axis on the project website's
place-in-bowl grid, not a bound stated in the paper. The grasp and volume-ratio
gates below are this implementation's checks and are labeled that way.

The second sister method (Gemini image edits lifted by Meshy, or Meshy-6
retexture) is a later phase. It is not called here.
"""

from __future__ import annotations

import numpy as np

# Implementation gates. Not published in the paper.
GRASP_MIN_EXTENT_M = 0.005
VOLUME_RATIO_BAND = (0.5, 1.5)
WEBSITE_LENGTH_SCALES = (0.95, 1.05)


def box_mesh(size):
    """Axis-aligned box centered at the origin. ``size`` is full edge lengths."""
    hx, hy, hz = (float(v) / 2.0 for v in size)
    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=int,
    )
    return vertices, faces


def mesh_volume(vertices, faces):
    verts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)
    total = 0.0
    for a, b, c in tris:
        total += float(np.dot(verts[a], np.cross(verts[b], verts[c]))) / 6.0
    return abs(total)


def mesh_extent(vertices):
    verts = np.asarray(vertices, dtype=float)
    return (verts.max(axis=0) - verts.min(axis=0)).astype(float)


def procedural_texture(hue_degrees, mottle, speckle, noise):
    return {
        "kind": "procedural_vertex_color",
        "hue_degrees": float(hue_degrees),
        "mottle": float(mottle),
        "speckle": float(speckle),
        "noise": float(noise),
        "geometry_unchanged": True,
        "source": "section_3.8_model_free_texture",
    }


def check_sister(extent, volume, original_volume, target_inner=None):
    """Extent, volume-ratio, grasp, and optional fit against a target box."""
    extent = np.asarray(extent, dtype=float)
    reasons = []
    if float(extent.min()) < GRASP_MIN_EXTENT_M:
        reasons.append("extent_below_grasp_minimum")
    if original_volume <= 0:
        reasons.append("original_volume_not_positive")
        ratio = None
    else:
        ratio = float(volume) / float(original_volume)
        low, high = VOLUME_RATIO_BAND
        if ratio < low or ratio > high:
            reasons.append("volume_ratio_outside_band")
    fits = None
    if target_inner is not None:
        inner = np.asarray(target_inner, dtype=float)
        fits = bool(np.all(extent <= inner + 1e-9))
        if not fits:
            reasons.append("does_not_fit_target")
    return {
        "extent_m": extent.tolist(),
        "volume_m3": float(volume),
        "volume_ratio": ratio,
        "fits_target": fits,
        "graspable": "extent_below_grasp_minimum" not in reasons and "volume_ratio_outside_band" not in reasons,
        "accepted": len(reasons) == 0,
        "reasons": reasons,
        "gates": {
            "grasp_min_extent_m": GRASP_MIN_EXTENT_M,
            "volume_ratio_band": list(VOLUME_RATIO_BAND),
            "published_in_paper": False,
        },
    }


def geometry_sisters(size, target_inner=None, length_scales=WEBSITE_LENGTH_SCALES):
    """Scale the box's first axis and score each result.

    The original mesh is included so the grid has the twin plus sisters.
    """
    base_verts, faces = box_mesh(size)
    base_volume = mesh_volume(base_verts, faces)
    specs = [("original", (1.0, 1.0, 1.0))]
    for scale in length_scales:
        label = f"length {scale - 1.0:+.0%}"
        specs.append((label, (float(scale), 1.0, 1.0)))
    sisters = []
    for label, factors in specs:
        scaled = base_verts * np.asarray(factors, dtype=float)
        extent = mesh_extent(scaled)
        volume = mesh_volume(scaled, faces)
        report = check_sister(extent, volume, base_volume, target_inner=target_inner)
        report.update({
            "name": label,
            "kind": "bounded_axis_scale",
            "scale": list(factors),
            "texture": procedural_texture(0.0, 0.0, 0.0, 0.0),
        })
        sisters.append(report)
    return sisters


def appearance_sister(size, hue_degrees, mottle, speckle, noise, name):
    """Retexture parameters on the unchanged mesh, as in the model-free path."""
    verts, faces = box_mesh(size)
    volume = mesh_volume(verts, faces)
    extent = mesh_extent(verts)
    report = check_sister(extent, volume, volume, target_inner=None)
    report.update({
        "name": name,
        "kind": "procedural_texture",
        "scale": [1.0, 1.0, 1.0],
        "texture": procedural_texture(hue_degrees, mottle, speckle, noise),
    })
    return report


def scene_sisters(anchor_xy, gap_m, relations=None):
    """Layouts around a task anchor.

    Section 3.8 and skill SCENEAGENT-SCENE-SISTERS sample layouts from spatial
    relations. The relation names used here are a phase-1 set. Offsets are in
    meters on the horizontal plane.
    """
    relations = relations or ("left_of", "right_of", "behind", "in_front_of")
    offsets = {
        "left_of": (-gap_m, 0.0),
        "right_of": (gap_m, 0.0),
        "behind": (0.0, gap_m),
        "in_front_of": (0.0, -gap_m),
    }
    ax, ay = float(anchor_xy[0]), float(anchor_xy[1])
    layouts = []
    for name in relations:
        if name not in offsets:
            raise ValueError(f"unsupported relation {name!r}")
        dx, dy = offsets[name]
        layouts.append({
            "relation": name,
            "anchor_xy_m": [ax, ay],
            "object_xy_m": [ax + dx, ay + dy],
            "gap_m": float(gap_m),
        })
    return layouts


def task_sisters(instruction, manipulated, target):
    """Instruction variants that keep the inside-container goal.

    Section 3.10 samples a natural-language instruction from alternative task
    descriptions. These three strings are phase-1 templates, not a VLM sample.
    """
    return [
        {
            "instruction": instruction,
            "goal": "inside",
            "manipulated": manipulated,
            "target": target,
            "source": "capture",
        },
        {
            "instruction": f"put the {manipulated} into the {target}",
            "goal": "inside",
            "manipulated": manipulated,
            "target": target,
            "source": "phase1_template",
        },
        {
            "instruction": f"move the {manipulated} so it ends inside the {target}",
            "goal": "inside",
            "manipulated": manipulated,
            "target": target,
            "source": "phase1_template",
        },
    ]
