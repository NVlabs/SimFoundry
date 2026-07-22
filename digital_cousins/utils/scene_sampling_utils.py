# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for validated scene pose resampling.

These utilities are used by:
  - scripts/pipeline/B_augmentation/stages/6b_sample_asset_scene.py — offline sampling of scene variants
  - scripts/pipeline/C_application/stages/5_generate_demos.py — online resampling between demo batches

The sampler groups task objects by their OnTop relationships (union-find), applies
a per-group XY translation + Z rotation around the group centroid, reloads object
state from the scene JSON, settles physics, and re-extracts relationships to validate
the result.
"""

import numpy as np
import torch as th

import omnigibson as og
from omnigibson.object_states import OnTop
import omnigibson.utils.transform_utils as T


# ---------------------------------------------------------------------------
# Scene JSON object classification
# ---------------------------------------------------------------------------

def _is_task_object(obj_name, obj_info):
    """Return True if this init_info entry is a task USDObject (not robot/background).

    Includes fixed_base objects — they are handled separately during randomization.
    """
    class_module = obj_info.get("class_module", "")
    if "usd_object" not in class_module:
        return False
    if "background" in obj_name.lower():
        return False
    return True


def get_task_object_names(scene_json):
    """Return ordered list of task object names from the scene JSON's init_info."""
    init_info = scene_json.get("objects_info", {}).get("init_info", {})
    return [name for name, info in init_info.items() if _is_task_object(name, info)]


def get_fixed_base_names(scene_json):
    """Return set of object names that have fixed_base=True in the scene JSON."""
    init_info = scene_json.get("objects_info", {}).get("init_info", {})
    names = set()
    for name, info in init_info.items():
        if info.get("args", {}).get("fixed_base", False):
            names.add(name)
    return names


# ---------------------------------------------------------------------------
# Pose capture + randomized pose computation
# ---------------------------------------------------------------------------

def capture_settled_poses(task_objects):
    """Return {obj_name: [pos_list, ori_list]} for all task objects."""
    poses = {}
    for name, obj in task_objects.items():
        pos, ori = obj.get_position_orientation()
        poses[name] = [
            pos.tolist() if isinstance(pos, th.Tensor) else list(pos),
            ori.tolist() if isinstance(ori, th.Tensor) else list(ori),
        ]
    return poses


def _sample_z_rotation_rad(z_rotation_deg):
    if z_rotation_deg is None or z_rotation_deg >= 360:
        return np.random.uniform(0, 2 * np.pi)
    half_rad = np.deg2rad(float(z_rotation_deg))
    return np.random.uniform(-half_rad, half_rad)


def compute_group_aabb(poses, group_names, margin=0.0):
    """Compute 2D (XY) axis-aligned bounding box for a group of objects.

    Args:
        poses: {name: [pos_list, ori_list]} poses dict.
        group_names: Iterable of object names in this group.
        margin: Extra padding in meters on each side. Default is 0 — the AABB
            covers only the object origins. Keep this small relative to
            ``xy_range`` to avoid starving later groups of valid placements.

    Returns:
        (min_xy, max_xy) as numpy arrays of shape (2,), or None if no valid objects.
    """
    pts = [np.array(poses[n][0][:2]) for n in group_names if n in poses]
    if not pts:
        return None
    pts = np.stack(pts)
    return pts.min(axis=0) - margin, pts.max(axis=0) + margin


def aabbs_overlap(aabb_a, aabb_b):
    """Return True if two 2D AABBs overlap (each is (min_xy, max_xy))."""
    if aabb_a is None or aabb_b is None:
        return False
    return (aabb_a[0][0] <= aabb_b[1][0] and aabb_a[1][0] >= aabb_b[0][0] and
            aabb_a[0][1] <= aabb_b[1][1] and aabb_a[1][1] >= aabb_b[0][1])


def compute_single_group_pose(settled_poses, group, xy_range, z_rotation_deg,
                               fixed_base_names=None, include_fixed_base_groups=False,
                               placed_aabbs=None, min_group_distance=0.0,
                               max_rejection_samples=50):
    """Sample a collision-free pose for a single group via rejection sampling.

    Applies a random XY translation + Z rotation (same math as
    ``compute_randomized_poses``) and checks that the resulting AABB does not
    overlap any already-placed group AABBs.

    The AABB check is a lightweight pre-filter to avoid gross collisions.
    Physics settling + relationship validation provide the real safety net.
    Keep ``min_group_distance`` small (or 0) relative to ``xy_range`` to avoid
    starving later groups of valid placements in dense scenes.

    Args:
        settled_poses: Baseline poses for all task objects.
        group: Set of object names in this group.
        xy_range: XY translation range in meters.
        z_rotation_deg: Z rotation range in +/- degrees.
        fixed_base_names: Set of fixed-base object names.
        include_fixed_base_groups: Whether to perturb fixed-base groups.
        placed_aabbs: List of (min_xy, max_xy) AABBs for already-placed groups.
        min_group_distance: Minimum clearance between group AABBs in meters.
            Default 0 — relies on physics validation instead.
        max_rejection_samples: Max rejection samples before giving up.

    Returns:
        (poses_dict, group_aabb) on success, or (None, None) if all samples
        were rejected.
    """
    fixed_base_names = set(fixed_base_names or [])
    placed_aabbs = placed_aabbs or []

    # Fixed-base groups that should not be perturbed — return settled poses directly
    group_has_fixed = any(n in fixed_base_names for n in group)
    if group_has_fixed and not include_fixed_base_groups:
        poses = {}
        for name in group:
            if name in settled_poses:
                poses[name] = settled_poses[name]
        aabb = compute_group_aabb(poses, list(group))
        return poses, aabb

    group_names = [n for n in group if n in settled_poses]
    if not group_names:
        return None, None

    positions = {n: th.tensor(settled_poses[n][0]) for n in group_names}
    centroid_xy = th.stack(list(positions.values()))[:, :2].mean(dim=0)

    for _ in range(max_rejection_samples):
        dx = np.random.uniform(-xy_range, xy_range)
        dy = np.random.uniform(-xy_range, xy_range)
        angle = _sample_z_rotation_rad(z_rotation_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_quat = th.tensor([0.0, 0.0, float(np.sin(angle / 2)), float(np.cos(angle / 2))])

        candidate = {}
        for name in group_names:
            pos = positions[name]
            ori = th.tensor(settled_poses[name][1])
            rel_x = float(pos[0]) - float(centroid_xy[0])
            rel_y = float(pos[1]) - float(centroid_xy[1])
            new_pos = pos.clone()
            new_pos[0] = float(centroid_xy[0]) + cos_a * rel_x - sin_a * rel_y + dx
            new_pos[1] = float(centroid_xy[1]) + sin_a * rel_x + cos_a * rel_y + dy
            new_ori = T.quat_multiply(rot_quat, ori)
            candidate[name] = [new_pos.tolist(), new_ori.tolist()]

        cand_aabb = compute_group_aabb(candidate, group_names)
        # Check overlap with all already-placed AABBs (inflated by min_group_distance)
        collision = False
        if cand_aabb is not None:
            inflated = (cand_aabb[0] - min_group_distance, cand_aabb[1] + min_group_distance)
            for placed in placed_aabbs:
                if aabbs_overlap(inflated, placed):
                    collision = True
                    break
        if not collision:
            return candidate, cand_aabb

    return None, None


def filter_relationships_for_group(rel_set, group_names):
    """Return subset of a relationship frozenset involving only the given group names.

    Since groups are built via union-find over OnTop relationships, all
    relationships are intra-group by construction.
    """
    group_set = set(group_names)
    return frozenset(r for r in rel_set if r[0] in group_set and r[1] in group_set)


def compute_randomized_poses(settled_poses, groups, xy_range, z_rotation_deg,
                              fixed_base_names=None, include_fixed_base_groups=False):
    """
    Compute new poses by applying a random XY translation + Z rotation per group.

    Args:
        settled_poses: {name: [pos_list, ori_list]} baseline poses for every task object.
        groups: List of sets — each set is a union-find group of object names.
        xy_range: Per-group uniform XY translation range (meters).
        z_rotation_deg: Per-group Z rotation range in +/- degrees (None or >=360 = full rotation).
        fixed_base_names: Set of object names whose scene-level ``fixed_base`` flag is True.
            These still need to be tracked so the caller can load them in the correct
            order, but whether they get perturbed depends on ``include_fixed_base_groups``.
        include_fixed_base_groups: If False (default), groups that contain any fixed_base
            object keep their settled poses (the legacy scene_sampling.py behavior).
            If True, those groups are perturbed like any other group — use this when you
            want fixtures (e.g. desks, organizers) to shift with their on-top objects as
            a rigid cluster. The caller is still responsible for pinning the fixed objects
            after settling (``obj.keep_still()`` + ``env.scene.update_initial_file()``).
    """
    fixed_base_names = set(fixed_base_names or [])
    new_poses = {}
    for group in groups:
        # Legacy behavior: skip randomization for groups that contain a fixed_base object.
        group_has_fixed = any(n in fixed_base_names for n in group)
        if group_has_fixed and not include_fixed_base_groups:
            for name in group:
                if name in settled_poses:
                    new_poses[name] = settled_poses[name]
            continue

        dx = np.random.uniform(-xy_range, xy_range)
        dy = np.random.uniform(-xy_range, xy_range)
        angle = _sample_z_rotation_rad(z_rotation_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_quat = th.tensor([0.0, 0.0, float(np.sin(angle / 2)), float(np.cos(angle / 2))])

        group_names = [n for n in group if n in settled_poses]
        if not group_names:
            continue

        positions = {n: th.tensor(settled_poses[n][0]) for n in group_names}
        centroid_xy = th.stack(list(positions.values()))[:, :2].mean(dim=0)

        for name in group_names:
            pos = positions[name]
            ori = th.tensor(settled_poses[name][1])
            rel_x = float(pos[0]) - float(centroid_xy[0])
            rel_y = float(pos[1]) - float(centroid_xy[1])
            new_pos = pos.clone()
            new_pos[0] = float(centroid_xy[0]) + cos_a * rel_x - sin_a * rel_y + dx
            new_pos[1] = float(centroid_xy[1]) + sin_a * rel_x + cos_a * rel_y + dy
            new_ori = T.quat_multiply(rot_quat, ori)
            new_poses[name] = [new_pos.tolist(), new_ori.tolist()]

    return new_poses


# ---------------------------------------------------------------------------
# Object reload with new poses
# ---------------------------------------------------------------------------

def reload_task_objects_with_poses(env, task_objects, scene_json, new_poses,
                                   settle_steps=100, fixed_base_names=None,
                                   bake_initial_file=True):
    """
    Restore full state for task objects from scene JSON, then override with new randomized
    poses. Fixed_base objects are loaded first and settled before placing others, so that
    movable objects don't intersect with them.

    Args:
        env: OmniGibson environment.
        task_objects: {name: obj} dict of live task object references.
        scene_json: Parsed scene JSON dict (used to look up per-object state).
        new_poses: {name: [pos_list, ori_list]} poses to apply after state restore.
        settle_steps: Physics steps to run for settling after reload.
        fixed_base_names: Set of names to load first (before movable objects).
        bake_initial_file: If True (default), call env.scene.update_initial_file() at
            the end of the reload so future env.reset() calls restore to this state.
            Pass False when the caller does its own validation before committing.
    """
    fixed_base_names = set(fixed_base_names or [])
    obj_registry = (
        scene_json.get("state", {})
        .get("registry", {})
        .get("object_registry", {})
    )

    fixed_objs = {n: o for n, o in task_objects.items() if n in fixed_base_names}
    movable_objs = {n: o for n, o in task_objects.items() if n not in fixed_base_names}

    def _load_and_place(objs):
        for obj_name, obj in objs.items():
            obj_state = obj_registry.get(obj_name)
            if obj_state is not None:
                obj.load_state(obj_state)
            if obj_name in new_poses:
                pos = th.tensor(new_poses[obj_name][0])
                ori = th.tensor(new_poses[obj_name][1])
                obj.set_position_orientation(position=pos, orientation=ori)

    # Phase 1: Load fixed_base objects first and let them settle
    if fixed_objs:
        _load_and_place(fixed_objs)
        for obj in fixed_objs.values():
            obj.keep_still()
        og.sim.step()

    # Phase 2: Load movable objects
    _load_and_place(movable_objs)
    for obj in task_objects.values():
        obj.wake()
    og.sim.step()

    # Settle
    print(f"  Letting objects settle ({settle_steps} steps)...")
    for _ in range(settle_steps):
        og.sim.step()
        og.sim.render()
    for obj in task_objects.values():
        obj.keep_still()

    if bake_initial_file:
        # Bake the post-settle state into the initial file.
        env.scene.update_initial_file()


# ---------------------------------------------------------------------------
# Relationship extraction + grouping
# ---------------------------------------------------------------------------

def extract_relationships(task_objects):
    """Return list of {object, other, state} dicts for OnTop relationships between task objects."""
    for obj in task_objects.values():
        obj.wake()
    og.sim.step()

    relationships = []
    names = list(task_objects.keys())
    for obj_name in names:
        obj = task_objects[obj_name]
        for other_name in names:
            if obj_name == other_name:
                continue
            other_obj = task_objects[other_name]
            if OnTop in obj.states:
                try:
                    if obj.states[OnTop].get_value(other_obj):
                        relationships.append({"object": obj_name, "other": other_name, "state": "OnTop"})
                        print(f"  FOUND: {obj_name} OnTop {other_name}")
                except Exception:
                    pass
    return relationships


def extract_relationships_for_group(task_objects, group_names):
    """Extract OnTop relationships where both objects belong to ``group_names``.

    This is a scoped version of :func:`extract_relationships` — more efficient
    when only a subset of objects needs validation.
    """
    group_set = set(group_names)
    for name in group_set:
        if name in task_objects:
            task_objects[name].wake()
    og.sim.step()

    relationships = []
    for obj_name in group_set:
        if obj_name not in task_objects:
            continue
        obj = task_objects[obj_name]
        for other_name in group_set:
            if other_name == obj_name or other_name not in task_objects:
                continue
            other_obj = task_objects[other_name]
            if OnTop in obj.states:
                try:
                    if obj.states[OnTop].get_value(other_obj):
                        relationships.append({"object": obj_name, "other": other_name, "state": "OnTop"})
                except Exception:
                    pass
    return relationships


def relationships_to_set(relationships):
    """Convert relationship list to a frozenset of (object, other, state) tuples."""
    return frozenset((r["object"], r["other"], r["state"]) for r in relationships)


def build_groups_from_relationships(relationships, all_obj_names):
    """Union-find grouping from OnTop relationships."""
    parent = {n: n for n in all_obj_names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for rel in relationships:
        if rel["object"] in parent and rel["other"] in parent:
            union(rel["object"], rel["other"])

    groups_dict = {}
    for name in all_obj_names:
        root = find(name)
        groups_dict.setdefault(root, set()).add(name)
    return list(groups_dict.values())
