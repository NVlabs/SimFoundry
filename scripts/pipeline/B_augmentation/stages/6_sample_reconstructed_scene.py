# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from b1k

Requires installing:

- BEHAVIOR-1K, see https://github.com/StanfordVL/BEHAVIOR-1K

This script extracts physical relationships (specifically "on_top") between objects
in the scene after settling, using OmniGibson's object state system.
"""
import os

# Point OmniGibson at this checkout's BEHAVIOR-1K datasets so generated
# real2sim assets resolve even if OmniGibson was installed from another clone.
from simfoundry import configure_omnigibson_data_path

configure_omnigibson_data_path(force=True)

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.lazy as lazy
from omnigibson.scenes import Scene
from omnigibson.objects import DatasetObject
from omnigibson.robots import REGISTERED_ROBOTS
import omnigibson.utils.transform_utils as T
from omnigibson.utils.config_utils import parse_config
from omnigibson.object_states import OnTop
from pathlib import Path
import torch as th
import json
import subprocess
import hydra
from omegaconf import OmegaConf
from simfoundry import CFG_DIR as SIMFOUNDRY_CFG_DIR
from simfoundry.pipeline.frame_selection import resolve_img_idx
from simfoundry.utils.processing_utils import dump_json
from simfoundry.utils.og_utils import set_obj_materials
import numpy as np
import logging

logger = logging.getLogger(__name__)

from simfoundry import CFG_DIR


### At the start of every script, we cd into the scripts/config directory
scripts_dir = os.path.dirname(os.path.abspath(__file__))
cfg_dir = CFG_DIR
os.chdir(cfg_dir)


def _cat(name, name_to_category):
    """Helper to format 'name (category)' for printing."""
    cat = name_to_category.get(name, "")
    return f"{name} ({cat})" if cat else name


def extract_on_top_relationships(objs, table=None, name_to_category=None):
    """
    Extract "on_top" relationships between all objects in the scene.
    
    Args:
        objs: dict mapping object names to DatasetObject instances
        table: optional table object to include in relationship checking
        name_to_category: optional dict mapping object names to category strings
    
    Returns:
        dict with "on_top_relationships" list, each entry includes "category" and
        "on_top_of_category" fields.
    """
    if name_to_category is None:
        name_to_category = {}

    relationships = []
    
    all_objs = dict(objs)
    if table is not None:
        all_objs["conference_table"] = table

    for obj in all_objs.values():
        obj.wake()
    og.sim.step()
    
    print(f"Checking on_top relationships for {len(all_objs)} objects...")
    
    for obj_name, obj in all_objs.items():
        if OnTop not in obj.states:
            print(f"  WARNING: {_cat(obj_name, name_to_category)} does not have OnTop state, skipping...")
            continue
     
        for other_name, other_obj in all_objs.items():
            if obj_name == other_name:
                continue
            
            try:
                is_on_top = obj.states[OnTop].get_value(other_obj)
                if is_on_top:
                    relationships.append({
                        "object": obj_name,
                        "category": name_to_category.get(obj_name, ""),
                        "on_top_of": other_name,
                        "on_top_of_category": name_to_category.get(other_name, ""),
                    })
                    print(f"  FOUND: {_cat(obj_name, name_to_category)} is ON TOP OF "
                          f"{_cat(other_name, name_to_category)}")
            except Exception as e:
                pass
    
    return {"on_top_relationships": relationships}


def extract_touching_relationships(objs, table=None, name_to_category=None):
    """
    Extract "touching" relationships between all object pairs.
    Only records each pair once (A touching B, not also B touching A).
    
    Args:
        objs: dict mapping object names to DatasetObject instances
        table: optional table object to include
        name_to_category: optional dict mapping object names to category strings
    
    Returns:
        list of dicts with object_a, category_a, object_b, category_b
    """
    from omnigibson.object_states import Touching

    if name_to_category is None:
        name_to_category = {}

    all_objs = dict(objs)
    if table is not None:
        all_objs["conference_table"] = table

    for obj in all_objs.values():
        obj.wake()
    og.sim.step()

    touching_pairs = set()
    names = sorted(all_objs.keys())

    for i, name_a in enumerate(names):
        obj_a = all_objs[name_a]
        if Touching not in obj_a.states:
            continue
        for name_b in names[i + 1:]:
            obj_b = all_objs[name_b]
            try:
                if obj_a.states[Touching].get_value(obj_b):
                    touching_pairs.add((name_a, name_b))
            except Exception:
                pass

    return [
        {
            "object_a": a,
            "category_a": name_to_category.get(a, ""),
            "object_b": b,
            "category_b": name_to_category.get(b, ""),
        }
        for a, b in sorted(touching_pairs)
    ]


def extract_all_relationships(objs, table=None, name_to_category=None, check_touching=True):
    """
    Extract both on_top and (optionally) touching relationships from the current scene state.

    Args:
        check_touching: If True, extract touching relationships; if False, touching_relationships is [].
    
    Returns:
        dict with keys "on_top_relationships" and "touching_relationships"
    """
    on_top_result = extract_on_top_relationships(objs, table=table, name_to_category=name_to_category)
    touching_result = (
        extract_touching_relationships(objs, table=table, name_to_category=name_to_category)
        if check_touching
        else []
    )
    return {
        "on_top_relationships": on_top_result["on_top_relationships"],
        "touching_relationships": touching_result,
    }


def relationships_to_hashable(relationships_dict):
    """
    Convert a relationships dict to a hashable representation for comparison.
    Returns a frozenset of tuples for both on_top and touching.
    """
    on_top = frozenset(
        (r["object"], r["on_top_of"]) for r in relationships_dict.get("on_top_relationships", [])
    )
    touching = frozenset(
        (r["object_a"], r["object_b"]) for r in relationships_dict.get("touching_relationships", [])
    )
    return (on_top, touching)


def calibrate_original_relationships(
    objs, settled_poses, table, scene, scene_objects_info, obj_frictions,
    sim_dir,
    n_trials=5, perturbation=0.005, settle_steps=50, name_to_category=None,
    check_touching=True,
):
    """
    Reload the original scene multiple times with small pose perturbations to
    determine the most stable/common relationship structure.

    Args:
        check_touching: If True, extract and calibrate touching relationships; if False, touching_relationships is [].
    
    Returns:
        dict: the most common relationships dict (on_top + touching)
    """
    from collections import Counter

    if name_to_category is None:
        name_to_category = {}

    print("\n" + "=" * 60)
    print(f"Calibrating original scene relationships ({n_trials} trials, perturbation={perturbation}m)")
    print("=" * 60)

    all_results = []
    hash_counter = Counter()
    hash_to_index = {}

    for trial in range(n_trials):
        print(f"\n  Trial {trial + 1}/{n_trials}...")

        perturbed_poses = {}
        for name, (pos, ori) in settled_poses.items():
            p = np.array(pos, dtype=np.float64)
            p[:3] += np.random.uniform(-perturbation, perturbation, size=3)
            perturbed_poses[name] = [p.tolist(), list(ori)]

        reload_objects_with_poses(
            scene=scene,
            objs=objs,
            scene_objects_info=scene_objects_info,
            new_poses=perturbed_poses,
            obj_frictions=obj_frictions,
            settle_steps=settle_steps,
        )

        rels = extract_all_relationships(
            objs, table=table, name_to_category=name_to_category, check_touching=check_touching
        )
        h = relationships_to_hashable(rels)

        all_results.append(rels)
        hash_counter[h] += 1
        if h not in hash_to_index:
            hash_to_index[h] = trial

        n_on_top = len(rels["on_top_relationships"])
        n_touching = len(rels["touching_relationships"]) if check_touching else None
        touch_str = f"touching: {n_touching}" if check_touching else "touching: (skipped)"
        print(f"    on_top: {n_on_top}, {touch_str}")

    most_common_hash = max(hash_counter, key=lambda h: (hash_counter[h], -hash_to_index[h]))
    best_idx = hash_to_index[most_common_hash]
    best_rels = all_results[best_idx]
    count = hash_counter[most_common_hash]

    print(f"\n  Most common relationship appeared {count}/{n_trials} times (from trial {best_idx + 1})")
    print(f"    on_top: {len(best_rels['on_top_relationships'])}")
    for r in best_rels["on_top_relationships"]:
        print(f"      {_cat(r['object'], name_to_category)} ON TOP OF "
              f"{_cat(r['on_top_of'], name_to_category)}")
    if check_touching:
        print(f"    touching: {len(best_rels['touching_relationships'])}")
        for r in best_rels["touching_relationships"]:
            print(f"      {_cat(r['object_a'], name_to_category)} TOUCHING "
                  f"{_cat(r['object_b'], name_to_category)}")
    else:
        print("    touching: (skipped)")
    print("=" * 60)

    return best_rels


def build_groups_from_relationships(relationships, all_obj_names):
    """
    Build groups of objects that are physically connected via on_top relationships.
    Uses union-find to merge objects that are transitively connected.
    
    Args:
        relationships: list of {"object": ..., "on_top_of": ...} dicts
        all_obj_names: list of all object names in the scene
    
    Returns:
        list of sets, each set contains object names belonging to one group
    """
    parent = {name: name for name in all_obj_names}

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
        obj_a = rel["object"]
        obj_b = rel["on_top_of"]
        if obj_a in parent and obj_b in parent:
            union(obj_a, obj_b)

    groups_dict = {}
    for name in all_obj_names:
        root = find(name)
        groups_dict.setdefault(root, set()).add(name)

    return list(groups_dict.values())


def _sample_z_rotation_rad(z_rotation_deg):
    """Sample Z rotation in radians. 360 or None => [0, 2*pi); else => ±z_rotation_deg in radians."""
    if z_rotation_deg is None or z_rotation_deg >= 360:
        return np.random.uniform(0, 2 * np.pi)
    half_rad = np.deg2rad(float(z_rotation_deg))
    return np.random.uniform(-half_rad, half_rad)


def apply_random_group_transform(objs, groups, xy_range=0.3, settle_steps=50, z_rotation_deg=360):
    """
    Apply a random XY translation and Z-axis rotation to each group,
    maintaining relative poses within the group.

    Args:
        objs: dict mapping object names to DatasetObject instances
        groups: list of sets of object names
        xy_range: max displacement in meters for X and Y
        settle_steps: number of sim steps to let objects settle after moving
        z_rotation_deg: Z rotation range in degrees; 360 = full [0,360°), else ±N degrees
    """
    for group in groups:
        dx = np.random.uniform(-xy_range, xy_range)
        dy = np.random.uniform(-xy_range, xy_range)
        angle = _sample_z_rotation_rad(z_rotation_deg)

        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Compute group centroid in XY for rotation pivot
        positions = {}
        orientations = {}
        for name in group:
            if name not in objs:
                continue
            pos, ori = objs[name].get_position_orientation()
            positions[name] = pos.clone()
            orientations[name] = ori.clone()

        if not positions:
            continue

        centroid_xy = th.stack(list(positions.values()))[:, :2].mean(dim=0)

        # Z-axis rotation quaternion in OmniGibson's (x, y, z, w) convention
        rot_quat = th.tensor([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)])

        for name in group:
            if name not in objs:
                continue
            pos = positions[name]
            ori = orientations[name]

            # Rotate XY around group centroid
            rel_x = pos[0] - centroid_xy[0]
            rel_y = pos[1] - centroid_xy[1]
            new_x = centroid_xy[0] + cos_a * rel_x - sin_a * rel_y + dx
            new_y = centroid_xy[1] + sin_a * rel_x + cos_a * rel_y + dy

            new_pos = pos.clone()
            new_pos[0] = new_x
            new_pos[1] = new_y

            new_ori = T.quat_multiply(rot_quat, ori)

            objs[name].set_position_orientation(position=new_pos, orientation=new_ori)

    # Let objects settle briefly (don't wake objects so they can reach zero velocity)
    for _ in range(settle_steps):
        og.sim.step()


def capture_current_poses(objs, table=None, robot=None):
    """Capture the current poses of all objects, table, and robot."""
    poses = {}
    for obj_name, obj in objs.items():
        pos, ori = obj.get_position_orientation()
        poses[obj_name] = [
            pos.tolist() if isinstance(pos, th.Tensor) else list(pos),
            ori.tolist() if isinstance(ori, th.Tensor) else list(ori),
        ]
    if table is not None:
        pos, ori = table.get_position_orientation()
        poses["conference_table"] = [
            pos.tolist() if isinstance(pos, th.Tensor) else list(pos),
            ori.tolist() if isinstance(ori, th.Tensor) else list(ori),
        ]
    if robot is not None:
        pos, ori = robot.get_position_orientation()
        poses["robot"] = [
            pos.tolist() if isinstance(pos, th.Tensor) else list(pos),
            ori.tolist() if isinstance(ori, th.Tensor) else list(ori),
        ]
    return poses


def restore_poses(objs, saved_poses):
    """Restore objects to previously saved poses."""
    for obj_name, obj in objs.items():
        if obj_name in saved_poses:
            pos = th.tensor(saved_poses[obj_name][0])
            ori = th.tensor(saved_poses[obj_name][1])
            obj.set_position_orientation(position=pos, orientation=ori)


def compute_randomized_poses(settled_poses, groups, xy_range=0.3, z_rotation_deg=360):
    """
    Compute new poses by applying random XY shift + Z rotation per group,
    without touching the simulation.

    Args:
        settled_poses: dict of obj_name -> [pos_list, ori_list]
        groups: list of sets of object names
        xy_range: max XY displacement in meters
        z_rotation_deg: Z rotation range in degrees; 360 = full [0,360°), else ±N degrees

    Returns:
        dict: new_poses mapping obj_name -> [pos_list, ori_list]
    """
    new_poses = {}

    for group in groups:
        dx = np.random.uniform(-xy_range, xy_range)
        dy = np.random.uniform(-xy_range, xy_range)
        angle = _sample_z_rotation_rad(z_rotation_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Z-axis rotation quaternion in OmniGibson (x, y, z, w)
        rot_quat = th.tensor([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)])

        # Collect positions for group centroid
        group_names = [n for n in group if n in settled_poses]
        if not group_names:
            continue

        positions = {n: th.tensor(settled_poses[n][0]) for n in group_names}
        centroid_xy = th.stack(list(positions.values()))[:, :2].mean(dim=0)

        for name in group_names:
            pos = positions[name]
            ori = th.tensor(settled_poses[name][1])

            rel_x = pos[0] - centroid_xy[0]
            rel_y = pos[1] - centroid_xy[1]
            new_x = centroid_xy[0] + cos_a * rel_x - sin_a * rel_y + dx
            new_y = centroid_xy[1] + sin_a * rel_x + cos_a * rel_y + dy

            new_pos = pos.clone()
            new_pos[0] = new_x
            new_pos[1] = new_y

            new_ori = T.quat_multiply(rot_quat, ori)

            new_poses[name] = [new_pos.tolist(), new_ori.tolist()]

    return new_poses


def reload_objects_with_poses(scene, objs, scene_objects_info, new_poses, obj_frictions, settle_steps=100):
    """
    Remove all objects and reload them with the given poses, then let settle.

    Args:
        scene: OmniGibson scene
        objs: dict of object name -> DatasetObject (modified in-place)
        scene_objects_info: original scene_objects_info dict
        new_poses: dict of obj_name -> [pos_list, ori_list] to load with
        obj_frictions: original obj_frictions dict
        settle_steps: number of steps to let simulation settle
    """
    # Remove existing objects
    for obj_name in list(objs.keys()):
        scene.remove_object(objs[obj_name])
    objs.clear()

    # Re-add objects with new poses
    with og.sim.stopped():
        for idx, obj_info in reversed(list(scene_objects_info.items())):
            obj_category = obj_info["category"]
            obj_model = obj_info["model"]
            obj_name = obj_info["name"]

            obj = DatasetObject(
                name=obj_name,
                dataset_name="real2sim-assets",
                category=obj_category,
                model=obj_model,
            )
            scene.add_object(obj)

            pos = th.tensor(new_poses[obj_name][0])
            ori = th.tensor(new_poses[obj_name][1])
            pos[2] -= 0.005
            obj.set_position_orientation(position=pos, orientation=ori)
            og.sim.step()

            obj_aabb = obj.aabb

            objs[obj_name] = obj
            for i in range(10):
                og.sim.step()

    # Re-apply materials
    set_obj_materials(objs=objs, obj_frictions=obj_frictions)

    # Let simulation settle (don't wake objects so they can reach zero velocity)
    print("  Letting objects settle after reload...")
    for i in range(settle_steps):
        og.sim.step()


def interactive_randomize_loop(
    objs, groups, settled_poses, out_dir,
    scene=None, scene_objects_info=None, obj_poses=None, obj_frictions=None,
    sim_dir=None,
    table=None, robot=None, xy_range=0.3, settle_steps=50, z_rotation_deg=360,
):
    """
    Interactive loop: press Enter to randomize groups, press 's' to save, Ctrl+C to exit.
    Each randomization reloads objects from scratch, then applies random group transforms.

    Args:
        z_rotation_deg: Z rotation range in degrees; 360 = full [0,360°), else ±N degrees
    """
    save_counter = 0

    # Build name -> category lookup
    name_to_category = {}
    if scene_objects_info:
        for _, obj_info in scene_objects_info.items():
            name_to_category[obj_info["name"]] = obj_info["category"]

    print("\n" + "=" * 60)
    print("Interactive group randomization")
    print("=" * 60)
    print(f"  {len(groups)} groups found:")
    for i, g in enumerate(groups):
        members = [f"{n} ({name_to_category.get(n, '?')})" for n in sorted(g)]
        print(f"    Group {i}: {', '.join(members)}")
    print()
    print("  Press ENTER to generate a random arrangement")
    print("  Type 's' then ENTER to save current arrangement")
    print("  Press Ctrl+C to exit")
    print("=" * 60)

    import select
    import sys

    def check_stdin():
        """Non-blocking check if there's input available on stdin."""
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        return bool(ready)

    try:
        print("\n> ", end="", flush=True)
        while True:
            # Keep simulation running
            og.sim.step()
            print(og.sim.viewer_camera.get_position_orientation())

            # Check for user input without blocking
            if check_stdin():
                user_input = sys.stdin.readline().strip().lower()

                if user_input == "s":
                    save_counter += 1
                    poses = capture_current_poses(objs, table=table, robot=robot)
                    save_path = f"{out_dir}/settled_poses_random_{save_counter:03d}.json"
                    with open(save_path, "w") as f:
                        json.dump(poses, f, indent=2)
                    print(f"  Saved arrangement to: {save_path}")
                else:
                    # Compute randomized poses from the original settled poses
                    new_poses = compute_randomized_poses(
                        settled_poses, groups, xy_range=xy_range, z_rotation_deg=z_rotation_deg
                    )

                    # Reload objects with the transformed poses and let them settle
                    print("  Reloading objects with randomized poses...")
                    reload_objects_with_poses(
                        scene=scene,
                        objs=objs,
                        scene_objects_info=scene_objects_info,
                        new_poses=new_poses,
                        obj_frictions=obj_frictions,
                        settle_steps=settle_steps,
                    )
                    print("  Done. Press ENTER for another, 's' to save, Ctrl+C to exit.")

                print("\n> ", end="", flush=True)

    except (KeyboardInterrupt, EOFError):
        print("\nExiting randomization loop.")


def take_scene_picture(save_path, camera_pos=None, camera_ori=None, focal_length=10.0):
    """
    Set viewer camera pose, render, and save the current view as a PNG.
    Uses a lower focal length (wider FOV) so all tabletop objects fit in frame.

    Args:
        save_path: Full path for the output image (e.g. .../scene_000.png).
        camera_pos: (3,) position; default fixed pose if None.
        camera_ori: (4,) quaternion orientation; default if None.
        focal_length: Camera focal length in mm; lower = wider FOV (default 10.0 so scene fits).
    """
    from PIL import Image as PILImage
    if camera_pos is None:
        camera_pos = th.tensor([-0.0466, -0.7612, 0.5355])
    if camera_ori is None:
        camera_ori = th.tensor([0.5219, -0.0213, -0.0347, 0.8521])
    cam = og.sim.viewer_camera
    orig_focal = cam.focal_length
    try:
        cam.focal_length = focal_length
        cam.set_position_orientation(
            position=camera_pos,
            orientation=camera_ori,
        )
        for _ in range(5):
            og.sim.render()
        obs, _ = cam.get_obs()
        rgb = obs["rgb"]
        if isinstance(rgb, th.Tensor):
            rgb_np = rgb.cpu().numpy()
        else:
            rgb_np = np.array(rgb)
        if rgb_np.ndim == 3 and rgb_np.shape[-1] == 4:
            rgb_np = rgb_np[:, :, :3]
        if rgb_np.dtype != np.uint8:
            if rgb_np.max() <= 1.0:
                rgb_np = (rgb_np * 255).astype(np.uint8)
            else:
                rgb_np = rgb_np.astype(np.uint8)
        PILImage.fromarray(rgb_np).save(save_path)
        print(f"  Saved image: {save_path}")
    finally:
        cam.focal_length = orig_focal


def automated_generation(
    objs, groups, settled_poses, out_dir,
    scene=None, scene_objects_info=None, obj_poses=None, obj_frictions=None,
    sim_dir=None,
    table=None, xy_range=0.3, settle_steps=50, iter_num=10,
    camera_pos=None, camera_ori=None,
    calibrated_relationships=None, name_to_category=None,
    check_touching=True, z_rotation_deg=360,
):
    """
    Automated mode: randomize objects, take a picture, validate on_top (and
    optionally touching) relationships against the calibrated original, and save results.

    A scene is valid if on_top matches and, when check_touching is True, touching also matches.
    """
    if camera_pos is None:
        camera_pos = th.tensor([-0.0466, -0.7612, 0.5355])
    if camera_ori is None:
        camera_ori = th.tensor([0.5219, -0.0213, -0.0347, 0.8521])
    if name_to_category is None:
        name_to_category = {}

    auto_dir = f"{out_dir}/auto_generation"
    Path(auto_dir).mkdir(parents=True, exist_ok=True)

    if calibrated_relationships is not None:
        original_hash = relationships_to_hashable(calibrated_relationships)
        original_on_top = calibrated_relationships["on_top_relationships"]
        original_touching = calibrated_relationships["touching_relationships"]
    else:
        original_on_top = []
        original_touching = []
        original_hash = relationships_to_hashable({
            "on_top_relationships": [], "touching_relationships": []
        })

    original_groups = [frozenset(g) for g in groups]
    original_groups_sorted = sorted(original_groups, key=lambda s: sorted(s)[0])

    results = []

    print("\n" + "=" * 60)
    print(f"Automated generation: {iter_num} iterations")
    if check_touching:
        print(f"  Validating against calibrated on_top ({len(original_on_top)}) "
              f"and touching ({len(original_touching)}) relationships")
    else:
        print(f"  Validating against calibrated on_top ({len(original_on_top)}) only (touching skipped)")
    print("=" * 60)

    for iteration in range(iter_num):
        print(f"\n--- Iteration {iteration + 1}/{iter_num} ---")

        new_poses = compute_randomized_poses(
            settled_poses, groups, xy_range=xy_range, z_rotation_deg=z_rotation_deg
        )

        print("  Reloading objects with randomized poses...")
        reload_objects_with_poses(
            scene=scene,
            objs=objs,
            scene_objects_info=scene_objects_info,
            new_poses=new_poses,
            obj_frictions=obj_frictions,
            settle_steps=settle_steps,
        )

        # Validate on_top and optionally touching
        new_rels = extract_all_relationships(
            objs, table=table, name_to_category=name_to_category, check_touching=check_touching
        )
        new_hash = relationships_to_hashable(new_rels)

        new_on_top_set, new_touching_set = new_hash
        orig_on_top_set, orig_touching_set = original_hash

        on_top_match = (new_on_top_set == orig_on_top_set)
        touching_match = (new_touching_set == orig_touching_set) if check_touching else True
        is_valid = on_top_match and touching_match

        all_obj_names = list(objs.keys())
        new_groups = build_groups_from_relationships(
            new_rels["on_top_relationships"], all_obj_names
        )
        new_groups_sorted = sorted(
            [frozenset(g) for g in new_groups], key=lambda s: sorted(s)[0]
        )

        print(f"  on_top  match: {on_top_match}  "
              f"(orig={len(orig_on_top_set)}, new={len(new_on_top_set)})")
        if check_touching:
            print(f"  touching match: {touching_match}  "
                  f"(orig={len(orig_touching_set)}, new={len(new_touching_set)})")
        print(f"  Valid: {is_valid}")

        if not on_top_match:
            missing_on_top = orig_on_top_set - new_on_top_set
            extra_on_top = new_on_top_set - orig_on_top_set
            if missing_on_top:
                print(f"    Missing on_top: "
                      f"{[f'{_cat(a, name_to_category)} ON {_cat(b, name_to_category)}' for a, b in missing_on_top]}")
            if extra_on_top:
                print(f"    Extra on_top:   "
                      f"{[f'{_cat(a, name_to_category)} ON {_cat(b, name_to_category)}' for a, b in extra_on_top]}")

        if check_touching and not touching_match:
            missing_touch = orig_touching_set - new_touching_set
            extra_touch = new_touching_set - orig_touching_set
            if missing_touch:
                print(f"    Missing touching: "
                      f"{[f'{_cat(a, name_to_category)} <-> {_cat(b, name_to_category)}' for a, b in missing_touch]}")
            if extra_touch:
                print(f"    Extra touching:   "
                      f"{[f'{_cat(a, name_to_category)} <-> {_cat(b, name_to_category)}' for a, b in extra_touch]}")

        # Keep objects still and zero out velocities before saving so the JSON has clean state
        og.sim.step_physics()
        for obj in objs.values():
            obj.keep_still()
        if table is not None:
            table.keep_still()

        # Take picture and capture poses AFTER relationships are extracted and objects are stilled,
        # so the image and saved poses match the scene JSON that will be loaded by step 14.
        img_filename = f"scene_{iteration:03d}.png"
        img_path = f"{auto_dir}/{img_filename}"
        take_scene_picture(img_path, camera_pos=camera_pos, camera_ori=camera_ori)

        poses = capture_current_poses(objs, table=table)
        poses_filename = f"settled_poses_{iteration:03d}.json"
        poses_path = f"{auto_dir}/{poses_filename}"
        with open(poses_path, "w") as f:
            json.dump(poses, f, indent=2)

        # Save scene state as JSON (same format as reconstructed_og_scene.json for use by step 14)
        scene_json_filename = f"scene_{iteration:03d}.json"
        scene_json_path = f"{auto_dir}/{scene_json_filename}"
        scene.update_initial_file()
        og.sim.save(json_paths=[scene_json_path])
        print(f"  Saved scene: {scene_json_path}")

        # Groups with categories for JSON
        groups_with_categories = []
        for g in new_groups_sorted:
            groups_with_categories.append([
                {"name": n, "category": name_to_category.get(n, "")} for n in sorted(g)
            ])

        results.append({
            "iteration": iteration,
            "image": img_filename,
            "poses": poses_filename,
            "valid": is_valid,
            "scene_json": scene_json_filename,
            "on_top_match": on_top_match,
            "touching_match": touching_match,
            "on_top_relationships": new_rels["on_top_relationships"],
            "touching_relationships": new_rels["touching_relationships"],
            "groups": groups_with_categories,
        })

    # Summary: valid count and per-scene valid + scene_json (for use by step 14)
    summary = {
        "valid_count": sum(1 for r in results if r["valid"]),
        "invalid_count": sum(1 for r in results if not r["valid"]),
        "scenes": [
            {"iteration": r["iteration"], "valid": r["valid"], "scene_json": r["scene_json"]}
            for r in results
        ],
    }
    summary_path = f"{auto_dir}/generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Automated generation complete!")
    print(f"  Valid: {summary['valid_count']}/{iter_num}")
    print(f"  Invalid: {summary['invalid_count']}/{iter_num}")
    print(f"  Summary saved to: {summary_path}")
    print("=" * 60)


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    sim_dir = cfg.s11_sim.out_dir
    physics_dir = cfg.s12_physics.out_dir
    out_dir = cfg.s14_og.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)


    # Load physics objects info
    include_gs = cfg.s14_og.include_gs

    # TODO: Remove this once include_gs is fully validated
    # assert not include_gs, "TODO: Gaussian background still needs to be validated!"


    scene_objects_info_fpath = f"{sim_dir}/scene_objects_info.json"
    obj_frictions = dict()
    if os.path.exists(scene_objects_info_fpath):
        with open(scene_objects_info_fpath, "r") as f:
            scene_objects_info = json.load(f)

        for _, obj_info in scene_objects_info.items():
            obj_frictions[obj_info["name"]] = obj_info["friction"]

        # Load pose info
        obj_poses_fpath = f"{physics_dir}/pb_scene_poses.json"
        with open(obj_poses_fpath, "r") as f:
            obj_poses = json.load(f)
    else:
        scene_objects_info = dict()
        obj_frictions = dict()
        obj_poses = dict()

    # Build name -> category lookup from scene_objects_info
    name_to_category = {}
    for _, obj_info in scene_objects_info.items():
        name_to_category[obj_info["name"]] = obj_info["category"]

    # Create scene config

    scene_cfg = {
        "type": "Scene",
        "use_floor_plane": True,
        "floor_plane_visible": not include_gs,
        "use_skybox": not include_gs,
    }
    robots_cfg = []
    if cfg.s14_og.include_robot:
        # Add the robot we want to load
        robot_cfg = OmegaConf.to_container(cfg.s14_og.robot_config, resolve=True)
        arms = ["0"]        # TODO: Support bimanual robots
        robot_cfg["controller_config"] = {}
        for arm in arms:
            # TODO: Make this use Joint Position later
            # robot_cfg["controller_config"][f"arm_{arm}"] = {
            #     "name": "JointController",
            #     "command_input_limits": None,
            #     "use_delta_commands": False,
            # }
            robot_cfg["controller_config"][f"arm_{arm}"] = {
                "name": "InverseKinematicsController",
                "command_input_limits": None,
                "mode": "pose_delta_ori",
                "smoothing_filter_size": 5,
                # "use_delta_commands": True,
            }
            robot_cfg["controller_config"][f"gripper_{arm}"] = {
                "name": "MultiFingerGripperController",
                "command_input_limits": (0.0, 1.0),
                "mode": "smooth",
            }
        robots_cfg.append(robot_cfg)

    external_sensors_cfg_path = f"{SIMFOUNDRY_CFG_DIR}/external_sensors/{cfg.s14_og.external_sensors_cfg}.yaml"
    env_cfg = {
        "external_sensors": parse_config(external_sensors_cfg_path)["external_sensors"]
    }

    og_cfg = dict(env=env_cfg, scene=scene_cfg, robots=robots_cfg)

    # Create the environment
    env = og.Environment(configs=og_cfg)
    env.reset()
    

    scene = env.scene
    z_offset = 0.0
    table = None  # Initialize table variable

    if not include_gs:
        if cfg.s14_og.include_table:
            # Add conference table to the scene
            table = DatasetObject(
                name="conference_table",
                dataset_name="behavior-1k-assets",
                category="conference_table",  # Using breakfast_table as it's more common; can use conference_table if available
                model="qzmjrj",  # A specific model; will use first available if this doesn't exist
            )
            scene.add_object(table)

            # Position table at origin with standard orientation
            table_position = th.tensor([0.5, 0.0, 0.0])
            table_orientation = th.tensor([0.0, 0.0, 0.0, 1.0])  # Identity quaternion (w, x, y, z)
            table.set_position_orientation(position=table_position, orientation=table_orientation)

            # Step simulation to get table dimensions
            og.sim.play()
            for i in range(10):
                og.sim.step()

            # Get table's top surface Z coordinate (assumes table is axis-aligned)
            table_aabb = table.aabb
            z_offset = table_aabb[1][2]  # Max Z of bounding box

            print(f"Table positioned at {table_position}")
            print(f"Table top surface at Z = {z_offset}")
    else:
        og.sim.play()
        for i in range(10):
            og.sim.step()

    objs = dict()
    with og.sim.stopped():
        for idx, obj_info in reversed(list(scene_objects_info.items())):
            
            obj_category = obj_info["category"]
            obj_model = obj_info["model"]
            obj_name = obj_info["name"]

            # if "banana" not in obj_category and "plate" not in obj_category:
            #     continue

            # Create and import object
            obj = DatasetObject(
                name=obj_name,
                dataset_name="real2sim-assets",
                category=obj_category,
                model=obj_model,
                # visual_only=True,
            )
            scene.add_object(obj)
            
            # Get original pose
            original_pos = th.tensor(obj_poses[obj_name][0])
            original_ori = th.tensor(obj_poses[obj_name][1])
            
            # Preserve X, Y but adjust Z to be on table
            # We'll adjust after getting object's own height
            adjusted_pos = original_pos.clone()
            adjusted_pos[2] -= 0.005
            # Set position temporarily to get object dimensions
            obj.set_position_orientation(position=adjusted_pos, orientation=original_ori)
            og.sim.step()
            
            # Get object's bottom Z coordinate
            obj_aabb = obj.aabb
            # obj_bottom_z = obj_aabb[0][2]  # Min Z of bounding box
            # obj_height_offset = adjusted_pos[2] - obj_bottom_z  # How much above its bottom the center is
            
            # # Place object on table: table_top_z + offset from bottom to center
            # if not include_gs:
            #     adjusted_pos[2] = z_offset + obj_height_offset + 0.02  # Small gap to prevent initial collision
            
            # # Set final position
            # obj.set_position_orientation(position=adjusted_pos, orientation=original_ori)
            
            # print(f"Placed {obj_name} at position {adjusted_pos} (original Z: {original_pos[2]:.3f}, new Z: {adjusted_pos[2]:.3f})")

            objs[obj_name] = obj
            for i in range(10):
                og.sim.step()

    # Set object materials
    obj_materials = set_obj_materials(objs=objs, obj_frictions=obj_frictions)

    # Let simulation settle before capturing final poses
    print("\nLetting simulation settle...")
    settle_steps = cfg.s14_og.get("settle_steps", 60)
    for i in range(settle_steps):
        og.sim.step()
        if (i + 1) % 20 == 0:
            print(f"  Settling step {i + 1}/{settle_steps}...")

    # Capture final poses after settling
    print("\nCapturing final object poses...")
    settled_poses = {}
    for obj_name, obj in objs.items():
        pos, ori = obj.get_position_orientation()
        # Convert to lists (position is tensor, orientation is quaternion (w, x, y, z))
        settled_poses[obj_name] = [
            pos.tolist() if isinstance(pos, th.Tensor) else list(pos),
            ori.tolist() if isinstance(ori,th.Tensor) else list(ori)
        ]
        cat = name_to_category.get(obj_name, "")
        label = f"{obj_name} ({cat})" if cat else obj_name
        print(f"  {label}: pos={pos.numpy() if isinstance(pos, th.Tensor) else pos}, ori={ori.numpy() if isinstance(ori, th.Tensor) else ori}")

    # Also capture table pose if it exists
    if cfg.s14_og.include_table and not include_gs and table is not None:
        table_pos, table_ori = table.get_position_orientation()
        settled_poses["conference_table"] = [
            table_pos.tolist() if isinstance(table_pos, th.Tensor) else list(table_pos),
            table_ori.tolist() if isinstance(table_ori, th.Tensor) else list(table_ori)
        ]
        print(f"  conference_table: pos={table_pos.numpy() if isinstance(table_pos, th.Tensor) else table_pos}, ori={table_ori.numpy() if isinstance(table_ori, th.Tensor) else table_ori}")

    # Also capture robot pose if it exists
    if cfg.s14_og.include_robot:
        robot = env.robots[0]
        robot_pos, robot_ori = robot.get_position_orientation()
        settled_poses["robot"] = [
            robot_pos.tolist() if isinstance(robot_pos, th.Tensor) else list(robot_pos),
            robot_ori.tolist() if isinstance(robot_ori, th.Tensor) else list(robot_ori)
        ]
        print(f"  robot: pos={robot_pos.numpy() if isinstance(robot_pos, th.Tensor) else robot_pos}, ori={robot_ori.numpy() if isinstance(robot_ori, th.Tensor) else robot_ori}")

    # Save settled poses
    settled_poses_path = f"{out_dir}/settled_poses.json"
    with open(settled_poses_path, "w") as f:
        json.dump(settled_poses, f, indent=2)
    print(f"\nSaved settled poses to: {settled_poses_path}")

    # Extract physical relationships (on_top). Save to disk only when not running auto_generation
    # (when auto_generation runs, we save the calibrated relationships to physical_relationships.json instead)
    print("\nExtracting physical relationships...")
    relationships = extract_on_top_relationships(
        objs,
        table=table if cfg.s14_og.include_table and not include_gs else None,
        name_to_category=name_to_category,
    )
    print(f"Found {len(relationships['on_top_relationships'])} on_top relationships")
    if not cfg.s14_og.get("auto_generation", False):
        relationships_path = f"{out_dir}/physical_relationships.json"
        with open(relationships_path, "w") as f:
            json.dump(relationships, f, indent=2)
        print(f"Saved physical relationships to: {relationships_path}")

    # Build groups from on_top relationships
    all_obj_names = list(objs.keys())
    groups = build_groups_from_relationships(
        relationships["on_top_relationships"], all_obj_names
    )

    # Save original relationship groups (used for randomization) under s14_og
    relationship_groups_path = f"{out_dir}/relationship_group.json"
    relationship_groups_data = {
        "groups": [sorted(g) for g in groups],
    }
    with open(relationship_groups_path, "w") as f:
        json.dump(relationship_groups_data, f, indent=2)
    print(f"Saved relationship groups to: {relationship_groups_path}")

    # Include 3DGS background if specified
    if include_gs:
        cam2world_tf_fpath = f"{cfg.s4_frame.out_dir}/image_{resolve_img_idx(cfg)}_cam2world.npy"
        cam2world_tf = th.from_numpy(np.load(cam2world_tf_fpath)).float()
        gs_path = os.path.abspath(f"{cfg.s2_da.out_dir}/gs_outputs/gaussian.usdz")
        gs_path_da3 = os.path.abspath(f"{cfg.s2_da.out_dir}/gs_outputs/gaussian_da3.usdz")

        from omnigibson.objects import USDObject

        # gs_background = USDObject(
        #     name="background",
        #     usd_path=gs_path,
        # )

        # Same guard stage 14 uses. Without it, a missing splat is a FileNotFoundError inside
        # USDObject followed by a segfault, rather than a scene without a background.
        if not os.path.exists(gs_path_da3):
            logger.warning(
                "include_gs=True but the GS USDZ was not found at %s. Run the "
                "auto_bg_reconstruction pipeline to produce it, or set s14_og.include_gs=false. "
                "Skipping the GS background for this run.",
                gs_path_da3,
            )
        else:
            gs_background_da3 = USDObject(
                name="background_da3",
                usd_path=gs_path_da3,
            )

            env.scene.add_object(gs_background_da3)
            og.sim.step()
            gs_background_da3.set_position_orientation(*T.mat2pose(cam2world_tf))
        og.sim.step()
        og.sim.render()
        og.sim.render()
        og.sim.render()

        # TODO: Get camK2cam0 tf as well, since 3DGS is taken wrt first cam frame


    if cfg.s14_og.include_robot:
        # Spawn / position the robot
        robot = env.robots[0]
        og.sim.play()
        robot.reset()
        robot.keep_still()
        robot_position = th.tensor(cfg.s14_og.robot_config.position)
        robot.set_position_orientation(
            position=robot_position,
            orientation=th.tensor(cfg.s14_og.robot_config.orientation),
        )
        print(f"Robot '{robot.name}' positioned at {robot_position}")
        # Let robot settle
        for i in range(20):
            og.sim.step()

    print("\n" + "="*60)
    print("Scene created successfully!")
    print("="*60)
    if cfg.s14_og.include_table and not include_gs:
        print(f"Table: conference_table at {table_position}")
        print(f"Table top surface Z: {z_offset:.3f}")
    print(f"N Objects: {len(objs)}")
    print(f"  - {', '.join(list(objs.keys())[:5])}{'...' if len(objs) > 5 else ''}")
    if cfg.s14_og.include_robot:
        print(f"Robot: {robot.name} at {robot_position}")
    print("="*60 + "\n")

    # Save this config
    env.scene.update_initial_file()
    json_path = f"{out_dir}/reconstructed_og_scene.json"
    og.sim.save(json_paths=[json_path])

    robot_ref = env.robots[0] if cfg.s14_og.include_robot else None
    table_ref = table if cfg.s14_og.include_table and not include_gs else None

    auto_mode = cfg.s14_og.get("auto_generation", False)
    if auto_mode:
        iter_num = cfg.s14_og.get("auto_iter_num", 10)
        cam_pos_list = cfg.s14_og.get("auto_camera_pos", [-0.0466, -0.7612, 0.5355])
        cam_ori_list = cfg.s14_og.get("auto_camera_ori", [0.5219, -0.0213, -0.0347, 0.8521])
        n_calibration_trials = cfg.s14_og.get("auto_calibration_trials", 5)
        calibration_perturbation = cfg.s14_og.get("auto_calibration_perturbation", 0.001)

        check_touching = cfg.s14_og.get("check_touching", True)
        if n_calibration_trials <= 0:
            print("Skipping calibration trials; using relationships extracted from the settled original scene.")
            calibrated_rels = {
                "on_top_relationships": relationships["on_top_relationships"],
                "touching_relationships": [],
            }
        else:
            # Calibrate: reload original scene multiple times with small perturbations
            # to find the most stable/common relationship structure
            calibrated_rels = calibrate_original_relationships(
                objs=objs,
                settled_poses=settled_poses,
                table=table_ref,
                scene=scene,
                scene_objects_info=scene_objects_info,
                obj_frictions=obj_frictions,
                sim_dir=sim_dir,
                n_trials=n_calibration_trials,
                perturbation=calibration_perturbation,
                settle_steps=cfg.s14_og.get("randomize_settle_steps", 20),
                name_to_category=name_to_category,
                check_touching=check_touching,
            )

        # Save calibrated relationships as physical_relationships.json (single relationship file for s14_og)
        relationships_path = f"{out_dir}/physical_relationships.json"
        with open(relationships_path, "w") as f:
            json.dump(calibrated_rels, f, indent=2)
        print(f"Saved calibrated relationships to: {relationships_path}")

        # Rebuild groups from calibrated on_top relationships
        all_obj_names = list(objs.keys())
        groups = build_groups_from_relationships(
            calibrated_rels["on_top_relationships"], all_obj_names
        )

        automated_generation(
            objs=objs,
            groups=groups,
            settled_poses=settled_poses,
            out_dir=out_dir,
            scene=scene,
            scene_objects_info=scene_objects_info,
            obj_poses=obj_poses,
            obj_frictions=obj_frictions,
            sim_dir=sim_dir,
            table=table_ref,
            xy_range=cfg.s14_og.get("randomize_xy_range", 0.02),
            settle_steps=cfg.s14_og.get("randomize_settle_steps", 20),
            iter_num=iter_num,
            camera_pos=th.tensor(cam_pos_list),
            camera_ori=th.tensor(cam_ori_list),
            calibrated_relationships=calibrated_rels,
            name_to_category=name_to_category,
            check_touching=check_touching,
            z_rotation_deg=cfg.s14_og.get("randomize_z_rotation_deg", 360),
        )
    else:
        interactive_randomize_loop(
            objs=objs,
            groups=groups,
            settled_poses=settled_poses,
            out_dir=out_dir,
            scene=scene,
            scene_objects_info=scene_objects_info,
            obj_poses=obj_poses,
            obj_frictions=obj_frictions,
            sim_dir=sim_dir,
            table=table_ref,
            robot=robot_ref,
            xy_range=cfg.s14_og.get("randomize_xy_range", 0.02),
            settle_steps=cfg.s14_og.get("randomize_settle_steps", 20),
            z_rotation_deg=cfg.s14_og.get("randomize_z_rotation_deg", 360),
        )

    og.shutdown()
    
    end_stage(cfg, success=True)

def end_stage(cfg, success=False, additional_info: dict = None):
    """
    Function that is run at the end of every stage to save the stage config and additional info.

    # TODO: should this be a general function that can be used for all stages? or should we have a separate function for each stage?
    """
    save_dir = cfg.s14_og.out_dir
    stage_cfg = OmegaConf.to_object(cfg.s14_og)
    stage_cfg['success'] = success
    if additional_info is not None:
        stage_cfg.update(additional_info)
    dump_json(stage_cfg, f"{save_dir}/stage_info.json")


if __name__ == "__main__":
    main()
