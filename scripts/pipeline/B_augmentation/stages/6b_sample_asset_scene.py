# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from simfoundry env

Sample randomized pose variants from a pre-built asset scene JSON.

Works like 13b_og_scene_sampling.py but loads objects from an existing scene JSON
(e.g. assets/scenes/nv_desk_.../..._scene_state_latest.json) instead of rebuilding
objects from DatasetObject + scene_objects_info.json.

Each iteration removes all task objects from the scene and re-adds them at new poses
(pos[2] -= 0.005 to ensure surface contact), exactly mirroring 13b's
reload_objects_with_poses approach.

Output is written to <input_scene_dir>/sampling_scene/ in the same format that
14_teleop_og_scene.py expects when scene_source=load_sampling.

Usage:
  python scene_sampling.py scene_name=nv_desk \\
      scene_sampling.input_scene_json=assets/scenes/nv_desk/nv_desk_scene_state_latest.json \\
      scene_sampling.n_iterations=10

Then run step 14 with:
  python scripts/pipeline/C_application/stages/2_teleop_og_scene.py scene_name=nv_desk task.task_name=<task> \\
      s14_teleop.scene_json_name=nv_desk s14_teleop.scene_source=load_sampling

  To teleop only specific sampled scenes (e.g. scenes 0 and 3):
  python scripts/pipeline/C_application/stages/2_teleop_og_scene.py ... s14_teleop.sampling_scene_indices=[0,3]

Requires installing:
- BEHAVIOR-1K, see https://github.com/StanfordVL/BEHAVIOR-1K
"""

import os
import json
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path

import torch as th
import hydra
from omegaconf import OmegaConf
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.robots import BaseRobot

from simfoundry import import_og_dependencies, REPO_DIR
from simfoundry.utils.og_utils import apply_teleop_omnigibson_macros
from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths
from simfoundry.utils.scene_sampling_utils import (
    get_task_object_names,
    get_fixed_base_names,
    capture_settled_poses,
    compute_randomized_poses,
    reload_task_objects_with_poses,
    extract_relationships,
    relationships_to_set,
    build_groups_from_relationships,
)

import_og_dependencies()

gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128

from simfoundry import CFG_DIR

scripts_dir = os.path.dirname(os.path.abspath(__file__))
cfg_dir = CFG_DIR
os.chdir(cfg_dir)


# ---------------------------------------------------------------------------
# Object helpers
# ---------------------------------------------------------------------------

def restore_poses_from_json(env, scene_json):
    """
    Restore full state (pose + joints + velocities) for all objects from scene_json.
    Uses OmniGibson's scene.restore() which properly converts JSON lists to tensors
    via recursively_convert_to_torch before calling load_state.
    """
    env.scene.restore(scene_file=scene_json)
    for obj in env.scene.objects:
        obj.wake()
    og.sim.step()
    print("Letting scene settle after load...")
    for _ in range(500):
        og.sim.step()
        og.sim.render()
    for obj in env.scene.objects:
        obj.keep_still()


# ---------------------------------------------------------------------------
# Image capture
# ---------------------------------------------------------------------------

def take_scene_picture(save_path, camera_pos=None, camera_ori=None, focal_length=10.0):
    """Render the current view and save as PNG (same as 13b)."""
    from PIL import Image as PILImage

    if og.sim.viewer_camera is None:
        print(f"  Skipping image capture (headless mode, no viewer camera)")
        return

    if camera_pos is None:
        camera_pos = th.tensor([-0.0466, -0.7612, 0.5355])
    if camera_ori is None:
        camera_ori = th.tensor([0.5219, -0.0213, -0.0347, 0.8521])

    cam = og.sim.viewer_camera
    orig_focal = cam.focal_length
    try:
        cam.focal_length = focal_length
        cam.set_position_orientation(position=camera_pos, orientation=camera_ori)
        for _ in range(5):
            og.sim.render()
        obs, _ = cam.get_obs()
        rgb = obs["rgb"]
        rgb_np = rgb.cpu().numpy() if isinstance(rgb, th.Tensor) else np.array(rgb)
        if rgb_np.ndim == 3 and rgb_np.shape[-1] == 4:
            rgb_np = rgb_np[:, :, :3]
        if rgb_np.dtype != np.uint8:
            rgb_np = (rgb_np * 255).astype(np.uint8) if rgb_np.max() <= 1.0 else rgb_np.astype(np.uint8)
        PILImage.fromarray(rgb_np).save(save_path)
        print(f"  Saved image: {save_path}")
    finally:
        cam.focal_length = orig_focal


def save_scene_json(env, json_path):
    """
    Save the scene state to a JSON file, matching interactive_scene_editor.py's approach:
      1. update_initial_file() so og.sim.save() captures current state
      2. og.sim.save()
      3. Convert all USD paths to relative (portable scenes)
      4. Save ground plane info
    """
    env.scene.update_initial_file()
    og.sim.save(json_paths=[str(json_path)])

    json_path = Path(json_path)
    json_dir = json_path.parent

    with open(json_path, "r") as f:
        scene_data = json.load(f)

    # Convert USD paths to relative
    if "objects_info" in scene_data and "init_info" in scene_data["objects_info"]:
        for obj_info in scene_data["objects_info"]["init_info"].values():
            if "args" in obj_info and "usd_path" in obj_info["args"]:
                usd_path = obj_info["args"]["usd_path"]
                if usd_path:
                    try:
                        obj_info["args"]["usd_path"] = os.path.relpath(usd_path, json_dir)
                    except ValueError:
                        pass  # different drives on Windows

    # Save ground plane info
    if og.sim.floor_plane is not None:
        floor_pos, floor_ori = og.sim.floor_plane.get_position_orientation()
        scene_data["ground_plane_info"] = {
            "position": floor_pos.tolist(),
            "orientation": floor_ori.tolist(),
        }

    with open(json_path, "w") as f:
        json.dump(scene_data, f, indent=2)


# ---------------------------------------------------------------------------
# Main sampling loop
# ---------------------------------------------------------------------------

def run_sampling(env, scene_json, task_objects, groups, settled_poses, original_rel_set,
                 out_dir, num_samples, xy_range, z_rotation_deg, settle_steps,
                 camera_pos, camera_ori, fixed_base_names=None):
    """
    Randomize, validate, and save scene_NNN.json files.
    Only saves successful (valid) scenes. Stops after num_samples successes
    or 2*num_samples total attempts, whichever comes first.
    Writes generation_summary.json in the same format as 13b_og_scene_sampling.py.
    """
    auto_dir = Path(out_dir) / "auto_generation"
    auto_dir.mkdir(parents=True, exist_ok=True)

    max_attempts = 2 * num_samples
    results = []
    success_count = 0
    for attempt in range(max_attempts):
        if success_count >= num_samples:
            break

        print(f"\n--- Attempt {attempt + 1}/{max_attempts} (successes: {success_count}/{num_samples}) ---")

        # Compute new poses analytically from settled_poses (fixed_base groups keep their poses)
        new_poses = compute_randomized_poses(settled_poses, groups, xy_range, z_rotation_deg,
                                             fixed_base_names=fixed_base_names)

        # Load fixed_base objects first, then movable objects
        print("  Reloading objects with randomized poses...")
        reload_task_objects_with_poses(
            env=env,
            task_objects=task_objects,
            scene_json=scene_json,
            new_poses=new_poses,
            settle_steps=settle_steps,
            fixed_base_names=fixed_base_names,
        )

        # Validate OnTop + Touching relationships
        new_rels = extract_relationships(task_objects)
        new_rel_set = relationships_to_set(new_rels)
        is_valid = (new_rel_set == original_rel_set)

        print(f"  Relationships match: {is_valid}  (orig={len(original_rel_set)}, new={len(new_rel_set)})")
        if not is_valid:
            missing = original_rel_set - new_rel_set
            extra = new_rel_set - original_rel_set
            if missing:
                print(f"    Missing: {list(missing)}")
            if extra:
                print(f"    Extra:   {list(extra)}")
            print("  Skipping (invalid)")
            continue

        # Valid scene — save it
        for obj in task_objects.values():
            obj.keep_still()
        og.sim.step_physics()

        scene_idx = success_count
        img_filename = f"scene_{scene_idx:03d}.png"
        take_scene_picture(str(auto_dir / img_filename), camera_pos=camera_pos, camera_ori=camera_ori)

        scene_json_filename = f"scene_{scene_idx:03d}.json"
        scene_json_path = auto_dir / scene_json_filename
        save_scene_json(env, scene_json_path)
        print(f"  Saved scene: {scene_json_path}")

        results.append({
            "iteration": scene_idx,
            "image": img_filename,
            "valid": True,
            "scene_json": scene_json_filename,
        })
        success_count += 1

    # Write generation_summary.json (same format as 13b, consumed by step 14)
    summary = {
        "valid_count": success_count,
        "total_attempts": attempt + 1 if max_attempts > 0 else 0,
        "scenes": [
            {"iteration": r["iteration"], "valid": r["valid"], "scene_json": r["scene_json"]}
            for r in results
        ],
    }
    summary_path = str(auto_dir / "generation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Sampling complete: {success_count}/{num_samples} valid scenes collected ({attempt + 1} attempts)")
    print(f"Summary: {summary_path}")
    print(f"{'='*60}")
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    ss_cfg = cfg.scene_sampling
    input_scene_json_path = ss_cfg.input_scene_json
    n_iterations = int(ss_cfg.get("n_iterations", 10))
    xy_range = float(ss_cfg.get("xy_range", 0.01))
    z_rotation_deg = float(ss_cfg.get("z_rotation_deg", 50))
    settle_steps = int(ss_cfg.get("settle_steps", 100))
    cam_pos = list(ss_cfg.get("camera_pos", [-0.0466, -0.7612, 0.5355]))
    cam_ori = list(ss_cfg.get("camera_ori", [0.5219, -0.0213, -0.0347, 0.8521]))

    # Resolve input path (allow paths relative to repo root)
    if not os.path.isabs(input_scene_json_path):
        input_scene_json_path = os.path.join(REPO_DIR, input_scene_json_path)
    if not os.path.exists(input_scene_json_path):
        raise FileNotFoundError(f"Input scene JSON not found: {input_scene_json_path}")

    # Output always goes to <scene_asset_dir>/sampling_scene/ so step 14 can find it
    # via scene_json_name=<name> + scene_source=load_sampling
    out_dir = os.path.join(os.path.dirname(input_scene_json_path), "sampling_scene")

    print(f"Input scene JSON: {input_scene_json_path}")
    print(f"Output directory: {out_dir}")
    print(f"Num samples: {n_iterations}, max attempts: {2 * n_iterations}, xy_range: {xy_range}m, z_rotation: {z_rotation_deg}deg")

    # Load scene JSON — converts relative USD paths to absolute so og.sim.save() bakes them in
    scene_json = load_json_with_absolute_usd_paths(input_scene_json_path)

    task_object_names = get_task_object_names(scene_json)
    fixed_base_names = get_fixed_base_names(scene_json)
    if not task_object_names:
        raise RuntimeError("No task objects found in scene JSON init_info.")
    print(f"\nTask objects ({len(task_object_names)}): {task_object_names}")
    if fixed_base_names:
        print(f"Fixed base objects (auto-detected): {sorted(fixed_base_names)}")

    # Create environment from the scene JSON (same pattern as step 14)
    apply_teleop_omnigibson_macros(enable_tr=False)
    # Must be set AFTER apply_teleop_omnigibson_macros since OMNIGIBSON_MACROS sets it True
    gm.ENABLE_FLATCACHE = False
    env = og.Environment(configs={
        "env": {
            "action_frequency": 30,
            "rendering_frequency": 30,
            "physics_frequency": 120,
        },
        "scene": {
            "type": "Scene",
            "scene_file": scene_json,
            "use_floor_plane": True,
            "floor_plane_visible": False,
            "use_skybox": False,
            "include_robots": True,
        },
    })
    env.reset()

    # Restore settled positions from scene JSON (env.reset() uses USD defaults)
    restore_poses_from_json(env, scene_json)

    # task_objects dict is live reference, updated by reload_task_objects_with_poses each iteration
    task_objects = {name: env.scene.object_registry("name", name) for name in task_object_names}
    task_objects = {k: v for k, v in task_objects.items() if v is not None}

    og.sim.step()
    env.scene.update_initial_file()

    

    # Capture settled poses (base for all randomizations, same as 13b's settled_poses)
    settled_poses = capture_settled_poses(task_objects)

    # Detect OnTop + Touching relationships and groups from the settled original scene
    print("\nDetecting OnTop + Touching relationships (original scene)...")
    original_rels = extract_relationships(task_objects)
    original_rel_set = relationships_to_set(original_rels)
    groups = build_groups_from_relationships(original_rels, list(task_objects.keys()))
    print(f"Found {len(groups)} group(s), {len(original_rels)} relationships")
    for i, g in enumerate(groups):
        print(f"  Group {i}: {sorted(g)}")

    # Run sampling
    run_sampling(
        env=env,
        scene_json=scene_json,
        task_objects=task_objects,
        groups=groups,
        settled_poses=settled_poses,
        original_rel_set=original_rel_set,
        out_dir=out_dir,
        num_samples=n_iterations,
        xy_range=xy_range,
        z_rotation_deg=z_rotation_deg,
        settle_steps=settle_steps,
        camera_pos=th.tensor(cam_pos),
        camera_ori=th.tensor(cam_ori),
        fixed_base_names=fixed_base_names,
    )

    og.shutdown()


if __name__ == "__main__":
    main()
