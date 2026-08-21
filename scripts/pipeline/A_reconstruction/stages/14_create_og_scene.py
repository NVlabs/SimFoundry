# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from b1k

Requires installing:

- BEHAVIOR-1K, see https://github.com/StanfordVL/BEHAVIOR-1K
"""
import os

# Keep pipeline invocations non-interactive and avoid X/viewport dependencies by
# default. Interactive users can override this before launching the script.
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

from simfoundry import CFG_DIR as SIMFOUNDRY_CFG_DIR, configure_omnigibson_data_path

configure_omnigibson_data_path(force=True)

import omnigibson as og
from omnigibson.objects import DatasetObject
import omnigibson.utils.transform_utils as T
from omnigibson.utils.asset_utils import get_dataset_path
from omnigibson.utils.config_utils import parse_config
from pathlib import Path
import torch as th
import json
import hydra
from omegaconf import OmegaConf
from simfoundry.pipeline.stage_utils import StageResult, bootstrap_hydra_workdir, finalize_stage
from simfoundry.pipeline.frame_selection import resolve_img_idx
from simfoundry.utils.og_utils import set_obj_materials
import numpy as np
import sys

from simfoundry import CFG_DIR


bootstrap_hydra_workdir(__file__)


def take_scene_picture(save_path, camera_pos=None, camera_ori=None, focal_length=12.0):
    """
    Set viewer camera pose, render, and save the current view as a PNG.
    Same logic as in B_augmentation/stages/6b_sample_asset_scene.py for consistency.
    Uses a lower focal length (wider FOV) so all tabletop objects fit in frame.

    Args:
        save_path: Full path for the output image (e.g. .../reconstructed_scene.png).
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
    if cam is None:
        print(f"Skipping scene image capture because no viewer camera is available: {save_path}")
        return False

    orig_focal = cam.focal_length
    try:
        cam.focal_length = focal_length
        cam.set_position_orientation(position=camera_pos, orientation=camera_ori)
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
        print(f"Saved scene image: {save_path}")
        return True
    finally:
        cam.focal_length = orig_focal


def run_interactive_viewer(env):
    """Keep the simulator alive for manual inspection when explicitly requested."""
    print("Simulation running. Press Ctrl+C to exit...")
    try:
        while True:
            action = th.zeros(7)
            env.step(action)
            print(og.sim.viewer_camera.get_position_orientation())
    except KeyboardInterrupt:
        print("\nShutting down simulation...")


def exit_after_success(cfg):
    """Bypass brittle OmniGibson/Python atexit teardown in batch pipeline runs."""
    if not cfg.s14_og.get("fast_process_exit", True):
        og.shutdown()
        return

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def validate_scene_assets(scene_objects_info, dataset_name):
    dataset_path = get_dataset_path(dataset_name)
    missing = []
    for obj_info in scene_objects_info.values():
        category = obj_info["category"]
        model = obj_info["model"]
        usd_path = os.path.join(dataset_path, "objects", category, model, "usd", f"{model}.usd")
        if not os.path.exists(usd_path):
            missing.append(usd_path)
    if missing:
        missing_lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Stage 14 cannot load {len(missing)} USD asset(s):\n{missing_lines}")


def validate_robot_assets(robot_cfg):
    if robot_cfg.get("type") != "FrankaPanda":
        return
    end_effector = robot_cfg.get("end_effector", "gripper")
    model_by_end_effector = {
        "gripper": "franka_panda",
        "robotiq": "franka_robotiq",
    }
    model_name = model_by_end_effector.get(end_effector)
    if model_name is None:
        return
    robot_assets_path = get_dataset_path("omnigibson-robot-assets")
    usd_path = os.path.join(robot_assets_path, "models", "franka", model_name, "usd", f"{model_name}.usda")
    if os.path.exists(usd_path):
        return
    raise FileNotFoundError(
        "Stage 14 cannot load the configured Franka robot asset:\n"
        f"  - end_effector: {end_effector}\n"
        f"  - expected USD: {usd_path}\n"
        "The default 'gripper' end effector is included in the public OmniGibson robot assets. "
        "Robotiq configs additionally require the SimFoundry robot asset bundle, which "
        "scripts/installation/install_simfoundry.sh fetches automatically. Re-run that installer, "
        "or pass --robot-asset-fallback-root <repo-with-assets> if you have a local copy, "
        "or override s14_og.robot_config.end_effector=gripper."
    )

@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    sim_dir = cfg.s11_sim.out_dir
    physics_dir = cfg.s12_physics.out_dir
    out_dir = cfg.s14_og.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)


    # Load physics objects info
    include_gs = cfg.s14_og.include_gs


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

    dataset_name = str(cfg.s14_og.get("dataset_name", "real2sim-assets"))
    validate_scene_assets(scene_objects_info, dataset_name)

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
        validate_robot_assets(robot_cfg)
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

    # Create the environment. Skip obs collection in reset: external sensors may be
    # non-functional due to an Isaac Sim 5.1 zero-dtype bug in SyntheticData — the
    # sensor prims still exist in the USD scene for downstream stages.
    env = og.Environment(configs=og_cfg)
    env.reset(get_obs=False)
    

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
                dataset_name=dataset_name,
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
    settle_steps = cfg.s14_og.get("settle_steps", 100)
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
            ori.tolist() if isinstance(ori, th.Tensor) else list(ori)
        ]
        print(f"  {obj_name}: pos={pos.numpy() if isinstance(pos, th.Tensor) else pos}, ori={ori.numpy() if isinstance(ori, th.Tensor) else ori}")

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

    # Include 3DGS background if specified
    if include_gs:
        # Preferred: USDZ produced by the auto_bg_reconstruction pipeline (script 7).
        # Fallback: nerfstudio-trained splat from stage 2c (splat.ply) — stored as a
        # reference path only; full OG integration still requires the USDZ conversion.
        cam2world_tf_fpath = f"{cfg.s4_frame.out_dir}/image_{resolve_img_idx(cfg)}_cam2world.npy"
        cam2world_tf = th.from_numpy(np.load(cam2world_tf_fpath)).float()
        gs_path_da3 = os.path.abspath(f"{cfg.s2_da.out_dir}/gs_outputs/gaussian_da3.usdz")
        gs_path_2c_ply = os.path.abspath(f"{cfg.s2c_gs.out_dir}/export/splat.ply")

        from omnigibson.objects import USDObject

        if not os.path.exists(gs_path_da3):
            print(
                f"[WARN] include_gs=True but GS USDZ not found at {gs_path_da3}. "
                f"Run the auto_bg_reconstruction pipeline (scripts 1-7) to produce it. "
                f"Stage 2c PLY exists: {os.path.exists(gs_path_2c_ply)}. "
                "Skipping GS background for this run."
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

   

    # joint_targets = th.tensor([])
    if cfg.s14_og.include_robot:
        # Spawn the robot
        robot = env.robots[0]

        # # Position robot relative to the table
        # robot_position = table_position.clone()
        # robot_position[0] += 0.55
        # robot_position[1] -= 0.10
        # robot_position[2] = 0.60  # Slightly off ground

        # robot_position = th.tensor([2.0, 2.0, -0.861])
        og.sim.play()
        robot.reset()
        robot.keep_still()

        # robot.set_joint_positions(robot.default_arm_poses["home"])
        # joint_targets = robot.default_arm_poses["gripper_down"]
        # joint_targets = robot.default_arm_poses["home"]
        # joint_targets = th.cat([robot.reset_joint_pos[:-2], th.tensor([1.0])])
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
    if cfg.s14_og.include_table:
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

    img_path = f"{out_dir}/reconstructed_scene.png"
    image_captured = False
    if cfg.s14_og.get("capture_image", True):
        image_captured = take_scene_picture(
            img_path,
            camera_pos=cfg.s14_og.auto_camera_pos,
            camera_ori=cfg.s14_og.auto_camera_ori,
        )
    
    finalize_stage(
        stage_cfg=cfg.s14_og,
        out_dir=cfg.s14_og.out_dir,
        result=StageResult(
            success=True,
            additional_info={
                "scene_json": json_path,
                "preview_image": img_path if image_captured else None,
                "num_objects": len(objs),
                "interactive": bool(cfg.s14_og.get("interactive", False)),
            },
        ),
    )

    if cfg.s14_og.get("interactive", False):
        run_interactive_viewer(env)
        og.shutdown()
    else:
        exit_after_success(cfg)


if __name__ == "__main__":
    main()
