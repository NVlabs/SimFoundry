# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from simfoundry

Requires installing:

- simfoundry, see the main README
"""
import numpy as np
import os
from pathlib import Path
import json
import time
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
from simfoundry.utils.asset_conversion_utils import import_custom_object
import hydra
from simfoundry.pipeline.stage_utils import StageResult, bootstrap_hydra_workdir, finalize_stage
from simfoundry.pipeline.frame_selection import resolve_img_idx
import logging
import os

# see https://github.com/facebookresearch/hydra/issues/2949#issue-2516892001
if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

from simfoundry import CFG_DIR

logger = logging.getLogger(__name__)
bootstrap_hydra_workdir(__file__)

def compute_delta_ori(curr_orientation, init_orientation):
    ori_err_normalized = np.linalg.norm(
        R.from_matrix(R.from_quat(init_orientation).as_matrix().T @ R.from_quat(curr_orientation).as_matrix()).as_rotvec()
    ).item() / (np.pi * 2)
    ori_err = np.abs(np.pi * 2 * (np.round(ori_err_normalized) - ori_err_normalized))
    return ori_err


def stabilize_object(obj_id, pos_delta_threshold=1e-4, quat_delta_threshold=1e-3):
    # Step sim until objects settle
    n_steps = 0
    last_pos = np.zeros(3)
    last_quat = np.array([0, 0, 0, 1.0])
    last_pos_delta = 1000.0
    last_quat_delta = 1000.0
    while np.abs(last_pos_delta) > pos_delta_threshold or np.abs(last_quat_delta) > quat_delta_threshold:
        # Step sim
        p.stepSimulation()
        # Keep still object
        p.resetBaseVelocity(objectUniqueId=obj_id, linearVelocity=np.zeros(3), angularVelocity=np.zeros(3))
        # Calculate current pose and update values
        pos, quat = p.getBasePositionAndOrientation(obj_id)
        last_pos_delta = np.linalg.norm(np.array(pos) - last_pos)
        last_quat_delta = compute_delta_ori(np.array(quat), last_quat)
        last_pos = pos
        last_quat = quat
        n_steps += 1

    return n_steps


def stabilize_all_objects(obj_ids, pos_delta_threshold=1e-4, quat_delta_threshold=1e-3, max_steps=10000):
    """
    Stabilize all objects simultaneously until they all settle.
    
    Args:
        obj_ids: List of PyBullet object IDs to stabilize
        pos_delta_threshold: Position change threshold for convergence
        quat_delta_threshold: Orientation change threshold for convergence
        max_steps: Maximum simulation steps before giving up
        
    Returns:
        n_steps: Number of steps taken to stabilize all objects
    """
    n_steps = 0
    n_objs = len(obj_ids)
    
    # Track last pose for each object
    last_poses = {obj_id: (np.zeros(3), np.array([0, 0, 0, 1.0])) for obj_id in obj_ids}
    last_deltas = {obj_id: (1000.0, 1000.0) for obj_id in obj_ids}  # (pos_delta, quat_delta)
    
    while n_steps < max_steps:
        # Step simulation
        p.stepSimulation()
        
        # Dampen velocities for all objects
        for obj_id in obj_ids:
            p.resetBaseVelocity(objectUniqueId=obj_id, linearVelocity=np.zeros(3), angularVelocity=np.zeros(3))
        
        # Check convergence for all objects
        all_converged = True
        for obj_id in obj_ids:
            pos, quat = p.getBasePositionAndOrientation(obj_id)
            pos = np.array(pos)
            quat = np.array(quat)
            
            last_pos, last_quat = last_poses[obj_id]
            pos_delta = np.linalg.norm(pos - last_pos)
            quat_delta = compute_delta_ori(quat, last_quat)
            
            last_poses[obj_id] = (pos, quat)
            last_deltas[obj_id] = (pos_delta, quat_delta)
            
            if np.abs(pos_delta) > pos_delta_threshold or np.abs(quat_delta) > quat_delta_threshold:
                all_converged = False
        
        n_steps += 1
        
        if all_converged:
            break
    
    return n_steps


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    ground_img_idx = resolve_img_idx(cfg)
    frame_dir = cfg.s4_frame.out_dir
    if cfg.s9_compile.use_interactive_pose:
        pose_dir = cfg.s8_pose.out_dir + f"/info_interactive{cfg.s9_compile.interactive_suffix}"
    else:
        pose_dir = cfg.s8_pose.out_dir + "/info"
    sim_dir = cfg.s10_sim.out_dir
    out_dir = cfg.s11_physics.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("Starting physics stabilization pipeline...")
    logger.info(f"Output directory: {out_dir}")
    logger.info("="*60)

    # Load scene objects info
    scene_objects_info_fpath = f"{sim_dir}/scene_objects_info.json"
    with open(scene_objects_info_fpath, "r") as f:
        scene_objects_info = json.load(f)

    # Load cam2world tf
    cam2world_tf = np.load(f"{frame_dir}/image_{ground_img_idx}_cam2world.npy")

    # Connect to pybullet (use DIRECT mode if not visualizing to avoid X server requirement)
    if cfg.visualize:
        physics_client = p.connect(p.GUI)
    else:
        physics_client = p.connect(p.DIRECT)
        logger.info("Running in headless mode (no X server required)")

    # Configure the simulation
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)  # Disable real-time simulation for better control

    # Load ground plane
    plane_id = p.loadURDF("plane.urdf")

    # Check loading mode
    load_at_once = cfg.s11_physics.get("load_at_once", False)
    z_offset = 0.05
    objs_pb_info = dict()

    if load_at_once:
        # ==================== Mode 1: Load All Objects At Once ====================
        # Load all objects first, then stabilize them together
        # This allows objects to interact with each other during settling
        obj_ids = []
        
        logger.info("Mode: load_at_once=True - Loading all objects simultaneously...")
        for idx, obj_info in reversed(scene_objects_info.items()):
            obj_category = obj_info["category"]
            obj_model = obj_info["model"]
            obj_name = obj_info["name"]

            obj_urdf_fpath = f"{sim_dir}/objects/{obj_category}/{obj_model}/urdf/{obj_model}.urdf"
            with open(f"{pose_dir}/{obj_name}.json", "r") as f:
                obj_tf_info = json.load(f)["y_up"]

            obj2scene_tf = np.eye(4)
            obj2scene_tf[:3, 3] = np.array(obj_tf_info["trans"])
            obj2scene_tf[:3, :3] = np.array(obj_tf_info["rot"])
            obj2scene_tf = cam2world_tf @ obj2scene_tf

            logger.info(f"Loading object {obj_category} / {obj_model} / {obj_name} from {obj_urdf_fpath}...")
            obj_id = p.loadURDF(
                obj_urdf_fpath,
                useFixedBase=False,
            )

            # Get mass / friction info
            obj_dynamics_info = p.getDynamicsInfo(
                bodyUniqueId=obj_id,
                linkIndex=-1,   # root link
            )
            mass = obj_dynamics_info[0]
            friction = obj_dynamics_info[1]
            inertia = obj_dynamics_info[2]
            inertia_pos = obj_dynamics_info[3]
            inertia_quat = obj_dynamics_info[4]
            logger.info(f"Obj {obj_name} [({obj_category}) / ({obj_model})]:")
            logger.info(f"\tmass: {mass}")
            logger.info(f"\tfriction: {friction}")
            logger.info(f"\tinertia: {inertia}")
            p.changeDynamics(
                bodyUniqueId=obj_id,
                linkIndex=-1,   # root link
                contactProcessingThreshold=0.0,
                collisionMargin=0.0,
                lateralFriction=obj_info["friction"],
            )

            # Update initial pose based on inertia pos / quat
            inertia2obj_tf = np.eye(4)
            inertia2obj_tf[:3, :3] = R.from_quat(inertia_quat).as_matrix()
            inertia2obj_tf[:3, 3] = inertia_pos
            obj_tf = obj2scene_tf @ inertia2obj_tf
            obj_pos = obj_tf[:3, 3]
            obj_quat = R.from_matrix(obj_tf[:3, :3]).as_quat()

            # Query native AABB
            obj_aabb = p.getAABB(obj_id)
            obj_aabb_center_offset = (np.array(obj_aabb[1]) + np.array(obj_aabb[0])) / 2

            # Update base pose based on cached obj info
            p.resetBasePositionAndOrientation(
                obj_id,
                obj_pos + np.array([0, 0, z_offset]),
                obj_quat,
            )

            objs_pb_info[obj_name] = {
                "id": obj_id,
                "urdf_fpath": obj_urdf_fpath,
                "friction": obj_info["friction"],
                "init_pos": obj_pos,
                "init_quat": obj_quat,
                "aabb_center_offset": obj_aabb_center_offset,
                "aabb_center_init": R.from_quat(obj_quat).as_matrix() @ obj_aabb_center_offset + obj_pos,
                "inertia2obj_tf": inertia2obj_tf,
            }
            obj_ids.append(obj_id)
        
        logger.info(f"Loaded {len(obj_ids)} objects. Now stabilizing all objects together...")
        
        # Stabilize all objects simultaneously - this allows objects to interact with each other
        n_steps = stabilize_all_objects(obj_ids=obj_ids)
        logger.info(f"Stabilized all {len(obj_ids)} objects after {n_steps} steps!")

    else:
        # ==================== Mode 2: Load Objects One-by-One (Original) ====================
        # Load objects sequentially, stabilize each one, then fix in place with constraints
        # Objects are placed in reverse order (top-to-bottom from scene decomposition)
        fixed_constraint_ids = []
        
        logger.info("Mode: load_at_once=False - Loading objects one-by-one with constraints...")
        for idx, obj_info in reversed(scene_objects_info.items()):
            obj_category = obj_info["category"]
            obj_model = obj_info["model"]
            obj_name = obj_info["name"]

            obj_urdf_fpath = f"{sim_dir}/objects/{obj_category}/{obj_model}/urdf/{obj_model}.urdf"
            with open(f"{pose_dir}/{obj_name}.json", "r") as f:
                obj_tf_info = json.load(f)["y_up"]

            obj2scene_tf = np.eye(4)
            obj2scene_tf[:3, 3] = np.array(obj_tf_info["trans"])
            obj2scene_tf[:3, :3] = np.array(obj_tf_info["rot"])
            obj2scene_tf = cam2world_tf @ obj2scene_tf

            logger.info(f"Loading object {obj_category} / {obj_model} / {obj_name} from {obj_urdf_fpath}...")
            obj_id = p.loadURDF(
                obj_urdf_fpath,
                useFixedBase=False,
            )

            # Get mass / friction info
            obj_dynamics_info = p.getDynamicsInfo(
                bodyUniqueId=obj_id,
                linkIndex=-1,   # root link
            )
            mass = obj_dynamics_info[0] * 10
            friction = obj_dynamics_info[1]
            inertia = obj_dynamics_info[2]
            inertia_pos = obj_dynamics_info[3]
            inertia_quat = obj_dynamics_info[4]
            logger.info(f"Obj {obj_name} [({obj_category}) / ({obj_model})]:")
            logger.info(f"\tmass: {mass}")
            logger.info(f"\tfriction: {friction}")
            logger.info(f"\tinertia: {inertia}")
            p.changeDynamics(
                bodyUniqueId=obj_id,
                linkIndex=-1,   # root link
                contactProcessingThreshold=0.0,
                collisionMargin=0.0,
                lateralFriction=obj_info["friction"],
            )

            # Update initial pose based on inertia pos / quat
            inertia2obj_tf = np.eye(4)
            inertia2obj_tf[:3, :3] = R.from_quat(inertia_quat).as_matrix()
            inertia2obj_tf[:3, 3] = inertia_pos
            obj_tf = obj2scene_tf @ inertia2obj_tf
            obj_pos = obj_tf[:3, 3]
            obj_quat = R.from_matrix(obj_tf[:3, :3]).as_quat()

            # Query native AABB
            obj_aabb = p.getAABB(obj_id)
            obj_aabb_center_offset = (np.array(obj_aabb[1]) + np.array(obj_aabb[0])) / 2

            # Update base pose based on cached obj info
            p.resetBasePositionAndOrientation(
                obj_id,
                obj_pos + np.array([0, 0, z_offset]),
                obj_quat,
            )

            objs_pb_info[obj_name] = {
                "id": obj_id,
                "urdf_fpath": obj_urdf_fpath,
                "friction": obj_info["friction"],
                "init_pos": obj_pos,
                "init_quat": obj_quat,
                "aabb_center_offset": obj_aabb_center_offset,
                "aabb_center_init": R.from_quat(obj_quat).as_matrix() @ obj_aabb_center_offset + obj_pos,
                "inertia2obj_tf": inertia2obj_tf,
            }

            # Step slowly until the object stabilizes
            n_steps = stabilize_object(obj_id=obj_id)
            logger.info(f"Stabilized {obj_name} after {n_steps} steps!")

            # Grab updated object pos / quat
            pos, quat = p.getBasePositionAndOrientation(obj_id)

            # Fix this object to the scene with a fixed joint constraint
            constraint_id = p.createConstraint(
                parentBodyUniqueId=obj_id,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,  # World
                childLinkIndex=-1,  # Not applicable for world
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=np.zeros(3),
                parentFrameOrientation=np.array([0, 0, 0, 1.0]),
                childFramePosition=pos,
                childFrameOrientation=quat,
            )
            fixed_constraint_ids.append(constraint_id)

        # Remove all the fixed joints to allow final settling
        for constraint_id in fixed_constraint_ids:
            p.removeConstraint(constraint_id)

    # Keep all objects still
    p.stepSimulation()
    for _ in range(50):
        for obj_info in objs_pb_info.values():
            p.resetBaseVelocity(objectUniqueId=obj_info["id"], linearVelocity=np.zeros(3), angularVelocity=np.zeros(3))
        p.stepSimulation()

    # Save this information -- all positions relative to the ground plane in the world frame!
    obj_state = dict()
    for obj_name, obj_info in objs_pb_info.items():
        pos, quat = p.getBasePositionAndOrientation(obj_info["id"])
        inertia2scene_tf = np.eye(4)
        inertia2scene_tf[:3, :3] = R.from_quat(quat).as_matrix()
        inertia2scene_tf[:3, 3] = pos
        obj2scene_tf = inertia2scene_tf @ np.linalg.inv(obj_info["inertia2obj_tf"])
        obj2scene_pos = obj2scene_tf[:3, 3]
        obj2scene_quat = R.from_matrix(obj2scene_tf[:3, :3]).as_quat()
        obj_state[obj_name] = [obj2scene_pos.tolist(), obj2scene_quat.tolist()]
        # obj_state[obj_name] = p.getBasePositionAndOrientation(obj_info["id"])
    with open(f"{out_dir}/pb_scene_poses.json", "w") as f:
        json.dump(obj_state, f, indent=4)

    # Visualization code only runs if cfg.visualize is True
    if cfg.visualize:
        # Toggle for collision visualization
        collision_toggle = p.addUserDebugParameter("Show Collision", 0, 1, 0)
        print("\nControls:")
        print("- Use the sliders to control joint positions")
        print("- Mouse: Rotate view (left click + drag), Zoom (scroll), Pan (right click + drag)")
        print("- Press Ctrl+C to exit")

        # Main simulation loop
        prev_collision_state = 0

        while True:
            # Handle collision visualization toggle
            collision_state = p.readUserDebugParameter(collision_toggle)
            if collision_state != prev_collision_state:
                if collision_state > 0.5:
                    # Show collision shapes
                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
                    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)
                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
                else:
                    # Hide collision shapes
                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
                    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
                prev_collision_state = collision_state

            # p.setTimeStep(0.0)
            p.stepSimulation()

            time.sleep(1. / 240.)  # 240 Hz
    else:
        # Disconnect from PyBullet when not visualizing
        p.disconnect()

    logger.info("="*60)
    logger.info("Physics stabilization complete!")
    logger.info("="*60)
    
    finalize_stage(
        stage_cfg=cfg.s11_physics,
        out_dir=cfg.s11_physics.out_dir,
        result=StageResult(success=True),
    )


if __name__ == "__main__":
    main()
