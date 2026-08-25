# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate a Gr00t Nx pretrained checkpoint on a simulated OmniGibson environment.

Should be run from simfoundry

NOTES FROM JOSIAH on tuning sim:

- Tuning render settings post-processing color correction params
- Replaying robot actions from real dataset and making sure the overall trajectories align with what the sim setup can do (esp. in z-axis)
- Replace 3DGS background with new mesh background
- Change assisted -> physical grasping
- Add friction to all objects in the scene
- Tune poses of all scene objects, robot pose
- Tune scales of all scene objects to match the real world
- Lower robot finger max velocity (TBD: Somehow this is being set external from somewhere? Need to manually override each time)

"""
import omnigibson as og
from omnigibson import shutdown_handler
from omnigibson.macros import gm
import omnigibson.lazy as lazy
from omnigibson.scenes import Scene
from omnigibson.objects import DatasetObject
from omnigibson.robots import REGISTERED_ROBOTS, BaseRobot, LocomotionRobot, MobileManipulationRobot
from omnigibson.controllers import InverseKinematicsController, OperationalSpaceController
import omnigibson.utils.transform_utils as T
from omnigibson.utils.config_utils import parse_config
from omnigibson.sensors import VisionSensor
from omnigibson.envs import HDF5CollectionWrapper
from pathlib import Path
import torch as th
import json
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from simfoundry import import_og_dependencies, CFG_DIR as SIMFOUNDRY_CFG_DIR, ASSET_DIR as SIMFOUNDRY_ASSET_DIR
from simfoundry.utils.processing_utils import dump_json, resize_with_pad
from simfoundry.utils.python_utils import resolve_task_config_path
from simfoundry.utils.og_utils import (
    apply_teleop_omnigibson_macros,
    set_obj_materials,
    setup_task_status_ui,
    update_task_status,
    update_reward_ui,
)
from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths
# object_swap_utils is not part of this tree, and object_swap_json defaults to
# null in every config -- so importing it here failed the stage before it had
# read a single argument. Imported at the two places that actually swap instead.
from simfoundry.policies.gr00t import Gr00tClient
from simfoundry.policies.openpi import OpenPIClient
from simfoundry.policies.dreamzero import DreamZeroClient
import os
from datetime import datetime
import signal
import numpy as np
from scipy.spatial.transform import Rotation
import dataclasses
from typing_extensions import override
import pathlib
import imageio
from PIL import Image


gm.ENABLE_CCD = True

# Needed so custom tasks can be instantiated properly
import_og_dependencies()


# DROID Franka EEF convention: rotate panda_link8 frame so the canonical
# DROID/Pinocchio EEF frame (used by N1.7 oxe_droid_relative_eef policies)
# is recovered. Matches simfoundry.policies.gr00t.DROID_EEF_ROTATION_CORRECT.
_DROID_EEF_ROTATION_CORRECT = np.array(
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    dtype=np.float64,
)
# panda_link7 -> panda_link8 (EEF flange) is a +107 mm translation along link7's local Z.
_PANDA_LINK7_TO_LINK8_Z = 0.107


def compute_droid_eef_9d(robot):
    """Compute DROID-convention eef_9d (xyz + rot6d) from a Franka OG robot.
    Needed for N1.7.
    TODO: refactor into gr00t.py

    Returns None if the robot does not expose `panda_link7` (e.g. non-Franka).
    """
    if "panda_link7" not in robot.links:
        return None
    pos7, quat7 = robot.links["panda_link7"].get_position_orientation()
    pos7_np = pos7.cpu().numpy().astype(np.float32)
    quat7_np = quat7.cpu().numpy().astype(np.float32)  # xyzw
    rot7 = Rotation.from_quat(quat7_np)
    eef_pos = (pos7_np + rot7.apply(np.array([0.0, 0.0, _PANDA_LINK7_TO_LINK8_Z]))).astype(np.float32)
    rot_mat = rot7.as_matrix() @ _DROID_EEF_ROTATION_CORRECT
    rot6d = rot_mat[:2, :].flatten().astype(np.float32)
    return np.concatenate([eef_pos, rot6d])


class PolicyRolloutHDF5CollectionWrapper(HDF5CollectionWrapper):
    """
    Extended HDF5CollectionWrapper for policy rollouts.
    """
    def _parse_step_data(self, action, obs, reward, terminated, truncated, info):
        step_data = super()._parse_step_data(action, obs, reward, terminated, truncated, info)
        # Add milestones from obs dict if present (from task._get_obs)
        if "milestones_satisfied" in obs:
            step_data["milestones"] = obs["milestones_satisfied"]
        elif info and "milestones" in info and info["milestones"] is not None:
            step_data["milestones"] = th.tensor(list(info["milestones"].values()), dtype=th.float32)
        else:
            step_data["milestones"] = th.tensor([0.0], dtype=th.float32)
        return step_data


CFG_DIR = SIMFOUNDRY_CFG_DIR

### At the start of every script, we cd into the scripts/config directory
scripts_dir = os.path.dirname(os.path.abspath(__file__))
cfg_dir = os.path.join(scripts_dir, "..", "cfg")
# os.chdir(cfg_dir)


def load_policy(cfg):
    if cfg.s15_eval.policy == "gr00t":
        return Gr00tClient(host=cfg.s15_eval.host, port=cfg.s15_eval.port, api_token=None, open_loop_horizon=cfg.s15_eval.execute_horizon)
    elif cfg.s15_eval.policy == "openpi":
        return OpenPIClient(host=cfg.s15_eval.host, port=cfg.s15_eval.port, open_loop_horizon=cfg.s15_eval.execute_horizon)
    elif cfg.s15_eval.policy == "dreamzero":
        return DreamZeroClient(host=cfg.s15_eval.host, port=cfg.s15_eval.port, open_loop_horizon=cfg.s15_eval.execute_horizon)
    else:
        raise ValueError(f"Invalid policy: {cfg.s15_eval.policy}")

@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    sim_dir = cfg.s10_sim.out_dir
    og_dir = cfg.s13_og.out_dir
    out_dir = cfg.s15_eval.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load policy
    policy = load_policy(cfg)
    

    use_absolute_actions = True

    # Resolve the scene JSON path. ``s15_eval.scene_json`` may be:
    #   - ``null``  -> use the s13_og output ``reconstructed_og_scene.json``.
    #   - a bare scene name (no separators, no .json) -> the canonical layout
    #     ``<SIMFOUNDRY_ASSET_DIR>/scenes/<name>/<name>_scene_state_latest.json``.
    #   - a path ending in ``.json`` (abs path, or relative to ``SIMFOUNDRY_ASSET_DIR`` /
    #     cwd) -> used verbatim. Useful for tuned cousin variants saved as
    #     ``<scene>/<scene>_scene_state_<suffix>_latest.json``.
    scene_json_name = cfg.s15_eval.scene_json
    if scene_json_name is None:
        og_scene_json_path = os.path.join(og_dir, "reconstructed_og_scene.json")
    elif str(scene_json_name).endswith(".json") or os.sep in str(scene_json_name):
        candidate = str(scene_json_name)
        if os.path.isabs(candidate) and os.path.exists(candidate):
            og_scene_json_path = candidate
        elif os.path.exists(os.path.join(SIMFOUNDRY_ASSET_DIR, candidate)):
            og_scene_json_path = os.path.join(SIMFOUNDRY_ASSET_DIR, candidate)
        else:
            og_scene_json_path = candidate  # let load fail with a clear message
    else:
        og_scene_json_path = f"{SIMFOUNDRY_ASSET_DIR}/scenes/{scene_json_name}/{scene_json_name}_scene_state_latest.json"

    print(f"[s15_eval] Loading scene JSON from: {og_scene_json_path}")
    og_scene_json = load_json_with_absolute_usd_paths(og_scene_json_path)

    # Optionally swap scene objects with alternative USDs (e.g., cousin assets).
    # See assets/cousins/<Task>/swaps/cousin_combo_*.json for examples. Paths
    # may be absolute or relative to SIMFOUNDRY_ASSET_DIR.
    swap_info = {}
    swap_stem = None
    swap_json_path = cfg.s15_eval.get("object_swap_json", None)
    if swap_json_path is not None:
        if not os.path.isabs(swap_json_path):
            swap_json_path = os.path.join(SIMFOUNDRY_ASSET_DIR, swap_json_path)
        if os.path.exists(swap_json_path):
            from simfoundry.utils.object_swap_utils import apply_object_swaps
            swap_info = apply_object_swaps(og_scene_json, swap_json_path)
            swap_stem = Path(swap_json_path).stem
            print(f"[ObjectSwap] Applied {len(swap_info)} swap(s) from {swap_json_path}")
        else:
            print(f"[ObjectSwap] Warning: swap JSON not found at {swap_json_path}")


    # Modify robot config to use physical grasping instead of assisted grasping
    grasping_mode_override = cfg.s15_eval.get("grasping_mode", None)
    if grasping_mode_override is not None:
        if "objects_info" in og_scene_json and "init_info" in og_scene_json["objects_info"]:
            for obj_name, obj_info in og_scene_json["objects_info"]["init_info"].items():
                if obj_name.startswith("robot") and "args" in obj_info:
                    if "grasping_mode" in obj_info["args"]:
                        print(f"Changing grasping mode for {obj_name} from '{obj_info['args']['grasping_mode']}' to '{grasping_mode_override}'")
                        obj_info["args"]["grasping_mode"] = grasping_mode_override



    # Include the swap stem in the modified-scene filename so concurrent/sequential
    # sweeps over multiple swap combos do not clobber each other's scene JSONs.
    swap_suffix = f"_{swap_stem}" if swap_stem else ""
    modified_scene_json_path = f"{out_dir}/modified_scene{swap_suffix}.json"
    with open(modified_scene_json_path, "w") as f:
        json.dump(og_scene_json, f, indent=2)
    print(f"Saved modified scene JSON to: {modified_scene_json_path}")

    scene_objects_info_fpath = f"{sim_dir}/scene_objects_info.json"
    obj_frictions = dict()
    if os.path.exists(scene_objects_info_fpath):
        with open(scene_objects_info_fpath, "r") as f:
            scene_objects_info = json.load(f)

        for _, obj_info in scene_objects_info.items():
            obj_frictions[obj_info["name"]] = obj_info["friction"]


    scene_cfg = {
        "type": "Scene",
        "scene_file": modified_scene_json_path,
        "use_floor_plane": cfg.s15_eval.use_floor_plane,
        "floor_plane_visible": cfg.s15_eval.use_floor_plane and cfg.s15_eval.floor_plane_visible,
        "use_skybox": True,
        "include_robots": True,
    }

    # Load task configuration
    task_name = cfg.task.task_name
    og_task_cfg_path = resolve_task_config_path(
        SIMFOUNDRY_CFG_DIR, task_name,
        group_choice=HydraConfig.get().runtime.choices.get("task"))
    task_cfg = parse_config(og_task_cfg_path)["og_task_config"]
    action_freq = cfg.s15_eval.action_freq
    n_steps = int(cfg.s15_eval.timeout_s * action_freq)
    task_cfg["termination_config"]["max_steps"] = n_steps

    # Load external sensors configuration
    external_sensors_cfg_path = f"{SIMFOUNDRY_CFG_DIR}/external_sensors/{cfg.s15_eval.external_sensors_cfg}.yaml"
    external_sensors_cfg = parse_config(external_sensors_cfg_path)["external_sensors"]
    
    # Set image resolution to 224x224 as expected by DROID/OpenPI models
    for sensor_cfg in external_sensors_cfg:
        if "sensor_kwargs" in sensor_cfg:
            sensor_cfg["sensor_kwargs"]["image_height"] = 720
            sensor_cfg["sensor_kwargs"]["image_width"] = 1280
    

    env_cfg = {
        "external_sensors": external_sensors_cfg,
        "action_frequency": action_freq,
        "rendering_frequency": action_freq,
        "physics_frequency": 120,
    }


    og_cfg = dict(env=env_cfg, scene=scene_cfg, task=task_cfg)

    # Create the environment
    env = og.Environment(configs=og_cfg)
    env.reset()

    # Get robot reference for controller setup
    assert len(env.robots) == 1, "Evaluation only supports a single robot!"
    robot = env.robots[0]

    # Update robot control mode BEFORE wrapping with HDF5CollectionWrapper
    # This ensures the scene_file captured by the wrapper has the correct controller config
    # Configuration matches sim-evals/Isaac Lab ImplicitActuatorCfg exactly:
    #   - stiffness=400.0, damping=80.0 for arm joints
    #   - Gripper uses MultiFingerGripperController for binary open/close
    controller_cfg = dict()
    for arm in robot.arm_names:
        controller_cfg[f"arm_{arm}"] = {
            "name": "JointController",
            "motor_type": "position",
            "command_input_limits": None, 
            "use_delta_commands": False,
            "smoothing_filter_size": 5,
        }
        # controller_cfg[f"arm_{arm}"] = {
        #     "name": "ImplicitPDController",
        #     "stiffness": 400.0,           # Matches Isaac Lab panda_shoulder/panda_forearm
        #     "damping": 80.0,              # Matches Isaac Lab panda_shoulder/panda_forearm
        #     "command_input_limits": None,
        #     "use_delta_commands": False,
        #     # use_explicit_pd defaults to False (implicit mode - physics does PD)
        # }
        controller_cfg[f"gripper_{arm}"] = {
            "name": "MultiFingerGripperController",
            "command_input_limits": (0.0, 1.0),
            "mode": "smooth",
            "inverted": True, # False,
        }
    robot.reload_controllers(controller_cfg)
    env.scene.update_initial_file()
    print(f"[DEBUG] Reloaded controllers to JointController before HDF5 wrapping")

    # Wrap environment with data collection wrapper to save rollouts
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create checkpoint-specific output directory. When a cousin swap JSON is
    # active, suffix the checkpoint dir with the swap stem so a sweep over
    # multiple cousin_combo_*.json files keeps each combo's results separate.
    checkpoint_name = cfg.s15_eval.checkpoint
    results_dir = f"{out_dir}/{checkpoint_name}{swap_suffix}/results_{timestamp}"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    rollout_dataset_path = f"{results_dir}/rollouts.hdf5"
    
    print(f"\n{'='*60}")
    print(f"Results will be saved to:")
    print(f"  Directory: {results_dir}/")
    print(f"  Rollouts HDF5: {rollout_dataset_path}")
    print(f"  Results JSON: {results_dir}/eval_results.json")
    print(f"  Videos: {results_dir}/videos/")
    print(f"{'='*60}\n")
    
    
    data_env_kwargs = OmegaConf.to_container(cfg.s14_teleop.data_env_kwargs, resolve=True)
    data_env_kwargs["only_successes"] = False
    data_env_kwargs["flush_every_n_traj"] = 1  
    
    env = PolicyRolloutHDF5CollectionWrapper(
        env=env,
        output_path=rollout_dataset_path,
        **data_env_kwargs,
    )

    # Re-enable camera rendering
    for sensor in VisionSensor.SENSORS.values():
        sensor.render_product.hydra_texture.set_updates_enabled(True)
    env.is_recording = False
    env.reset()
    env.is_recording = True
    print(f"Data collection enabled. Rollouts will be saved to: {rollout_dataset_path}")

    # AABB-based z adjustment for swapped objects that lacked pre-computed size info
    if swap_info and any(info["needs_aabb_adjustment"] for info in swap_info.values()):
        gp_info = og_scene_json.get("ground_plane_info", {})
        gp_z = gp_info.get("position", [0, 0, 0])[2] if gp_info else 0.0
        from simfoundry.utils.object_swap_utils import adjust_swapped_objects_z
        adjust_swapped_objects_z(env, swap_info, ground_plane_z=gp_z)
    
    

    # Set object materials
    objs = {name: env.scene.object_registry("name", name) for name in obj_frictions.keys()}
    objs = {k: v for k, v in objs.items() if v is not None}

    with og.sim.stopped():
        for obj in env.scene.objects:
            for link in obj.links.values():
                link.ccd_enabled = True

        # # Set object materials
        # obj_materials = set_obj_materials(objs=objs, obj_frictions=obj_frictions)

        # Create friction material for robot fingers
        robot = env.robots[0]
        finger_friction = 2.0
        finger_mat = lazy.isaacsim.core.api.materials.PhysicsMaterial(
            prim_path=f"{robot.prim_path}/Looks/finger_mat",
            name="finger_mat",
            static_friction=finger_friction,
            dynamic_friction=finger_friction,
            restitution=None,
        )
        # Apply friction material to gripper finger links
        for link_name, link in robot.links.items():
            if "finger" in link_name.lower() or "gripper" in link_name.lower():
                for mesh in link.collision_meshes.values():
                    mesh.apply_physics_material(finger_mat)

        # Apply default friction material of 0.5 friction to all other objects in the scene if the name is not gs_background
        default_friction = 0.5
        default_mat = lazy.isaacsim.core.api.materials.PhysicsMaterial(
            prim_path=f"/World/Looks/default_friction_mat",
            name="default_friction_mat",
            static_friction=default_friction,
            dynamic_friction=default_friction,
            restitution=None,
        )
        for obj in env.scene.objects:
            if obj.name == "gs_background":
                continue
            if isinstance(obj, BaseRobot):
                continue
            # if obj.name in objs:
            #     continue  # Already has custom friction applied
            for link in obj.links.values():
                for mesh in link.collision_meshes.values():
                    mesh.apply_physics_material(default_mat)

    # Update camera params
    for sensor_name, sensor in robot.sensors.items():
        if isinstance(sensor, VisionSensor):
            sensor.focal_length=2.8
            # sensor.focus_distance=28.0
            sensor.horizontal_aperture=5.376
            # sensor.vertical_aperture=3.024

    # Load init states for deterministic evaluation if provided
    init_states_path = cfg.s15_eval.get("init_states_path", None)
    init_states = None
    if init_states_path is not None:
        init_states_file = Path(init_states_path)
        if init_states_file.exists():
            with open(init_states_file, "r") as f:
                init_states_data = json.load(f)
            init_states = init_states_data["init_states"]
            print(f"Loaded {len(init_states)} deterministic init states from: {init_states_path}")
        else:
            print(f"WARNING: init_states_path not found: {init_states_path}, falling back to random resets")

    # Get evaluation parameters

    execute_horizon = cfg.s15_eval.execute_horizon
    n_episodes = cfg.s15_eval.n_episodes
    prompt = cfg.s15_eval.get("prompt", None)
    if prompt is None:
        prompt = cfg.task.get("language_instruction", None)
    if prompt is None:
        raise ValueError(
            "No prompt specified. Set s15_eval.prompt or add language_instruction to the task YAML."
        )
    base_camera_1_name = cfg.s15_eval.base_camera_1_name
    base_camera_2_name = cfg.s15_eval.base_camera_2_name
    interactive = cfg.s15_eval.get("interactive", False)
    save_video = cfg.s15_eval.get("save_video", False)
    video_fps = cfg.s15_eval.get("video_fps", 10)
    video_resolution = cfg.s15_eval.get("video_resolution", None)
    save_per_camera_video = cfg.s15_eval.get("save_per_camera_video", False)

    # If init states are provided, override n_episodes to match
    if init_states is not None:
        if n_episodes != len(init_states):
            print(f"Overriding n_episodes from {n_episodes} to {len(init_states)} (number of init states)")
            n_episodes = len(init_states)

    # Setup UI
    overlay_window, text_labels, instance_id_label, demo_count_label, reward_label, prev_status = setup_task_status_ui(
        task_name=task_name,
        env=env,
        instance_id=None,
        show_reward=True,
    )

    # Modifies OG native handler for clean shutdown
    def clean_shutdown_handler(*args, **kwargs):
        print("Shutting down...")
        return shutdown_handler(*args, **kwargs)

    signal.signal(signal.SIGINT, clean_shutdown_handler)

    # Results tracking
    results = {
        "checkpoint": cfg.s15_eval.checkpoint,
        "task_name": task_name,
        "prompt": prompt,
        "n_steps": n_steps,
        "execute_horizon": execute_horizon,
        "n_episodes": n_episodes,
        "episodes": [],
    }

    print(f"\n{'='*60}")
    print(f"Starting Gr00t Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint: {cfg.s15_eval.checkpoint}")
    print(f"Action mode: {'ABSOLUTE' if use_absolute_actions else 'DELTA'}")
    print(f"Task: {task_name}")
    print(f"Prompt: {prompt}")
    print(f"Max steps per episode: {n_steps}")
    print(f"Execute horizon: {execute_horizon}")
    print(f"Number of episodes: {n_episodes}")
    print(f"{'='*60}\n")

    # Track interrupt state for keyboard callbacks
    interrupt_flag = {"reset": False, "debug": False}
    
    def setup_keyboard_callbacks():
        """Setup keyboard callbacks for R (reset) and B (debug/IPython embed)."""
        if gm.HEADLESS:
            return
        
        def trigger_reset():
            nonlocal interrupt_flag
            interrupt_flag["reset"] = True
            print("\n[Keyboard] Reset triggered (R pressed)...")
        
        def trigger_debug():
            nonlocal interrupt_flag
            interrupt_flag["debug"] = True
            print("\n[Keyboard] Debug mode triggered (B pressed)...")
        
        from omnigibson.utils.ui_utils import KeyboardEventHandler
        KeyboardEventHandler.initialize()
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.R,
            trigger_reset
        )
        KeyboardEventHandler.add_keyboard_callback(
            lazy.carb.input.KeyboardInput.B,
            trigger_debug
        )
        print("Keyboard callbacks registered: R=reset, B=debug (IPython)")
    
    # Initialize keyboard callbacks
    setup_keyboard_callbacks()
    
    def run_episode(episode_idx: int):
        """Run a single evaluation episode."""
        nonlocal interrupt_flag, prev_status
        print(f"\n--- Episode {episode_idx + 1}/{n_episodes} ---")
        
        # Reset state
        interrupt_flag["reset"] = False
        interrupt_flag["debug"] = False
        prev_status["success"] = False
        
        obs, _ = env.reset()
        for i in range(10):
            og.sim.step()
            og.sim.render()
        obs, _ = env.reset()

        # Apply deterministic init state if provided
        if init_states is not None and episode_idx < len(init_states):
            state = init_states[episode_idx]
            state_idx = state.get("state_idx", episode_idx)
            print(f"  Applying init state {state_idx} (grid_row={state.get('grid_row')}, grid_col={state.get('grid_col')})")
            for obj_name, obj_pose in state["objects"].items():
                obj = env.scene.object_registry("name", obj_name, None)
                if obj is not None:
                    pos = th.tensor(obj_pose["pos"], dtype=th.float32)
                    ori = th.tensor(obj_pose["ori"], dtype=th.float32)
                    obj.set_position_orientation(position=pos, orientation=ori)
                    obj.keep_still()
                else:
                    print(f"  WARNING: Object '{obj_name}' from init state not found in scene")
            # Step physics briefly to let objects settle
            for _ in range(5):
                og.sim.step_physics()
                og.sim.render()

        policy.reset()
        
        episode_result = {
            "episode_idx": episode_idx,
            "steps": 0,
            "success": False,
            "actions": [],
            "interrupted": False,
            "milestones": {},  # Will store final milestone status
            "milestone_progress": 0.0,  # Will store final milestone progress
        }
        
       
        video_frames = [] if save_video else None
        per_camera_frames = {"ext_0": [], "ext_1": [], "wrist": []} if (save_video and save_per_camera_video) else None
        
        j = 0

        # Get normalized gripper range
        gripper_limit = robot.joint_upper_limits[robot.gripper_control_idx[robot.default_arm]].mean()


        import sys
        from tqdm import tqdm
        pbar = tqdm(total=n_steps, desc=f"Episode {episode_idx + 1}/{n_episodes}", 
                    file=sys.stdout, dynamic_ncols=False, leave=True)
        
        # Track max reward for this episode (episode reward = max reward over all timesteps)
        max_episode_reward = 0.0

        while j < n_steps:
            # Wake up objects to ensure proper contact detection
            for obj in env.scene.objects:
                obj.wake()
            
            # Get robot state
            qpos = robot.get_joint_positions()
            gripper_norm = (qpos[-2:].mean() / gripper_limit).item()
            
            # Construct observation for policy
            wrist_cam_key = None
            for key in obs[robot.name].keys():
                if "Camera" in key or "wrist" in key.lower():
                    wrist_cam_key = key
                    break
            
            if wrist_cam_key is None:
                # Fallback: use first available camera
                wrist_cam_key = list(obs[robot.name].keys())[0]
            
            wrist_rgb = obs[robot.name][wrist_cam_key]["rgb"][:, :, :3]
            
            # Get base/external camera observations (DROID uses two external cameras)
            external_obs = obs.get("external", {})
            external_cams = list(external_obs.keys())
            
            # Get first external camera (exterior_image_1)
            if base_camera_1_name in external_obs:
                base_1_rgb = external_obs[base_camera_1_name]["rgb"][:, :, :3]
            elif len(external_cams) >= 1:
                base_1_rgb = external_obs[external_cams[0]]["rgb"][:, :, :3]
            else:
                base_1_rgb = wrist_rgb  # Use wrist as fallback
            
            # Get second external camera (exterior_image_2)
            if base_camera_2_name in external_obs:
                base_2_rgb = external_obs[base_camera_2_name]["rgb"][:, :, :3]
            elif len(external_cams) >= 2:
                base_2_rgb = external_obs[external_cams[1]]["rgb"][:, :, :3]
            else:
                base_2_rgb = base_1_rgb  # Use first external camera as fallback
            
            
            # Convert joints to numpy and ensure correct dtype
            joints_np = qpos[:7].numpy() if hasattr(qpos, 'numpy') else np.array(qpos[:7])
            joints_np = joints_np.astype(np.float32)
            gripper_np = np.array([gripper_norm*2], dtype=np.float32)
        

            # TODO: currently only works for DROID and pi05_droid_jointpos. Where should the observation keys be defined?
            curr_obs = {
                "exterior_image_1_left": base_1_rgb,
                "exterior_image_2_left": base_2_rgb,
                "wrist_image_left": wrist_rgb,
                "joint_position": joints_np,
                "gripper_position": gripper_np,
            }

            # Some policies (e.g. Gr00t N1.7 oxe_droid_relative_eef_*) require
            # eef_9d in the state. Compute from Franka FK; clients that don't
            # need it will simply ignore the extra key.
            eef_9d = compute_droid_eef_9d(robot)
            if eef_9d is not None:
                curr_obs["eef_9d"] = eef_9d
            
            # Helper function to capture video frame
            def capture_video_frame():
                if not save_video:
                    return
                imgs = [base_1_rgb.numpy(), base_2_rgb.numpy(), wrist_rgb.numpy()]
                if video_resolution is not None:
                    vh, vw = int(video_resolution[0]), int(video_resolution[1])
                    imgs = [resize_with_pad(img, vh, vw) for img in imgs]
                concat_frame = np.concatenate(imgs, axis=1)
                video_frames.append(concat_frame)
                if per_camera_frames is not None:
                    per_camera_frames["ext_0"].append(imgs[0])
                    per_camera_frames["ext_1"].append(imgs[1])
                    per_camera_frames["wrist"].append(imgs[2])
            

            capture_video_frame()
            
            # Debug output on first inference of first episode
            if j == 0 and episode_idx == 0:
                print(f"  [Debug] wrist_rgb shape: {wrist_rgb.shape}, dtype: {wrist_rgb.dtype}")
                print(f"  [Debug] base_1_rgb shape: {base_1_rgb.shape}, dtype: {base_1_rgb.dtype}")
                print(f"  [Debug] base_2_rgb shape: {base_2_rgb.shape}, dtype: {base_2_rgb.dtype}")
                print(f"  [Debug] joints shape: {joints_np.shape}, values: {joints_np}")
                print(f"  [Debug] gripper: {gripper_np}")
                print(f"  [Debug] prompt: {prompt}")
            

            inference_result = policy.infer(curr_obs, prompt) # should return action chunk and viz images
            action_chunk = inference_result["action"]
            n_actions_per_chunk = min(execute_horizon, len(action_chunk))
            
            # Debug output on first inference
            if j == 0:
                print(f"  [Debug] Action chunk shape: {action_chunk.shape}")
                print(f"  [Debug] Current qpos[:7]: {qpos[:7].numpy() if hasattr(qpos, 'numpy') else qpos[:7]}")
                print(f"  [Debug] Action chunk[0]: {action_chunk[0]}")
                print(f"  [Debug] Gripper norm: {gripper_np}")
            
            # Execute actions
            for i in range(n_actions_per_chunk):
                if j >= n_steps:
                    break
                
                # Check for keyboard interrupts
                if interrupt_flag["reset"]:
                    print("  Episode interrupted by reset request.")
                    episode_result["interrupted"] = True
                    break
                
                if interrupt_flag["debug"]:
                    print("  Entering debug mode (IPython)...")
                    interrupt_flag["debug"] = False  # Reset flag after entering
                    from IPython import embed
                    embed()


                # Convert action to float32 to match HDF5/LeRobot expectations
                action = th.from_numpy(action_chunk[i].astype(np.float32))

                # Gripper post-processing (configurable via hydra cfg)
                # DROID convention: 0=open, 1=closed
                # OmniGibson MultiFingerGripperController: 0=closed, 1=open
                # Default: invert + binarize at 0.5
                gripper_invert = cfg.s15_eval.get("gripper_invert", True)
                gripper_binarize = cfg.s15_eval.get("gripper_binarize", True)
                gripper_threshold = cfg.s15_eval.get("gripper_threshold", 0.5)

                if gripper_invert:
                    action[7] = 1 - action[7]

                action[7] = th.clip(action[7], 0, 1)
                raw_gripper_action = action[7].item()

                if gripper_binarize:
                    action[7] = 1.0 if action[7] > gripper_threshold else 0.0
                
                obs, reward, terminated, truncated, info = env.step(action)
                j += 1
                max_episode_reward = max(max_episode_reward, reward)
                pbar.update(1)
                pbar.set_postfix(max_reward=f"{max_episode_reward:.3f}, gripper_action={raw_gripper_action:.3f}")
                update_reward_ui(reward_label, reward)  # Show current timestep reward in UI
                
                episode_result["actions"].append(action.numpy().tolist())
                
                # Update milestones from info (keep latest)
                if "milestones" in info:
                    episode_result["milestones"] = info["milestones"]
                if "milestone_progress" in info:
                    episode_result["milestone_progress"] = info["milestone_progress"]
                
               # Update status
                status = {"success": env.task.success}
                if status["success"] and not prev_status["success"]:
                    episode_result["success"] = True
                    print(f"  Success at step {j}! (reward={reward:.3f})")
                    update_task_status(text_labels=text_labels, goal_status=status, prev_goal_status=prev_status, env=env)
                    prev_status["success"] = True  # Update prev_status to avoid repeated messages
                

                if terminated or truncated:
                    break
                
                # Break early if max reward reaches 1.0 (all milestones satisfied)
                if max_episode_reward >= 1.0 and env.task.success:
                    if not episode_result["success"]:
                        episode_result["success"] = True
                        print(f"  Success! Max reward reached 1.0 at step {j}!")
                    break
            
            # Check for reset interrupt at outer loop level
            if interrupt_flag["reset"] or episode_result["interrupted"]:
                break
            
            if terminated or truncated or episode_result["success"]:
                break
        
        pbar.close()
        episode_result["steps"] = j
        episode_result["episode_reward"] = max_episode_reward  # Episode reward = max reward over all timesteps
        
        # Capture final video frame after episode ends (shows final state)
        if save_video and video_frames is not None:
            # Get final observation state
            wrist_cam_key = None
            for key in obs[robot.name].keys():
                if "Camera" in key or "wrist" in key.lower():
                    wrist_cam_key = key
                    break
            if wrist_cam_key is None:
                wrist_cam_key = list(obs[robot.name].keys())[0]
            
            wrist_rgb_final = obs[robot.name][wrist_cam_key]["rgb"][:, :, :3]
            external_obs_final = obs.get("external", {})
            
            # Get base cameras
            if base_camera_1_name in external_obs_final:
                base_1_rgb_final = external_obs_final[base_camera_1_name]["rgb"][:, :, :3]
            elif len(list(external_obs_final.keys())) >= 1:
                base_1_rgb_final = external_obs_final[list(external_obs_final.keys())[0]]["rgb"][:, :, :3]
            else:
                base_1_rgb_final = wrist_rgb_final
            
            if base_camera_2_name in external_obs_final:
                base_2_rgb_final = external_obs_final[base_camera_2_name]["rgb"][:, :, :3]
            elif len(list(external_obs_final.keys())) >= 2:
                base_2_rgb_final = external_obs_final[list(external_obs_final.keys())[1]]["rgb"][:, :, :3]
            else:
                base_2_rgb_final = base_1_rgb_final
            
            # Capture final frame
            imgs_final = [base_1_rgb_final.numpy(), base_2_rgb_final.numpy(), wrist_rgb_final.numpy()]
            if video_resolution is not None:
                vh, vw = int(video_resolution[0]), int(video_resolution[1])
                imgs_final = [resize_with_pad(img, vh, vw) for img in imgs_final]
            concat_frame = np.concatenate(imgs_final, axis=1)
            video_frames.append(concat_frame)
            if per_camera_frames is not None:
                per_camera_frames["ext_0"].append(imgs_final[0])
                per_camera_frames["ext_1"].append(imgs_final[1])
                per_camera_frames["wrist"].append(imgs_final[2])
        
        # Final check for task success (in case it wasn't captured in the loop)
        if not episode_result["success"] and env.task.success:
            episode_result["success"] = True
            print(f"  Success detected at episode end! (reward={max_episode_reward:.3f})")
        
        # Calculate milestone progress from achieved milestones
        if episode_result["milestones"]:
            milestones_achieved = sum(1 for achieved in episode_result["milestones"].values() if achieved)
            total_milestones = len(episode_result["milestones"])
            calculated_progress = milestones_achieved / total_milestones if total_milestones > 0 else 0.0
            # Override with calculated progress (more reliable than env's info)
            episode_result["milestone_progress"] = calculated_progress
        
        print(f"  Episode {episode_idx + 1}/{n_episodes} completed: steps={j}, success={episode_result['success']}, episode_reward={max_episode_reward:.3f}")
        
        # Print milestone status
        if episode_result["milestones"]:
            print(f"  Milestones achieved:")
            for milestone_name, achieved in episode_result["milestones"].items():
                status = "✓" if achieved else "✗"
                print(f"    {status} {milestone_name}")
            print(f"  Milestone progress: {episode_result['milestone_progress']:.2%}")
        
        # Save video for this episode if enabled
        if save_video and video_frames:
            video_dir = f"{results_dir}/videos"
            Path(video_dir).mkdir(parents=True, exist_ok=True)
            success_str = "success" if episode_result["success"] else "fail"
            video_path = f"{video_dir}/episode_{episode_idx:03d}_{success_str}.mp4"
            print(f"  Saving video to: {video_path}")
            imageio.mimwrite(video_path, video_frames, fps=video_fps)
            episode_result["video_path"] = video_path
            
            if per_camera_frames is not None:
                cameras_root = f"{video_dir}/cameras"
                for cam_name, frames in per_camera_frames.items():
                    if not frames:
                        continue
                    cam_video_dir = f"{cameras_root}/{cam_name}"
                    Path(cam_video_dir).mkdir(parents=True, exist_ok=True)
                    cam_video_path = f"{cam_video_dir}/episode_{episode_idx:03d}_{success_str}.mp4"
                    imageio.mimwrite(cam_video_path, frames, fps=video_fps)
                print(f"  Saved per-camera videos under: {cameras_root}/{{{','.join(per_camera_frames.keys())}}}/")
        
        return episode_result

    # Run evaluation episodes
    if interactive:
        print("\nInteractive mode enabled. Use IPython to run evaluations.")
        print("Call run_episode(0) to run an episode, or use env.reset() and policy.infer() directly.")
        print("Note: Using Gr00t model.")
        from IPython import embed
        embed()
    else:
        for episode_idx in range(n_episodes):
            episode_result = run_episode(episode_idx)
            results["episodes"].append(episode_result)

        # Compute summary statistics
        successes = sum(1 for ep in results["episodes"] if ep["success"])
        results["success_rate"] = successes / n_episodes
        results["avg_steps"] = sum(ep["steps"] for ep in results["episodes"]) / n_episodes
        results["avg_episode_reward"] = sum(ep["episode_reward"] for ep in results["episodes"]) / n_episodes
        results["avg_milestone_progress"] = sum(ep.get("milestone_progress", 0.0) for ep in results["episodes"]) / n_episodes
        
        # Compute per-milestone success rates
        if results["episodes"] and results["episodes"][0].get("milestones"):
            milestone_names = list(results["episodes"][0]["milestones"].keys())
            milestone_success_rates = {}
            for milestone_name in milestone_names:
                milestone_successes = sum(1 for ep in results["episodes"] if ep.get("milestones", {}).get(milestone_name, False))
                milestone_success_rates[milestone_name] = milestone_successes / n_episodes
            results["milestone_success_rates"] = milestone_success_rates
        
        print(f"\n{'='*60}")
        print(f"Evaluation Complete")
        print(f"{'='*60}")
        print(f"Success rate: {successes}/{n_episodes} ({results['success_rate']*100:.1f}%)")
        print(f"Average steps: {results['avg_steps']:.1f}")
        print(f"Average episode reward: {results['avg_episode_reward']:.3f}")
        print(f"Average milestone progress: {results['avg_milestone_progress']:.1%}")
        
        if "milestone_success_rates" in results:
            print(f"\nPer-milestone success rates:")
            for milestone_name, rate in results["milestone_success_rates"].items():
                print(f"  {milestone_name}: {rate*100:.1f}%")
        
        print(f"{'='*60}\n")

        # Save results
        results_path = f"{results_dir}/eval_results.json"
        dump_json(results, results_path)
        print(f"Results saved to: {results_path}")

    # Save data and close data collection wrapper to finalize dataset
    env.save_data()  # This flushes remaining trajectory and writes n_episodes attribute
    print(f"Rollout states/actions saved to: {rollout_dataset_path}")

    # Construct output path for LeRobot dataset (within results directory)
    lerobot_dataset_name = f"eval_rollouts"
    lerobot_output_path = f"{results_dir}/lerobot/{lerobot_dataset_name}"

    # Build the replay command for manual execution
    # Note: This uses the rollout HDF5 path which is now in results_dir
    replay_cmd = (
        f"python 18_replay_dataset.py "
        f"s14_teleop.out_dir={results_dir} "
        f"s18_replay.out_dir={results_dir} "
        f"s18_replay.dataset_name={lerobot_dataset_name} "
        f"s18_replay.external_sensors_cfg={cfg.s15_eval.external_sensors_cfg} "
        f"s18_replay.include_mimicgen_data=false "
        f"s18_replay.playback_kwargs.only_successes=false"
    )
    
    # Save stage info
    end_stage(cfg, success=True, additional_info={
        **(results if not interactive else {}),
        "rollout_hdf5": rollout_dataset_path,
        "lerobot_output": lerobot_output_path,
        "replay_cmd": replay_cmd,
    })

    # Print instructions for manual replay
    print(f"\n{'='*60}")
    print("To replay rollouts and save observations, run:")
    print(f"  cd {cfg_dir}")
    print(f"  {replay_cmd}")
    print(f"{'='*60}\n")

    og.shutdown()


def end_stage(cfg, success=False, additional_info: dict = None):
    """
    Function that is run at the end of every stage to save the stage config and additional info.
    """
    save_dir = cfg.s15_eval.out_dir
    stage_cfg = OmegaConf.to_object(cfg.s15_eval)
    stage_cfg['success'] = success
    if additional_info is not None:
        stage_cfg.update(additional_info)
    dump_json(stage_cfg, f"{save_dir}/stage_info.json")


if __name__ == "__main__":
    main()
