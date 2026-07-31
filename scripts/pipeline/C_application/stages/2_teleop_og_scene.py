# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from b1k

Requires installing:

- BEHAVIOR-1K, see https://github.com/StanfordVL/BEHAVIOR-1K
"""
import omnigibson as og
from omnigibson import shutdown_handler
from omnigibson.macros import gm
import omnigibson.lazy as lazy
from omnigibson.scenes import Scene
from omnigibson.objects import DatasetObject
from omnigibson.robots import REGISTERED_ROBOTS, BaseRobot, LocomotionRobot, MobileManipulationRobot, Yam, FrankaPanda
from omnigibson.controllers import InverseKinematicsController, OperationalSpaceController
import omnigibson.utils.transform_utils as T
from omnigibson.utils.ui_utils import choose_from_options, KeyboardEventHandler
from omnigibson.utils.config_utils import parse_config
from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths
from omnigibson.envs import HDF5CollectionWrapper
from pathlib import Path
import torch as th
import json
import subprocess
import time
import hydra
from omegaconf import OmegaConf
from simfoundry import import_og_dependencies, CFG_DIR as SIMFOUNDRY_CFG_DIR, ASSET_DIR as SIMFOUNDRY_ASSET_DIR
from simfoundry.utils.processing_utils import dump_json
from simfoundry.utils.og_utils import (
    apply_teleop_omnigibson_macros,
    set_obj_materials,
    setup_task_status_ui,
    update_task_status,
    update_demo_success_counter,
    update_reward_ui,
    setup_robot_visualizers,
    update_in_hand_status,
    update_grasp_status,
)
from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths
from simfoundry.utils.object_swap_utils import (
    apply_object_swaps,
    adjust_swapped_objects_z,
)
import os
from datetime import datetime
import signal

# Needed so custom tasks can be instantiated properly
import_og_dependencies()

CFG_DIR = SIMFOUNDRY_CFG_DIR


### At the start of every script, we cd into the scripts/config directory
scripts_dir = os.path.dirname(os.path.abspath(__file__))
cfg_dir = os.path.join(scripts_dir, "..", "cfg")
os.chdir(cfg_dir)


def restore_scene_object_poses(env, scene_json):
    """
    Restore non-robot object positions and joint states from a scene JSON state section.

    Skips robots so that robot state and task success flags are not affected.
    Tensorizes joint state lists from JSON before calling load_state to avoid
    'list has no attribute to' errors from the physics backend.

    Args:
        env: OmniGibson Environment
        scene_json (dict): loaded scene JSON (the same dict passed as scene_file)
    """
    robot_names = {robot.name for robot in env.robots}
    obj_registry = (
        scene_json.get("state", {})
        .get("registry", {})
        .get("object_registry", {})
    )
    for obj_name, obj_state in obj_registry.items():
        if obj_name in robot_names:
            continue
        obj = env.scene.object_registry("name", obj_name)
        if obj is None:
            continue
        # JSON deserializes tensors as plain lists, but load_state passes them
        # directly to set_joint_positions which expects tensors
        for key in ("joint_pos", "joint_vel", "joint_effort"):
            if key in obj_state and isinstance(obj_state[key], list):
                obj_state[key] = th.tensor(obj_state[key], dtype=th.float32)
        obj.load_state(obj_state)
        obj.keep_still()



TELEOP_METHOD = {
    "keyboard": "Keyboard (default)",
    "spacemouse": "SpaceMouse",
    "oculus": "Oculus Quest",
}

EXTERNAL_CAMERA_CONFIG = {
    # "external_cam0_opposite": {
    #     "position": [-0.58,  0.37,  0.66046],
    #     "orientation": [-0.36555,  0.38907,  0.64547, -0.54624],
    # },
    "external_cam1_left": {
        "position": [-0.1838, -0.6026,  0.4259],
        "orientation": [ 0.5413, -0.3535,  0.4049,  0.6466],
    },
    "external_cam2_opposite": {
        "position": [-0.48,  -0.82,  0.66046],
        "orientation": [0.5044,  -0.20347,  -0.35345, 0.76109],
    },
}

# Viewer/perspective camera config (set to None to skip override)
VIEWER_CAMERA_CONFIG = {
    "position": [-1.07841,  -0.78161,  0.88046],
    "orientation": [36.539,  -60.566,  -49.607],  # euler angles in degrees
}

HIGHLIGHT_COLOR = (0.6, 0.9, 0.6)  # Light green RGB for object highlighting


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    sim_dir = cfg.s10_sim.out_dir
    og_dir = cfg.s13_og.out_dir
    out_dir = cfg.s14_teleop.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load physics objects info and resolve scene JSON path
    scene_json_name = cfg.s14_teleop.scene_json_name
    scene_source = cfg.s14_teleop.get("scene_source", "reconstructed")
    success_demo_per_sampling = int(cfg.s14_teleop.get("success_demo_per_sampling", 1))
    sampling_scene_indices = cfg.s14_teleop.get("sampling_scene_indices", None)
    if sampling_scene_indices is not None:
        sampling_scene_indices = list(sampling_scene_indices)
    obj_frictions = dict()
    valid_sampling_scene_paths = []  # When scene_source == "load_sampling", list of paths to valid scene JSONs

    if scene_json_name is not None:
        scene_asset_dir = os.path.join(SIMFOUNDRY_ASSET_DIR, "scenes", scene_json_name)
        og_scene_json_path = os.path.join(scene_asset_dir, f"{scene_json_name}_scene_state_latest.json")
        if scene_source == "load_sampling":
            sampling_dir = os.path.join(scene_asset_dir, "sampling_scene/auto_generation")
            summary_path = os.path.join(sampling_dir, "generation_summary.json")
            if not os.path.exists(summary_path):
                raise FileNotFoundError(
                    f"generation_summary.json not found at {summary_path}. "
                    f"Run scene_sampling.py with input_scene_json=assets/scenes/{scene_json_name}/"
                    f"{scene_json_name}_scene_state_latest.json first."
                )
            with open(summary_path, "r") as f:
                generation_summary = json.load(f)
            valid_entries = [s for s in generation_summary.get("scenes", []) if s.get("valid", False)]
            if not valid_entries:
                raise RuntimeError(f"No valid scenes in {summary_path}. Run scene_sampling.py first.")
            # Optionally filter to specific scene indices
            if sampling_scene_indices is not None:
                max_idx = len(valid_entries) - 1
                bad_indices = [i for i in sampling_scene_indices if i < 0 or i > max_idx]
                if bad_indices:
                    raise IndexError(
                        f"sampling_scene_indices {bad_indices} out of range for {len(valid_entries)} valid scene(s) (0..{max_idx})"
                    )
                valid_entries = [valid_entries[i] for i in sampling_scene_indices]
                print(f"[load_sampling] Filtered to scene indices {sampling_scene_indices} ({len(valid_entries)} scene(s))")
            valid_sampling_scene_paths = [
                os.path.join(sampling_dir, s["scene_json"]) for s in valid_entries
            ]
            for p in valid_sampling_scene_paths:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Valid scene JSON not found: {p}")
            og_scene_json_path = valid_sampling_scene_paths[0]
            print(f"[load_sampling] Found {len(valid_sampling_scene_paths)} valid scene(s) in {sampling_dir}, "
                  f"will collect {success_demo_per_sampling} success(es) per scene into one HDF5.")
    elif scene_source == "load_sampling":
        raise ValueError(
            "scene_source=load_sampling requires scene_json_name to be set. "
            "Use scene_sampling.py to generate samples under assets/scenes/<name>/sampling_scene/, "
            "then set scene_json_name=<name> with scene_source=load_sampling."
        )
    else:
        og_scene_json_path = os.path.join(og_dir, "reconstructed_og_scene.json")
    og_scene_json = load_json_with_absolute_usd_paths(og_scene_json_path)

    # Optionally swap scene objects with alternative USDs (e.g., cousin assets)
    swap_info = {}
    swap_json_path = cfg.s14_teleop.get("object_swap_json", None)
    if swap_json_path is not None:
        if not os.path.isabs(swap_json_path):
            swap_json_path = os.path.join(SIMFOUNDRY_ASSET_DIR, swap_json_path)
        if os.path.exists(swap_json_path):
            swap_info = apply_object_swaps(og_scene_json, swap_json_path)
            print(f"[ObjectSwap] Applied {len(swap_info)} object swap(s) from {swap_json_path}")
        else:
            print(f"[ObjectSwap] Warning: swap JSON not found at {swap_json_path}")
        
    scene_objects_info = {}
    if scene_json_name is None:
        scene_objects_info_fpath = f"{sim_dir}/scene_objects_info.json"
        if os.path.exists(scene_objects_info_fpath):
            with open(scene_objects_info_fpath, "r") as f:
                scene_objects_info = json.load(f)
            for _, obj_info in scene_objects_info.items():
                obj_frictions[obj_info["name"]] = obj_info["friction"]

    # grasping mode (e.g., from assisted to physical)
    grasping_mode_override = cfg.s14_teleop.get("grasping_mode", None)
    if grasping_mode_override is not None:
        if "objects_info" in og_scene_json and "init_info" in og_scene_json["objects_info"]:
            for obj_name, obj_info in og_scene_json["objects_info"]["init_info"].items():
                if obj_name.startswith("robot") and "args" in obj_info:
                    if "grasping_mode" in obj_info["args"]:
                        print(f"Changing grasping mode for {obj_name} from '{obj_info['args']['grasping_mode']}' to '{grasping_mode_override}'")
                        obj_info["args"]["grasping_mode"] = grasping_mode_override

    # Create scene config
    scene_cfg = {
        "type": "Scene",
        "scene_file": og_scene_json,
        "use_floor_plane": cfg.s14_teleop.use_floor_plane,
        "floor_plane_visible": cfg.s14_teleop.floor_plane_visible,
        "use_skybox": True,
        "include_robots": True,
    }
    # Task name for loading YAML (override with task.task_name=serve_the_orange for scene-specific tasks)
    task_name = cfg.task.task_name
    scene_name = cfg.get("scene_name", "")
    # Prefer scene-specific task config if present: task/<scene_name>/<task_name>.yaml
    og_task_cfg_path = os.path.join(SIMFOUNDRY_CFG_DIR, "task", scene_name, f"{task_name}.yaml") if scene_name else ""
    if not (og_task_cfg_path and os.path.exists(og_task_cfg_path)):
        og_task_cfg_path = os.path.join(SIMFOUNDRY_CFG_DIR, "task", f"{task_name}.yaml")
    task_cfg = parse_config(og_task_cfg_path)["og_task_config"]
    task_cfg["termination_config"]["max_steps"] = cfg.s14_teleop.max_steps

    # Deterministic pose schedule setup (disabled when load_sampling: one trajectory per valid scene instead)
    use_deterministic_schedule = cfg.s14_teleop.use_deterministic_pose_schedule and (scene_source != "load_sampling")
    n_deterministic_samples = cfg.s14_teleop.n_deterministic_samples
    det_schedule = None

    if use_deterministic_schedule:
        task_cfg["ground_plane_z_randomization"] = 0
        group_z_rot_cfg = task_cfg.get("group_z_rot_randomization", {}) or {}
        schedule = []
        for i in range(n_deterministic_samples):
            z_rots = {}
            for group, max_z_rot in group_z_rot_cfg.items():
                if max_z_rot is not None and max_z_rot > 0:
                    if n_deterministic_samples > 1:
                        offset = -max_z_rot + (2.0 * max_z_rot * i / (n_deterministic_samples - 1))
                    else:
                        offset = 0.0
                    z_rots[group] = offset
            schedule.append(z_rots)
        det_schedule = {
            "schedule": schedule,
            "idx": 0,
            "n_samples": n_deterministic_samples,
            "done": False,
        }
        print(f"[Deterministic Schedule] Enabled with {n_deterministic_samples} configurations")
        print(f"[Deterministic Schedule] Ground plane z-randomization overridden to 0")
        for i, config in enumerate(schedule):
            config_str = ", ".join(f"{g}: {v:.4f} rad" for g, v in config.items())
            print(f"  Config {i+1}: {config_str}")

    external_sensors_cfg_path = f"{SIMFOUNDRY_CFG_DIR}/external_sensors/{cfg.s14_teleop.external_sensors_cfg}.yaml"
    action_freq = cfg.s14_teleop.action_freq
    env_cfg = {
        "external_sensors": parse_config(external_sensors_cfg_path)["external_sensors"],
        "action_frequency": action_freq,
        "rendering_frequency": action_freq,
        "physics_frequency": 120,
    }

    # TODO: Place robot at specific locations? Randomize etc?

    og_cfg = dict(env=env_cfg, scene=scene_cfg, task=task_cfg)

    # Create the environment
    apply_teleop_omnigibson_macros()
    env = og.Environment(configs=og_cfg)

    # Set deterministic override for first reset
    if det_schedule is not None:
        env.task._override_group_z_rotations = det_schedule["schedule"][det_schedule["idx"]]

    env.reset()
    # scene.reset() (called inside env.reset()) resets objects to their _default_state which
    # is set from USD file defaults at creation time, not from the settled positions in the JSON.
    # Restore the correct settled positions from the scene JSON state section.
    restore_scene_object_poses(env, og_scene_json)

    # AABB-based z adjustment for swapped objects that lacked pre-computed size info
    if swap_info and any(info["needs_aabb_adjustment"] for info in swap_info.values()):
        gp_info = og_scene_json.get("ground_plane_info", {})
        gp_z = gp_info.get("position", [0, 0, 0])[2] if gp_info else 0.0
        adjust_swapped_objects_z(env, swap_info, ground_plane_z=gp_z)

    # Override external camera positions/orientations from EXTERNAL_CAMERA_CONFIG
    if env.external_sensors is not None:
        for cam_name, cam_params in EXTERNAL_CAMERA_CONFIG.items():
            if cam_name in env.external_sensors:
                sensor = env.external_sensors[cam_name]
                pos = th.tensor(cam_params["position"])
                ori = cam_params["orientation"]
                # Convert euler angles (degrees) to quaternion if needed
                if len(ori) == 3:
                    ori_rad = th.deg2rad(th.tensor(ori))
                    quat = T.euler2quat(ori_rad)
                else:
                    quat = th.tensor(ori)
                sensor.set_position_orientation(position=pos, orientation=quat, frame="parent")  # Local coordinates
                print(f"Set camera '{cam_name}' (local) position={pos.tolist()}, orientation={quat.tolist()}")

    # Override viewer/perspective camera from VIEWER_CAMERA_CONFIG
    if VIEWER_CAMERA_CONFIG is not None:
        pos = th.tensor(VIEWER_CAMERA_CONFIG["position"])
        ori = VIEWER_CAMERA_CONFIG["orientation"]
        # Convert euler angles (degrees) to quaternion if needed
        if len(ori) == 3:
            ori_rad = th.deg2rad(th.tensor(ori))
            quat = T.euler2quat(ori_rad)
        else:
            quat = th.tensor(ori)
        og.sim.viewer_camera.set_position_orientation(position=pos, orientation=quat)
        print(f"Set viewer camera position={pos.tolist()}, orientation={quat.tolist()}")

    objs = {name: env.scene.object_registry("name", name) for name in obj_frictions.keys()}
    objs = {k: v for k, v in objs.items() if v is not None}

    # If there are multiple robots, ask user which robot to control
    n_robots = len(env.robots)
    action_dim = 0
    action_start_end_idxs = None
    if n_robots > 1:
        print("\nMultiple robots detected:")
        actions_start_end_idxs = dict()
        for i, r in enumerate(env.robots):
            print(f"  [{i}]: {getattr(r, 'name', repr(r))}")
            actions_start_end_idxs[i] = (action_dim, action_dim + r.action_dim)
            action_dim += r.action_dim
        while True:
            try:
                selected_idx = int(input(f"Enter index of robot to control (0 - {n_robots - 1}): "))
                if 0 <= selected_idx < n_robots:
                    break
                else:
                    print(f"Index must be between 0 and {n_robots - 1}")
            except Exception as e:
                print("Invalid input. Try again.")
        robot = env.robots[selected_idx]
        action_start_end_idxs = actions_start_end_idxs[selected_idx]
    else:
        robot = env.robots[0]
        action_dim = robot.action_dim
        action_start_end_idxs = (0, action_dim)

    # Set object materials and visualizers
    with og.sim.stopped():
        obj_materials = set_obj_materials(objs=objs, obj_frictions=obj_frictions)
        vis_elements = setup_robot_visualizers(robot=robot, scene=env.scene, offset=th.tensor([0.0, -0.041, 0.13]) if isinstance(robot, Yam) else None)

        # Highlight all non-robot/non-background objects in light green for teleop visibility
        if cfg.s14_teleop.highlight_objects:
            highlight_color = th.tensor(HIGHLIGHT_COLOR)
            n_highlighted = 0
            for obj in env.scene.objects:
                if isinstance(obj, BaseRobot):
                    continue
                if "background" in obj.name.lower():
                    continue
                for material in obj.materials:
                    try:
                        input_names = material.shader_input_names
                        if "diffuse_texture" in input_names:
                            material.set_input("diffuse_texture", lazy.pxr.Sdf.AssetPath(""))
                        if "diffuse_color_constant" in input_names:
                            material.diffuse_color_constant = highlight_color
                        elif "diffuse_tint" in input_names:
                            material.diffuse_tint = highlight_color
                    except Exception as e:
                        print(f"Warning: Could not highlight material on '{obj.name}': {e}")
                n_highlighted += 1
            print(f"Highlighted {n_highlighted} scene objects in light green for teleop visibility")

        # Hide robot arm body links, keeping only gripper/EEF links visible
        if cfg.s14_teleop.hide_robot_body:
            arm_link_set = set()
            for arm, link_names in robot.arm_link_names.items():
                arm_link_set.update(link_names)
            if isinstance(robot, FrankaPanda) and robot.end_effector == "robotiq":
                arm_link_set.update(["camera_link"])
            n_hidden = 0
            for link_name, link in robot.links.items():
                if link_name not in arm_link_set:
                    continue
                for mesh in link.visual_meshes.values():
                    mesh.visible = False
                n_hidden += 1
            print(f"Hidden {n_hidden} robot body links (gripper links remain visible)")

    eef_cylinder_geoms = vis_elements["eef_cylinder_geoms"]
    vis_mats = vis_elements["vis_mats"]
    prev_grasp_status = {arm: False for arm in robot.arm_names}
    prev_in_hand_status = {arm: False for arm in robot.arm_names}

    assert len(robot.arm_names) == 1, "Teleop only supports controlling a single arm robot for now!"

    # Load IK controller to control robots
    arm_teleop_method = cfg.s14_teleop.device #choose_from_options(options=TELEOP_METHOD, name="robot arm teleop method")
    assert arm_teleop_method in ["spacemouse", "oculus"], "Must use spacemouse or oculus for teleop!"
    for r in env.robots:
        if isinstance(r, Yam):
            open_qpos = [-0.045, 0.045]
            closed_qpos = [0.0, 0.0]
            inverted = [False, True]
        else:
            open_qpos = None
            closed_qpos = None
            inverted = [True, True] #False
        controller_config = {
            f"arm_{r.default_arm}": {
                "name": "InverseKinematicsController",
                "command_input_limits": None,
                "smoothing_filter_size": 5,
            },
            f"gripper_{r.default_arm}": {
                "name": "MultiFingerGripperController",
                "command_input_limits": (0.0, 1.0),
                "mode": "smooth",
                "open_qpos": open_qpos,
                "closed_qpos": closed_qpos,
                "inverted": inverted,
            },
        }
        if arm_teleop_method == "spacemouse":
            controller_config[f"arm_{r.default_arm}"]["mode"] = "pose_delta_ori"
        else:
            controller_config[f"arm_{r.default_arm}"]["command_output_limits"] = None
            controller_config[f"arm_{r.default_arm}"]["mode"] = "absolute_pose"

        r.reload_controllers(controller_config=controller_config)
        assert isinstance(r.controllers[f"arm_{r.default_arm}"], InverseKinematicsController),"Teleop only support controlling with IK Controller!"

    env.scene.update_initial_file()

    # Re-apply joint positions AFTER update_initial_file(), because the with og.sim.stopped()
    # block above resets joints to 0 (from the initial file), and update_initial_file() would
    # otherwise bake joints=0. Re-applying here and calling update_initial_file() again bakes
    # the correct joint positions so all subsequent env.reset() calls restore them properly.
    restore_scene_object_poses(env, og_scene_json)
    og.sim.step()
    env.scene.update_initial_file()

    # # TEMPORARY: collect fixed articulated objects (non-robot, non-background, has joints)
    # # keep_still() is called on them every frame to prevent drawers/doors from drifting.
    # _fixed_articulated_objs = [
    #     obj for obj in env.scene.objects
    #     if not isinstance(obj, BaseRobot)
    #     and "background" not in obj.name.lower()
    #     and hasattr(obj, "joints") and obj.joints
    #     and getattr(obj, "fixed_base", False)
    # ]
    # print(f"\n[TEMP] Will call keep_still() each frame on {len(_fixed_articulated_objs)} fixed articulated object(s): "
    #       f"{[o.name for o in _fixed_articulated_objs]}")


    # See if we have a yam workstation ("usd_workstation_0") -- if so, set one of the walls to false so we can teleop easily
    workstation = env.scene.object_registry("name", "usd_workstation_0")
    if workstation is not None:
        workstation.root_link.visual_meshes["visuals_Plane"].visible = False

    # Wrap in data environment
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    swap_suffix = ""
    if swap_info:
        swap_stem = Path(swap_json_path).stem
        swap_suffix = f"_swap_{swap_stem}"
    if det_schedule is not None:
        hdf5_path = f"{out_dir}/{task_name}_{now_str}{swap_suffix}_deterministic_{n_deterministic_samples}configs.hdf5"
    elif scene_source == "load_sampling":
        hdf5_path = f"{out_dir}/{task_name}_{now_str}{swap_suffix}_load_sampling.hdf5"
    else:
        hdf5_path = f"{out_dir}/{task_name}_{now_str}{swap_suffix}.hdf5"
    env = HDF5CollectionWrapper(
        env=env,
        output_path=hdf5_path,
        **OmegaConf.to_container(cfg.s14_teleop.data_env_kwargs, resolve=True),
    )
    env.is_recording = False
    env.reset()
    # Restore correct object poses (env.reset() uses USD defaults, not JSON settled poses)
    restore_scene_object_poses(env, og_scene_json)
    env.is_recording = cfg.s14_teleop.save_data

    # Create a teleop system
    try:
        from telemoma.configs.base_config import teleop_config
    except ImportError as e:
        raise ImportError(
            "TeleMoMa is required for teleoperation but is not installed.\n\n"
            "TeleMoMa (https://github.com/UT-Austin-RobIn/telemoma) ships no license file\n"
            "and is therefore all-rights-reserved. SimFoundry does not install, distribute,\n"
            "or mirror it, and grants no rights to it.\n\n"
            "If you have established your own right to use TeleMoMa, install it separately:\n"
            "    pip install --no-deps telemoma==0.3.0\n\n"
            "See THIRD_PARTY_LICENSES.md and INSTALL.md."
        ) from e
    from omnigibson.utils.teleop_utils import TeleopSystem

    if isinstance(robot, (LocomotionRobot, MobileManipulationRobot)):
        base_teleop_method = choose_from_options(options=TELEOP_METHOD, name="robot base teleop method")
    else:
        base_teleop_method = "keyboard"  # Dummy value since FrankaPanda does not have a base
    # Generate teleop config
    teleop_config.arm_left_controller = arm_teleop_method
    teleop_config.arm_right_controller = arm_teleop_method
    teleop_config.base_controller = base_teleop_method
    teleop_config.interface_kwargs["keyboard"] = {"arm_speed_scaledown": cfg.s14_teleop.arm_speed_scaledown}
    teleop_config.interface_kwargs["spacemouse"] = {"arm_speed_scaledown": cfg.s14_teleop.arm_speed_scaledown, "arm_rot_speed_scaledown": cfg.s14_teleop.arm_rot_speed_scaledown}

    # Initialize teleoperation system
    teleop_sys = TeleopSystem(config=teleop_config, robot=robot, show_control_marker=True)
    teleop_sys.start()
    print("Teleop system started!")

    # Prevent swapping between controllers
    teleop_sys.interfaces[arm_teleop_method].controllable_robot_parts = ["right"]

    # Left spacemouse button is grasp, so make sure right button maps to save episode / env reset

    # Modifies OG native handler to make sure we save the data before shutting down
    def data_shutdown_handler(*args, **kwargs):
        # Save data
        if env.is_recording:
            print("Saving data before shutting down...")
            env.save_data()
            print("Successfully saved data!")

        return shutdown_handler(*args, **kwargs)

    # Register keyboard callback for reset without saving
    is_success = False
    def reset_without_saving():
        """Reset the environment without saving the current demonstration"""
        nonlocal is_success, cumulative_reward
        was_recording = env.is_recording
        env.is_recording = False
        print("Resetting environment without saving...")
        if det_schedule is not None:
            env.task._override_group_z_rotations = det_schedule["schedule"][det_schedule["idx"]]
        env.reset()
        teleop_sys.reset()
        is_success = False
        cumulative_reward = 0.0
        update_reward_ui(reward_label, cumulative_reward)
        update_task_status(text_labels=text_labels, goal_status={"success": False}, prev_goal_status={"success": True}, env=env)
        env.is_recording = was_recording
        print("Environment reset complete.")

    # Register 'R' key to reset without saving
    from omnigibson.utils.ui_utils import KeyboardEventHandler
    KeyboardEventHandler.initialize()
    KeyboardEventHandler.add_keyboard_callback(
        lazy.carb.input.KeyboardInput.R,
        reset_without_saving
    )

    # Register 'B' key to drop into IPython debugging session
    def drop_into_debugger():
        nonlocal env, robot, teleop_sys
        """Drop into an IPython embed session for debugging"""
        print("Dropping into IPython debugger (press Ctrl+D to continue)...")
        from IPython import embed; embed()
        print("Resuming teleop...")

    KeyboardEventHandler.add_keyboard_callback(
        lazy.carb.input.KeyboardInput.B,
        drop_into_debugger
    )

    # Something somewhere disables the default SIGINT handler, so we need to re-enable it
    signal.signal(signal.SIGINT, data_shutdown_handler)

    # Setup UI
    overlay_window, text_labels, instance_id_label, demo_count_label, reward_label, prev_status = setup_task_status_ui(
        task_name=task_name,
        env=env,
        instance_id=None,
        show_reward=True,
    )

    # Initialize demo success counter and cumulative reward
    demo_success_count = 0
    cumulative_reward = 0.0

    print(f"Starting teleop!\n\n")
    if scene_source == "load_sampling" and valid_sampling_scene_paths:
        print(f"[load_sampling] Scene 1/{len(valid_sampling_scene_paths)}: {os.path.basename(valid_sampling_scene_paths[0])} (need {success_demo_per_sampling} success(es) to advance)\n")
    prev_reset_button = 0
    fps = env.env_config["action_frequency"]
    load_sampling_state = {"index": 0, "done": False, "scene_success_count": 0}  # used when scene_source == "load_sampling"

    def check_reset(teleop_sys, prev_reset_button, is_success, demo_success_count, cumulative_reward, env):
        # Check for reset
        if arm_teleop_method == "spacemouse":
            l_button, _ = teleop_sys.interfaces["spacemouse"].raw_data.buttons
            if l_button and not prev_reset_button:
                # Advance deterministic schedule before reset
                if det_schedule is not None:
                    if is_success:
                        det_schedule["idx"] += 1
                    if det_schedule["idx"] < det_schedule["n_samples"]:
                        env.task._override_group_z_rotations = det_schedule["schedule"][det_schedule["idx"]]
                        print(f"[Deterministic] {'Advancing to' if is_success else 'Retrying'} config {det_schedule['idx']+1}/{det_schedule['n_samples']}")
                # Reset! For load_sampling, only save successful episodes to keep HDF5 clean.
                if scene_source == "load_sampling" and not is_success:
                    _was_rec = env.is_recording
                    env.is_recording = False
                    env.reset()
                    env.is_recording = _was_rec
                else:
                    env.reset()
                teleop_sys.reset()
                was_success = is_success
                if is_success:
                    demo_success_count += 1
                if det_schedule is not None and det_schedule["idx"] >= det_schedule["n_samples"]:
                    det_schedule["done"] = True
                update_demo_success_counter(demo_count_label, demo_success_count)
                is_success = False
                cumulative_reward = 0.0
                update_reward_ui(reward_label, cumulative_reward)
                update_task_status(text_labels=text_labels, goal_status={"success": False}, prev_goal_status={"success": True}, env=env)
                # load_sampling: retry same scene or advance when enough successes collected
                if scene_source == "load_sampling" and valid_sampling_scene_paths:
                    cur_idx = load_sampling_state["index"]
                    if was_success:
                        load_sampling_state["scene_success_count"] += 1
                        print(f"[load_sampling] Scene {cur_idx+1}/{len(valid_sampling_scene_paths)}: success {load_sampling_state['scene_success_count']}/{success_demo_per_sampling}")
                    if load_sampling_state["scene_success_count"] >= success_demo_per_sampling:
                        # Advance to next scene
                        load_sampling_state["index"] += 1
                        load_sampling_state["scene_success_count"] = 0
                        if load_sampling_state["index"] >= len(valid_sampling_scene_paths):
                            load_sampling_state["done"] = True
                            print(f"[load_sampling] All {len(valid_sampling_scene_paths)} scenes done. Save and exit.")
                        else:
                            next_path = valid_sampling_scene_paths[load_sampling_state["index"]]
                            next_scene_json = load_json_with_absolute_usd_paths(next_path)
                            # Disable recording: this reset just sets up the next scene, not an episode
                            _was_rec = env.is_recording
                            env.is_recording = False
                            env.reset()
                            env.is_recording = _was_rec
                            restore_scene_object_poses(env, next_scene_json)
                            # apply_joint_friction_scale(env)
                            teleop_sys.reset()
                            print(f"[load_sampling] Scene {load_sampling_state['index']+1}/{len(valid_sampling_scene_paths)}: {os.path.basename(next_path)} (need {success_demo_per_sampling} success(es))")
                    else:
                        # Retry same scene - restore poses after env.reset() above
                        cur_path = valid_sampling_scene_paths[cur_idx]
                        cur_scene_json = load_json_with_absolute_usd_paths(cur_path)
                        restore_scene_object_poses(env, cur_scene_json)
                        # apply_joint_friction_scale(env)
                        remaining = success_demo_per_sampling - load_sampling_state["scene_success_count"]
                        print(f"[load_sampling] Retrying scene {cur_idx+1}/{len(valid_sampling_scene_paths)} ({remaining} success(es) remaining)")
            prev_reset_button = l_button
        elif arm_teleop_method == "oculus":
            raw_action = teleop_sys.interfaces['oculus'].get_action(teleop_sys.get_obs())
            buttons = raw_action.extra['buttons']
            reset_button = buttons['A'] or buttons['B']
            is_success = is_success and not buttons['B'] # B can be used to cancel the success
            if reset_button and not prev_reset_button:
                # Advance deterministic schedule before reset
                if det_schedule is not None:
                    if is_success:
                        det_schedule["idx"] += 1
                    if det_schedule["idx"] < det_schedule["n_samples"]:
                        env.task._override_group_z_rotations = det_schedule["schedule"][det_schedule["idx"]]
                        print(f"[Deterministic] {'Advancing to' if is_success else 'Retrying'} config {det_schedule['idx']+1}/{det_schedule['n_samples']}")
                # Reset! For load_sampling, only save successful episodes to keep HDF5 clean.
                if scene_source == "load_sampling" and not is_success:
                    _was_rec = env.is_recording
                    env.is_recording = False
                    env.reset()
                    env.is_recording = _was_rec
                else:
                    env.reset()
                teleop_sys.reset()
                was_success = is_success
                if is_success:
                    demo_success_count += 1
                if det_schedule is not None and det_schedule["idx"] >= det_schedule["n_samples"]:
                    det_schedule["done"] = True
                update_demo_success_counter(demo_count_label, demo_success_count)
                update_task_status(text_labels=text_labels, goal_status={"success": False}, prev_goal_status={"success": was_success}, env=env)
                is_success = False
                cumulative_reward = 0.0
                update_reward_ui(reward_label, cumulative_reward)
                teleop_sys.interfaces[arm_teleop_method].reset_state()
                # load_sampling: retry same scene or advance when enough successes collected
                if scene_source == "load_sampling" and valid_sampling_scene_paths:
                    cur_idx = load_sampling_state["index"]
                    if was_success:
                        load_sampling_state["scene_success_count"] += 1
                        print(f"[load_sampling] Scene {cur_idx+1}/{len(valid_sampling_scene_paths)}: success {load_sampling_state['scene_success_count']}/{success_demo_per_sampling}")
                    if load_sampling_state["scene_success_count"] >= success_demo_per_sampling:
                        # Advance to next scene
                        load_sampling_state["index"] += 1
                        load_sampling_state["scene_success_count"] = 0
                        if load_sampling_state["index"] >= len(valid_sampling_scene_paths):
                            load_sampling_state["done"] = True
                            print(f"[load_sampling] All {len(valid_sampling_scene_paths)} scenes done. Save and exit.")
                        else:
                            next_path = valid_sampling_scene_paths[load_sampling_state["index"]]
                            next_scene_json = load_json_with_absolute_usd_paths(next_path)
                            # Disable recording: this reset just sets up the next scene, not an episode
                            _was_rec = env.is_recording
                            env.is_recording = False
                            env.reset()
                            env.is_recording = _was_rec
                            restore_scene_object_poses(env, next_scene_json)
                            # apply_joint_friction_scale(env)
                            teleop_sys.reset()
                            print(f"[load_sampling] Scene {load_sampling_state['index']+1}/{len(valid_sampling_scene_paths)}: {os.path.basename(next_path)} (need {success_demo_per_sampling} success(es))")
                    else:
                        # Retry same scene - restore poses after env.reset() above
                        cur_path = valid_sampling_scene_paths[cur_idx]
                        cur_scene_json = load_json_with_absolute_usd_paths(cur_path)
                        restore_scene_object_poses(env, cur_scene_json)
                        # apply_joint_friction_scale(env)
                        remaining = success_demo_per_sampling - load_sampling_state["scene_success_count"]
                        print(f"[load_sampling] Retrying scene {cur_idx+1}/{len(valid_sampling_scene_paths)} ({remaining} success(es) remaining)")
            prev_reset_button = reset_button
                
        
        return prev_reset_button, is_success, demo_success_count, cumulative_reward

    # Reset the env one final time before starting teleop
    env.reset()
    # Restore correct object poses (env.reset() uses USD defaults, not JSON settled poses)
    restore_scene_object_poses(env, og_scene_json)

    # # TEMPORARY: disable gravity for all links of wooden_organizer_with_drawer_vmufpr_13
    # _no_gravity_obj_name = "wooden_organizer_with_drawer_vmufpr_13"
    # _no_gravity_obj = env.scene.object_registry("name", _no_gravity_obj_name)
    # if _no_gravity_obj is not None:
    #     # _drawer_link_names = ["drawer_1_link", "drawer_2_link", "drawer_3_link"]
    #     # for link_name in _drawer_link_names:
    #     for link_name in _no_gravity_obj.links:
    #         _no_gravity_obj.links[link_name].prim.GetAttribute("physxRigidBody:disableGravity").Set(True)
    #         print(f"[TEMP] Disabled gravity for '{_no_gravity_obj_name}/{link_name}'")
    # else:
    #     print(f"[TEMP] Object '{_no_gravity_obj_name}' not found in scene, skipping gravity disable")

    # calibrate oculus
    if arm_teleop_method == "oculus":
        print("Align the controller in the forward direction of the robot and press the joystick to calibrate")
        press_joystick = False
        while True:
            raw_action = teleop_sys.interfaces['oculus'].get_action(teleop_sys.get_obs())
            buttons = raw_action.extra['buttons']
            if buttons['RJ'] and not press_joystick:
                press_joystick = True
                print("Joystick pressed, calibrating...")
            elif not buttons['RJ'] and press_joystick:
                break
            # Get current robot EEF pose in base frame and command it to stay there
            arm_name = robot.arm_names[0]  # right arm
            abs_eef_pos, abs_eef_orn = robot.eef_links[arm_name].get_position_orientation()
            base_pos, base_orn = robot.get_position_orientation()
            rel_eef_pos, rel_eef_orn = T.relative_pose_transform(abs_eef_pos, abs_eef_orn, base_pos, base_orn)
            # Build action: [pos(3), axis_angle(3), gripper(1)]
            hold_action = th.zeros(action_dim)
            hold_action[0:3] = rel_eef_pos
            hold_action[3:6] = T.quat2axisangle(rel_eef_orn)
            hold_action[6] = 1.0  # keep gripper open
            env.step(hold_action)
        print("Calibration complete!")


    while True:
        # Stupid @#$!%& isaac doesn't report contact anymore for sleeping objects, so we wake them up here
        for obj in env.scene.objects:
            obj.wake()
        # # TEMPORARY: keep fixed articulated objects (drawers/doors) still each frame
        # for obj in _fixed_articulated_objs:
        #     obj.keep_still()
        action = teleop_sys.get_action(teleop_sys.get_obs())

        if arm_teleop_method == "oculus":
            # TODO: only works for right arm for now
            # action.right is (x, y, z, qx, qy, qz, qw, gripper) - 8 values
            raw_teleop_action = teleop_sys.interfaces['oculus'].get_action(teleop_sys.get_obs())
            action = raw_teleop_action.right
            if action is not None and len(action) == 8:
                action_vec = th.zeros(action_dim)        
                action_vec[0:3] = th.tensor(action[0:3])
                action_quat = th.tensor(action[3:7])
                action_aa = T.quat2axisangle(action_quat)
                action_vec[3:6] = action_aa
                action_vec[6] = action[7]
            else:
                print(f"action is None or length is not 8: {action}")
                action_vec = th.zeros(action_dim)
        else:
            action_vec = th.zeros(action_dim)
            arm_action = action.right if hasattr(action, 'right') else action
            action_vec[action_start_end_idxs[0]:action_start_end_idxs[1]] = th.tensor(arm_action) if not isinstance(arm_action, th.Tensor) else arm_action


        obs, reward, done, truncated, info = env.step(action_vec)
        
        # Update cumulative reward and display
        cumulative_reward += reward
        update_reward_ui(reward_label, cumulative_reward)

        # Update status
        status = {
            "success": env.task.success,
        }

        # Track demo successes
        if status["success"] and not prev_status["success"]:
            is_success = True
            update_task_status(text_labels=text_labels, goal_status=status, prev_goal_status=prev_status, env=env)
            print(f"Demo success #{demo_success_count} achieved! Cumulative reward: {cumulative_reward:.3f}")
        elif not status["success"] and prev_status["success"]:
            is_success = False
            update_task_status(text_labels=text_labels, goal_status=status, prev_goal_status=prev_status, env=env)
            print("Goal condition broken - success reverted to False")

        prev_status = status

        # TODO: Force 30FPS frame rate
        prev_in_hand_status = update_in_hand_status(
            robot,
            vis_mats,
            prev_in_hand_status,
        )
        prev_grasp_status = update_grasp_status(
            robot,
            eef_cylinder_geoms,
            prev_grasp_status,
        )

        # Check for reset
        prev_reset_button, is_success, demo_success_count, cumulative_reward = check_reset(teleop_sys, prev_reset_button, is_success, demo_success_count, cumulative_reward, env)

        # Auto-terminate if deterministic schedule is complete
        if det_schedule is not None and det_schedule.get("done", False):
            print(f"\n[Deterministic Schedule] All {det_schedule['n_samples']} configurations completed successfully!")
            print(f"Total successful demos: {demo_success_count}")
            if env.is_recording:
                env.save_data()
            break

        # Auto-terminate if load_sampling: all valid scenes collected into one HDF5
        if scene_source == "load_sampling" and load_sampling_state.get("done", False):
            total_demos = len(valid_sampling_scene_paths) * success_demo_per_sampling
            print(f"\n[load_sampling] Collected {total_demos} trajectories ({success_demo_per_sampling} per scene x {len(valid_sampling_scene_paths)} scenes) into {hdf5_path}")
            if env.is_recording:
                env.save_data()
            break

        # l_button, r_button = teleop_sys.interfaces["spacemouse"].raw_data.buttons
        # if l_button and not prev_l_button:
        #     # Reset!
        #     env.reset()
        #     if is_success:
        #         demo_success_count += 1
        #     update_demo_success_counter(demo_count_label, demo_success_count)
        #     is_success = False
        #     update_task_status(text_labels=text_labels, goal_status={"success": False}, prev_goal_status={"success": True}, env=env)
        # prev_l_button = l_button

    # Write load_sampling metadata sidecar so steps 15-17 can identify per-scene episode groups
    if scene_source == "load_sampling" and valid_sampling_scene_paths:
        sidecar_path = hdf5_path.replace(".hdf5", "_meta.json")
        with open(sidecar_path, "w") as f:
            json.dump({
                "valid_sampling_scene_paths": valid_sampling_scene_paths,
                "success_demo_per_sampling": success_demo_per_sampling,
            }, f, indent=2)
        print(f"[load_sampling] Wrote metadata sidecar: {sidecar_path}")

    og.shutdown()

    end_stage(cfg, success=True)


def end_stage(cfg, success=False, additional_info: dict = None):
    """
    Function that is run at the end of every stage to save the stage config and additional info.

    # TODO: should this be a general function that can be used for all stages? or should we have a separate function for each stage?
    """
    save_dir = cfg.s14_teleop.out_dir
    stage_cfg = OmegaConf.to_object(cfg.s14_teleop)
    stage_cfg['success'] = success
    if additional_info is not None:
        stage_cfg.update(additional_info)
    dump_json(stage_cfg, f"{save_dir}/stage_info.json")


if __name__ == "__main__":
    main()

