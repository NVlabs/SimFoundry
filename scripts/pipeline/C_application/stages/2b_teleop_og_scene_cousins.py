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
from omnigibson.objects import DatasetObject, USDObject
from omnigibson.robots import REGISTERED_ROBOTS, BaseRobot, LocomotionRobot, MobileManipulationRobot
from omnigibson.controllers import InverseKinematicsController, OperationalSpaceController
import omnigibson.utils.transform_utils as T
from omnigibson.utils.ui_utils import choose_from_options, KeyboardEventHandler
from omnigibson.utils.config_utils import parse_config
from omnigibson.envs import HDF5CollectionWrapper
from pathlib import Path
import torch as th
import json
import subprocess
import hydra
from omegaconf import OmegaConf
from simfoundry import import_og_dependencies, CFG_DIR as SIMFOUNDRY_CFG_DIR
from simfoundry.utils.processing_utils import dump_json
from omnigibson.examples.objects.import_custom_object import import_custom_object
from simfoundry.utils.og_utils import (
    apply_teleop_omnigibson_macros,
    set_obj_materials,
    setup_task_status_ui,
    update_task_status,
    update_demo_success_counter,
    setup_robot_visualizers,
    update_in_hand_status,
    update_grasp_status,
)
import os
from datetime import datetime
import signal

# Needed so custom tasks can be instantiated properly
import_og_dependencies()

from simfoundry import CFG_DIR


### At the start of every script, we cd into the scripts/config directory
scripts_dir = os.path.dirname(os.path.abspath(__file__))
cfg_dir = CFG_DIR
os.chdir(cfg_dir)


TELEOP_METHOD = {
    "keyboard": "Keyboard (default)",
    "spacemouse": "SpaceMouse",
}

def parse_cousin_from_path(usd_path):
    """
    usd_path = ../../deps/BEHAVIOR-1K/datasets/{DATASET_NAME}/objects/beige_cylindrical_object_cousin_012_v3

    category should be something like "brown_teapot_cousin_003_v3"
    model    should be something like "wuujbj"
    """
    parts = Path(usd_path).parts
    category = parts[-1]                                             # brown_teapot_cousin_003_v3
    model = next(p.name for p in Path(usd_path).iterdir() if p.is_dir())   # wuujbj
    return category, model

def find_object_by_prefix(scene, prefix: str):
    matches = [
        obj for obj in scene.objects
        if obj.name == prefix or obj.name.startswith(prefix + "_")
    ]

    if len(matches) == 0:
        print(f"[DEBUG] Nothing found to swap for prefix={prefix}")
        return None

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple objects found for prefix {prefix}: "
            f"{[o.name for o in matches]}"
        )

    return matches[0]

@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    sim_dir = cfg.s11_sim.out_dir
    physics_dir = cfg.s12_physics.out_dir
    og_dir = cfg.s14_og.out_dir
    out_dir = cfg.s14_teleop.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load physics objects info
    og_scene_json = os.path.join(og_dir, "reconstructed_og_scene.json")
    scene_objects_info_fpath = f"{sim_dir}/scene_objects_info.json"
    obj_frictions = dict()
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
        "use_floor_plane": True,
        # TODO: Change once GS is working
        "floor_plane_visible": True,
        "use_skybox": True,
        "include_robots": True,
    }
    # # Add the robot we want to load
    # robot_name = cfg.s14_og.robot
    # robot_cfg = {
    #     "type": robot_name,
    #     "obs_modalities": ["rgb"],
    #     "action_normalize": False,
    #     "grasping_mode": "assisted",
    # }
    # arms = ["left", "right"] if robot_name == "Tiago" else ["0"]
    # robot_cfg["controller_config"] = {}
    # for arm in arms:
    #     robot_cfg["controller_config"][f"arm_{arm}"] = {
    #         "name": "InverseKinematicsController",
    #         "command_input_limits": None,
    #     }
    #     robot_cfg["controller_config"][f"gripper_{arm}"] = {
    #         "name": "MultiFingerGripperController",
    #         "command_input_limits": (0.0, 1.0),
    #         "mode": "smooth",
    #     }
    task_name = cfg.s14_teleop.task_name
    og_task_cfg_path = f"{SIMFOUNDRY_CFG_DIR}/task/{task_name}.yaml"
    task_cfg = parse_config(og_task_cfg_path)["og_task_config"]
    task_cfg["termination_config"]["max_steps"] = cfg.s14_teleop.max_steps

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
    env.reset()

    objs = {name: env.scene.object_registry("name", name) for name in obj_frictions.keys()}
    objs = {k: v for k, v in objs.items() if v is not None}

    # Set object materials
    with og.sim.stopped():
        obj_materials = set_obj_materials(objs=objs, obj_frictions=obj_frictions)

    assert len(env.robots) == 1, "Teleop only supports controlling a single robot for now!"
    robot = env.robots[0]
    assert isinstance(robot.controllers[f"arm_{robot.default_arm}"], InverseKinematicsController),"Teleop only support controlling with IK Controller!"

    # Wrap in data environment
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    hdf5_path = f"{out_dir}/{task_name}_{now_str}.hdf5"
    env = HDF5CollectionWrapper(
        env=env,
        output_path=hdf5_path,
        **OmegaConf.to_container(cfg.s14_teleop.data_env_kwargs, resolve=True),
    )
    env.is_recording = False
    env.reset()
    env.is_recording = cfg.s14_teleop.save_data

    # ================================
    # HOT SWAP COUSIN
    # ================================
    print("\n=== Objects currently in scene ===")
    for obj in env.scene.objects:
        print(f"name = {obj.name}, category = {obj.category}")
    print("=================================\n")

    # MUST initialize keyboard first
    KeyboardEventHandler.initialize()

    # ================================
    # LOAD SWAP COMBINATIONS
    # ================================
    with open(f"{cfg.generate_cousins_combination.out_dir}/combinations.json", "r") as f:
        swap_combinations = json.load(f)
    assert len(swap_combinations) > 0

    DATASET_NAME = "custom-assets"

    # choose the object to swap
    # swap_obj_name = "masked_object_iter4"
    # swap_obj = env.scene.object_registry("name", swap_obj_name)
    # assert swap_obj is not None, f"{swap_obj_name} not found in scene"

    # choose the cousin
    curr_cousin_idx = 0  # closure state
    imported_models = set()  # cache: avoid re-importing
    
    pending_swap = False
    is_swapping = False

    def hot_swap_cousin():
        nonlocal curr_cousin_idx, is_swapping #, swap_obj

        if is_swapping:
            return
        is_swapping = True

        # cousins = [
        #     dict(
        #         category="brown_teapot_cousin_003_v3",
        #         model="wuujbj",
        #     ),
        #     dict(
        #         category="brown_teapot_cousin_001_v1",
        #         model="vosrsj", 
        #     ),
        # ]

        curr_cousin_idx = (curr_cousin_idx + 1) % len(swap_combinations)
        combo = swap_combinations[curr_cousin_idx]

        print(f"[HOT SWAP] combo idx = {curr_cousin_idx}")

        # ----------------------------
        # 1. stop recording + teleop
        # ----------------------------
        was_recording = env.is_recording
        env.is_recording = False
        # teleop_sys.stop()

        # ============================
        # Stop sim during cousin swapping
        # ============================
        with og.sim.stopped():
            # old_obj = swap_obj
            # pos, orn = old_obj.get_position_orientation()

            # og.sim._physics_sim_view = None
            # og.sim._tensor_api = None

            # env.scene.remove_object(old_obj)
            # og.sim.stage.RemovePrim(f"/World/{old_obj.name}")
            # og.sim.stage.Flatten()

            # swap_obj = DatasetObject(
            #     name=f"{swap_obj_name}_{cousin['model']}",
            #     category=cousin["category"],
            #     model=cousin["model"],
            #     dataset_name=DATASET_NAME,
            # )

            # env.scene.add_object(swap_obj)
            # swap_obj.set_position_orientation(pos, orn)

            # swap objects in combo list
            for obj_name, cousin_path in combo.items():
                '''
                e.g.
                obj_name   : masked_object_iter1
                cousin_path: masked_object_iter1/visual/cousin_012_v3_transparent.png
                cousin_name: cousin_012_v3
                '''
                cousin_name = "_".join(Path(cousin_path).stem.split("_")[:3])
                usd_path = f"../../deps/BEHAVIOR-1K/datasets/{DATASET_NAME}/objects"
                folder = next(
                    p for p in Path(usd_path).iterdir()
                    if p.is_dir() and p.name.endswith(cousin_name)
                )
                # usd_path = usd_path + "/" + str(folder)
                print(f"[DEBUG] folder: {folder}")

                old_obj = find_object_by_prefix(env.scene, obj_name)
                if old_obj is None:
                    print(f"[WARN] {obj_name} not found, skipping")
                    continue

                pos, orn = old_obj.get_position_orientation()
                old_category = old_obj.category
                env.scene.remove_object(old_obj)
                # og.sim.stage.RemovePrim(f"/World/{old_obj.name}")
                # og.sim.stage.Flatten()

                
                _, model = parse_cousin_from_path(str(folder))
                usd_path = os.path.abspath(str(folder / model / "usd" / f"{model}.usd"))

                print(f"  └─ swapping {obj_name} → {model}")

                new_obj = USDObject(
                    name=f"{obj_name}_{model}",
                    usd_path=usd_path,
                    # name=obj_name,
                    category=old_category,
                    # model=model,
                    dataset_name=DATASET_NAME,
                )

                env.scene.add_object(new_obj)
                new_obj.set_position_orientation(pos, orn)

            og.sim.initialize_physics()

        env.scene.reset(hard=False)
        env.task.update_scene()
        env.scene.update_initial_file()
        env.is_recording = was_recording

        is_swapping = False

    def request_hot_swap():
        nonlocal pending_swap
        pending_swap = True

    # press H for hot-swap
    KeyboardEventHandler.add_keyboard_callback(
        lazy.carb.input.KeyboardInput.H,
        request_hot_swap
    )

    # Register keyboard callback for reset without saving
    is_success = False
    def reset_without_saving():
        """Reset the environment without saving the current demonstration"""
        was_recording = env.is_recording
        env.is_recording = False
        print("Resetting environment without saving...")
        env.reset()
        is_success = False
        env.is_recording = was_recording
        print("Environment reset complete.")

    # Register 'R' key to reset without saving
    # from omnigibson.utils.ui_utils import KeyboardEventHandler
    # KeyboardEventHandler.initialize()
    KeyboardEventHandler.add_keyboard_callback(
        lazy.carb.input.KeyboardInput.R,
        reset_without_saving
    )

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

    arm_teleop_method = cfg.s14_teleop.device #choose_from_options(options=TELEOP_METHOD, name="robot arm teleop method")
    assert arm_teleop_method == "spacemouse", "Must use spacemouse for teleop!"

    if isinstance(robot, (LocomotionRobot, MobileManipulationRobot)):
        base_teleop_method = choose_from_options(options=TELEOP_METHOD, name="robot base teleop method")
    else:
        base_teleop_method = "keyboard"  # Dummy value since FrankaPanda does not have a base
    # Generate teleop config
    teleop_config.arm_left_controller = arm_teleop_method
    teleop_config.arm_right_controller = arm_teleop_method
    teleop_config.base_controller = base_teleop_method
    teleop_config.interface_kwargs["keyboard"] = {"arm_speed_scaledown": 0.04}
    teleop_config.interface_kwargs["spacemouse"] = {"arm_speed_scaledown": 0.04}

    # Initialize teleoperation system
    teleop_sys = TeleopSystem(config=teleop_config, robot=robot, show_control_marker=True)
    teleop_sys.start()

    # Prevent swapping between controllers
    teleop_sys.interfaces["spacemouse"].controllable_robot_parts = ["right"]

    # Left spacemouse button is grasp, so make sure right button maps to save episode / env reset

    # Modifies OG native handler to make sure we save the data before shutting down
    def data_shutdown_handler(*args, **kwargs):
        # Save data
        if env.is_recording:
            print("Saving data before shutting down...")
            env.save_data()
            print("Successfully saved data!")

        return shutdown_handler(*args, **kwargs)

    # Something somewhere disables the default SIGINT handler, so we need to re-enable it
    signal.signal(signal.SIGINT, data_shutdown_handler)

    # Setup robot visualizers
    vis_elements = setup_robot_visualizers(robot=robot, scene=env.scene)
    eef_cylinder_geoms = vis_elements["eef_cylinder_geoms"]
    vis_mats = vis_elements["vis_mats"]
    prev_grasp_status = {arm: False for arm in robot.arm_names}
    prev_in_hand_status = {arm: False for arm in robot.arm_names}

    # Setup UI
    overlay_window, text_labels, instance_id_label, demo_count_label, prev_status = setup_task_status_ui(
        task_name=task_name,
        env=env,
        instance_id=cfg.s14_teleop.instance_id,
    )

    # Initialize demo success counter
    demo_success_count = 0

    # Print camera positions
    def print_camera_positions():
        print("\n=== Camera Positions ===")
        if env.external_sensors is not None:
            for cam_name, sensor in env.external_sensors.items():
                pos, orn = sensor.get_position_orientation()
                print(f"  [{cam_name}] position={pos.tolist()}, orientation={orn.tolist()}")
        viewer_pos, viewer_orn = og.sim.viewer_camera.get_position_orientation()
        print(f"  [viewer_camera] position={viewer_pos.tolist()}, orientation={viewer_orn.tolist()}")
        print("========================\n")

    print_camera_positions()

    # Register 'C' key to print camera positions on demand
    KeyboardEventHandler.add_keyboard_callback(
        lazy.carb.input.KeyboardInput.C,
        print_camera_positions,
    )

    print(f"Starting teleop!\n\n")
    print("Press 'C' to print camera positions at any time.\n")
    prev_l_button = 0
    fps = env.env_config["action_frequency"]
    step_count = 0

    while True:
        if pending_swap:
            print("SWAPPING COUSIN...")
            hot_swap_cousin()
            pending_swap = False

        # Stupid @#$!%& isaac doesn't report contact anymore for sleeping objects, so we wake them up here
        for obj in env.scene.objects:
            obj.wake()
        action = teleop_sys.get_action(teleop_sys.get_obs())
        env.step(action)

        # Print camera positions every 300 steps (~10 seconds at 30fps)
        step_count += 1
        if step_count % 300 == 0:
            print_camera_positions()

        # Update status
        status = {
            "success": env.task.success,
        }

        # Track demo successes
        if status["success"] and not prev_status["success"]:
            is_success = True
            update_task_status(text_labels=text_labels, goal_status=status, prev_goal_status=prev_status, env=env)
            print(f"Demo success #{demo_success_count} achieved!")

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
        l_button, r_button = teleop_sys.interfaces["spacemouse"].raw_data.buttons
        if l_button and not prev_l_button:
            # Reset!
            env.reset()
            if is_success:
                demo_success_count += 1
            update_demo_success_counter(demo_count_label, demo_success_count)
            is_success = False
            update_task_status(text_labels=text_labels, goal_status={"success": False}, prev_goal_status={"success": True}, env=env)
        prev_l_button = l_button

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
