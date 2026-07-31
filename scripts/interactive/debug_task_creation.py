# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Interactive task debugging script for OmniGibson.

Loads:
- A task config from scripts/cfg/task/<task_config_name>.yaml
- A scene state from assets/scenes/<scene_name>/<scene_name>_scene_state_latest.json

Then creates the OmniGibson environment and drops into IPython.
"""

import argparse
import os
from pathlib import Path

import omnigibson as og
from omnigibson.utils.config_utils import parse_config

from simfoundry import ASSET_DIR, CFG_DIR, import_og_dependencies
from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an OmniGibson env with a task + scene and open IPython for debugging."
    )
    parser.add_argument(
        "--task_config_name",
        type=str,
        required=True,
        help="Task config name in scripts/cfg/task (with or without .yaml).",
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        required=True,
        help="Scene name in assets/scenes. Expects <scene_name>_scene_state_latest.json.",
    )
    parser.add_argument(
        "--use_floor_plane",
        action="store_true",
        help="Enable floor plane in scene.",
    )
    parser.add_argument(
        "--floor_plane_visible",
        action="store_true",
        help="Show floor plane (only used if --use_floor_plane is set).",
    )
    parser.add_argument(
        "--action_freq",
        type=int,
        default=15,
        help="Action/render frequency for environment.",
    )
    parser.add_argument(
        "--physics_freq",
        type=int,
        default=120,
        help="Physics frequency for environment.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional override for task termination max_steps.",
    )
    parser.add_argument(
        "--grasping_mode",
        type=str,
        default="physical",
        choices=["physical", "assisted", "sticky", "none"],
        help="Override robot grasping mode in the loaded scene json.",
    )
    return parser.parse_args()


def _normalize_task_config_name(task_config_name: str) -> str:
    return task_config_name if task_config_name.endswith(".yaml") else f"{task_config_name}.yaml"


def _resolve_task_config_path(task_config_name: str) -> str:
    task_cfg_name = _normalize_task_config_name(task_config_name)
    task_cfg_path = f"{CFG_DIR}/task/{task_cfg_name}"
    if not os.path.exists(task_cfg_path):
        raise FileNotFoundError(f"Task config not found: {task_cfg_path}")
    return task_cfg_path


def _resolve_scene_json_path(scene_name: str) -> str:
    scene_json_path = f"{ASSET_DIR}/scenes/{scene_name}/{scene_name}_scene_state_latest.json"
    if not os.path.exists(scene_json_path):
        raise FileNotFoundError(f"Scene json not found: {scene_json_path}")
    return scene_json_path


def _override_robot_grasping_mode(scene_json_dict, grasping_mode: str):
    objects_info = scene_json_dict.get("objects_info", {})
    init_info = objects_info.get("init_info", {})
    for obj_name, obj_info in init_info.items():
        if not obj_name.startswith("robot"):
            continue
        obj_args = obj_info.get("args", {})
        if "grasping_mode" in obj_args and obj_args["grasping_mode"] != grasping_mode:
            print(
                f"Changing grasping mode for {obj_name} from "
                f"'{obj_args['grasping_mode']}' to '{grasping_mode}'"
            )
            obj_args["grasping_mode"] = grasping_mode


def main():
    args = parse_args()

    # Needed so custom tasks (e.g., PickPlaceTask) can be instantiated by OG.
    import_og_dependencies()

    # Keep behavior consistent with existing scripts that expect cfg-relative paths.
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_dir = os.path.join(scripts_dir, "..", "cfg")
    os.chdir(cfg_dir)

    task_cfg_path = _resolve_task_config_path(args.task_config_name)
    scene_json_path = _resolve_scene_json_path(args.scene_name)

    print(f"Loading task config: {task_cfg_path}")
    task_cfg = parse_config(task_cfg_path)["og_task_config"]
    if args.max_steps is not None:
        task_cfg["termination_config"]["max_steps"] = args.max_steps

    print(f"Loading scene json: {scene_json_path}")
    scene_json = load_json_with_absolute_usd_paths(scene_json_path)
    _override_robot_grasping_mode(scene_json, args.grasping_mode)

    scene_cfg = {
        "type": "Scene",
        "scene_file": scene_json,
        "use_floor_plane": args.use_floor_plane,
        "floor_plane_visible": args.use_floor_plane and args.floor_plane_visible,
        "use_skybox": True,
        "include_robots": True,
    }
    env_cfg = {
        "action_frequency": args.action_freq,
        "rendering_frequency": args.action_freq,
        "physics_frequency": args.physics_freq,
    }

    og_cfg = dict(env=env_cfg, scene=scene_cfg, task=task_cfg)

    print("Creating environment...")
    env = og.Environment(configs=og_cfg)
    obs, info = env.reset()

    assert len(env.robots) >= 1, "Expected at least one robot in scene."
    robot = env.robots[0]

    print("\nEnvironment ready.")
    print(f"Task: {env.task.__class__.__name__}")
    print(f"Scene: {args.scene_name}")
    print(f"Robot count: {len(env.robots)}")
    print(f"First robot: {robot.name}")
    print("Launching IPython...")
    print("Available vars: env, obs, info, robot, scene, task, og")

    scene = env.scene
    task = env.task
    from IPython import embed

    try:
        embed()
    finally:
        print("Shutting down OmniGibson...")
        og.shutdown()


if __name__ == "__main__":
    main()

