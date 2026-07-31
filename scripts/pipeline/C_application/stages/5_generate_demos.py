# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from simfoundry env

Generate synthetic demonstrations by replaying object-centric waypoints
in randomized scenes using CuRobo motion planning.

Requires installing:
- BEHAVIOR-1K, see https://github.com/StanfordVL/BEHAVIOR-1K
"""

import os
import sys
import json
import random
import subprocess as sp
from pathlib import Path
from copy import deepcopy

from glob import glob

import yaml
import hydra
import numpy as np
import torch as th
import omnigibson as og
from omnigibson.macros import gm
import omnigibson.lazy as lazy
from omnigibson.robots import BaseRobot
from omnigibson.macros import gm
from omnigibson.envs import HDF5CollectionWrapper
from omnigibson.object_states import Open
import omnigibson.utils.transform_utils as T
import omnigibson.utils.transform_utils_np as NT
from omnigibson.controllers.ik_controller import _compute_ik_qpos_numpy

from simfoundry import import_og_dependencies, REPO_DIR
from simfoundry.utils.og_utils import apply_teleop_omnigibson_macros, setup_wrist_camera_viewport
from simfoundry.utils.scene_sampling_utils import (
    get_task_object_names,
    get_fixed_base_names,
    capture_settled_poses,
    compute_single_group_pose,
    compute_group_aabb,
    filter_relationships_for_group,
    reload_task_objects_with_poses,
    extract_relationships,
    extract_relationships_for_group,
    relationships_to_set,
    build_groups_from_relationships,
)

gm.ENABLE_CCD = True

# Needed so custom tasks can be instantiated properly
import_og_dependencies()

from simfoundry.utils.data_gen_utils import (
    init_curobo,
    plan_to_pose,
    transform_waypoints_to_robot_frame,
    load_waypoints_from_hdf5,
    solve_ik_step,
    compute_linear_freespace_trajectory,
    apply_eef_z_offset,
    apply_tcp_action_noise,
)

# Configure viewer
gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128

DEFAULT_MOTION_PLANNER_ATTACHED_OBJECT_SCALE = 0.8
from simfoundry import CFG_DIR

# At the start of every script, we cd into the scripts/config directory
scripts_dir = os.path.dirname(os.path.abspath(__file__))
cfg_dir = CFG_DIR
os.chdir(cfg_dir)

# Known top-level subdirectories within the SimFoundry repo, used to infer the old repo root
# from absolute paths baked into serialized configs. _detect_old_repo_root returns on the
# first entry that matches anywhere in the path, so ordering matters: the package directory
# is listed last because a checkout is commonly named `simfoundry` too, and matching that
# first would resolve the repo's PARENT as the old root. 'digital_cousins/' is retained so
# configs serialized before the package rename can still be rebased.
_SIMFOUNDRY_REPO_SUBDIRS = (
    'assets/', 'Data/', 'checkpoints/', 'scripts/', 'deps/', 'digital_cousins/',
)


class NoisyActionHDF5Wrapper(HDF5CollectionWrapper):
    """HDF5CollectionWrapper that executes noised actions but records clean ones.

    Overrides DataWrapper.step() to decouple the action sent to physics
    (with TCP-bounded noise) from the action saved to the HDF5 dataset
    (clean, unnoised).  All other wrapper behaviour (reset, flush,
    checkpointing, etc.) is inherited unchanged.
    """

    def __init__(self, *args, noise_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._noise_fn = noise_fn
        self.noise_enabled = True

    def step(self, action, n_render_iterations=1):
        if self._noise_fn is not None and self.noise_enabled:
            noised_action = self._noise_fn(action)
        else:
            noised_action = action
        next_obs, reward, terminated, truncated, info = self.env.step(
            noised_action, n_render_iterations=n_render_iterations,
        )
        self.step_count += 1
        self._record_step_trajectory(action, next_obs, reward, terminated, truncated, info)
        return next_obs, reward, terminated, truncated, info


def _detect_old_repo_root(obj):
    """Walk a nested dict/list to find the first absolute .usd path and infer the old repo root."""
    if isinstance(obj, dict):
        for v in obj.values():
            result = _detect_old_repo_root(v)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for v in obj:
            result = _detect_old_repo_root(v)
            if result is not None:
                return result
    elif isinstance(obj, str) and obj.startswith('/') and obj.lower().endswith('.usd'):
        for subdir in _SIMFOUNDRY_REPO_SUBDIRS:
            marker = '/' + subdir
            idx = obj.find(marker)
            if idx >= 0:
                return obj[:idx]
    return None


def _rebase_strings(obj, old_root, new_root):
    """Recursively replace old_root prefix with new_root in all string values."""
    count = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and v.startswith(old_root):
                obj[k] = new_root + v[len(old_root):]
                count += 1
            else:
                count += _rebase_strings(v, old_root, new_root)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str) and v.startswith(old_root):
                obj[i] = new_root + v[len(old_root):]
                count += 1
            else:
                count += _rebase_strings(v, old_root, new_root)
    return count


def rebase_config_paths(config, new_repo_root):
    """
    Make a serialized env config portable by rebasing absolute paths.

    Detects the repo root that was baked into the config (from the first .usd path found)
    and replaces it with new_repo_root in every string value throughout the nested structure.

    Returns the number of strings that were updated.
    """
    old_root = _detect_old_repo_root(config)
    if old_root is None:
        return 0
    if old_root == new_repo_root:
        return 0
    count = _rebase_strings(config, old_root, new_repo_root)
    print(f"  Rebased {count} absolute paths: {old_root} -> {new_repo_root}")
    return count


def solve_ik_step_og(
    robot,
    target_pos_robot: np.ndarray,
    target_quat_robot: np.ndarray,
    gripper_cmd: float
) -> np.ndarray:
    """
    Solve IK for a single waypoint using Jacobian pseudo-inverse.
    
    This is a simpler/faster version of solve_ik_step that uses pure
    Jacobian pseudo-inverse instead of cvxpy QP optimization.
    
    Args:
        robot: Robot object
        target_pos_robot: Target EEF position in robot frame [3]
        target_quat_robot: Target EEF quaternion in robot frame [4]
        gripper_cmd: Gripper command (1.0 = open, 0.0 = close)
        
    Returns:
        Action array for the robot
    """
    from omnigibson.controllers import ControlType
    
    # Get current EEF pose in robot frame
    curr_pos_robot, curr_quat_robot = robot.get_relative_eef_pose(arm="default")
    curr_pos_robot = curr_pos_robot.numpy()
    curr_quat_robot = curr_quat_robot.numpy()
    control_dict = robot.get_control_dict()
    
    # Get arm controller and DOF indices
    arm_controller = robot.controllers[f"arm_{robot.default_arm}"]
    arm_dof_idx = arm_controller.dof_idx
    
    # Get Jacobian and joint state
    j_eef = control_dict[f"eef_{robot.default_arm}_jacobian_relative"][:, arm_dof_idx]
    q = control_dict["joint_position"][arm_dof_idx]
    q_lower_limit = arm_controller._control_limits[ControlType.get_type("position")][0][arm_dof_idx]
    q_upper_limit = arm_controller._control_limits[ControlType.get_type("position")][1][arm_dof_idx]
    
    # Convert current and target orientations to rotation matrices
    curr_mat = NT.quat2mat(curr_quat_robot)
    target_mat = NT.quat2mat(target_quat_robot)
    
    # Solve IK using Jacobian pseudo-inverse
    target_joint_pos = _compute_ik_qpos_numpy(
        q=q,
        j_eef=j_eef,
        ee_pos=curr_pos_robot,
        ee_mat=curr_mat,
        goal_pos=target_pos_robot,
        goal_ori_mat=target_mat,
        q_lower_limit=q_lower_limit,
        q_upper_limit=q_upper_limit,
    )
    
    # Create full joint position vector
    full_joint_pos = control_dict["joint_position"].copy()
    full_joint_pos[arm_dof_idx] = np.array(target_joint_pos)
    
    # Convert to action using robot's q_to_action method
    action = robot.q_to_action(th.from_numpy(full_joint_pos), controller_name=f'arm_{robot.default_arm}')
    
    # Append gripper action
    action = th.cat([action, th.tensor([gripper_cmd])], dim=0)
    
    return action.cpu().numpy() if isinstance(action, th.Tensor) else action


# # @jit(nopython=True)
# def _compute_ik_qpos_numpy(
#     q,
#     j_eef,
#     ee_pos,
#     ee_mat,
#     goal_pos,
#     goal_ori_mat,
#     q_lower_limit,
#     q_upper_limit,
#     solver=None,
# ):
#     # Compute the pose error. Note that this is computed NOT in the EEF frame but still
#     # in the base frame.
#     pos_err = goal_pos - ee_pos
#     ori_err = NT.orientation_error(goal_ori_mat, ee_mat).astype(np.float32)
#     err = np.concatenate((pos_err, ori_err))

#     # Use the jacobian to compute a local approximation
#     if solver is not None:
#         j_eef_pinv = solver.compute(j_eef)
#     else:
#         j_eef_pinv = np.linalg.pinv(j_eef)
#     delta_j = j_eef_pinv @ err
#     target_joint_pos = q + delta_j

#     # Clip values to be within the joint limits
#     return target_joint_pos.clip(q_lower_limit, q_upper_limit)



# def _compute_ik_qpos_torch(
#     q: th.Tensor,
#     j_eef: th.Tensor,
#     ee_pos: th.Tensor,
#     ee_mat: th.Tensor,
#     goal_pos: th.Tensor,
#     goal_ori_mat: th.Tensor,
#     q_lower_limit: th.Tensor,
#     q_upper_limit: th.Tensor,
# ) -> th.Tensor:
#     """
#     Compute target joint positions using Jacobian pseudo-inverse IK.
    
#     This replicates OmniGibson's IK controller logic.
    
#     Args:
#         q: Current joint positions
#         j_eef: End-effector Jacobian [6, n_joints]
#         ee_pos: Current EEF position [3]
#         ee_mat: Current EEF rotation matrix [3, 3]
#         goal_pos: Target EEF position [3]
#         goal_ori_mat: Target EEF rotation matrix [3, 3]
#         q_lower_limit: Joint position lower limits
#         q_upper_limit: Joint position upper limits
        
#     Returns:
#         Target joint positions clipped to joint limits
#     """
#     # Compute the pose error. Note that this is computed NOT in the EEF frame but still
#     # in the base frame.
#     pos_err = goal_pos - ee_pos
#     ori_err = T.orientation_error(goal_ori_mat, ee_mat)
#     err = th.cat([pos_err, ori_err])

#     # Use the jacobian to compute a local approximation
#     j_eef_pinv = th.linalg.pinv(j_eef)
#     delta_j = j_eef_pinv @ err
#     target_joint_pos = q + delta_j

#     # Clip values to be within the joint limits
#     return target_joint_pos.clip(
#         min=q_lower_limit,
#         max=q_upper_limit,
#     )


def check_joint_velocity_exceeded(robot, arm_dof_idx, prev_arm_joint_pos, max_joint_velocity):
    """
    Check if any arm joint's per-step position change exceeds the threshold.

    Args:
        robot: Robot object
        arm_dof_idx: Indices of the arm joints
        prev_arm_joint_pos: Previous arm joint positions (tensor), or None for first step
        max_joint_velocity: Maximum allowed per-step position change (rad/step)

    Returns:
        Tuple of (current_arm_joint_pos, exceeded_bool)
    """
    curr = robot.get_joint_positions()[arm_dof_idx].clone()
    if prev_arm_joint_pos is None:
        return curr, False
    delta = th.abs(curr - prev_arm_joint_pos)
    if th.any(delta > max_joint_velocity):
        worst_joint = delta.argmax().item()
        print(f"  EARLY TERMINATION: Joint velocity exceeded — joint {worst_joint} "
              f"moved {delta[worst_joint].item():.4f} rad/step (limit: {max_joint_velocity})")
        return curr, True
    return curr, False


def resolve_knn_param(value, subtask_type, subtask_idx, default):
    """
    Resolve a KNN parameter that may be a scalar, per-subtask-type dict, or per-subtask-index list.

    Args:
        value: One of:
            - Scalar (int/float): used for all subtasks
            - Dict/DictConfig mapping subtask type strings to values
            - List/ListConfig with one entry per subtask index
        subtask_type: The subtask type string (e.g. "pick", "place", "open", "close",
            or merged like "pick+open"). Used for dict lookups; for merged types the
            last component is used as the key.
        subtask_idx: Integer index of the current subtask. Used for list lookups.
        default: Fallback value if the subtask type is not found in a dict

    Returns:
        Resolved scalar value for this subtask
    """
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return default
    # List-like (but not dict-like): index by subtask position
    if isinstance(value, (list, tuple)):
        return value[subtask_idx]
    # Hydra ListConfig also needs handling
    try:
        from omegaconf import ListConfig
        if isinstance(value, ListConfig):
            return value[subtask_idx]
    except ImportError:
        pass
    # Dict-like: look up by subtask type, falling back to the last component
    # of merged types (e.g. "pick+open" -> "open")
    lookup_key = subtask_type.split('+')[-1] if '+' in subtask_type else subtask_type
    if hasattr(value, '__getitem__'):
        if lookup_key in value:
            return value[lookup_key]
        return default
    return value


def select_subtask_knn(all_waypoints, episode_keys, subtask_idx, scene, k, pos_weight, ori_weight, blacklist_ids=None):
    """
    Select a subtask from the top-K nearest source demos based on reference object pose similarity.

    For the given subtask index, compares the reference object's current pose in the scene
    with its stored pose from each source demo. Returns a randomly chosen subtask from the
    top-K closest matches.

    Args:
        all_waypoints: Dict mapping episode keys to lists of subtask data
        episode_keys: List of available episode keys
        subtask_idx: Index of the subtask to select
        scene: Current OmniGibson scene (for querying live object poses)
        k: Number of nearest neighbors to sample from
        pos_weight: Weight for position L2 distance
        ori_weight: Weight for orientation angular distance (geodesic)
        blacklist_ids: Optional set/list of episode IDs (ints) to exclude for this subtask

    Returns:
        Tuple of (selected_subtask_dict, selected_demo_key) or (None, None) if no valid candidates
    """
    blacklist_keys = set()
    if blacklist_ids:
        blacklist_keys = {f"demo_{eid}" for eid in blacklist_ids}

    candidates = []
    distances = []

    for demo_key in episode_keys:
        if demo_key in blacklist_keys:
            continue
        subtasks = all_waypoints[demo_key]
        if subtask_idx >= len(subtasks):
            continue
        subtask = subtasks[subtask_idx]
        ref_info = subtask['reference_object']
        stored_pos = ref_info.get('pos', None)
        stored_quat = ref_info.get('quat', None)
        if stored_pos is None or stored_quat is None:
            continue

        ref_obj = scene.object_registry("name", ref_info['name'], None)
        if ref_obj is None:
            continue

        curr_pos, curr_quat = ref_obj.get_position_orientation()
        curr_pos = curr_pos.cpu().numpy() if hasattr(curr_pos, 'cpu') else np.asarray(curr_pos)
        curr_quat = curr_quat.cpu().numpy() if hasattr(curr_quat, 'cpu') else np.asarray(curr_quat)
        stored_pos = np.asarray(stored_pos, dtype=np.float64)
        stored_quat = np.asarray(stored_quat, dtype=np.float64)

        pos_dist = np.linalg.norm(curr_pos - stored_pos)
        dot = np.clip(np.abs(np.dot(curr_quat, stored_quat)), 0.0, 1.0)
        ori_dist = 2.0 * np.arccos(dot)

        weighted_dist = pos_weight * pos_dist + ori_weight * ori_dist
        candidates.append((demo_key, subtask))
        distances.append(weighted_dist)

    if not candidates:
        return None, None

    distances = np.asarray(distances)
    k_actual = min(k, len(candidates))
    kth_distance = np.partition(distances, k_actual - 1)[k_actual - 1]
    tied_indices = np.where(distances <= kth_distance)[0]
    selected_idx = random.choice(tied_indices.tolist())
    demo_key, subtask = candidates[selected_idx]
    return subtask, demo_key


def _resolve_generation_datasets(annotation_dir, waypoints_dir, filter_str):
    """
    Resolve (config_path, waypoints_path, stem_or_None) tuples for generation.

    When filter_str is set, finds all waypoint files matching the filter and pairs
    each with its corresponding env_config from stage 15.

    Returns:
        List of (config_path, waypoints_path, stem_or_None) tuples.
    """
    if filter_str:
        waypoint_files = sorted(glob(f"{waypoints_dir}/waypoints_*.hdf5"))
        matched = []
        for wp_path in waypoint_files:
            basename = os.path.basename(wp_path)
            stem = basename.replace("waypoints_", "").replace(".hdf5", "")
            if filter_str not in stem:
                continue
            config_path = Path(annotation_dir) / f"env_config_{stem}.json"
            if not config_path.exists():
                print(f"WARNING: Missing env_config for {stem}: {config_path}, skipping")
                continue
            matched.append((str(config_path), str(wp_path), stem))

        if not matched:
            all_waypoints = [os.path.basename(f) for f in waypoint_files]
            raise FileNotFoundError(
                f"No waypoint files matching filter '{filter_str}' in {waypoints_dir}. "
                f"Available: {all_waypoints}"
            )
        print(f"Filter '{filter_str}' matched {len(matched)} dataset(s):")
        for _, _, stem in matched:
            print(f"  {stem}")
        return matched
    else:
        config_path = Path(annotation_dir) / "env_config.json"
        waypoints_path = Path(waypoints_dir) / "waypoints.hdf5"
        if not config_path.exists(): # might be in a nested directory
            for dir in Path(annotation_dir).iterdir():
                if "load_sampling" in dir.name:
                    continue
                else:
                    if (Path(dir) / "env_config.json").exists():
                        config_path = Path(dir) / "env_config.json"
                        break
        if not config_path.exists():    # still not found
            raise FileNotFoundError(f"Environment config not found: {config_path}. Please run stage 15 first.")
        if not waypoints_path.exists():
            raise FileNotFoundError(f"Waypoints file not found: {waypoints_path}. Please run stage 16 first.")
        print(f"No filter set — using default files:")
        print(f"  Config: {config_path}")
        print(f"  Waypoints: {waypoints_path}")
        return [(str(config_path), str(waypoints_path), None)]


def _prepare_generation_config(config_path, cfg):
    """
    Load and prepare environment config for demo generation.

    Handles path rebasing, task config hot-loading, frequency overrides,
    and robot controller configuration.

    Returns:
        Prepared config dict ready for og.Environment.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    rebase_config_paths(config, REPO_DIR)

    print("\nHot loading task configuration...")
    task_cfg_path = Path(CFG_DIR) / "task" / f"{cfg.task.task_name}.yaml"
    if not task_cfg_path.exists():
        print(f"WARNING: Task config not found at {task_cfg_path}")
        print("Using task config from env_config.json (may be incorrect!)")
    else:
        with open(task_cfg_path, 'r') as f:
            task_yaml = yaml.safe_load(f)
        config['task'] = task_yaml['og_task_config']
        print(f"Successfully loaded task config from: {task_cfg_path}")
        print(f"Task activity: {config['task']['activity_name']}")
        if 'workspace_bounds' in config['task']:
            print(f"[DEBUG] workspace_bounds in task config: {config['task']['workspace_bounds']}")
        else:
            print(f"[DEBUG] workspace_bounds NOT found in task config!")

    config['env']['physics_frequency'] = 120
    config['env']['action_frequency'] = cfg.s17_generate.action_frequency
    config['env']['rendering_frequency'] = cfg.s17_generate.action_frequency

    robot_id = cfg.s15_annotate.robot_id
    scene_file = config['scene']['scene_file']
    assert isinstance(scene_file, dict), "Scene file must be a dictionary"
    objects_info = scene_file['objects_info']
    assert isinstance(objects_info, dict), "Objects info must be a dictionary"
    init_info = objects_info['init_info']
    assert isinstance(init_info, dict), "Init info must be a dictionary"

    grasping_type = cfg.s17_generate.get('grasping_type', None)
    if grasping_type is not None:
        assert grasping_type in ["physical", "assisted"], \
            f"Invalid grasping_type: {grasping_type}. Must be 'physical' or 'assisted'"
        print(f"Overriding robot grasping mode to: {grasping_type}")

    for obj_name, obj_info in init_info.items():
        if not obj_name.startswith('robot'):
            continue
        robot_info = init_info[obj_name]
        assert isinstance(robot_info, dict), "Robot info must be a dictionary"
        if grasping_type is not None:
            robot_info['args']['grasping_mode'] = grasping_type
        else:
            robot_info['args']['grasping_mode'] = "assisted"
        controller_config = robot_info['args']['controller_config']
        assert isinstance(controller_config, dict), "Controller config must be a dictionary"
        is_yam = obj_info['class_name'] == "Yam"
        extra_controller_kwargs = {}
        if is_yam:
            extra_controller_kwargs["open_qpos"] = [-0.045, 0.045]
            extra_controller_kwargs["closed_qpos"] = [0.0, 0.0]
            extra_controller_kwargs["inverted"] = [False, True]
        else:
            orig_gripper_cfg = controller_config.get('gripper_0', {})
            if 'inverted' in orig_gripper_cfg:
                extra_controller_kwargs["inverted"] = orig_gripper_cfg['inverted']

        if obj_name == f'robot{robot_id}':
            controller_config['arm_0'] = {
                "name": "JointController",
                "motor_type": "position",
                "command_input_limits": None,
                "use_delta_commands": False,
                "use_impedances": False,
            }
            controller_config['gripper_0'] = {
                "name": "MultiFingerGripperController",
                "command_input_limits": (0.0, 1.0),
                "mode": "smooth",
                **extra_controller_kwargs,
            }
        else:
            controller_config['arm_0'] = {
                "name": "InverseKinematicsController",
                "command_input_limits": None,
                "mode": "pose_delta_ori",
                "smoothing_filter_size": 5,
            }
            controller_config['gripper_0'] = {
                "name": "MultiFingerGripperController",
                "command_input_limits": (0.0, 1.0),
                "mode": "smooth",
                **extra_controller_kwargs,
            }

    return config


def _run_filter_subprocesses(stems, stage_cfg_key):
    """Spawn a subprocess for each stem, re-invoking this script in single-target mode."""
    script_path = os.path.join(scripts_dir, os.path.basename(__file__))
    base_overrides = [
        arg for arg in sys.argv[1:]
        if not arg.startswith(f'{stage_cfg_key}.filter=')
        and not arg.lstrip('+').startswith(f'{stage_cfg_key}._target_stem=')
    ]

    results = {}
    for stem in stems:
        cmd = [sys.executable, script_path] + base_overrides + [f'++{stage_cfg_key}._target_stem={stem}']
        print(f"\n{'#'*80}")
        print(f"[SUBPROCESS] Processing: {stem}")
        print(f"[SUBPROCESS] Command: {' '.join(cmd)}")
        print(f"{'#'*80}\n")
        result = sp.run(cmd)
        results[stem] = result.returncode
        if result.returncode != 0:
            print(f"\nWARNING: Subprocess for '{stem}' exited with code {result.returncode}")

    n_ok = sum(1 for rc in results.values() if rc == 0)
    print(f"\n{'='*80}")
    print(f"SUBPROCESS SUMMARY: {n_ok}/{len(results)} succeeded")
    for stem, rc in results.items():
        status = "OK" if rc == 0 else f"FAILED (exit code {rc})"
        print(f"  {stem}: {status}")
    print(f"{'='*80}")


def _sort_groups_for_progressive(groups, fixed_base_names, include_fixed_base_groups):
    """Order groups for progressive placement: fixed-base first, then largest."""
    fixed = []
    movable = []
    for g in groups:
        if any(n in fixed_base_names for n in g) and not include_fixed_base_groups:
            fixed.append(g)
        else:
            movable.append(g)
    movable.sort(key=len, reverse=True)
    return fixed + movable


def _resample_valid_scene(env, task_objects, scene_json, groups, settled_poses,
                          original_rel_set, fixed_base_names,
                          xy_range, z_rotation_deg, settle_steps, max_attempts,
                          include_fixed_base_groups=False,
                          per_group_attempts=10,
                          min_group_distance=0.0,
                          max_rejection_samples=50):
    """
    Try up to ``max_attempts`` to find a new scene config whose OnTop relationships
    match the original settled-scene relationships.

    Groups are placed one at a time (progressive sampling). Each group is
    validated individually before moving on to the next. If a group fails after
    ``per_group_attempts``, the entire sequence restarts. Already-placed group
    AABBs are used for collision avoidance when sampling subsequent groups. On
    failure of any group, the scene is rolled back to the last stable checkpoint
    (the state after all previously-successful groups).

    On success, the valid pose state is baked into ``env.scene``'s initial file so that
    subsequent ``env.reset()`` calls restore to the new config.

    On failure (no valid config found within ``max_attempts``), the previous initial
    file state is left untouched (reloads during attempts use bake_initial_file=False),
    but object positions may still be left in a perturbed state. The caller should
    rely on the next ``env.reset()`` to restore to the previously baked state.

    Returns:
        True if a valid config was found and baked; False otherwise.
    """
    ordered_groups = _sort_groups_for_progressive(groups, fixed_base_names, include_fixed_base_groups)
    print(f"  [Resample] Progressive: {len(ordered_groups)} group(s) to place")

    for overall_attempt in range(max_attempts):
        placed_aabbs = []
        placed_poses = {}
        all_groups_ok = True

        for group_idx, group in enumerate(ordered_groups):
            group_placed = False
            group_names = list(group)

            for group_attempt in range(per_group_attempts):
                # Step 1: Sample a collision-free pose for this group
                group_poses, group_aabb = compute_single_group_pose(
                    settled_poses, group, xy_range, z_rotation_deg,
                    fixed_base_names=fixed_base_names,
                    include_fixed_base_groups=include_fixed_base_groups,
                    placed_aabbs=placed_aabbs,
                    min_group_distance=min_group_distance,
                    max_rejection_samples=max_rejection_samples,
                )
                if group_poses is None:
                    print(f"  [Resample] Group {group_idx} ({group_names}): "
                          f"rejection sampling exhausted (attempt {group_attempt + 1}/{per_group_attempts})")
                    continue

                # Step 2: Build full pose dict and reload scene
                # Already-placed groups use their validated (post-settle) poses,
                # unplaced groups use settled_poses, current group uses candidate.
                all_poses = dict(settled_poses)
                all_poses.update(placed_poses)
                all_poses.update(group_poses)

                reload_task_objects_with_poses(
                    env=env,
                    task_objects=task_objects,
                    scene_json=scene_json,
                    new_poses=all_poses,
                    settle_steps=settle_steps,
                    fixed_base_names=fixed_base_names,
                    bake_initial_file=False,
                )

                # Step 3: Validate relationships for this group
                expected_group_rels = filter_relationships_for_group(original_rel_set, group_names)
                actual_group_rels = relationships_to_set(
                    extract_relationships_for_group(task_objects, group_names)
                )

                if actual_group_rels != expected_group_rels:
                    missing = list(expected_group_rels - actual_group_rels)
                    extra = list(actual_group_rels - expected_group_rels)
                    print(f"  [Resample] Group {group_idx} attempt {group_attempt + 1}/{per_group_attempts} "
                          f"invalid (missing={missing}, extra={extra})")
                    continue

                # Step 4: Also verify previously-placed groups still hold
                prev_ok = True
                for prev_idx, prev_group in enumerate(ordered_groups[:group_idx]):
                    prev_names = list(prev_group)
                    expected_prev = filter_relationships_for_group(original_rel_set, prev_names)
                    actual_prev = relationships_to_set(
                        extract_relationships_for_group(task_objects, prev_names)
                    )
                    if actual_prev != expected_prev:
                        print(f"  [Resample] Group {group_idx} placement broke group {prev_idx} "
                              f"(attempt {group_attempt + 1}/{per_group_attempts})")
                        prev_ok = False
                        break

                if not prev_ok:
                    continue

                # Success — capture post-settle poses for this group
                settled_group_poses = capture_settled_poses(
                    {n: task_objects[n] for n in group_names if n in task_objects}
                )
                placed_poses.update(settled_group_poses)
                placed_aabbs.append(
                    compute_group_aabb(placed_poses, group_names)
                )
                group_placed = True
                print(f"  [Resample] Group {group_idx} ({group_names}) placed on "
                      f"attempt {group_attempt + 1}/{per_group_attempts}")
                break

            if not group_placed:
                all_groups_ok = False
                print(f"  [Resample] Failed to place group {group_idx} ({group_names}), "
                      f"restarting (overall attempt {overall_attempt + 1}/{max_attempts})")
                break

        if not all_groups_ok:
            continue

        # Final full-scene validation — physics coupling may have shifted things
        final_rel_set = relationships_to_set(extract_relationships(task_objects))
        if final_rel_set == original_rel_set:
            env.scene.update_initial_file()
            print(f"  [Resample] Valid scene found on overall attempt "
                  f"{overall_attempt + 1}/{max_attempts}")
            return True
        else:
            missing = list(original_rel_set - final_rel_set)
            extra = list(final_rel_set - original_rel_set)
            print(f"  [Resample] Final validation failed on overall attempt "
                  f"{overall_attempt + 1}/{max_attempts} (missing={missing}, extra={extra})")

    print(f"  [Resample] WARNING: no valid config found in {max_attempts} overall attempts, "
          f"keeping previous initial-file state")
    return False


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    if not cfg.s17_generate.rerun:
        print("Skipping s17_generate (rerun=False)")
        return

    annotation_dir = cfg.s15_annotate.out_dir
    waypoints_dir = cfg.s16_waypoints.out_dir
    out_dir = cfg.s17_generate.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    target_stem = cfg.s17_generate.get("_target_stem", None)
    filter_str = cfg.s17_generate.get("filter", None)

    # Filter mode: spawn a subprocess per matching dataset and return
    if filter_str and not target_stem:
        datasets = _resolve_generation_datasets(annotation_dir, waypoints_dir, filter_str)
        stems = [stem for _, _, stem in datasets]
        print(f"Spawning {len(stems)} subprocess(es):")
        for s in stems:
            print(f"  {s}")
        _run_filter_subprocesses(stems, "s17_generate")
        return

    # Single-target mode (subprocess) or default mode
    if target_stem:
        config_path = str(Path(annotation_dir) / f"env_config_{target_stem}.json")
        waypoints_path = str(Path(waypoints_dir) / f"waypoints_{target_stem}.hdf5")
        for p, desc in [(config_path, "env_config"), (waypoints_path, "waypoints")]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Target {desc} not found: {p}")
        datasets = [(config_path, waypoints_path, target_stem)]
        print(f"Processing single target: {target_stem}")
    else:
        datasets = _resolve_generation_datasets(annotation_dir, waypoints_dir, None)

    robot_id = cfg.s15_annotate.robot_id
    n_demos_to_generate = cfg.s17_generate.n_demos
    max_attempts = cfg.s17_generate.max_attempts
    use_curobo = cfg.s17_generate.use_curobo
    freespace_velocity = cfg.s17_generate.freespace_velocity
    smooth_freespace = cfg.s17_generate.smooth_freespace
    action_frequency = cfg.s17_generate.action_frequency
    eef_z_offset = cfg.s17_generate.eef_z_offset
    inverted_gripper = cfg.s17_generate.inverted_gripper
    max_joint_velocity = cfg.s17_generate.get('max_joint_velocity', None)
    smooth_n_boundary_waypoints = cfg.s17_generate.get('smooth_n_boundary_waypoints', 5)
    smooth_velocity_ramp_fraction = cfg.s17_generate.get('smooth_velocity_ramp_fraction', 0.25)
    smooth_spline_tangent_scale = cfg.s17_generate.get('smooth_spline_tangent_scale', 1.0)
    smooth_spline_samples = cfg.s17_generate.get('smooth_spline_samples', 1000)
    knn_cfg_k = cfg.s17_generate.get('k_nearest_neighbors', None)
    knn_cfg_pos_weight = cfg.s17_generate.get('knn_pos_weight', 1.0)
    knn_cfg_ori_weight = cfg.s17_generate.get('knn_ori_weight', 1.0)
    knn_enabled = knn_cfg_k is not None
    blacklist_ids_cfg = cfg.s17_generate.get('blacklist_ids', None)

    # Action noise (opt-in) — TCP-bounded noise for domain randomization
    action_noise_cfg = cfg.s17_generate.get('action_noise', None)
    action_noise_enabled = bool(action_noise_cfg and action_noise_cfg.get('enabled', False))
    if action_noise_enabled:
        action_noise_params = {
            'action_scale_factor': float(action_noise_cfg.get('action_scale_factor', 0.1)),
            'rotation_noise_scale': float(action_noise_cfg.get('rotation_noise_scale', 0.1)),
            'max_tcp_position_noise': float(action_noise_cfg.get('max_tcp_position_noise', 0.02)),
            'max_tcp_rotation_noise': float(action_noise_cfg.get('max_tcp_rotation_noise', 0.1)),
        }
        action_noise_apply_to_freespace = bool(action_noise_cfg.get('apply_to_freespace', True))
        print(f"\n[Action Noise] Enabled: scale={action_noise_params['action_scale_factor']}, "
              f"rot_scale={action_noise_params['rotation_noise_scale']}, "
              f"max_pos={action_noise_params['max_tcp_position_noise']}m, "
              f"max_rot={action_noise_params['max_tcp_rotation_noise']}rad, "
              f"freespace={action_noise_apply_to_freespace}")

    # Scene resampling (opt-in) — every N successful demos, sample a new validated scene config
    resample_cfg = cfg.s17_generate.get('resample_scene', None)
    resample_enabled = bool(resample_cfg and resample_cfg.get('enabled', False))
    if resample_enabled:
        resample_every = int(resample_cfg.get('every_n_demos', 10))
        resample_xy_range = float(resample_cfg.get('xy_range', 0.01))
        resample_z_rot_deg = float(resample_cfg.get('z_rotation_deg', 50))
        resample_settle_steps = int(resample_cfg.get('settle_steps', 100))
        resample_max_attempts = int(resample_cfg.get('max_attempts', 20))
        resample_initial = bool(resample_cfg.get('resample_initial', False))
        resample_include_fixed = bool(resample_cfg.get('include_fixed_base_groups', False))
        resample_max_consec_failures = resample_cfg.get('max_consecutive_failures', 3)
        resample_max_consec_failures = int(resample_max_consec_failures) if resample_max_consec_failures else 0
        resample_per_group_attempts = int(resample_cfg.get('per_group_attempts', 10))
        resample_min_group_distance = float(resample_cfg.get('min_group_distance', 0.0))
        resample_max_rejection_samples = int(resample_cfg.get('max_rejection_samples', 50))

    apply_teleop_omnigibson_macros(enable_tr=False)

    total_success = 0
    total_attempts = 0
    base_env = None

    try:
        for dataset_idx, (config_path, waypoints_path, stem) in enumerate(datasets):
            suffix = f"_{stem}" if stem else ""
            print(f"\n{'#'*80}")
            print(f"DATASET {dataset_idx + 1}/{len(datasets)}: {stem or 'default'}")
            print(f"  Config: {config_path}")
            print(f"  Waypoints: {waypoints_path}")
            print(f"{'#'*80}")

            # Load waypoints for this dataset
            print("\nLoading waypoints...")
            all_waypoints = load_waypoints_from_hdf5(str(waypoints_path))
            episode_keys = list(all_waypoints.keys())
            assert len(episode_keys) > 0, f"No waypoints found in {waypoints_path}!"
            print(f"Loaded waypoints for {len(episode_keys)} episodes: {episode_keys}")

            # Prepare environment config
            config = _prepare_generation_config(config_path, cfg)

            # Create or reload environment
            if base_env is None:
                print("\nCreating environment...")
                base_env = og.Environment(configs=config)

                if len(base_env.robots[robot_id].sensors) == 1:
                    setup_wrist_camera_viewport(base_env, robot_id=robot_id)
                else:
                    print(f"WARNING: Expected exactly one sensor on robot {robot_id}, "
                          f"got {len(base_env.robots[robot_id].sensors)}. Skipping wrist camera viewport setup.")
            else:
                print("\nReloading environment with new config...")
                base_env.reload(config)

            # Determine output HDF5 filename. Explicit output_name takes priority;
            # otherwise derive a tag from resample parameters for backward compatibility.
            output_name = cfg.s17_generate.get("output_name", None)
            if output_name:
                output_tag = f"_{output_name}"
            elif resample_enabled:
                output_tag = (
                    f"_resample_n{resample_every}"
                    f"_xy{resample_xy_range:g}"
                    f"_z{resample_z_rot_deg:g}"
                )
            else:
                output_tag = ""
            output_hdf5_path = Path(out_dir) / f"generated_demos{suffix}{output_tag}.hdf5"

            env = NoisyActionHDF5Wrapper(
                env=base_env,
                output_path=str(output_hdf5_path),
                viewport_camera_path=None,
                overwrite=True,
                only_successes=True,
                flush_every_n_traj=10,
            )

            robot = env.robots[robot_id]
            robot_action_offset = 0
            for r in env.robots[:robot_id]:
                robot_action_offset += r.action_dim
            scene = env.scene
            task = env.task
            default_action = []
            for r in env.robots:
                r_default_action = th.zeros(r.action_dim)
                r_default_action[-1] = 1.0
                default_action.append(r_default_action)
            default_action = th.cat(default_action, dim=0)

            # Initialize CuRobo if enabled
            cmg = None
            if use_curobo:
                cmg = init_curobo(robot, batch_size=2)
            else:
                print(f"\n[Freespace Planning] Using linear interpolation (velocity={freespace_velocity} m/s, smooth={smooth_freespace})")

            if eef_z_offset != 0.0:
                print(f"[EEF Z Offset] Applying inverse offset of {-eef_z_offset}m to waypoints (original offset: {eef_z_offset}m)")

            arm_controller = robot.controllers[f"arm_{robot.default_arm}"]
            arm_dof_idx = arm_controller.dof_idx
            if max_joint_velocity is not None:
                print(f"\n[Joint Velocity Check] Enabled: max {max_joint_velocity} rad/step across {len(arm_dof_idx)} arm joints")

            if action_noise_enabled:
                from functools import partial
                env._noise_fn = partial(
                    apply_tcp_action_noise,
                    robot=robot,
                    arm_dof_idx=arm_dof_idx,
                    robot_action_offset=robot_action_offset,
                    noise_cfg=action_noise_params,
                )

            n_subtasks_per_episode = len(all_waypoints[episode_keys[0]])

            if knn_enabled:
                if isinstance(knn_cfg_k, (int, float)):
                    assert int(knn_cfg_k) > 0, f"k_nearest_neighbors must be positive, got {knn_cfg_k}"
                _list_types = (list, tuple)
                try:
                    from omegaconf import ListConfig
                    _list_types = (list, tuple, ListConfig)
                except ImportError:
                    print(f"WARNING: omegaconf not found, using list type instead")
                    pass
                for param_name, param_val in [
                    ('k_nearest_neighbors', knn_cfg_k),
                    ('knn_pos_weight', knn_cfg_pos_weight),
                    ('knn_ori_weight', knn_cfg_ori_weight),
                ]:
                    if isinstance(param_val, _list_types):
                        assert len(param_val) == n_subtasks_per_episode, \
                            f"{param_name} list length ({len(param_val)}) must equal number of subtasks ({n_subtasks_per_episode})"
                if blacklist_ids_cfg is not None:
                    blacklist_ids_resolved = [list(bl) for bl in blacklist_ids_cfg]
                    assert len(blacklist_ids_resolved) == n_subtasks_per_episode, \
                        f"blacklist_ids outer length ({len(blacklist_ids_resolved)}) must equal number of subtasks ({n_subtasks_per_episode})"
                else:
                    blacklist_ids_resolved = [None] * n_subtasks_per_episode

                bl_summary = [len(bl) if bl else 0 for bl in blacklist_ids_resolved]
                print(f"\n[KNN Selection] Enabled: K={knn_cfg_k}, pos_weight={knn_cfg_pos_weight}, ori_weight={knn_cfg_ori_weight}")
                print(f"  Blacklisted episode counts per subtask: {bl_summary}")

            # --- Per-dataset scene resampling setup (opt-in) ---
            resample_state = None
            if resample_enabled:
                scene_json_for_sampling = config['scene']['scene_file']
                assert isinstance(scene_json_for_sampling, dict), \
                    "resample_scene requires config['scene']['scene_file'] to be a dict (serialized scene)"
                rs_task_object_names = get_task_object_names(scene_json_for_sampling)
                rs_fixed_base_names = get_fixed_base_names(scene_json_for_sampling)
                rs_task_objects = {
                    name: env.scene.object_registry("name", name) for name in rs_task_object_names
                }
                rs_task_objects = {k: v for k, v in rs_task_objects.items() if v is not None}

                # Baseline: reset env and capture settled poses + relationships once
                env.reset()
                og.sim.step()
                rs_settled_poses = capture_settled_poses(rs_task_objects)
                rs_original_rels = extract_relationships(rs_task_objects)
                rs_original_rel_set = relationships_to_set(rs_original_rels)
                rs_groups = build_groups_from_relationships(
                    rs_original_rels, list(rs_task_objects.keys())
                )
                print(f"\n[Resample] Enabled — every {resample_every} demos, "
                      f"xy±{resample_xy_range}m, z±{resample_z_rot_deg}°, "
                      f"max_attempts={resample_max_attempts}, "
                      f"include_fixed_base_groups={resample_include_fixed}, "
                      f"max_consecutive_failures={resample_max_consec_failures or 'disabled'}")
                print(f"[Resample] {len(rs_task_objects)} task objects, "
                      f"{len(rs_groups)} group(s), "
                      f"{len(rs_original_rel_set)} OnTop relationship(s), "
                      f"fixed_base={sorted(rs_fixed_base_names)}")
                print(f"[Resample] NOTE: task-level group_xyz/z_rot randomization still runs "
                      f"on each env.reset(); set it to 0 in your task YAML if you want only "
                      f"the validated resampling.")

                resample_state = {
                    "scene_json": scene_json_for_sampling,
                    "task_objects": rs_task_objects,
                    "groups": rs_groups,
                    "settled_poses": rs_settled_poses,
                    "original_rel_set": rs_original_rel_set,
                    "fixed_base_names": rs_fixed_base_names,
                    "demos_since_resample": resample_every if resample_initial else 0,
                    "consecutive_failures": 0,
                }

            # --- Per-dataset generation loop ---
            success_count = 0
            attempt_count = 0

            print(f"\n{'='*80}")
            print(f"GENERATING {n_demos_to_generate} DEMONSTRATIONS for {stem or 'default'}")
            print(f"{'='*80}")

            while success_count < n_demos_to_generate and attempt_count < max_attempts:
                attempt_count += 1
                print(f"\n{'='*80}")
                print(f"Attempt {attempt_count}/{max_attempts} (Successes: {success_count}/{n_demos_to_generate})")
                print(f"{'='*80}")

                if resample_state is not None:
                    _trigger_reason = None
                    if resample_state["demos_since_resample"] >= resample_every:
                        _trigger_reason = (
                            f"demos_since_resample={resample_state['demos_since_resample']}"
                        )
                    elif (resample_max_consec_failures > 0
                          and resample_state["consecutive_failures"] >= resample_max_consec_failures):
                        _trigger_reason = (
                            f"consecutive_failures={resample_state['consecutive_failures']} "
                            f">= max_consecutive_failures={resample_max_consec_failures}"
                        )
                    if _trigger_reason is not None:
                        print(f"\n[Resample] Triggering scene resample ({_trigger_reason})")
                        # Reset env first so the robot arm returns to its home
                        # pose and doesn't interfere with object settling.
                        env.reset()
                        _resample_valid_scene(
                            env=env,
                            task_objects=resample_state["task_objects"],
                            scene_json=resample_state["scene_json"],
                            groups=resample_state["groups"],
                            settled_poses=resample_state["settled_poses"],
                            original_rel_set=resample_state["original_rel_set"],
                            fixed_base_names=resample_state["fixed_base_names"],
                            xy_range=resample_xy_range,
                            z_rotation_deg=resample_z_rot_deg,
                            settle_steps=resample_settle_steps,
                            max_attempts=resample_max_attempts,
                            include_fixed_base_groups=resample_include_fixed,
                            per_group_attempts=resample_per_group_attempts,
                            min_group_distance=resample_min_group_distance,
                            max_rejection_samples=resample_max_rejection_samples,
                        )
                        resample_state["demos_since_resample"] = 0
                        resample_state["consecutive_failures"] = 0

                env.reset()
                prev_arm_joint_pos = robot.get_joint_positions()[arm_dof_idx].clone() if max_joint_velocity is not None else None
                prev_subtask_positions = None
                prev_subtask_quats = None

                if not knn_enabled:
                    source_demo_key = random.choice(episode_keys)
                    subtasks = all_waypoints[source_demo_key]
                    print(f"Selected source demo: {source_demo_key} with {len(subtasks)} subtasks")

                action = None

                gripper_cmd = None
                for subtask_idx in range(n_subtasks_per_episode):
                    if knn_enabled:
                        subtask_type_for_knn = next(
                            wps[subtask_idx]['type'] for wps in all_waypoints.values() if subtask_idx < len(wps)
                        )
                        k_resolved = int(resolve_knn_param(knn_cfg_k, subtask_type_for_knn, subtask_idx, 1))
                        pos_w_resolved = float(resolve_knn_param(knn_cfg_pos_weight, subtask_type_for_knn, subtask_idx, 1.0))
                        ori_w_resolved = float(resolve_knn_param(knn_cfg_ori_weight, subtask_type_for_knn, subtask_idx, 1.0))

                        subtask, selected_demo_key = select_subtask_knn(
                            all_waypoints, episode_keys, subtask_idx, scene,
                            k=k_resolved, pos_weight=pos_w_resolved, ori_weight=ori_w_resolved,
                            blacklist_ids=blacklist_ids_resolved[subtask_idx],
                        )
                        if subtask is None:
                            print(f"  WARNING: No valid KNN candidates for subtask {subtask_idx}, skipping episode")
                            break
                        print(f"\n[Subtask {subtask_idx + 1}/{n_subtasks_per_episode}] {subtask['type'].upper()} "
                              f"(KNN K={k_resolved} pos_w={pos_w_resolved} ori_w={ori_w_resolved}, selected from {selected_demo_key})")
                    else:
                        subtask = subtasks[subtask_idx]
                        print(f"\n[Subtask {subtask_idx + 1}/{n_subtasks_per_episode}] {subtask['type'].upper()}")

                    print(f"  Reference object: {subtask['reference_object']['name']}")
                    print(f"  Waypoints: {len(subtask['waypoints'])}")
                    failed_step = False

                    ref_obj_name = subtask['reference_object']['name']
                    ref_obj = scene.object_registry("name", ref_obj_name, None)
                    assert ref_obj is not None, f"Reference object '{ref_obj_name}' not found in scene!"

                    eef_pos_robot, eef_quat_robot, gripper_actions = transform_waypoints_to_robot_frame(
                        subtask['waypoints'],
                        ref_obj,
                        robot,
                        subtask_type=subtask['type']
                    )

                    if eef_z_offset != 0.0:
                        eef_pos_robot, eef_quat_robot = apply_eef_z_offset(
                            eef_pos_robot, eef_quat_robot, -eef_z_offset
                        )

                    start_pos_robot = eef_pos_robot[0]
                    start_quat_robot = eef_quat_robot[0]

                    import omnigibson.utils.transform_utils as T
                    robot_pos, robot_quat = robot.get_position_orientation()
                    if not isinstance(robot_pos, th.Tensor):
                        robot_pos = th.tensor(robot_pos, dtype=th.float32)
                    if not isinstance(robot_quat, th.Tensor):
                        robot_quat = th.tensor(robot_quat, dtype=th.float32)

                    robot_T_world = T.pose2mat((robot_pos, robot_quat))
                    start_T_robot = T.pose2mat((start_pos_robot, start_quat_robot))
                    start_T_world = robot_T_world @ start_T_robot
                    start_pos_world, start_quat_world = T.mat2pose(start_T_world)

                    print(f"  Target start pose (world): pos={start_pos_world.cpu().numpy()}")

                    from simfoundry.utils.og_utils import draw_trajectory_gradient
                    eef_pos_world = (th.cat([eef_pos_robot, th.ones(eef_pos_robot.shape[0], 1)], dim=1) @ robot_T_world.T)[:, :3]
                    draw_trajectory_gradient(
                        eef_pos_world.tolist(),
                        color_scheme="green_to_red",
                        point_size=10,
                        clear_previous=True
                    )

                    attached_obj = None
                    attached_obj_scale = None
                    if subtask['type'] == 'place' and use_curobo:
                        placed_obj_name = subtask.get('placed_object', {}).get('name', None)
                        placed_obj = scene.object_registry("name", placed_obj_name, None)
                        assert placed_obj is not None, f"Placed object '{placed_obj_name}' not found in scene!"

                        eef_link_name = robot.eef_link_names[robot.default_arm]
                        attached_obj = {eef_link_name: placed_obj.root_link}
                        attached_obj_scale = {eef_link_name: DEFAULT_MOTION_PLANNER_ATTACHED_OBJECT_SCALE}
                        print(f"  Planning with attached object: {placed_obj_name}")

                    print("  Planning freespace motion to start...")

                    if action_noise_enabled and not action_noise_apply_to_freespace:
                        env.noise_enabled = False

                    if use_curobo:
                        joint_trajectory = plan_to_pose(
                            cmg,
                            robot,
                            start_pos_world,
                            start_quat_world,
                            max_attempts=cfg.s17_generate.curobo_max_attempts,
                            timeout=cfg.s17_generate.curobo_timeout,
                            attached_obj=attached_obj,
                            attached_obj_scale=attached_obj_scale
                        )

                        if joint_trajectory is None:
                            print("  WARNING: CuRobo failed to plan to start pose, skipping this attempt")
                            break

                        assert len(joint_trajectory) > 0, "CuRobo returned empty trajectory"
                        print(f"  Planned trajectory with {len(joint_trajectory)} waypoints")

                        freespace_actions = []
                        for i in range(len(joint_trajectory)):
                            action = robot.q_to_action(joint_trajectory[i], controller_name='arm_0')
                            if subtask['type'] in ['pick', 'open', 'close']:
                                gripper_action = 1.0 if not inverted_gripper else 0.0
                            else:
                                gripper_action = gripper_cmd if gripper_cmd is not None else (0.0 if not inverted_gripper else 1.0)
                            action = th.cat([action, th.tensor([gripper_action])], dim=0)
                            full_action = default_action.clone()
                            full_action[robot_action_offset:robot_action_offset + robot.action_dim] = action
                            freespace_actions.append(full_action)
                        freespace_actions = np.array(freespace_actions)

                        print("  Executing freespace motion (joint control)...")
                        for i in range(freespace_actions.shape[0]):
                            fa = freespace_actions[i]
                            env.step(fa.cpu().numpy() if isinstance(fa, th.Tensor) else fa)
                            if max_joint_velocity is not None:
                                prev_arm_joint_pos, exceeded = check_joint_velocity_exceeded(
                                    robot, arm_dof_idx, prev_arm_joint_pos, max_joint_velocity)
                                if exceeded:
                                    failed_step = True
                                    break
                    else:
                        freespace_positions, freespace_quats = compute_linear_freespace_trajectory(
                            robot=robot,
                            target_pos_robot=start_pos_robot,
                            target_quat_robot=start_quat_robot,
                            freespace_velocity=freespace_velocity,
                            action_frequency=action_frequency,
                            smooth=smooth_freespace,
                            preceding_positions=prev_subtask_positions if smooth_freespace else None,
                            preceding_quats=prev_subtask_quats if smooth_freespace else None,
                            waypoint_positions=eef_pos_robot if smooth_freespace else None,
                            waypoint_quats=eef_quat_robot if smooth_freespace else None,
                            n_boundary_waypoints=smooth_n_boundary_waypoints,
                            velocity_ramp_fraction=smooth_velocity_ramp_fraction,
                            spline_tangent_scale=smooth_spline_tangent_scale,
                            n_spline_samples=smooth_spline_samples,
                        )

                        print(f"  Planned linear trajectory with {len(freespace_positions)} waypoints")

                        print("  Executing freespace motion (IK to joint space)...")
                        freespace_gripper_is_open = any(subtask['type'].startswith(subtask_type) for subtask_type in ['pick', 'open', 'close'])
                        for i in range(len(freespace_positions)):
                            target_pos = freespace_positions[i]
                            target_quat = freespace_quats[i]

                            if freespace_gripper_is_open:
                                gripper_cmd = 1.0 if not inverted_gripper else 0.0
                            else:
                                gripper_cmd = 0.0 if not inverted_gripper else 1.0

                            try:
                                action = solve_ik_step_og(robot, target_pos.numpy(), target_quat.numpy(), gripper_cmd)
                            except Exception as e:
                                print(f"  WARNING: IK solver failed for waypoint {i}: {e}")
                                failed_step = True
                                break

                            full_action = default_action.clone()
                            full_action[robot_action_offset:robot_action_offset + robot.action_dim] = th.from_numpy(action)
                            env.step(full_action)
                            if max_joint_velocity is not None:
                                prev_arm_joint_pos, exceeded = check_joint_velocity_exceeded(
                                    robot, arm_dof_idx, prev_arm_joint_pos, max_joint_velocity)
                                if exceeded:
                                    failed_step = True
                                    break

                    if action_noise_enabled and not action_noise_apply_to_freespace:
                        env.noise_enabled = True

                    if failed_step:
                        break

                    print("  Executing waypoint trajectory (IK to joint space)...")

                    for i in range(len(eef_pos_robot)):
                        target_pos_robot = eef_pos_robot[i]
                        target_quat_robot = eef_quat_robot[i]
                        gripper_cmd = gripper_actions[i].item()

                        try:
                            action = solve_ik_step_og(robot, target_pos_robot.numpy(), target_quat_robot.numpy(), gripper_cmd)
                        except Exception as e:
                            print(f"  WARNING: IK solver failed for waypoint {i}: {e}")
                            failed_step = True
                            break

                        full_action = default_action.clone()
                        full_action[robot_action_offset:robot_action_offset + robot.action_dim] = th.from_numpy(action)
                        env.step(full_action)
                        if max_joint_velocity is not None:
                            prev_arm_joint_pos, exceeded = check_joint_velocity_exceeded(
                                robot, arm_dof_idx, prev_arm_joint_pos, max_joint_velocity)
                            if exceeded:
                                failed_step = True
                                break

                    print(f"  Subtask {subtask_idx + 1} completed")

                    prev_subtask_positions = eef_pos_robot
                    prev_subtask_quats = eef_quat_robot

                    if not failed_step:
                        subtask_type = subtask['type']
                        arm = robot.default_arm

                        if subtask_type == 'pick':
                            pick_obj_name = subtask['reference_object']['name']
                            pick_obj = scene.object_registry("name", pick_obj_name, None)
                            obj_in_hand = robot._ag_obj_in_hand.get(arm, None)
                            is_holding = (obj_in_hand is not None and obj_in_hand == pick_obj)
                            if not is_holding:
                                from omnigibson.controllers import IsGraspingState
                                for a in robot.arm_names:
                                    if pick_obj is not None and robot.is_grasping(arm=a, candidate_obj=pick_obj) == IsGraspingState.TRUE:
                                        is_holding = True
                                        break
                            if not is_holding:
                                print(f"  EARLY TERMINATION: Pick failed — robot is not holding '{pick_obj_name}'")
                                failed_step = True

                        elif subtask_type == 'place':
                            placed_obj_name = subtask.get('placed_object', {}).get('name', None)
                            if placed_obj_name is not None:
                                placed_obj = scene.object_registry("name", placed_obj_name, None)
                                obj_in_hand = robot._ag_obj_in_hand.get(arm, None)
                                still_holding = (obj_in_hand is not None and obj_in_hand == placed_obj)
                                if still_holding:
                                    print(f"  EARLY TERMINATION: Place failed — robot still holding '{placed_obj_name}'")
                                    failed_step = True

                        elif subtask_type == 'open':
                            open_obj_name = subtask['reference_object']['name']
                            open_obj = scene.object_registry("name", open_obj_name, None)
                            if open_obj is not None and Open in open_obj.states:
                                if not open_obj.states[Open].get_value():
                                    print(f"  EARLY TERMINATION: Open failed — '{open_obj_name}' is not open")
                                    failed_step = True

                        elif subtask_type == 'close':
                            close_obj_name = subtask['reference_object']['name']
                            close_obj = scene.object_registry("name", close_obj_name, None)
                            if close_obj is not None and Open in close_obj.states:
                                if close_obj.states[Open].get_value():
                                    print(f"  EARLY TERMINATION: Close failed — '{close_obj_name}' is still open")
                                    failed_step = True

                    if failed_step:
                        break

                if action is not None:
                    for obj in env.scene.objects:
                        obj.wake()
                    og.sim.step()
                    task.step(env, action)

                task_success = task._success

                if task_success:
                    print(f"\n{'='*40}")
                    print(f"SUCCESS! Demo {success_count + 1} completed")
                    print(f"{'='*40}")
                    success_count += 1
                    if resample_state is not None:
                        resample_state["demos_since_resample"] += 1
                        resample_state["consecutive_failures"] = 0
                else:
                    print("\nTask check failed - demo will not be saved")
                    if resample_state is not None:
                        resample_state["consecutive_failures"] += 1
                        if resample_max_consec_failures > 0:
                            print(f"  [Resample] consecutive_failures="
                                  f"{resample_state['consecutive_failures']}/"
                                  f"{resample_max_consec_failures}")

                if attempt_count >= max_attempts:
                    print(f"\nReached maximum attempts ({max_attempts})")
                    break

            # Save collected data for this dataset
            print(f"\nSaving generated demonstrations for {stem or 'default'}...")
            env.save_data()

            total_success += success_count
            total_attempts += attempt_count

            print(f"\nDataset '{stem or 'default'}': {success_count}/{n_demos_to_generate} successful demos "
                  f"in {attempt_count} attempts -> {output_hdf5_path}")

    except Exception as e:
        print(f"ERROR: {e}")
        from IPython import embed; embed()

    finally:
        if base_env is not None:
            try:
                og.clear()
            except KeyError as e:
                print(f"Warning: Cleanup error (can be safely ignored): {e}")
            og.shutdown()

    print("\n" + "="*80)
    print(f"DEMO GENERATION COMPLETE")
    print(f"Processed {len(datasets)} dataset(s): {total_success} total successful demos in {total_attempts} total attempts")
    print(f"Output directory: {out_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
