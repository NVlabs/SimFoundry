# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from b1k

Requires installing:

- BEHAVIOR-1K, see https://github.com/StanfordVL/BEHAVIOR-1K
"""
import omnigibson as og
from omnigibson.macros import gm
from pathlib import Path
import os
import sys
import shutil
import subprocess as sp
import numpy as np
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
from simfoundry import import_og_dependencies, CFG_DIR as SIMFOUNDRY_CFG_DIR
from simfoundry.utils.processing_utils import dump_json
from simfoundry.utils.og_utils import apply_teleop_omnigibson_macros
from omnigibson.envs.data_wrapper import LeRobotPlaybackWrapper
from simfoundry.envs.omnigibson.lerobot_pointbridge_playback_wrapper import (
    LeRobotPointbridgePlaybackWrapper,
    LeRobotPlaybackWrapperWithTransforms,
)
from omnigibson.utils.config_utils import parse_config
from omnigibson.learning.utils import obs_utils as OU
from scipy.spatial.transform import Rotation as R
from glob import glob
import av

# Suppress verbose libx264 encoder output
av.logging.set_level(av.logging.ERROR)

# Needed so custom tasks can be instantiated properly
import_og_dependencies()

from simfoundry import CFG_DIR


### At the start of every script, we cd into the scripts/config directory
scripts_dir = os.path.dirname(os.path.abspath(__file__))
cfg_dir = CFG_DIR
os.chdir(cfg_dir)


def build_domain_randomization_manager(dr_cfg):
    """Build a DomainRandomizationManager from a Hydra domain_randomization config.

    Args:
        dr_cfg: The s18_replay.domain_randomization OmegaConf node.

    Returns:
        DomainRandomizationManager instance (may be disabled if dr_cfg.enabled is False).
    """
    from simfoundry.domain_randomization import (
        DomainRandomizationCfg,
        MaterialRandomizationCfg,
        LightingRandomizationCfg,
        DomainRandomizationManager,
    )

    dr_cfg = OmegaConf.to_container(dr_cfg, resolve=True)

    mat_cfg_dict = dr_cfg.get("materials", {})
    light_cfg_dict = dr_cfg.get("lighting", {})

    object_names = mat_cfg_dict.get("object_names", None)
    if object_names is None:
        scene = og.sim.scenes[0]
        object_names = [obj.name for obj in scene.objects if ("robot" not in obj.name and "background" not in obj.name)]

    mat_cfg = MaterialRandomizationCfg(
        enabled=dr_cfg.get("enabled", False) and mat_cfg_dict.get("enabled", False),
        object_names=object_names,
        randomize_robot=mat_cfg_dict.get("randomize_robot", False),
        material_preset=mat_cfg_dict.get("material_preset", None),
        categories=mat_cfg_dict.get("categories", None),
        num_variants_per_material=mat_cfg_dict.get("num_variants_per_material", 3),
        include_wood=mat_cfg_dict.get("include_wood", True),
        include_metal=mat_cfg_dict.get("include_metal", True),
        include_plastic=mat_cfg_dict.get("include_plastic", True),
        include_paint=mat_cfg_dict.get("include_paint", True),
    )

    light_cfg = LightingRandomizationCfg(
        enabled=dr_cfg.get("enabled", False) and light_cfg_dict.get("enabled", False),
        dome_intensity_range=tuple(light_cfg_dict.get("dome_intensity_range", [300.0, 1200.0])),
        dome_color_temperature_range=tuple(light_cfg_dict.get("dome_color_temperature_range", [4000.0, 8000.0])),
        use_hdri_textures=light_cfg_dict.get("use_hdri_textures", False),
        randomize_hdri_rotation=light_cfg_dict.get("randomize_hdri_rotation", True),
        hdri_rotation_range=tuple(light_cfg_dict.get("hdri_rotation_range", [0.0, 360.0])),
    )

    cfg = DomainRandomizationCfg(
        enabled=dr_cfg.get("enabled", False),
        materials=mat_cfg,
        lighting=light_cfg,
        randomize_on_reset=True,
        randomize_interval_steps=dr_cfg.get("randomize_interval_steps", 0),
    )

    return DomainRandomizationManager(cfg)


def randomize_camera_poses(env, cam_pose_cfg, default_cam_poses):
    """Apply small random perturbations to external sensor camera poses.

    Args:
        env: The playback wrapper environment.
        cam_pose_cfg: Dict with camera_pose randomization config
            (position_noise_std, orientation_noise_std).
        default_cam_poses: Dictionary of default camera (pos, quat) poses.
    """
    import torch as th
    from omnigibson.utils.transform_utils import quat_multiply, euler2quat

    pos_std = cam_pose_cfg.get("position_noise_std", 0.0)
    ori_std_deg = cam_pose_cfg.get("orientation_noise_std", 0.0)
    ori_std_rad = np.radians(ori_std_deg)

    scene = og.sim.scenes[0]
    if scene is None:
        return

    for name, sensor in env.external_sensors.items():
        pos, ori = default_cam_poses[name]
        pos_noise = th.tensor(np.random.normal(0, pos_std, 3), dtype=pos.dtype, device=pos.device)
        euler_noise = th.tensor(np.random.normal(0, ori_std_rad, 3), dtype=ori.dtype, device=ori.device)
        ori_noise = euler2quat(euler_noise)
        sensor.set_position_orientation(position=pos + pos_noise, orientation=quat_multiply(ori_noise, ori))


def playback_dataset_with_dr(env, dr_manager, cam_pose_cfg, record_data=True, n_episodes=None, random_sample=True):
    """Replay episodes with optional domain randomization applied per-episode and per-step.

    Replaces env.playback_dataset() to inject randomization at episode boundaries
    and optionally every N simulation steps.

    Args:
        env: The playback wrapper (LeRobotPlaybackWrapper or subclass).
        dr_manager: Initialized DomainRandomizationManager.
        cam_pose_cfg: Camera pose randomization config dict, or None if disabled.
        record_data: Whether to record observation data.
        n_episodes: Number of episodes to replay (None = all).
        random_sample: Whether to randomly sample episode IDs.
    """
    n_total = env.input_hdf5["data"].attrs["n_episodes"]
    if n_episodes is None:
        episode_ids = range(n_total)
    else:
        if n_episodes > n_total:
            print(f"[WARNING] n_episodes ({n_episodes}) > available ({n_total}), clamping")
            n_episodes = n_total
        if random_sample:
            episode_ids = np.random.choice(n_total, size=n_episodes, replace=False)
        else:
            episode_ids = range(n_episodes)

    interval = dr_manager.cfg.randomize_interval_steps if dr_manager.is_enabled else 0
    per_step_dr = dr_manager.is_enabled and interval > 0

    # Wrap env.env.step to inject per-step randomization
    original_step = env.env.step
    episode_first_step = [True]
    step_count = [0]

    default_cam_poses = {name: sensor.get_position_orientation() for name, sensor in env.external_sensors.items()}

    def _apply_all_randomizations():
        dr_manager.randomize()
        if cam_pose_cfg is not None:
            randomize_camera_poses(env, cam_pose_cfg, default_cam_poses)

    def step_with_dr(*args, **kwargs):
        if episode_first_step[0]:
            episode_first_step[0] = False
            _apply_all_randomizations()
        elif per_step_dr:
            step_count[0] += 1
            if step_count[0] % interval == 0:
                _apply_all_randomizations()
        return original_step(*args, **kwargs)

    if dr_manager.is_enabled:
        env.env.step = step_with_dr

    try:
        for episode_id in tqdm(episode_ids, desc="Playing back episodes:"):
            dr_manager.reset_step_count()
            episode_first_step[0] = True
            step_count[0] = 0
            env.playback_episode(
                episode_id=episode_id,
                record_data=record_data,
            )
    finally:
        env.env.step = original_step


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


def _resolve_replay_inputs(teleop_dir, mimicgen_dir, include_mimicgen_data, filter_str):
    """
    Resolve (input_hdf5_path, stem_or_None) tuples for replay.

    When filter_str is set, finds all HDF5 files matching the filter (from s17
    if include_mimicgen_data, otherwise from s14).

    Returns:
        List of (input_hdf5_path, stem_or_None) tuples.
    """
    if include_mimicgen_data:
        search_dir = mimicgen_dir
        prefix = "generated_demos_"
        default_file = "generated_demos.hdf5"
    else:
        search_dir = teleop_dir
        prefix = ""
        default_file = None

    if filter_str:
        hdf5_files = sorted(glob(f"{search_dir}/*.hdf5"))
        matched = []
        for f in hdf5_files:
            basename = os.path.basename(f)
            if filter_str not in basename:
                continue
            if include_mimicgen_data and basename.startswith(prefix):
                stem = basename[len(prefix):].replace(".hdf5", "")
            else:
                stem = Path(f).stem
            matched.append((f, stem))

        if not matched:
            available = [os.path.basename(f) for f in hdf5_files]
            raise FileNotFoundError(
                f"No HDF5 files matching filter '{filter_str}' in {search_dir}. "
                f"Available: {available}"
            )
        print(f"Filter '{filter_str}' matched {len(matched)} file(s):")
        for _, stem in matched:
            print(f"  {stem}")
        return matched
    else:
        if include_mimicgen_data:
            default_path = f"{mimicgen_dir}/{default_file}"
            if os.path.exists(default_path):
                path = default_path
            else:
                # Fall back to the most recent generated_demos*.hdf5 (picks up
                # tagged outputs from s17 like generated_demos_resample_n2_xy0.03_z20.hdf5)
                hdf5_files = glob(f"{mimicgen_dir}/generated_demos*.hdf5")
                if not hdf5_files:
                    raise FileNotFoundError(
                        f"No generated_demos*.hdf5 found in {mimicgen_dir}"
                    )
                path = max(hdf5_files, key=os.path.getmtime)
                print(f"'{default_file}' not found; using most recent match: {os.path.basename(path)}")
        else:
            hdf5_files = glob(f"{teleop_dir}/*.hdf5")
            if not hdf5_files:
                raise FileNotFoundError(f"No HDF5 files found in {teleop_dir}")
            path = max(hdf5_files, key=os.path.getmtime)
        return [(path, None)]


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    teleop_dir = cfg.s14_teleop.out_dir
    out_dir = cfg.s18_replay.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    target_stem = cfg.s18_replay.get("_target_stem", None)
    filter_str = cfg.s18_replay.get("filter", None)
    include_mimicgen_data = cfg.s18_replay.include_mimicgen_data
    mimicgen_dir = cfg.s17_generate.out_dir if include_mimicgen_data else None

    # Filter mode: spawn a subprocess per matching HDF5 and return
    if filter_str and not target_stem:
        inputs = _resolve_replay_inputs(teleop_dir, mimicgen_dir, include_mimicgen_data, filter_str)
        stems = [stem for _, stem in inputs]
        print(f"Spawning {len(stems)} subprocess(es):")
        for s in stems:
            print(f"  {s}")
        _run_filter_subprocesses(stems, "s18_replay")
        return

    # Single-target mode (subprocess) or default mode
    if target_stem:
        if include_mimicgen_data:
            input_hdf5_path = f"{mimicgen_dir}/generated_demos_{target_stem}.hdf5"
        else:
            input_hdf5_path = str(Path(teleop_dir) / f"{target_stem}.hdf5")
        if not os.path.exists(input_hdf5_path):
            raise FileNotFoundError(f"Target HDF5 not found: {input_hdf5_path}")
        print(f"Processing single target: {target_stem}")
    else:
        inputs = _resolve_replay_inputs(teleop_dir, mimicgen_dir, include_mimicgen_data, None)
        input_hdf5_path = inputs[0][0]

    print(f"Found input HDF5 file: {input_hdf5_path}")

    # Set OG macros
    OU.USE_DEPTH_RGB_ENCODING = cfg.s18_replay.depth_as_rgb

    # Grab external sensors config
    external_sensors_cfg_path = f"{SIMFOUNDRY_CFG_DIR}/external_sensors/{cfg.s18_replay.external_sensors_cfg}.yaml"
    external_sensors_cfg = parse_config(external_sensors_cfg_path)["external_sensors"]
    cfg.s18_replay.playback_kwargs.external_sensors_config = external_sensors_cfg

    # Determine which wrapper class to use
    use_pointbridge = cfg.s18_replay.get("use_pointbridge", False)

    # Construct output path for LeRobot dataset
    # LeRobot expects a repo_id-like path (e.g., "username/dataset_name")
    dataset_name = cfg.s18_replay.dataset_name
    if use_pointbridge:
        dataset_name += "_pointbridge"
    if include_mimicgen_data:
        dataset_name += "_mimicgen"
    if target_stem:
        dataset_name += f"_{target_stem}"
    output_path = f"{cfg.s18_replay.repo_prefix}/{dataset_name}"

    print(f"Creating LeRobot dataset at: {out_dir}/{output_path}")

    # Apply macros for OmniGibson teleop mode
    apply_teleop_omnigibson_macros(enable_tr=False)
    
    # Resolve language instruction: explicit config > task yaml > wrapper auto-infer
    task_name_for_dataset = cfg.s18_replay.playback_kwargs.task_name
    if task_name_for_dataset is None:
        task_name_for_dataset = cfg.task.get("language_instruction", None)
    if task_name_for_dataset:
        print(f"Language instruction for dataset: '{task_name_for_dataset}'")

    # Common kwargs for both wrappers
    robot_sensor_config = OmegaConf.to_container(cfg.s18_replay.playback_kwargs.robot_sensor_config, resolve=True) if cfg.s18_replay.playback_kwargs.robot_sensor_config is not None else None
    common_kwargs = dict(
        input_path=input_hdf5_path,
        output_path=output_path,
        robot_obs_modalities=cfg.s18_replay.playback_kwargs.robot_obs_modalities,
        robot_grasping_mode="physical",
        robot_proprio_keys=cfg.s18_replay.playback_kwargs.robot_proprio_keys,
        robot_sensor_config=robot_sensor_config,
        external_sensors_config=external_sensors_cfg,
        include_sensor_names=cfg.s18_replay.playback_kwargs.include_sensor_names,
        exclude_sensor_names=cfg.s18_replay.playback_kwargs.exclude_sensor_names,
        n_render_iterations=cfg.s18_replay.playback_kwargs.n_render_iterations,
        overwrite=cfg.s18_replay.playback_kwargs.overwrite,
        only_successes=False,  # cfg.s18_replay.playback_kwargs.only_successes,
        flush_every_n_traj=cfg.s18_replay.playback_kwargs.flush_every_n_traj,
        flush_every_n_steps=cfg.s18_replay.playback_kwargs.flush_every_n_steps,
        include_env_wrapper=cfg.s18_replay.playback_kwargs.include_env_wrapper,
        additional_wrapper_configs=None,
        full_scene_file=None,
        include_task=cfg.s18_replay.playback_kwargs.include_task,
        include_task_obs=cfg.s18_replay.playback_kwargs.include_task_obs,
        include_robot_control=cfg.s18_replay.playback_kwargs.include_robot_control,
        include_contacts=cfg.s18_replay.playback_kwargs.include_contacts,
        root_dir=out_dir,
        robot_type=cfg.s18_replay.playback_kwargs.robot_type,
        image_writer_threads=cfg.s18_replay.playback_kwargs.image_writer_threads,
        image_writer_processes=cfg.s18_replay.playback_kwargs.image_writer_processes,
        task_name=task_name_for_dataset,
        lerobot_version=cfg.s18_replay.playback_kwargs.lerobot_version,
        use_videos=cfg.s18_replay.playback_kwargs.use_videos,
        include_multi_action_representation=cfg.s18_replay.playback_kwargs.include_multi_action_representation,
        include_modalities=cfg.s18_replay.playback_kwargs.include_modalities,
    )

    # Data transformation options (applied during saving only)
    invert_gripper_action = cfg.s18_replay.get("invert_gripper_action", False)
    scale_gripper_state = cfg.s18_replay.get("scale_gripper_state", 1.0)
    
    if invert_gripper_action or scale_gripper_state != 1.0:
        print(f"Data transformations enabled:")
        if invert_gripper_action:
            print(f"  - Invert gripper action: action[-1] = 1 - action[-1]")
        if scale_gripper_state != 1.0:
            print(f"  - Scale gripper state: state[-1] *= {scale_gripper_state}")

    # Create the playback wrapper from HDF5
    if use_pointbridge:
        # Get pointbridge-specific kwargs
        pointbridge_kwargs = cfg.s18_replay.get("pointbridge_kwargs", {})
        if pointbridge_kwargs is not None:
            pointbridge_kwargs = OmegaConf.to_container(pointbridge_kwargs, resolve=True)
        else:
            pointbridge_kwargs = {}

        # Make sure all sensor modalities include seg_semantic
        for sensor_cfg in common_kwargs["external_sensors_config"]:
            if "seg_semantic" not in sensor_cfg["modalities"]:
                sensor_cfg["modalities"].append("seg_semantic")
        if "seg_semantic" not in common_kwargs["robot_sensor_config"]["VisionSensor"]["modalities"]:
            common_kwargs["robot_sensor_config"]["VisionSensor"]["modalities"].append("seg_semantic")
        
        print("Using LeRobotPointbridgePlaybackWrapper for point cloud generation")
        env = LeRobotPointbridgePlaybackWrapper.create_from_hdf5(
            **common_kwargs,
            num_points_obj=pointbridge_kwargs.get("num_points_obj", 128),
            num_objects=pointbridge_kwargs.get("num_objects", 1),
            include_categories=pointbridge_kwargs.get("include_categories", None),
            pc_x_min=pointbridge_kwargs.get("pc_x_min", -2.0),
            pc_x_max=pointbridge_kwargs.get("pc_x_max", 2.0),
            pc_y_min=pointbridge_kwargs.get("pc_y_min", -2.0),
            pc_y_max=pointbridge_kwargs.get("pc_y_max", 2.0),
            pc_z_min=pointbridge_kwargs.get("pc_z_min", 0.001),
            pc_z_max=pointbridge_kwargs.get("pc_z_max", 2.0),
            pointcloud_sensors=pointbridge_kwargs.get("pointcloud_sensors", None),
            invert_gripper_action=invert_gripper_action,
            scale_gripper_state=scale_gripper_state,
        )
    else:
        # Use wrapper with transformation support for non-pointbridge case
        env = LeRobotPlaybackWrapperWithTransforms.create_from_hdf5(
            **common_kwargs,
            invert_gripper_action=invert_gripper_action,
            scale_gripper_state=scale_gripper_state,
        )

    print(f"\nStarting dataset replay and observation collection...")
    print(f"Input: {input_hdf5_path}")
    print(f"Output: {out_dir}/{output_path}")
    print("="*60)

    # Hide the ground plane
    og.sim.floor_plane.visible = False

    # Build and initialize domain randomization
    dr_cfg = cfg.s18_replay.get("domain_randomization", {})
    dr_manager = build_domain_randomization_manager(dr_cfg)

    if dr_manager.is_enabled:
        print("\nInitializing domain randomization...")
        dr_manager.initialize()
        interval = dr_manager.cfg.randomize_interval_steps
        if interval == 0:
            print("  Randomization will be applied once at the start of each episode")
        else:
            print(f"  Randomization will be applied every {interval} step(s)")

    cam_pose_cfg = None
    if dr_cfg and OmegaConf.to_container(dr_cfg, resolve=True).get("enabled", False):
        cam_raw = OmegaConf.to_container(dr_cfg, resolve=True).get("camera_pose", {})
        if cam_raw.get("enabled", False):
            cam_pose_cfg = cam_raw
            print(f"  Camera pose noise: pos_std={cam_pose_cfg['position_noise_std']}m, "
                  f"ori_std={cam_pose_cfg['orientation_noise_std']}deg")

    # Playback all episodes and record observations (with optional DR)
    if dr_manager.is_enabled or cam_pose_cfg is not None:
        playback_dataset_with_dr(
            env=env,
            dr_manager=dr_manager,
            cam_pose_cfg=cam_pose_cfg,
            record_data=True,
            n_episodes=cfg.s18_replay.playback_kwargs.n_episodes,
            random_sample=True,
        )
    else:
        env.playback_dataset(record_data=True, n_episodes=cfg.s18_replay.playback_kwargs.n_episodes, random_sample=True)

    # Save the dataset
    env.save_data()

    # Optionally replace the auto-generated modality.json with a user-provided one
    modality_json = cfg.s18_replay.get("modality_json", None)
    if modality_json:
        modality_json = str(modality_json)
        if not os.path.exists(modality_json):
            print(f"WARNING: modality_json not found: {modality_json}, skipping replacement")
        else:
            dest = os.path.join(out_dir, output_path, "meta", "modality.json")
            shutil.copy2(modality_json, dest)
            print(f"Replaced modality.json with: {modality_json}")

    print("\n" + "="*60)
    print(f"Dataset replay complete!")
    print(f"LeRobot dataset saved to: {out_dir}/{output_path}")
    print("="*60 + "\n")

    if dr_manager.is_enabled:
        dr_manager.cleanup()

    og.shutdown()

    end_stage(cfg, success=True, additional_info={"input_hdf5": input_hdf5_path, "output_dataset": f"{out_dir}/{output_path}"})


def end_stage(cfg, success=False, additional_info: dict = None):
    """
    Function that is run at the end of every stage to save the stage config and additional info.
    """
    save_dir = cfg.s18_replay.out_dir
    stage_cfg = OmegaConf.to_object(cfg.s18_replay)
    stage_cfg['success'] = success
    if additional_info is not None:
        stage_cfg.update(additional_info)
    dump_json(stage_cfg, f"{save_dir}/stage_info.json")


if __name__ == "__main__":
    main()
