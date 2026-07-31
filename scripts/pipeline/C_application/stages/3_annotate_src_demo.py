# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from simfoundry env

Annotate source demonstrations with pick/place signals for MimicGen.

Requires installing:
- BEHAVIOR-1K, see https://github.com/StanfordVL/BEHAVIOR-1K
"""

import os
import sys
import json
import subprocess as sp
from copy import deepcopy
from pathlib import Path

import h5py
import hydra
from glob import glob
import omnigibson as og
from omnigibson.macros import gm

from simfoundry import import_og_dependencies
from simfoundry.utils.annotation_utils import MinimalPlaybackWrapper, AnnotationManager
from simfoundry.utils.og_utils import apply_teleop_omnigibson_macros, setup_wrist_camera_viewport
from simfoundry.utils.processing_utils import make_json_serializable
from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths

# Needed so custom tasks can be instantiated properly
import_og_dependencies()

# Configure viewer
gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128

from simfoundry import CFG_DIR

# At the start of every script, we cd into the scripts/config directory
scripts_dir = os.path.dirname(os.path.abspath(__file__))
cfg_dir = CFG_DIR
os.chdir(cfg_dir)


def _load_config_from_hdf5(input_hdf5):
    """Extract and prepare environment config from an HDF5 file for playback."""
    config = json.loads(input_hdf5["data"].attrs["config"])

    config["env"]["action_frequency"] = 1000.0
    config["env"]["rendering_frequency"] = 1000.0
    config["env"]["physics_frequency"] = 1000.0
    config["env"]["flatten_obs_space"] = True

    config["scene"]["scene_file"] = json.loads(input_hdf5["data"].attrs["scene_file"])

    scene_file = config["scene"]["scene_file"]
    if "objects_info" in scene_file and "init_info" in scene_file["objects_info"]:
        for obj_name, obj_info in scene_file["objects_info"]["init_info"].items():
            if "args" in obj_info and "grasping_mode" in obj_info["args"]:
                obj_info["args"]["grasping_mode"] = "physical"
                print(f"Set grasping_mode to 'physical' for: {obj_name}")

    if config["task"]["type"] == "BehaviorTask":
        config["task"]["online_object_sampling"] = False
        config["task"]["use_presampled_robot_pose"] = False

    config["objects"] = []
    return config


def _annotate_single_hdf5(hdf_path, out_dir, cfg, env=None):
    """
    Run annotation on a single HDF5 file.

    Args:
        hdf_path: Path to the input HDF5 file.
        out_dir: Base output directory for annotation files. A subfolder named after
            the HDF5 stem will be created inside this directory.
        cfg: The s15_annotate config.
        env: Existing OmniGibson environment to reuse (None to create a new one).

    Returns:
        The OmniGibson environment (for reuse across files).
    """
    hdf_path = Path(hdf_path)
    hdf_stem = hdf_path.stem
    hdf_out_dir = Path(out_dir) / hdf_stem
    hdf_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nProcessing: {hdf_path}")
    print(f"Output directory: {hdf_out_dir}")

    input_hdf5 = h5py.File(str(hdf_path), "r")
    config = _load_config_from_hdf5(input_hdf5)

    robot_id = cfg.robot_id
    check_gripper_open_during_place = cfg.get("check_gripper_open_during_place", True)

    if env is None:
        apply_teleop_omnigibson_macros(enable_tr=False)
        env = og.Environment(configs=config)

        if len(env.robots[robot_id].sensors) == 1:
            setup_wrist_camera_viewport(env, robot_id=robot_id)
        else:
            print(f"WARNING: Expected exactly one sensor on robot {robot_id}, "
                  f"got {len(env.robots[robot_id].sensors)}. Skipping wrist camera viewport setup.")
    else:
        env.reload(config)

    config_output_path = hdf_out_dir / "env_config.json"
    with open(config_output_path, 'w') as f:
        json.dump(make_json_serializable(config), f, indent=2)
    print(f"Saved environment config to: {config_output_path}")

    annotation_manager = AnnotationManager(
        env,
        robot_id=robot_id,
        invert_gripper_action=False,
        check_gripper_open_during_place=check_gripper_open_during_place,
    )

    wrapper = MinimalPlaybackWrapper(
        env,
        input_hdf5,
        step_callback=annotation_manager.step,
        episode_start_callback=annotation_manager.episode_start_callback,
    )

    all_annotations = {}
    n_episodes = input_hdf5["data"].attrs["n_episodes"]
    assert n_episodes > 0, f"No episodes found in {hdf_path}!"

    for episode_id in range(n_episodes):
        print(f"\nProcessing episode {episode_id}...")
        wrapper.playback_episode(episode_id)
        signals = annotation_manager.get_annotations()
        all_annotations[f"demo_{episode_id}"] = signals
        print(f"  Found {len(signals)} signals")

    output_path = hdf_out_dir / "annotations.json"
    with open(output_path, 'w') as f:
        json.dump(all_annotations, f, indent=2)
    print(f"\nSaved annotations to: {output_path}")

    input_hdf5.close()
    return env


def _load_sampling_sidecar(hdf_path):
    """Return the load_sampling metadata dict if a sidecar exists next to the HDF5, else None."""
    sidecar_path = str(hdf_path).replace(".hdf5", "_meta.json")
    if os.path.exists(sidecar_path):
        with open(sidecar_path) as f:
            return json.load(f)
    return None


def _annotate_load_sampling_hdf5(hdf_path, out_dir, cfg, sidecar, env=None):
    """
    Annotate a load_sampling HDF5 file with per-scene env configs and annotation files.

    For each sampled scene, creates:
      - out_dir/env_config_<scene_stem>.json   (env config with that scene's scene_file)
      - out_dir/annotations_<scene_stem>.json  (annotations for the scene's episodes,
                                                with a '_source_hdf5' key for step 16)

    Args:
        hdf_path: Path to the load_sampling HDF5 from step 14.
        out_dir: Base output directory (flat files, no subdir).
        cfg: The s15_annotate config section.
        sidecar: Metadata dict from _load_sampling_sidecar().
        env: Existing OmniGibson environment to reuse (None to create new).

    Returns:
        The OmniGibson environment (for reuse).
    """
    hdf_path = Path(hdf_path)
    # Mirror the per-HDF5 subdirectory convention used by _annotate_single_hdf5
    out_dir = Path(out_dir) / hdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_scene_paths = sidecar["valid_sampling_scene_paths"]
    K = int(sidecar.get("success_demo_per_sampling", 1))
    n_scenes = len(valid_scene_paths)

    input_hdf5 = h5py.File(str(hdf_path), "r")
    base_config = _load_config_from_hdf5(input_hdf5)
    n_episodes = int(input_hdf5["data"].attrs["n_episodes"])

    robot_id = cfg.robot_id
    check_gripper_open_during_place = cfg.get("check_gripper_open_during_place", True)

    print(f"\n[load_sampling] {hdf_path.name}: {n_scenes} scenes × {K} episode(s) = {n_episodes} total episodes")

    for scene_idx, scene_path in enumerate(valid_scene_paths):
        episode_ids = list(range(scene_idx * K, min((scene_idx + 1) * K, n_episodes)))
        if not episode_ids:
            print(f"\n[load_sampling] Scene {scene_idx}: no episodes in HDF5, skipping")
            continue

        scene_stem = Path(scene_path).stem  # e.g. "scene_000"
        print(f"\n[load_sampling] Scene {scene_idx} ({scene_stem}): episodes {episode_ids}")

        # Build env config with this scene's JSON as the scene_file
        scene_json = load_json_with_absolute_usd_paths(scene_path)
        scene_config = deepcopy(base_config)
        scene_config["scene"]["scene_file"] = scene_json

        if env is None:
            apply_teleop_omnigibson_macros(enable_tr=False)
            env = og.Environment(configs=scene_config)
            if len(env.robots[robot_id].sensors) == 1:
                setup_wrist_camera_viewport(env, robot_id=robot_id)
            else:
                print(f"WARNING: Expected exactly one sensor on robot {robot_id}, "
                      f"got {len(env.robots[robot_id].sensors)}. Skipping wrist camera viewport setup.")
        # For subsequent scenes we do NOT reload the env: all load_sampling scenes share the
        # same scene structure, and playback_episode restores per-episode positions via
        # og.sim.load_state(), so the live env can stay as-is.  The per-scene env_config is
        # still saved below for downstream steps 16/17.

        # Save env_config for this scene
        config_output_path = out_dir / f"env_config_{scene_stem}.json"
        with open(config_output_path, "w") as f:
            json.dump(make_json_serializable(scene_config), f, indent=2)
        print(f"Saved env config: {config_output_path}")

        annotation_manager = AnnotationManager(
            env,
            robot_id=robot_id,
            invert_gripper_action=False,
            check_gripper_open_during_place=check_gripper_open_during_place,
        )

        wrapper = MinimalPlaybackWrapper(
            env,
            input_hdf5,
            step_callback=annotation_manager.step,
            episode_start_callback=annotation_manager.episode_start_callback,
        )
        # Override scene_file so playback restores this scene's structure, not the first scene's
        wrapper.scene_file = scene_json

        all_annotations = {}
        for episode_id in episode_ids:
            print(f"  Processing episode {episode_id} (scene {scene_stem})...")
            wrapper.playback_episode(episode_id)
            signals = annotation_manager.get_annotations()
            all_annotations[f"demo_{episode_id}"] = signals
            print(f"  Found {len(signals)} signals")

        # Store source HDF5 path so step 16 can locate it via stem lookup
        all_annotations["_source_hdf5"] = str(hdf_path.resolve())

        annotation_output_path = out_dir / f"annotations_{scene_stem}.json"
        with open(annotation_output_path, "w") as f:
            json.dump(all_annotations, f, indent=2)
        print(f"Saved annotations: {annotation_output_path}")

    input_hdf5.close()
    return env


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


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    if not cfg.s15_annotate.rerun:
        print("Skipping s15_annotate (rerun=False)")
        return

    teleop_dir = cfg.s14_teleop.out_dir
    out_dir = cfg.s15_annotate.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    target_stem = cfg.s15_annotate.get("_target_stem", None)
    filter_str = cfg.s15_annotate.get("filter", None)

    # Filter mode: spawn a subprocess per matching file and return
    # Special cases "load_sampling" and "non_load_sampling" are handled below (pick latest only)
    if filter_str and filter_str not in ("load_sampling", "non_load_sampling") and not target_stem:
        all_hdf5_files = sorted(glob(f"{teleop_dir}/*.hdf5"))
        if not all_hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in {teleop_dir}")
        matched = [f for f in all_hdf5_files if filter_str in os.path.basename(f)]
        if not matched:
            raise FileNotFoundError(
                f"No HDF5 files matching filter '{filter_str}' in {teleop_dir}. "
                f"Available: {[os.path.basename(f) for f in all_hdf5_files]}"
            )
        stems = [Path(f).stem for f in matched]
        print(f"Filter '{filter_str}' matched {len(matched)} file(s), spawning subprocesses:")
        for s in stems:
            print(f"  {s}")
        _run_filter_subprocesses(stems, "s15_annotate")
        return

    # Single-target mode (subprocess) or default mode (most recent file)
    if target_stem:
        hdf_path = os.path.join(teleop_dir, f"{target_stem}.hdf5")
        if not os.path.exists(hdf_path):
            raise FileNotFoundError(f"Target HDF5 not found: {hdf_path}")
        files_to_process = [hdf_path]
        print(f"Processing single target: {target_stem}")
    else:
        all_hdf5_files = sorted(glob(f"{teleop_dir}/*.hdf5"))
        if not all_hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in {teleop_dir}")
        
        # If filter is "load_sampling", find latest load_sampling file
        # If filter is "non_load_sampling" or not set, find latest non-load_sampling file
        if filter_str == "load_sampling":
            candidate_files = [f for f in all_hdf5_files if "_load_sampling" in os.path.basename(f)]
            mode_desc = "load_sampling"
        else:
            candidate_files = [f for f in all_hdf5_files if "_load_sampling" not in os.path.basename(f)]
            mode_desc = "non-load_sampling"
        
        if not candidate_files:
            raise FileNotFoundError(
                f"No {mode_desc} HDF5 files found in {teleop_dir}. "
                f"Available: {[os.path.basename(f) for f in all_hdf5_files]}"
            )
        
        hdf_path = max(candidate_files, key=os.path.getmtime)
        files_to_process = [hdf_path]
        print(f"Using most recent {mode_desc} HDF5: {os.path.basename(hdf_path)}")

    env = None
    try:
        for hdf_path in files_to_process:
            sidecar = _load_sampling_sidecar(hdf_path)
            if sidecar is not None:
                print(f"[load_sampling] Detected sidecar for {os.path.basename(hdf_path)}, using per-scene annotation mode")
                env = _annotate_load_sampling_hdf5(hdf_path, out_dir, cfg.s15_annotate, sidecar, env=env)
            else:
                env = _annotate_single_hdf5(hdf_path, out_dir, cfg.s15_annotate, env=env)
    finally:
        if env is not None:
            try:
                og.clear()
            except KeyError as e:
                print(f"Warning: Cleanup error (can be safely ignored): {e}")
            og.shutdown()

    print("\n" + "="*80)
    print("Stage 15 complete!")
    print("="*80)


if __name__ == "__main__":
    main()
