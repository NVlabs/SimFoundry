# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from simfoundry env

Extract object-centric waypoints from source demonstrations.

Requires installing:
- BEHAVIOR-1K, see https://github.com/StanfordVL/BEHAVIOR-1K
"""

import os
import sys
import json
import subprocess as sp
from pathlib import Path

from glob import glob
import h5py
import hydra
import numpy as np
import omnigibson as og
from omnigibson.macros import gm

from simfoundry import import_og_dependencies
from simfoundry.utils.annotation_utils import MinimalPlaybackWrapper, WaypointExtractor
from simfoundry.utils.og_utils import apply_teleop_omnigibson_macros, setup_wrist_camera_viewport

# Needed so custom tasks can be instantiated properly
import_og_dependencies()

# Configure viewer
gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128

from simfoundry import CFG_DIR

# Expected action sequences for each task
TASK_EXPECTED_SEQUENCES = {
    'serve_fruit': ['pick', 'place'],
    'hide_banana': ['open', 'pick', 'place', 'close'],
    'yam_workstation_stack_bowls_minimal': ['pick', 'place'],
    'yam_workstation_stack_bowls_rescaled_minimal': ['pick', 'place'],
    'yam_workstation_stack_bowls': ['pick', 'place', 'pick', 'place'],
    'droid_desk_stack_dishware': ['pick', 'place', 'pick', 'place'],
    'droid_desk_put_away_trash': ['pick', 'open', 'pick', 'place', 'close'],
    'droid_desk_put_away_marker': ['open', 'pick', 'place', 'close'],
    'place_eraser_on_tray': ['pick', 'place'],
    'nv_desk_place_baseball_in_bowl': ['pick', 'place'],
    'nv_desk_place_black_eraser_on_organizer': ['pick', 'place'],
    'nv_desk_place_black_marker_on_plate': ['pick', 'place'],
    'nv_desk_place_burger_in_bowl': ['pick', 'place'],
    'nv_desk_place_orange_marker_on_organizer': ['pick', 'place'],
    'nv_desk_place_red_bottle_in_bowl': ['pick', 'place'],
    'nv_desk_place_red_marker_on_organizer': ['pick', 'place'],
    'nv_desk_place_red_marker_on_teal_tray': ['pick', 'place'],
    'nv_desk_place_tennis_ball_in_yellow_bowl': ['pick', 'place'],
    'nv_desk_place_tennis_ball_on_organizer': ['pick', 'place'],
    'nv_desk_place_red_marker_in_blue_cup': ['pick', 'place'],
    'nv_desk_place_black_eraser_in_bowl': ['pick', 'place'],
    'nv_desk_place_orange_cup_in_bowl': ['pick', 'place'],
    'nv_desk_place_orange_cup_on_organizer': ['pick', 'place']
}


def find_subsequence(signals, expected_sequence):
    """
    Find the expected subsequence in the list of signals.

    Args:
        signals: List of signal dicts with 'type' field
        expected_sequence: List of expected signal types in order

    Returns:
        Tuple of (start_idx, matching_signals) if found, None otherwise
    """
    signal_types = [s['type'] for s in signals]

    # Try to find the expected subsequence starting at each position
    for start_idx in range(len(signal_types)):
        # Check if we have enough signals remaining
        if start_idx + len(expected_sequence) > len(signal_types):
            break

        # Check if subsequence matches
        match = True
        for i, expected_type in enumerate(expected_sequence):
            if signal_types[start_idx + i] != expected_type:
                match = False
                break

        if match:
            # Found the subsequence!
            matching_signals = signals[start_idx:start_idx + len(expected_sequence)]
            return (start_idx, matching_signals)

    return None


# At the start of every script, we cd into the scripts/config directory
scripts_dir = os.path.dirname(os.path.abspath(__file__))
cfg_dir = CFG_DIR
os.chdir(cfg_dir)


def _save_waypoints_hdf5(all_waypoints, output_path):
    """Save extracted waypoints to an HDF5 file."""
    print(f"\nSaving waypoints to: {output_path}")

    with h5py.File(output_path, 'w') as f:
        data_grp = f.create_group("data")
        data_grp.attrs["n_episodes"] = len(all_waypoints)

        print("\n" + "="*80)
        print("WAYPOINT EXTRACTION SUMMARY")
        print("="*80)

        for episode_key, waypoints in all_waypoints.items():
            print(f"{episode_key}: {len(waypoints)} subtasks")

            ep_grp = data_grp.create_group(episode_key)
            ep_grp.attrs["n_subtasks"] = len(waypoints)

            for i, subtask in enumerate(waypoints):
                print(f"  Subtask {i}: {subtask['type']} with {len(subtask['waypoints'])} waypoints")

                subtask_grp = ep_grp.create_group(f"subtask_{i}")
                subtask_grp.attrs["type"] = subtask["type"]
                subtask_grp.attrs["signal_frame_idx"] = subtask["signal_frame_idx"]
                subtask_grp.attrs["actual_frames_before"] = subtask["actual_frames_before"]
                subtask_grp.attrs["actual_frames_after"] = subtask["actual_frames_after"]
                subtask_grp.attrs["extraction_params"] = json.dumps(subtask["extraction_params"])
                subtask_grp.attrs["reference_object"] = json.dumps(subtask["reference_object"])

                if "placed_object" in subtask:
                    subtask_grp.attrs["placed_object"] = json.dumps(subtask["placed_object"])

                frame_indices = np.array([w["frame_idx"] for w in subtask["waypoints"]], dtype=np.int32)
                eef_pos = np.array([w["eef_pos_obj"] for w in subtask["waypoints"]], dtype=np.float32)
                eef_quat = np.array([w["eef_quat_obj"] for w in subtask["waypoints"]], dtype=np.float32)
                gripper = np.array([w["gripper_action"] for w in subtask["waypoints"]], dtype=np.float32)

                subtask_grp.create_dataset("frame_indices", data=frame_indices)
                subtask_grp.create_dataset("eef_pos_obj", data=eef_pos)
                subtask_grp.create_dataset("eef_quat_obj", data=eef_quat)
                subtask_grp.create_dataset("gripper_action", data=gripper)

    print(f"\nSaved waypoints to: {output_path}")


def _extract_waypoints_single(config_path, annotation_path, hdf_path, cfg, env=None):
    """
    Extract waypoints from a single annotation + HDF5 pair.

    Args:
        config_path: Path to env_config JSON from stage 15.
        annotation_path: Path to annotations JSON from stage 15.
        hdf_path: Path to source HDF5 from stage 14.
        cfg: Full Hydra config.
        env: Existing OmniGibson environment to reuse (None to create new).

    Returns:
        Tuple of (env, extracted_waypoints_dict) where extracted_waypoints_dict maps
        episode keys to their waypoint data.
    """
    hdf_path = Path(hdf_path)
    print(f"\n{'='*80}")
    print(f"Processing dataset: {hdf_path.stem}")
    print(f"  Config: {config_path}")
    print(f"  Annotations: {annotation_path}")
    print(f"  Episode data: {hdf_path}")
    print(f"{'='*80}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    with open(annotation_path, 'r') as f:
        all_annotations = json.load(f)

    robot_id = cfg.s15_annotate.robot_id

    input_hdf5 = h5py.File(str(hdf_path), "r")

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

    extracted_waypoints = {}
    n_episodes = input_hdf5["data"].attrs["n_episodes"]
    assert n_episodes > 0, f"No episodes found in {hdf_path}!"

    # Only process episodes that have annotations (handles load_sampling where
    # each scene's annotation file only covers specific episodes)
    annotated_episodes = [k for k in all_annotations.keys() if k.startswith("demo_")]
    if not annotated_episodes:
        print(f"  No annotated episodes found in {annotation_path}")
        input_hdf5.close()
        return env, extracted_waypoints

    for episode_key in sorted(annotated_episodes):
        # Extract episode ID from key (e.g., "demo_0" -> 0)
        episode_id = int(episode_key.split("_")[1])

        if episode_key not in input_hdf5["data"]:
            print(f"  Episode {episode_key} not found in HDF5, skipping")
            continue

        episode_annotations = all_annotations[episode_key]
        if not episode_annotations:
            print(f"No signals found in {episode_key}, skipping")
            continue

        task_name = cfg.task.task_name
        expected_sequence = TASK_EXPECTED_SEQUENCES.get(task_name, None)
        if expected_sequence is None:
            print(f"WARNING: No expected sequence defined for task '{task_name}', skipping {episode_key}")
            continue

        result = find_subsequence(episode_annotations, expected_sequence)
        if result is None:
            signal_types = [s['type'] for s in episode_annotations]
            print(f"Episode {episode_key} does not contain expected sequence {expected_sequence}")
            print(f"  Found signals: {signal_types}")
            print(f"  Skipping this episode")
            continue

        start_idx, matching_signals = result
        print(f"\nProcessing {episode_key}:")
        print(f"  Found expected sequence {expected_sequence} starting at signal index {start_idx}")
        print(f"  Using {len(matching_signals)} signals for waypoint extraction")

        episode_grp = input_hdf5["data"][episode_key]
        max_frames = episode_grp.attrs.get("num_samples", None)

        waypoint_extractor = WaypointExtractor(
            env,
            signals=matching_signals,
            pick_min_start_distance=cfg.s16_waypoints.pick_min_start_distance,
            pick_max_distance=cfg.s16_waypoints.pick_max_distance,
            place_min_start_distance=cfg.s16_waypoints.place_min_start_distance,
            place_max_distance=cfg.s16_waypoints.place_max_distance,
            open_min_start_distance=cfg.s16_waypoints.open_min_start_distance,
            open_max_distance=cfg.s16_waypoints.open_max_distance,
            close_min_start_distance=cfg.s16_waypoints.close_min_start_distance,
            close_max_distance=cfg.s16_waypoints.close_max_distance,
            end_z_threshold=cfg.s16_waypoints.end_z_threshold,
            max_frames=max_frames,
            robot_id=robot_id,
            eef_z_offset=cfg.s16_waypoints.eef_z_offset,
            merge_sequences=cfg.s16_waypoints.merge_sequences,
            use_open_grasp_signal=cfg.s16_waypoints.use_open_grasp_signal,
            open_grasp_max_distance=cfg.s16_waypoints.open_grasp_max_distance,
        )

        wrapper = MinimalPlaybackWrapper(
            env,
            input_hdf5,
            step_callback=waypoint_extractor.step,
            episode_start_callback=waypoint_extractor.episode_start_callback,
        )

        wrapper.playback_episode(episode_id)
        waypoint_extractor.finalize()
        waypoints = waypoint_extractor.get_waypoints()

        extracted_waypoints[episode_key] = waypoints
        print(f"  Extracted {len(waypoints)} subtask trajectories")

    input_hdf5.close()
    return env, extracted_waypoints


def _resolve_dataset_triples(teleop_dir, annotation_dir, filter_str):
    """
    Resolve (config_path, annotation_path, hdf_path, stem) tuples for processing.

    When filter_str is set, finds all annotation files matching the filter and pairs
    each with its corresponding env_config and source HDF5.
    
    Special cases:
    - filter_str="load_sampling": finds the latest _load_sampling.hdf5 and its annotations
    - filter_str="non_load_sampling": finds the latest non-load_sampling HDF5 and its annotations

    Returns:
        List of (config_path, annotation_path, hdf_path, stem_or_None) tuples.
    """
    # Special handling for load_sampling / non_load_sampling: find latest HDF5 first
    if filter_str in ("load_sampling", "non_load_sampling"):
        all_hdf5_files = sorted(glob(f"{teleop_dir}/*.hdf5"))
        if not all_hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in {teleop_dir}")
        
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
        
        # Get the latest HDF5 file
        latest_hdf5 = max(candidate_files, key=os.path.getmtime)
        latest_stem = Path(latest_hdf5).stem
        print(f"Using most recent {mode_desc} HDF5: {os.path.basename(latest_hdf5)}")
        
        # Now search for annotations in the subdirectory matching this HDF5 stem
        annotation_subdir = Path(annotation_dir) / latest_stem
        if not annotation_subdir.exists():
            raise FileNotFoundError(
                f"Annotation directory not found for {latest_stem}: {annotation_subdir}. "
                f"Please run stage 15 first."
            )
        
        # Find all annotation files in this subdirectory
        # Support both per-scene layout (annotations_<scene>.json) and plain layout (annotations.json)
        annotation_files = sorted(glob(f"{annotation_subdir}/annotations_*.json"))
        if annotation_files:
            matched = []
            for ann_path in annotation_files:
                basename = os.path.basename(ann_path)
                stem = basename.replace("annotations_", "").replace(".json", "")
                config_path = annotation_subdir / f"env_config_{stem}.json"
                if not config_path.exists():
                    print(f"WARNING: Missing env_config for {stem}: {config_path}, skipping")
                    continue
                matched.append((str(config_path), str(ann_path), str(latest_hdf5), stem))
        else:
            # Plain layout from stage 15: annotations.json + env_config.json
            plain_ann = annotation_subdir / "annotations.json"
            plain_cfg = annotation_subdir / "env_config.json"
            if not plain_ann.exists() or not plain_cfg.exists():
                raise FileNotFoundError(
                    f"No annotation files found in {annotation_subdir}. Please run stage 15 first."
                )
            matched = [(str(plain_cfg), str(plain_ann), str(latest_hdf5), latest_stem)]
        
        if not matched:
            raise FileNotFoundError(
                f"No valid annotation/config pairs found in {annotation_subdir}"
            )
        
        print(f"Found {len(matched)} scene(s) in {annotation_subdir.name}:")
        for _, _, _, stem in matched:
            print(f"  {stem}")
        return matched
    
    elif filter_str:
        # Search both flat (legacy) and per-HDF5 subdirectory layouts
        annotation_files = sorted(
            glob(f"{annotation_dir}/annotations_*.json") +
            glob(f"{annotation_dir}/*/annotations_*.json")
        )
        matched = []
        for ann_path in annotation_files:
            basename = os.path.basename(ann_path)
            stem = basename.replace("annotations_", "").replace(".json", "")
            # Match filter against stem or parent subdirectory name
            parent_dir_name = os.path.basename(os.path.dirname(ann_path))
            if filter_str not in stem and filter_str not in parent_dir_name:
                continue
            # env_config lives in the same directory as the annotation file
            ann_dir = os.path.dirname(ann_path)
            config_path = Path(ann_dir) / f"env_config_{stem}.json"
            # For load_sampling annotations, _source_hdf5 points to the actual HDF5 file
            with open(ann_path) as _f:
                _ann_data = json.load(_f)
            source_hdf5 = _ann_data.get("_source_hdf5", None)
            if source_hdf5 and os.path.exists(source_hdf5):
                hdf_path = Path(source_hdf5)
            else:
                hdf_path = Path(teleop_dir) / f"{stem}.hdf5"
            if not config_path.exists():
                print(f"WARNING: Missing env_config for {stem}: {config_path}, skipping")
                continue
            if not hdf_path.exists():
                print(f"WARNING: Missing HDF5 for {stem}: {hdf_path}, skipping")
                continue
            matched.append((str(config_path), str(ann_path), str(hdf_path), stem))

        if not matched:
            all_annotations = [os.path.basename(f) for f in annotation_files]
            raise FileNotFoundError(
                f"No annotation files matching filter '{filter_str}' in {annotation_dir}. "
                f"Available: {all_annotations}"
            )
        print(f"Filter '{filter_str}' matched {len(matched)} dataset(s):")
        for _, _, hdf, stem in matched:
            print(f"  {stem}")
        return matched
    else:
        config_path = Path(annotation_dir) / "env_config.json"
        annotation_path = Path(annotation_dir) / "annotations.json"

        if config_path.exists() and annotation_path.exists():
            # Flat/legacy layout: files directly in annotation_dir
            hdf5_files = glob(f"{teleop_dir}/*.hdf5")
            if not hdf5_files:
                raise FileNotFoundError(f"No HDF5 files found in {teleop_dir}")
            hdf_path = max(hdf5_files, key=os.path.getmtime)
            print(f"No filter set — using default files:")
            print(f"  Config: {config_path}")
            print(f"  Annotations: {annotation_path}")
            print(f"  Episode data: {os.path.basename(hdf_path)}")
            return [(str(config_path), str(annotation_path), hdf_path, None)]

        # Per-HDF5 subdirectory layout: find latest HDF5 and its matching subdirectory
        hdf5_files = glob(f"{teleop_dir}/*.hdf5")
        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in {teleop_dir}")
        hdf_path = max(hdf5_files, key=os.path.getmtime)
        hdf_stem = Path(hdf_path).stem

        subdir = Path(annotation_dir) / hdf_stem
        sub_config = subdir / "env_config.json"
        sub_annotation = subdir / "annotations.json"
        if sub_config.exists() and sub_annotation.exists():
            print(f"No filter set — using per-HDF5 subdirectory layout:")
            print(f"  Config: {sub_config}")
            print(f"  Annotations: {sub_annotation}")
            print(f"  Episode data: {os.path.basename(hdf_path)}")
            return [(str(sub_config), str(sub_annotation), hdf_path, hdf_stem)]

        raise FileNotFoundError(
            f"Environment config not found. Checked:\n"
            f"  Flat layout: {config_path}\n"
            f"  Subdirectory layout: {sub_config}\n"
            f"Please run stage 15 first."
        )


def _run_extraction_subprocesses(datasets, out_dir):
    """
    Spawn a subprocess for each dataset to extract waypoints.

    Each subprocess runs in single-scene mode and saves to a temp file.

    Args:
        datasets: List of (config_path, annotation_path, hdf_path, stem) tuples.
        out_dir: Output directory for temp files.

    Returns:
        Dict mapping stem to temp file path (or None if failed).
    """
    script_path = os.path.join(scripts_dir, os.path.basename(__file__))
    # Strip filter args - we'll pass explicit paths via temp file instead
    base_overrides = [
        arg for arg in sys.argv[1:]
        if not arg.startswith('s16_waypoints.filter=')
        and not arg.lstrip('+').startswith('s16_waypoints._dataset_file=')
    ]

    results = {}
    for config_path, annotation_path, hdf_path, stem in datasets:
        # Write dataset info to a temp JSON file (avoids Hydra parsing issues with JSON on command line)
        dataset_file = os.path.join(out_dir, f"_dataset_info_{stem}.json")
        with open(dataset_file, 'w') as f:
            json.dump({
                'config_path': config_path,
                'annotation_path': annotation_path,
                'hdf_path': hdf_path,
                'stem': stem,
            }, f)

        cmd = [sys.executable, script_path] + base_overrides + [f"++s16_waypoints._dataset_file={dataset_file}"]
        print(f"\n{'#'*80}")
        print(f"[SUBPROCESS] Processing: {stem}")
        print(f"{'#'*80}\n")
        result = sp.run(cmd)

        # Clean up dataset info file
        if os.path.exists(dataset_file):
            os.remove(dataset_file)

        temp_path = os.path.join(out_dir, f"_temp_waypoints_{stem}.hdf5")
        if result.returncode == 0 and os.path.exists(temp_path):
            results[stem] = temp_path
        else:
            print(f"\nWARNING: Subprocess for '{stem}' failed (exit code {result.returncode})")
            results[stem] = None

    n_ok = sum(1 for v in results.values() if v is not None)
    print(f"\n{'='*80}")
    print(f"SUBPROCESS SUMMARY: {n_ok}/{len(results)} succeeded")
    for stem, path in results.items():
        status = "OK" if path else "FAILED"
        print(f"  {stem}: {status}")
    print(f"{'='*80}")

    return results


def _load_and_rename_waypoints(hdf5_path, stem):
    """
    Load waypoints from an HDF5 file and rename episode keys with stem suffix.

    Args:
        hdf5_path: Path to waypoints HDF5 file.
        stem: Scene stem to append to episode keys (e.g., "scene_000").

    Returns:
        Dict mapping renamed episode keys to waypoint data.
    """
    waypoints = {}
    with h5py.File(hdf5_path, 'r') as f:
        data_grp = f['data']
        for episode_key in data_grp.keys():
            ep_grp = data_grp[episode_key]
            n_subtasks = ep_grp.attrs['n_subtasks']

            subtasks = []
            for i in range(n_subtasks):
                subtask_grp = ep_grp[f'subtask_{i}']
                subtask_data = {
                    'type': subtask_grp.attrs['type'],
                    'signal_frame_idx': int(subtask_grp.attrs['signal_frame_idx']),
                    'actual_frames_before': int(subtask_grp.attrs['actual_frames_before']),
                    'actual_frames_after': int(subtask_grp.attrs['actual_frames_after']),
                    'extraction_params': json.loads(subtask_grp.attrs['extraction_params']),
                    'reference_object': json.loads(subtask_grp.attrs['reference_object']),
                }
                if 'placed_object' in subtask_grp.attrs:
                    subtask_data['placed_object'] = json.loads(subtask_grp.attrs['placed_object'])

                # Load waypoint arrays and convert to list format
                frame_indices = subtask_grp['frame_indices'][:]
                eef_pos = subtask_grp['eef_pos_obj'][:]
                eef_quat = subtask_grp['eef_quat_obj'][:]
                gripper = subtask_grp['gripper_action'][:]

                waypoints_list = []
                for j in range(len(frame_indices)):
                    waypoints_list.append({
                        'frame_idx': int(frame_indices[j]),
                        'eef_pos_obj': eef_pos[j].tolist(),
                        'eef_quat_obj': eef_quat[j].tolist(),
                        'gripper_action': float(gripper[j]),
                    })
                subtask_data['waypoints'] = waypoints_list
                subtasks.append(subtask_data)

            # Rename episode key with stem suffix
            new_key = f"{episode_key}_{stem}"
            waypoints[new_key] = subtasks

    return waypoints


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    if not cfg.s16_waypoints.rerun:
        print("Skipping s16_waypoints (rerun=False)")
        return

    teleop_dir = cfg.s14_teleop.out_dir
    annotation_dir = cfg.s15_annotate.out_dir
    out_dir = cfg.s16_waypoints.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    filter_str = cfg.s16_waypoints.get("filter", None)
    dataset_file = cfg.s16_waypoints.get("_dataset_file", None)

    # Check if we're in subprocess mode (single dataset passed via file)
    if dataset_file:
        # Subprocess mode: read dataset info from temp file
        with open(dataset_file, 'r') as f:
            dataset_info = json.load(f)
        config_path = dataset_info['config_path']
        annotation_path = dataset_info['annotation_path']
        hdf_path = dataset_info['hdf_path']
        stem = dataset_info['stem']

        env = None
        try:
            env, all_waypoints = _extract_waypoints_single(
                config_path, annotation_path, hdf_path, cfg,
            )
            output_path = Path(out_dir) / f"_temp_waypoints_{stem}.hdf5"
            _save_waypoints_hdf5(all_waypoints, output_path)
        finally:
            if env is not None:
                try:
                    og.clear()
                except KeyError as e:
                    print(f"Warning: Cleanup error (can be safely ignored): {e}")
                og.shutdown()

        print("\n" + "="*80)
        print(f"Stage 16 subprocess complete: {stem}")
        print(f"Output: {output_path}")
        print("="*80)
        return

    # Resolve datasets to process
    datasets = _resolve_dataset_triples(teleop_dir, annotation_dir, filter_str)

    # Determine if we're in consolidated mode (filter with multiple scenes)
    consolidate_mode = filter_str is not None and len(datasets) > 1

    if consolidate_mode:
        # Parent process: spawn subprocesses for each scene, then consolidate
        print(f"\n{'#'*80}")
        print(f"CONSOLIDATED MODE: Processing {len(datasets)} scene(s) via subprocesses")
        print(f"{'#'*80}")

        temp_files = _run_extraction_subprocesses(datasets, out_dir)

        # Consolidate all temp waypoint files into one
        all_waypoints = {}
        for stem, temp_path in temp_files.items():
            if temp_path and os.path.exists(temp_path):
                scene_waypoints = _load_and_rename_waypoints(temp_path, stem)
                all_waypoints.update(scene_waypoints)
                os.remove(temp_path)  # Clean up temp file

        output_path = Path(out_dir) / "waypoints.hdf5"
        _save_waypoints_hdf5(all_waypoints, output_path)

        print("\n" + "="*80)
        print("Stage 16 complete!")
        print(f"Consolidated {len(datasets)} scene(s) -> {len(all_waypoints)} episode(s)")
        print(f"Output: {output_path}")
        print("="*80)

    else:
        # Single scene mode (no filter or single match)
        env = None
        all_waypoints = {}

        try:
            for config_path, annotation_path, hdf_path, stem in datasets:
                env, extracted = _extract_waypoints_single(
                    config_path, annotation_path, hdf_path, cfg,
                    env=env,
                )
                all_waypoints.update(extracted)

            output_path = Path(out_dir) / "waypoints.hdf5"
            _save_waypoints_hdf5(all_waypoints, output_path)

        finally:
            if env is not None:
                try:
                    og.clear()
                except KeyError as e:
                    print(f"Warning: Cleanup error (can be safely ignored): {e}")
                og.shutdown()

        print("\n" + "="*80)
        print("Stage 16 complete!")
        print(f"Output: {output_path}")
        print("="*80)


if __name__ == "__main__":
    main()
