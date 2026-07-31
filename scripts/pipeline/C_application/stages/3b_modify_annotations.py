# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from simfoundry env

Interactively review and modify annotations from step 15.

Plays back episodes from step 14, displays existing annotations, and allows
the user to keep/remove annotations or add new ones via keyboard interaction.

Controls:
  - When an annotation is found: prompted in terminal to keep or remove
  - SPACE: pause playback and add a new annotation at the current frame
  - C: skip the rest of the current episode and move to the next one

Requires installing:
- BEHAVIOR-1K, see https://github.com/StanfordVL/BEHAVIOR-1K
"""

import os
import sys
import json
import shutil
import time
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

import h5py
import hydra
from glob import glob
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.robots import BaseRobot
from omnigibson.utils.ui_utils import KeyboardEventHandler

from simfoundry import import_og_dependencies, REPO_DIR, CFG_DIR as SIMFOUNDRY_CFG_DIR
from omnigibson.utils.config_utils import parse_config
from simfoundry.utils.annotation_utils import (
    MinimalPlaybackWrapper,
    PickSignal,
    PlaceSignal,
    OpenSignal,
    CloseSignal,
)
from simfoundry.utils.og_utils import apply_teleop_omnigibson_macros, setup_wrist_camera_viewport
from simfoundry.utils.processing_utils import make_json_serializable

import_og_dependencies()

# Parse script-specific args before Hydra consumes sys.argv.
# add_help=False so we don't conflict with Hydra's --help / -h.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--episode_ids", type=str, default=None,
    help="Comma-separated episode IDs to review (e.g. '0,1,2'). Default: all episodes.",
)
_parser.add_argument(
    "--playback_fps", type=float, default=30.0,
    help="Playback speed in FPS for episode visualization. Default: 30.",
)
_parser.add_argument(
    "--annotations_file", type=str, default=None,
    help="Name of annotation JSON file in the s15 output dir to modify "
         "(e.g. 'annotations_cousin_combo_001.json'). Default: 'annotations.json'.",
)
_parser.add_argument(
    "--load_sampling_dir", type=str, default=None,
    help="Directory name (inside s15 output dir) containing load_sampling per-scene annotation files "
         "(e.g. 'nv_desk_place_black_marker_on_plate_2026-03-10_22-12-56_load_sampling'). "
         "If specified, will iterate through all annotations_scene_XXX.json files.",
)
_parser.add_argument(
    "--scene_ids", type=str, default=None,
    help="When using --load_sampling_dir, comma-separated scene IDs to review (e.g. '0,2,5'). "
         "Default: all scenes.",
)
_script_args, _hydra_argv = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _hydra_argv

gm.RENDER_VIEWER_CAMERA = True
gm.DEFAULT_VIEWER_WIDTH = 960
gm.DEFAULT_VIEWER_HEIGHT = 540

from simfoundry import CFG_DIR

scripts_dir = os.path.dirname(os.path.abspath(__file__))
cfg_dir = CFG_DIR
os.chdir(cfg_dir)

ANNOTATION_TYPES = ["pick", "place", "open", "close"]


class EpisodeSkipException(Exception):
    """Raised by the step callback to cut the current episode short."""
    pass


def get_annotation_frame(ann):
    """Return the primary frame index for an annotation."""
    if ann["type"] == "pick":
        return ann["frame_idx"]
    elif ann["type"] == "place":
        return ann["contact_frame_idx"]
    elif ann["type"] in ("open", "close"):
        return ann.get("contact_frame_idx", ann.get("release_frame_idx", 0))
    return -1


def format_annotation(ann):
    """Return a human-readable string for an annotation."""
    lines = []
    for k, v in ann.items():
        lines.append(f"    {k}: {v}")
    return "\n".join(lines)


def build_frame_to_annotation_map(annotations):
    """Map frame indices to their annotation list indices."""
    mapping = {}
    for i, ann in enumerate(annotations):
        frame = get_annotation_frame(ann)
        mapping.setdefault(frame, []).append(i)
    return mapping


class AnnotationModifier:
    """Manages interactive annotation review/modification during episode playback."""

    def __init__(self, env, episode_annotations, episode_key, robot_id=0, playback_fps=30):
        self.env = env
        self.episode_annotations = list(episode_annotations)
        self.episode_key = episode_key
        self.playback_fps = playback_fps
        self.frame_delay = 1.0 / playback_fps if playback_fps > 0 else 0

        self.indices_to_remove = set()
        self.annotations_to_add = []
        self.space_pressed = False
        self.skip_requested = False
        self.current_frame = 0

        self.frame_to_ann_indices = build_frame_to_annotation_map(self.episode_annotations)

        self.scene_objects = [
            obj for obj in env.scene.objects if not isinstance(obj, BaseRobot)
        ]

    def on_space_pressed(self):
        self.space_pressed = True

    def on_c_pressed(self):
        self.skip_requested = True

    def step(self, frame_idx, action, env=None):
        if self.skip_requested:
            raise EpisodeSkipException()

        self.current_frame = frame_idx

        # Check for existing annotations at this frame
        if frame_idx in self.frame_to_ann_indices:
            for ann_idx in self.frame_to_ann_indices[frame_idx]:
                if ann_idx in self.indices_to_remove:
                    continue
                ann = self.episode_annotations[ann_idx]
                self._prompt_keep_or_remove(ann_idx, ann)

        # Check if SPACE was pressed (flag set during previous og.sim.step())
        if self.space_pressed:
            self.space_pressed = False
            self._prompt_add_annotation(frame_idx)

        # Throttle playback so the user can see what's happening
        if self.frame_delay > 0:
            time.sleep(self.frame_delay)

    def episode_start_callback(self, episode_id, env):
        print(f"\n{'='*70}")
        print(f"  Reviewing {self.episode_key}  ({len(self.episode_annotations)} annotations)")
        print(f"  SPACE = add annotation at current frame  |  C = skip to next episode")
        print(f"{'='*70}")

    def _prompt_keep_or_remove(self, ann_idx, ann):
        frame = get_annotation_frame(ann)
        print(f"\n{'='*60}")
        print(f"  ANNOTATION at frame {frame}  [{self.episode_key}]")
        print(format_annotation(ann))
        print(f"{'='*60}")
        response = input("  Keep this annotation? [Y/n]: ").strip().lower()
        if response == "n":
            self.indices_to_remove.add(ann_idx)
            print("  >>> REMOVED")
        else:
            print("  >>> KEPT")

    def _prompt_add_annotation(self, frame_idx):
        print(f"\n{'='*60}")
        print(f"  ADD NEW ANNOTATION at frame {frame_idx}  [{self.episode_key}]")
        print(f"{'='*60}")
        print("  Annotation types:")
        for i, t in enumerate(ANNOTATION_TYPES):
            print(f"    [{i}] {t}")
        print(f"    [q] Cancel")
        choice = input("  Select type: ").strip().lower()

        type_map = {str(i): t for i, t in enumerate(ANNOTATION_TYPES)}
        if choice not in type_map:
            print("  >>> Cancelled")
            return
        ann_type = type_map[choice]

        # Show scene objects
        print("\n  Scene objects:")
        for i, obj in enumerate(self.scene_objects):
            print(f"    [{i}] {obj.name}  (category: {obj.category})")

        ref_input = input(f"  Reference object ID for '{ann_type}': ").strip()
        try:
            ref_obj = self.scene_objects[int(ref_input)]
        except (ValueError, IndexError):
            print("  >>> Invalid object ID, cancelled")
            return

        if ann_type == "pick":
            ann = asdict(PickSignal(
                frame_idx=frame_idx,
                object_name=ref_obj.name,
                object_category=ref_obj.category,
            ))
        elif ann_type == "place":
            print("\n  For 'place', also specify the object being placed:")
            released_input = input("  Released/placed object ID: ").strip()
            try:
                released_obj = self.scene_objects[int(released_input)]
            except (ValueError, IndexError):
                print("  >>> Invalid object ID, cancelled")
                return
            ann = asdict(PlaceSignal(
                release_frame_idx=frame_idx,
                contact_frame_idx=frame_idx,
                released_object_name=released_obj.name,
                released_object_category=released_obj.category,
                target_object_name=ref_obj.name,
                target_object_category=ref_obj.category,
            ))
        elif ann_type in ("open", "close"):
            link_names = list(ref_obj.links.keys())
            print(f"\n  Links for '{ref_obj.name}':")
            for i, ln in enumerate(link_names):
                print(f"    [{i}] {ln}")
            link_input = input("  Contact link ID: ").strip()
            try:
                contact_link = link_names[int(link_input)]
            except (ValueError, IndexError):
                print("  >>> Invalid link ID, cancelled")
                return
            SignalCls = OpenSignal if ann_type == "open" else CloseSignal
            ann = asdict(SignalCls(
                contact_frame_idx=frame_idx,
                release_frame_idx=frame_idx,
                object_name=ref_obj.name,
                object_category=ref_obj.category,
                contact_link=contact_link,
            ))
        else:
            return

        self.annotations_to_add.append(ann)
        print(f"\n  Added annotation:")
        print(format_annotation(ann))
        print("  >>> ADDED")

    def get_updated_annotations(self):
        """Return the final annotation list with removals applied and additions merged."""
        result = [
            ann for i, ann in enumerate(self.episode_annotations)
            if i not in self.indices_to_remove
        ]
        result.extend(self.annotations_to_add)
        result.sort(key=get_annotation_frame)
        return result


def _process_single_annotation_file(
    annotation_path, config_path, hdf_path, episode_ids, playback_fps, cfg, robot_id, env=None
):
    """
    Process a single annotation file: review, modify, and save.
    
    Args:
        annotation_path: Path to the annotation JSON file.
        config_path: Path to the environment config JSON file.
        hdf_path: Path to the HDF5 episode data.
        episode_ids: List of episode IDs to review, or None for all.
        playback_fps: Playback speed in FPS.
        cfg: Hydra config.
        robot_id: Robot ID for annotation.
        env: Existing OmniGibson environment to reuse (None to create new).
    
    Returns:
        The OmniGibson environment (for reuse).
    """
    print(f"\n{'='*80}")
    print(f"Processing: {annotation_path.name}")
    print(f"Environment config: {config_path}")
    print(f"Using episode data from: {hdf_path}")
    print(f"{'='*80}")

    # --- Backup annotations with wallclock timestamp ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = annotation_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_name = f"{annotation_path.stem}_backup_{timestamp}.json"
    backup_path = backup_dir / backup_name
    shutil.copy2(str(annotation_path), str(backup_path))
    print(f"Backed up annotations to: {backup_path}")

    # Load annotations
    with open(annotation_path, "r") as f:
        all_annotations = json.load(f)

    # Load environment config from step 15
    with open(config_path, "r") as f:
        config = json.load(f)

    input_hdf5 = h5py.File(str(hdf_path), "r")
    n_episodes = input_hdf5["data"].attrs["n_episodes"]

    # Determine which episodes to review
    if episode_ids is not None:
        review_episode_ids = episode_ids
    else:
        # For load_sampling, only review episodes that exist in this annotation file
        review_episode_ids = []
        for key in all_annotations.keys():
            if key.startswith("demo_"):
                try:
                    eid = int(key.split("_")[1])
                    review_episode_ids.append(eid)
                except (IndexError, ValueError):
                    pass
        review_episode_ids.sort()
        if not review_episode_ids:
            review_episode_ids = list(range(n_episodes))

    # Validate
    for eid in review_episode_ids:
        assert 0 <= eid < n_episodes, f"Episode ID {eid} out of range [0, {n_episodes})"

    print(f"\nWill review episodes: {review_episode_ids}")
    print(f"Playback FPS: {playback_fps}")

    # Create or reuse environment
    if env is None:
        env = og.Environment(configs=config)
        if len(env.robots[robot_id].sensors) == 1:
            setup_wrist_camera_viewport(env, robot_id=robot_id)

        # Register keyboard handlers (only once when creating env)
        KeyboardEventHandler.initialize()

    # For load_sampling, all scenes share the same structure, so we don't need to reload.
    # The playback_episode will restore per-episode state via og.sim.load_state().

    # Set up active_modifier reference for keyboard callbacks
    active_modifier = {"ref": None}

    def _on_space():
        if active_modifier["ref"] is not None:
            active_modifier["ref"].on_space_pressed()

    def _on_c():
        if active_modifier["ref"] is not None:
            active_modifier["ref"].on_c_pressed()

    # Re-register callbacks (they may have been cleared)
    KeyboardEventHandler.add_keyboard_callback(
        lazy.carb.input.KeyboardInput.SPACE,
        _on_space,
    )
    KeyboardEventHandler.add_keyboard_callback(
        lazy.carb.input.KeyboardInput.C,
        _on_c,
    )

    # Create a base wrapper (callbacks will be swapped per episode)
    wrapper = MinimalPlaybackWrapper(env, input_hdf5)

    # Process each episode
    for episode_id in review_episode_ids:
        episode_key = f"demo_{episode_id}"

        if episode_key not in all_annotations:
            print(f"\nNo annotations for {episode_key}, skipping")
            continue

        episode_anns = all_annotations[episode_key]

        modifier = AnnotationModifier(
            env=env,
            episode_annotations=episode_anns,
            episode_key=episode_key,
            robot_id=robot_id,
            playback_fps=playback_fps,
        )
        active_modifier["ref"] = modifier

        wrapper.step_callback = modifier.step
        wrapper.episode_start_callback = modifier.episode_start_callback

        try:
            wrapper.playback_episode(episode_id)
        except EpisodeSkipException:
            print(f"\n  >>> Episode {episode_key} skipped by user (C pressed)")

        updated = modifier.get_updated_annotations()
        n_removed = len(modifier.indices_to_remove)
        n_added = len(modifier.annotations_to_add)
        all_annotations[episode_key] = updated
        print(f"\n  {episode_key}: {n_removed} removed, {n_added} added, {len(updated)} total")

    # Write updated annotations back to the original file
    with open(annotation_path, "w") as f:
        json.dump(all_annotations, f, indent=2)
    print(f"\nUpdated annotations written to: {annotation_path}")
    print(f"Original backup at: {backup_path}")

    # Cleanup HDF5 file handle
    input_hdf5.close()
    return env


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    teleop_dir = cfg.s14_teleop.out_dir
    annotation_dir = cfg.s15_annotate.out_dir

    # Task name for loading YAML (same resolution as 14_teleop: scene-specific then fallback)
    task_name = cfg.task.task_name
    scene_name = cfg.get("scene_name", "")
    og_task_cfg_path = os.path.join(SIMFOUNDRY_CFG_DIR, "task", scene_name, f"{task_name}.yaml") if scene_name else ""
    if not (og_task_cfg_path and os.path.exists(og_task_cfg_path)):
        og_task_cfg_path = os.path.join(SIMFOUNDRY_CFG_DIR, "task", f"{task_name}.yaml")
    task_cfg = parse_config(og_task_cfg_path)["og_task_config"]
    print(f"Loaded task config from: {og_task_cfg_path}")

    episode_ids_str = _script_args.episode_ids
    playback_fps = _script_args.playback_fps
    annotations_file = _script_args.annotations_file
    load_sampling_dir = _script_args.load_sampling_dir
    scene_ids_str = _script_args.scene_ids
    robot_id = cfg.s15_annotate.robot_id

    # Handle load_sampling directory mode
    if load_sampling_dir:
        load_sampling_path = Path(annotation_dir) / load_sampling_dir
        if not load_sampling_path.exists():
            raise FileNotFoundError(f"Load sampling directory not found: {load_sampling_path}")
        
        # Find all annotation files in the directory
        annotation_files = sorted(load_sampling_path.glob("annotations_scene_*.json"))
        if not annotation_files:
            raise FileNotFoundError(f"No annotation files found in {load_sampling_path}")
        
        # Filter by scene_ids if specified
        if scene_ids_str:
            scene_ids = [int(x.strip()) for x in scene_ids_str.split(",")]
            annotation_files = [
                f for f in annotation_files
                if any(f"scene_{sid:03d}" in f.name for sid in scene_ids)
            ]
            print(f"Filtered to scene IDs: {scene_ids}")
        
        print(f"\n{'='*80}")
        print(f"Load sampling mode: {load_sampling_dir}")
        print(f"Found {len(annotation_files)} annotation file(s)")
        for f in annotation_files:
            print(f"  - {f.name}")
        print(f"{'='*80}")
        
        # Parse episode_ids once (applies to all scenes)
        episode_ids = None
        if episode_ids_str is not None:
            episode_ids = [int(x.strip()) for x in str(episode_ids_str).split(",")]
        
        env = None
        try:
            apply_teleop_omnigibson_macros(enable_tr=False)
            for annotation_path in annotation_files:
                # Derive config and HDF5 paths
                ann_stem = annotation_path.stem  # e.g. "annotations_scene_000"
                data_stem = ann_stem[len("annotations_"):]  # e.g. "scene_000"
                config_path = annotation_path.parent / f"env_config_{data_stem}.json"
                
                if not config_path.exists():
                    print(f"WARNING: Config not found for {annotation_path.name}, skipping")
                    continue
                
                # Get HDF5 path from _source_hdf5 in annotation file
                with open(annotation_path) as _f:
                    _ann = json.load(_f)
                source_hdf5 = _ann.get("_source_hdf5", None)
                if source_hdf5 and os.path.exists(source_hdf5):
                    hdf_path = Path(source_hdf5)
                else:
                    # Fallback: try to find HDF5 from directory name
                    hdf_stem = load_sampling_dir  # e.g. "nv_desk_..._load_sampling"
                    hdf_path = Path(teleop_dir) / f"{hdf_stem}.hdf5"
                    if not hdf_path.exists():
                        print(f"WARNING: HDF5 not found for {annotation_path.name}, skipping")
                        continue
                
                # Pass env to reuse across scenes (load_sampling scenes share structure)
                env = _process_single_annotation_file(
                    annotation_path=annotation_path,
                    config_path=config_path,
                    hdf_path=hdf_path,
                    episode_ids=episode_ids,
                    playback_fps=playback_fps,
                    cfg=cfg,
                    robot_id=robot_id,
                    env=env,
                )
        finally:
            if env is not None:
                try:
                    og.clear()
                except (KeyError, RuntimeError) as e:
                    print(f"Warning: Cleanup error (can be safely ignored): {e}")
                og.shutdown()
        
        print(f"\n{'='*80}")
        print("Stage 15b complete (load_sampling mode)!")
        print(f"{'='*80}")
        return

    # Single annotation file mode (original behavior)
    # Resolve annotation file, env_config, and HDF5 paths
    if annotations_file:
        annotation_path = Path(annotation_dir) / annotations_file
        # env_config lives in the same directory as the annotation file (may be a subdir)
        ann_dir = annotation_path.parent
        ann_stem = annotation_path.stem  # e.g. "annotations_scene_000" or "annotations"
        if ann_stem.startswith("annotations_"):
            data_stem = ann_stem[len("annotations_"):]
            config_path = ann_dir / f"env_config_{data_stem}.json"
            hdf_path = Path(teleop_dir) / f"{data_stem}.hdf5"
        else:
            config_path = ann_dir / "env_config.json"
            hdf_path = None
    else:
        annotation_path = Path(annotation_dir) / "annotations.json"
        config_path = Path(annotation_dir) / "env_config.json"
        hdf_path = None

    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}. Run stage 15 first.")
    if not config_path.exists():
        raise FileNotFoundError(f"Environment config not found: {config_path}. Run stage 15 first.")

    # For load_sampling annotations, _source_hdf5 points to the correct HDF5
    if hdf_path is None or not hdf_path.exists():
        with open(annotation_path) as _f:
            _ann = json.load(_f)
        source_hdf5 = _ann.get("_source_hdf5", None)
        if source_hdf5 and os.path.exists(source_hdf5):
            hdf_path = Path(source_hdf5)
        else:
            hdf5_files = glob(f"{teleop_dir}/*.hdf5")
            if not hdf5_files:
                raise FileNotFoundError(f"No HDF5 files found in {teleop_dir}")
            hdf_path = Path(max(hdf5_files, key=os.path.getmtime))

    # Parse episode_ids
    episode_ids = None
    if episode_ids_str is not None:
        episode_ids = [int(x.strip()) for x in str(episode_ids_str).split(",")]

    env = None
    try:
        apply_teleop_omnigibson_macros(enable_tr=False)
        env = _process_single_annotation_file(
            annotation_path=annotation_path,
            config_path=config_path,
            hdf_path=hdf_path,
            episode_ids=episode_ids,
            playback_fps=playback_fps,
            cfg=cfg,
            robot_id=robot_id,
            env=None,
        )
    finally:
        if env is not None:
            try:
                og.clear()
            except (KeyError, RuntimeError) as e:
                print(f"Warning: Cleanup error (can be safely ignored): {e}")
            og.shutdown()

    print(f"\n{'='*80}")
    print("Stage 15b complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
