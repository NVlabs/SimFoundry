# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from `simfoundry` env

Requires installing:

- simfoundry, see the main README
"""
from copy import deepcopy
from pathlib import Path
from typing import Tuple, Optional

import torch
import cv2
import os
import trimesh
import numpy as np
from PIL import Image
from supervision import draw_rectangle
import shutil

from simfoundry.models.vlm import Gemini, Imagen3, FLUX1
from simfoundry.models.sam_v3_gmask import SAM3
from simfoundry.models.clip import CLIPEncoder
from simfoundry.models.sbert import SBERTEncoder
from simfoundry.utils.prompt_utils import prompt_object_setlist, prompt_object_removal_selection, \
    prompt_remove_object, prompt_classify_annotated_object, prompt_classify_outlined_object, \
    prompt_classify_number_in_circle, prompt_object_setlist_specific_tabletop, prompt_object_setlist_specific_floorplane, prompt_object_setlist_specific_floorplane_v3, \
    prompt_check_phrase_removed, prompt_flux_object_removal_bbox, prompt_flux_object_removal_outline, \
    prompt_flux_object_removal_mask, parse_json_response
from simfoundry.utils.processing_utils import draw_numbered_circle, erode_mask, get_largest_island, \
    geometric_median_mask, draw_mask_boundary, select_best_generated_background_infill_image, pad_image_to_ratio, \
    get_num_pixels_with_color, compute_point_cloud_from_depth, annotate, transform_point_cloud, denoise_obj_point_cloud, \
    draw_bounding_box, unpad_image, dilate_mask, vis_disparity, crop_image_with_mask, denoise_obj_point_cloud_v2
from torchvision.ops.boxes import box_convert
import tempfile
import json
import re
from collections import defaultdict, deque
from matplotlib import colormaps
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from prior_depth_anything import PriorDepthAnything, Arguments
from sentence_transformers import SentenceTransformer
from simfoundry.utils.python_utils import assert_valid_key
from simfoundry.pipeline.stage_utils import StageResult, bootstrap_hydra_workdir, finalize_stage
from simfoundry.pipeline.frame_selection import resolve_img_idx
from simfoundry.utils.python_utils import atomic_output_path
import hydra
import logging
import os
import glob

logger = logging.getLogger(__name__)

# Feedback file name (saved by 5b_visualize_decomposition.py)
FEEDBACK_FILENAME = "rerun_feedback.json"
bootstrap_hydra_workdir(__file__)


def get_primary_object_name(name: str) -> str:
    """
    Return the first segment of a slash-delimited object description.
    """
    if not isinstance(name, str):
        return name
    return name.split(" / ")[0].strip()


def canonicalize_vlm_bbox_xyxy(
    raw_bbox,
    img_w: int,
    img_h: int,
    forced_scale: Optional[str] = None,
    forced_order: Optional[str] = None,
    pixel_rescale: Optional[Tuple[float, float]] = None,
):
    """
    Normalize VLM bbox into pixel-space xyxy.
    Handles common format drifts:
    - coordinate order: xyxy vs yxyx
    - coordinate scale: pixel vs [0,1] vs [0,1000]
    """
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None, None

    try:
        raw = np.array([float(v) for v in raw_bbox], dtype=np.float32)
    except (TypeError, ValueError):
        return None, None

    if not np.all(np.isfinite(raw)):
        return None, None

    # Candidate coordinate orders
    all_order_candidates = {
        "xyxy": raw,
        "yxyx": np.array([raw[1], raw[0], raw[3], raw[2]], dtype=np.float32),
    }
    if forced_order in all_order_candidates:
        order_candidates = {forced_order: all_order_candidates[forced_order]}
    else:
        order_candidates = all_order_candidates

    # Candidate scales
    all_scale_candidates = {
        "pixel": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }
    raw_max = float(np.max(np.abs(raw)))
    if raw_max <= 1.5:
        all_scale_candidates["unit"] = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    if raw_max <= 1000.0:
        all_scale_candidates["thousand"] = np.array([img_w / 1000.0, img_h / 1000.0, img_w / 1000.0, img_h / 1000.0], dtype=np.float32)

    if forced_scale in all_scale_candidates:
        scale_candidates = {forced_scale: all_scale_candidates[forced_scale]}
    else:
        scale_candidates = all_scale_candidates

    best = None
    best_meta = None

    for order_name, coords in order_candidates.items():
        for scale_name, scale in scale_candidates.items():
            c = coords * scale
            x1, y1, x2, y2 = c.tolist()
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            w = x2 - x1
            h = y2 - y1
            if w <= 1e-3 or h <= 1e-3:
                continue

            area_ratio = (w * h) / max(float(img_w * img_h), 1.0)
            in_bounds = sum(
                [
                    0.0 <= x1 <= img_w,
                    0.0 <= x2 <= img_w,
                    0.0 <= y1 <= img_h,
                    0.0 <= y2 <= img_h,
                ]
            )

            # Heuristic score:
            # - prefer boxes inside image bounds
            # - prefer plausible area
            # - gently prefer pixel/xyxy defaults
            score = float(in_bounds)
            if 1e-4 <= area_ratio <= 0.95:
                score += 2.0
            if scale_name == "pixel":
                score += 0.8
            if order_name == "xyxy":
                score += 0.6

            if best is None or score > best[0]:
                best = (score, [x1, y1, x2, y2])
                best_meta = {"order": order_name, "scale": scale_name, "score": score}

    if best is None:
        return None, None

    x1, y1, x2, y2 = best[1]
    if pixel_rescale is not None and best_meta is not None and best_meta["scale"] == "pixel":
        sx, sy = pixel_rescale
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy
        best_meta = {**best_meta, "pixel_rescale": (sx, sy)}
    x1 = float(np.clip(x1, 0, img_w - 1))
    y1 = float(np.clip(y1, 0, img_h - 1))
    x2 = float(np.clip(x2, 0, img_w - 1))
    y2 = float(np.clip(y2, 0, img_h - 1))
    if x2 <= x1 or y2 <= y1:
        return None, None

    return [x1, y1, x2, y2], best_meta


def canonicalize_vlm_point_xy(raw_point, img_w: int, img_h: int):
    """
    Normalize VLM point into pixel-space (x, y).
    Handles format drifts:
    - coordinate order: xy vs yx
    - coordinate scale: pixel vs [0,1] vs [0,1000]
    """
    if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
        return None, None

    try:
        raw = np.array([float(v) for v in raw_point], dtype=np.float32)
    except (TypeError, ValueError):
        return None, None

    if not np.all(np.isfinite(raw)):
        return None, None

    order_candidates = {
        "xy": raw,
        "yx": np.array([raw[1], raw[0]], dtype=np.float32),
    }
    scale_candidates = {
        "pixel": np.array([1.0, 1.0], dtype=np.float32),
    }
    raw_max = float(np.max(np.abs(raw)))
    if raw_max <= 1.5:
        scale_candidates["unit"] = np.array([img_w, img_h], dtype=np.float32)
    if raw_max <= 1000.0:
        scale_candidates["thousand"] = np.array([img_w / 1000.0, img_h / 1000.0], dtype=np.float32)

    best = None
    best_meta = None
    for order_name, coords in order_candidates.items():
        for scale_name, scale in scale_candidates.items():
            x, y = (coords * scale).tolist()
            in_bounds = (0.0 <= x <= img_w) + (0.0 <= y <= img_h)
            score = float(in_bounds)
            if scale_name == "pixel":
                score += 0.8
            if order_name == "xy":
                score += 0.6
            if best is None or score > best[0]:
                best = (score, [x, y])
                best_meta = {"order": order_name, "scale": scale_name, "score": score}

    if best is None:
        return None, None

    x, y = best[1]
    x = float(np.clip(x, 0, img_w - 1))
    y = float(np.clip(y, 0, img_h - 1))
    return (int(np.round(x)), int(np.round(y))), best_meta


def convert_gemini_box2d_to_xyxy(raw_box2d, img_w: int, img_h: int):
    """
    Convert Gemini `box_2d` from [ymin, xmin, ymax, xmax] normalized to [0, 1000]
    into pixel-space xyxy using the same preprocessing as the reference example.
    """
    if not isinstance(raw_box2d, (list, tuple)) or len(raw_box2d) != 4:
        return None
    try:
        y1_n, x1_n, y2_n, x2_n = [float(v) for v in raw_box2d]
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite([y1_n, x1_n, y2_n, x2_n])):
        return None

    y1 = int(y1_n / 1000.0 * img_h)
    x1 = int(x1_n / 1000.0 * img_w)
    y2 = int(y2_n / 1000.0 * img_h)
    x2 = int(x2_n / 1000.0 * img_w)

    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    x_min = int(np.clip(x_min, 0, img_w - 1))
    y_min = int(np.clip(y_min, 0, img_h - 1))
    x_max = int(np.clip(x_max, 0, img_w - 1))
    y_max = int(np.clip(y_max, 0, img_h - 1))
    if x_max <= x_min or y_max <= y_min:
        return None
    return [float(x_min), float(y_min), float(x_max), float(y_max)]


def choose_prompt_point_from_bbox(
    target_bbox,
    other_bboxes,
    img_w: int,
    img_h: int,
):
    """
    Pick a robust point inside target bbox that avoids overlapping bboxes when possible.
    Falls back to bbox center if no collision-free point is found.
    """
    x1, y1, x2, y2 = [float(v) for v in target_bbox]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    def _inside_box(px, py, box):
        bx1, by1, bx2, by2 = [float(v) for v in box]
        bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
        by1, by2 = min(by1, by2), max(by1, by2)
        return (bx1 <= px <= bx2) and (by1 <= py <= by2)

    def _same_box(a, b, eps=1.0):
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        return (
            abs(ax1 - bx1) <= eps and
            abs(ay1 - by1) <= eps and
            abs(ax2 - bx2) <= eps and
            abs(ay2 - by2) <= eps
        )

    def _is_collision_free(px, py):
        for box in other_bboxes:
            # Ignore duplicate/alias boxes from the same object.
            if _same_box(target_bbox, box):
                continue
            if _inside_box(px, py, box):
                return False
        return True

    # Prefer off-center candidates first; keep center as a later fallback.
    candidate_fracs = [
        (0.35, 0.50),
        (0.65, 0.50),
        (0.50, 0.35),
        (0.50, 0.65),
        (0.35, 0.35),
        (0.65, 0.35),
        (0.35, 0.65),
        (0.65, 0.65),
        (0.20, 0.50),
        (0.80, 0.50),
        (0.50, 0.20),
        (0.50, 0.80),
        (0.50, 0.50),
    ]

    for fx, fy in candidate_fracs:
        px = int(np.clip(np.round(x1 + fx * (x2 - x1)), 0, img_w - 1))
        py = int(np.clip(np.round(y1 + fy * (y2 - y1)), 0, img_h - 1))
        if _is_collision_free(px, py):
            return (px, py), {"fallback_to_center": False}

    cx = int(np.clip(np.round((x1 + x2) / 2.0), 0, img_w - 1))
    cy = int(np.clip(np.round((y1 + y2) / 2.0), 0, img_h - 1))
    return (cx, cy), {"fallback_to_center": True}


def compute_object_removal_order(
    obj_chs,
    cam2world_tf,
    n_ray_samples=1000,
    occlusion_ray_fraction_threshold=0.05,
    support_ray_fraction_threshold=0.20,
    containment_fraction_threshold=0.50,
):
    """
    Determines which object to remove next using three prioritized heuristics:
      1. Occlusion: prefer non-occluded objects (visible from camera)
      2. Support level: prefer objects on top of / inside other objects (topmost first)
      3. Camera depth: prefer objects closer to the camera (foreground first)

    Support relationships are computed first so that the occlusion check can exclude
    supporting objects. For example, a croissant inside an open box will have camera-
    directed rays that hit the box's convex hull, but this is expected geometry (the box
    supports the croissant), not true visual occlusion.

    Args:
        obj_chs: dict mapping phrase_unique_id -> trimesh convex hull mesh
        cam2world_tf: 4x4 camera-to-world transformation matrix
        n_ray_samples: number of ray samples for each test
        occlusion_ray_fraction_threshold: fraction of camera-directed rays that must be
            blocked for an object to be considered occluded
        support_ray_fraction_threshold: fraction of downward rays from B that must hit A
            for B to be considered "on top of" A
        containment_fraction_threshold: fraction of B's interior points that must be
            inside A's convex hull for B to be considered "inside" A

    Returns:
        str: phrase_unique_id of the object to remove
    """
    phrases = list(obj_chs.keys())

    if len(phrases) == 1:
        logger.info(f"Only one object detected: {phrases[0]}")
        return phrases[0]

    cam_pos = cam2world_tf[:3, 3]

    # Pre-compute intersectors and interior sample points for each object
    intersectors = {}
    interior_points = {}
    for phrase, ch in obj_chs.items():
        intersectors[phrase] = trimesh.ray.ray_pyembree.RayMeshIntersector(ch)
        min_coords = ch.bounds[0]
        max_coords = ch.bounds[1]
        candidates = np.random.uniform(low=min_coords, high=max_coords, size=(n_ray_samples, 3))
        inside = intersectors[phrase].contains_points(candidates)
        interior_points[phrase] = candidates[inside]

    # --- Heuristic 2 (computed first): Support relationship detection ---
    # Build a directed graph: on_top_of[B] contains A means B is on top of A
    # (i.e. A supports B, so B should be removed before A)
    # This must be computed before occlusion so we can exclude supporting objects
    # from the occlusion check.
    on_top_of = defaultdict(set)   # B -> set of objects that B sits on top of
    supports = defaultdict(set)    # A -> set of objects that sit on top of A

    for phrase_b in phrases:
        ch_b = obj_chs[phrase_b]
        pts_b = interior_points[phrase_b]
        if len(pts_b) == 0:
            continue
        centroid_b_z = ch_b.centroid[2]

        for phrase_a in phrases:
            if phrase_a == phrase_b:
                continue
            ch_a = obj_chs[phrase_a]

            # Skip if B's centroid is well below A's bottom -- B can't be on top of A
            if centroid_b_z < ch_a.bounds[0][2] - 0.02:
                continue

            # Test 1: Downward rays from B's interior hitting A's convex hull
            down_dirs = np.zeros((len(pts_b), 3))
            down_dirs[:, 2] = -1.0
            down_hits = intersectors[phrase_a].intersects_any(
                ray_origins=pts_b,
                ray_directions=down_dirs,
            )
            down_fraction = down_hits.sum() / len(pts_b)

            # Test 2: Containment -- fraction of B's interior inside A's convex hull
            containment_fraction = 0.0
            if len(pts_b) > 0:
                inside_a = intersectors[phrase_a].contains_points(pts_b)
                containment_fraction = inside_a.sum() / len(pts_b)

            if down_fraction >= support_ray_fraction_threshold or containment_fraction >= containment_fraction_threshold:
                on_top_of[phrase_b].add(phrase_a)
                supports[phrase_a].add(phrase_b)
                reason = []
                if down_fraction >= support_ray_fraction_threshold:
                    reason.append(f"downward rays={down_fraction:.2f}")
                if containment_fraction >= containment_fraction_threshold:
                    reason.append(f"containment={containment_fraction:.2f}")
                logger.info(f"Support: {phrase_b} is on top of {phrase_a} ({', '.join(reason)})")

    # Compute support levels via BFS from leaves (objects with nothing on top)
    # Level 0 = nothing on top (remove first), higher = deeper in stack
    support_level = {}
    # Objects with nothing on top of them are level 0
    leaves = [p for p in phrases if len(supports.get(p, set())) == 0]
    queue = deque()
    for leaf in leaves:
        support_level[leaf] = 0
        queue.append(leaf)

    while queue:
        current = queue.popleft()
        current_level = support_level[current]
        # Objects that current sits on top of get level = max(their current level, current_level + 1)
        for below in on_top_of.get(current, set()):
            new_level = current_level + 1
            if below not in support_level or support_level[below] < new_level:
                support_level[below] = new_level
                queue.append(below)

    # Any object not yet assigned (e.g. in a cycle) gets level 0
    for p in phrases:
        if p not in support_level:
            support_level[p] = 0

    for p in phrases:
        logger.info(f"Support level: {p} = {support_level[p]}")

    # --- Heuristic 1: Fractional occlusion via camera-directed rays ---
    # When checking if object A is occluded, we skip objects that A sits on top of.
    # Rays from A hitting a supporting object's convex hull are expected (the object
    # is inside/on-top-of the supporter), not true visual occlusion.
    occlusion_fractions = {}
    for phrase_a in phrases:
        pts = interior_points[phrase_a]
        if len(pts) == 0:
            occlusion_fractions[phrase_a] = 0.0
            continue
        ray_dirs = cam_pos.reshape(1, 3) - pts
        ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=-1, keepdims=True)
        # Objects that A sits on top of should not count as occluders
        a_supporters = on_top_of.get(phrase_a, set())
        max_blocked = 0.0
        for phrase_b in phrases:
            if phrase_b == phrase_a:
                continue
            if phrase_b in a_supporters:
                continue
            hits = intersectors[phrase_b].intersects_any(
                ray_origins=pts,
                ray_directions=ray_dirs,
            )
            fraction = hits.sum() / len(pts)
            max_blocked = max(max_blocked, fraction)
        occlusion_fractions[phrase_a] = max_blocked

    is_occluded = {p: frac > occlusion_ray_fraction_threshold for p, frac in occlusion_fractions.items()}
    for p in phrases:
        if is_occluded[p]:
            logger.info(f"Occlusion: {p} is occluded (max blocked fraction: {occlusion_fractions[p]:.2f})")
        else:
            logger.info(f"Occlusion: {p} is NOT occluded (max blocked fraction: {occlusion_fractions[p]:.2f})")

    # --- Heuristic 3: Camera distance ---
    # Use direct distance instead of a camera-axis projection because project data
    # may use either +Z-forward or -Z-forward camera transforms.
    camera_depths = {}
    for phrase, ch in obj_chs.items():
        centroid = ch.centroid
        camera_depths[phrase] = np.linalg.norm(centroid - cam_pos)

    for p in phrases:
        logger.info(f"Camera depth: {p} = {camera_depths[p]:.4f}")

    # --- Combined multi-level sort ---
    # Priority: (1) non-occluded first, (2) lowest support level first, (3) closest to camera first
    sorted_phrases = sorted(phrases, key=lambda p: (
        is_occluded[p],         # False (0) < True (1) -> non-occluded first
        support_level[p],       # 0 < 1 < 2 ... -> topmost objects first
        camera_depths[p],       # smaller -> closer to camera
    ))

    logger.info(f"\nObject removal priority order:")
    for i, p in enumerate(sorted_phrases):
        logger.info(f"  {i+1}. {p} (occluded={is_occluded[p]}, support_level={support_level[p]}, cam_depth={camera_depths[p]:.4f})")

    selected = sorted_phrases[0]
    logger.info(f"\nSelected for removal: {selected}")
    return selected

#TODO: handle these better
REMOVAL_MODELS = {
    "gemini",
    "gemini-2.5-flash-image",
    "gemini-3-pro-image",
    "flux",
}

DETECTION_MODELS = {
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
}

PDA_GEOMETRIC_BACKENDS = {
    "depth_pro",
    "da3",
}


def load_rerun_feedback(out_dir: str) -> dict:
    """
    Load rerun feedback from the visualization script.
    
    Args:
        out_dir: Path to the s5_scene output directory
        
    Returns:
        Dictionary with rerun requests, or empty dict if no feedback
    """
    feedback_path = os.path.join(out_dir, FEEDBACK_FILENAME)
    if not os.path.exists(feedback_path):
        logger.info(f"No feedback file found at {feedback_path}")
        return {}
    
    try:
        with open(feedback_path, "r") as f:
            feedback = json.load(f)
        
        # Convert string keys back to integers
        rerun_requests = {int(k): v for k, v in feedback.get("rerun_requests", {}).items()}
        logger.info(f"Loaded feedback with {len(rerun_requests)} rerun request(s)")
        return rerun_requests
    except Exception as e:
        logger.error(f"Failed to load feedback: {e}")
        return {}


def delete_iteration_data(out_dir: str, start_iter: int, end_iter: int = None):
    """
    Delete iteration data for specified iterations.
    
    Args:
        out_dir: Path to the s5_scene output directory
        start_iter: First iteration to delete (inclusive)
        end_iter: Last iteration to delete (inclusive). If None, deletes all from start_iter onwards.
    """
    # Directories that contain per-iteration data
    iter_dirs = [
        "masked_object", "masked_object_focus", "masked_object_background",
        "detected_phrases", "removal_mask", "removal_mask_resized",
        "pre_object_removal", "post_object_removal", "metric_depth", "obj_cat_list",
        "skipped_iterations",
    ]
    
    # Find all iteration files if end_iter not specified
    if end_iter is None:
        # Find the maximum iteration
        obj_cat_dir = os.path.join(out_dir, "obj_cat_list")
        if os.path.exists(obj_cat_dir):
            iter_files = glob.glob(os.path.join(obj_cat_dir, "iter_*.json"))
            if iter_files:
                max_iter = max(int(os.path.basename(f).split("_")[1].split(".")[0]) for f in iter_files)
                end_iter = max_iter
            else:
                end_iter = start_iter
        else:
            end_iter = start_iter
    
    deleted_count = 0
    for iter_idx in range(start_iter, end_iter + 1):
        for dir_name in iter_dirs:
            dir_path = os.path.join(out_dir, dir_name)
            if not os.path.exists(dir_path):
                continue
            
            # Try different file extensions
            for ext in [".png", ".json", ".npy"]:
                file_path = os.path.join(dir_path, f"iter_{iter_idx}{ext}")
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        logger.debug(f"Deleted {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
    
    logger.info(f"Deleted {deleted_count} files for iterations {start_iter} to {end_iter}")


DECOMPOSITION_FRAME_FILENAME = "decomposition_frame.json"


def read_decomposition_frame(out_dir: str) -> int | None:
    """Frame index the artifacts already in `out_dir` were decomposed from, if recorded."""
    fpath = os.path.join(out_dir, DECOMPOSITION_FRAME_FILENAME)
    if not os.path.isfile(fpath):
        return None
    try:
        with open(fpath, "r") as f:
            return int(json.load(f)["img_idx"])
    except Exception as e:
        logger.warning(f"Could not read {fpath}: {e}")
        return None


def write_decomposition_frame(out_dir: str, img_idx: int) -> None:
    with open(os.path.join(out_dir, DECOMPOSITION_FRAME_FILENAME), "w") as f:
        json.dump({"img_idx": int(img_idx)}, f, indent=4)


def check_resume_frame_matches(out_dir: str, img_idx: int) -> None:
    """Refuse to resume on top of a decomposition taken from a different frame.

    infer_resume_state below decides where to resume purely from filenames, so it cannot tell
    that the existing iterations came from another viewpoint. With s3_ground.img_idx: auto the
    canonical frame can change between runs, and resuming across that change silently mixes
    object crops from two viewpoints -- or, when the old run consumed every category, skips
    decomposition altogether and leaves the previous frame's crops for stages 6-8 to build on.
    """
    previous = read_decomposition_frame(out_dir)
    if previous is not None and previous != int(img_idx):
        raise RuntimeError(
            f"{out_dir} holds a decomposition of frame {previous}, but this run is built on "
            f"frame {img_idx}. Resuming would mix object crops from two viewpoints. Move or "
            f"delete {out_dir} (and the s6-s13 outputs derived from it) and rerun, or pin "
            f"s3_ground.img_idx={previous} to keep the existing decomposition."
        )
    if previous is None and infer_resume_state(out_dir)[0] > 0:
        logger.warning(
            "%s holds a previous decomposition with no recorded frame index, so it cannot be "
            "checked against the current frame (%s). If it predates frame selection and was "
            "built from a different frame, clear it before rerunning.", out_dir, img_idx,
        )


def infer_resume_state(out_dir: str) -> tuple:
    """
    Infer where to resume the decomposition pipeline based on existing outputs.

    An iteration is considered complete only if its core outputs exist:
    - post_object_removal/iter_{i}.png
    - metric_depth/iter_{i}.npy
    - obj_cat_list/iter_{i}.json

    Returns:
        Tuple of (resume_iteration, stale_iterations, last_iter_metadata)
    """
    required_outputs = {
        "post_object_removal": ".png",
        "metric_depth": ".npy",
        "obj_cat_list": ".json",
    }
    iter_pattern = re.compile(r"iter_(\d+)$")

    existing_iterations = set()
    complete_iteration_sets = []

    for dir_name, ext in required_outputs.items():
        dir_path = os.path.join(out_dir, dir_name)
        iter_indices = set()
        if os.path.exists(dir_path):
            for fname in os.listdir(dir_path):
                stem, suffix = os.path.splitext(fname)
                if suffix != ext:
                    continue
                match = iter_pattern.match(stem)
                if match:
                    iter_indices.add(int(match.group(1)))
        complete_iteration_sets.append(iter_indices)
        existing_iterations |= iter_indices

    for dir_name in [
        "masked_object", "masked_object_focus", "masked_object_background",
        "detected_phrases", "removal_mask", "removal_mask_resized", "pre_object_removal",
    ]:
        dir_path = os.path.join(out_dir, dir_name)
        if not os.path.exists(dir_path):
            continue
        for fname in os.listdir(dir_path):
            stem = os.path.splitext(fname)[0]
            match = iter_pattern.match(stem)
            if match:
                existing_iterations.add(int(match.group(1)))

    complete_iterations = set.intersection(*complete_iteration_sets) if complete_iteration_sets else set()
    last_complete_iter = max(complete_iterations) if complete_iterations else None
    resume_iteration = 0 if last_complete_iter is None else last_complete_iter + 1
    stale_iterations = sorted(
        iter_idx for iter_idx in existing_iterations
        if last_complete_iter is None or iter_idx > last_complete_iter
    )

    last_iter_metadata = None
    if last_complete_iter is not None:
        metadata_path = os.path.join(out_dir, "obj_cat_list", f"iter_{last_complete_iter}.json")
        try:
            with open(metadata_path, "r") as f:
                last_iter_metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata for completed iteration {last_complete_iter}: {e}")

    return resume_iteration, stale_iterations, last_iter_metadata


def process_feedback_requests(rerun_requests: dict, out_dir: str) -> tuple:
    """
    Process feedback requests and determine what needs to be rerun.
    
    Args:
        rerun_requests: Dictionary of {iter_idx: {rerun, category, rerun_downstream, point_prompt}}
        out_dir: Path to the s5_scene output directory
        
    Returns:
        Tuple of (start_iteration, category_overrides, point_prompts, iterations_to_delete)
        - start_iteration: The earliest iteration that needs to be rerun
        - category_overrides: Dict mapping iter_idx to category string
        - point_prompts: Dict mapping iter_idx to (x, y) point coordinates
        - iterations_to_delete: Set of iteration indices to delete before rerun
    """
    if not rerun_requests:
        return None, {}, {}, set()
    
    # Find earliest iteration to rerun
    iterations_to_rerun = [idx for idx, req in rerun_requests.items() if req.get("rerun", False)]
    if not iterations_to_rerun:
        return None, {}, {}, set()
    
    start_iteration = min(iterations_to_rerun)
    
    # Collect category overrides
    category_overrides = {}
    for iter_idx, req in rerun_requests.items():
        if req.get("category"):
            category_overrides[iter_idx] = req["category"]
    
    # Collect point prompts
    point_prompts = {}
    for iter_idx, req in rerun_requests.items():
        if req.get("point_prompt"):
            point_prompts[iter_idx] = tuple(req["point_prompt"])
    
    # Determine which iterations to delete
    iterations_to_delete = set()
    for iter_idx, req in rerun_requests.items():
        if req.get("rerun", False):
            if req.get("rerun_downstream", False):
                # Delete this iteration and all subsequent ones
                # We'll need to find max iteration
                obj_cat_dir = os.path.join(out_dir, "obj_cat_list")
                if os.path.exists(obj_cat_dir):
                    iter_files = glob.glob(os.path.join(obj_cat_dir, "iter_*.json"))
                    if iter_files:
                        max_iter = max(int(os.path.basename(f).split("_")[1].split(".")[0]) for f in iter_files)
                        for i in range(iter_idx, max_iter + 1):
                            iterations_to_delete.add(i)
            else:
                # Just delete this single iteration
                iterations_to_delete.add(iter_idx)
    
    return start_iteration, category_overrides, point_prompts, iterations_to_delete

# Start with: RGB frame (s1), metric depth + intrinsics (s2), floor category (s3),
# cam2world + floor-aligned point cloud (s4). The frame is padded / resized to the removal
# model's nearest supported resolution and upsampled by that model before the loop starts.
# While true:
#  1. Detect object categories and 2D boxes in the current RGB using Gemini
#     (cfg.s5_scene.detection_model). Re-detected every update_categories_period iterations
#     while under min_detection_iterations; overridable by force_categories or by feedback
#     from 5b_visualize_decomposition.py.
#  2. Segment instances with SAM3 text prompts, one prompt per category. If a category yields
#     no mask, fall back to a point prompt (the feedback point if one was given for this
#     iteration, else a point derived from the VLM box).
#  3. Prune masks: redundant / duplicate / boundary masks, then masks below the floor plane
#     (point-cloud z), then disentangle masks that subsume others. Terminate once nothing
#     survives detection or pruning (past min_detection_iterations), or at end_iter.
#  4. Build a denoised point cloud and convex hull per object, then pick the object to remove
#     via ray-based heuristics: non-occluded first, then topmost (on / inside others), then
#     closest to camera.
#  5a. Isolate the object: crop around its mask and save masked / background / mask variants
#  5b. Annotate the object (bbox, outline, or filled mask per obj_removal_prompt_order) and
#     remove it with Gemini or FLUX.1 Kontext, retrying seeds and rejecting outputs that keep
#     the annotation color; blend the result back into the current RGB over the dilated mask
#  6. Generate new depth: zero the removed region, run the geometric backend (DepthPro or DA3),
#     refine with Prior-Depth-Anything conditioned on the surviving depth as prior, and keep the
#     prior where it is still valid -> new depth map

#################


from simfoundry import CFG_DIR


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    raw_img_dir = f"{cfg.s1_video.out_dir}/frames_subsampled_{cfg.s1_video.n_subsampled_frames}"
    raw_imgs = list(sorted([f"{raw_img_dir}/{fname}" for fname in os.listdir(raw_img_dir) if fname.endswith(".png")]))
    source_dir = cfg.s2_da.out_dir
    # source_image_fpath = f"{cfg.s1_zed.out_dir}/{cfg.s5_scene.img_name}_l.png"
    out_dir = cfg.s5_scene.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Other hyperparams we need
    scene_img_idx = resolve_img_idx(cfg, stage_key="s5_scene")
    ground_img_idx = resolve_img_idx(cfg)
    if scene_img_idx != ground_img_idx:
        # This stage indexes object masks taken from the scene frame into the point cloud of the
        # ground frame, so the two only line up when they are the same frame.
        logger.warning(
            "s5_scene.img_idx (%s) differs from s3_ground.img_idx (%s); object masks and the "
            "point cloud come from different viewpoints and will not align.",
            scene_img_idx, ground_img_idx,
        )

    dirs_to_create = ["masked_object", "masked_object_focus", "masked_object_background", "detected_phrases", "gemini_detected_phrases", "removal_mask", "removal_mask_resized",
    "pre_object_removal", "post_object_removal", "metric_depth", "obj_cat_list", "skipped_iterations"]
    for dir_name in dirs_to_create:
        Path(f"{out_dir}/{dir_name}").mkdir(parents=True, exist_ok=True)

    # Checked before the expensive source-image upsample below, so a stale output dir fails
    # fast rather than after a remote image call.
    check_resume_frame_matches(out_dir, scene_img_idx)
    write_decomposition_frame(out_dir, scene_img_idx)

    # Set hyperparams
    RATIO = cfg.s5_scene.ratio
    REMOVAL_OUTLINE_COLOR = cfg.s5_scene.removal_outline_color
    BOUNDARY_PROPORTION = cfg.s5_scene.boundary_proportion

    logger.info("="*60)
    logger.info("Starting scene decomposition pipeline...")
    logger.info(f"Output directory: {out_dir}")
    logger.info("="*60)

    # Create SAM model
    sam3 = SAM3(
        confidence_threshold=cfg.s5_scene.sam_conf_threshold,
        device="cuda",
        video=False,
    )

    # Object detection model: configurable via cfg.s5_scene.detection_model.
    # Needs strong vision + structured JSON output. Uses Vertex AI Gemini and must be
    # one of Gemini's supported model ids (DETECTION_MODELS).
    detection_model_name = cfg.s5_scene.detection_model
    assert_valid_key(key=detection_model_name, valid_keys=DETECTION_MODELS, name="detection model")
    vlm_pro = Gemini(
        project=cfg.gcloud_project,
        location="global",
        model=detection_model_name,
    )

    # Define removal model
    removal_model_name = cfg.s5_scene.removal_model
    assert_valid_key(key=removal_model_name, valid_keys=REMOVAL_MODELS, name="removal model")
    if removal_model_name == "gemini":
        removal_model = Gemini(
            project=cfg.gcloud_project,
            location="global",
            model="gemini-3-pro-image",
        )
    elif "gemini" in removal_model_name:
        removal_model = Gemini(
            project=cfg.gcloud_project,
            location="global",
            model=removal_model_name,
        )
    elif removal_model_name == "flux":
        removal_model = FLUX1(
            model="FLUX.1-Kontext-dev",
            dtype=torch.bfloat16,
            device="cuda",
        )
    else:
        raise NotImplementedError

    # Create text encoder
    text_encoder = SBERTEncoder()

    # Models for generating synthetic depth
    args = Arguments()
    args.K = cfg.s5_scene.pda_K

    # Select geometric depth backend for PriorDepthAnything
    pda_geometric_backend = cfg.s5_scene.get("pda_geometric_backend", "depth_pro")
    assert_valid_key(key=pda_geometric_backend, valid_keys=PDA_GEOMETRIC_BACKENDS, name="pda_geometric_backend")
    logger.info(f"Using PDA geometric depth backend: {pda_geometric_backend}")

    dp = None
    dp_transform = None
    da3_geometric_model = None

    if pda_geometric_backend == "depth_pro":
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        DEFAULT_MONODEPTH_CONFIG_DICT.checkpoint_uri = f"{depth_pro.__path__[0]}/../../checkpoints/depth_pro.pt"
        dp, dp_transform = depth_pro.create_model_and_transforms()
        dp.eval()
    elif pda_geometric_backend == "da3":
        from simfoundry.models.depth_anything_v3 import DepthAnythingV3
        da3_geometric_model = DepthAnythingV3(model_name="DA3NESTED-GIANT-LARGE-1.1", device="cuda")
    else:
        raise ValueError(f"Unknown pda_geometric_backend: {pda_geometric_backend}")

    priorda = PriorDepthAnything(
        device="cuda",
        frozen_model_size="vitl",
        conditioned_model_size="vitb",
        coarse_only=False,
        version="1.0",
        args=args,
    )
    def _calc_scale_shift_fp32(k_sparse_targets, k_pred_targets, currk_dists=None, knn=False):
        # Prior-Depth-Anything can run under autocast (fp16/bf16), but lstsq only
        # supports float/double. Recompute in fp32 with autocast disabled.
        with torch.cuda.amp.autocast(enabled=False):
            k_sparse_targets = k_sparse_targets.to(torch.float32)
            k_pred_targets = k_pred_targets.to(torch.float32)
            if currk_dists is not None:
                currk_dists = currk_dists.to(torch.float32)

            k_pred_targets = k_pred_targets + torch.rand_like(k_pred_targets) * 1e-5
            X = torch.stack(
                [
                    k_pred_targets,
                    torch.ones_like(k_pred_targets),
                ],
                dim=2,
            )

            if knn > 0:
                k_sparse_targets, X = priorda.completion.perform_weighted(k_sparse_targets, X, currk_dists)
            elif k_pred_targets.shape[0] > 1:
                k_sparse_targets = k_sparse_targets.unsqueeze(-1)

            solution = torch.linalg.lstsq(X, k_sparse_targets)
            scale = solution[0][:, 0].squeeze()
            shift = solution[0][:, 1].squeeze()
            return scale, shift

    priorda.completion.calc_scale_shift = _calc_scale_shift_fp32

    # Load raw inputs
    # Make sure to resize image first so that it's standardized with Imagen native resolutions
    padding_color = (255, 255, 255)
    original_img = np.array(Image.open(raw_imgs[scene_img_idx]))
    original_H, original_W, _ = original_img.shape
    original_ratio = original_W / original_H
    ratio_options = {w / h: (w, h) for (w, h) in removal_model.IMAGE_SHAPES}
    ratio_differences = np.abs([original_ratio - ratio_option for ratio_option in ratio_options.keys()])
    closest_ratio = list(ratio_options.keys())[ratio_differences.argmin()]
    resolution = ratio_options[closest_ratio]
    # resolution = imagen.RESOLUTIONS[RATIO]
    # w_rel, h_rel = RATIO.split(":")
    target_W, target_H = resolution
    target_ratio = target_W / target_H        # W / H

    resized_padded_img = cv2.resize(pad_image_to_ratio(original_img, target_ratio=target_ratio, padding_color=padding_color), resolution)
    H, W, _ = resized_padded_img.shape
    img_scale = np.sqrt(H * W)
    source_resized_image_fpath = f"{out_dir}/source_padded_resized.png"
    Image.fromarray(resized_padded_img).save(source_resized_image_fpath)

    # Pass through VLM to upsample this image, unless disabled via cfg.s5_scene.use_upsampled_source_image
    use_upsampled_source_image = cfg.s5_scene.get("use_upsampled_source_image", True)
    source_resized_upsampled_image_fpath = f"{out_dir}/source_padded_resized_upsampled.png"
    if not use_upsampled_source_image:
        logger.info(
            "s5_scene.use_upsampled_source_image=false; seeding decomposition with "
            f"{source_resized_image_fpath} directly, skipping the source-image upsample call."
        )
        upsampled_obj_img_raw = resized_padded_img
    else:
        upsample_kwargs = {
            "prompt": "Significantly improve the image resolution, highlighting fine-grained details",
            "seed": 0,
            "print_results": cfg.visualize,
        }
        if "gemini" in removal_model_name:
            logger.info(f"Calling {removal_model_name} to upsample source image (may take up to 5 min)...")
            result = removal_model(
                image_paths=source_resized_image_fpath,
                temperature=0,
                top_p=0,
                **upsample_kwargs,
            )
            logger.info("Source image upsample done.")
            images = [] if result is None else removal_model.get_result_images(result=result)
            if not images:
                logger.warning("Image model returned no images, falling back to original.")
                upsampled_obj_img_raw = resized_padded_img
            else:
                upsampled_obj_img_raw = np.array(images[0])
        elif removal_model_name == "flux":
            upsampled_obj_img_raw = np.array(removal_model(
                image_path=source_resized_image_fpath,
                guidance_scale=2.5,
                num_inference_steps=20,
                max_sequence_length=512,
                **upsample_kwargs,
            ))
        else:
            raise NotImplementedError

        # Resize to expected resolution (FLUX/Gemini may return slightly different dimensions)
        upsampled_obj_img_raw = cv2.resize(upsampled_obj_img_raw, resolution)
        Image.fromarray(upsampled_obj_img_raw).save(source_resized_upsampled_image_fpath)

    # Load the rotated point cloud (i.e.: "floor" is located at origin and z-direction is aligned with the plane
    results = np.load(f"{source_dir}/da/exports/npz/results.npz")
    K = results["intrinsics"][ground_img_idx]
    np.save(f"{out_dir}/original_depth.npy", results["depth"][scene_img_idx])
    # K_fpath = f"{source_dir}/{cfg.s3_ground.img_name}_K.npy"
    # scene_img_name = cfg.s5_scene.img_name
    # K = np.load(K_fpath)
    # depth_path = f"{source_dir}/{scene_img_name}_depth_meter.npy"
    # shutil.copy2(depth_path, f"{out_dir}/original_depth.npy")
    # pc_name = cfg.s3_ground.img_name
    cam2world_tf_fpath = f"{cfg.s4_frame.out_dir}/image_{ground_img_idx}_cam2world.npy"
    cam2world_tf = np.load(cam2world_tf_fpath)

    pcd_fpath = f"{cfg.s4_frame.out_dir}/image_{ground_img_idx}_pc_raw_rotated.ply"
    # da_rgb_fpath = f"{cfg.s2_da.out_dir}/image_{ground_img_idx}_rgb.png"
    # da_rgb_img = np.array(Image.open(fs_rgb_fpath))
    da_rgb_img = results["image"][ground_img_idx]
    pcd = o3d.io.read_point_cloud(pcd_fpath)
    pc = np.asarray(pcd.points).reshape(*da_rgb_img.shape)  # (H, W, 3)
    padded_pc, (delta_w, delta_h) = pad_image_to_ratio(pc, target_ratio=target_ratio, padding_color=padding_color, return_padding_size=True)
    padded_H, padded_W, _ = padded_pc.shape
    resized_padded_pc = cv2.resize(padded_pc, resolution)

    # Iterate
    current_iteration = 0 if cfg.s5_scene.start_iter is None else cfg.s5_scene.start_iter
    end_iteration = cfg.s5_scene.end_iter
    detection_iteration = 0
    resumed_obj_metadata = None
    feedback_resume_applied = False
    
    # Handle feedback from visualization script if enabled
    use_feedback = cfg.s5_scene.get("use_feedback", False)
    category_overrides = {}
    point_prompts = {}
    vlm_obj_prompt_points = {}
    vlm_obj_prompt_bboxes = {}
    if use_feedback:
        logger.info("="*60)
        logger.info("Processing feedback from 5b_visualize_decomposition.py...")
        logger.info("="*60)
        
        rerun_requests = load_rerun_feedback(out_dir)
        if rerun_requests:
            feedback_start_iter, category_overrides, point_prompts, iterations_to_delete = process_feedback_requests(rerun_requests, out_dir)
            if feedback_start_iter is not None:
                logger.info(f"Feedback requests starting from iteration {feedback_start_iter}")
                
                # Delete iteration data as requested
                if iterations_to_delete:
                    logger.info(f"Deleting data for iterations: {sorted(iterations_to_delete)}")
                    for iter_idx in sorted(iterations_to_delete):
                        delete_iteration_data(out_dir, iter_idx, iter_idx)
                
                # Override start iteration
                current_iteration = feedback_start_iter
                detection_iteration = 0
                feedback_resume_applied = True
                logger.info(f"Starting decomposition from iteration {current_iteration}")
                
                if category_overrides:
                    logger.info(f"Category overrides: {category_overrides}")
                
                if point_prompts:
                    logger.info(f"Point prompts: {point_prompts}")
                
            else:
                logger.info("No valid rerun requests in feedback file")
        else:
            logger.info("No feedback file found or empty feedback")
        
        logger.info("="*60)

    # Automatically continue from the last complete iteration unless the user explicitly
    # set start_iter or provided a feedback-based rerun request.
    if cfg.s5_scene.start_iter is None and not feedback_resume_applied:
        inferred_resume_iter, stale_iterations, resumed_obj_metadata = infer_resume_state(out_dir)
        if stale_iterations:
            logger.info(f"Cleaning up incomplete iteration data before resume: {stale_iterations}")
            for iter_idx in stale_iterations:
                delete_iteration_data(out_dir, iter_idx, iter_idx)
        if inferred_resume_iter > 0:
            current_iteration = inferred_resume_iter
            detection_iteration = 0
            logger.info(f"Auto-resuming scene decomposition from iteration {current_iteration}")
    
    # The iteration whose post-removal outputs the next detection runs against. 
    last_kept_iteration = max(
        (
            int(p.stem.split("_")[-1])
            for p in Path(f"{out_dir}/post_object_removal").glob("iter_*.png")
            if int(p.stem.split("_")[-1]) < current_iteration
        ),
        default=None,
    )
    if last_kept_iteration is None:
        current_rgb_fpath = source_resized_upsampled_image_fpath if use_upsampled_source_image else source_resized_image_fpath
        current_rgb = upsampled_obj_img_raw
        current_depth = results["depth"][scene_img_idx]
    else:
        current_rgb_fpath = f"{out_dir}/post_object_removal/iter_{last_kept_iteration}.png"
        current_rgb = np.array(Image.open(current_rgb_fpath))
        current_depth = np.load(f"{out_dir}/metric_depth/iter_{last_kept_iteration}.npy")
    current_pc = compute_point_cloud_from_depth(
        depth=current_depth,
        K=K,
        world_to_cam_tf=cam2world_tf,
    )
    update_categories_period = cfg.s5_scene.update_categories_period
    merge_output_images = cfg.s5_scene.merge_output_images
    crop_removal_images = cfg.s5_scene.crop_removal_images
    obj_removal_prompt_order = cfg.s5_scene.obj_removal_prompt_order
    min_valid_pixel_prop = cfg.s5_scene.min_valid_pixel_prop
    force_categories = cfg.s5_scene.force_categories
    obj_cat_list = None
    if current_iteration > 0:
        # The preceding index may have been a skipped iteration with no manifest, so
        # resume category state from the last kept iteration, not current_iteration - 1.
        if resumed_obj_metadata is None and last_kept_iteration is not None:
            prior_metadata_fpath = f"{out_dir}/obj_cat_list/iter_{last_kept_iteration}.json"
            if os.path.exists(prior_metadata_fpath):
                try:
                    with open(prior_metadata_fpath, "r") as f:
                        resumed_obj_metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load previous iteration metadata from {prior_metadata_fpath}: {e}")
        if resumed_obj_metadata is not None:
            obj_cat_list = list(
                resumed_obj_metadata.get(
                    "vlm_categories_full",
                    resumed_obj_metadata.get("vlm_categories", []),
                )
            )
            removed_obj_phrase = resumed_obj_metadata.get(
                "removed_obj_phrase_full",
                resumed_obj_metadata.get("removed_obj_phrase"),
            )
            if removed_obj_phrase in obj_cat_list:
                obj_cat_list.remove(removed_obj_phrase)
            logger.info(f"Resumed category state from iteration {current_iteration - 1}: {obj_cat_list}")
            if (
                force_categories is None
                and current_iteration not in category_overrides
                and len(obj_cat_list) == 0
            ):
                logger.info(
                    "Previous complete iteration removed the last known object; "
                    "skipping residual-image VLM detection on resume."
                )
                finalize_stage(
                    stage_cfg=cfg.s5_scene,
                    out_dir=cfg.s5_scene.out_dir,
                    result=StageResult(success=True),
                )
                return

    floor_info_fpath = f"{cfg.s3_ground.out_dir}/image_{ground_img_idx}_floor_info.json"
    with open(floor_info_fpath, "r") as f:
        floor_info = json.load(f)
    floor_category = floor_info["floor_category"]
    while True:
        # Break if we have specified an end iteration and we're past it
        if end_iteration is not None and current_iteration >= end_iteration:
            print(f"End iteration = {end_iteration} reached, terminating early.")
            break

        # Check for category override from feedback (takes highest priority)
        if current_iteration in category_overrides:
            override_category = category_overrides[current_iteration]
            obj_cat_list = [override_category]
            logger.info(f"\n Using feedback category override for iteration {current_iteration}: {obj_cat_list}")
            detection_iteration += 1
        elif force_categories is None:
            # TODO: add step to prune hallucinated objects
            if obj_cat_list is None or (current_iteration % update_categories_period == 0 and detection_iteration <= cfg.s5_scene.min_detection_iterations):
                # 1. Detect object categories and 2D boxes in the current image using the VLM
                logger.info("Detecting objects in image...")
                result = vlm_pro(
                    prompt=prompt_object_setlist_specific_floorplane_v3(floor_category),
                    image_paths=current_rgb_fpath,
                    temperature=0,
                    top_p=0,
                    seed=0,
                    print_results=True,
                )
                if result is None:
                    logger.error(
                        f"Failed to query Gemini during object detection at iteration {current_iteration}. "
                        "Stopping so the run can be resumed later."
                    )
                    finalize_stage(
                        stage_cfg=cfg.s5_scene,
                        out_dir=cfg.s5_scene.out_dir,
                        result=StageResult(
                            success=False,
                            additional_info={
                                "resume_iteration": current_iteration,
                                "failure_reason": "vlm_detection_no_response",
                            },
                        ),
                    )
                    return
                # pattern = r'[\[\]\.`"]'  # characters to remove
                result_text = vlm_pro.get_result_text(result=result)
                result_json = parse_json_response(result_text)
                # Handle both {"objects":[], "detections":[]} and a bare list of detections
                if isinstance(result_json, list):
                    result_json = {"detections": result_json}
                obj_cat_list = result_json.get("objects", [])
                detections = result_json.get("detections", [])

                # Backward/forward-compatible parsing in case the VLM omits one field.
                if len(obj_cat_list) == 0 and len(detections) > 0:
                    obj_cat_list = [det.get("name", "") for det in detections if det.get("name")]

                # Build object->point lookup from VLM detections.
                # Prefer bbox center (especially Gemini box_2d). Fallback to point_xy for compatibility.
                numeric_bboxes = []
                has_box_2d = False
                has_bbox_xyxy = False
                for det in detections:
                    b = det.get("bbox_xyxy")
                    if isinstance(b, (list, tuple)) and len(b) == 4:
                        has_bbox_xyxy = True
                    else:
                        b = det.get("box_2d")
                        if isinstance(b, (list, tuple)) and len(b) == 4:
                            has_box_2d = True
                    if not isinstance(b, (list, tuple)) or len(b) != 4:
                        continue
                    try:
                        numeric_bboxes.append([float(v) for v in b])
                    except (TypeError, ValueError):
                        continue

                forced_scale = None
                forced_order = None
                pixel_rescale = None
                if len(numeric_bboxes) > 0:
                    bbox_arr = np.array(numeric_bboxes, dtype=np.float32)
                    max_abs = float(np.max(np.abs(bbox_arr)))

                    # Gemini `box_2d` follows [ymin, xmin, ymax, xmax] in [0, 1000].
                    if has_box_2d and not has_bbox_xyxy:
                        forced_order = "yxyx"
                        if max_abs <= 1.5:
                            forced_scale = "unit"
                        elif max_abs <= 1000.0:
                            forced_scale = "thousand"
                        else:
                            forced_scale = "pixel"
                    else:
                        forced_order = "xyxy"
                        if max_abs <= 1.5:
                            forced_scale = "unit"
                        elif max_abs <= 1000.0 and max_abs < max(W, H) * 0.8:
                            forced_scale = "thousand"
                        else:
                            forced_scale = "pixel"

                    if forced_scale == "pixel":
                        max_x = float(np.max(np.maximum(bbox_arr[:, 0], bbox_arr[:, 2])))
                        max_y = float(np.max(np.maximum(bbox_arr[:, 1], bbox_arr[:, 3])))
                        if max_x > W * 1.02 or max_y > H * 1.02:
                            sx = W / max(max_x, 1.0)
                            sy = H / max(max_y, 1.0)
                            pixel_rescale = (sx, sy)
                            logger.info(
                                f"Applying global pixel bbox rescale from inferred frame "
                                f"({max_x:.1f}, {max_y:.1f}) -> ({W}, {H}) with factors ({sx:.4f}, {sy:.4f})"
                            )

                    logger.info(
                        f"VLM bbox mode inference: forced_order={forced_order}, forced_scale={forced_scale}, "
                        f"n_boxes={len(numeric_bboxes)}, has_bbox_xyxy={has_bbox_xyxy}, has_box_2d={has_box_2d}"
                    )

                parsed_vlm_points = {}
                parsed_vlm_bboxes = {}
                n_point_entries = 0
                n_bbox_fallback_entries = 0
                for det in detections:
                    det_name = det.get("name")
                    if not det_name:
                        continue
                    det_point = det.get("point_xy")
                    det_bbox = det.get("bbox_xyxy")
                    det_bbox_key = "bbox_xyxy"
                    if not isinstance(det_bbox, (list, tuple)) or len(det_bbox) != 4:
                        det_bbox = det.get("box_2d")
                        det_bbox_key = "box_2d"

                    canonical_bbox = None
                    bbox_meta = None
                    point_meta = None
                    if isinstance(det_bbox, (list, tuple)) and len(det_bbox) == 4:
                        if det_bbox_key == "box_2d":
                            canonical_bbox = convert_gemini_box2d_to_xyxy(
                                raw_box2d=det_bbox,
                                img_w=W,
                                img_h=H,
                            )
                            bbox_meta = {"order": "yxyx", "scale": "thousand", "score": 0.0}
                        else:
                            canonical_bbox, bbox_meta = canonicalize_vlm_bbox_xyxy(
                                raw_bbox=det_bbox,
                                img_w=W,
                                img_h=H,
                                forced_scale=forced_scale,
                                forced_order=forced_order,
                                pixel_rescale=pixel_rescale,
                            )
                        if canonical_bbox is None:
                            logger.warning(f"Skipping invalid VLM bbox for '{det_name}': {det_bbox}")
                            continue
                        n_bbox_fallback_entries += 1
                    else:
                        canonical_point, point_meta = canonicalize_vlm_point_xy(
                            raw_point=det_point,
                            img_w=W,
                            img_h=H,
                        ) if isinstance(det_point, (list, tuple)) and len(det_point) == 2 else (None, None)
                        if canonical_point is None:
                            logger.warning(
                                f"Skipping VLM detection without valid bbox_xyxy/box_2d/point_xy for '{det_name}': {det}"
                            )
                            continue
                        n_point_entries += 1
                        cx, cy = canonical_point
                        half = 8
                        canonical_bbox = [
                            float(np.clip(cx - half, 0, W - 1)),
                            float(np.clip(cy - half, 0, H - 1)),
                            float(np.clip(cx + half, 0, W - 1)),
                            float(np.clip(cy + half, 0, H - 1)),
                        ]

                    parsed_vlm_bboxes[det_name] = canonical_bbox
                    if bbox_meta is not None:
                        logger.info(
                            f"VLM bbox normalized for '{det_name}' ({det_bbox_key}): raw={det_bbox} -> "
                            f"{parsed_vlm_bboxes[det_name]} ({bbox_meta['order']}/{bbox_meta['scale']})"
                        )
                    elif point_meta is not None:
                        logger.info(
                            f"VLM point normalized for '{det_name}': raw={det_point} -> "
                            f"{parsed_vlm_points[det_name]} ({point_meta['order']}/{point_meta['scale']})"
                        )
                    primary_name = get_primary_object_name(det_name)
                    if primary_name and primary_name not in parsed_vlm_bboxes:
                        parsed_vlm_bboxes[primary_name] = list(canonical_bbox)

                # Derive robust prompt points from all parsed bboxes:
                # prefer points that stay inside own bbox but outside others.
                parsed_keys = list(parsed_vlm_bboxes.keys())
                for key in parsed_keys:
                    target_bbox = parsed_vlm_bboxes[key]
                    other_bboxes = [parsed_vlm_bboxes[k] for k in parsed_keys if k != key]
                    robust_point, robust_meta = choose_prompt_point_from_bbox(
                        target_bbox=target_bbox,
                        other_bboxes=other_bboxes,
                        img_w=W,
                        img_h=H,
                    )
                    parsed_vlm_points[key] = robust_point
                    if robust_meta["fallback_to_center"]:
                        logger.info(
                            f"Robust bbox point fallback to center for '{key}': point={robust_point}, bbox={target_bbox}"
                        )
                    else:
                        logger.info(
                            f"Robust bbox point selected for '{key}': point={robust_point}, bbox={target_bbox}"
                        )
                vlm_obj_prompt_points = parsed_vlm_points
                vlm_obj_prompt_bboxes = parsed_vlm_bboxes

                logger.info(f"\n Detected categories: {obj_cat_list}")
                if len(detections) > 0:
                    logger.info(
                        f"Detected VLM entries: total={len(detections)}, "
                        f"point_xy={n_point_entries}, bbox_fallback={n_bbox_fallback_entries}"
                    )
                if len(vlm_obj_prompt_points) > 0:
                    logger.info(f"Detected VLM-guided points: {len(vlm_obj_prompt_points)}")
                if len(obj_cat_list) == 0:
                    logger.warning("No objects detected")
                    detection_iteration += 1
                    continue

                detection_iteration += 1
        else:
            # Use hardcoded categories specified
            obj_cat_list = [cat for cat in force_categories]
            print(f"\n Using forced categories: {obj_cat_list}")
            detection_iteration += 1

        # 2. Segment object instances / masks using SAM3 -> N objects detected to decompose
        pil_img = Image.open(current_rgb_fpath)
        np_img = np.array(pil_img)
        all_masks, all_boxes_xyxy, all_logits, phrases = np.zeros((0, 1, H, W), dtype=bool), np.zeros((0, 4)), np.zeros((0,)), []

        # Point source priority:
        # (1) user-selected point from 5b feedback (iteration-level)
        # (2) VLM bbox-derived robust point for each object category (object-level)
        # Text-prompt masks are always preferred. Point prompts are used only as fallback.
        user_point_xy = None
        used_vlm_point_guidance = False
        if current_iteration in point_prompts:
            point_coords = point_prompts[current_iteration]
            x = int(np.clip(np.round(point_coords[0]), 0, W - 1))
            y = int(np.clip(np.round(point_coords[1]), 0, H - 1))
            user_point_xy = (x, y)
            logger.info(f"\n Using user hybrid text+point prompt for iteration {current_iteration}: {user_point_xy}")

        if user_point_xy is not None:
            # Keep existing feedback behavior: one user point guides this iteration.
            for obj_cat in obj_cat_list:
                masks, boxes_xyxy, logits = sam3.predict_segmentation(pil_img=pil_img, text_prompt=obj_cat)
                all_masks = np.append(all_masks, masks, axis=0)
                all_boxes_xyxy = np.append(all_boxes_xyxy, boxes_xyxy, axis=0)
                all_logits = np.append(all_logits, logits, axis=0)
                phrases += [obj_cat] * len(masks)

            if len(all_masks) > 0:
                logger.info("Using text-prompt masks; skipping user point filtering because text segmentation succeeded")
            else:
                logger.info("No text masks found; falling back to point-only segmentation with user point")
                masks, boxes_xyxy, logits = sam3.predict_segmentation_with_point(
                    pil_img=pil_img,
                    point_coords=user_point_xy,
                    point_labels=[1],  # foreground point
                    multimask_output=True,  # Get multiple mask candidates
                )
                if len(masks) > 0:
                    best_idx = np.argmax(logits)
                    masks = masks[best_idx:best_idx + 1]
                    boxes_xyxy = boxes_xyxy[best_idx:best_idx + 1]
                    logits = logits[best_idx:best_idx + 1]
                all_masks = masks
                all_boxes_xyxy = boxes_xyxy
                all_logits = logits
                if current_iteration in category_overrides:
                    phrases = [category_overrides[current_iteration]] * len(masks)
                else:
                    phrases = ["point_selected_object"] * len(masks)
        else:
            # No user point: use text prompts first, then fall back to VLM point guidance only if text fails.
            for obj_cat in obj_cat_list:
                masks, boxes_xyxy, logits = sam3.predict_segmentation(pil_img=pil_img, text_prompt=obj_cat)

                obj_point_xy = vlm_obj_prompt_points.get(obj_cat)
                if obj_point_xy is None:
                    obj_point_xy = vlm_obj_prompt_points.get(get_primary_object_name(obj_cat))

                if obj_point_xy is not None and len(masks) == 0:
                    used_vlm_point_guidance = True
                    logger.info(
                        f"No text mask for '{obj_cat}'; using point-only segmentation with VLM point {obj_point_xy}"
                    )
                    masks_pt, boxes_pt, logits_pt = sam3.predict_segmentation_with_point(
                        pil_img=pil_img,
                        point_coords=obj_point_xy,
                        point_labels=[1],
                        multimask_output=True,
                    )
                    if len(masks_pt) > 0:
                        best_idx = np.argmax(logits_pt)
                        masks = masks_pt[best_idx:best_idx + 1]
                        boxes_xyxy = boxes_pt[best_idx:best_idx + 1]
                        logits = logits_pt[best_idx:best_idx + 1]
                    else:
                        masks = np.zeros((0, 1, H, W), dtype=bool)
                        boxes_xyxy = np.zeros((0, 4))
                        logits = np.zeros((0,))

                all_masks = np.append(all_masks, masks, axis=0)
                all_boxes_xyxy = np.append(all_boxes_xyxy, boxes_xyxy, axis=0)
                all_logits = np.append(all_logits, logits, axis=0)
                phrases += [obj_cat] * len(masks)

        # Debug: when Gemini bbox-point guidance is used (and no 5b user point), dump VLM bbox overlay per iteration
        if user_point_xy is None and used_vlm_point_guidance:
            debug_boxes = []
            debug_phrases = []
            debug_points = []
            for obj_cat in obj_cat_list:
                bbox_xyxy = vlm_obj_prompt_bboxes.get(obj_cat)
                if bbox_xyxy is None:
                    bbox_xyxy = vlm_obj_prompt_bboxes.get(get_primary_object_name(obj_cat))
                if bbox_xyxy is None:
                    continue
                point_xy = vlm_obj_prompt_points.get(obj_cat)
                if point_xy is None:
                    point_xy = vlm_obj_prompt_points.get(get_primary_object_name(obj_cat))
                debug_boxes.append(bbox_xyxy)
                debug_phrases.append(obj_cat)
                debug_points.append(point_xy)
            if len(debug_boxes) > 0:
                debug_boxes = np.array(debug_boxes, dtype=np.float32)
                debug_logits = np.ones((len(debug_boxes),), dtype=np.float32)
                # annotate() returns BGR image. Draw selected prompt points in BGR, then convert to RGB for PIL.
                gemini_annotated_img_bgr = annotate(np_img, debug_boxes, debug_phrases, debug_logits)
                for i, point_xy in enumerate(debug_points):
                    if point_xy is None:
                        continue
                    px, py = int(point_xy[0]), int(point_xy[1])
                    cv2.circle(gemini_annotated_img_bgr, (px, py), 6, (0, 0, 255), -1)  # red dot
                    cv2.circle(gemini_annotated_img_bgr, (px, py), 10, (255, 255, 255), 2)  # white ring
                    cv2.putText(
                        gemini_annotated_img_bgr,
                        f"p{i+1}",
                        (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                gemini_annotated_img = gemini_annotated_img_bgr[:, :, ::-1]
                gemini_annotated_img_fpath = f"{out_dir}/gemini_detected_phrases/iter_{current_iteration}.png"
                Image.fromarray(gemini_annotated_img).save(gemini_annotated_img_fpath)
                logger.info(f"Saved Gemini debug bbox image: {gemini_annotated_img_fpath}")

        
        # image_source, image = gsam.load_image(current_rgb_fpath)
        # boxes_xyxy, logits, gsam_phrases = gsam.predict_boxes(image, obj_cat_list)

        # If no objects are detected and we have reached the minimum number of detection iterations, we are done
        if len(all_masks) == 0 and detection_iteration >= cfg.s5_scene.min_detection_iterations:
            break
        elif len(all_masks) == 0:
            # A rerun can skip an index a previous run kept; drop any stale kept
            # artifacts so they never coexist with the skip record.
            delete_iteration_data(out_dir, current_iteration, current_iteration)
            # Record json for skipped iterations
            _skip_fpath = f"{out_dir}/skipped_iterations/iter_{current_iteration}.json"
            with atomic_output_path(_skip_fpath) as _tmp_skip, open(_tmp_skip, "w+") as f:
                json.dump({
                    "iteration": current_iteration,
                    "reason": "no_masks_detected",
                    "detection_iteration": detection_iteration,
                    "base_iter": last_kept_iteration,
                }, f, indent=4)
            current_iteration += 1
            continue

        # all_masks = gsam.predict_segmentation(image_source, boxes_xyxy, multimask_output=False)

        # Sanity check to make sure we have no degenerate masks
        if all(mask.squeeze(axis=0).sum() == 0 for mask in all_masks):
            break

        # 3. Prune the masks
        pruned_masks, pruned_boxes_xyxy, pruned_logits, pruned_phrases = sam3.prune_redundant_masks(
            masks=[mask.squeeze(axis=0) for mask in all_masks],
            boxes_xyxy=all_boxes_xyxy,
            logits=all_logits,
            phrases=phrases,
            polygon_relative_intersection_threshold=0.97,
            polygon_relative_area_threshold=0.9,
            obj_mask_intersect_area_threshold=0.8,
            duplicate_mask_intersect_area_threshold=cfg.s5_scene.duplicate_mask_intersect_area_threshold,
            boundary_proportion=BOUNDARY_PROPORTION,
            text_encoder=text_encoder,
            text_encoder_similarity_threshold=0.85,
            verbose=True,
        )
        if len(pruned_masks) == 0:
            break

        # Prune any below surface as well
        cumulative_remaining_mask = np.ones_like(pruned_masks[0])
        pruned_masks, pruned_boxes_xyxy, pruned_logits, pruned_phrases = sam3.prune_masks_below_surface(
            masks=pruned_masks,
            boxes_xyxy=pruned_boxes_xyxy,
            logits=pruned_logits,
            phrases=pruned_phrases,
            pc=resized_padded_pc,
            valid_mask=cumulative_remaining_mask,
            z_threshold=cfg.s5_scene.z_prune_threshold,
            z_threshold_proportion=cfg.s5_scene.z_prune_threshold_proportion,
            verbose=True,
        )
        if len(pruned_masks) == 0:
            break

        # Disentangle masks that might subsume others
        pruned_masks = sam3.disentangle_masks(
            masks=pruned_masks,
            phrases=pruned_phrases,
            overlap_threshold_proportion=cfg.s5_scene.overlap_threshold_proportion,
            verbose=True,
        )


        pruned_phrases_unique_ids = [f"{phrase}_{i}" for i, phrase in enumerate(pruned_phrases)]

        # Annotate
        annotated_img_fpath = f"{out_dir}/detected_phrases/iter_{current_iteration}.png"
        annotated_img = annotate(np_img, pruned_boxes_xyxy, pruned_phrases, pruned_logits)[:,:,::-1]
        Image.fromarray(annotated_img).save(annotated_img_fpath)

        # 4. Use analytical heuristics using each objects' convex hull to determine which object should be removed (prioritize foreground + objects with nothing beneath them)
        # TODO: Need to handle special case when there is only 1 obj left (so nothing to choose between)
        obj_chs = dict()
        skipped_objects = []
        for mask, phrase in zip(pruned_masks, pruned_phrases_unique_ids):
            if not np.any(mask):
                logger.warning(f"Skipping object '{phrase}' because mask is empty after disentangling.")
                skipped_objects.append({"phrase": phrase, "reason": "empty_mask_after_disentangle"})
                continue

            mask_eroded_255 = erode_mask((mask * 255).astype(np.uint8), kernel_size=3)

            # Depth is a different shape so need to resize RGB and mask now
            rgb_resized = cv2.resize(current_rgb, (padded_W, padded_H))
            rgb_resized_unpadded = unpad_image(rgb_resized, delta_w, delta_h)
            mask_eroded_resized_255 = cv2.resize(mask_eroded_255, (padded_W, padded_H))
            mask_eroded_resized_unpadded = unpad_image(mask_eroded_resized_255, delta_w, delta_h).astype(bool)
            if not np.any(mask_eroded_resized_unpadded):
                logger.warning(f"Skipping object '{phrase}' because eroded mask has no valid pixels.")
                skipped_objects.append({"phrase": phrase, "reason": "empty_eroded_mask"})
                continue

            obj_pc = current_pc[mask_eroded_resized_unpadded]
            obj_rgb = rgb_resized_unpadded[mask_eroded_resized_unpadded] / 255
            obj_pcd = o3d.geometry.PointCloud()
            obj_pcd.points = o3d.utility.Vector3dVector(obj_pc)
            obj_pcd.colors = o3d.utility.Vector3dVector(obj_rgb)


            obj_pcd_pruned, indices = denoise_obj_point_cloud_v2(
                pcd_obj=obj_pcd,
                visualize_process=False,
                visualize_result=False,
                target_density_cm2=5.0,
                outlier_removal_n_neighbors=None,
                outlier_removal_std_ratio=1.0, #0.75,
                dbscan_eps=0.005, #None, #0.01,
                dbscan_min_pts=None, #3, #None, #None, #10, #None,
                cluster_proportion_threshold=0.06,
            )

            obj_pc = np.asarray(obj_pcd_pruned.points)
            if len(obj_pc) < 4:
                logger.warning(f"Skipping object '{phrase}' because point cloud has only {len(obj_pc)} points after pruning.")
                skipped_objects.append({"phrase": phrase, "reason": "insufficient_point_cloud", "n_points": int(len(obj_pc))})
                continue

            # Compute convex hull via trimesh
            obj_ch = trimesh.convex.convex_hull(obj_pc)

            # Erode the convex hull slightly to avoid slight overlaps
            obj_ch = trimesh.convex.convex_hull(np.asarray(obj_ch.vertices) - 0.0025 * np.asarray(obj_ch.vertex_normals))

            obj_chs[phrase] = obj_ch

        if not obj_chs:
            logger.warning("No valid object hulls remain after mask/point-cloud pruning; ending decomposition early.")
            break

        # With these convex hulls, determine which object to remove using prioritized heuristics:
        #   1. Prefer non-occluded objects (visible from camera)
        #   2. Prefer objects on top of / inside other objects (topmost first)
        #   3. Prefer objects closer to camera (foreground first)
        obj_phrase_unique_id_to_remove = compute_object_removal_order(
            obj_chs=obj_chs,
            cam2world_tf=cam2world_tf,
        )
        obj_idx_to_remove = pruned_phrases_unique_ids.index(obj_phrase_unique_id_to_remove)
        obj_phrase_to_remove = pruned_phrases[obj_idx_to_remove]
        logger.info(f"\n ({obj_phrase_to_remove})\n")

        # 5a. Isolate the object
        removal_mask = pruned_masks[obj_idx_to_remove]
        removal_mask_255 = removal_mask.astype(np.uint8) * 255
        Image.fromarray(removal_mask_255).save(f"{out_dir}/removal_mask/iter_{current_iteration}.png")
        # Save the masked object -- crop to center and resize
        mask_h_idxs, mask_w_idxs = np.where(removal_mask)
        min_h, max_h = mask_h_idxs.min(), mask_h_idxs.max()
        min_w, max_w = mask_w_idxs.min(), mask_w_idxs.max()
        margin = int(np.ceil(max(max_h - min_h, max_w - min_w) * 0.5)) #0.05))
        margin_focus = int(np.ceil(max(max_h - min_h, max_w - min_w) * 0.1))
        masked_images = {
            "with_background": deepcopy(current_rgb),
            "no_background": deepcopy(current_rgb),
            "no_background_focus": deepcopy(current_rgb),
            "resized_removal_mask": deepcopy(removal_mask).astype(np.uint8) * 255,
            "resized_removal_mask_focus": deepcopy(removal_mask).astype(np.uint8) * 255,
        }

        masked_coords = dict()
        for mask_name, masked_image in masked_images.items():
            cropped_masked_image, masked_coords[mask_name] = crop_image_with_mask(
                image=masked_image,
                mask=removal_mask,
                buffer_pixels=margin_focus if "focus" in mask_name else margin,
                target_aspect_ratio=target_ratio,
            )
            resized_cropped_masked_image = cv2.resize(cropped_masked_image, resolution)
            masked_images[mask_name] = resized_cropped_masked_image

        # Make sure to actually remove the background
        for suffix in {"", "_focus"}:
            masked_images[f"no_background{suffix}"][~masked_images[f"resized_removal_mask{suffix}"].astype(bool)] = 255
            masked_images[f"no_background{suffix}"] = np.concatenate([masked_images[f"no_background{suffix}"], masked_images[f"resized_removal_mask{suffix}"][..., np.newaxis]], axis=-1)


        Image.fromarray(masked_images["no_background"]).save(f"{out_dir}/masked_object/iter_{current_iteration}.png")
        Image.fromarray(masked_images["no_background_focus"]).save(f"{out_dir}/masked_object_focus/iter_{current_iteration}.png")
        Image.fromarray(masked_images["with_background"]).save(f"{out_dir}/masked_object_background/iter_{current_iteration}.png")
        Image.fromarray(masked_images["resized_removal_mask"]).save(f"{out_dir}/removal_mask_resized/iter_{current_iteration}.png")


        if crop_removal_images:
            annotated_removal_image_raw = deepcopy(masked_images["with_background"])
            mask_resized_h_idxs, mask_resized_w_idxs = np.where(masked_images["resized_removal_mask"])
            min_resized_h, max_resized_h = mask_resized_h_idxs.min(), mask_resized_h_idxs.max()
            min_resized_w, max_resized_w = mask_resized_w_idxs.min(), mask_resized_w_idxs.max()
            bbox_x_range = (min_resized_w, max_resized_w)
            bbox_y_range = (min_resized_h, max_resized_h)
        else:
            annotated_removal_image_raw = deepcopy(current_rgb)
            bbox_x_range = (min_w, max_w)
            bbox_y_range = (min_h, max_h)

        # 5b. Annotate the object and remove it with the image model
        success = False
        for obj_removal_prompt_type in obj_removal_prompt_order:
            annotated_removal_image = deepcopy(annotated_removal_image_raw)
            if obj_removal_prompt_type == "bbox":
                draw_bounding_box(
                    image=annotated_removal_image,
                    x_range=bbox_x_range,
                    y_range=bbox_y_range,
                    color=REMOVAL_OUTLINE_COLOR,
                    thickness=int(np.round(img_scale / 400)),
                )
                prompt_fcn = prompt_flux_object_removal_bbox
            elif obj_removal_prompt_type in {"outline", "mask"}:
                obj_mask = masked_images["resized_removal_mask"] if crop_removal_images else removal_mask.astype(np.uint8)
                draw_mask_boundary(
                    image=annotated_removal_image,
                    mask=obj_mask,
                    color=REMOVAL_OUTLINE_COLOR,
                    thickness=int(np.ceil(img_scale / 500)),
                )
                if obj_removal_prompt_type == "mask":
                    obj_mask_dilated = dilate_mask(obj_mask, kernel_width=3, iterations=5)      # 5
                    annotated_removal_image = np.where(obj_mask_dilated[..., np.newaxis], np.array(REMOVAL_OUTLINE_COLOR, dtype=np.uint8).reshape(1, 1, 3), annotated_removal_image)
                    prompt_fcn = prompt_flux_object_removal_mask
                else:
                    prompt_fcn = prompt_flux_object_removal_outline
            else:
                raise ValueError(f"Unknown removal prompt type, got: {obj_removal_prompt_type}!")

            # Pixels we actually painted with REMOVAL_OUTLINE_COLOR.
            annotation_pixels = np.all(
                annotated_removal_image == np.array(REMOVAL_OUTLINE_COLOR, dtype=np.uint8), axis=-1
            )
            annotation_check_mask = dilate_mask(
                annotation_pixels.astype(np.uint8), kernel_width=5, iterations=4
            ).astype(bool)

            # Write this image to disk
            annotated_removal_image_fpath = f"{out_dir}/pre_object_removal/iter_{current_iteration}.png"
            Image.fromarray(annotated_removal_image).save(annotated_removal_image_fpath)

            logger.info(f"Removing selected object from image using prompt type: {obj_removal_prompt_type}...")
            n_attempts = 3
            start_seed = 0
            prompt = prompt_fcn(obj_phrase=obj_phrase_to_remove)
            for i in range(n_attempts):
                logger.info(f"Attempt {i + 1} / {n_attempts}...")
                removal_seed = start_seed + i
                removal_kwargs = {
                    "prompt": prompt,
                    "seed": removal_seed,
                    "print_results": cfg.visualize,
                }
                if removal_model_name == "gemini":
                    logger.info(f"Calling {removal_model_name} to remove '{obj_phrase_to_remove}' (may take up to 5 min)...")
                    result = removal_model(
                        image_paths=annotated_removal_image_fpath,
                        temperature=0,
                        top_p=0,
                        **removal_kwargs,
                    )
                    logger.info(f"Object removal done for '{obj_phrase_to_remove}'.")
                    images = [] if result is None else removal_model.get_result_images(result=result)
                    if not images:
                        logger.error("Failed to query image model for image generation.")
                        continue
                    removed_obj_img_raw = np.array(images[0])
                elif "gemini" in removal_model_name:
                    result = removal_model(
                        image_paths=annotated_removal_image_fpath,
                        temperature=0,
                        top_p=0,
                        **removal_kwargs,
                    )
                    images = [] if result is None else removal_model.get_result_images(result=result)
                    if not images:
                        logger.error("Failed to query Gemini for image generation.")
                        continue
                    removed_obj_img_raw = np.array(images[0])
                elif removal_model_name == "flux":
                    removed_obj_img_raw = np.array(removal_model(
                        image_path=annotated_removal_image_fpath,
                        guidance_scale=2.5,
                        num_inference_steps=20,
                        max_sequence_length=512,
                        **removal_kwargs,
                    ))
                else:
                    raise NotImplementedError
                # Resize to expected resolution (FLUX/Gemini may return slightly different dimensions)
                removed_obj_img_raw = cv2.resize(removed_obj_img_raw, resolution)
                
                # Check if any residual highlights
                if get_num_pixels_with_color(
                    removed_obj_img_raw, REMOVAL_OUTLINE_COLOR, tolerance=50, mask=annotation_check_mask
                ) > (img_scale * 0.33):
                    logger.warning("Generated image rejected because found bright green pixels (likely representing annotation bbox / outline / mask)")
                    rejected_fpath = f"{out_dir}/pre_object_removal/iter_{current_iteration}_rejected_{obj_removal_prompt_type}_seed{removal_seed}.png"
                    Image.fromarray(removed_obj_img_raw).save(rejected_fpath)
                    logger.info(f"Saved rejected candidate for inspection: {rejected_fpath}")
                    continue

                success = True
                break

            if success:
                break

        if not success:
            logger.error(
                f"Failed to generate a valid object-removal image at iteration {current_iteration}. "
                "Stopping so the run can be resumed later."
            )
            finalize_stage(
                stage_cfg=cfg.s5_scene,
                out_dir=cfg.s5_scene.out_dir,
                result=StageResult(
                    success=False,
                    additional_info={
                        "resume_iteration": current_iteration,
                        "failure_reason": "object_removal_generation_failed",
                    },
                ),
            )
            return

        if crop_removal_images:
            # Need to invert the cropping operation to "splice" back in the removed obj img_raw
            coords = masked_coords["with_background"]
            removed_obj_img_raw_resized = cv2.resize(removed_obj_img_raw, (coords[2] - coords[0] + 1, coords[3] - coords[1] + 1))
            removed_obj_img_raw_spliced = deepcopy(current_rgb)
            removed_obj_img_raw_spliced[coords[1]:coords[3] + 1, coords[0]:coords[2] + 1, :] = removed_obj_img_raw_resized
        else:
            removed_obj_img_raw_spliced = removed_obj_img_raw

        # If we're merging, dilate the mask slightly and then only replace the pixels from the old image within the mask
        # with the pixels from the removed_obj_img
        if merge_output_images:
            mask_dilated_255 = dilate_mask(removal_mask_255, kernel_width=int(np.round(img_scale / 70)), iterations=5)
            # Blur using Gaussian blur
            mask_dilated_blurred = (cv2.GaussianBlur(mask_dilated_255, (99, 99), 0) / 255)[..., np.newaxis]
            removed_obj_img = (mask_dilated_blurred * removed_obj_img_raw_spliced + (1 - mask_dilated_blurred) * current_rgb).astype(np.uint8)

        else:
            raise ValueError("Must merge images!")
            removed_obj_img = removed_obj_img_raw

        removed_obj_fpath = f"{out_dir}/post_object_removal/iter_{current_iteration}.png"
        Image.fromarray(removed_obj_img).save(removed_obj_fpath)

        # 6. Generate new synthetic depth map using Prior-Depth-Anything + geometric backend (DepthPro or DA3)
        img_resized = cv2.resize(removed_obj_img, (padded_W, padded_H))
        img_resized_unpadded = unpad_image(img_resized, delta_w, delta_h)
        mask_resized = cv2.resize(removal_mask_255, (padded_W, padded_H))
        mask_resized_unpadded = unpad_image(mask_resized, delta_w, delta_h)
        mask_resized_unpadded_dilated = dilate_mask(mask_resized_unpadded, kernel_width=3, iterations=5).astype(bool)
        metric_depth = deepcopy(current_depth)
        metric_depth[mask_resized_unpadded_dilated] = 0

        # Compute geometric depth using selected backend
        if pda_geometric_backend == "depth_pro":
            # Load and preprocess an image.
            # image, _, f_px = depth_pro.load_rgb(image_original_path)
            dp_image = dp_transform(img_resized_unpadded)

            # TODO: Get f_px from GT stereo camera?
            dp_prediction = dp.infer(dp_image, f_px=None)  # tbd: get GT f_px from stereo camera
            geometric_depth = dp_prediction["depth"].detach().cpu().numpy()  # Depth in [m].
            # focallength_px = dp_prediction["focallength_px"]  # Focal length in pixels.
        elif pda_geometric_backend == "da3":
            geometric_depth = da3_geometric_model.infer_depth(
                image=img_resized_unpadded,
                resolution=cfg.s5_scene.pda_da3_resolution,
                resize_to_input=True,
            )
        else:
            raise ValueError(f"Unknown pda_geometric_backend: {pda_geometric_backend}")

        # torch.linalg.lstsq (inside PriorDepthAnything) only supports float/double.
        # Ensure both inputs are fp32 to avoid float16/bfloat16 runtime errors.
        metric_depth = metric_depth.astype(np.float32, copy=False)
        geometric_depth = geometric_depth.astype(np.float32, copy=False)

        # Run prior depth anything
        output = priorda.infer_one_sample(
            image=img_resized_unpadded,  # image_path,
            prior=metric_depth,  # prior_masked_path,
            geometric=geometric_depth,
            # geometric=metric_depth_path,
            # double_global=True,
            # prior_cover=True,
            visualize=cfg.visualize,
        ).detach().cpu().numpy()

        merged_output = np.where(metric_depth == 0, deepcopy(output), metric_depth)

        # Check pixels against original image to see if we segmented a valid image, and not an "imagined" object
        # due to the inpainting process accidentally adding a spurious object. We enforce a minimum proportion of pixels
        # to completely align with the original image
        mask_size = removal_mask.sum()
        # removal_original_pixel_alignment = np.all(resized_padded_img == current_rgb, axis=-1)
        # n_aligned_pixels = removal_original_pixel_alignment[removal_mask].sum()
        # Both images are uint8; subtract as int16 so negative channel differences don't wrap to ~255.
        removal_original_pixel_alignment = np.linalg.norm(
            resized_padded_img.astype(np.int16) - current_rgb.astype(np.int16), axis=-1) < 50
        n_aligned_pixels = removal_original_pixel_alignment[removal_mask].sum()
        valid_pixel_proportion = float(n_aligned_pixels / mask_size)
        is_valid_removed_obj = bool(valid_pixel_proportion > min_valid_pixel_prop)
        if not is_valid_removed_obj:
            logger.warning(
                f"Object {obj_phrase_to_remove} is not a valid removed obj! "
                f"(valid_pixel_proportion={valid_pixel_proportion:.3f}, threshold={min_valid_pixel_prop})"
            )
            # from IPython import embed; embed()

        # Save output
        out_depth_fpath = f"{out_dir}/metric_depth/iter_{current_iteration}.npy"
        np.save(out_depth_fpath, merged_output)
        # imageio.imwrite(out_depth_fpath.replace(".npy", ".png"), cv2.cvtColor(vis_disparity(merged_output), cv2.COLOR_BGR2RGB))
        Image.fromarray(cv2.cvtColor(vis_disparity(merged_output), cv2.COLOR_BGR2RGB)).save(
            out_depth_fpath.replace(".npy", ".png"))

        # Visualize if requested
        if cfg.visualize:
            # vis = np.concatenate([metric_depth, geometric_depth, output, merged_output], axis=1)
            vis = np.concatenate([metric_depth, geometric_depth, merged_output], axis=1)
            Image.fromarray((vis * 180).astype(np.uint8)).show()
            pc = compute_point_cloud_from_depth(merged_output, K)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc.reshape(-1, 3))
            pcd.colors = o3d.utility.Vector3dVector(img_resized_unpadded.reshape(-1, 3) / 255)
            o3d.visualization.draw_geometries([pcd])

        # Update values
        current_rgb_fpath = removed_obj_fpath
        current_rgb = removed_obj_img
        current_depth = merged_output
        current_pc = compute_point_cloud_from_depth(
            depth=merged_output,
            K=K,
            world_to_cam_tf=cam2world_tf,
        )

        # A rerun can detect an object at an index a previous run skipped; drop the
        # stale skip record so the two never coexist for the same iteration.
        _stale_skip_fpath = f"{out_dir}/skipped_iterations/iter_{current_iteration}.json"
        if os.path.exists(_stale_skip_fpath):
            os.remove(_stale_skip_fpath)

        # Published atomically: this file is what the stage-6 listener watches for, so a
        # consumer must never see it half-written.
        _obj_cat_fpath = f"{out_dir}/obj_cat_list/iter_{current_iteration}.json"
        with atomic_output_path(_obj_cat_fpath) as _tmp_obj_cat, open(_tmp_obj_cat, "w+") as f:
            saved_vlm_categories = [get_primary_object_name(name) for name in obj_cat_list]
            saved_pruned_phrases = [get_primary_object_name(name) for name in pruned_phrases]
            saved_removed_obj_phrase = get_primary_object_name(obj_phrase_to_remove)
            json.dump({
                "vlm_categories": saved_vlm_categories,
                "pruned_phrases": saved_pruned_phrases,
                "removed_obj_phrase": saved_removed_obj_phrase,
                "vlm_categories_full": obj_cat_list,
                "pruned_phrases_full": pruned_phrases,
                "removed_obj_phrase_full": obj_phrase_to_remove,
                "is_valid_removed_obj": is_valid_removed_obj,
                "valid_pixel_proportion": valid_pixel_proportion,
                "valid_pixel_proportion_threshold": min_valid_pixel_prop,
                "skipped_objects": skipped_objects,
                # Which iteration's post-removal image/depth this object was detected
                # against (null = the original source frame). Iteration numbers can have
                # gaps, so consumers (stage 8/8b) must use this rather than idx - 1.
                "base_iter": last_kept_iteration,
            }, f, indent=4)

        # Only remove the category from the list if this was the sole instance detected.
        # If multiple instances of the same category were found, keep the category so that
        # remaining instances can still be selected in subsequent iterations.
        instances_of_removed_category = pruned_phrases.count(obj_phrase_to_remove)
        if instances_of_removed_category <= 1 and obj_phrase_to_remove in obj_cat_list:
            obj_cat_list.remove(obj_phrase_to_remove)
        elif instances_of_removed_category > 1:
            logger.info(f"Keeping category '{obj_phrase_to_remove}' in list — "
                        f"{instances_of_removed_category - 1} other instance(s) of the same category remain")

        if cfg.debug:
            inp = input(f"Press ENTER to continue, or b to breakpoint...")
            if "b" in inp:
                from IPython import embed; embed()

        last_kept_iteration = current_iteration
        current_iteration += 1

    # Done!
    logger.info(f"\n{'*' * 30}\nCompleted scene decomposition!\n{'*' * 30}\n")

    finalize_stage(
        stage_cfg=cfg.s5_scene,
        out_dir=cfg.s5_scene.out_dir,
        result=StageResult(success=True),
    )

if __name__ == "__main__":
    main()

# TODO: Handle incomplete masks (e.g.: cup with cropped out pen masks inside it) - maybe check object against VLM , then potentially complete object? Maybe check via analytical depth check -- check depth gradient at the edge of the object mask edge, and if it's negative (i.e.: the pixels right next to the object have "less depth" i.e.: are closer to the camera), then it's likely that the object is occluded. But what about "objectA resting on objectB" edge case? Idea: Secondary check where we check whether those pixel Z values are above (which implies occlusion) or below objectA's min z value observed. Another obvious occlusion check: multiple islands of pixels
