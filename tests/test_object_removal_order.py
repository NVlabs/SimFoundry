# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script for the object removal ordering heuristics in stage 5.

Tests the compute_object_removal_order() function using:
  1. Synthetic convex hulls (no GPU/model dependencies)
  2. Real droid_desk_1 data (requires simfoundry env + SAM3)

Usage:
  # Quick synthetic tests only (no GPU needed):
    python tests/test_object_removal_order.py --synthetic-only

  # Full test including real data (needs simfoundry env):
    python tests/test_object_removal_order.py
"""
import sys
import os
import argparse
import importlib.util
import logging
import numpy as np
import pytest
import trimesh

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
DATA_ROOT = os.environ.get("SIMFOUNDRY_TEST_DATA_ROOT", os.path.join(project_root, "Data"))

# Import compute_object_removal_order from 5_decompose_scene.py (can't use normal
# import since the filename starts with a digit)
_module_path = os.path.join(
    project_root,
    "scripts",
    "pipeline",
    "A_reconstruction",
    "stages",
    "5_decompose_scene.py",
)
_spec = importlib.util.spec_from_file_location("_decompose_scene", _module_path,
                                                submodule_search_locations=[])
# We only need the function, not the full module (which has heavy deps like hydra/torch).
# So we extract it by reading the source and exec'ing just the function + its deps.
# This avoids loading SAM3/Gemini/etc just to test the ordering logic.
def _load_ordering_function():
    """Load compute_object_removal_order without importing the full pipeline module."""
    from collections import defaultdict, deque

    # Read the source
    with open(_module_path, "r") as f:
        source = f.read()

    # Extract the function: starts at "def compute_object_removal_order(" and ends
    # at the next top-level def/class/constant (identified by lines starting with
    # a keyword or uppercase letter, excluding closing parens and blank lines)
    import re
    func_start = source.index("def compute_object_removal_order(")
    # Find the end by locating the next top-level definition after the function
    rest = source[func_start:]
    # Match next top-level def, class, or UPPER_CASE constant assignment
    match = re.search(r"\n(?:def |class |[A-Z][A-Z_]+ *=)", rest[1:])
    if match:
        func_source = rest[:match.start() + 1]
    else:
        func_source = rest
    func_source = func_source.rstrip()

    # Execute in a namespace with required dependencies
    namespace = {
        "np": np,
        "trimesh": trimesh,
        "logger": logging.getLogger("5_decompose_scene"),
        "defaultdict": defaultdict,
        "deque": deque,
    }
    exec(func_source, namespace)
    return namespace["compute_object_removal_order"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Load the function once
compute_object_removal_order = _load_ordering_function()


def _require_scene_fixture(scene_name, required_relative_paths):
    data_root = os.path.join(DATA_ROOT, scene_name)
    missing = [
        relpath
        for relpath in required_relative_paths
        if not os.path.exists(os.path.join(data_root, relpath))
    ]
    if missing:
        pytest.skip(
            f"Optional reference fixture '{scene_name}' is missing under {DATA_ROOT}; "
            f"first missing path: {missing[0]}"
        )
    return data_root


def make_box_hull(center, half_extents):
    """Create a convex hull from an axis-aligned box."""
    cx, cy, cz = center
    hx, hy, hz = half_extents
    corners = np.array([
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz - hz],
        [cx - hx, cy + hy, cz - hz],
        [cx + hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz],
        [cx + hx, cy - hy, cz + hz],
        [cx - hx, cy + hy, cz + hz],
        [cx + hx, cy + hy, cz + hz],
    ])
    return trimesh.convex.convex_hull(corners)


def make_bowl_hull(center, radius, height):
    """Create a convex hull approximating a bowl (hemisphere + cylinder top)."""
    # Sample points on a hemisphere + flat rim
    n = 200
    phi = np.random.uniform(0, 2 * np.pi, n)
    theta = np.random.uniform(0, np.pi / 2, n)
    x = radius * np.sin(theta) * np.cos(phi) + center[0]
    y = radius * np.sin(theta) * np.sin(phi) + center[1]
    z = -radius * np.cos(theta) + center[2] + height  # hemisphere bottom at center_z
    # Add rim points at the top
    n_rim = 50
    phi_rim = np.linspace(0, 2 * np.pi, n_rim)
    x_rim = radius * np.cos(phi_rim) + center[0]
    y_rim = radius * np.sin(phi_rim) + center[1]
    z_rim = np.full(n_rim, center[2] + height)
    pts = np.column_stack([
        np.concatenate([x, x_rim]),
        np.concatenate([y, y_rim]),
        np.concatenate([z, z_rim]),
    ])
    return trimesh.convex.convex_hull(pts)


def make_camera_transform(cam_pos, look_at, up=np.array([0, 0, 1])):
    """Create a 4x4 cam2world transform from camera position and look-at point."""
    forward = look_at - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    # Camera convention: -Z is forward, X is right, Y is up
    cam2world = np.eye(4)
    cam2world[:3, 0] = right
    cam2world[:3, 1] = true_up
    cam2world[:3, 2] = -forward  # camera looks along -Z
    cam2world[:3, 3] = cam_pos
    return cam2world


# ---------------------------------------------------------------------------
# Synthetic tests
# ---------------------------------------------------------------------------

def test_toast_on_plate():
    """
    Scenario: Toast sitting on top of a flat plate. Both visible from camera.
    Expected: Toast should be removed first (on top, non-occluded, foreground).
    This is the exact failure case from droid_desk_1.
    """
    logger.info("=" * 60)
    logger.info("TEST: Toast on plate (flat surface)")
    logger.info("=" * 60)

    # Plate: flat disc-like object on table (z=0 is table surface)
    plate = make_box_hull(center=[0.0, 0.3, 0.01], half_extents=[0.10, 0.10, 0.01])
    # Toast: smaller, sitting on top of plate
    toast = make_box_hull(center=[0.02, 0.31, 0.04], half_extents=[0.04, 0.04, 0.02])

    # Camera looking down at ~45 degrees from front
    cam2world = make_camera_transform(
        cam_pos=np.array([0.0, -0.3, 0.5]),
        look_at=np.array([0.0, 0.3, 0.0]),
    )

    obj_chs = {"green plate_0": plate, "sliced toast_1": toast}

    selected = compute_object_removal_order(obj_chs=obj_chs, cam2world_tf=cam2world)

    assert selected == "sliced toast_1", f"FAIL: Expected 'sliced toast_1' but got '{selected}'"
    logger.info("PASS: Toast correctly selected before plate\n")
    return True


def test_cup_in_bowl():
    """
    Scenario: Cup sitting inside a deep concave bowl.
    Expected: Cup should be removed first (inside/on-top-of bowl).
    """
    logger.info("=" * 60)
    logger.info("TEST: Cup inside deep bowl (concave surface)")
    logger.info("=" * 60)

    # Bowl: deep concave container (convex hull fills interior)
    bowl = make_bowl_hull(center=[0.0, 0.3, 0.0], radius=0.10, height=0.08)
    # Cup: small cylinder inside the bowl, bottom sitting low inside
    cup = make_box_hull(center=[0.0, 0.3, 0.04], half_extents=[0.03, 0.03, 0.04])

    cam2world = make_camera_transform(
        cam_pos=np.array([0.0, -0.3, 0.5]),
        look_at=np.array([0.0, 0.3, 0.0]),
    )

    obj_chs = {"deep bowl_0": bowl, "small cup_1": cup}

    selected = compute_object_removal_order(obj_chs=obj_chs, cam2world_tf=cam2world)

    assert selected == "small cup_1", f"FAIL: Expected 'small cup_1' but got '{selected}'"
    logger.info("PASS: Cup correctly selected before bowl\n")
    return True


def test_occluded_on_top_vs_non_occluded():
    """
    Scenario: Object A is on top of C but occluded by B. Object B is non-occluded
    and not on top of anything.
    Expected: B should be removed first (non-occluded takes priority over on-top-of).
    """
    logger.info("=" * 60)
    logger.info("TEST: Occlusion priority over support level")
    logger.info("=" * 60)

    # C: base object at back
    obj_c = make_box_hull(center=[0.0, 0.5, 0.02], half_extents=[0.08, 0.08, 0.02])
    # A: on top of C, but behind B from camera's view
    obj_a = make_box_hull(center=[0.0, 0.5, 0.06], half_extents=[0.04, 0.04, 0.02])
    # B: tall wall-like object in front, blocking view of A, not on top of anything
    obj_b = make_box_hull(center=[0.0, 0.3, 0.10], half_extents=[0.10, 0.04, 0.10])

    # Camera at a low angle so B actually blocks the view of A
    cam2world = make_camera_transform(
        cam_pos=np.array([0.0, -0.3, 0.15]),
        look_at=np.array([0.0, 0.4, 0.05]),
    )

    obj_chs = {"obj_A_0": obj_a, "obj_B_1": obj_b, "obj_C_2": obj_c}

    selected = compute_object_removal_order(obj_chs=obj_chs, cam2world_tf=cam2world)

    # B should be selected: it's non-occluded (highest priority)
    assert selected == "obj_B_1", f"FAIL: Expected 'obj_B_1' but got '{selected}'"
    logger.info("PASS: Non-occluded object B correctly selected over occluded-but-on-top A\n")
    return True


def test_foreground_tiebreaker():
    """
    Scenario: Two objects side by side, both non-occluded, neither on top of the other.
    One is closer to the camera.
    Expected: The closer object should be removed first.
    """
    logger.info("=" * 60)
    logger.info("TEST: Foreground tiebreaker")
    logger.info("=" * 60)

    # Object near camera
    obj_near = make_box_hull(center=[0.1, 0.2, 0.03], half_extents=[0.03, 0.03, 0.03])
    # Object far from camera
    obj_far = make_box_hull(center=[-0.1, 0.5, 0.03], half_extents=[0.03, 0.03, 0.03])

    cam2world = make_camera_transform(
        cam_pos=np.array([0.0, -0.3, 0.5]),
        look_at=np.array([0.0, 0.3, 0.0]),
    )

    obj_chs = {"near_obj_0": obj_near, "far_obj_1": obj_far}

    selected = compute_object_removal_order(obj_chs=obj_chs, cam2world_tf=cam2world)

    assert selected == "near_obj_0", f"FAIL: Expected 'near_obj_0' but got '{selected}'"
    logger.info("PASS: Closer object correctly selected as tiebreaker\n")
    return True


def test_stacked_three_objects():
    """
    Scenario: Three objects stacked: A on B on C.
    Expected: A (topmost) removed first.
    """
    logger.info("=" * 60)
    logger.info("TEST: Three-level stack (A on B on C)")
    logger.info("=" * 60)

    obj_c = make_box_hull(center=[0.0, 0.3, 0.02], half_extents=[0.10, 0.10, 0.02])
    obj_b = make_box_hull(center=[0.0, 0.3, 0.06], half_extents=[0.07, 0.07, 0.02])
    obj_a = make_box_hull(center=[0.0, 0.3, 0.10], half_extents=[0.04, 0.04, 0.02])

    cam2world = make_camera_transform(
        cam_pos=np.array([0.0, -0.3, 0.5]),
        look_at=np.array([0.0, 0.3, 0.0]),
    )

    obj_chs = {"top_A_0": obj_a, "mid_B_1": obj_b, "base_C_2": obj_c}

    selected = compute_object_removal_order(obj_chs=obj_chs, cam2world_tf=cam2world)

    assert selected == "top_A_0", f"FAIL: Expected 'top_A_0' but got '{selected}'"
    logger.info("PASS: Topmost object A correctly selected in 3-level stack\n")
    return True


def test_object_in_open_container():
    """
    Scenario: Small object (croissant) inside an open box.
    The box's convex hull fills the open interior, so camera-directed rays from
    the croissant would hit the box's hull. The support-aware occlusion exclusion
    must prevent the croissant from being tagged as occluded by the box.
    Expected: Croissant should be removed first (on top of box, not occluded).
    This is the exact failure case from droid_desk_2.
    """
    logger.info("=" * 60)
    logger.info("TEST: Object in open container (croissant in box)")
    logger.info("=" * 60)

    # Open box: tall container (convex hull fills the open top)
    box = make_box_hull(center=[0.0, 0.3, 0.08], half_extents=[0.12, 0.10, 0.08])
    # Croissant: small object sitting inside the box, near the top
    croissant = make_box_hull(center=[0.0, 0.3, 0.08], half_extents=[0.04, 0.03, 0.02])

    cam2world = make_camera_transform(
        cam_pos=np.array([0.0, -0.4, 0.5]),
        look_at=np.array([0.0, 0.3, 0.05]),
    )

    obj_chs = {"open cardboard box_0": box, "golden-brown croissant_1": croissant}

    selected = compute_object_removal_order(obj_chs=obj_chs, cam2world_tf=cam2world)

    assert selected == "golden-brown croissant_1", \
        f"FAIL: Expected 'golden-brown croissant_1' but got '{selected}'"
    logger.info("PASS: Croissant correctly selected before box\n")
    return True


# ---------------------------------------------------------------------------
# Real data tests
# ---------------------------------------------------------------------------

def test_real_droid_desk_1():
    """
    Test with real droid_desk_1 data: reconstruct the iter_7 scenario
    where toast and plate are both detected. Verify toast is selected first.
    Requires: simfoundry environment with SAM3 model.
    """
    logger.info("=" * 60)
    logger.info("TEST: Real droid_desk_1 data (iter 7: toast vs plate)")
    logger.info("=" * 60)

    data_root = _require_scene_fixture("droid_desk_1", [
        "s1_video/frames_subsampled_15",
        "s2_da/da/exports/npz/results.npz",
        "s4_frame/image_0_cam2world.npy",
        "s4_frame/image_0_pc_raw_rotated.ply",
        "s5_scene/metric_depth/iter_6.npy",
        "s5_scene/post_object_removal/iter_6.png",
    ])

    import cv2
    import open3d as o3d
    from PIL import Image

    s5_dir = os.path.join(data_root, "s5_scene")
    s4_dir = os.path.join(data_root, "s4_frame")
    s2_dir = os.path.join(data_root, "s2_da")

    # Load camera data
    cam2world_tf = np.load(os.path.join(s4_dir, "image_0_cam2world.npy"))

    # Load intrinsics
    results = np.load(os.path.join(s2_dir, "da", "exports", "npz", "results.npz"))
    K = results["intrinsics"][0]

    # Load iter_6 depth (input depth for iter_7)
    current_depth = np.load(os.path.join(s5_dir, "metric_depth", "iter_6.npy"))

    # Load point cloud and reshape
    da_rgb_img = results["image"][0]
    pcd = o3d.io.read_point_cloud(os.path.join(s4_dir, "image_0_pc_raw_rotated.ply"))
    pc = np.asarray(pcd.points).reshape(*da_rgb_img.shape)  # (280, 504, 3)

    # Load the post-iter-6 image (input for iter 7)
    current_rgb = np.array(Image.open(os.path.join(s5_dir, "post_object_removal", "iter_6.png")))
    H, W, _ = current_rgb.shape

    # Compute target ratio and padding (matching the pipeline logic)
    original_img_path = os.path.join(data_root, "s1_video", "frames_subsampled_15")
    original_imgs = sorted([f for f in os.listdir(original_img_path) if f.endswith(".png")])
    original_img = np.array(Image.open(os.path.join(original_img_path, original_imgs[0])))
    original_H, original_W, _ = original_img.shape
    original_ratio = original_W / original_H

    # Gemini image shapes (from the pipeline)
    IMAGE_SHAPES = [(1024, 1024), (896, 1152), (1152, 896), (768, 1408), (1408, 768)]
    ratio_options = {w / h: (w, h) for (w, h) in IMAGE_SHAPES}
    ratio_diffs = np.abs([original_ratio - r for r in ratio_options.keys()])
    closest_ratio = list(ratio_options.keys())[ratio_diffs.argmin()]
    resolution = ratio_options[closest_ratio]
    target_W, target_H = resolution
    target_ratio = target_W / target_H

    from simfoundry.utils.processing_utils import (
        pad_image_to_ratio, unpad_image, erode_mask, compute_point_cloud_from_depth,
        denoise_obj_point_cloud_v2,
    )

    # Pad and compute sizes (matching pipeline)
    padding_color = (255, 255, 255)
    padded_pc, (delta_w, delta_h) = pad_image_to_ratio(
        pc, target_ratio=target_ratio, padding_color=padding_color, return_padding_size=True,
    )
    padded_H, padded_W, _ = padded_pc.shape

    # Compute current point cloud from depth
    current_pc = compute_point_cloud_from_depth(
        depth=current_depth, K=K, world_to_cam_tf=cam2world_tf,
    )

    # Use SAM3 to segment the two objects of interest
    logger.info("Loading SAM3 model...")
    from simfoundry.models.sam_v3 import SAM3

    sam3 = SAM3(confidence_threshold=0.4, device="cuda", video=False)
    pil_img = Image.fromarray(current_rgb)

    target_phrases = ["sliced toast", "green plate"]
    obj_chs = {}

    for phrase in target_phrases:
        logger.info(f"Segmenting '{phrase}'...")
        masks, boxes_xyxy, logits = sam3.predict_segmentation(pil_img=pil_img, text_prompt=phrase)
        if len(masks) == 0:
            logger.warning(f"No mask found for '{phrase}', skipping")
            continue

        # Take the best mask
        best_idx = np.argmax(logits)
        mask = masks[best_idx].squeeze(axis=0)

        # Build convex hull (matching pipeline logic)
        mask_eroded_255 = erode_mask((mask * 255).astype(np.uint8), kernel_size=3)
        mask_eroded_resized_255 = cv2.resize(mask_eroded_255, (padded_W, padded_H))
        mask_eroded_resized_unpadded = unpad_image(mask_eroded_resized_255, delta_w, delta_h).astype(bool)

        obj_pc = current_pc[mask_eroded_resized_unpadded]
        obj_rgb = cv2.resize(current_rgb, (padded_W, padded_H))
        obj_rgb = unpad_image(obj_rgb, delta_w, delta_h)[mask_eroded_resized_unpadded] / 255

        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(obj_pc)
        obj_pcd.colors = o3d.utility.Vector3dVector(obj_rgb)

        obj_pcd_pruned, indices = denoise_obj_point_cloud_v2(
            pcd_obj=obj_pcd,
            visualize_process=False,
            visualize_result=False,
            target_density_cm2=5.0,
            outlier_removal_n_neighbors=None,
            outlier_removal_std_ratio=1.0,
            dbscan_eps=0.005,
            dbscan_min_pts=None,
            cluster_proportion_threshold=0.06,
        )

        obj_pc_clean = np.asarray(obj_pcd_pruned.points)
        obj_ch = trimesh.convex.convex_hull(obj_pc_clean)
        obj_ch = trimesh.convex.convex_hull(
            np.asarray(obj_ch.vertices) - 0.0025 * np.asarray(obj_ch.vertex_normals),
        )
        unique_id = f"{phrase}_{target_phrases.index(phrase)}"
        obj_chs[unique_id] = obj_ch
        logger.info(f"  Hull bounds: Z=[{obj_ch.bounds[0][2]:.4f}, {obj_ch.bounds[1][2]:.4f}], "
                     f"centroid={obj_ch.centroid}")

    if len(obj_chs) < 2:
        logger.error("Could not segment both toast and plate. Test inconclusive.")
        return False

    # Run the new ordering
    selected = compute_object_removal_order(obj_chs=obj_chs, cam2world_tf=cam2world_tf)

    expected = "sliced toast_0"
    if selected == expected:
        logger.info(f"PASS: Toast correctly selected before plate (selected: {selected})\n")
        return True
    else:
        logger.error(f"FAIL: Expected '{expected}' but got '{selected}'\n")
        return False


def _build_convex_hulls_from_scene(scene_name, input_iter, target_phrases):
    """
    Helper to load scene data and build convex hulls for specified object phrases.
    Returns (obj_chs, cam2world_tf) or raises on failure.
    """
    data_root = _require_scene_fixture(scene_name, [
        "s1_video/frames_subsampled_15",
        "s2_da/da/exports/npz/results.npz",
        "s4_frame/image_0_cam2world.npy",
        "s4_frame/image_0_pc_raw_rotated.ply",
        f"s5_scene/metric_depth/iter_{input_iter}.npy",
        f"s5_scene/post_object_removal/iter_{input_iter}.png",
    ])

    import cv2
    import open3d as o3d
    from PIL import Image
    from simfoundry.utils.processing_utils import (
        pad_image_to_ratio, unpad_image, erode_mask, compute_point_cloud_from_depth,
        denoise_obj_point_cloud_v2,
    )
    from simfoundry.models.sam_v3 import SAM3

    s5_dir = os.path.join(data_root, "s5_scene")
    s4_dir = os.path.join(data_root, "s4_frame")
    s2_dir = os.path.join(data_root, "s2_da")

    cam2world_tf = np.load(os.path.join(s4_dir, "image_0_cam2world.npy"))
    results = np.load(os.path.join(s2_dir, "da", "exports", "npz", "results.npz"))
    K = results["intrinsics"][0]
    current_depth = np.load(os.path.join(s5_dir, "metric_depth", f"iter_{input_iter}.npy"))
    da_rgb_img = results["image"][0]
    pcd = o3d.io.read_point_cloud(os.path.join(s4_dir, "image_0_pc_raw_rotated.ply"))
    pc = np.asarray(pcd.points).reshape(*da_rgb_img.shape)

    current_rgb = np.array(Image.open(
        os.path.join(s5_dir, "post_object_removal", f"iter_{input_iter}.png")))

    # Compute target ratio / padding (matching pipeline logic)
    original_img_path = os.path.join(data_root, "s1_video", "frames_subsampled_15")
    original_imgs = sorted([f for f in os.listdir(original_img_path) if f.endswith(".png")])
    original_img = np.array(Image.open(os.path.join(original_img_path, original_imgs[0])))
    original_H, original_W, _ = original_img.shape
    original_ratio = original_W / original_H

    IMAGE_SHAPES = [(1024, 1024), (896, 1152), (1152, 896), (768, 1408), (1408, 768)]
    ratio_options = {w / h: (w, h) for (w, h) in IMAGE_SHAPES}
    ratio_diffs = np.abs([original_ratio - r for r in ratio_options.keys()])
    closest_ratio = list(ratio_options.keys())[ratio_diffs.argmin()]
    resolution = ratio_options[closest_ratio]
    target_W, target_H = resolution
    target_ratio = target_W / target_H

    padding_color = (255, 255, 255)
    padded_pc, (delta_w, delta_h) = pad_image_to_ratio(
        pc, target_ratio=target_ratio, padding_color=padding_color, return_padding_size=True,
    )
    padded_H, padded_W, _ = padded_pc.shape
    current_pc = compute_point_cloud_from_depth(depth=current_depth, K=K, world_to_cam_tf=cam2world_tf)

    logger.info("Loading SAM3 model...")
    sam3 = SAM3(confidence_threshold=0.4, device="cuda", video=False)
    pil_img = Image.fromarray(current_rgb)

    obj_chs = {}
    for phrase in target_phrases:
        logger.info(f"Segmenting '{phrase}'...")
        masks, boxes_xyxy, logits = sam3.predict_segmentation(pil_img=pil_img, text_prompt=phrase)
        if len(masks) == 0:
            logger.warning(f"No mask found for '{phrase}', skipping")
            continue

        best_idx = np.argmax(logits)
        mask = masks[best_idx].squeeze(axis=0)
        mask_eroded_255 = erode_mask((mask * 255).astype(np.uint8), kernel_size=3)
        mask_eroded_resized_255 = cv2.resize(mask_eroded_255, (padded_W, padded_H))
        mask_eroded_resized_unpadded = unpad_image(mask_eroded_resized_255, delta_w, delta_h).astype(bool)

        obj_pc = current_pc[mask_eroded_resized_unpadded]
        obj_rgb = cv2.resize(current_rgb, (padded_W, padded_H))
        obj_rgb = unpad_image(obj_rgb, delta_w, delta_h)[mask_eroded_resized_unpadded] / 255

        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(obj_pc)
        obj_pcd.colors = o3d.utility.Vector3dVector(obj_rgb)

        obj_pcd_pruned, _ = denoise_obj_point_cloud_v2(
            pcd_obj=obj_pcd, visualize_process=False, visualize_result=False,
            target_density_cm2=5.0, outlier_removal_n_neighbors=None,
            outlier_removal_std_ratio=1.0, dbscan_eps=0.005, dbscan_min_pts=None,
            cluster_proportion_threshold=0.06,
        )

        obj_pc_clean = np.asarray(obj_pcd_pruned.points)
        obj_ch = trimesh.convex.convex_hull(obj_pc_clean)
        obj_ch = trimesh.convex.convex_hull(
            np.asarray(obj_ch.vertices) - 0.0025 * np.asarray(obj_ch.vertex_normals),
        )
        unique_id = f"{phrase}_{target_phrases.index(phrase)}"
        obj_chs[unique_id] = obj_ch
        logger.info(f"  Hull bounds: Z=[{obj_ch.bounds[0][2]:.4f}, {obj_ch.bounds[1][2]:.4f}], "
                     f"centroid={obj_ch.centroid}")

    return obj_chs, cam2world_tf


def test_real_droid_desk_2():
    """
    Test with real droid_desk_2 data: reconstruct the iter_3 scenario
    where croissant and open cardboard box are both detected.
    Verify croissant is selected first (it sits inside the open box).
    Requires: simfoundry environment with SAM3 model.
    """
    logger.info("=" * 60)
    logger.info("TEST: Real droid_desk_2 data (iter 3: croissant vs open box)")
    logger.info("=" * 60)

    target_phrases = ["golden-brown croissant", "open cardboard box"]
    obj_chs, cam2world_tf = _build_convex_hulls_from_scene(
        scene_name="droid_desk_2",
        input_iter=2,  # iter_2 output is the input for iter_3
        target_phrases=target_phrases,
    )

    if len(obj_chs) < 2:
        logger.error("Could not segment both croissant and box. Test inconclusive.")
        return False

    selected = compute_object_removal_order(obj_chs=obj_chs, cam2world_tf=cam2world_tf)

    expected = "golden-brown croissant_0"
    if selected == expected:
        logger.info(f"PASS: Croissant correctly selected before box (selected: {selected})\n")
        return True
    else:
        logger.error(f"FAIL: Expected '{expected}' but got '{selected}'\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test object removal ordering heuristics")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Only run synthetic tests (no GPU/SAM3 needed)")
    args = parser.parse_args()

    results = {}
    synthetic_tests = [
        ("toast_on_plate", test_toast_on_plate),
        ("cup_in_bowl", test_cup_in_bowl),
        ("object_in_open_container", test_object_in_open_container),
        ("occlusion_priority", test_occluded_on_top_vs_non_occluded),
        ("foreground_tiebreaker", test_foreground_tiebreaker),
        ("three_level_stack", test_stacked_three_objects),
    ]

    for name, test_fn in synthetic_tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            logger.error(f"FAIL: {name} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    if not args.synthetic_only:
        real_tests = [
            ("real_droid_desk_1", test_real_droid_desk_1),
            ("real_droid_desk_2", test_real_droid_desk_2),
        ]
        for name, test_fn in real_tests:
            try:
                results[name] = test_fn()
            except Exception as e:
                logger.error(f"FAIL: {name} raised exception: {e}")
                import traceback
                traceback.print_exc()
                results[name] = False

    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {status}: {name}")
        if not passed:
            all_pass = False

    if all_pass:
        logger.info(f"\nAll {len(results)} tests passed!")
    else:
        n_fail = sum(1 for v in results.values() if not v)
        logger.info(f"\n{n_fail}/{len(results)} tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
