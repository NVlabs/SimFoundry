# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from `simfoundry` env

Requires installing:

- simfoundry, see the main README
"""
from simfoundry.models.sam_v3 import SAM3
from simfoundry.utils.processing_utils import compute_point_cloud_from_depth, annotate
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image
import json
import hydra
import os
from copy import deepcopy
from simfoundry.pipeline.stage_utils import StageResult, bootstrap_hydra_workdir, finalize_stage
from simfoundry.pipeline.frame_selection import (
    is_auto_img_idx,
    select_canonical_frame,
    write_selection,
)
from simfoundry import CFG_DIR
import logging
import einops


logger = logging.getLogger(__name__)

bootstrap_hydra_workdir(__file__)


def predict_floor_masks(rgb_fpath, sam3, floor_categories, floor_threshold, visualize=False):
    pil_img = Image.open(rgb_fpath)
    np_img = np.array(pil_img)
    for floor_category in floor_categories:
        masks, boxes_xyxy, logits = sam3.predict_segmentation(pil_img=pil_img, text_prompt=floor_category)
        n_masks = len(masks)
        if n_masks > 0:
            if logits.max() < floor_threshold:
                continue
            if visualize:
                annotated_img = annotate(np_img, boxes_xyxy, [floor_category] * n_masks, logits)[:,:,::-1]
                Image.fromarray(annotated_img).show()
            break

    return masks, boxes_xyxy, logits, floor_category


def predict_masks_with_llm(rgb_fpath, cfg, sam3):
    from simfoundry.models.vlm import Gemini
    from simfoundry.utils.prompt_utils import prompt_floor_setlist
    vlm = Gemini(
        project=cfg.gcloud_project,
        location="global",
        model=cfg.s3_ground.detection_model,
    )
    result = vlm(
        prompt=prompt_floor_setlist(),
        image_paths=str(rgb_fpath),
        temperature=0,
        top_p=0,
        seed=0,
        print_results=True,
    )
    floor_categories = vlm.get_result_text(result=result).split("\n")[-1].strip("[]").replace("'", "").replace('"', "").replace(".", "").split(", ")

    masks, boxes_xyxy, logits, phrase = predict_floor_masks(rgb_fpath, sam3, floor_categories, cfg.s3_ground.floor_threshold, cfg.visualize)

    if len(masks) == 0:
        print(f"Found no masks for floor categories: {floor_categories} with threshold: {cfg.s3_ground.floor_threshold}.")
        return masks, boxes_xyxy, logits, phrase

    return masks, boxes_xyxy, logits, phrase


def predict_masks_interactive(rgb_fpath, sam3):
    """
    Interactive fallback: user clicks on the floor in a matplotlib window,
    SAM3 segments based on point prompts, and the result is displayed.
    The user can refine by clicking additional points. Press Enter to accept,
    or 'r' to reset all points and start over.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("TkAgg")

    pil_img = Image.open(rgb_fpath)
    np_img = np.array(pil_img)

    point_coords = []
    point_labels = []  # 1=foreground, 0=background
    current_masks = None
    current_boxes = None
    current_scores = None

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_title("Left-click: foreground | Right-click: background | Enter: accept | R: reset")
    img_display = ax.imshow(np_img)
    ax.axis("off")
    scatter_fg = ax.scatter([], [], c="lime", marker="o", s=80, edgecolors="white", linewidths=1.5, zorder=5)
    scatter_bg = ax.scatter([], [], c="red", marker="x", s=80, linewidths=2, zorder=5)
    accepted = [False]

    def _update_overlay():
        overlay = np_img.copy()
        if current_masks is not None and len(current_masks) > 0:
            best_idx = np.argmax(current_scores)
            mask = current_masks[best_idx][0].astype(bool)  # (H, W)
            color = np.array([0, 255, 0], dtype=np.uint8)
            overlay[mask] = (overlay[mask] * 0.5 + color * 0.5).astype(np.uint8)
        img_display.set_data(overlay)

        fg_pts = [(x, y) for (x, y), l in zip(point_coords, point_labels) if l == 1]
        bg_pts = [(x, y) for (x, y), l in zip(point_coords, point_labels) if l == 0]
        scatter_fg.set_offsets(fg_pts if fg_pts else np.empty((0, 2)))
        scatter_bg.set_offsets(bg_pts if bg_pts else np.empty((0, 2)))
        fig.canvas.draw_idle()

    def _run_prediction():
        nonlocal current_masks, current_boxes, current_scores
        if not point_coords:
            return
        masks, boxes, scores = sam3.predict_segmentation_with_point(
            pil_img, point_coords, point_labels, multimask_output=True,
        )
        current_masks = masks
        current_boxes = boxes
        current_scores = scores
        _update_overlay()

    def on_click(event):
        if event.inaxes != ax or accepted[0]:
            return
        x, y = int(event.xdata), int(event.ydata)
        label = 1 if event.button == 1 else 0  # left=fg, right=bg
        point_coords.append((x, y))
        point_labels.append(label)
        print(f"  Point ({x}, {y}) label={'fg' if label == 1 else 'bg'} — running SAM3...")
        _run_prediction()

    def on_key(event):
        nonlocal current_masks, current_boxes, current_scores
        if event.key == "enter":
            if current_masks is not None and len(current_masks) > 0:
                accepted[0] = True
                plt.close(fig)
            else:
                print("  No mask to accept yet — click on the floor first.")
        elif event.key == "r":
            point_coords.clear()
            point_labels.clear()
            current_masks = None
            current_boxes = None
            current_scores = None
            _update_overlay()
            print("  Reset all points.")

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if not accepted[0] or current_masks is None or len(current_masks) == 0:
        raise RuntimeError("Interactive segmentation was cancelled or produced no mask.")

    best_idx = np.argmax(current_scores)
    masks = current_masks[best_idx:best_idx + 1].astype(bool)
    boxes = current_boxes[best_idx:best_idx + 1]
    scores = current_scores[best_idx:best_idx + 1]

    return masks, boxes, scores, "floor (interactive)"


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    floor_categories = cfg.s3_ground.floor_categories
    out_dir = cfg.s3_ground.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    use_interactive = cfg.s3_ground.get("use_interactive_segmentation", False)

    # Create sam3. Built before the frame is picked because automatic selection segments the
    # support surface in every candidate to score it.
    sam3 = SAM3(
        confidence_threshold=0.50,
        device="cuda",
        video=False,
        enable_inst_interactivity=use_interactive,
    )

    # Every stage from here to 13 reconstructs the scene from this one frame, so a bad choice
    # (blurry, shot from far away, objects occluding each other) caps the quality of all of
    # them. `img_idx: auto` scores the candidates and commits the winner to disk for the
    # downstream stages; an explicit integer still pins the frame.
    if is_auto_img_idx(cfg.s3_ground.img_idx):
        selection = select_canonical_frame(cfg, sam3)
        img_idx = selection.selected_idx
        logger.info("Wrote frame selection to %s", write_selection(cfg, selection))
    else:
        img_idx = int(cfg.s3_ground.img_idx)
        logger.info("Using the frame pinned by s3_ground.img_idx: %s", img_idx)

    # Check if we should use FoundationStereo or Depth Anything output
    use_fs = cfg.s3_ground.get("use_fs", False)

    if use_fs:
        logger.info("Using FoundationStereo output")
        fs_dir = cfg.s2_fs.out_dir
        
        # Load FS outputs
        rgb = np.load(f"{fs_dir}/image_{img_idx}_rgb.npy")
        depth = np.load(f"{fs_dir}/image_{img_idx}_depth_meter.npy")
        K = np.load(f"{fs_dir}/image_{img_idx}_K.npy")
    else:
        logger.info("Using Depth Anything output")
        pc_dir = cfg.s2_da.out_dir
        
        # Load DA outputs
        results = np.load(f"{pc_dir}/da/exports/npz/results.npz")
        rgb = results["image"][img_idx]
        depth = results["depth"][img_idx]
        K = results["intrinsics"][img_idx]

    # Copy raw RGB image
    rgb_fpath = f"{out_dir}/raw_img.png"
    Image.fromarray(rgb).save(rgb_fpath)

    # Infer floor possibilities
    masks, boxes_xyxy, logits, phrase = predict_floor_masks(rgb_fpath, sam3, floor_categories, cfg.s3_ground.floor_threshold, cfg.visualize)

    if len(masks) == 0:
        print(f"Found no masks for floor categories: {floor_categories} with threshold: {cfg.s3_ground.floor_threshold}.")
        if use_interactive:
            print("Falling back to interactive point-click segmentation.")
            masks, boxes_xyxy, logits, phrase = predict_masks_interactive(rgb_fpath, sam3)
        else:
            print("Trying with LLM categories")
            masks, boxes_xyxy, logits, phrase = predict_masks_with_llm(rgb_fpath, cfg, sam3)

    if len(masks) == 0:
        if use_interactive:
            print("Trying with LLM categories")
            masks, boxes_xyxy, logits, phrase = predict_masks_with_llm(rgb_fpath, cfg, sam3)
        else:
            print("LLM categories also failed. Falling back to interactive point-click segmentation.")
            masks, boxes_xyxy, logits, phrase = predict_masks_interactive(rgb_fpath, sam3)

    n_masks = len(masks)
    assert n_masks > 0, f"Expected at least one bounding box amongst floor categories: {floor_categories}, but found none!"

    # Annotate images with masks
    np_img = np.array(Image.open(rgb_fpath))
    annotated_img = annotate(np_img, boxes_xyxy, [phrase] * n_masks, logits)[:,:,::-1]
    annotated_img_fpath = f"{out_dir}/annotated_img.png"
    Image.fromarray(annotated_img).save(annotated_img_fpath)

    # For each object, samv2 returns 3 masks
    # We merge all three as the object mask
    if cfg.visualize:
        Image.fromarray(annotate(np_img, boxes_xyxy, [phrase] * n_masks, logits)[:,:,::-1]).show()
    if n_masks != 1:
        logger.warning(f"Expected a single mask for category: {phrase}, but got: {n_masks} -- selecting the largest mask")
        mask_sums = einops.reduce(masks, "N 1 H W -> N", "sum")
        mask_idx = np.argmax(mask_sums)
        floor_mask = masks[mask_idx][0]     # (H, W)
    else:
        floor_mask = masks[0][0]            # (H, W)

    # Only keep points belonging to the floor masks
    pts = compute_point_cloud_from_depth(depth, K).reshape(-1, 3)[floor_mask.flatten()]
    rgbs = rgb.reshape(-1, 3)[floor_mask.flatten()] / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    rgbs[inliers] = np.array([1.0, 0, 0]).reshape(1, 3)
    pcd.colors = o3d.utility.Vector3dVector(rgbs)
    [a, b, c, d] = plane_model
    z_dir_plane = np.array([a, b, c])
    # print(f"Estimated floor plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    inlier_cloud = pcd.select_by_index(inliers)
    pc_floor = np.asarray(inlier_cloud.points)
    pc_floor_mean = np.mean(pc_floor, axis=0)
    # origin_pos = pc_floor[int(len(pc_floor) // 2)] + pc_floor_mean

    # Two possible normals -- pointing up or down from the plane
    # Because the image z axis points into the frame and towards the floor, we expect the native camera Z vector to have
    # a negative dot product with the computed z direction. If positive, flip
    assert np.isclose(np.linalg.norm(z_dir_plane), 1.0)
    dot_product = np.dot(z_dir_plane, np.array([0, 0, 1.0]))
    if dot_product > 0.0:
        print("Found inverted floor z plane normal, un-inverting so that it points upwards")
        z_dir_plane *= -1

    print(f"Estimated z-direction computed from floor point cloud: {z_dir_plane}")

    # Visualize
    start_point = np.mean(pc_floor, axis=0)
    vector = z_dir_plane / np.linalg.norm(z_dir_plane)  # Normalize
    arrow_length = np.linalg.norm(vector)  # Length of the arrow is the magnitude of the vector
    arrow_radius = 0.1  # Radius of the cylinder and cone
    arrow_cone_radius = 0.2  # Radius of the cone at the arrowhead
    arrow_cone_height = 0.5  # Height of the arrowhead

    # The arrow is initially aligned with the Z-axis. We need to rotate it to align with the vector.
    # Compute the rotation between the Z-axis and the vector
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, vector)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rotation_angle = np.arccos(np.dot(z_axis, vector))
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)

    # Create the arrow and apply the rotation and translation
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=arrow_radius, cone_radius=arrow_cone_radius,
                                                   cylinder_height=arrow_length - arrow_cone_height,
                                                   cone_height=arrow_cone_height)
    arrow.rotate(rotation_matrix, center=(0, 0, 0))
    arrow.translate(start_point)

    # Convert the arrow mesh to a point cloud if you want it to have a similar appearance to the original point cloud
    # You might skip this conversion if you prefer the mesh appearance
    arrow_pcd = arrow.sample_points_poisson_disk(number_of_points=1000)

    if cfg.visualize:
        o3d.visualization.draw_geometries([pcd, arrow_pcd], point_show_normal=True)

    # Make sure there is minimal roll in the angle
    assert abs(z_dir_plane[0]) < cfg.s3_ground.floor_tilt_threshold, f"got tilted floor: {z_dir_plane[0]}"

    # Project pc_floor_mean point onto z_plane to get final origin pos
    plane_point = np.array([-d / a, 0, 0])
    plane_to_floor_mean = pc_floor_mean - plane_point
    proj_dist = np.dot(plane_to_floor_mean, z_dir_plane)
    origin_pos = pc_floor_mean - proj_dist * z_dir_plane

    # Store result
    out_info = {
        "floor_category": phrase,
        "origin": origin_pos.tolist(),
        "z_dir": z_dir_plane.tolist(),
    }

    with open(f"{out_dir}/image_{img_idx}_floor_info.json", "w") as f:
        json.dump(out_info, f, indent=4)
    
    finalize_stage(
        stage_cfg=cfg.s3_ground,
        out_dir=cfg.s3_ground.out_dir,
        result=StageResult(success=True, additional_info={"resolved_img_idx": int(img_idx)}),
    )


if __name__ == "__main__":
    main()
