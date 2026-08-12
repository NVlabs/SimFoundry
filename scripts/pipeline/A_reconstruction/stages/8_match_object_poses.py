# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from simfoundry or from foundationpose if using foundationpose

Requires installing:

- FoundationPose, see https://github.com/NVlabs/FoundationPose
"""
from urllib.parse import uses_relative

import numpy as np
from matplotlib import colormaps
import open3d as o3d
import os
import sys
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation as R
import copy
from copy import deepcopy
import trimesh
import tempfile
import shutil
from probreg import cpd, gmmtree, filterreg, bcpd, l2dist_regs, transformation
import cv2
import json
import hydra
from simfoundry.utils.faiss_utils import l2_search
from simfoundry.utils.processing_utils import compute_point_cloud_from_depth, pad_image_to_ratio, unpad_image, \
    dilate_mask, erode_mask, extract_numbers_from_str, denoise_obj_point_cloud
from simfoundry.utils.prompt_utils import prompt_topk_image_select
from simfoundry.pipeline.stage_utils import StageResult, bootstrap_hydra_workdir, finalize_stage
from simfoundry.pipeline.frame_selection import resolve_img_idx
from simfoundry.utils.python_utils import atomic_output_path
import multiprocessing
from tqdm import trange
import logging
import os

logger = logging.getLogger(__name__)
# To import FoundationPose
from simfoundry import REPO_DIR
foundationpose_dir = f"{REPO_DIR}/deps/FoundationPose"
sys.path.append(foundationpose_dir)


# see https://github.com/facebookresearch/hydra/issues/2949#issue-2516892001
if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

from simfoundry import CFG_DIR

bootstrap_hydra_workdir(__file__)


tf_y_up = transformation.RigidTransformation(rot=R.from_euler("xyz", np.array([-np.pi / 2, 0, 0])).as_matrix())

import numpy as np


def resolve_requested_indices(cfg):
    """Resolve optional per-object filter for streaming/partial reruns."""
    raw = cfg.s8_pose.get("object_indices", None)
    if raw is None:
        return None
    if isinstance(raw, int):
        return {raw}
    return {int(v) for v in raw}


def quaternion_distance(q1, q2):
    """Compute angular distance between quaternions"""
    # Handle double cover: q and -q represent same rotation
    dot_product = np.abs(np.dot(q1, q2))
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return 2 * np.arccos(dot_product)


def quaternion_mean(quaternions, weights=None):
    """Compute mean quaternion using averaging in R⁴ followed by normalization"""
    if weights is None:
        weights = np.ones(len(quaternions)) / len(quaternions)

    # Ensure all quaternions have positive scalar part (to handle double cover)
    aligned_quats = []
    for q in quaternions:
        if q[0] < 0:  # Assuming scalar-first convention
            aligned_quats.append(-q)
        else:
            aligned_quats.append(q)

    # Weighted average
    mean_q = np.sum([w * q for w, q in zip(weights, aligned_quats)], axis=0)

    # Normalize
    return mean_q / np.linalg.norm(mean_q)


def quaternion_kmeans(quaternions, k, max_iters=100):
    n = len(quaternions)

    # Initialize centers randomly
    indices = np.random.choice(n, k, replace=False)
    centers = [quaternions[i].copy() for i in indices]

    for iteration in range(max_iters):
        # Assignment step
        clusters = [[] for _ in range(k)]
        for i, q in enumerate(quaternions):
            distances = [quaternion_distance(q, c) for c in centers]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(i)

        # Update step
        new_centers = []
        for i, cluster in enumerate(clusters):
            if cluster:
                cluster_quats = [quaternions[idx] for idx in cluster]
                new_centers.append(quaternion_mean(cluster_quats))
            else:
                new_centers.append(centers[i])  # Keep old center if empty

        # Check convergence
        if all(np.allclose(new_centers[i], centers[i]) for i in range(k)):
            break

        centers = new_centers

    # Prune any empty entries
    for i in reversed(range(k)):
        if len(clusters[i]) == 0:
            del centers[i]
            del clusters[i]

    return centers, clusters


def _sample_fits(source_pts, target_pts, ori_mats, process_id):
    info = []
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_pts)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_pts)
    for i in trange(len(ori_mats), desc=f"Pre-tokenizing Thread {process_id}"):
        ori_mat = ori_mats[i]
        tf = transformation.RigidTransformation(rot=ori_mat)
        # print(f"testing ori #{j}...")
        # Run CPD
        source_copy = copy.deepcopy(source)
        source_copy.points = tf.transform(source_copy.points)
        target_copy = copy.deepcopy(target)

        # print(f"Running CPD...")
        for i in range(1):
            # print(f"Iter {i}...")
            # tf_param, _, _ = filterreg.registration_filterreg(source, target, feature_fn=probreg.features.FPFH())
            tf_param, _, _ = cpd.registration_cpd(
                source_copy,
                target_copy,
                tf_type_name="rigid",
                use_color=False,
                # tol=0.0001,
                # maxiter=200,
                use_cuda=False,
                # update_scale=True,
            )  # , w=0.05, maxiter=200, tol=0.0001)
            # tf_param = bcpd.registration_bcpd(source_copy, target_copy)
            result = copy.deepcopy(source_copy)
            result.points = tf_param.transform(result.points)
            source_copy = result
            # tf = tf_param
            tf = tf_param * tf

        # Compute chamfer distance
        # print(f"Computing chamfer distance...")
        target_points = np.array(target_copy.points)
        result_points = np.array(result.points)
        dists, idxs = l2_search(target_points, result_points, 1)
        dist = np.sum(dists)
        # dist = np.sum(result.compute_point_cloud_distance(target_copy))
        info.append({"result": result, "dist": dist, "tf_z_up": tf, "tf_y_up": tf * tf_y_up})

    # Sort the values
    info_sorted = sorted(info, key=lambda x: x["dist"])
    top_info = info_sorted[0]

    return top_info


def sample_fits_multiprocess(source, target, n_samples=10, n_threads=8):
    # Transform to offset automatic rotation from +Y-up / +Z-forward to +Z-up / -Y-forward
    ori_mats = R.random(n_samples).as_matrix()
    ori_mats_batches = np.array_split(ori_mats, n_threads)

    with multiprocessing.Pool(processes=n_threads) as pool:
        pid = 0
        results = []
        for ori_mat_batch in ori_mats_batches:
            results.append(pool.apply_async(_sample_fits, args=(np.asarray(source.points), np.asarray(target.points), ori_mat_batch, pid)))
            pid += 1

        pool.close()

        # Loop until all threads are completed, aggregating results along the way
        info = []
        while len(results) > 0:
            idx = 0
            while idx < len(results):
                if results[idx].ready():
                    # Dynamically synthesize results
                    new_info = results.pop(idx).get()
                    info.append(new_info)
                else:
                    idx += 1

        # Sort and return
        info_sorted = sorted(info, key=lambda x: x["dist"])
        top_info = info_sorted[0]

    return top_info


def sample_fits(source, target, n_samples=10, update_scale=True, return_inverse=False):
    # Transform to offset automatic rotation from +Y-up / +Z-forward to +Z-up / -Y-forward
    ori_mats = R.random(n_samples).as_matrix()

    # Each entry is dict with keys {"result", "dist"}
    info = []
    for j, ori_mat in enumerate(ori_mats):
        tf = transformation.RigidTransformation(rot=ori_mat)
        logger.info(f"testing ori #{j}...")
        # Run CPD
        source_copy = copy.deepcopy(source)
        source_copy.points = tf.transform(source_copy.points)
        target_copy = copy.deepcopy(target)

        logger.info(f"Running CPD...")
        for i in range(1):
            logger.info(f"Iter {i}...")
            # tf_param, _, _ = filterreg.registration_filterreg(source, target, feature_fn=probreg.features.FPFH())
            tf_param, _, _ = cpd.registration_cpd(
                source_copy,
                target_copy,
                tf_type_name="rigid",
                use_color=False,
                # tol=0.0001,
                # maxiter=200,
                use_cuda=False,
                update_scale=update_scale,
            )  # , w=0.05, maxiter=200, tol=0.0001)
            # tf_param = bcpd.registration_bcpd(source_copy, target_copy)
            result = copy.deepcopy(source_copy)
            result.points = tf_param.transform(result.points)
            source_copy = result
            # tf = tf_param
            tf = tf_param * tf

        # Compute chamfer distance
        logger.info(f"Computing chamfer distance...")
        target_points = np.array(target_copy.points)
        result_points = np.array(result.points)
        dists, idxs = l2_search(target_points, result_points, 1)
        dist = np.sum(dists)
        # dist = np.sum(result.compute_point_cloud_distance(target_copy))
        tf_z = tf
        tf_y = tf * tf_y_up
        if return_inverse:
            tf_z = tf_z.inverse()
            tf_y = tf_y.inverse()
        info.append({"result": result, "dist": dist, "tf_z_up": tf_z, "tf_y_up": tf_y})

    # Sort the values
    info_sorted = sorted(info, key=lambda x: x["dist"])
    top_info = info_sorted[0]

    return info_sorted


# Minimum number of finite source points required to attempt pose matching.
# Open3D's OBB needs ≥4; CPD needs enough for a meaningful registration.
_MIN_SOURCE_POINTS = 10


def _write_object_failure(out_dir: str, img_name: str, reason: str, **counts) -> None:
    """Write a failure record to the per-object info JSON and log a warning."""
    info = {"error": "pose_matching_skipped", "reason": reason, **counts}
    with atomic_output_path(f"{out_dir}/info/{img_name}.json") as _tmp_info, open(_tmp_info, "w") as f:
        json.dump(info, f, indent=4)
    logger.warning("Skipping pose matching for %s: %s", img_name, reason)


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    source_dir = cfg.s2_da.out_dir
    scene_dir = cfg.s5_scene.out_dir
    mesh_dir = f"{cfg.s7_mesh.out_dir}/textured_mesh/{cfg.s7_mesh.texture_model}"
    out_dir = cfg.s8_pose.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    dirs_to_create = ["pc", "canonical_mesh", "fit", "info"]
    for dir_name in dirs_to_create:
        Path(f"{out_dir}/{dir_name}").mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("Starting object pose matching pipeline...")
    logger.info(f"Output directory: {out_dir}")
    logger.info("="*60)

    # Store hyperparam data
    source_voxel_scale_factor = cfg.s8_pose.source_voxel_scale_factor
    # voxel_size = cfg.s8_pose.voxel_size

    # Load source files
    source_padded_resized = np.array(Image.open(f"{scene_dir}/source_padded_resized.png"))
    resolution = (source_padded_resized.shape[1], source_padded_resized.shape[0])

    img_idx = resolve_img_idx(cfg)
    
    # Check if using FoundationStereo or Depth Anything
    use_fs = cfg.s3_ground.get("use_fs", False)
    
    if use_fs:
        logger.info("Using FoundationStereo (FS) output")
        fs_dir = cfg.s2_fs.out_dir
        rgb_da = np.array(Image.open(f"{fs_dir}/image_{img_idx}_rgb.png"))
        K = np.load(f"{fs_dir}/image_{img_idx}_K.npy")
    else:
        logger.info("Using Depth Anything (DA) output")
        results = np.load(f"{source_dir}/da/exports/npz/results.npz")
        rgb_da = results["image"][img_idx]
        K = results["intrinsics"][img_idx]
    
    # K_fpath = f"{source_dir}/{cfg.s3_ground.img_name}_K.npy"
    # K = np.load(K_fpath)
    # resolution = Imagen3.RESOLUTIONS[cfg.s5_scene.ratio]
    target_ratio = resolution[0] / resolution[1]
    # scene_img_name = cfg.s5_scene.img_name
    # source_disp_fpath = f"{source_dir}/{scene_img_name}_disp.png"
    # source_disp = np.array(Image.open(source_disp_fpath))
    padded_rgb_da, (delta_w, delta_h) = pad_image_to_ratio(rgb_da, target_ratio=target_ratio, padding_color=(0, 0, 0), return_padding_size=True)
    H, W, _ = padded_rgb_da.shape

    # # Load cam2world tf
    # cam2world_tf_fpath = f"{cfg.s4_frame.out_dir}/image_{img_idx}_cam2world.npy"
    # cam2world_tf = np.load(cam2world_tf_fpath)

    prev_index = 0
    requested_indices = resolve_requested_indices(cfg)
    if requested_indices is not None:
        logger.info("Filtering to requested object indices: %s", sorted(requested_indices))

    # Helper function to extract iteration number for proper numerical sorting
    def get_iter_num(filename):
        """Extract iteration number from filename for numerical sorting"""
        if "iter" not in filename or not filename.endswith(".glb"):
            return float('inf')  # Non-iter files go to the end
        try:
            return int(filename.split("iter_")[1].split("_")[0])
        except (IndexError, ValueError):
            return float('inf')

    # Iterate over all meshes, and match their pose (sorted numerically by iteration)
    for filename in sorted(os.listdir(mesh_dir), key=get_iter_num):
        if not filename.endswith(".glb"):
            continue
        img_name = filename.split("_mesh.")[0]
        if "iter" not in filename or "untextured" in filename:
            continue
        idx = int(img_name.split("iter_")[-1])
        if requested_indices is not None and idx not in requested_indices:
            continue

        # # TODO: Remove!!
        # if idx != 12:
        #     continue

        mesh_path = f"{mesh_dir}/{filename}"
        prev_index = idx - 1
        depth_filename = "original_depth.npy" if idx == 0 else f"metric_depth/iter_{prev_index}.npy"
        depth_path = f"{scene_dir}/{depth_filename}"
        rgb_filename = "source_padded_resized.png" if idx == 0 else f"post_object_removal/iter_{prev_index}.png"
        rgb_path = f"{scene_dir}/{rgb_filename}"
        mask_path = f"{scene_dir}/removal_mask/iter_{idx}.png"

        # Extract point cloud from masked depth
        rgb = np.array(Image.open(rgb_path))
        depth = np.load(depth_path)
        mask = np.array(Image.open(mask_path))
        # The mask / rgb is based on the padded / resized version, so we need to reverse the process to get the original
        # dimensions as the depth map
        rgb_resized = cv2.resize(rgb, (W, H))
        rgb_resized_unpadded = unpad_image(rgb_resized, delta_w, delta_h)
        mask_resized = cv2.resize(mask, (W, H))
        mask_resized_unpadded = unpad_image(mask_resized, delta_w, delta_h)
        # Erode mask slightly to remove noise at boundary edges
        mask_resized_unpadded_eroded = erode_mask(mask_resized_unpadded, kernel_size=3).astype(bool)
        # Image.fromarray((mask_resized_unpadded_eroded * 255).astype(np.uint8)).show()

        # Compose object point cloud
        pc = compute_point_cloud_from_depth(depth=depth, K=K).reshape(-1, 3) #, world_to_cam_tf=cam2world_tf).reshape(-1, 3)
        obj_pc = pc[mask_resized_unpadded_eroded.flatten()]
        obj_rgb = (rgb_resized_unpadded.reshape(-1, 3) / 255)[mask_resized_unpadded_eroded.flatten()]
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(obj_pc)
        obj_pcd.colors = o3d.utility.Vector3dVector(obj_rgb)

        if len(obj_pc) == 0:
            _write_object_failure(
                out_dir, img_name,
                "Depth map produced 0 points in the masked region — "
                "depth estimation likely failed for this object (too small or specular surface).",
                mask_pixels=int(mask_resized_unpadded_eroded.sum()),
            )
            continue

        # Prune outlier / noisy values
        # TODO: Can try either (a) o3d denoise or (b) depth map gradient-based pruning
        obj_pcd, indices = denoise_obj_point_cloud(pcd_obj=obj_pcd)

        obj_pc_path = f"{out_dir}/pc/{img_name}.ply"
        o3d.io.write_point_cloud(obj_pc_path, obj_pcd)

        # Read source (real world) partial point cloud
        # source_pts = np.load(source_fpath)
        # source = o3d.geometry.PointCloud()
        # source.points = o3d.utility.Vector3dVector(source_pts) # * np.array([1.0, 1.0, 1.0]) * SOURCE_VOXEL_SCALE_FACTOR)
        source = o3d.io.read_point_cloud(obj_pc_path)
        source = source.remove_non_finite_points()
        # o3d.visualization.draw_geometries([source])

        n_source_pts = len(source.points)
        if n_source_pts < _MIN_SOURCE_POINTS:
            _write_object_failure(
                out_dir, img_name,
                f"Only {n_source_pts} finite point(s) remain after denoising — "
                f"too few for pose matching (need ≥ {_MIN_SOURCE_POINTS}). "
                "Depth values in the masked region may be unreliable (small or specular object).",
                mask_pixels=int(mask_resized_unpadded_eroded.sum()),
                source_points_after_denoise=n_source_pts,
            )
            continue

        # Grab oriented bounding box so we know how to (partially) normalize the target mesh initially with the same
        # rough voxel density
        source_obb_diag = np.linalg.norm(source.get_oriented_bounding_box().extent)

        # Read target mesh, convert to point cloud
        mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
        target = mesh.sample_points_poisson_disk(number_of_points=10000)
        target = target.remove_non_finite_points()
        # o3d.visualization.draw_geometries([target])

        # Re-scale target mesh down to roughly the scale of the source based on ratio of diagonals
        target_obb_diag = np.linalg.norm(target.get_oriented_bounding_box().extent)
        pre_scale_factor = source_obb_diag / target_obb_diag
        target = target.scale(pre_scale_factor, center=np.zeros(3))

        # Unify sampling
        voxel_size = source_obb_diag / 30
        source = source.voxel_down_sample(voxel_size=voxel_size) # * source_voxel_scale_factor) #voxel_size * source_voxel_scale_factor)
        target = target.voxel_down_sample(voxel_size=voxel_size) #voxel_size * source_voxel_scale_factor)
        target = target.farthest_point_down_sample(num_samples=min(len(source.points), len(target.points)))

        # TODO: Try keeping source to be denser?

        # # TODO: Fix hacky swapping
        # source_tmp = target
        # target = source
        # source = source_tmp

        # Sample fits
        # We return the inverse because sample fits computes the tf from source (partial PC) -> target (mesh PC), but
        # we want the inverse operation (mesh fit to partial PC)
        
        # First, run initial sample_fits at the base pre_scale_factor
        logger.info(f"Running initial sample_fits at base pre_scale_factor={pre_scale_factor:.4f}")
        info_unscaled_sorted = sample_fits(source, target, cfg.s8_pose.n_samples, update_scale=False, return_inverse=True)
        info_scaled_sorted = sample_fits(source, target, cfg.s8_pose.n_samples, update_scale=True, return_inverse=True)
        # Prune any scaled values that are degenerate -- this is any computed TF with scale < 0.1
        # We expect this to be degenerate because we already pre-scale the raw mesh to a "reasonable" order of magnitude
        # based on the real-world partial point cloud...so it shouldn't be more than +/- 1 order of magnitude
        info_scaled_sorted = [info for info in info_scaled_sorted if (info["tf_z_up"].scale > 0.1 and info["tf_z_up"].scale < 10.0)]
        
        # Add pre_scale_factor metadata to each info entry for the base scale (multiplier=1.0)
        for info in info_unscaled_sorted + info_scaled_sorted:
            info["pre_scale_factor"] = pre_scale_factor
            info["scale_multiplier"] = 1.0
        
        all_info = info_unscaled_sorted + info_scaled_sorted
        
        # Now run sample_fits for different scale multipliers (+/- 0.05 increments, up to +/- 0.2)
        # This gives multipliers: 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2 (excluding 1.0 which we already did)
        scale_multipliers = np.arange(0.8, 1.25, 0.05)
        scale_multipliers = scale_multipliers[np.abs(scale_multipliers - 1.0) > 0.01]  # Exclude 1.0 (already computed)
        
        logger.info(f"Running scale grid search with multipliers: {scale_multipliers}")
        
        # for scale_mult in scale_multipliers:
        #     # Compute the new pre_scale_factor for this multiplier
        #     new_pre_scale_factor = pre_scale_factor * scale_mult
            
        #     # Re-read and re-scale the target mesh with the new scale factor
        #     target_scaled = mesh.sample_points_poisson_disk(number_of_points=10000)
        #     target_scaled = target_scaled.remove_non_finite_points()
        #     target_scaled = target_scaled.scale(new_pre_scale_factor, center=np.zeros(3))
            
        #     # Unify sampling (same as before)
        #     target_scaled = target_scaled.voxel_down_sample(voxel_size=voxel_size)
        #     target_scaled = target_scaled.farthest_point_down_sample(num_samples=min(len(source.points), len(target_scaled.points)))
            
        #     logger.info(f"  Running sample_fits at scale_multiplier={scale_mult:.2f} (pre_scale_factor={new_pre_scale_factor:.4f})")
            
        #     # Run sample_fits with update_scale=False for this scaled target
        #     info_at_scale = sample_fits(source, target_scaled, cfg.s8_pose.n_samples, update_scale=False, return_inverse=True)
            
        #     # Add metadata to track which scale was used
        #     for info in info_at_scale:
        #         info["pre_scale_factor"] = new_pre_scale_factor
        #         info["scale_multiplier"] = scale_mult
            
        #     all_info.extend(info_at_scale)
        
        # Sort all results by distance and select the best
        info_sorted = sorted(all_info, key=lambda x: x["dist"])
        top_info = info_sorted[0]
        
        # Extract the final pre_scale_factor from the best result
        final_pre_scale_factor = top_info["pre_scale_factor"]
        final_scale_multiplier = top_info["scale_multiplier"]
        logger.info(f"Best fit found at scale_multiplier={final_scale_multiplier:.2f} "
                   f"(final_pre_scale_factor={final_pre_scale_factor:.4f}, dist={top_info['dist']:.6f})")

        # Update mesh -- this is pre-scaled, pre-transform (canonical object frame) -- save this
        canonical_mesh_fpath = f"{out_dir}/canonical_mesh/{img_name}.glb"

        # Scale and save via trimesh using the FINAL pre_scale_factor (best from grid search)
        canonical_mesh_tm = trimesh.load(mesh_path)
        canonical_mesh_tm.apply_scale(final_pre_scale_factor)
        canonical_mesh_tm.export(canonical_mesh_fpath)

        # Load canonical mesh in open3d
        canonical_mesh = o3d.io.read_triangle_mesh(canonical_mesh_fpath, enable_post_processing=True)

        # # shutil.copy2(src=mesh_path, dst=canonical_mesh_fpath)
        # canonical_mesh = deepcopy(mesh)
        # canonical_mesh = canonical_mesh.scale(pre_scale_factor, center=np.zeros(3))
        # o3d.io.write_triangle_mesh(canonical_mesh_fpath, canonical_mesh)

        if cfg.visualize:
            # draw result
            for top_info in info_sorted:
                result = top_info["result"]
                source.paint_uniform_color([1, 0, 0])
                target.paint_uniform_color([0, 1, 0])
                result.paint_uniform_color([0, 0, 1])
                result_check = copy.deepcopy(target)
                pts = np.asarray(result_check.points)
                pts = top_info["tf_z_up"].scale * np.dot(pts, top_info["tf_z_up"].rot.T) + top_info["tf_z_up"].t
                result_check.points = o3d.utility.Vector3dVector(pts)
                # result_check.points = top_info["tf"].transform(result_check.points)
                result_check.paint_uniform_color([0, 1, 1])
                # o3d.visualization.draw_geometries([mesh, source, target, result, result_check])
                # o3d.visualization.draw_geometries([source, target, result, result_check])
                print(f"SCALE: {top_info['tf_z_up'].scale}")
                o3d.visualization.draw_geometries([source, target, result_check])
        
                # from IPython import embed; embed()

        # Potentially load FoundationPose if requested
        use_foundationpose = cfg.s8_pose.use_foundationpose
        if use_foundationpose:
            import nvdiffrast.torch as dr
            from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
            from Utils import set_logging_format, set_seed

            set_logging_format()
            set_seed(0)

            # Export updated mesh to load into trimesh and run FoundationPose
            with tempfile.TemporaryDirectory() as tmpdir:
                canonical_mesh_scaled = deepcopy(canonical_mesh)
                vertices = np.asarray(canonical_mesh_scaled.vertices)
                vertices = top_info["tf_z_up"].scale * vertices
                canonical_mesh_scaled.vertices = o3d.utility.Vector3dVector(vertices)

                tmp_obj_path = f"{tmpdir}/{img_name}_canonical_mesh.obj"
                o3d.io.write_triangle_mesh(tmp_obj_path, canonical_mesh_scaled)
                canonical_mesh_scaled_tm = trimesh.load(tmp_obj_path)

                # Initialize FoundationPose components
                scorer = ScorePredictor()
                refiner = PoseRefinePredictor()
                glctx = dr.RasterizeCudaContext()
                
                debug_level = 3 if cfg.visualize else 1
                debug_dir = f"{out_dir}/debug/{img_name}"
                os.makedirs(debug_dir, exist_ok=True)
                
                est = FoundationPose(
                    model_pts=canonical_mesh_scaled_tm.vertices,
                    model_normals=canonical_mesh_scaled_tm.vertex_normals,
                    mesh=canonical_mesh_scaled_tm,
                    scorer=scorer,
                    refiner=refiner,
                    debug_dir=debug_dir,
                    debug=debug_level,
                    glctx=glctx,
                )
                logger.info("FoundationPose estimator initialization done")

                # Register pose using FoundationPose
                est_refine_iter = getattr(cfg.s8_pose, "est_refine_iter", 5)
                foundationpose_tf = est.register(
                    K=K.astype(float), #float),
                    rgb=rgb_resized_unpadded,
                    depth=depth,
                    ob_mask=mask_resized_unpadded.astype(bool),
                    iteration=est_refine_iter,
                )

                # Save debug visualization if enabled
                if cfg.visualize and debug_level >= 2:
                    # Export transformed mesh for visualization
                    m = canonical_mesh_scaled_tm.copy()
                    m.apply_transform(foundationpose_tf)
                    m.export(f'{debug_dir}/model_tf.obj')
                    
                    # Compute bounding box for visualization
                    to_origin, extents = trimesh.bounds.oriented_bounds(canonical_mesh_scaled_tm)
                    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
                    
                    # Draw posed 3D box on the image
                    from Utils import draw_posed_3d_box, draw_xyz_axis
                    center_pose = foundationpose_tf @ np.linalg.inv(to_origin)
                    vis = draw_posed_3d_box(K, img=rgb_resized_unpadded.copy(), ob_in_cam=center_pose, bbox=bbox)
                    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
                    
                    # Save visualization
                    Image.fromarray(vis).save(f"{out_dir}/fit/{img_name}_foundationpose_fit.png")

                # Update canonical mesh
                est.mesh_ori.export(canonical_mesh_fpath)
                canonical_mesh = o3d.io.read_triangle_mesh(canonical_mesh_fpath, enable_post_processing=True)

                # Update transform
                foundationpose_tf_rt = transformation.RigidTransformation(
                    scale=np.ones(3), 
                    rot=foundationpose_tf[:3, :3], 
                    t=foundationpose_tf[:3, 3]
                )
                top_info["tf_y_up"] = foundationpose_tf_rt * tf_y_up
                top_info["tf_z_up"] = foundationpose_tf_rt

        # Update mesh
        mesh_updated = deepcopy(canonical_mesh)
        vertices = np.asarray(mesh_updated.vertices)
        vertices = top_info["tf_z_up"].scale * vertices @ top_info["tf_z_up"].rot.T + top_info["tf_z_up"].t
        mesh_updated.vertices = o3d.utility.Vector3dVector(vertices)

        if cfg.visualize:
            # # draw result
            # result = top_info["result"]
            # source.paint_uniform_color([1, 0, 0])
            # target.paint_uniform_color([0, 1, 0])
            # result.paint_uniform_color([0, 0, 1])
            # result_check = copy.deepcopy(source)
            # pts = np.asarray(result_check.points)
            # pts = top_info["tf_z_up"].scale * np.dot(pts, top_info["tf_z_up"].rot.T) + top_info["tf_z_up"].t
            # result_check.points = o3d.utility.Vector3dVector(pts)
            # # result_check.points = top_info["tf"].transform(result_check.points)
            # result_check.paint_uniform_color([0, 1, 1])
            # o3d.visualization.draw_geometries([mesh, source, target, result, result_check])

            # Visualize overlay with scene
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(pc.reshape(-1, 3))
            scene_pcd.colors = o3d.utility.Vector3dVector(rgb_resized_unpadded.reshape(-1, 3) / 255)
            o3d.visualization.draw_geometries([scene_pcd, mesh_updated])

        # Store result
        out_info = {
            "z_up": {
                "trans": top_info["tf_z_up"].t.tolist(),
                "rot": top_info["tf_z_up"].rot.tolist(),
                "scale": np.asarray(top_info["tf_z_up"].scale).tolist(),
            },
            "y_up": {
                "trans": top_info["tf_y_up"].t.tolist(),
                "rot": top_info["tf_y_up"].rot.tolist(),
                "scale": np.asarray(top_info["tf_y_up"].scale).tolist(),
            },
            # Total scale = pre_scale_factor * tf_scale
            "pre_scale_factor": float(pre_scale_factor),
        }

        with atomic_output_path(f"{out_dir}/info/{img_name}.json") as _tmp_info, open(_tmp_info, "w") as f:
            json.dump(out_info, f, indent=4)

    logger.info("="*60)
    logger.info("Object pose matching complete!")
    logger.info("="*60)
    
    finalize_stage(
        stage_cfg=cfg.s8_pose,
        out_dir=cfg.s8_pose.out_dir,
        result=StageResult(success=True),
    )


if __name__ == "__main__":
    main()
