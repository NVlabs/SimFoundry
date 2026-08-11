# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from `simfoundry` env

Requires installing:

- simfoundry, see the main README
"""
import os.path

"""
Unifies the world frame -- the ground truth point cloud is natively in the camera frame (Z points into the scene),
but we want Z to point up from the ground plane to be consistent with gravity in physics simulation
So we take the convention to be the new world X as pointing in the direction that is the projection of the camera's
native Z axis onto the ground plane. This is equivalent to some single rotation about the camera frame's X axis
"""
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import json
from pathlib import Path
import hydra
from simfoundry.utils.processing_utils import compute_point_cloud_from_depth
from simfoundry.pipeline.stage_utils import StageResult, bootstrap_hydra_workdir, finalize_stage
from simfoundry.pipeline.frame_selection import resolve_img_idx
import logging
logger = logging.getLogger(__name__)

# see https://github.com/facebookresearch/hydra/issues/2949#issue-2516892001
if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

from simfoundry import CFG_DIR

bootstrap_hydra_workdir(__file__)


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    planes_dir = cfg.s3_ground.out_dir
    img_idx = resolve_img_idx(cfg)
    floor_info_fpath = f"{planes_dir}/image_{img_idx}_floor_info.json"
    out_dir = cfg.s4_frame.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pc_fpath = f"{out_dir}/image_{img_idx}_pc_raw.ply"
    pc_pruned_fpath = f"{out_dir}/image_{img_idx}_pc_pruned.ply"
    pc_denoised_fpath = f"{out_dir}/image_{img_idx}_pc_denoised.ply"
    out_pc_fpath = f"{out_dir}/image_{img_idx}_pc_raw_rotated.ply"

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
    
    # Construct pruned point cloud and then transform it
    pc = compute_point_cloud_from_depth(depth=depth, K=K).reshape(-1, 3)
    keep_mask = (pc[:, 2] > 0) & (pc[:, 2] <= cfg.s4_frame.z_far)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255.0)
    o3d.io.write_point_cloud(pc_fpath, pcd)

    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
    o3d.io.write_point_cloud(pc_pruned_fpath, pcd)
    logger.info(f"PCL saved to {out_dir}")

    logging.info("Denoising point cloud...")
    cl, ind = pcd.remove_radius_outlier(nb_points=cfg.s4_frame.denoise_nb_points, radius=cfg.s4_frame.denoise_radius)
    inlier_cloud = pcd.select_by_index(ind)
    o3d.io.write_point_cloud(pc_denoised_fpath, inlier_cloud)
    pcd = inlier_cloud

    # Rotate the point cloud based on the z direction such that z points upwards and x points in the direction of the
    #   original camera z-axis projected onto the computed ground floor plane
    with open(floor_info_fpath, "r") as f:
        floor_info = json.load(f)
    z_dir = floor_info["z_dir"]

    # Compensate for roll angle as well
    roll_angle = np.arctan2(z_dir[0], np.linalg.norm(z_dir[1:]))        # angle between X and YZ vector
    roll_mat = R.from_euler("xyz", np.array([0, 0, -roll_angle])).as_matrix()
    tilt_angle = np.arctan2(z_dir[1], z_dir[2])
    tilt_mat = R.from_euler("xyz", np.array([tilt_angle, 0, 0])).as_matrix()
    rot_mat = tilt_mat @ roll_mat
    pcd = o3d.io.read_point_cloud(pc_fpath)
    # Translation needs to happen FIRST because it's in the camera frame
    pcd.points = o3d.utility.Vector3dVector((np.asarray(pcd.points) - np.array(floor_info["origin"]).reshape(1, 3)) @ rot_mat.T)
    if cfg.visualize:
        o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(out_pc_fpath, pcd)

    # Also save the cam2world transform for transforming points / objects
    # (Negative) translation needs to happen FIRST because it's in the camera frame -- so we need to compile tf matrix
    # as a composition of two matrices: R @ T, NOT T @ R which would be the normal [[R T] [0 1]] block matrix
    rot_tf = np.eye(4)
    rot_tf[:3, :3] = rot_mat
    trans_tf = np.eye(4)
    trans_tf[:3, 3] = -np.array(floor_info["origin"])
    cam2world_tf = rot_tf @ trans_tf
    cam2world_tf_fpath = f"{out_dir}/image_{img_idx}_cam2world.npy"
    np.save(cam2world_tf_fpath, cam2world_tf)

    finalize_stage(
        stage_cfg=cfg.s4_frame,
        out_dir=cfg.s4_frame.out_dir,
        result=StageResult(success=True),
    )


if __name__ == "__main__":
    main()
