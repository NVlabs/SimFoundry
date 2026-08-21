# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from simfoundry

Requires installing:

- simfoundry, see the main README
"""
import numpy as np
import open3d as o3d
import os
from pathlib import Path
import json
from scipy.spatial.transform import Rotation as R
import hydra
from simfoundry.pipeline.stage_utils import StageResult, bootstrap_hydra_workdir, finalize_stage
from simfoundry.pipeline.frame_selection import resolve_img_idx
import logging
import os
# see https://github.com/facebookresearch/hydra/issues/2949#issue-2516892001
if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

from simfoundry import CFG_DIR

logger = logging.getLogger(__name__)
bootstrap_hydra_workdir(__file__)


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    ground_img_idx = resolve_img_idx(cfg)
    pc_dir = cfg.s4_frame.out_dir
    pc_fpath = f"{pc_dir}/image_{ground_img_idx}_pc_raw_rotated.ply"
    mesh_dir = cfg.s8_pose.out_dir + "/canonical_mesh"
    
    # Check if we should use interactive poses
    use_interactive_pose = cfg.s10_compile.get("use_interactive_pose", False)
    interactive_suffix = cfg.s10_compile.get("interactive_suffix", "")
    
    if use_interactive_pose:
        pose_dir = cfg.s8_pose.out_dir + f"/info_interactive{interactive_suffix}"
        logger.info(f"Using INTERACTIVE poses from: {pose_dir}")
        if not os.path.exists(pose_dir):
            logger.error(f"Interactive pose directory not found: {pose_dir}")
            logger.error("Available interactive directories:")
            base_dir = cfg.s8_pose.out_dir
            for d in sorted(os.listdir(base_dir)):
                if d.startswith("info_interactive"):
                    logger.error(f"  - {d}")
            raise FileNotFoundError(f"Interactive pose directory not found: {pose_dir}")
    else:
        pose_dir = cfg.s8_pose.out_dir + "/info"
        logger.info(f"Using ORIGINAL poses from: {pose_dir}")
    
    out_dir = cfg.s10_compile.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("Starting scene compilation pipeline...")
    logger.info(f"Output directory: {out_dir}")
    logger.info("="*60)

    # Load cam2world tf
    cam2world_tf_fpath = f"{cfg.s4_frame.out_dir}/image_{ground_img_idx}_cam2world.npy"
    cam2world_tf = np.load(cam2world_tf_fpath)

    # Load original point cloud
    scene_pcd = o3d.io.read_point_cloud(pc_fpath)
    obj_meshes = dict()

    # Helper function to extract iteration number for proper numerical sorting
    def get_iter_num(filename):
        """Extract iteration number from filename for numerical sorting"""
        if "iter" not in filename or not filename.endswith(".glb"):
            return float('inf')
        try:
            return int(filename.split("iter_")[1].split("_")[0])
        except (IndexError, ValueError):
            return float('inf')

    # Iterate over all meshes, and transform them accordingly (sorted numerically by iteration)
    for filename in sorted(os.listdir(mesh_dir), key=get_iter_num):
        if not filename.endswith(".glb"):
            continue
        if "iter" not in filename:
            continue
        img_name = filename.split(".")[0]
        idx = int(img_name.split("iter_")[-1])
        mesh_path = f"{mesh_dir}/{filename}"
        obj = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
        # if idx == 4:
        #     from IPython import embed; embed()
        with open(f"{pose_dir}/{img_name}.json", "r") as f:
            pose_info = json.load(f)["z_up"]
            vertices = np.asarray(obj.vertices)
            # Convert from obj canonical frame -> scene pose in cam frame
            vertices = pose_info["scale"] * vertices @ np.array(pose_info["rot"]).T + np.array(pose_info["trans"])
            # Convert from cam frame -> world frame
            vertices = vertices @ cam2world_tf[:3, :3].T + cam2world_tf[:3, 3].reshape(1, 3)
            obj.vertices = o3d.utility.Vector3dVector(vertices)
            # obj_tf = np.eye(4)
            # obj_tf[:3, :3] = np.array(pose_info["rot"])
            # obj_tf[:3, 3] = np.array(pose_info["trans"])
            # obj = obj.transform(obj_tf)
            # obj = obj.scale(pose_info["scale"], center=obj.get_center())
            obj_meshes[img_name] = obj

    # Visualize
    geometries = [scene_pcd, *list(obj_meshes.values())]
    if cfg.visualize:
        o3d.visualization.draw_geometries(geometries)
    
    logger.info("="*60)
    logger.info("Scene compilation complete!")
    logger.info("="*60)

    finalize_stage(
        stage_cfg=cfg.s10_compile,
        out_dir=cfg.s10_compile.out_dir,
        result=StageResult(success=True),
    )


if __name__ == "__main__":
    main()
