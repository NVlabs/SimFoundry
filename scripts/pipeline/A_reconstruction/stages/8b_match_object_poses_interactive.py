# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Interactive version of 8_match_object_poses.py

Should be run from simfoundry or from any6d if using any6d

Enables users to interactively move, scale, and rotate objects in the visualization window.
Controls:
- Arrow keys / WASD: Translate object
- Q/E: Rotate around Z-axis (roll)
- R/F: Rotate around Y-axis (yaw)  
- T/G: Rotate around X-axis (pitch)
- Z/X: Scale up/down
- +/-: Fine scale adjustment
- S: Save current pose
- R: Reset to initial pose
- N: Next object
- ESC/Q: Quit

Requires installing:
- Any6D, see https://github.com/taeyeopl/Any6D
"""
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
from omegaconf import OmegaConf
from simfoundry.models.vlm import Gemini # , Imagen3
from simfoundry.utils.faiss_utils import l2_search
from simfoundry.utils.processing_utils import compute_point_cloud_from_depth, pad_image_to_ratio, unpad_image, \
    dilate_mask, erode_mask, extract_numbers_from_str, denoise_obj_point_cloud, dump_json
from simfoundry.utils.prompt_utils import prompt_topk_image_select
from simfoundry.pipeline.stage_utils import resolve_base_iteration
import multiprocessing
from tqdm import trange
import logging
import os

logger = logging.getLogger(__name__)
# TO import Any6D
from simfoundry import REPO_DIR
any6d_dir = f"{REPO_DIR}/deps/Any6D"
sys.path.append(any6d_dir)


# see https://github.com/facebookresearch/hydra/issues/2949#issue-2516892001
if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

from simfoundry import CFG_DIR
from simfoundry.pipeline.frame_selection import resolve_img_idx

### At the start of every script, we cd into the scripts/config directory
scripts_dir = os.path.dirname(os.path.abspath(__file__))
cfg_dir = CFG_DIR
os.chdir(cfg_dir)


tf_y_up = transformation.RigidTransformation(rot=R.from_euler("xyz", np.array([-np.pi / 2, 0, 0])).as_matrix())

import numpy as np


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


class InteractivePoseEditor:
    """Interactive pose editor using Open3D visualization"""
    
    @staticmethod
    def _copy_scale(scale):
        """Copy scale (can be scalar or array)"""
        if isinstance(scale, np.ndarray):
            return scale.copy()
        elif hasattr(scale, '__len__') and not isinstance(scale, str):
            return np.array(scale).copy()
        else:
            return scale  # Scalar, return as-is
    
    def __init__(self, scene_pcd, canonical_mesh, initial_tf_z_up, initial_tf_y_up, img_name, out_dir, interactive_suffix=""):
        self.scene_pcd = scene_pcd
        self.canonical_mesh = canonical_mesh
        # Manually copy transformation (can't use deepcopy on RigidTransformation)
        self.initial_tf_z_up = transformation.RigidTransformation(
            scale=self._copy_scale(initial_tf_z_up.scale),
            rot=initial_tf_z_up.rot.copy(),
            t=initial_tf_z_up.t.copy()
        )
        self.initial_tf_y_up = transformation.RigidTransformation(
            scale=self._copy_scale(initial_tf_y_up.scale),
            rot=initial_tf_y_up.rot.copy(),
            t=initial_tf_y_up.t.copy()
        )
        self.current_tf_z_up = transformation.RigidTransformation(
            scale=self._copy_scale(initial_tf_z_up.scale),
            rot=initial_tf_z_up.rot.copy(),
            t=initial_tf_z_up.t.copy()
        )
        self.current_tf_y_up = transformation.RigidTransformation(
            scale=self._copy_scale(initial_tf_y_up.scale),
            rot=initial_tf_y_up.rot.copy(),
            t=initial_tf_y_up.t.copy()
        )
        self.img_name = img_name
        self.out_dir = out_dir
        self.interactive_suffix = interactive_suffix
        
        # Transformation parameters
        self.translation_step = 0.002  # meters (reduced for finer control)
        self.rotation_step = np.pi / 90  # 2 degrees (reduced for finer control)
        self.scale_step = 0.05
        
        # Create transformed mesh (use Open3D's copy method)
        self.mesh_updated = o3d.geometry.TriangleMesh(canonical_mesh)
        self.update_mesh()
        
        # Visualization state
        self.vis = None
        self.saved = False
        
        # Point cloud visibility and density controls
        self.pcd_visible = True
        self.pcd_density = 1.0  # 1.0 = full density, lower = sparser
        self.pcd_density_step = 0.05
        self.original_pcd_points = np.asarray(scene_pcd.points).copy()
        self.original_pcd_colors = np.asarray(scene_pcd.colors).copy() if scene_pcd.has_colors() else None
        
    def update_mesh(self):
        """Update mesh vertices based on current transformation"""
        vertices = np.asarray(self.canonical_mesh.vertices)
        # Apply transformation: scale, rotate, translate
        # Scale can be scalar or 3D array - use uniform scaling
        scale_val = self.current_tf_z_up.scale
        if hasattr(scale_val, '__len__') and not isinstance(scale_val, str):
            # If it's an array, use first element (assuming uniform scale)
            scale_val = scale_val[0] if len(scale_val) > 0 else 1.0
        vertices = scale_val * vertices @ self.current_tf_z_up.rot.T + self.current_tf_z_up.t
        self.mesh_updated.vertices = o3d.utility.Vector3dVector(vertices)
        self.mesh_updated.compute_vertex_normals()
        
    def toggle_pcd_visibility(self, vis):
        """Toggle point cloud visibility"""
        self.pcd_visible = not self.pcd_visible
        if self.pcd_visible:
            # Restore point cloud with current density
            self.update_pcd_density(vis)
            logger.info("Point cloud: VISIBLE")
        else:
            # Hide point cloud by setting empty points
            self.scene_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            self.scene_pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            vis.update_geometry(self.scene_pcd)
            logger.info("Point cloud: HIDDEN")
    
    def update_pcd_density(self, vis):
        """Update point cloud density"""
        if not self.pcd_visible:
            return
            
        n_total = len(self.original_pcd_points)
        n_sample = max(1, int(n_total * self.pcd_density))
        
        if self.pcd_density >= 1.0:
            # Full density
            self.scene_pcd.points = o3d.utility.Vector3dVector(self.original_pcd_points)
            if self.original_pcd_colors is not None:
                self.scene_pcd.colors = o3d.utility.Vector3dVector(self.original_pcd_colors)
        else:
            # Subsample
            indices = np.random.choice(n_total, n_sample, replace=False)
            self.scene_pcd.points = o3d.utility.Vector3dVector(self.original_pcd_points[indices])
            if self.original_pcd_colors is not None:
                self.scene_pcd.colors = o3d.utility.Vector3dVector(self.original_pcd_colors[indices])
        
        vis.update_geometry(self.scene_pcd)
        # logger.info(f"Point cloud density: {self.pcd_density*100:.0f}% ({n_sample}/{n_total} points)")
    
    def change_pcd_density(self, vis, increase=True):
        """Change point cloud density"""
        if increase:
            self.pcd_density = min(1.0, self.pcd_density + self.pcd_density_step)
        else:
            self.pcd_density = max(0.1, self.pcd_density - self.pcd_density_step)
        self.update_pcd_density(vis)
    
    def change_translation_step(self, increase=True):
        """Change translation step size"""
        if increase:
            self.translation_step *= 2.0
        else:
            self.translation_step /= 2.0
        # Clamp to reasonable range (0.0001m to 0.1m)
        self.translation_step = max(0.0001, min(0.1, self.translation_step))
        logger.info(f"Translation step: {self.translation_step*1000:.2f} mm")
    
    def translate(self, dx, dy, dz):
        """Translate object"""
        self.current_tf_z_up.t += np.array([dx, dy, dz])
        self.update_mesh()
        
    def rotate(self, axis, angle):
        """Rotate object around axis in world frame"""
        if axis == 'x':
            rot_mat = R.from_euler('x', angle).as_matrix()
        elif axis == 'y':
            rot_mat = R.from_euler('y', angle).as_matrix()
        elif axis == 'z':
            rot_mat = R.from_euler('z', angle).as_matrix()
        else:
            return
            
        # Apply rotation relative to current rotation (compose rotations)
        # Rotate in world frame: R_new = R_delta @ R_current
        new_rot = rot_mat @ self.current_tf_z_up.rot
        self.current_tf_z_up.rot = new_rot
        self.update_mesh()
        
    def scale(self, factor):
        """Scale object uniformly"""
        scale_val = self.current_tf_z_up.scale
        if hasattr(scale_val, '__len__') and not isinstance(scale_val, str):
            # If it's an array, scale all elements uniformly
            self.current_tf_z_up.scale = scale_val * factor
        else:
            # If it's a scalar
            self.current_tf_z_up.scale = scale_val * factor
        self.update_mesh()
        
    def reset(self):
        """Reset to initial pose"""
        # Manually copy transformation components
        self.current_tf_z_up = transformation.RigidTransformation(
            scale=self._copy_scale(self.initial_tf_z_up.scale),
            rot=self.initial_tf_z_up.rot.copy(),
            t=self.initial_tf_z_up.t.copy()
        )
        self.current_tf_y_up = transformation.RigidTransformation(
            scale=self._copy_scale(self.initial_tf_y_up.scale),
            rot=self.initial_tf_y_up.rot.copy(),
            t=self.initial_tf_y_up.t.copy()
        )
        self.update_mesh()
        
    def save(self):
        """Save current pose"""
        # Update y_up transform
        self.current_tf_y_up = self.current_tf_z_up * tf_y_up
        
        # Handle scale format (can be scalar or array) - convert to list format
        scale_z = self.current_tf_z_up.scale
        if hasattr(scale_z, 'tolist'):
            scale_z = scale_z.tolist()
        elif not isinstance(scale_z, (list, tuple, np.ndarray)):
            # If scalar, convert to 3D array
            scale_z = [float(scale_z), float(scale_z), float(scale_z)]
        else:
            scale_z = list(scale_z)
            
        scale_y = self.current_tf_y_up.scale
        if hasattr(scale_y, 'tolist'):
            scale_y = scale_y.tolist()
        elif not isinstance(scale_y, (list, tuple, np.ndarray)):
            # If scalar, convert to 3D array
            scale_y = [float(scale_y), float(scale_y), float(scale_y)]
        else:
            scale_y = list(scale_y)
        
        out_info = {
            "z_up": {
                "trans": self.current_tf_z_up.t.tolist(),
                "rot": self.current_tf_z_up.rot.tolist(),
                "scale": scale_z,
            },
            "y_up": {
                "trans": self.current_tf_y_up.t.tolist(),
                "rot": self.current_tf_y_up.rot.tolist(),
                "scale": scale_y,
            },
        }
        
        # Save to interactive output directory with suffix
        info_dir = f"{self.out_dir}/info_interactive{self.interactive_suffix}"
        Path(info_dir).mkdir(parents=True, exist_ok=True)
        info_path = f"{info_dir}/{self.img_name}.json"
        with open(info_path, "w") as f:
            json.dump(out_info, f, indent=4)
        
        logger.info(f"Saved interactive pose to {info_path}")
        self.saved = True
        
    def key_callback(self, vis, key, action):
        """Handle keyboard input - single callback for all keys"""
        if action == 0:  # KeyEvent.DOWN
            # Translation
            if key == ord('W') or key == 265:  # Up arrow
                self.translate(0, 0, self.translation_step)
                vis.update_geometry(self.mesh_updated)
            elif key == ord('S') or key == 264:  # Down arrow
                self.translate(0, 0, -self.translation_step)
                vis.update_geometry(self.mesh_updated)
            elif key == ord('A') or key == 263:  # Left arrow
                self.translate(-self.translation_step, 0, 0)
                vis.update_geometry(self.mesh_updated)
            elif key == ord('D') or key == 262:  # Right arrow
                self.translate(self.translation_step, 0, 0)
                vis.update_geometry(self.mesh_updated)
            elif key == ord('Q'):  # Move up (Y+)
                self.translate(0, self.translation_step, 0)
                vis.update_geometry(self.mesh_updated)
            elif key == ord('E'):  # Move down (Y-)
                self.translate(0, -self.translation_step, 0)
                vis.update_geometry(self.mesh_updated)
                
            # Rotation
            elif key == ord('R'):
                self.rotate('y', self.rotation_step)  # Yaw
                vis.update_geometry(self.mesh_updated)
            elif key == ord('F'):
                self.rotate('y', -self.rotation_step)  # Yaw
                vis.update_geometry(self.mesh_updated)
            elif key == ord('T'):
                self.rotate('x', self.rotation_step)  # Pitch
                vis.update_geometry(self.mesh_updated)
            elif key == ord('G'):
                self.rotate('x', -self.rotation_step)  # Pitch
                vis.update_geometry(self.mesh_updated)
            elif key == ord('Y'):
                self.rotate('z', self.rotation_step)  # Roll
                vis.update_geometry(self.mesh_updated)
            elif key == ord('H'):
                self.rotate('z', -self.rotation_step)  # Roll
                vis.update_geometry(self.mesh_updated)
                
            # Scaling
            elif key == ord('Z'):
                self.scale(1.0 + self.scale_step)
                vis.update_geometry(self.mesh_updated)
            elif key == ord('X'):
                self.scale(1.0 / (1.0 + self.scale_step))
                vis.update_geometry(self.mesh_updated)
            elif key == ord('=') or key == ord('+'):
                self.scale(1.0 + self.scale_step * 0.1)  # Fine scale
                vis.update_geometry(self.mesh_updated)
            elif key == ord('-'):
                self.scale(1.0 / (1.0 + self.scale_step * 0.1))  # Fine scale
                vis.update_geometry(self.mesh_updated)
                
            # Save
            elif key == ord('P'):
                self.save()
                
            # Reset
            elif key == ord('B'):
                self.reset()
                vis.update_geometry(self.mesh_updated)
            
            # Point cloud visibility toggle
            elif key == ord('V'):
                self.toggle_pcd_visibility(vis)
            
            # Point cloud density control
            elif key == ord(']'):
                self.change_pcd_density(vis, increase=True)
            elif key == ord('['):
                self.change_pcd_density(vis, increase=False)
            
            # Translation step size control
            elif key == ord('.'):
                self.change_translation_step(increase=True)
            elif key == ord(','):
                self.change_translation_step(increase=False)
                
            # Quit
            elif key == 27:  # ESC
                return False  # Close window
                
        return True
        
    def run(self):
        """Run interactive visualization"""
        print("\n" + "="*60)
        print("INTERACTIVE POSE EDITOR")
        print("="*60)
        print("Controls:")
        print("  Translation: W/S (Z axis), A/D (X axis), Q/E (Y axis)")
        print("  Translation Step: , (decrease), . (increase)")
        print("  Rotation: R/F (yaw), T/G (pitch), Y/H (roll)")
        print("  Scaling: Z/X (coarse), +/- (fine)")
        print("  Point Cloud: V (toggle visibility), [ ] (decrease/increase density)")
        print("  Save: P key")
        print("  Reset: B key")
        print("  Quit: ESC")
        print(f"  Current translation step: {self.translation_step*1000:.2f} mm")
        print("="*60 + "\n")
        
        # Create visualization
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=f"Interactive Pose Editor: {self.img_name}", width=1920, height=1080)
        
        # Add geometries
        # Keep original colors from point cloud (don't paint uniform color)
        self.mesh_updated.paint_uniform_color([1.0, 0.0, 0.0])  # Red mesh
        self.vis.add_geometry(self.scene_pcd)
        self.vis.add_geometry(self.mesh_updated)
        
        # Register key callbacks - create proper closures for each key
        def make_callback(key_code):
            return lambda vis: self.key_callback(vis, key_code, 0)
        
        # Register all keys
        self.vis.register_key_callback(ord('W'), make_callback(ord('W')))
        self.vis.register_key_callback(ord('A'), make_callback(ord('A')))
        self.vis.register_key_callback(ord('S'), make_callback(ord('S')))
        self.vis.register_key_callback(ord('D'), make_callback(ord('D')))
        self.vis.register_key_callback(ord('Q'), make_callback(ord('Q')))
        self.vis.register_key_callback(ord('E'), make_callback(ord('E')))
        self.vis.register_key_callback(ord('R'), make_callback(ord('R')))
        self.vis.register_key_callback(ord('F'), make_callback(ord('F')))
        self.vis.register_key_callback(ord('T'), make_callback(ord('T')))
        self.vis.register_key_callback(ord('G'), make_callback(ord('G')))
        self.vis.register_key_callback(ord('Y'), make_callback(ord('Y')))
        self.vis.register_key_callback(ord('H'), make_callback(ord('H')))
        self.vis.register_key_callback(ord('Z'), make_callback(ord('Z')))
        self.vis.register_key_callback(ord('X'), make_callback(ord('X')))
        self.vis.register_key_callback(ord('='), make_callback(ord('=')))
        self.vis.register_key_callback(ord('+'), make_callback(ord('+')))
        self.vis.register_key_callback(ord('-'), make_callback(ord('-')))
        self.vis.register_key_callback(ord('P'), make_callback(ord('P')))  # Save
        self.vis.register_key_callback(ord('B'), make_callback(ord('B')))  # Reset
        self.vis.register_key_callback(ord('V'), make_callback(ord('V')))  # Toggle point cloud visibility
        self.vis.register_key_callback(ord('['), make_callback(ord('[')))  # Decrease point cloud density
        self.vis.register_key_callback(ord(']'), make_callback(ord(']')))  # Increase point cloud density
        self.vis.register_key_callback(ord(','), make_callback(ord(',')))  # Decrease translation step
        self.vis.register_key_callback(ord('.'), make_callback(ord('.')))  # Increase translation step
        self.vis.register_key_callback(27, make_callback(27))  # ESC
        
        # Arrow keys
        self.vis.register_key_callback(265, make_callback(265))  # Up
        self.vis.register_key_callback(264, make_callback(264))  # Down
        self.vis.register_key_callback(263, make_callback(263))  # Left
        self.vis.register_key_callback(262, make_callback(262))  # Right
        
        # Set up view
        ctr = self.vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.7)
        
        # Run visualization loop
        self.vis.run()
        self.vis.destroy_window()
        
        # Auto-save if not saved
        if not self.saved:
            response = input(f"\nPose not saved. Save current pose for {self.img_name}? (y/n): ")
            if response.lower() == 'y':
                self.save()


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

    # Optional: resume from a specific info_interactive folder
    # resume_idx: -1 -> load from info/, save to info_interactive/
    # resume_idx: 0 -> load from info_interactive/, save to info_interactive_1/
    # resume_idx: 1 -> load from info_interactive_1/, save to info_interactive_2/
    resume_idx = cfg.s8_pose.get("resume_idx", None)
    
    if resume_idx is not None:
        # Determine source folder to load from
        if resume_idx == -1:
            # Special case: load from non-interactive info folder
            load_dir = f"{out_dir}/info"
            load_suffix = "_non_interactive"  # Special marker for non-interactive
            save_suffix = ""
            interactive_suffix = save_suffix
            
            if not os.path.exists(load_dir):
                raise ValueError(f"Non-interactive info folder does not exist: {load_dir}")
            
            logger.info("="*60)
            logger.info("Starting INTERACTIVE object pose matching pipeline (RESUME FROM NON-INTERACTIVE)...")
            logger.info(f"Output directory: {out_dir}")
            logger.info(f"Loading poses from: info/")
            logger.info(f"Saving adjusted poses to: info_interactive/")
            logger.info("="*60)
        else:
            # Load from info_interactive folders
            load_suffix = "" if resume_idx == 0 else f"_{resume_idx}"
            load_dir = f"{out_dir}/info_interactive{load_suffix}"
            
            if not os.path.exists(load_dir):
                raise ValueError(f"Resume folder does not exist: {load_dir}")
            
            # Save to next folder
            save_suffix = f"_{resume_idx + 1}"
            interactive_suffix = save_suffix
            
            logger.info("="*60)
            logger.info("Starting INTERACTIVE object pose matching pipeline (RESUME MODE)...")
            logger.info(f"Output directory: {out_dir}")
            logger.info(f"Loading poses from: info_interactive{load_suffix}/")
            logger.info(f"Saving adjusted poses to: info_interactive{save_suffix}/")
            logger.info("="*60)
    else:
        # Normal mode: generate new interactive suffix based on existing directories
        interactive_suffix = ""
        interactive_idx = 1
        while os.path.exists(f"{out_dir}/info_interactive{interactive_suffix}"):
            interactive_suffix = f"_{interactive_idx}"
            interactive_idx += 1
        load_suffix = None  # No loading, run pose fitting
        
        logger.info("="*60)
        logger.info("Starting INTERACTIVE object pose matching pipeline...")
        logger.info(f"Output directory: {out_dir}")
        logger.info(f"Interactive poses will be saved to: info_interactive{interactive_suffix}/")
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

    # Create VLM
    vlm = Gemini(
        project=cfg.gcloud_project,
        location="global",
        model="gemini-2.5-flash",
    )

    # Helper function to extract iteration number for proper numerical sorting
    def get_iter_num(filename):
        """Extract iteration number from filename for numerical sorting"""
        if "iter" not in filename or not filename.endswith(".glb"):
            return float('inf')  # Non-iter files go to the end
        try:
            # Extract number between "iter_" and "_mesh"
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

        mesh_path = f"{mesh_dir}/{filename}"
        base_iter = resolve_base_iteration(scene_dir, idx)
        depth_filename = "original_depth.npy" if base_iter is None else f"metric_depth/iter_{base_iter}.npy"
        depth_path = f"{scene_dir}/{depth_filename}"
        rgb_filename = "source_padded_resized.png" if base_iter is None else f"post_object_removal/iter_{base_iter}.png"
        rgb_path = f"{scene_dir}/{rgb_filename}"
        mask_path = f"{scene_dir}/removal_mask/iter_{idx}.png"
        
        # Path to canonical mesh (always same location)
        canonical_mesh_fpath = f"{out_dir}/canonical_mesh/{img_name}.glb"
        
        # Path to existing pose file (for resume mode)
        if load_suffix is not None:
            if load_suffix == "_non_interactive":
                # Special case: load from non-interactive info folder
                existing_pose_path = f"{out_dir}/info/{img_name}.json"
            else:
                existing_pose_path = f"{out_dir}/info_interactive{load_suffix}/{img_name}.json"
        else:
            existing_pose_path = None
        
        # Always load rgb/depth/mask for scene visualization
        rgb = np.array(Image.open(rgb_path))
        depth = np.load(depth_path)
        mask = np.array(Image.open(mask_path))
        rgb_resized = cv2.resize(rgb, (W, H))
        rgb_resized_unpadded = unpad_image(rgb_resized, delta_w, delta_h)
        mask_resized = cv2.resize(mask, (W, H))
        mask_resized_unpadded = unpad_image(mask_resized, delta_w, delta_h)
        mask_resized_unpadded_eroded = erode_mask(mask_resized_unpadded, kernel_size=3).astype(bool)
        
        # Compute full scene point cloud (needed for visualization in both modes)
        pc = compute_point_cloud_from_depth(depth=depth, K=K).reshape(-1, 3)
        
        # Check if we're resuming and existing pose file exists
        is_resuming = (existing_pose_path is not None and 
                       os.path.exists(existing_pose_path) and 
                       os.path.exists(canonical_mesh_fpath))
        
        if is_resuming:
            # RESUME MODE: Load existing pose and mesh, skip pose fitting
            logger.info(f"Loading existing pose from {existing_pose_path}")
            
            with open(existing_pose_path, "r") as f:
                existing_pose = json.load(f)
            
            # Load existing canonical mesh
            canonical_mesh = o3d.io.read_triangle_mesh(canonical_mesh_fpath, enable_post_processing=True)
            
            # Reconstruct transformation from saved pose
            top_info = {
                "tf_z_up": transformation.RigidTransformation(
                    scale=np.array(existing_pose["z_up"]["scale"]),
                    rot=np.array(existing_pose["z_up"]["rot"]),
                    t=np.array(existing_pose["z_up"]["trans"])
                ),
                "tf_y_up": transformation.RigidTransformation(
                    scale=np.array(existing_pose["y_up"]["scale"]),
                    rot=np.array(existing_pose["y_up"]["rot"]),
                    t=np.array(existing_pose["y_up"]["trans"])
                )
            }
        else:
            # NORMAL MODE: Run full pose fitting pipeline
            # Compose object point cloud from masked region
            obj_pc = pc[mask_resized_unpadded_eroded.flatten()]
            obj_rgb = (rgb_resized_unpadded.reshape(-1, 3) / 255)[mask_resized_unpadded_eroded.flatten()]
            obj_pcd = o3d.geometry.PointCloud()
            obj_pcd.points = o3d.utility.Vector3dVector(obj_pc)
            obj_pcd.colors = o3d.utility.Vector3dVector(obj_rgb)

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

            # Sample fits
            # We return the inverse because sample fits computes the tf from source (partial PC) -> target (mesh PC), but
            # we want the inverse operation (mesh fit to partial PC)
            # info_sorted = sample_fits(source, target, cfg.s8_pose.n_samples, update_scale=False)
            info_unscaled_sorted = sample_fits(source, target, cfg.s8_pose.n_samples, update_scale=False, return_inverse=True)
            info_scaled_sorted = sample_fits(source, target, cfg.s8_pose.n_samples, update_scale=True, return_inverse=True)
            # Prune any scaled values that are degenerate -- this is any computed TF with scale < 0.1
            # We expect this to be degenerate because we already pre-scale the raw mesh to a "reasonable" order of magnitude
            # based on the real-world partial point cloud...so it shouldn't be more than +/- 1 order of magnitude
            info_scaled_sorted = [info for info in info_scaled_sorted if (info["tf_z_up"].scale > 0.1 and info["tf_z_up"].scale < 10.0)]
            info_sorted = sorted(info_unscaled_sorted + info_scaled_sorted, key=lambda x: x["dist"])
            top_info = info_sorted[0]
            # top_info = sample_fits_multiprocess(source, target, cfg.s8_pose.n_samples, 8)

            # Update mesh -- this is pre-scaled, pre-transform (canonical object frame) -- save this

            # Scale and save via trimesh because open3d can't save glb files
            canonical_mesh_tm = trimesh.load(mesh_path)
            canonical_mesh_tm.apply_scale(pre_scale_factor)
            canonical_mesh_tm.export(canonical_mesh_fpath)

            # Load canonical mesh in open3d
            canonical_mesh = o3d.io.read_triangle_mesh(canonical_mesh_fpath, enable_post_processing=True)

        # Potentially load Any6D if requested (skip when resuming)
        use_any6d = cfg.s8_pose.use_any6d
        if use_any6d and not is_resuming:
            from estimater import Any6D

            # Export updated mesh to load into trimesh and run Any6D
            with tempfile.TemporaryDirectory() as tmpdir:
                canonical_mesh_scaled = o3d.geometry.TriangleMesh(canonical_mesh)
                vertices = np.asarray(canonical_mesh_scaled.vertices)
                vertices = top_info["tf_z_up"].scale * vertices
                canonical_mesh_scaled.vertices = o3d.utility.Vector3dVector(vertices)

                tmp_obj_path = f"{tmpdir}/{img_name}_canonical_mesh.obj"
                o3d.io.write_triangle_mesh(tmp_obj_path, canonical_mesh_scaled)
                canonical_mesh_scaled_tm = trimesh.load(tmp_obj_path)
                est = Any6D(symmetry_tfs=None, mesh=canonical_mesh_scaled_tm, debug_dir=f"{out_dir}/debug", debug=3)
                # any6d_tf = est.register_any6d(
                #     K=K,
                #     rgb=rgb_resized_unpadded,
                #     depth=depth,
                #     ob_mask=mask_resized_unpadded.astype(bool), #_eroded,
                #     iteration=5,
                #     name=img_name,
                #     coarse_est=False,
                #     refinement=False,
                # )

                pose_data, ids_ordered = est.register(
                    K=K,
                    rgb=rgb_resized_unpadded,
                    depth=depth,
                    ob_mask=mask_resized_unpadded.astype(bool),  # _eroded,
                    iteration=5,
                    name=img_name,
                    return_all_poses=True,
                )
                ids_ordered = ids_ordered.cpu().numpy()
                # ids_ranking_mapping = {v: idx for idx, v in enumerate(ids_ordered)}
                poses = pose_data.poseA.cpu().numpy()
                # Only keep top-20 idxs
                topk_candidates = 25
                quats = R.from_matrix(poses[ids_ordered[:topk_candidates], :3, :3]).as_quat()
                n_clusters = 3
                centers, clusters = quaternion_kmeans(quats, n_clusters, max_iters=100)
                if len(clusters) == 0:
                    logger.warning("NO CLUSTERS")
                    from IPython import embed; embed()
                # Update n_clusters to reflect actual number after pruning empty clusters
                n_clusters = len(clusters)
                logger.info(f"K-means produced {n_clusters} non-empty clusters")
                # Keep the top-1 from each cluster
                # candidate_poses = dict()
                candidate_idxs = []
                candidate_img_fpaths = []
                img_stack = []
                for i, cluster in enumerate(clusters):
                    # sorted_cluster = sorted(cluster, key=lambda c: ids_ranking_mapping[c])
                    # sorted_cluster = sorted(cluster, key=lambda c: ids_ordered[c])
                    # best_pose_idx = sorted_cluster[0]
                    best_pose_idx = ids_ordered[min(cluster)]
                    candidate_idxs.append(best_pose_idx)
                    img = (pose_data.rgbAs[best_pose_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    img_stack.append(img)
                    tmp_img_path = f"{tmpdir}/{img_name}_candidate_pose_{i}.png"
                    Image.fromarray(img).save(tmp_img_path)
                    candidate_img_fpaths.append(tmp_img_path)

                # Visualize stacked image
                img_stacked = Image.fromarray(np.concatenate(img_stack, axis=1))
                img_stacked.save(f"{out_dir}/fit/{img_name}_cand_fit.png")
                if cfg.visualize:
                    img_stacked.show()

                # Pass top-scoring value from each cluster through VLM to sanity check values
                reference_img = (pose_data.rgbBs[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                tmp_ref_img_path = f"{tmpdir}/{img_name}_reference_pose.png"
                Image.fromarray(reference_img).save(tmp_ref_img_path)
                result = vlm(
                    prompt=prompt_topk_image_select(n_candidates=n_clusters),
                    image_paths=[tmp_ref_img_path] + candidate_img_fpaths,
                    temperature=0,
                    top_p=0,
                    seed=0,
                    print_results=True,
                )
                # Sanitize output: Extract number from the last line of the answer
                result_str = vlm.get_result_text(result=result).split("\n")[-1].strip()
                numbers = extract_numbers_from_str(result_str)
                assert len(numbers) == 1
                # Numbered from 2 - N_candidates + 1 and this is 1-indexed so we need to offset by 2 to get the true idx
                best_candidate_idx = numbers[0] - 2
                any6d_tf = poses[candidate_idxs[best_candidate_idx]]
                best_img_fpath = candidate_img_fpaths[best_candidate_idx]

                # Save this image file
                best_img_cat = Image.fromarray(np.concatenate([np.array(Image.open(fpath)) for fpath in (tmp_ref_img_path, best_img_fpath)], axis=1))
                best_img_cat.save(f"{out_dir}/fit/{img_name}_best_fit.png")

                if cfg.visualize:
                    best_img_cat.show()
                    # Image.open(best_img_fpath).show()
                    # Image.open(tmp_ref_img_path).show()
                # Sto

                # Update canonical mesh: re-export the GLB at Any6D's scale and centering
                # (est.mesh is the centered copy) to maintain metallic appearance
                a6d_scaled_mesh = trimesh.load(canonical_mesh_fpath)
                a6d_scaled_mesh.apply_scale(top_info["tf_z_up"].scale)
                a6d_scaled_mesh.apply_translation(-(a6d_scaled_mesh.bounds[0] + a6d_scaled_mesh.bounds[1]) / 2.0)
                a6d_scaled_mesh.export(canonical_mesh_fpath)
                canonical_mesh = o3d.io.read_triangle_mesh(canonical_mesh_fpath, enable_post_processing=True)

                # Update transform
                any6d_tf_rt = transformation.RigidTransformation(scale=np.ones(3), rot=any6d_tf[:3, :3], t=any6d_tf[:3, 3])
                top_info["tf_y_up"] = any6d_tf_rt * tf_y_up
                top_info["tf_z_up"] = any6d_tf_rt

        # Create scene point cloud for visualization
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(pc.reshape(-1, 3))
        scene_pcd.colors = o3d.utility.Vector3dVector(rgb_resized_unpadded.reshape(-1, 3) / 255)
        
        # Downsample point cloud slightly for better visualization (so mesh is more visible)
        # Use voxel downsampling with size based on bounding box diagonal
        scene_bbox = scene_pcd.get_axis_aligned_bounding_box()
        scene_diag = np.linalg.norm(scene_bbox.get_extent())
        voxel_size = scene_diag / 3000  # Downsample to ~200 voxels across diagonal (denser point cloud)
        scene_pcd = scene_pcd.voxel_down_sample(voxel_size=voxel_size)
        # Colors are automatically preserved by averaging within each voxel

        # Run interactive editor
        editor = InteractivePoseEditor(
            scene_pcd=scene_pcd,
            canonical_mesh=canonical_mesh,
            initial_tf_z_up=top_info["tf_z_up"],
            initial_tf_y_up=top_info["tf_y_up"],
            img_name=img_name,
            out_dir=out_dir,
            interactive_suffix=interactive_suffix
        )
        logger.info(f"Processing {img_name}")
        editor.run()

    logger.info("="*60)
    logger.info("Interactive object pose matching complete!")
    logger.info("="*60)
    
    end_stage(cfg, success=True)

def end_stage(cfg, success=False, additional_info: dict = None):
    """
    Function that is run at the end of every stage to save the stage config and additional info.

    # TODO: should this be a general function that can be used for all stages? or should we have a separate function for each stage?
    """
    save_dir = cfg.s8_pose.out_dir
    stage_cfg = OmegaConf.to_object(cfg.s8_pose)
    stage_cfg['success'] = success
    if additional_info is not None:
        stage_cfg.update(additional_info)
    dump_json(stage_cfg, f"{save_dir}/stage_info.json")


if __name__ == "__main__":
    main()
