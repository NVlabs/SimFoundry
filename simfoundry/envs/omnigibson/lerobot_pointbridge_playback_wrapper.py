# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence
import torch
import torch as th
import numpy as np
import open3d as o3d
from pytorch3d.ops import sample_farthest_points
import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm

from omnigibson.envs.data_wrapper import LeRobotPlaybackWrapper
from omnigibson.sensors.vision_sensor import VisionSensor

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME


class LeRobotPlaybackWrapperWithTransforms(LeRobotPlaybackWrapper):
    """
    Extension of LeRobotPlaybackWrapper that supports data transformations during saving.
    
    This wrapper allows inverting gripper actions and scaling gripper states when saving
    data to disk, without affecting the actual environment execution.
    """
    
    def __init__(self, *args, invert_gripper_action: bool = False, scale_gripper_state: float = 1.0, **kwargs):
        """
        Args:
            *args: Arguments passed to parent LeRobotPlaybackWrapper
            invert_gripper_action (bool): If True, invert gripper action when saving: action[-1] = 1 - action[-1]
            scale_gripper_state (float): Factor to multiply gripper state by when saving: state[-1] *= scale
            **kwargs: Keyword arguments passed to parent LeRobotPlaybackWrapper
        """
        self.invert_gripper_action = invert_gripper_action
        self.scale_gripper_state = scale_gripper_state
        super().__init__(*args, **kwargs)
    
    def process_traj_to_dataset(self, traj_data, nested_keys=("obs",)):
        """
        Process trajectory with optional data transformations for gripper action/state.
        """
        invert_gripper_action = getattr(self, 'invert_gripper_action', False)
        scale_gripper_state = getattr(self, 'scale_gripper_state', 1.0)
        
        if invert_gripper_action or scale_gripper_state != 1.0:
            for step_data in traj_data:
                # Transform action (gripper is last element)
                if invert_gripper_action and "action" in step_data:
                    action = step_data["action"]
                    if hasattr(action, '__len__') and len(action) > 0:
                        if isinstance(action, th.Tensor):
                            step_data["action"] = action.clone()
                            step_data["action"][-1] = 1.0 - step_data["action"][-1]
                        else:
                            step_data["action"] = np.array(action).copy()
                            step_data["action"][-1] = 1.0 - step_data["action"][-1]
                
                # Transform observation.state (gripper is last element)
                if scale_gripper_state != 1.0 and "obs" in step_data:
                    obs = step_data["obs"]
                    for key in list(obs.keys()):
                        if "state" in key.lower() or "proprio" in key.lower() or "gripper" in key.lower():
                            state = obs[key]
                            if hasattr(state, '__len__') and len(state) > 0:
                                if isinstance(state, th.Tensor):
                                    obs[key] = state.clone()
                                    obs[key][-1] = obs[key][-1] * scale_gripper_state
                                else:
                                    obs[key] = np.array(state).copy()
                                    obs[key][-1] = obs[key][-1] * scale_gripper_state
        
        super().process_traj_to_dataset(traj_data, nested_keys)


class PointCloudVisualizer:
    """
    Non-blocking Open3D visualizer for point clouds that automatically updates.
    
    Display modes (toggle with 'V' key):
        - "both": Show both red (before FPS) and green (after FPS) point clouds overlaid
        - "before": Show only red (before FPS) point cloud
        - "after": Show only green (after FPS) point cloud
    """
    
    _instance = None  # Singleton instance
    DISPLAY_MODES = ["both", "before", "after"]
    
    def __init__(self, window_name: str = "Point Cloud Visualization", width: int = 1280, height: int = 720):
        """
        Initialize the visualizer window.
        
        Args:
            window_name (str): Name of the visualization window
            width (int): Window width in pixels
            height (int): Window height in pixels
        """
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=window_name, width=width, height=height)
        
        # Display mode: "both", "before", "after"
        self._display_mode_idx = 0  # Start with "both"
        
        # Register keyboard callback for 'V' key (ASCII 86) to toggle display mode
        self.vis.register_key_callback(ord('V'), self._toggle_display_mode_callback)
        self.vis.register_key_callback(ord('v'), self._toggle_display_mode_callback)
        
        # Create persistent geometry objects
        self.pcd_before = o3d.geometry.PointCloud()
        self.pcd_after = o3d.geometry.PointCloud()
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        
        # Add geometries to visualizer
        self.vis.add_geometry(self.pcd_before)
        self.vis.add_geometry(self.pcd_after)
        self.vis.add_geometry(self.coord_frame)
        
        self._initialized = True
        self._first_update = True
        
        # Store last point cloud data for re-rendering on mode change
        self._last_pc_before = None
        self._last_pc_after = None
        self._last_before_color = [1.0, 0.0, 0.0]
        self._last_after_color = [0.0, 1.0, 0.0]
        
        print("[PointCloudVisualizer] Press 'V' to toggle display mode: both (overlay) -> before only (red) -> after only (green)")
    
    @property
    def display_mode(self) -> str:
        """Get current display mode."""
        return self.DISPLAY_MODES[self._display_mode_idx]
    
    def _toggle_display_mode_callback(self, vis):
        """Callback for toggling display mode when 'V' is pressed."""
        self._display_mode_idx = (self._display_mode_idx + 1) % len(self.DISPLAY_MODES)
        mode = self.display_mode
        print(f"[PointCloudVisualizer] Display mode: {mode}")
        
        # Re-apply visibility based on new mode
        self._apply_display_mode()
        return False  # Return False to allow other callbacks
    
    def _apply_display_mode(self):
        """Apply current display mode by showing/hiding point clouds."""
        mode = self.display_mode
        
        if mode == "both":
            # Show both point clouds
            if self._last_pc_before is not None:
                self.pcd_before.points = o3d.utility.Vector3dVector(self._last_pc_before)
                self.pcd_before.paint_uniform_color(self._last_before_color)
            if self._last_pc_after is not None:
                self.pcd_after.points = o3d.utility.Vector3dVector(self._last_pc_after)
                self.pcd_after.paint_uniform_color(self._last_after_color)
        elif mode == "before":
            # Show only before (red), hide after
            if self._last_pc_before is not None:
                self.pcd_before.points = o3d.utility.Vector3dVector(self._last_pc_before)
                self.pcd_before.paint_uniform_color(self._last_before_color)
            self.pcd_after.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        elif mode == "after":
            # Show only after (green), hide before
            self.pcd_before.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            if self._last_pc_after is not None:
                self.pcd_after.points = o3d.utility.Vector3dVector(self._last_pc_after)
                self.pcd_after.paint_uniform_color(self._last_after_color)
        
        # Update geometries
        self.vis.update_geometry(self.pcd_before)
        self.vis.update_geometry(self.pcd_after)
        self.vis.poll_events()
        self.vis.update_renderer()
    
    @classmethod
    def get_instance(cls):
        """Get or create the singleton visualizer instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def update(
        self,
        pc_before: torch.Tensor,
        pc_after: torch.Tensor,
        title: str = "Point Cloud Visualization",
        before_color: list = [1.0, 0.0, 0.0],  # Red for before FPS
        after_color: list = [0.0, 1.0, 0.0],   # Green for after FPS
    ):
        """
        Update the point cloud visualization with new data.
        
        Args:
            pc_before (torch.Tensor): Point cloud before FPS of shape (N, 3)
            pc_after (torch.Tensor): Point cloud after FPS of shape (M, 3)
            title (str): Title for debug output
            before_color (list): RGB color for before FPS points
            after_color (list): RGB color for after FPS points
        """
        # Store data for display mode changes
        self._last_pc_before = pc_before.cpu().numpy()
        self._last_pc_after = pc_after.cpu().numpy()
        self._last_before_color = before_color
        self._last_after_color = after_color
        
        # Apply current display mode
        mode = self.display_mode
        
        if mode == "both":
            self.pcd_before.points = o3d.utility.Vector3dVector(self._last_pc_before)
            self.pcd_before.paint_uniform_color(before_color)
            self.pcd_after.points = o3d.utility.Vector3dVector(self._last_pc_after)
            self.pcd_after.paint_uniform_color(after_color)
        elif mode == "before":
            self.pcd_before.points = o3d.utility.Vector3dVector(self._last_pc_before)
            self.pcd_before.paint_uniform_color(before_color)
            self.pcd_after.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        elif mode == "after":
            self.pcd_before.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            self.pcd_after.points = o3d.utility.Vector3dVector(self._last_pc_after)
            self.pcd_after.paint_uniform_color(after_color)
        
        # Update geometries in visualizer
        self.vis.update_geometry(self.pcd_before)
        self.vis.update_geometry(self.pcd_after)
        
        # Reset view on first update to frame all geometry
        if self._first_update:
            self.vis.reset_view_point(True)
            self._first_update = False
        
        # Poll events and update renderer (non-blocking)
        self.vis.poll_events()
        self.vis.update_renderer()
        
        # Print debug info
        mode_str = f"[{mode.upper()}]"
        print(f"[DEBUG] {mode_str} {title} - Before FPS (red): {pc_before.shape[0]} pts, After FPS (green): {pc_after.shape[0]} pts")
    
    def close(self):
        """Close the visualizer window."""
        if self._initialized:
            self.vis.destroy_window()
            self._initialized = False
            PointCloudVisualizer._instance = None


def visualize_point_clouds_o3d(
    pc_before: torch.Tensor,
    pc_after: torch.Tensor,
    title: str = "Point Cloud Visualization",
    before_color: list = [1.0, 0.0, 0.0],  # Red for before FPS
    after_color: list = [0.0, 1.0, 0.0],   # Green for after FPS
):
    """
    Visualize point clouds before and after farthest point sampling using Open3D.
    Uses a non-blocking visualizer that automatically updates.
    
    Args:
        pc_before (torch.Tensor): Point cloud before FPS of shape (N, 3)
        pc_after (torch.Tensor): Point cloud after FPS of shape (M, 3)
        title (str): Window title
        before_color (list): RGB color for before FPS points
        after_color (list): RGB color for after FPS points
    """
    # Get or create the singleton visualizer
    visualizer = PointCloudVisualizer.get_instance()
    
    # Update the visualization with new point cloud data
    visualizer.update(
        pc_before=pc_before,
        pc_after=pc_after,
        title=title,
        before_color=before_color,
        after_color=after_color,
    )


def compute_uvk_inv(K, h, w, device="cuda"):
    """
    Computes projection matrix to multiply by depth values

    Args:
        K (th.Tensor): (N,3,3) batched cam intrinsics matrices
        h (int): height to compute in pixels
        w (int): width to compute in pixels

    Returns:
        th.Tensor: (N,H*W,3) pre-computed projection matrices
    """
    K = K.to(device)
    K_inv = torch.linalg.inv(K)
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    u = x
    v = y
    uv = torch.dstack((u, v, torch.ones_like(u))).float()

    return uv.view(1, -1, 3) @ K_inv.transpose(-2, -1)  # (N,H*W,3)

@torch.compile
def compute_point_cloud_from_depth_torch_batch(depth, K=None, uvk_inv=None, cam_to_img_tf=None, world_to_cam_tf=None, device="cuda"):
    """
    Computes point cloud from depth image.

    Args:
        depth (th.Tensor): (N,H,W) Input depth map with normalized values (the output of @process_depth_linear)
        K (th.Tensor): (N,3,3) cam intrinsics matrices. Either this or @uvk_inv (not both) must be specified
        uvk_inv (None or th.Tensor): (N,H*W,3) optional pre-computed projection matrices.
            Either this or @K (not both) must be specified
        cam_to_img_tf (th.Tensor): (N,4,4) Camera to image coordinate transformation matrix.
                    omni cam_to_img_tf is T.pose2mat(([0, 0, 0], T.euler2quat([np.pi, 0, 0])))
        world_to_cam_tf (th.Tensor): (N,4,4) World to camera coordinate transformation matrix

    Returns:
        th.Tensor: Resulting point cloud of shape (N, H, W, 3).
    """
    depth = depth.to(device)
    N, h, w = depth.shape
    assert depth.min() >= 0

    if uvk_inv is None:
        uvk_inv = compute_uvk_inv(K, h, w, device=device)

    pc = depth.view(N, -1, 1) * uvk_inv         # (N,H*W,3)
    pc = torch.concatenate([pc, torch.ones((N, h * w, 1), device=device)], dim=-1)  # shape (N,H*W,4)

    # Convert using camera transform
    if cam_to_img_tf is not None:
        pc = pc @ cam_to_img_tf.to(device).transpose(-2, -1)
    if world_to_cam_tf is not None:
        pc = pc @ world_to_cam_tf.to(device).transpose(-2, -1)
    # Create (N,H,W,3) vector from pc
    pc = pc[:, :, :3].view(N, h, w, 3)

    return pc


def pointcloud_from_depth_segmented(
    depth,
    seg_mask,
    robot2cam_tf,
    K=None,
    uvk_inv=None,
    id2labels=None,
    cam_to_img_tf=None,
    num_points: int = 1024,
    include_categories: Sequence[str] | None = None,
    x_min: float = -2.0,
    x_max: float = 2.0,
    y_min: float = -2.0,
    y_max: float = 2.0,
    z_min: float = 0.001,
    z_max: float = 2.0,
    device: str = "cuda",
    sensor_name: str = "unknown",
) -> torch.Tensor:
    """
    Create a point cloud from depth, filtered by semantic segmentation.

    Args:
        depth (th.Tensor): (H, W) Depth image
        seg_mask (th.Tensor): (H, W) Segmentation mask with integer class IDs
        robot2cam_tf (th.Tensor): (4, 4) Robot to camera transformation matrix
        K (th.Tensor): (3, 3) Camera intrinsics matrix
        uvk_inv (th.Tensor): (H*W, 3) Pre-computed projection matrices
        id2labels (dict or None): Mapping from segmentation ID to label info. If None, all pixels are included.
        num_points (int): Number of points to sample in the final point cloud
        include_categories (list[str] or None): Categories to INCLUDE (only these will be in point cloud).
                        If None, all categories are included.
        cam_to_img_tf (th.Tensor): (4, 4) Camera to image transformation matrix. If None, the default is used.
        x_min, x_max: X workspace bounds (in robot frame).
        y_min, y_max: Y workspace bounds (in robot frame).
        z_min, z_max: Z workspace bounds (in robot frame).
        device (str): Device to use for computation
        sensor_name (str): Name of the sensor (used for debug visualization title)

    Returns:
        th.Tensor: Point cloud of shape (num_points, 3)
    """
    depth = depth.to(device)
    seg_mask = seg_mask.to(device)
    K = K.to(device) if K is not None else None
    uvk_inv = uvk_inv.to(device) if uvk_inv is not None else None
    robot2cam_tf = robot2cam_tf.to(device)
    cam_to_img_tf = cam_to_img_tf.to(device) if cam_to_img_tf is not None else None
    h, w = depth.shape

    # Build object mask based on segmentation categories
    if include_categories is not None and id2labels is not None:
        # Get all ids that correspond to requested categories
        include_ids = [
            k
            for k, v in id2labels.items()
            if (v.get('class', v) if isinstance(v, dict) else str(v)) in include_categories
        ]
        # Make tensor on same device, same type as seg_mask
        if include_ids:
            include_ids_tensor = torch.tensor(include_ids, dtype=seg_mask.dtype, device=seg_mask.device)
            obj_mask = (seg_mask.unsqueeze(-1) == include_ids_tensor).any(dim=-1)
        else:
            obj_mask = torch.zeros_like(seg_mask, dtype=torch.bool)
    else:
        # Include all non-zero depth pixels if no filtering specified
        obj_mask = depth > 0

    num_obj_pixels = obj_mask.sum().item()
    if num_obj_pixels == 0:
        return torch.zeros((num_points, 3), device=device)

    # Mask depth - set background to 0 (invalid)
    depth_masked = depth.clone()
    depth_masked[~obj_mask] = 0.0

    # Create point cloud
    # Add batch dimension for the function
    depth_batched = depth_masked.unsqueeze(0)  # (1, H, W)
    # world_to_cam_tf is the inverse of robot2cam_tf (cam_to_robot)
    # We need to go from camera frame to robot frame
    cam_to_robot_tf = robot2cam_tf  # This is already robot-to-camera, we need to invert it
    point_cloud_kwargs = dict()
    if uvk_inv is not None:
        point_cloud_kwargs["uvk_inv"] = uvk_inv
    else:
        assert K is not None, "Either uvk_inv or K must be provided"
        point_cloud_kwargs["K"] = K
    point_cloud = compute_point_cloud_from_depth_torch_batch(
        depth=depth_batched,
        cam_to_img_tf=cam_to_img_tf.unsqueeze(0),
        world_to_cam_tf=cam_to_robot_tf.unsqueeze(0),
        device=device,
        **point_cloud_kwargs,
    )  # (1, H, W, 3)
    point_cloud = point_cloud.squeeze(0).view(-1, 3)  # (H*W, 3)

    # Filter valid points (xyz bounds, finite, non-masked)
    depth_valid = depth_masked.view(-1) > 0
    valid = (
        (point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] <= x_max) &
        (point_cloud[:, 1] >= y_min) & (point_cloud[:, 1] <= y_max) &
        (point_cloud[:, 2] >= z_min) & (point_cloud[:, 2] <= z_max) &
        torch.isfinite(point_cloud).all(dim=1) &
        depth_valid
    )
    filtered_pc = point_cloud[valid]
    # Downsample using farthest point sampling
    n = filtered_pc.shape[0]
    if n > num_points:
        sampled_pc = farthest_point_sampling(filtered_pc, num_points, device)
    elif n > 0:
        # Repeat to fill num_points
        sampled_pc = filtered_pc.repeat((num_points // n) + 1, 1)[:num_points]
    else:
        sampled_pc = torch.zeros((num_points, 3), device=device)
    # Debug visualization if gm.DEBUG is set
    if gm.DEBUG and filtered_pc.shape[0] > 0:
        visualize_point_clouds_o3d(
            pc_before=filtered_pc,
            pc_after=sampled_pc,
            title=f"Point Cloud - {sensor_name} (Before vs After FPS)",
        )

    return sampled_pc


def farthest_point_sampling_o3d(points: torch.Tensor, num_points: int, device: str) -> torch.Tensor:
    """Farthest point sampling using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    pcd = pcd.farthest_point_down_sample(num_points)
    return torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device=device)


def farthest_point_sampling(points: torch.Tensor, num_points: int, device: str) -> torch.Tensor:
    """Farthest point sampling using PyTorch3D."""
    sampled, _ = sample_farthest_points(points.unsqueeze(0), K=num_points)
    return sampled.squeeze(0)


class LeRobotPointbridgePlaybackWrapper(LeRobotPlaybackWrapper):
    """
    An OmniGibson environment wrapper for playing back data and collecting observations 
    with point cloud generation from depth and segmentation maps.
    
    This wrapper extends LeRobotPlaybackWrapper to:
    1. Generate segmented point clouds from depth images and segmentation masks
    2. Store point cloud data in the LeRobot dataset format
    
    NOTE: This assumes a DataCollectionWrapper environment has been used to collect data
    and that the robot has depth_linear and seg_semantic modalities enabled.
    """

    CAM_TO_IMG_TF = T.pose2mat((torch.tensor([0, 0, 0]), T.euler2quat(torch.tensor([th.pi, 0, 0]))))

    def __init__(
        self,
        env,
        input_path,
        output_path,
        n_render_iterations=5,
        overwrite=True,
        only_successes=False,
        flush_every_n_traj=10,
        flush_every_n_steps=0,
        full_scene_file=None,
        load_room_instances=None,
        include_robot_control=True,
        include_contacts=True,
        robot_grasping_mode=None,
        root_dir=HF_LEROBOT_HOME,
        robot_type=None,
        image_writer_threads=10,
        image_writer_processes=5,
        task_name=None,
        lerobot_version="v3.0",
        use_videos=True,
        include_multi_action_representation=False,
        include_modalities=("pointcloud", "proprio"),
        # Point cloud specific parameters
        num_points_obj: int = 128,
        num_objects: int = 1,
        include_categories: list[str] | None = None,
        pc_x_min: float = -2.0,
        pc_x_max: float = 2.0,
        pc_y_min: float = -2.0,
        pc_y_max: float = 2.0,
        pc_z_min: float = 0.001,
        pc_z_max: float = 2.0,
        pointcloud_sensors: list[str] | None = None,
        # Data transformation options (applied during saving only)
        invert_gripper_action: bool = False,
        scale_gripper_state: float = 1.0,
    ):
        """
        Args:
            env (Environment): The environment to wrap
            input_path (str): path to input hdf5 collected data file
            output_path (str): path to the output lerobot dataset
            n_render_iterations (int): Number of rendering iterations
            overwrite (bool): Whether to overwrite existing data
            only_successes (bool): Whether to only save successful episodes
            flush_every_n_traj (int): How often to flush across episodes
            flush_every_n_steps (int): How often to flush within episodes
            full_scene_file (None or str): Full scene file for playback
            load_room_instances (None or str): Room instances to load
            include_robot_control (bool): Whether to include robot control
            include_contacts (bool): Whether to include contacts
            robot_grasping_mode (None or str): If specified, the grasping mode to use for all robots. This will override
                the grasping mode set during robot initialization. Valid modes include: "physical", "assisted", "sticky".
            root_dir (str): Root directory for dataset
            robot_type (None or str): Robot type name
            image_writer_threads (int): Threads for image writing
            image_writer_processes (int): Processes for image writing
            task_name (None or str): Task name
            lerobot_version (str): LeRobot version
            use_videos (bool): Whether to save as videos
            include_multi_action_representation (bool): Whether to include multi action representation
            include_modalities (None or list of str): If specified, only include these observation modalities
                in the dataset. Valid modalities include: "rgb", "depth_linear", "proprio", "pointcloud", etc.
                Note: depth_linear and seg_semantic are always processed internally for point cloud generation,
                but may be excluded from the final dataset if not in this list.
                If None, all modalities are included.
            num_points_obj (int): Number of points to sample for each object in point cloud
            num_objects (int): Number of objects to sample in point cloud
            include_categories (list[str] or None): Categories to include in point cloud segmentation
            pc_x_min, pc_x_max (float): X workspace bounds for point cloud
            pc_y_min, pc_y_max (float): Y workspace bounds for point cloud
            pc_z_min, pc_z_max (float): Z workspace bounds for point cloud
            pointcloud_sensors (list[str] or None): Names of sensors to generate point clouds from.
                If None, generates from all sensors with depth_linear modality.
            invert_gripper_action (bool): If True, invert gripper action when saving: action[-1] = 1 - action[-1]
            scale_gripper_state (float): Factor to multiply gripper state by when saving: state[-1] *= scale
        """
        # Store data transformation options
        self.invert_gripper_action = invert_gripper_action
        self.scale_gripper_state = scale_gripper_state
        
        # Store point cloud parameters
        self.num_points_obj = num_points_obj
        self.num_points = num_points_obj * num_objects
        self.num_objs = num_objects
        self.include_categories = set(include_categories) if include_categories is not None else None
        self.pc_bounds = {
            "x_min": pc_x_min,
            "x_max": pc_x_max,
            "y_min": pc_y_min,
            "y_max": pc_y_max,
            "z_min": pc_z_min,
            "z_max": pc_z_max,
        }
        self.pointcloud_sensors = pointcloud_sensors
        
        # Store depth sensor info for point cloud generation
        self.depth_sensor_info = {}  # Maps sensor_name -> {"K": intrinsic_matrix}
        # obs_mapping_full stores the full obs mapping (without modality filtering) for point cloud processing
        self.obs_mapping_full = None
        
        # Run super
        super().__init__(
            env=env,
            input_path=input_path,
            output_path=output_path,
            n_render_iterations=n_render_iterations,
            overwrite=overwrite,
            only_successes=only_successes,
            flush_every_n_traj=flush_every_n_traj,
            flush_every_n_steps=flush_every_n_steps,
            full_scene_file=full_scene_file,
            load_room_instances=load_room_instances,
            include_robot_control=include_robot_control,
            include_contacts=include_contacts,
            robot_grasping_mode=robot_grasping_mode,
            root_dir=root_dir,
            robot_type=robot_type,
            image_writer_threads=image_writer_threads,
            image_writer_processes=image_writer_processes,
            task_name=task_name,
            lerobot_version=lerobot_version,
            use_videos=use_videos,
            include_multi_action_representation=include_multi_action_representation,
            include_modalities=include_modalities,
        )
        
        # Validate that required modalities are present
        self._validate_sensor_modalities()

    def _validate_sensor_modalities(self):
        """
        Validates that required observation modalities (depth_linear, seg_semantic) 
        are available for point cloud generation.
        
        Supports multiple robots - sensor names are prefixed with robot name for disambiguation.
        """
        n_robots = len(self.env.robots)
        
        # Check robot sensors for each robot
        for robot in self.env.robots:
            robot_prefix = f"{robot.name}_" if n_robots > 1 else ""
            
            for sensor_name, sensor in robot.sensors.items():
                if isinstance(sensor, VisionSensor):
                    if self.pointcloud_sensors is not None:
                        # Check if this sensor is in our list
                        remapped_name = "_".join(sensor_name.split(":")[1:]).lower()
                        full_name = f"{robot_prefix.lower()}{remapped_name}"
                        if (remapped_name not in self.pointcloud_sensors and 
                            sensor_name not in self.pointcloud_sensors and
                            full_name not in self.pointcloud_sensors):
                            continue
                    
                    modalities = sensor.modalities
                    if "depth_linear" in modalities:
                        # Store intrinsic matrix for this sensor
                        remapped_name = "_".join(sensor_name.split(":")[1:]).lower()
                        # Add robot prefix for multi-robot disambiguation
                        sensor_key = f"{robot_prefix.lower()}{remapped_name}"
                        uvk_inv = compute_uvk_inv(K=sensor.intrinsic_matrix.clone().float().unsqueeze(0), h=sensor.image_height, w=sensor.image_width)
                        self.depth_sensor_info[sensor_key] = {
                            "uvk_inv": uvk_inv,
                            "sensor_name": sensor_name,
                            "robot_name": robot.name,
                        }
                        
                        # Warn if seg_semantic is not available
                        if "seg_semantic" not in modalities and self.include_categories is not None:
                            print(f"Warning: Sensor {sensor_name} does not have seg_semantic modality. "
                                  f"Point cloud will not be filtered by category.")
        
        # Check external sensors
        if self.env.external_sensors is not None:
            for sensor_name, sensor in self.env.external_sensors.items():
                if isinstance(sensor, VisionSensor):
                    if self.pointcloud_sensors is not None:
                        if sensor_name not in self.pointcloud_sensors:
                            continue
                    
                    modalities = sensor.modalities
                    if "depth_linear" in modalities:
                        self.depth_sensor_info[sensor_name] = {
                            "uvk_inv": compute_uvk_inv(K=sensor.intrinsic_matrix.clone().float().unsqueeze(0), h=sensor.image_height, w=sensor.image_width),
                            "sensor_name": sensor_name,
                            "robot_name": None,  # External sensors are not associated with a specific robot
                        }
        
        if len(self.depth_sensor_info) == 0:
            print("Warning: No sensors with depth_linear modality found. Point cloud generation will be disabled.")

    @classmethod
    def get_lerobot_obs_mapping(cls, env, use_videos=True, num_points=1024, include_modalities=None, num_points_obj=128, num_objects=1):
        """
        Extended observation mapping that includes point cloud features.
        
        Supports multiple robots - sensor names are prefixed with robot name for disambiguation.
        
        Args:
            env: The environment
            use_videos (bool): Whether to use videos for image data
            num_points (int): Number of points in point cloud
            include_modalities (None or list of str): If specified, only include these modalities.
                If None, all modalities are included. "pointcloud" can be included to add point cloud features.
            num_points_obj (int): Number of points to sample for each object in point cloud
            num_objects (int): Number of objects to sample in point cloud
        Returns:
            tuple: (obs_mapping, obs_features) dictionaries
        """
        # Get base observation mapping (with modality filtering)
        obs_mapping, obs_features = LeRobotPlaybackWrapper.get_lerobot_obs_mapping(
            env=env, 
            use_videos=use_videos,
            include_modalities=include_modalities,
        )
        
        n_robots = len(env.robots)
        
        # Check if pointcloud should be included
        pointcloud_included = include_modalities is None or any("pointcloud" in mod for mod in include_modalities)
        
        if pointcloud_included:
            # Add point cloud features for each robot's depth sensors
            for robot in env.robots:
                robot_prefix = f"{robot.name}_" if n_robots > 1 else ""
                
                for sensor_name, sensor in robot.sensors.items():
                    if isinstance(sensor, VisionSensor) and "depth_linear" in sensor.modalities:
                        remapped_name = "_".join(sensor_name.split(":")[1:]).lower()
                        # Add robot prefix for multi-robot disambiguation
                        sensor_key = f"{robot_prefix.lower()}{remapped_name}"
                        pc_feature_name = f"observation.pointcloud.{sensor_key}"
                        obs_features[pc_feature_name] = {
                            "dtype": "float32",
                            "shape": (num_points, 3),
                            "names": ["num_points", "xyz"],
                        }
            
            # Check external sensors
            if env.external_sensors is not None:
                for sensor_name, sensor in env.external_sensors.items():
                    if isinstance(sensor, VisionSensor) and "depth_linear" in sensor.modalities:
                        pc_feature_name = f"observation.pointcloud.{sensor_name}"
                        obs_features[pc_feature_name] = {
                            "dtype": "float32",
                            "shape": (num_objects, num_points_obj, 3),
                            "names": ["num_objects", "num_points_obj", "xyz"],
                        }
        
        return obs_mapping, obs_features

    def create_dataset(self, output_path, env, overwrite=True):
        """
        Extended dataset creation that adds point cloud modality info and features.
        
        This method overrides the parent to:
        1. First validate sensor modalities to populate depth_sensor_info
        2. Create the base dataset with extended features including point clouds
        3. Add point cloud modality metadata to modality.json
        
        Supports multiple robots - sensor and proprioception names are prefixed for disambiguation.
        """
        import os
        import json
        import shutil
        
        from omnigibson.learning.utils.lerobot_utils import OmniGibsonLeRobotV2Dataset, OmniGibsonLeRobotV3Dataset
        
        # Sanity checks
        assert (
            output_path == self.lerobot_dataset_kwargs["repo_id"]
        ), f"Expected LeRobot repo_id path ({self.lerobot_dataset_kwargs['repo_id']}) to match output_path ({output_path})!"

        abs_output_path = f"{self.lerobot_dataset_kwargs['root']}"

        if os.path.exists(abs_output_path):
            if overwrite:
                shutil.rmtree(abs_output_path)
            else:
                raise ValueError(f"Found pre-existing LeRobot dataset at: {abs_output_path}")

        # Support multiple robots
        n_robots = len(env.robots)
        assert n_robots >= 1, "At least one robot must be present in the environment!"

        modality_info = {
            "annotation": {
                "language.language_instruction": {},
                "language.language_instruction_2": {},
                "language.language_instruction_3": {},
            },
        }

        # Add video modality info (filtered by include_modalities if specified)
        video_modality_info = dict()
        for i, (sensor_name, sensor) in enumerate(env.external_sensors.items()):
            if isinstance(sensor, VisionSensor):
                for mod in ["rgb", "depth_linear"]:
                    if mod in sensor.modalities:
                        # Skip if include_modalities is specified and this modality is not included
                        if self.include_modalities is not None:
                            if not any(inc_mod in mod for inc_mod in self.include_modalities):
                                continue
                        mod_name = f"observation.{mod}.{sensor_name}"
                        key = f"exterior_image_{i}_{mod}"
                        video_modality_info[key] = {
                            "type": mod,
                            "original_key": mod_name,
                        }
        
        # Add video modality info for each robot's sensors
        robot_sensor_idx = 0
        for robot_idx, robot in enumerate(env.robots):
            robot_prefix = f"{robot.name}_" if n_robots > 1 else ""
            for i, (sensor_name, sensor) in enumerate(robot.sensors.items()):
                if isinstance(sensor, VisionSensor):
                    for mod in ["rgb", "depth_linear"]:
                        if mod in sensor.modalities:
                            # Skip if include_modalities is specified and this modality is not included
                            if self.include_modalities is not None:
                                if not any(inc_mod in mod for inc_mod in self.include_modalities):
                                    continue
                            remapped_sensor_name = "_".join(sensor_name.split(":")[1:]).lower()
                            # Add robot prefix for multi-robot disambiguation
                            obs_name = f"{robot_prefix.lower()}{remapped_sensor_name}"
                            mod_name = f"observation.{mod}.{obs_name}"
                            key = f"robot{robot_idx}_image_{i}_{mod}" if n_robots > 1 else f"robot_image_{robot_sensor_idx}_{mod}"
                            video_modality_info[key] = {
                                "type": mod,
                                "original_key": mod_name,
                            }
                            robot_sensor_idx += 1
        modality_info["video" if self.use_videos else "image"] = video_modality_info

        # Handle multi action representation if enabled
        if self.include_multi_action_representation:
            self.controller_action_start_idxs = dict()
            cmd_start_idx = 0
            for robot in env.robots:
                robot_prefix = f"{robot.name}_" if n_robots > 1 else ""
                for name, controller in robot.controllers.items():
                    controller_key = f"{robot_prefix}{name}"
                    self.controller_action_start_idxs[controller_key] = cmd_start_idx
                    cmd_start_idx += controller.command_dim
            # Compute multi-action shape across all robots
            action_shape = self._compute_multi_action_shape_all_robots(env.robots)
        else:
            # Compute total action shape across all robots
            total_action_dim = sum(env.action_space[robot.name].shape[0] for robot in env.robots)
            action_shape = (total_action_dim,)

        # Extract relevant info from original source env config
        config = json.loads(self.input_hdf5["data"].attrs["config"])

        # Create LeRobot dataset, define features to store
        features = {
            "action": {
                "dtype": "float32",
                "shape": action_shape,
                "names": ["action"],
            },
            "next.reward": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["reward"],
            },
            "next.terminated": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["done"],
            },
            "next.truncated": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["done"],
            },
            "annotation.language.language_instruction": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["language_instruction"],
            },
            "annotation.language.language_instruction_2": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["language_instruction_2"],
            },
            "annotation.language.language_instruction_3": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["language_instruction_3"],
            },
        }

        # Get observation mapping with point cloud features (filtered by include_modalities)
        obs_mapping, obs_features = self.get_lerobot_obs_mapping(
            env=env, 
            use_videos=self.use_videos,
            num_points=self.num_points,
            include_modalities=self.include_modalities,
            num_points_obj=self.num_points_obj,
            num_objects=self.num_objs,
        )
        features.update(obs_features)

        # Get observation mapping with ALL modalities (no filtering) -- needed for point cloud generation
        # which requires depth_linear and seg_semantic even if they're not in the final dataset
        obs_mapping_full, _ = self.get_lerobot_obs_mapping(
            env=env, 
            use_videos=self.use_videos,
            num_points=self.num_points,
            include_modalities=None,  # No filtering for full mapping
            num_points_obj=self.num_points_obj,
            num_objects=self.num_objs,
        )

        # Create the dataset
        if self.lerobot_version == "v2.1":
            dataset_cls = OmniGibsonLeRobotV2Dataset
        elif self.lerobot_version == "v3.0":
            dataset_cls = OmniGibsonLeRobotV3Dataset
        else:
            raise ValueError(f"Got invalid lerobot version: {self.lerobot_version}")
            
        self.dataset = dataset_cls.create(
            fps=config["env"]["action_frequency"],
            use_videos=self.use_videos,
            features=features,
            **self.lerobot_dataset_kwargs,
        )
        self.obs_mapping = obs_mapping
        self.obs_mapping_full = obs_mapping_full

        # Store proprio shape mapping for each robot (if proprio is included)
        # For multiple robots, concatenate all proprio observations with continuous indices
        proprio_included = self.include_modalities is None or any("proprio" in mod for mod in self.include_modalities)
        if proprio_included:
            proprio_shape_mapping = dict()
            idx = 0  # Continuous index across all robots
            for robot in env.robots:
                robot_prefix = f"{robot.name}_" if n_robots > 1 else ""
                proprio_dict = robot._get_proprioception_dict()
                for obs_key in robot.proprio_obs:
                    obs_dim = len(proprio_dict[obs_key])
                    proprio_key = f"{robot_prefix}{obs_key}" if n_robots > 1 else obs_key
                    proprio_shape_mapping[proprio_key] = {
                        "start": idx,
                        "end": idx + obs_dim,
                    }
                    idx += obs_dim
            modality_info["state"] = proprio_shape_mapping

        # Render to avoid degen intrinsic matrices
        for _ in range(10):
            og.sim.render()

        # Add camera intrinsics
        cam_intrinsics = dict()
        for sensor_name, sensor in env.external_sensors.items():
            if isinstance(sensor, VisionSensor):
                K = sensor.intrinsic_matrix.cpu()
                cam_intrinsics[sensor_name] = K.numpy().tolist()
        
        # Add camera intrinsics for each robot's sensors
        for robot in env.robots:
            robot_prefix = f"{robot.name}_" if n_robots > 1 else ""
            for sensor_name, sensor in robot.sensors.items():
                if isinstance(sensor, VisionSensor):
                    sensor_name_remapped = "_".join(sensor_name.split(":")[1:]).lower()
                    cam_key = f"{robot_prefix.lower()}{sensor_name_remapped}"
                    K = sensor.intrinsic_matrix.cpu()
                    cam_intrinsics[cam_key] = K.numpy().tolist()
        self.dataset.set_omnigibson_metadata(key="cam_intrinsics", value=cam_intrinsics)

        # Add point cloud modality info (if pointcloud is included)
        pointcloud_included = self.include_modalities is None or any("pointcloud" in mod for mod in self.include_modalities)
        pointcloud_modality_info = {}
        if pointcloud_included:
            for sensor_name in self.depth_sensor_info.keys():
                pc_name = f"observation.pointcloud.{sensor_name}"
                pointcloud_modality_info[sensor_name] = {
                    "type": "pointcloud",
                    "original_key": pc_name,
                    "num_points_obj": self.num_points_obj,
                    "num_objects": self.num_objs,
                    "bounds": self.pc_bounds,
                }
        modality_info["pointcloud"] = pointcloud_modality_info

        # Update obs_mapping with point cloud entries (if pointcloud is included)
        if pointcloud_included:
            for sensor_name in self.depth_sensor_info.keys():
                pc_key = f"pointcloud::{sensor_name}"
                pc_feature_name = f"observation.pointcloud.{sensor_name}"
                self.obs_mapping[pc_key] = pc_feature_name

        # Write modality data
        os.makedirs(os.path.join(abs_output_path, "meta"), exist_ok=True)
        with open(os.path.join(abs_output_path, "meta", "modality.json"), "w+") as f:
            json.dump(modality_info, f, indent=4)
    
    def _compute_multi_action_shape(self, robot):
        """
        Compute the action shape for multi-action representation for a single robot.
        
        Args:
            robot: The robot instance
            
        Returns:
            tuple: Shape of the action tensor
        """
        idx = 0
        for arm in robot.arm_names:
            arm_name = f"arm_{arm}"
            arm_controller = robot.controllers[arm_name]
            # eef_pos (3) + eef_aa (3) + delta_eef_pos (3) + delta_eef_aa (3) + 
            # gripper_pos (1) + joint_pos (control_dim) + delta_joint_pos (control_dim)
            idx += 3 + 3 + 3 + 3 + 1 + arm_controller.control_dim + arm_controller.control_dim
        return (idx,)
    
    def _compute_multi_action_shape_all_robots(self, robots):
        """
        Compute the action shape for multi-action representation across all robots.
        
        Args:
            robots: List of robot instances
            
        Returns:
            tuple: Shape of the action tensor for all robots combined
        """
        total_idx = 0
        for robot in robots:
            for arm in robot.arm_names:
                arm_name = f"arm_{arm}"
                arm_controller = robot.controllers[arm_name]
                # eef_pos (3) + eef_aa (3) + delta_eef_pos (3) + delta_eef_aa (3) + 
                # gripper_pos (1) + joint_pos (control_dim) + delta_joint_pos (control_dim)
                total_idx += 3 + 3 + 3 + 3 + 1 + arm_controller.control_dim + arm_controller.control_dim
        return (total_idx,)

    def _process_obs(self, obs, info):
        """
        Extended observation processing that generates point clouds from depth and segmentation.
        
        This method processes all observations (including depth_linear and seg_semantic for point cloud
        generation), then filters the output to only include modalities specified in include_modalities.
        
        Args:
            obs (dict): Raw observations from the environment
            info (dict): Additional observation info (contains segmentation label mappings)
            
        Returns:
            dict: Processed observations including point clouds (filtered by include_modalities)
        """
        # First, get the base lerobot-formatted observations using FULL obs_mapping
        # We need all modalities (especially depth_linear and seg_semantic) for point cloud generation
        original_obs_mapping = self.obs_mapping
        original_include_modalities = self.include_modalities
        
        # Temporarily use full mapping to get all observations
        self.obs_mapping = self.obs_mapping_full
        self.include_modalities = None  # No filtering during initial processing
        obs_full = super()._process_obs(obs, info)
        
        # Restore original mappings
        self.obs_mapping = original_obs_mapping
        self.include_modalities = original_include_modalities
        
        # Generate point clouds for each depth sensor
        for sensor_name, sensor_info in self.depth_sensor_info.items():
            depth_key = f"observation.depth_linear.{sensor_name}"
            seg_key = f"observation.seg_semantic.{sensor_name}"
            robot2cam_key = f"observation.robot2cam_pose.{sensor_name}"
            pc_key = f"observation.pointcloud.{sensor_name}"
            
            # Check if depth observation is available
            if depth_key not in obs_full:
                # Try alternative naming for external sensors
                alt_depth_key = None
                for key in obs_full.keys():
                    if "depth_linear" in key and sensor_name in key:
                        alt_depth_key = key
                        break
                if alt_depth_key is None:
                    print(f"Warning: Depth observation {depth_key} not found. Skipping point cloud for {sensor_name}.")
                    obs_full[pc_key] = th.zeros((self.num_objs, self.num_points_obj, 3), device="cpu")
                    continue
                depth_key = alt_depth_key
            
            # Get depth data - remove channel dimension if present
            depth = obs_full[depth_key]
            if depth.dim() == 3:
                depth = depth.squeeze(-1)  # Remove channel dim: (H, W, 1) -> (H, W)
            
            # Get segmentation data if available
            seg_mask = None
            id2labels = None
            if self.include_categories is not None:
                if seg_key in obs_full:
                    seg_mask = obs_full[seg_key]
                    if seg_mask.dim() == 3:
                        seg_mask = seg_mask.squeeze(-1)
                    
                    # Get id2labels from info if available
                    # The info structure varies, so we try different paths
                    id2labels = self._get_seg_id_labels(info["obs_info"], sensor_info["sensor_name"])
            
            # Get robot2cam transform
            if robot2cam_key in obs_full:
                robot2cam_pose = obs_full[robot2cam_key]  # (7,) - pos + quat
                robot2cam_tf = T.pose2mat((robot2cam_pose[:3], robot2cam_pose[3:]))
            else:
                # Fallback: use identity transform
                robot2cam_tf = th.eye(4)
            
            # Get intrinsic matrix
            uvk_inv = sensor_info["uvk_inv"]
            
            # Generate point cloud
            if seg_mask is not None:
                point_cloud = pointcloud_from_depth_segmented(
                    depth=depth,
                    seg_mask=seg_mask,
                    uvk_inv=uvk_inv,
                    cam_to_img_tf=self.CAM_TO_IMG_TF,
                    robot2cam_tf=robot2cam_tf,
                    id2labels=id2labels,
                    num_points=self.num_points,
                    include_categories=self.include_categories,
                    device="cuda", #depth.device if hasattr(depth, 'device') else "cpu",
                    sensor_name=sensor_name,
                    **self.pc_bounds,
                )
            else:
                # No segmentation, just use depth with all pixels
                point_cloud = pointcloud_from_depth_segmented(
                    depth=depth,
                    seg_mask=th.ones_like(depth, dtype=th.int64),  # Dummy mask
                    uvk_inv=uvk_inv,
                    cam_to_img_tf=self.CAM_TO_IMG_TF,
                    robot2cam_tf=robot2cam_tf,
                    id2labels=None,
                    num_points=self.num_points,
                    include_categories=None,
                    device="cuda", #depth.device if hasattr(depth, 'device') else "cpu",
                    sensor_name=sensor_name,
                    **self.pc_bounds,
                )
            
            # Store point cloud in observations (move to CPU for dataset storage)
            obs_full[pc_key] = point_cloud.view(self.num_objs, self.num_points_obj, 3).cpu()
        
        # Filter the observations to only include keys in self.obs_mapping (filtered by include_modalities)
        # Also include point cloud keys if pointcloud modality is included
        if self.include_modalities is not None:
            # Get the set of allowed lerobot observation names from obs_mapping
            allowed_keys = set(self.obs_mapping.values())
            
            # Also add point cloud keys if pointcloud is in include_modalities
            pointcloud_included = any("pointcloud" in mod for mod in self.include_modalities)
            if pointcloud_included:
                for sensor_name in self.depth_sensor_info.keys():
                    allowed_keys.add(f"observation.pointcloud.{sensor_name}")
            
            # Filter obs_full to only include allowed keys
            filtered_obs = {k: v for k, v in obs_full.items() if k in allowed_keys}
            return filtered_obs
        
        print(f"gripper qpos norm: {obs_full['observation.state'][-1]}")
        return obs_full
    
    def _get_seg_id_labels(self, info, sensor_name):
        """
        Extract segmentation ID to label mapping from observation info.
        
        Args:
            info (dict): Observation info dictionary
            sensor_name (str): Name of the sensor
            
        Returns:
            dict or None: ID to label mapping, or None if not found
        """
        # Try different paths to find the segmentation labels
        # Path 1: info[robot_name][sensor_name]["seg_semantic"]
        # Path 2: info["external"][sensor_name]["seg_semantic"]
        
        if info is None:
            return None
        
        # Try robot sensors first
        for robot_name, robot_info in info.items():
            if not isinstance(robot_info, dict):
                continue
            for sens_name, sens_info in robot_info.items():
                if sensor_name in sens_name and isinstance(sens_info, dict):
                    if "seg_semantic" in sens_info:
                        return sens_info["seg_semantic"]
        
        # Try external sensors
        if "external" in info:
            external_info = info["external"]
            if isinstance(external_info, dict):
                for sens_name, sens_info in external_info.items():
                    if sensor_name in sens_name and isinstance(sens_info, dict):
                        if "seg_semantic" in sens_info:
                            return sens_info["seg_semantic"]
        
        return None

    def process_traj_to_dataset(self, traj_data, nested_keys=("obs",)):
        """
        Extended trajectory processing that includes point cloud data.
        Also applies data transformations (gripper inversion/scaling) if configured.
        """
        # Apply data transformations to actions and states before saving
        # This modifies traj_data in place, affecting only what gets saved to disk
        # The environment execution uses the original data
        invert_gripper_action = getattr(self, 'invert_gripper_action', False)
        scale_gripper_state = getattr(self, 'scale_gripper_state', 1.0)
        
        if invert_gripper_action or scale_gripper_state != 1.0:
            for step_data in traj_data:
                # Transform action (gripper is last element)
                if invert_gripper_action and "action" in step_data:
                    action = step_data["action"]
                    if hasattr(action, '__len__') and len(action) > 0:
                        if isinstance(action, th.Tensor):
                            step_data["action"] = action.clone()
                            step_data["action"][-1] = 1.0 - step_data["action"][-1]
                        else:
                            import numpy as np
                            step_data["action"] = np.array(action).copy()
                            step_data["action"][-1] = 1.0 - step_data["action"][-1]
                
                # Transform observation.state (gripper is last element)
                if scale_gripper_state != 1.0 and "obs" in step_data:
                    obs = step_data["obs"]
                    # The obs dict may contain various keys; look for state-related ones
                    for key in list(obs.keys()):
                        if "state" in key.lower() or "proprio" in key.lower() or "gripper" in key.lower():
                            state = obs[key]
                            if hasattr(state, '__len__') and len(state) > 0:
                                if isinstance(state, th.Tensor):
                                    obs[key] = state.clone()
                                    obs[key][-1] = obs[key][-1] * scale_gripper_state
                                else:
                                    import numpy as np
                                    obs[key] = np.array(state).copy()
                                    obs[key][-1] = obs[key][-1] * scale_gripper_state
        
        # Process trajectory with point cloud observations included
        # The parent class handles the actual writing
        super().process_traj_to_dataset(traj_data, nested_keys)
