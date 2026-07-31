# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file (the abstract robot interface) are adapted from MolmoSpaces
# (https://github.com/allenai/molmospaces), Copyright 2026 Allen Institute for AI,
# licensed under the Apache License, Version 2.0.
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.

"""
Utilities for generating demonstration data from object-centric waypoints.

This module provides:
- Object pose randomization
- CuRobo motion planning for freespace navigation
- Waypoint replay with IK control
- Joint-space IK solver
"""

import json
from typing import Dict, List, Tuple, Optional, Union

import h5py
import numpy as np
import torch as th
import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.curobo import CuRoboMotionGenerator, CuRoboEmbodimentSelection
from omnigibson.controllers import ControlType


def init_curobo(robot, batch_size: int = 2) -> CuRoboMotionGenerator:
    """
    Initialize CuRobo motion generator.

    Args:
        robot: Robot object
        batch_size: Batch size for CuRobo

    Returns:
        CuRoboMotionGenerator instance
    """
    print("\n[CuRobo] Initializing motion generator...")
    print("[CuRobo] Note: First-time initialization may take 5-15 minutes for JIT compilation")

    cmg = CuRoboMotionGenerator(
        robot=robot,
        batch_size=batch_size,
        debug=False,
        use_cuda_graph=True,
        collision_activation_distance=0.075,
        use_default_embodiment_only=True,
    )

    print("[CuRobo] Initialization complete!")
    return cmg


def plan_to_pose(
    cmg: CuRoboMotionGenerator,
    robot,
    target_pos_world: th.Tensor,
    target_quat_world: th.Tensor,
    max_attempts: int = 100,
    timeout: float = 60.0,
    attached_obj = None,
    attached_obj_scale = None,
) -> Optional[th.Tensor]:
    """
    Plan a collision-free joint-space trajectory to target EEF pose using CuRobo.

    Args:
        cmg: CuRoboMotionGenerator instance
        robot: Robot object
        target_pos_world: Target EEF position in world frame [3]
        target_quat_world: Target EEF quaternion in world frame [4]
        max_attempts: Maximum planning attempts
        timeout: Planning timeout in seconds
        attached_obj: Dict mapping EEF link name to attached object's root link (for collision checking)
                     Example: {robot.eef_link_names[robot.default_arm]: obj.root_link}
        attached_obj_scale: Dict mapping EEF link name to scale factor for attached object collision checking
                           Example: {robot.eef_link_names[robot.default_arm]: 0.8}

    Returns:
        Joint trajectory [T, n_joints] if successful, None otherwise
    """
    # Ensure correct shape
    if len(target_pos_world.shape) == 1:
        target_pos_world = target_pos_world.unsqueeze(0)  # [1, 3]
    if len(target_quat_world.shape) == 1:
        target_quat_world = target_quat_world.unsqueeze(0)  # [1, 4]

    # Get EEF link name
    eef_link_name = robot.eef_link_names[robot.default_arm]

    # Create target pose dict for CuRobo
    target_pos_dict = {eef_link_name: target_pos_world}
    target_quat_dict = {eef_link_name: target_quat_world}

    # Compute trajectory in joint space
    # successes, traj_paths = cmg.compute_trajectories(
    successes, traj_paths = cmg.compute_trajectories(
        target_pos=target_pos_dict,
        target_quat=target_quat_dict,
        is_local=False,  # Using world frame coordinates
        max_attempts=max_attempts,
        timeout=timeout,
        ik_fail_return=50,
        enable_finetune_trajopt=True,
        finetune_attempts=1,
        return_full_result=False,
        success_ratio=1.0,
        attached_obj=attached_obj,
        attached_obj_scale=attached_obj_scale,
    )

    if not successes[0].item():
        return None

    # Extract joint trajectory from CuRobo path (returns JointState object)
    # Use path_to_joint_trajectory to get the actual joint positions tensor
    joint_trajectory = cmg.path_to_joint_trajectory(traj_paths[0]).cpu().float()  # [T, n_joints]

    # TODO: properly smooth out entire trajectory
    joint_trajectory = cmg.add_linearly_interpolated_waypoints(traj=joint_trajectory, max_inter_dist=0.02)

    return joint_trajectory


def transform_waypoints_to_robot_frame(
    waypoints: List[Dict],
    ref_obj,
    robot,
    subtask_type: str = "unknown"
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """
    Transform waypoints from object frame to robot frame.

    Args:
        waypoints: List of waypoint dicts with 'eef_pos_obj', 'eef_quat_obj', 'gripper_action'
        ref_obj: Reference object
        robot: Robot object

    Returns:
        Tuple of (eef_pos_robot, eef_quat_robot, gripper_actions)
        - eef_pos_robot: [n, 3] EEF positions in robot frame
        - eef_quat_robot: [n, 4] EEF quaternions in robot frame
        - gripper_actions: [n] gripper actions
    """
    n_waypoints = len(waypoints)

    # Get current object and robot poses
    obj_pos, obj_quat = ref_obj.get_position_orientation()
    robot_pos, robot_quat = robot.get_position_orientation()

    # Convert to tensors if needed
    if not isinstance(obj_pos, th.Tensor):
        obj_pos = th.tensor(obj_pos, dtype=th.float32)
    if not isinstance(obj_quat, th.Tensor):
        obj_quat = th.tensor(obj_quat, dtype=th.float32)
    if not isinstance(robot_pos, th.Tensor):
        robot_pos = th.tensor(robot_pos, dtype=th.float32)
    if not isinstance(robot_quat, th.Tensor):
        robot_quat = th.tensor(robot_quat, dtype=th.float32)

    # Build transformation matrices
    obj_T_world = T.pose2mat((obj_pos, obj_quat))  # Object → World
    robot_T_world = T.pose2mat((robot_pos, robot_quat))  # Robot → World
    world_T_robot = T.pose_inv(robot_T_world)  # World → Robot

    eef_pos_robot_list = []
    eef_quat_robot_list = []
    gripper_actions = []

    for wp in waypoints:
        # Get waypoint pose in object frame
        eef_pos_obj = th.tensor(wp['eef_pos_obj'], dtype=th.float32)
        eef_quat_obj = th.tensor(wp['eef_quat_obj'], dtype=th.float32)

        # Transform: Object frame → World frame → Robot frame
        eef_T_obj = T.pose2mat((eef_pos_obj, eef_quat_obj))
        eef_T_world = obj_T_world @ eef_T_obj
        eef_T_robot = world_T_robot @ eef_T_world

        # Extract pose in robot frame
        eef_pos_robot, eef_quat_robot = T.mat2pose(eef_T_robot)

        eef_pos_robot_list.append(eef_pos_robot)
        eef_quat_robot_list.append(eef_quat_robot)
        gripper_actions.append(wp['gripper_action'])

    # Stack into tensors
    eef_pos_robot = th.stack(eef_pos_robot_list, dim=0)  # [n, 3]
    eef_quat_robot = th.stack(eef_quat_robot_list, dim=0)  # [n, 4]
    gripper_actions = th.tensor(gripper_actions, dtype=th.float32)  # [n]

    return eef_pos_robot, eef_quat_robot, gripper_actions


def execute_trajectory_joint(actions: th.Tensor, env):
    """
    Execute joint-space trajectory using pre-assembled actions.

    Args:
        actions: [n, action_dim] pre-assembled actions (joint positions + gripper)
        env: Environment
    """
    assert actions.ndim == 2, f"Expected 2D actions array, got shape {actions.shape}"

    for i in range(actions.shape[0]):
        action = actions[i].cpu().numpy() if isinstance(actions[i], th.Tensor) else actions[i]
        env.step(action)


def execute_trajectory_delta(actions: th.Tensor, env):
    """
    Execute trajectory using pre-assembled delta IK actions.

    Args:
        actions: [n, action_dim] pre-assembled delta IK actions (delta_pos + delta_aa + gripper)
        env: Environment
    """
    assert actions.ndim == 2, f"Expected 2D actions array, got shape {actions.shape}"

    for i in range(actions.shape[0]):
        action = actions[i].cpu().numpy() if isinstance(actions[i], th.Tensor) else actions[i]
        env.step(action)


def apply_eef_z_offset(
    eef_pos: th.Tensor,
    eef_quat: th.Tensor,
    z_offset: float
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Apply a z-offset in the local EEF frame to the EEF pose(s).

    Can handle both single poses [3]/[4] and batched poses [n, 3]/[n, 4].

    Args:
        eef_pos: EEF position(s) [3] or [n, 3]
        eef_quat: EEF quaternion(s) [4] or [n, 4]
        z_offset: Z offset in local EEF frame (meters)

    Returns:
        Tuple of (new_eef_pos, eef_quat) - Updated EEF pose(s) with offset applied
    """
    if z_offset == 0.0:
        return eef_pos, eef_quat

    if not isinstance(eef_pos, th.Tensor):
        eef_pos = th.tensor(eef_pos, dtype=th.float32)
    if not isinstance(eef_quat, th.Tensor):
        eef_quat = th.tensor(eef_quat, dtype=th.float32)

    # Handle single vs batched input
    is_batched = eef_pos.dim() == 2

    offset_local = th.tensor([0.0, 0.0, z_offset], dtype=th.float32)
    eef_rot_mat = T.quat2mat(eef_quat)
    offset_transformed = eef_rot_mat @ offset_local
    new_eef_pos = eef_pos + offset_transformed

    # if not is_batched:
    #     # Single pose
    #     offset_local = th.tensor([0.0, 0.0, z_offset], dtype=th.float32)
    #     eef_rot_mat = T.quat2mat(eef_quat)
    #     offset_transformed = eef_rot_mat @ offset_local
    #     new_eef_pos = eef_pos + offset_transformed
    # else:
    #     # Batched poses
    #     n = eef_pos.shape[0]
    #     offset_local = th.tensor([0.0, 0.0, z_offset], dtype=th.float32)
    #     eef_
    #     new_eef_pos = eef_pos.clone()
    #     for i in range(n):
    #         eef_rot_mat = T.quat2mat(eef_quat[i])
    #         offset_transformed = eef_rot_mat @ offset_local
    #         new_eef_pos[i] = eef_pos[i] + offset_transformed

    return new_eef_pos, eef_quat


def _estimate_boundary_conditions(
    positions: th.Tensor,
    action_frequency: float,
    n_samples: int = 5,
    from_end: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Estimate travel direction and velocity from boundary waypoints of a segment.

    Fits a line through the boundary points via SVD (principal component) to get
    the travel direction, and computes mean per-step displacement for velocity.

    Args:
        positions: [N, 3] waypoint positions in robot frame
        action_frequency: Action frequency in Hz
        n_samples: Number of boundary waypoints to use
        from_end: If True use last n_samples points, otherwise first n_samples

    Returns:
        (unit_direction [3], velocity_m_per_s)
    """
    pts = positions.detach().cpu().numpy().astype(np.float64) if isinstance(positions, th.Tensor) else np.asarray(positions, dtype=np.float64)
    n_avail = min(n_samples, len(pts))
    if n_avail < 2:
        return np.zeros(3, dtype=np.float64), 0.0

    pts = pts[-n_avail:] if from_end else pts[:n_avail]

    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    direction = Vt[0].copy()

    if np.dot(direction, pts[-1] - pts[0]) < 0:
        direction = -direction

    nrm = np.linalg.norm(direction)
    direction = direction / nrm if nrm > 1e-8 else np.zeros(3, dtype=np.float64)

    step_dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    velocity = float(np.mean(step_dists)) * action_frequency

    return direction, velocity


def compute_linear_freespace_trajectory(
    robot,
    target_pos_robot: th.Tensor,
    target_quat_robot: th.Tensor,
    freespace_velocity: float,
    action_frequency: float,
    smooth: bool = False,
    preceding_positions: Optional[th.Tensor] = None,
    preceding_quats: Optional[th.Tensor] = None,
    waypoint_positions: Optional[th.Tensor] = None,
    waypoint_quats: Optional[th.Tensor] = None,
    n_boundary_waypoints: int = 5,
    velocity_ramp_fraction: float = 0.25,
    spline_tangent_scale: float = 1.0,
    n_spline_samples: int = 1000,
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Compute freespace trajectory from current EEF pose to target in robot frame.

    In default mode, generates a straight-line trajectory with uniform velocity.
    In smooth mode, uses a cubic Hermite spline whose boundary tangents match the
    travel direction of the preceding / proceeding waypoint segments, with a
    three-phase velocity profile (ramp-up from preceding velocity, cruise at
    freespace_velocity, ramp-down to proceeding velocity).

    Args:
        robot: Robot with ``get_relative_eef_pose`` method
        target_pos_robot: Target EEF position in robot frame [3]
        target_quat_robot: Target EEF quaternion in robot frame [4]
        freespace_velocity: Cruise EEF velocity in m/s
        action_frequency: Action frequency in Hz
        smooth: Enable spline-based smoothing with velocity profiling
        preceding_positions: [N, 3] preceding segment waypoint positions (robot frame)
        preceding_quats: [N, 4] preceding segment waypoint quaternions
        waypoint_positions: [N, 3] proceeding segment waypoint positions (robot frame)
        waypoint_quats: [N, 4] proceeding segment waypoint quaternions
        n_boundary_waypoints: How many waypoints to sample from each boundary
            for direction / velocity estimation
        velocity_ramp_fraction: Fraction of arc length used for velocity ramp at
            each end (0.25 means first 25 % ramps up, last 25 % ramps down)
        spline_tangent_scale: Multiplier for Hermite tangent magnitudes; larger
            values produce wider curves, smaller values approach a straight line
        n_spline_samples: Internal resolution for arc-length computation

    Returns:
        (positions [W, 3], quaternions [W, 4]) in robot frame
    """
    curr_pos_robot, curr_quat_robot = robot.get_relative_eef_pose(arm="default")
    if not isinstance(curr_pos_robot, th.Tensor):
        curr_pos_robot = th.tensor(curr_pos_robot, dtype=th.float32)
    if not isinstance(curr_quat_robot, th.Tensor):
        curr_quat_robot = th.tensor(curr_quat_robot, dtype=th.float32)
    if not isinstance(target_pos_robot, th.Tensor):
        target_pos_robot = th.tensor(target_pos_robot, dtype=th.float32)
    if not isinstance(target_quat_robot, th.Tensor):
        target_quat_robot = th.tensor(target_quat_robot, dtype=th.float32)

    total_distance = th.norm(target_pos_robot - curr_pos_robot).item()
    travel_time = total_distance / freespace_velocity if freespace_velocity > 0 else 0.0
    n_waypoints = max(2, int(np.ceil(travel_time * action_frequency)))

    if smooth:
        p0 = curr_pos_robot.detach().cpu().numpy().astype(np.float64)
        p1 = target_pos_robot.detach().cpu().numpy().astype(np.float64)
        chord = p1 - p0
        chord_len = np.linalg.norm(chord)

        if chord_len < 1e-6:
            return curr_pos_robot.unsqueeze(0), curr_quat_robot.unsqueeze(0)

        chord_dir = chord / chord_len

        # --- estimate boundary directions & velocities ---
        if preceding_positions is not None and len(preceding_positions) >= 2:
            dir_start, vel_start = _estimate_boundary_conditions(
                preceding_positions, action_frequency, n_boundary_waypoints, from_end=True)
        else:
            dir_start, vel_start = chord_dir.copy(), freespace_velocity

        if waypoint_positions is not None and len(waypoint_positions) >= 2:
            dir_end, vel_end = _estimate_boundary_conditions(
                waypoint_positions, action_frequency, n_boundary_waypoints, from_end=False)
        else:
            dir_end, vel_end = chord_dir.copy(), freespace_velocity

        # --- cubic Hermite spline ---
        tang_mag = spline_tangent_scale * chord_len
        m0 = dir_start * tang_mag
        m1 = dir_end * tang_mag

        u = np.linspace(0.0, 1.0, n_spline_samples)
        u2, u3 = u * u, u * u * u
        h00 = 2 * u3 - 3 * u2 + 1
        h10 = u3 - 2 * u2 + u
        h01 = -2 * u3 + 3 * u2
        h11 = u3 - u2

        spline = (h00[:, None] * p0 + h10[:, None] * m0
                  + h01[:, None] * p1 + h11[:, None] * m1)

        # --- arc-length parametrisation ---
        seg_lens = np.linalg.norm(np.diff(spline, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
        arc_total = arc[-1]

        if arc_total < 1e-8:
            return curr_pos_robot.unsqueeze(0), curr_quat_robot.unsqueeze(0)

        s_norm = arc / arc_total

        # --- three-phase velocity profile ---
        min_vel = max(freespace_velocity * 0.01, 1e-4)
        ramp = float(np.clip(velocity_ramp_fraction, 0.0, 0.5))

        def _vel(s):
            if s < ramp:
                frac = s / ramp if ramp > 0 else 1.0
                v = vel_start + (freespace_velocity - vel_start) * frac
            elif s < 1.0 - ramp:
                v = freespace_velocity
            else:
                frac = (s - (1.0 - ramp)) / ramp if ramp > 0 else 1.0
                v = freespace_velocity + (vel_end - freespace_velocity) * frac
            return max(v, min_vel)

        # integrate ds / v(s) to obtain elapsed time at each spline sample
        s_mid = 0.5 * (s_norm[:-1] + s_norm[1:])
        v_mid = np.array([_vel(si) for si in s_mid])
        dt_seg = np.diff(arc) / v_mid
        times = np.concatenate([[0.0], np.cumsum(dt_seg)])
        total_time = times[-1]

        # --- resample at action frequency ---
        step_dt = 1.0 / action_frequency
        n_out = max(2, int(np.ceil(total_time / step_dt)) + 1)
        t_query = np.linspace(0.0, total_time, n_out)
        arc_query = np.interp(t_query, times, arc)

        out_pos = np.column_stack([
            np.interp(arc_query, arc, spline[:, 0]),
            np.interp(arc_query, arc, spline[:, 1]),
            np.interp(arc_query, arc, spline[:, 2]),
        ])
        s_query = arc_query / arc_total

        positions = th.from_numpy(out_pos).float()
        s_th = th.from_numpy(s_query).float()

        quaternions = T.quat_slerp(
            curr_quat_robot.unsqueeze(0).expand(n_out, -1),
            target_quat_robot.unsqueeze(0).expand(n_out, -1),
            s_th.unsqueeze(-1),
        )

        return positions, quaternions

    # --- default: straight line with uniform velocity ---
    s_values = th.linspace(0, 1, n_waypoints)

    positions = []
    for s in s_values:
        pos = (1 - s) * curr_pos_robot + s * target_pos_robot
        positions.append(pos)
    positions = th.stack(positions, dim=0)

    quaternions = T.quat_slerp(
        curr_quat_robot.unsqueeze(0).repeat(s_values.shape[0], 1),
        target_quat_robot.unsqueeze(0).repeat(s_values.shape[0], 1),
        s_values.unsqueeze(-1),
    )

    return positions, quaternions


def solve_ik_step(
    robot,
    target_pos_robot: th.Tensor,
    target_quat_robot: th.Tensor,
    gripper_cmd: float
) -> np.ndarray:
    """
    Solve IK for a single waypoint using cvxpy-based quadratic programming.

    This replicates OmniGibson's IKController logic to convert a target EEF pose
    (in robot frame) to joint positions for a JointController.

    Args:
        robot: Robot object
        target_pos_robot: Target EEF position in robot frame [3]
        target_quat_robot: Target EEF quaternion in robot frame [4]
        gripper_cmd: Gripper command (1.0 = open, 0.0 = close)

    Returns:
        Action array for the robot
    """
    import cvxpy as cp
    # Get current EEF pose in robot frame
    curr_pos_robot, curr_quat_robot = robot.get_relative_eef_pose(arm="default")
    if not isinstance(curr_pos_robot, th.Tensor):
        curr_pos_robot = th.tensor(curr_pos_robot, dtype=th.float32)
    if not isinstance(curr_quat_robot, th.Tensor):
        curr_quat_robot = th.tensor(curr_quat_robot, dtype=th.float32)

    # Compute delta position in robot frame
    delta_pos = target_pos_robot - curr_pos_robot

    # Compute delta orientation using orientation_error
    delta_ori = T.orientation_error(T.quat2mat(target_quat_robot), T.quat2mat(curr_quat_robot))

    # Concatenate into error vector
    err = th.cat([delta_pos, delta_ori])

    # Get control dict for Jacobian and joint states
    control_dict = robot.get_control_dict()

    # Get arm controller and indices
    arm_controller = robot.controllers[f"arm_{robot.default_arm}"]
    arm_dof_idx = arm_controller.dof_idx
    manipulation_dof_idx = arm_dof_idx

    # Get Jacobian and joint state
    j_eef = control_dict[f"eef_{robot.default_arm}_jacobian_relative"][:, manipulation_dof_idx]
    q = control_dict["joint_position"][manipulation_dof_idx]
    q_lower_limit = arm_controller._control_limits[ControlType.get_type("position")][0][manipulation_dof_idx]
    q_upper_limit = arm_controller._control_limits[ControlType.get_type("position")][1][manipulation_dof_idx]
    q_dot_lower_limit = arm_controller._control_limits[ControlType.get_type("velocity")][0][manipulation_dof_idx]
    q_dot_upper_limit = arm_controller._control_limits[ControlType.get_type("velocity")][1][manipulation_dof_idx]

    # Solve IK using cvxpy (replicating IKController logic)
    vel_err = err.cpu().numpy() / og.sim.get_physics_dt()
    proportional_gain = 0.5

    n = j_eef.shape[1]
    epsilon = 1e-6
    P = j_eef.T @ j_eef + epsilon * np.eye(j_eef.shape[1])
    r = -proportional_gain * vel_err @ j_eef

    velocity_gain = 0.5
    q_dot_upper_limit_by_joint_limit = velocity_gain * (q_upper_limit - q) / og.sim.get_physics_dt()
    q_dot_lower_limit_by_joint_limit = velocity_gain * (q_lower_limit - q) / og.sim.get_physics_dt()

    q_dot_upper_limit = np.minimum(q_dot_upper_limit, q_dot_upper_limit_by_joint_limit)
    q_dot_lower_limit = np.maximum(q_dot_lower_limit, q_dot_lower_limit_by_joint_limit)

    G = np.vstack([np.eye(n), -np.eye(n)])
    h = np.concatenate([q_dot_upper_limit, -q_dot_lower_limit])

    q_dot = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(q_dot, P) + r.T @ q_dot), [G @ q_dot <= h])

    # Solve QP
    prob.solve()

    if prob.status == "optimal":
        q_dot_val = q_dot.value
        delta_j = q_dot_val * og.sim.get_physics_dt()
        target_joint_pos = q + delta_j
    else:
        # If solver fails, keep current position
        target_joint_pos = q

    # Clip to joint limits
    target_joint_pos = np.clip(target_joint_pos, q_lower_limit + 0.02, q_upper_limit - 0.02)

    # Pad with current gripper joint positions to get full joint vector
    full_joint_pos = control_dict["joint_position"].copy()
    full_joint_pos[manipulation_dof_idx] = target_joint_pos

    # Convert to action using q_to_action (expects full joint positions)
    action = robot.q_to_action(th.tensor(full_joint_pos, dtype=th.float32), controller_name='arm_0')
    gripper_action = gripper_cmd
    action = th.cat([action, th.tensor([gripper_action])], dim=0)

    return action.cpu().numpy() if isinstance(action, th.Tensor) else action


def apply_tcp_action_noise(
    action,
    robot,
    arm_dof_idx,
    robot_action_offset: int,
    noise_cfg: dict,
):
    """
    Apply TCP-bounded action noise to arm joint positions within a flat action array.

    samples noise proportional to the commanded TCP delta, bounded by configurable maximums, 
    then maps it back to joint space via Jacobian least-squares.
    Derived from https://github.com/allenai/molmospaces/blob/main/molmo_spaces/robots/abstract.py

    When the commanded TCP delta is zero (robot not moving), no noise is applied.

    Args:
        action: Flat action array (numpy or torch) for all robots.
        robot: OmniGibson robot object (the controlled robot).
        arm_dof_idx: Arm controller DOF indices (from arm_controller.dof_idx).
        robot_action_offset: Index offset into the flat action array for this robot.
        noise_cfg: Dict with keys: action_scale_factor, rotation_noise_scale,
            max_tcp_position_noise, max_tcp_rotation_noise.

    Returns:
        Copy of action with noised arm joint positions. Same type as input.
    """
    from scipy.stats import truncnorm
    import torch as th

    is_tensor = isinstance(action, th.Tensor)
    action_np = action.cpu().numpy().copy() if is_tensor else action.copy()

    n_arm_joints = len(arm_dof_idx)
    arm_start = robot_action_offset
    arm_end = arm_start + n_arm_joints

    commanded_joint_pos = action_np[arm_start:arm_end].astype(np.float64)

    current_joint_pos = robot.get_joint_positions()[arm_dof_idx]
    if isinstance(current_joint_pos, th.Tensor):
        current_joint_pos = current_joint_pos.cpu().numpy()
    current_joint_pos = current_joint_pos.astype(np.float64)

    joint_delta = commanded_joint_pos - current_joint_pos

    control_dict = robot.get_control_dict()
    J = control_dict[f"eef_{robot.default_arm}_jacobian_relative"][:, arm_dof_idx]
    if isinstance(J, th.Tensor):
        J = J.cpu().numpy()
    J = J.astype(np.float64)

    tcp_delta = J @ joint_delta
    tcp_pos_delta_norm = np.linalg.norm(tcp_delta[:3])

    scale_factor = noise_cfg["action_scale_factor"]
    position_noise_std = scale_factor * tcp_pos_delta_norm
    rotation_noise_std = position_noise_std * noise_cfg["rotation_noise_scale"]

    if position_noise_std > 0:
        pos_bound = noise_cfg["max_tcp_position_noise"] / position_noise_std
        position_noise = truncnorm.rvs(
            -pos_bound, pos_bound, scale=position_noise_std, size=3
        )
    else:
        position_noise = np.zeros(3)

    if rotation_noise_std > 0:
        rot_bound = noise_cfg["max_tcp_rotation_noise"] / rotation_noise_std
        rotation_noise = truncnorm.rvs(
            -rot_bound, rot_bound, scale=rotation_noise_std, size=3
        )
    else:
        rotation_noise = np.zeros(3)

    tcp_noise = np.concatenate([position_noise, rotation_noise])
    joint_noise, _, _, _ = np.linalg.lstsq(J, tcp_noise, rcond=None)

    noisy_joint_pos = commanded_joint_pos + joint_noise

    arm_controller = robot.controllers[f"arm_{robot.default_arm}"]
    q_lower = arm_controller._control_limits[ControlType.get_type("position")][0][arm_dof_idx]
    q_upper = arm_controller._control_limits[ControlType.get_type("position")][1][arm_dof_idx]
    if isinstance(q_lower, th.Tensor):
        q_lower = q_lower.cpu().numpy()
    if isinstance(q_upper, th.Tensor):
        q_upper = q_upper.cpu().numpy()
    noisy_joint_pos = np.clip(noisy_joint_pos, q_lower, q_upper)

    action_np[arm_start:arm_end] = noisy_joint_pos

    if is_tensor:
        return th.from_numpy(action_np).to(action.device, dtype=action.dtype)
    return action_np


def load_waypoints_from_hdf5(hdf5_path: str) -> Dict:
    """
    Load waypoints from HDF5 file.

    Args:
        hdf5_path: Path to waypoints HDF5 file

    Returns:
        Dict mapping demo_id to list of subtask data
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

                # Load metadata
                subtask_data = {
                    'type': subtask_grp.attrs['type'],
                    'signal_frame_idx': subtask_grp.attrs['signal_frame_idx'],
                    'reference_object': json.loads(subtask_grp.attrs['reference_object']),
                    'extraction_params': json.loads(subtask_grp.attrs['extraction_params']),
                }

                # Load placed_object if it exists (for place actions)
                if 'placed_object' in subtask_grp.attrs:
                    subtask_data['placed_object'] = json.loads(subtask_grp.attrs['placed_object'])

                # Load waypoint arrays
                frame_indices = subtask_grp['frame_indices'][:]
                eef_pos = subtask_grp['eef_pos_obj'][:]
                eef_quat = subtask_grp['eef_quat_obj'][:]
                gripper = subtask_grp['gripper_action'][:]

                # Convert to list of waypoint dicts
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

            waypoints[episode_key] = subtasks

    return waypoints
