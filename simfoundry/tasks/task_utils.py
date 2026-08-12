# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateless helper functions for SimFoundry tasks."""

import torch as th

import omnigibson.utils.transform_utils as T


def randomize_object_pose(default_pos, default_quat, max_xyz_offset=(0, 0, 0), max_z_rotation=0.0, bounds=None):
    """
    Randomizes the pose given @default_pos and @default_quat, based on max perturbations @max_xyz_offset and
    @max_z_rotation.  When @bounds is provided, the position offset is constrained so that the
    resulting position stays within the workspace bounds (uniform sampling within the intersection
    of the offset range and the valid workspace region).

    Args:
        default_pos (3-array): (x,y,z) position to perturb
        default_quat (4-array): (x,y,z,w) quaternion orientation to perturb
        max_xyz_offset (3-array): (x,y,z) maximum perturbation to sample
        max_z_rotation (float): maximum z-rotation to sample
        bounds (tuple or None): World-frame workspace bounds as ``(lower, upper)``
            tensors of shape ``(3,)``.  If provided, the sampled position is
            guaranteed to lie within these bounds.

    Returns:
        2-tuple:
            - torch.tensor: (x,y,z) perturbed position
            - torch.tensor: (x,y,z,w) perturbed quaternion
    """
    max_xyz_offset = th.tensor(max_xyz_offset, dtype=th.float)
    default_pos = th.as_tensor(default_pos, dtype=th.float)

    if bounds is not None:
        # Intersect the offset range [-max, +max] with the valid workspace region
        lo = th.maximum(-max_xyz_offset, bounds[0] - default_pos)
        hi = th.minimum(max_xyz_offset, bounds[1] - default_pos)
        pos_offset = th.rand(3) * (hi - lo) + lo
    else:
        pos_offset = th.rand(3) * (2.0 * max_xyz_offset) - max_xyz_offset

    rot_z_offset = T.euler2mat(th.tensor([0.0, 0.0, th.rand(1).item() * 2.0 * max_z_rotation - max_z_rotation], dtype=th.float))
    new_pos = default_pos + pos_offset
    new_quat = T.mat2quat(rot_z_offset @ T.quat2mat(default_quat))
    return new_pos, new_quat


def compute_look_at_orientation(cam_pos, look_at_point):
    """
    Compute the quaternion orientation for a camera at cam_pos to face look_at_point.
    Uses USD/OpenGL camera convention: -Z is forward (look direction), +Y is up.

    Args:
        cam_pos (torch.Tensor): Camera position (3,)
        look_at_point (torch.Tensor): Target point to look at (3,)

    Returns:
        torch.Tensor: Quaternion orientation (x, y, z, w)
    """
    forward = look_at_point - cam_pos
    forward = forward / th.norm(forward)

    world_up = th.tensor([0.0, 0.0, 1.0], dtype=th.float32)

    # Handle degenerate case where forward is nearly parallel to world_up
    if th.abs(th.dot(forward, world_up)) > 0.999:
        world_up = th.tensor([0.0, 1.0, 0.0], dtype=th.float32)

    right = th.cross(forward, world_up)
    right = right / th.norm(right)

    up = th.cross(right, forward)
    up = up / th.norm(up)

    # Camera convention: columns are [+X=right, +Y=up, +Z=-forward]
    rot_mat = th.stack([right, up, -forward], dim=1)
    quat = T.mat2quat(rot_mat)
    return quat


def obj_is_settled(obj, distractor_names):
    """Return True if *obj* should be considered settled for the reset velocity check."""
    if obj.fixed_base:
        return True
    if obj.name in distractor_names:
        return True
    try:
        return th.norm(obj.get_linear_velocity()).item() < 0.01
    except (AttributeError, Exception):
        # Kinematic prims (e.g. mesh_background) don't have get_linear_velocity
        return True
