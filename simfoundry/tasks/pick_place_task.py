# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import random
from collections import OrderedDict

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.object_states import Open, ToggledOn
from omnigibson.object_states.object_state_base import REGISTERED_OBJECT_STATES, RelativeObjectState
from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
from omnigibson.robots import BaseRobot, FrankaPanda, Yam
from omnigibson.scenes.scene_base import Scene
from omnigibson.tasks.task_base import BaseTask
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.ui_utils import create_module_logger

from simfoundry import ASSET_DIR
from simfoundry.tasks.macros import GAINS  # importing also applies the OG macro overrides
from simfoundry.tasks.predicates import (
    DirectPlacementPredicate,
    InsideAABBPredicate,
    SPECIAL_STATE_NAMES,
    make_aabb_predicate,
    MultiPredicate,
    Predicate,
    PredicateType,
    check_inside_aabb,
)
from simfoundry.tasks.task_utils import compute_look_at_orientation, obj_is_settled, randomize_object_pose
from simfoundry.utils.distractor_utils import (
    build_candidate_pool,
    place_distractor,
    sample_distractors,
)
from simfoundry.utils.placement_utils import place_with_predicate, resolve_gap, separate_overlapping_objects

# Create module logger
log = create_module_logger(module_name=__name__)


def _resolve_registered_state(state_name, context):
    """Look up an OmniGibson object state by name, with an actionable error for unknown names."""
    if state_name not in REGISTERED_OBJECT_STATES:
        raise ValueError(
            f"Unknown predicate state '{state_name}' ({context}). "
            f"SimFoundry special states: {', '.join(SPECIAL_STATE_NAMES)}. "
            f"Available OmniGibson object states: {', '.join(sorted(REGISTERED_OBJECT_STATES))}."
        )
    return REGISTERED_OBJECT_STATES[state_name]


class AbsoluteReward(BaseRewardFunction):
    """Reward that directly returns the value of a user-supplied function of the env."""

    def __init__(self, reward_fcn):
        self.reward_fcn = reward_fcn
        super().__init__()

    def _step(self, task, env, action):
        reward = self.reward_fcn(env)
        return reward, {}


class PickPlaceTask(BaseTask):
    """
    Task for pick-place behavior

    Args:
        activity_name (str): Name of the task to instantiate
        semantic_group_mapping (None or dict): If specified, maps group name to set of object categories / names.
            This will be used with predicate_* args to determine programmatic logic checks for determining success
            conditions
        init_predicates_all (None or list of dict): If specified, list of predicates which will be checked against for
            resetting env. Expected dict structure:

                "state": ObjectState (e.g.: "OnTop")
                "state_kwargs": Any additional args to pass into the object state
                "value": bool (True / False)                # Note that this must be True for init conditions
                "group": name of the group this applies to
                "other_group": For binary predicates, the other group which should map to exactly 1 single object
                    to pass into the object state checker

        init_predicates_any (None or list of dict): Same as @init_predicates_all, but checks for any amongst all valid candidates
        init_predicates_specific (None or list of dict): Same as @init_predicates_all, but checks for specific candidate
        goal_predicates_all (None or list of dict): If specified, list of predicates which will be checked against for
            success conditions. Expected dict structure:

                "state": ObjectState (e.g.: "OnTop")
                "state_kwargs": Any additional args to pass into the object state
                "value": bool (True / False)
                "group": name of the group this applies to
                "other_group": For binary predicates, the other group which should map to exactly 1 single object
                    to pass into the object state checker

        goal_predicates_any (None or list of dict): Same as @init_predicates_all, but checks for any amongst all valid candidates
        goal_predicates_specific (None or list of dict): Same as @init_predicates_all, but checks for specific candidate
        robot_pose (None or tuple): If specified, default (pos, quat) tuple where robot should be placed
        robot_xyz_randomization (None or 3-array): If specified, max (x,y,z) randomization to apply to robot.
            If None, will use 0
        robot_z_rot_randomization (None or float): If specified, max z-rotation randomization to apply to robot.
            If None, will use 0
        group_xyz_randomization (None or dict): If specified, maps group name to max (x,y,z) randomization to apply to
            group objects. If None, will use 0. Note this is applied after any init predicates are applied!
        group_z_rot_randomization (None or dict): If specified, maps group name to max z-rotation randomization
            to apply to group objects. If None, will use 0. Note this is applied after any init predicates are applied!
        group_init_relative_poses (None or dict): If specified, maps group name to initial (pos, quat) relative pose to robot's pose tuple to apply to
            group objects. If None, will use the object's default pose. Note this is applied after any init predicates are applied!
        group_init_joint_positions (None or dict): If specified, maps group name to initial joint positions array to apply to
            group objects with joints (e.g., to set a mailbox lid to open position). If None, objects will use their default joint positions.
        termination_config (None or dict): Keyword-mapped configuration to use to generate termination conditions. This
            should be specific to the task class. Default is None, which corresponds to a default config being usd.
            Note that any keyword required by a specific task class but not specified in the config will automatically
            be filled in with the default config. See cls.default_termination_config for default values used
        reward_config (None or dict): Keyword-mapped configuration to use to generate reward functions. This should be
            specific to the task class. Default is None, which corresponds to a default config being usd. Note that
            any keyword required by a specific task class but not specified in the config will automatically be filled
            in with the default config. See cls.default_reward_config for default values used
        include_obs (bool): Whether to include observations or not for this task
        hdr_background_name (None or str): If specified, HDR background texture to use (will look for values at ASSET_DIR/hdr_backgrounds/<NAME>.exr)
        hdr_background_z_rotation (None or float): If specified, z-rotation of the HDR background in degrees. If None, will use 0
        ground_plane_z_offset (None or float): If specified, z-offset (in meters) to apply to the ground plane. If None, will use 0
        ground_plane_z_randomization (None or float): If specified, max z-offset (in meters) to randomly apply to the
            ground plane and background mesh on each reset. The offset is sampled uniformly from
            [-ground_plane_z_randomization, +ground_plane_z_randomization]. If None or 0, no randomization is applied.
        camera_look_at_point (None or 3-array): If specified, the point (in parent frame) that external cameras
            should face. Required when camera_randomization is set.
        camera_randomization (None or dict): If specified, spherical coordinate perturbation to apply to external
            cameras on each reset. Keys: delta_r (float, meters), delta_theta (float, radians),
            delta_phi (float, radians). Each is the max uniform perturbation (+/-) applied to the camera's
            spherical coordinates relative to camera_look_at_point.
        reset_stability_max_attempts (int): Maximum number of reset attempts when the scene is detected as
            unstable (objects displaced beyond thresholds after stepping physics). Default is 5.
        reset_stability_translation_threshold (float): Maximum allowed object translation (in meters) during
            the post-reset stability check. Objects that move more than this are considered colliding/unstable.
            Default is 0.05m.
        reset_stability_rotation_threshold_deg (float): Maximum allowed object rotation (in degrees) during
            the post-reset stability check. Default is 30 degrees.
        reset_stability_n_physics_steps (int): Number of physics steps to simulate during the stability check.
            Default is 5.
    """

    def __init__(
        self,
        activity_name,
        semantic_group_mapping=None,
        init_predicates_all=None,
        init_predicates_any=None,
        init_predicates_specific=None,
        goal_predicates_all=None,
        goal_predicates_any=None,
        goal_predicates_specific=None,
        milestone_predicates=None,
        robot_pose=None,
        robot_xyz_randomization=None,
        robot_z_rot_randomization=None,
        robot_joint_randomization=None,
        group_xyz_randomization=None,
        group_z_rot_randomization=None,
        group_init_relative_poses=None,
        group_init_joint_positions=None,
        group_predicate_placement=None,
        workspace_bounds=None,
        prevent_predicate_overlap=False,
        overlap_separation_multiplier=1.5,
        additional_objects=None,
        termination_config=None,
        reward_config=None,
        include_obs=True,
        hdr_background_name=None,
        hdr_background_z_rotation=None,
        ground_plane_z_offset=None,
        ground_plane_z_randomization=None,
        camera_look_at_point=None,
        camera_randomization=None,
        reset_stability_max_attempts=5,
        reset_stability_translation_threshold=0.05,
        reset_stability_rotation_threshold_deg=30.0,
        reset_stability_n_physics_steps=5,
    ):
        # Make sure object states are enabled
        assert gm.ENABLE_OBJECT_STATES, f"Must set gm.ENABLE_OBJECT_STATES=True in order to use {self.__class__.__name__}!"

        # Store values
        self.activity_name = activity_name
        semantic_group_mapping = {} if semantic_group_mapping is None else semantic_group_mapping
        self.semantic_group_mapping = {k: set(v) for k, v in semantic_group_mapping.items()}
        self.init_predicates_all = init_predicates_all
        self.init_predicates_any = init_predicates_any
        self.init_predicates_specific = init_predicates_specific
        self.goal_predicates_all = goal_predicates_all
        self.goal_predicates_any = goal_predicates_any
        self.goal_predicates_specific = goal_predicates_specific
        self.milestone_predicates_config = milestone_predicates  # Store config for later initialization
        if robot_pose is not None:
            robot_pose = [[th.tensor(robot_pose[0]), th.tensor(robot_pose[1])]]
        self.robot_poses = robot_pose
        self.robot_xyz_randomization = None if robot_xyz_randomization is None else th.tensor(robot_xyz_randomization)
        self.robot_z_rot_randomization = robot_z_rot_randomization
        self.robot_joint_randomization = None if robot_joint_randomization is None else th.tensor(robot_joint_randomization)
        self.group_xyz_randomization = dict() if group_xyz_randomization is None else {k: th.tensor(v) for k, v in group_xyz_randomization.items()}
        self.group_z_rot_randomization = dict() if group_z_rot_randomization is None else group_z_rot_randomization
        self.group_init_relative_poses = dict()
        if group_init_relative_poses is not None:
            for k, v in group_init_relative_poses.items():
                assert not all(vv is None for vv in v), f"At least one of pos, quat in group_init_relative_poses for group {k} must not be None"
                pose = [th.tensor(v[0]) if v[0] is not None else th.zeros(3), th.tensor(v[1]) if v[1] is not None else th.tensor([0, 0, 0, 1.0])]
                self.group_init_relative_poses[k] = pose
        self.group_init_joint_positions = dict() if group_init_joint_positions is None else {k: {joint_name: th.tensor(joint_position) for joint_name, joint_position in v.items()} for k, v in group_init_joint_positions.items()}
        if self.group_init_joint_positions:
            log.info(f"[INIT] group_init_joint_positions configured: {list(self.group_init_joint_positions.keys())}")
        # Predicate-based spatial placement: maps group name -> config dict with
        #   reference_group (str) or reference_groups ([a, b], required for 'between'),
        #   predicates (list of str), gap (float, [min, max], or per-predicate dict),
        #   z_offset (float, optional), aligned (bool, optional),
        #   link_name (str, required for 'inside_link'), probability (float, optional)
        self.group_predicate_placement = dict() if group_predicate_placement is None else group_predicate_placement
        if self.group_predicate_placement:
            log.info(f"[INIT] group_predicate_placement configured: {list(self.group_predicate_placement.keys())}")
        # Workspace bounds: workspace limits in robot base frame (transformed to
        # world frame at reset time).  Constrains object placement and
        # randomization for all non-robot objects.
        if workspace_bounds is not None:
            self._workspace_bounds_local = (
                th.tensor(workspace_bounds["lower"], dtype=th.float32),
                th.tensor(workspace_bounds["upper"], dtype=th.float32),
            )
        else:
            self._workspace_bounds_local = None
        self.prevent_predicate_overlap = prevent_predicate_overlap
        self.overlap_separation_multiplier = overlap_separation_multiplier
        # Additional / distractor objects config and runtime state
        self.additional_objects_cfg = additional_objects  # raw config dict (or None)
        self._distractor_pool = []       # built once in _load
        self._distractor_objs = []       # OG objects added in current episode
        self.robots = None
        self.init_predicates = None
        self.group_objs = None
        self.hdr_background_name = hdr_background_name
        self.hdr_background_z_rotation = hdr_background_z_rotation
        self.ground_plane_z_offset = ground_plane_z_offset
        self.ground_plane_z_randomization = ground_plane_z_randomization

        # Base positions for ground plane randomization (stored during _load)
        self._base_floor_plane_pos = None
        self._base_floor_plane_ori = None
        self._background_pos = None
        self._background_ori = None
        self._background = None

        # Camera randomization (spherical perturbation)
        self.camera_look_at_point = th.tensor(camera_look_at_point, dtype=th.float32) if camera_look_at_point is not None else None
        self.camera_randomization = camera_randomization
        self._base_external_cam_poses = {}

        # Reset stability check parameters
        self.reset_stability_max_attempts = reset_stability_max_attempts
        self.reset_stability_translation_threshold = reset_stability_translation_threshold
        self.reset_stability_rotation_threshold_deg = reset_stability_rotation_threshold_deg
        self.reset_stability_n_physics_steps = reset_stability_n_physics_steps

        # Deterministic pose schedule override: when set to a dict mapping group_name -> z_rot_offset,
        # _reset_scene will apply those exact z-rotation offsets (no xyz randomization) instead of
        # random sampling. Set externally (e.g., from teleop script) before calling reset().
        self._override_group_z_rotations = None

        # Milestone tracking - will be initialized in update_scene
        self.milestone_predicates = {}  # name -> Predicate
        self.milestones_achieved = OrderedDict() # name -> 0 or 1 (0 if not achieved, 1 if achieved)

        # Run super init
        super().__init__(termination_config=termination_config, reward_config=reward_config, include_obs=include_obs)

    def update_scene(self):
        # Determine group objects
        self.group_objs = dict()
        log.info(f"Semantic group mapping: {self.semantic_group_mapping}")
        log.info(f"Scene objects: {[(obj.name, obj.category) for obj in og.sim.scenes[0].objects]}")
        for group, valid_keys in self.semantic_group_mapping.items():
            objs = []
            for obj in og.sim.scenes[0].objects:
                if obj.name in valid_keys or obj.category in valid_keys:
                    objs.append(obj)
            self.group_objs[group] = objs
            log.info(f"Group '{group}' matched objects: {[obj.name for obj in objs]}")

        # Create predicates for setting init conditions
        self.init_predicates = self._create_predicates(self.init_predicates_all, self.init_predicates_any, self.init_predicates_specific)

        # Update goal predicates
        goal_predicates = list(self._create_predicates(self.goal_predicates_all, self.goal_predicates_any, self.goal_predicates_specific).values())
        self._termination_conditions["predicates"].predicates = goal_predicates

        # Create milestone predicates for tracking sub-task completion
        self._create_milestone_predicates()

        # Make sure all toggle buttons are off
        for _obj in og.sim.scenes[0].objects:
            if ToggledOn in _obj.states:
                _obj.states[ToggledOn].link.visible = False

    def _load(self, env):
        # Apply ground plane position from scene file if available
        self._apply_ground_plane_from_scene(env)

        # If a gaussian splat background if detected (named "gs_background"), set the floor plane to be matte to enable shadows
        # TODO: Do not hardcode this!
        if self.hdr_background_name is not None:
            print("Setting up HDR background...")
            # Set background HDR
            domelight_texture_path = f"{ASSET_DIR}/hdr_backgrounds/{self.hdr_background_name}.exr"
            domelight_z_ori = self.hdr_background_z_rotation if self.hdr_background_z_rotation is not None else th.rand(1).item() * th.pi * 2
            og.sim.skybox.texture_file_path = domelight_texture_path
            og.sim.skybox.set_position_orientation(orientation=T.euler2quat(th.tensor([0, 0, domelight_z_ori])))
            og.sim.skybox.color = [1.0, 1.0, 1.0]
            og.sim.skybox.intensity = 1000.0

        gs_background = env.scene.object_registry("name", "gs_background")
        mesh_background = env.scene.object_registry("name", "mesh_background")
        if gs_background is not None:
            print("Setting up 3DGS background...")
            # Set proxy from 3DGS background to ground floorplane
            # See: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/neural-rendering.html
            floor_geom = og.sim.floor_plane.prim.GetChildren()[0]
            matte_attr_name = "primvars:isMatteObject"
            if not floor_geom.HasProperty(matte_attr_name):
                floor_geom.CreateAttribute(matte_attr_name, lazy.pxr.Sdf.ValueTypeNames.Bool)
            floor_geom.GetProperty(matte_attr_name).Set(True)
            floor_geom_sdf_path = floor_geom.GetPrimPath()
            gauss = gs_background.root_link.prim.GetChildren()[0]
            gauss.GetProperty("proxy").SetTargets([floor_geom_sdf_path])

            # Store base background position for ground plane z-randomization
            bg_pos, bg_ori = gs_background.get_position_orientation()
            self._background_pos = bg_pos.clone()
            self._background_ori = bg_ori.clone()
            self._background = gs_background
        elif mesh_background is not None:
            print("Setting up mesh background...")
            bg_pos, bg_ori = mesh_background.get_position_orientation()
            self._background_pos = bg_pos.clone()
            self._background_ori = bg_ori.clone()
            self._background = mesh_background
        else:
            print("No background found, setting to None")
            self._background = None

        # Update opacity threshold for all objects with opacity enabled to be 0.5
        for obj in og.sim.scenes[0].objects:
            for material in obj.materials:
                if "enable_opacity" in material.shader_input_names:
                    enable_opacity = material.get_input("enable_opacity")
                    if enable_opacity:
                        material.set_input("opacity_threshold", 0.5)

        # Grab robot
        if len(env.robots) > 1:
            assert self.robot_poses is None, "robot_pose must be None if multiple robots are present"
        if self.robot_poses is None:
            # Grab current robot poses as default pose
            self.robot_poses = [robot.get_position_orientation() for robot in env.robots]

        # Set robot max velocity / efforts
        # Step simulation to initialize articulations before setting joint limits
        og.sim.step()

        # Set hardcoded tuned gains
        for r in env.robots:
            assert isinstance(r, (FrankaPanda, Yam)), f"Robot {r.name} not supported!"
            gains = GAINS[r.model_name]
            idx = 0
            for jnt in r.joints.values():
                if jnt.driven:
                    jnt.max_velocity = gains["max_velocity"][idx]
                    jnt.max_effort = gains["max_effort"][idx]
                    
                    idx += 1

        # Cache base external camera poses for spherical randomization
        if hasattr(env, 'external_sensors') and env.external_sensors:
            for name, sensor in env.external_sensors.items():
                pos, ori = sensor.get_position_orientation(frame="parent")
                self._base_external_cam_poses[name] = (pos.clone(), ori.clone())
            if self.camera_randomization:
                log.info(f"Cached {len(self._base_external_cam_poses)} external camera base poses for randomization")

        # Auto-infer camera look-at point if not manually specified
        if self.camera_look_at_point is None and self.camera_randomization and self._base_external_cam_poses:
            self.camera_look_at_point = self._infer_camera_look_at_point()
            if self.camera_look_at_point is not None:
                log.info(f"Auto-inferred camera look-at point: {self.camera_look_at_point.tolist()}")
            else:
                log.warning("Failed to infer camera look-at point; camera randomization will be skipped")

        self.update_scene()

    def _apply_ground_plane_from_scene(self, env):
        """
        Apply ground plane position/orientation from the scene file's ground_plane_info if present.
        Also stores the base positions of the floor plane and background mesh for z-randomization.
        """
        if og.sim.floor_plane is None:
            return

        scene_file = env.scene.scene_file
        floor_pos = th.tensor([0.0, 0.0, 0.0], dtype=th.float32)
        floor_ori = th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32)
        if scene_file is not None:
            if isinstance(scene_file, str):
                with open(scene_file, "r") as f:
                    scene_info = json.load(f)
            else:
                scene_info = scene_file

            if "ground_plane_info" in scene_info:
                ground_plane_info = scene_info["ground_plane_info"]
                floor_pos = th.tensor(ground_plane_info["position"], dtype=th.float32)
                floor_ori = th.tensor(ground_plane_info["orientation"], dtype=th.float32)
        if self.ground_plane_z_offset is not None:
            floor_pos[2] = self.ground_plane_z_offset
        og.sim.floor_plane.set_position_orientation(position=floor_pos, orientation=floor_ori)
        log.info(f"Applied ground plane position from scene file: z={floor_pos[2]:.4f}m")

        # Store base floor plane position for z-randomization (even if no scene file)
        pos, ori = og.sim.floor_plane.get_position_orientation()
        self._base_floor_plane_pos = pos.clone()
        self._base_floor_plane_ori = ori.clone()

    def _compute_world_workspace_bounds(self):
        """
        Transform the workspace bounds from robot base frame to world frame.

        Returns:
            tuple or None: ``(world_lower, world_upper)`` tensors of shape ``(3,)``
            representing the axis-aligned bounding box in world frame, or ``None``
            if no workspace bounds are configured.
        """
        if self._workspace_bounds_local is None:
            return None

        local_lo, local_hi = self._workspace_bounds_local
        robot_pos, robot_quat = self.robot_poses[0]
        rot_mat = T.quat2mat(robot_quat)

        # Transform all 8 corners of the local bounding box to world frame,
        # then take the axis-aligned min/max to get the world-frame AABB.
        corners = []
        for x in (local_lo[0], local_hi[0]):
            for y in (local_lo[1], local_hi[1]):
                for z in (local_lo[2], local_hi[2]):
                    corner = th.tensor([x, y, z], dtype=th.float32)
                    world_corner = robot_pos + rot_mat @ corner
                    corners.append(world_corner)

        corners = th.stack(corners)
        world_lo = corners.min(dim=0).values
        world_hi = corners.max(dim=0).values
        return (world_lo, world_hi)

    def _infer_camera_look_at_point(self):
        """
        Infer the camera look-at point by ray-casting each external camera's forward axis (-Z)
        onto the ground plane and averaging the intersection points.

        Returns:
            torch.Tensor or None: The averaged look-at point (3,), or None if no valid intersections.
        """
        if self._base_floor_plane_pos is None or not self._base_external_cam_poses:
            return None

        z_ground = self._base_floor_plane_pos[2].item()
        intersections = []

        for name, (pos, ori) in self._base_external_cam_poses.items():
            # Camera forward direction is -Z column of the rotation matrix
            rot_mat = T.quat2mat(ori)
            forward = -rot_mat[:, 2]

            # Skip if ray is nearly parallel to the ground plane
            if abs(forward[2].item()) < 0.01:
                log.warning(f"Camera '{name}' forward ray is nearly parallel to ground plane; skipping")
                continue

            # Ray parameter: t = (z_ground - p_z) / d_z
            t = (z_ground - pos[2].item()) / forward[2].item()

            # Skip if intersection is behind the camera
            if t < 0:
                log.warning(f"Camera '{name}' forward ray points away from ground plane; skipping")
                continue

            hit = pos + t * forward
            intersections.append(hit)

        if not intersections:
            return None

        return th.stack(intersections).mean(dim=0)

    def _randomize_ground_plane_z(self, env):
        """
        Randomize the ground plane z-position and move the background mesh (gs_background) and all
        non-robot scene objects by the same offset so they stay on top of the ground plane.
        Called on each episode reset.
        """
        if not self.ground_plane_z_randomization:
            return
        if og.sim.floor_plane is None or self._base_floor_plane_pos is None:
            return

        # Sample uniform z-offset in [-max, +max]
        z_offset = (th.rand(1).item() * 2.0 - 1.0) * self.ground_plane_z_randomization

        # Apply to floor plane
        new_floor_pos = self._base_floor_plane_pos.clone()
        new_floor_pos[2] += z_offset
        if self.ground_plane_z_offset is not None:
            new_floor_pos[2] += self.ground_plane_z_offset
        og.sim.floor_plane.set_position_orientation(position=new_floor_pos, orientation=self._base_floor_plane_ori)

        # Apply same offset to background so it stays aligned with the ground plane
        if self._background is not None and self._background_pos is not None:
            new_bg_pos = self._background_pos.clone()
            new_bg_pos[2] += z_offset
            self._background.set_position_orientation(position=new_bg_pos, orientation=self._background_ori)

        # Shift all non-robot scene objects by the same z-offset so they remain on top of the ground plane
        for obj in env.scene.objects:
            if isinstance(obj, BaseRobot):
                continue
            if obj.name == "gs_background" or obj.name == "mesh_background":
                continue  # already shifted above
            obj_pos, obj_ori = obj.get_position_orientation()
            obj_pos[2] += z_offset
            obj.set_position_orientation(position=obj_pos, orientation=obj_ori)

        log.info(f"Randomized ground plane z: offset={z_offset:+.4f}m, new z={new_floor_pos[2]:.4f}m")

    def _randomize_external_cameras(self, env):
        """
        Randomize external camera positions using spherical coordinate perturbation around a
        configurable look-at point. The orientation is recomputed to face the look-at point.

        Spherical convention (physics):
            r: distance from look-at point to camera
            theta: polar angle from +Z axis (0=above, pi/2=horizontal)
            phi: azimuthal angle in XY plane from +X axis
        """
        if not self.camera_randomization:
            return
        if not hasattr(env, 'external_sensors') or not env.external_sensors:
            return

        #  auto-infer the look-at point on the first reset after scene is loaded
        if not self._base_external_cam_poses:
            for name, sensor in env.external_sensors.items():
                pos, ori = sensor.get_position_orientation(frame="parent")
                self._base_external_cam_poses[name] = (pos.clone(), ori.clone())
            if self._base_external_cam_poses:
                log.info(f"Lazy-cached {len(self._base_external_cam_poses)} external camera base poses")

        if self.camera_look_at_point is None and self._base_external_cam_poses:
            self.camera_look_at_point = self._infer_camera_look_at_point()
            if self.camera_look_at_point is not None:
                log.info(f"Auto-inferred camera look-at point: {self.camera_look_at_point.tolist()}")

        if self.camera_look_at_point is None:
            log.warning("camera_randomization is set but camera_look_at_point is None; skipping camera randomization")
            return

        delta_r = self.camera_randomization.get("delta_r", 0.0)
        delta_theta = self.camera_randomization.get("delta_theta", 0.0)
        delta_phi = self.camera_randomization.get("delta_phi", 0.0)

        for name, sensor in env.external_sensors.items():
            if name not in self._base_external_cam_poses:
                continue

            base_pos, _ = self._base_external_cam_poses[name]

            # Vector from look-at point to camera
            v = base_pos - self.camera_look_at_point

            # Convert to spherical coordinates
            r = th.norm(v)
            if r < 1e-6:
                log.warning(f"Camera '{name}' is at the look-at point; skipping randomization")
                continue
            theta = th.acos(th.clamp(v[2] / r, -1.0, 1.0))
            phi = th.atan2(v[1], v[0])

            # Apply uniform random perturbations
            r_new = r + (th.rand(1).item() * 2.0 - 1.0) * delta_r
            r_new = max(r_new, 0.05)  # Prevent camera from collapsing to the look-at point
            theta_new = theta + (th.rand(1).item() * 2.0 - 1.0) * delta_theta
            theta_new = th.clamp(th.tensor(theta_new), 0.01, th.pi - 0.01).item()  # Stay within valid range
            phi_new = phi + (th.rand(1).item() * 2.0 - 1.0) * delta_phi

            # Convert back to Cartesian offset from look-at point
            new_v = th.tensor([
                r_new * math.sin(theta_new) * math.cos(phi_new),
                r_new * math.sin(theta_new) * math.sin(phi_new),
                r_new * math.cos(theta_new),
            ], dtype=th.float32)

            new_pos = self.camera_look_at_point + new_v
            new_ori = compute_look_at_orientation(new_pos, self.camera_look_at_point)

            sensor.set_position_orientation(position=new_pos, orientation=new_ori, frame="parent")

        log.info(f"Randomized {len(self._base_external_cam_poses)} external camera(s) (spherical perturbation)")

    def _load_non_low_dim_observation_space(self):
        # No non-low dim observations so we return an empty dict
        return dict()

    def reset(self, env):
        # Reset milestone tracking for the new episode
        self.reset_milestones()

        for attempt in range(self.reset_stability_max_attempts):
            # Only reset with velocity checks if the init conditions are not specified
            if self.init_predicates is None or len(self.init_predicates) == 0:
                # Keep sampling until velocities are minimized.
                # Exclude distractor objects from the check — they are spawned inside
                # _reset_scene and may not have settled yet.  Re-running the full
                # reset loop would remove and re-add them each iteration, which
                # corrupts OmniGibson's physics tensor views and crashes the sim.
                success = False
                max_samples = 10
                curr_sample = 0
                while not success and curr_sample < max_samples:
                    super().reset(env)
                    og.sim.step_physics()
                    # Update distractor names after reset (they may have been re-created)
                    distractor_names = {o.name for o in self._distractor_objs}
                    success = all(obj_is_settled(obj, distractor_names) for obj in env.scene.objects)
                    curr_sample += 1
                if not success:
                    print("Failed to reset scene: velocities not minimized")
            else:
                # Reset without velocity checks
                super().reset(env)

            # Check scene stability by stepping physics and detecting large object displacements
            if self._check_scene_stability(env):
                break

            if attempt < self.reset_stability_max_attempts - 1:
                log.warning(
                    f"Scene unstable after reset (attempt {attempt + 1}/{self.reset_stability_max_attempts}), re-resetting..."
                )
            else:
                log.warning(
                    f"Scene still unstable after {self.reset_stability_max_attempts} attempts, proceeding anyway"
                )

        # Capture episode-start state for predicates that need it (e.g., Lifted).
        # OmniGibson's BaseTerminationCondition.reset() does not dispatch to _reset,
        # so the task triggers the capture itself, after init placement and settling.
        self._capture_predicate_start_states()

    def _capture_predicate_start_states(self):
        """Let goal and milestone predicates record episode-start state (e.g., Lifted's start Z)."""
        predicates = list(self._termination_conditions["predicates"].predicates)
        predicates += list(self.milestone_predicates.values())
        for predicate in predicates:
            if hasattr(predicate, "capture_start_state"):
                predicate.capture_start_state()

    def _check_scene_stability(self, env):
        """
        Check post-reset scene stability by stepping physics and detecting objects that have
        translated or rotated beyond configured thresholds (indicating collisions / instability).

        Returns:
            bool: True if stable (all objects within thresholds), False otherwise.
        """
        skip_names = {"gs_background", "mesh_background"}

        # Record pre-physics poses of all non-robot, non-fixed objects
        pre_poses = {}
        for obj in env.scene.objects:
            if isinstance(obj, BaseRobot) or obj.fixed_base or obj.name in skip_names:
                continue
            pos, quat = obj.get_position_orientation()
            pre_poses[obj.name] = (pos.clone(), quat.clone())

        if not pre_poses:
            return True

        # Let physics settle
        for _ in range(self.reset_stability_n_physics_steps):
            og.sim.step_physics()

        rotation_threshold_rad = math.radians(self.reset_stability_rotation_threshold_deg)

        for obj in env.scene.objects:
            if obj.name not in pre_poses:
                continue
            pre_pos, pre_quat = pre_poses[obj.name]
            cur_pos, cur_quat = obj.get_position_orientation()

            # Translation check
            translation = th.norm(cur_pos - pre_pos).item()
            if translation > self.reset_stability_translation_threshold:
                log.warning(
                    f"Stability check: '{obj.name}' translated {translation:.4f}m "
                    f"(threshold: {self.reset_stability_translation_threshold}m)"
                )
                return False

            # Rotation check: angle = 2 * arccos(|q1 · q2|)
            dot = th.abs(th.dot(pre_quat, cur_quat)).clamp(max=1.0).item()
            angle = 2.0 * math.acos(dot)
            if angle > rotation_threshold_rad:
                log.warning(
                    f"Stability check: '{obj.name}' rotated {math.degrees(angle):.1f}° "
                    f"(threshold: {self.reset_stability_rotation_threshold_deg}°)"
                )
                return False

        return True


    def _reset_scene(self, env):
        # Remove any distractor objects BEFORE scene restore, otherwise
        # scene.restore() will try to dump_state() on uninitialized objects
        # and crash with "Object must be initialized before dumping state!"
        self._remove_distractor_objects(env)

        # Call super first (this calls scene.restore which resets to initial state)
        super()._reset_scene(env)

        # Randomize ground plane z-position (and move background mesh to match)
        self._randomize_ground_plane_z(env)

        # Randomize external camera positions (spherical perturbation around look-at point)
        self._randomize_external_cameras(env)

        # Set hardcoded tuned gains
        for r in env.robots:
            gains = GAINS[r.model_name]
            idx = 0
            for jnt in r.joints.values():
                if jnt.driven:
                    jnt.stiffness = gains["kp"][idx]
                    jnt.damping = gains["kv"][idx]
                    if "joint_lower_limits" in gains:
                        jnt.lower_limit = gains["joint_lower_limits"][idx]
                    if "joint_upper_limits" in gains:
                        jnt.upper_limit = gains["joint_upper_limits"][idx]
                    idx += 1

        for group, pose in self.group_init_relative_poses.items():
            # For now, assert only one object per group to avoid overlapping poses
            assert len(self.group_objs[group]) == 1
            for obj in self.group_objs[group]:
                obj_pose = T.mat2pose(T.pose2mat(self.robot_poses[0]) @ T.pose2mat(pose))
                obj.set_position_orientation(*obj_pose)

        # Compute world-frame workspace bounds from robot-base-frame bounds
        # (used by both group randomization and predicate placement below)
        world_bounds = self._compute_world_workspace_bounds()

        # Apply group randomizations BEFORE init predicates so that predicates
        # (e.g., PlaceOnTop) place objects relative to already-randomized poses.
        # Non-robot groups are constrained to stay within workspace bounds.
        for group, objs in self.group_objs.items():
            xyz_randomization = self.group_xyz_randomization.get(group, None)
            z_rot_randomization = self.group_z_rot_randomization.get(group, None)
            is_robot = group == "robot"
            for obj in objs:
                obj.set_position_orientation(*self.randomize_object_pose(
                    *obj.get_position_orientation(),
                    max_xyz_offset=xyz_randomization if xyz_randomization is not None else th.zeros(3),
                    max_z_rotation=z_rot_randomization if z_rot_randomization is not None else 0.0,
                    bounds=world_bounds if not is_robot else None,
                ))

        # Apply predicate-based placement AFTER XYZ randomization of reference
        # objects so that the placed object uses the already-randomized reference
        # position (e.g., place marker left_of the desk_organizer).

        predicate_placed_objects = []  # (obj, predicate) for overlap prevention
        for group, placement_cfg in self.group_predicate_placement.items():
            probability = placement_cfg.get("probability", 1.0)
            if random.random() > probability:
                log.info(f"[RESET] Skipped predicate placement for '{group}' "
                         f"(p={probability:.2f}); using xyz randomization fallback")
                continue

            ref_cfg = placement_cfg.get("reference_groups", placement_cfg.get("reference_group"))
            assert ref_cfg is not None, (
                f"Predicate placement for group '{group}' requires reference_group or reference_groups"
            )
            ref_group_names = [ref_cfg] if isinstance(ref_cfg, str) else list(ref_cfg)
            predicates = placement_cfg["predicates"]
            gap_cfg = placement_cfg.get("gap", 0.05)
            z_offset = placement_cfg.get("z_offset", 0.0)
            aligned = placement_cfg.get("aligned", False)
            link_name = placement_cfg.get("link_name", None)

            # Randomly select a predicate for this reset
            predicate = random.choice(predicates)

            # Resolve gap (supports scalar, [min, max], or per-predicate dict)
            gap = resolve_gap(gap_cfg, predicate)

            ref_objs = []
            for ref_group in ref_group_names:
                objs_rg = self.group_objs.get(ref_group, [])
                assert len(objs_rg) == 1, (
                    f"Predicate placement for group '{group}' requires exactly 1 "
                    f"object in reference group '{ref_group}', found {len(objs_rg)}"
                )
                ref_objs.append(objs_rg[0])

            if predicate == "between":
                assert len(ref_objs) == 2, (
                    f"Predicate 'between' for group '{group}' requires reference_groups: [a, b]"
                )
            if predicate == "inside_link":
                assert link_name is not None, (
                    f"Predicate 'inside_link' for group '{group}' requires link_name"
                )

            # Non-'between' predicates in a mixed predicates list use the first reference group
            for obj in self.group_objs[group]:
                place_with_predicate(obj, ref_objs[0], predicate, gap=gap, z_offset=z_offset,
                                     bounds=world_bounds, aligned=aligned,
                                     reference_obj_2=ref_objs[1] if len(ref_objs) > 1 else None,
                                     link_name=link_name)
                predicate_placed_objects.append((obj, predicate))

            log.info(f"[RESET] Placed '{group}' {predicate} '{' / '.join(ref_group_names)}' (gap={gap:.3f}m)")

        # Resolve overlaps between predicate-placed objects (e.g., cup and bowl
        # both placed in_front_of the plate would stack without this).
        if self.prevent_predicate_overlap and len(predicate_placed_objects) > 1:
            separate_overlapping_objects(
                predicate_placed_objects,
                multiplier=self.overlap_separation_multiplier,
            )
            log.info(f"[RESET] Ran overlap prevention on {len(predicate_placed_objects)} predicate-placed objects")

        # Add distractor / additional objects (re-sampled every reset)
        self._add_distractor_objects(env)

        # Set init predicates AFTER randomization so placements (e.g., bowl on plate)
        # use the already-randomized target positions.
        if self.init_predicates:
            for name, predicate in self.init_predicates.items():
                log.info(f"Setting init predicate: {name}...")
                try:
                    success = predicate.set()
                    if not success:
                        log.warning(f"Init predicate {name} returned False (placement may have failed)")
                except Exception as e:
                    log.warning(f"Failed to set init predicate {name}: {e}")
        
        # Set initial joint positions for objects (e.g., to open mailbox lid)
        # joint positions is a dictionary of joint names and their positions
        log.info(f"[RESET] Applying group_init_joint_positions: {list(self.group_init_joint_positions.keys())}")
        for group, joint_positions in self.group_init_joint_positions.items():
            # Convert tensors to float for logging
            joint_positions_str = {k: float(v) for k, v in joint_positions.items()}
            log.info(f"[RESET] Processing group '{group}' with joint_positions: {joint_positions_str}")
            for obj in self.group_objs[group]:
                if hasattr(obj, 'n_joints') and obj.n_joints > 0:
                    log.info(f"Setting initial joint positions for '{obj.name}' (group '{group}'): {joint_positions_str}")
                    log.info(f"  Available joints for '{obj.name}': {list(obj.joints.keys())}")
                    
                    # Get all joint indices for this object
                    for joint_name, joint_position in joint_positions.items():
                        try:
                            if joint_name not in obj.joints:
                                log.warning(f"Joint '{joint_name}' not found in '{obj.name}'. Available: {list(obj.joints.keys())}")
                                continue
                            
                            # Get joint info before setting
                            joint = obj.joints[joint_name]
                            pos_before = float(joint.get_state()[0])
                            joint_position_float = float(joint_position)
                            log.info(f"  Joint '{joint_name}': position before={pos_before:.3f}, setting to={joint_position_float:.3f}")
                            
                            # Set position
                            joint.set_pos(joint_position)
                            
                            # Verify position was set
                            pos_after = float(joint.get_state()[0])
                            log.info(f"  Joint '{joint_name}': position after={pos_after:.3f}")
                            
                        except Exception as e:
                            log.warning(f"Failed to set initial joint position for '{joint_name}' of '{obj.name}': {e}")
                else:
                    log.warning(f"Object '{obj.name}' in group '{group}' has no joints - cannot set joint positions")

    # ------------------------------------------------------------------
    # Distractor / additional objects
    # ------------------------------------------------------------------

    def _remove_distractor_objects(self, env):
        """Remove any distractor objects that were added in a previous reset."""
        if not self._distractor_objs:
            return

        objs_to_remove = list(self._distractor_objs)
        self._distractor_objs.clear()

        # Separate initialized vs uninitialized objects
        initialized = [o for o in objs_to_remove if getattr(o, "_initialized", False)]
        uninitialized = [o for o in objs_to_remove if not getattr(o, "_initialized", False)]

        # Normal batch removal for initialized objects
        if initialized:
            try:
                og.sim.batch_remove_objects(initialized)
            except Exception as e:
                log.warning(f"[RESET] batch_remove_objects failed: {e}")
                uninitialized.extend(initialized)

        # Force-clean uninitialized objects (bypass dump_state)
        for obj in uninitialized:
            try:
                # Remove from scene registry
                try:
                    if obj.name in env.scene.object_registry.get_dict("name"):
                        env.scene.object_registry.remove(obj)
                except Exception:
                    pass
                # Remove USD prim
                if hasattr(obj, "prim_path") and obj.prim_path:
                    prim = og.sim.stage.GetPrimAtPath(obj.prim_path)
                    if prim and prim.IsValid():
                        og.sim.stage.RemovePrim(obj.prim_path)
                log.info(f"[RESET] Force-cleaned uninitialized distractor '{obj.name}'")
            except Exception as e:
                log.warning(f"[RESET] Failed to force-clean distractor '{obj.name}': {e}")

    def _add_distractor_objects(self, env):
        """Sample and place distractor objects for the current episode."""
        if not self.additional_objects_cfg:
            return

        # Lazy-init: build the candidate pool on the first call
        if not self._distractor_pool:
            cfg = self.additional_objects_cfg
            self._distractor_pool = build_candidate_pool(
                dataset_name=cfg.get("dataset_name", "behavior-1k-assets"),
                filters=cfg.get("filters"),
                category=cfg.get("category"),
                abilities=cfg.get("abilities"),
                specific_assets=cfg.get("specific_assets"),
            )
            log.info(f"[RESET] Built distractor candidate pool: {len(self._distractor_pool)} models")
            if not self._distractor_pool:
                log.warning("[RESET] Distractor pool is empty — no matching assets found")
                return

        from omnigibson.objects import DatasetObject

        cfg = self.additional_objects_cfg
        n = cfg.get("n", 1)
        placement_radius = cfg.get("placement_radius", 0.3)
        z_offset = cfg.get("z_offset", 0.02)
        max_attempts = cfg.get("max_placement_attempts", 20)
        reference_group = cfg.get("reference_group", None)

        # Compute world-frame XY placement bounds for distractors (if configured)
        distractor_bounds_world = None
        distractor_bounds_cfg = cfg.get("placement_bounds", None)
        if distractor_bounds_cfg is not None:
            local_lo_xy = th.tensor(distractor_bounds_cfg["lower"], dtype=th.float32)
            local_hi_xy = th.tensor(distractor_bounds_cfg["upper"], dtype=th.float32)
            robot_pos, robot_quat = self.robot_poses[0]
            rot_mat = T.quat2mat(robot_quat)
            # Transform all 4 corners of the 2D bounds to world XY
            corners_xy = []
            for x in (local_lo_xy[0], local_hi_xy[0]):
                for y in (local_lo_xy[1], local_hi_xy[1]):
                    corner = th.tensor([x, y, 0.0], dtype=th.float32)
                    world_corner = robot_pos + rot_mat @ corner
                    corners_xy.append(world_corner[:2])
            corners_xy = th.stack(corners_xy)
            distractor_bounds_world = (
                corners_xy.min(dim=0).values,
                corners_xy.max(dim=0).values,
            )

        # Derive support Z from ground plane height (stored during _load)
        ground_plane_z = None
        if self._base_floor_plane_pos is not None:
            ground_plane_z = float(self._base_floor_plane_pos[2].item())

        # Sample from pool
        selected = sample_distractors(self._distractor_pool, n)
        if not selected:
            return

        # Determine reference position for placement (used when no placement_bounds)
        centroid = th.zeros(3)
        if reference_group and reference_group in self.group_objs:
            ref_objs = self.group_objs[reference_group]
            if ref_objs:
                ref_pos, _ = ref_objs[0].get_position_orientation()
                centroid = ref_pos
                log.info(f"[RESET] Placing distractors around '{reference_group}' ({ref_objs[0].name}) at {centroid.tolist()}")


        # Add objects one at a time with sim stopped.
        # Some behavior-1k-assets objects crash during initialization (e.g. broken
        # emitter meshes), so we catch per-object failures and skip those.
        new_objs = []
        for i, entry in enumerate(selected):
            cat = entry["category"]
            model = entry["model"]
            ds = entry.get("dataset_name", "behavior-1k-assets")
            obj_name = f"distractor_{cat}_{model}_{i}"
            try:
                with og.sim.stopped():
                    obj = DatasetObject(
                        name=obj_name,
                        category=cat,
                        model=model,
                        dataset_name=ds,
                    )
                    env.scene.add_object(obj)

                og.sim.step_physics()
                new_objs.append(obj)
            except Exception as e:
                log.warning(f"[RESET] Distractor {cat}/{model} failed to initialise, skipping: {e}")
                # Force-clean the broken object from scene registry + USD
                try:
                    if obj.name in env.scene.object_registry.get_dict("name"):
                        env.scene.object_registry.remove(obj)
                    if hasattr(obj, "prim_path") and obj.prim_path:
                        prim = og.sim.stage.GetPrimAtPath(obj.prim_path)
                        if prim and prim.IsValid():
                            og.sim.stage.RemovePrim(obj.prim_path)
                except Exception:
                    pass

        if not new_objs:
            return

        # Let everything settle 
        for _ in range(5):
            og.sim.step_physics()

        # Place each object with collision avoidance
        added_count = 0
        for obj in new_objs:
            try:
                success = place_distractor(
                    obj=obj,
                    existing_objects=list(env.scene.objects),
                    centroid=centroid,
                    placement_radius=placement_radius,
                    z_offset=z_offset,
                    max_attempts=max_attempts,
                    support_z=ground_plane_z,
                    placement_bounds=distractor_bounds_world,
                )
                if success:
                    self._distractor_objs.append(obj)
                    added_count += 1
                    log.info(f"[RESET] Placed distractor '{obj.name}' at {obj.get_position_orientation()[0].tolist()}")
                else:
                    log.warning(f"[RESET] Could not place '{obj.name}' — removing")
                    env.scene.remove_object(obj)
            except Exception as e:
                log.warning(f"[RESET] Failed to place distractor '{obj.name}': {e}")


        if added_count > 0:
            for _ in range(5):
                og.sim.step_physics()
                og.sim.render()
            log.info(f"[RESET] Added {added_count}/{len(selected)} distractor objects")
        # If we have additional initial predicates specified, set these as well
        MAX_ATTEMPTS = 10
        overall_success = False
        for _ in range(MAX_ATTEMPTS):
            success = True
            for predicate in self.init_predicates.values():
                success = predicate.set()
                if not success:
                    break
            if success:
                # Terminate early
                overall_success = True
                break
        if not overall_success:
            log.warning(f"Failed to set initial predicates after {MAX_ATTEMPTS} attempts, but proceeding anyways")

    def _reset_agent(self, env):
        # Randomize
        for robot, robot_pose in zip(env.robots, self.robot_poses):
            robot.set_position_orientation(*self.randomize_object_pose(
                *robot_pose,
                max_xyz_offset=self.robot_xyz_randomization if self.robot_xyz_randomization is not None else th.zeros(3),
                max_z_rotation=self.robot_z_rot_randomization if self.robot_z_rot_randomization is not None else 0.0,
            ))

            # Reset qpos
            robot.reset()

            # Randomize qpos
            if self.robot_joint_randomization is not None:
                arm_idxs = th.cat([robot.arm_control_idx[arm] for arm in robot.arm_names])
                qpos = robot.get_joint_positions()[arm_idxs]
                noise = (th.rand_like(qpos) * 2 - 1) * self.robot_joint_randomization
                robot.set_joint_positions(qpos + noise, indices=arm_idxs)

            robot.keep_still()

    def _get_obs(self, env):
        # Check and update milestones every step (ensures they're always current)
        self.check_milestones(env)
        

        info = {
            "milestones": self.milestones_achieved.copy(),
            "milestone_progress": self.get_milestone_progress(),
        }
        return dict(), info

    def step(self, env, action):
        """Override step to add milestones to info dict."""
        reward, done, info = super().step(env, action)
        info["milestones"] = self.milestones_achieved.copy()
        return reward, done, info

    def _create_reward_functions(self):
        rewards = dict()

        def milestone_potential(env):
            """
            Returns value between 0 and 1 based on milestone completion.
            Milestones are "sticky" - once achieved, they stay achieved for the episode.
            This provides a monotonically increasing potential.
            """
            self.check_milestones(env)

            # Return milestone progress (0.0 to 1.0)
            return self.get_milestone_progress()

        rewards["milestone"] = AbsoluteReward(
            reward_fcn=milestone_potential,
        )

        return rewards

    def _create_termination_conditions(self):
        terminations = dict()

        terminations["timeout"] = Timeout(max_steps=self._termination_config["max_steps"])
        # Goal predicates are filled in later by update_scene(), once group objects are known
        terminations["predicates"] = MultiPredicate(predicates=[])

        return terminations

    def _create_predicates(self, predicates_all, predicates_any, predicates_specific):
        predicates = dict()
        for predicate_type, type_str, predicate_infos in zip(
                (PredicateType.ALL, PredicateType.ANY, PredicateType.SPECIFIC),
                ("all", "any", "specific"),
                (predicates_all, predicates_any, predicates_specific),
        ):
            if predicate_infos is not None:
                for i, predicate_info in enumerate(predicate_infos):
                    # Determine all valid object instances that satisfy the given group
                    group = predicate_info["group"]
                    objs = self.group_objs[group]
                    state_name = predicate_info["state"]
                    obj_state_val = predicate_info["value"]
                    obj_state_kwargs = predicate_info.get("state_kwargs", None)
                    obj_state_kwargs = dict() if obj_state_kwargs is None else obj_state_kwargs
                    
                    # Special handling for PlaceOnTop - direct AABB-based placement
                    # Much more reliable than OnTop.set_value() for custom-generated objects
                    if state_name == "PlaceOnTop":
                        other_group = predicate_info["other_group"]
                        other_objs = self.group_objs[other_group]
                        assert len(other_objs) == 1, \
                            f"PlaceOnTop requires exactly 1 object in other_group '{other_group}', found {len(other_objs)}"
                        z_offset = obj_state_kwargs.get("z_offset", 0.02)
                        for obj in objs:
                            predicates[f"predicate_{type_str}_{i}"] = DirectPlacementPredicate(
                                obj=obj,
                                target_obj=other_objs[0],
                                z_offset=z_offset,
                            )
                        continue

                    # Special handling for InsideAABB - pure AABB-based containment check
                    # Use this instead of Inside when container objects don't have meta links
                    if state_name == "InsideAABB":
                        other_group = predicate_info["other_group"]
                        other_objs = self.group_objs[other_group]
                        assert len(other_objs) == 1
                        link_name = obj_state_kwargs.get("link_name", None)
                        shrink_factor = obj_state_kwargs.get("shrink_factor", 0.0)
                        shrink_z_factor = obj_state_kwargs.get("shrink_z_factor", None)
                        volume_threshold = obj_state_kwargs.get("volume_threshold", None)
                        debug = obj_state_kwargs.get("debug", False)
                        predicates[f"predicate_{type_str}_{i}"] = InsideAABBPredicate(
                            inner_objs=objs,
                            outer_obj=other_objs[0],
                            outer_link_name=link_name,
                            shrink_factor=shrink_factor,
                            shrink_z_factor=shrink_z_factor,
                            volume_threshold=volume_threshold,
                            debug=debug,
                            expected_value=obj_state_val,
                        )
                        continue

                    # Special handling for the SimFoundry AABB-geometry states - reliable
                    # for custom meshes where OG's kinematic states misbehave
                    if state_name in ("OnTopAABB", "AboveAABB", "Lifted"):
                        other_obj = None
                        if state_name != "Lifted":
                            other_group = predicate_info["other_group"]
                            other_objs = self.group_objs[other_group]
                            assert len(other_objs) == 1, \
                                f"{state_name} requires exactly 1 object in other_group '{other_group}', found {len(other_objs)}"
                            other_obj = other_objs[0]
                        predicates[f"predicate_{type_str}_{i}"] = make_aabb_predicate(
                            state_name, objs, other_obj, obj_state_val, obj_state_kwargs,
                        )
                        continue

                    obj_state = _resolve_registered_state(state_name, f"predicates_{type_str}[{i}]")

                    # Special handling for Open state - dynamically add to objects with joints
                    if obj_state == Open:
                        for obj in objs:
                            if Open not in obj.states:
                                # Check if object has joints (required for Open state)
                                if hasattr(obj, 'n_joints') and obj.n_joints > 0:
                                    log.info(f"Dynamically adding Open state to object '{obj.name}' for goal predicate")
                                    open_state = Open(obj)
                                    open_state.initialize()
                                    obj.add_state(open_state)
                                else:
                                    log.warning(f"Cannot add Open state to '{obj.name}' - no joints found")
                    
                    # Add to kwargs if this is a relative object state
                    if issubclass(obj_state, RelativeObjectState):
                        other_group = predicate_info["other_group"]
                        other_objs = self.group_objs[other_group]
                        assert len(other_objs) == 1
                        obj_state_kwargs["other"] = other_objs[0]
                    predicates[f"predicate_{type_str}_{i}"] = Predicate(
                        objs=objs,
                        predicate_type=predicate_type,
                        obj_state=obj_state,
                        obj_state_val=obj_state_val,
                        obj_state_kwargs=obj_state_kwargs,
                    )

        return predicates

    def _create_milestone_predicates(self):
        """
        Creates milestone predicates from the milestone_predicates_config.
        Milestones are tracked during the episode - once achieved, they stay achieved.
        
        Special handling for IsGrasping: Since IsGrasping is not a default object state
        on robots, we store grasping milestones separately and check them using the
        robot's built-in is_grasping() method.
        
        Sequential dependencies: Each milestone can specify a 'requires' field with a list
        of prerequisite milestone names that must be achieved first.
        """
        self.milestone_predicates = {}
        self.milestone_grasping_checks = {}  # Special handling for IsGrasping
        self.milestones_achieved = OrderedDict()
        self.milestone_requires = {}  # Track prerequisite dependencies
        
        if self.milestone_predicates_config is None:
            return
        
        for milestone_info in self.milestone_predicates_config:
            name = milestone_info["name"]
            group = milestone_info["group"]
            objs = self.group_objs.get(group, [])
            state_name = milestone_info["state"]
            obj_state_val = milestone_info["value"]
            obj_state_kwargs = milestone_info.get("state_kwargs", None)
            obj_state_kwargs = dict() if obj_state_kwargs is None else obj_state_kwargs
            
            # Store prerequisite dependencies (if any)
            requires = milestone_info.get("requires", [])
            if isinstance(requires, str):
                requires = [requires]  # Convert single string to list
            self.milestone_requires[name] = requires
            
            # Special handling for IsGrasping - use robot's built-in method
            if state_name == "IsGrasping":
                other_group = milestone_info["other_group"]
                other_objs = self.group_objs.get(other_group, [])
                if len(other_objs) != 1:
                    log.warning(f"Milestone '{name}' requires exactly 1 object in other_group '{other_group}', found {len(other_objs)}")
                    continue
                # Store info for grasping check: (robot_objs, target_obj, expected_value)
                self.milestone_grasping_checks[name] = {
                    "robots": objs,  # robots from 'group'
                    "target_obj": other_objs[0],  # object to check if grasped
                    "expected_value": obj_state_val,
                }
                self.milestones_achieved[name] = 0
                continue
            
            # Special handling for InsideAABB - pure AABB-based containment check
            # Use this instead of Inside when container objects don't have meta links
            if state_name == "InsideAABB":
                other_group = milestone_info["other_group"]
                other_objs = self.group_objs.get(other_group, [])
                if len(other_objs) != 1 or len(objs) != 1:
                    log.warning(f"Milestone '{name}' requires exactly 1 object in each group")
                    continue
                # Store info for InsideAABB check
                if not hasattr(self, 'milestone_inside_aabb_checks'):
                    self.milestone_inside_aabb_checks = {}
                # Extract optional link_name, shrink_factor, shrink_z_factor, volume_threshold, and debug from state_kwargs
                link_name = obj_state_kwargs.get("link_name", None)
                shrink_factor = obj_state_kwargs.get("shrink_factor", 0.0)
                shrink_z_factor = obj_state_kwargs.get("shrink_z_factor", None)
                volume_threshold = obj_state_kwargs.get("volume_threshold", None)
                debug = obj_state_kwargs.get("debug", False)
                self.milestone_inside_aabb_checks[name] = {
                    "inner_obj": objs[0],
                    "outer_obj": other_objs[0],
                    "link_name": link_name,
                    "shrink_factor": shrink_factor,
                    "shrink_z_factor": shrink_z_factor,
                    "volume_threshold": volume_threshold,
                    "debug": debug,
                    "expected_value": obj_state_val,
                }
                self.milestones_achieved[name] = 0
                continue

            # Special handling for the SimFoundry AABB-geometry states. These classes
            # implement _step, so the standard check_milestones loop evaluates them
            # (including `requires` gating) - no bespoke check dict needed.
            if state_name in ("OnTopAABB", "AboveAABB", "Lifted"):
                other_obj = None
                if state_name != "Lifted":
                    other_group = milestone_info["other_group"]
                    other_objs = self.group_objs.get(other_group, [])
                    if len(other_objs) != 1:
                        log.warning(f"Milestone '{name}' requires exactly 1 object in other_group '{other_group}', found {len(other_objs)}")
                        continue
                    other_obj = other_objs[0]
                self.milestone_predicates[name] = make_aabb_predicate(
                    state_name, objs, other_obj, obj_state_val, obj_state_kwargs,
                )
                self.milestones_achieved[name] = 0
                continue

            # Standard object state predicate
            obj_state = _resolve_registered_state(state_name, f"milestone '{name}'")
            
            # Special handling for Open state - dynamically add to objects with joints
            if obj_state == Open:
                for obj in objs:
                    if Open not in obj.states:
                        # Check if object has joints (required for Open state)
                        if hasattr(obj, 'n_joints') and obj.n_joints > 0:
                            log.info(f"Dynamically adding Open state to object '{obj.name}' for milestone '{name}'")
                            open_state = Open(obj)
                            open_state.initialize()
                            obj.add_state(open_state)
                        else:
                            log.warning(f"Cannot add Open state to '{obj.name}' - no joints found")
                            continue
            
            # Add to kwargs if this is a relative object state
            if issubclass(obj_state, RelativeObjectState):
                other_group = milestone_info["other_group"]
                other_objs = self.group_objs.get(other_group, [])
                if len(other_objs) == 1:
                    obj_state_kwargs["other"] = other_objs[0]
                else:
                    log.warning(f"Milestone '{name}' requires exactly 1 object in other_group '{other_group}', found {len(other_objs)}")
                    continue
            
            self.milestone_predicates[name] = Predicate(
                objs=objs,
                predicate_type=PredicateType.ANY,  # Any object in group satisfying is enough
                obj_state=obj_state,
                obj_state_val=obj_state_val,
                obj_state_kwargs=obj_state_kwargs,
            )
            self.milestones_achieved[name] = 0
        
        log.info(f"Created {len(self.milestone_predicates)} milestone predicates and {len(self.milestone_grasping_checks)} grasping checks")

    def check_milestones(self, env):
        """
        Check all milestone predicates and update milestones_achieved.
        Once a milestone is achieved, it stays achieved for the episode.
        
        Sequential checking: Only checks milestones whose prerequisites have been achieved.
        
        Returns:
            dict: Current milestone status {name: achieved}
        """
        from omnigibson.controllers import IsGraspingState
        
        # Helper to check if prerequisites are satisfied
        def prerequisites_satisfied(milestone_name):
            required = self.milestone_requires.get(milestone_name, [])
            return all(self.milestones_achieved.get(req, 0) for req in required)
        
        # Check standard predicates
        for name, predicate in self.milestone_predicates.items():
            if not self.milestones_achieved[name]:
                # Only check if prerequisites are satisfied
                if not prerequisites_satisfied(name):
                    continue
                    
                # Check if milestone is now satisfied
                if predicate._step(self, env, None):
                    self.milestones_achieved[name] = 1
                    log.info(f"Milestone achieved: {name}")
        
        # Check grasping milestones using robot's built-in is_grasping() method
        # TODO: can this be simplified?
        for name, grasp_info in self.milestone_grasping_checks.items():
            if not self.milestones_achieved[name]:
                # Only check if prerequisites are satisfied
                if not prerequisites_satisfied(name):
                    continue
                    
                robots = grasp_info["robots"]
                target_obj = grasp_info["target_obj"]
                expected_value = grasp_info["expected_value"]
                
                # Check if any robot is grasping the target object
                # Try both physical grasping and assisted grasping (OR condition)
                is_grasping = False
                
                # Check physical grasping first
                for robot in robots:
                    for arm in robot.arm_names:
                        grasp_state = robot.is_grasping(arm=arm, candidate_obj=target_obj)
                        if grasp_state == IsGraspingState.TRUE:
                            is_grasping = True
                            log.debug(f"Physical grasp detected for {name}: {target_obj.name}")
                            break
                    if is_grasping:
                        break
                
                # If not physically grasping, check assisted grasping
                if not is_grasping:
                    for robot in robots:
                        for arm in robot.arm_names:
                            if robot._ag_obj_in_hand[arm] is not None:
                                if robot._ag_obj_in_hand[arm] == target_obj:
                                    is_grasping = True
                                    log.debug(f"Assisted grasp detected for {name}: {target_obj.name}")
                                    break
                        if is_grasping:
                            break
                
                if is_grasping == expected_value:
                    self.milestones_achieved[name] = 1
                    log.info(f"Milestone achieved: {name}")
        
        # Check InsideAABB milestones (pure AABB-based containment, no meta links required)
        if hasattr(self, 'milestone_inside_aabb_checks'):
            for name, aabb_info in self.milestone_inside_aabb_checks.items():
                if not self.milestones_achieved[name]:
                    # Only check if prerequisites are satisfied
                    if not prerequisites_satisfied(name):
                        continue
                        
                    inner_obj = aabb_info["inner_obj"]
                    outer_obj = aabb_info["outer_obj"]
                    link_name = aabb_info.get("link_name", None)
                    shrink_factor = aabb_info.get("shrink_factor", 0.0)
                    shrink_z_factor = aabb_info.get("shrink_z_factor", None)
                    volume_threshold = aabb_info.get("volume_threshold", None)
                    debug = aabb_info.get("debug", False)
                    expected_value = aabb_info["expected_value"]
                    
                    is_inside = check_inside_aabb(inner_obj, outer_obj, link_name, shrink_factor, shrink_z_factor, volume_threshold, debug)
                    
                    if is_inside == expected_value:
                        self.milestones_achieved[name] = 1
                        log.info(f"Milestone achieved: {name}")           
        
        return self.milestones_achieved.copy()

    def reset_milestones(self):
        """Reset all milestones to not achieved (called at episode start)."""
        for name in self.milestones_achieved:
            self.milestones_achieved[name] = 0

    def get_milestone_progress(self):
        """
        Returns the fraction of milestones achieved (0.0 to 1.0).
        """
        if not self.milestones_achieved:
            return 0.0
        achieved = sum(1 for v in self.milestones_achieved.values() if v)
        return achieved / len(self.milestones_achieved)

    # Implementation lives in task_utils; exposed as a static method so callers can keep
    # using self.randomize_object_pose / PickPlaceTask.randomize_object_pose.
    randomize_object_pose = staticmethod(randomize_object_pose)

    @classproperty
    def valid_scene_types(cls):
        # Any scene can be used
        return {Scene}

    @classproperty
    def default_termination_config(cls):
        return {
            "max_steps": 500,
        }

    @classproperty
    def default_reward_config(cls):
        return {
            "r_potential": 1.0,
        }
