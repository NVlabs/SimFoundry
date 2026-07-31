# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for annotating demonstrations with task-relevant signals.

This module provides detectors for identifying key events (pick, place, etc.)
during demonstration playback.
"""

import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

import h5py
import torch as th
import omnigibson as og
from omnigibson.macros import gm, macros
from omnigibson.utils.python_utils import h5py_group_to_torch
from omnigibson.object_states import ContactBodies, OnTop, Inside, Open
import omnigibson.utils.transform_utils as T
from pathlib import Path

from simfoundry.utils.og_utils import draw_trajectory_gradient, clear_trajectory_visualization


@dataclass
class PickSignal:
    """Annotation for a pick event"""
    type: str = "pick"
    frame_idx: int = 0
    object_name: str = ""
    object_category: str = ""
    # object_model: str = ""
    contact_link: str = ""


@dataclass
class PlaceSignal:
    """Annotation for a place event"""
    type: str = "place"
    release_frame_idx: int = 0
    contact_frame_idx: int = 0
    released_object_name: str = ""
    released_object_category: str = ""
    # released_object_model: str = ""
    target_object_name: str = ""
    target_object_category: str = ""
    # target_object_model: str = ""
    target_contact_link: str = ""


@dataclass
class OpenSignal:
    """Annotation for an open event (articulated object)"""
    type: str = "open"
    contact_frame_idx: int = 0
    release_frame_idx: int = 0
    object_name: str = ""
    object_category: str = ""
    # object_model: str = ""
    contact_link: str = ""


@dataclass
class CloseSignal:
    """Annotation for a close event (articulated object)"""
    type: str = "close"
    contact_frame_idx: int = 0
    release_frame_idx: int = 0
    object_name: str = ""
    object_category: str = ""
    # object_model: str = ""
    contact_link: str = ""


class AnnotationManager:
    """Detects pick/place/open/close by monitoring object state changes"""

    def __init__(self, env, robot_id: int = 0, invert_gripper_action: bool = False, check_gripper_open_during_place: bool = True):
        self.env = env
        self.robot = env.robots[robot_id]
        self.robot_action_idx_offset = 0
        for robot in env.robots[:robot_id]:
            self.robot_action_idx_offset += robot.action_dim
        self.prev_states = {}  # Track previous states of all objects
        self.signals = []
        self.invert_gripper_action = invert_gripper_action
        self.check_gripper_open_during_place = check_gripper_open_during_place
        self.pick_is_active = False

    def gripper_action_is_closing(self, gripper_action: float) -> bool:
        """Check if the gripper action is closing"""
        return gripper_action < 0.5 if not self.invert_gripper_action else gripper_action > 0.5
        
    def reset(self):
        """Reset for new episode"""
        self.prev_states = {}
        self.signals = []
        self.pick_is_active = False

    def step(self, frame_idx: int, action: th.Tensor, env: Any = None):
        """
        Detect state changes in objects and emit signals

        Args:
            frame_idx: Current frame index
            action: Action tensor
            env: Environment (unused, for callback compatibility)
        """
        self.current_frame_idx = frame_idx
        # Get gripper action
        arm = self.robot.default_arm
        gripper_action = action[self.robot_action_idx_offset + self.robot.gripper_action_idx[arm]].item()

        # PICK DETECTION: Check for grasps using contact-based heuristic
        # Pick happens when: gripper closing + both fingers contact object's base link
        if self.gripper_action_is_closing(gripper_action):  # Gripper is closing
            if not self.pick_is_active:
                picked_obj = self._detect_pick_from_contacts()
                if picked_obj is not None:
                    signal = PickSignal(
                        frame_idx=frame_idx,
                        object_name=picked_obj.name,
                        object_category=picked_obj.category,
                        # object_model=picked_obj.model,
                        contact_link=""  # Not tracked in contact-based approach
                    )
                    self.signals.append(asdict(signal))
                    print(f"  Detected PICK at frame {frame_idx}: {picked_obj.category} ({picked_obj.name})")
                    self.pick_is_active = True
        else:
            # Gripper is opening - reset picked flags for all objects to allow re-detection
            for obj_state in self.prev_states.values():
                if 'picked' in obj_state:
                    obj_state['picked'] = False
            self.pick_is_active = False

        for obj in self.env.scene.objects:
            if obj == self.robot:
                continue

            obj_name = obj.name

            # Get current states
            curr_on_top = self._get_object_on_top_of(obj) if OnTop in obj.states else None
            curr_inside = self._get_object_inside_of(obj) if Inside in obj.states else None
            curr_open = obj.states[Open].get_value() if Open in obj.states else None

            # Initialize tracking for new objects
            if obj_name not in self.prev_states:
                # self.prev_states[obj_name] = {
                #     'on_top_of': curr_on_top,
                #     'inside_of': curr_inside,
                #     'open': curr_open,
                #     'picked': False  # Track if we've already detected a pick for this object
                # }
                self.prev_states[obj_name] = {
                    'on_top_of': curr_on_top,
                    'inside_of': curr_inside,
                    'open': curr_open,
                    'picked': False,  # Track if we've already detected a pick for this object
                    'placed_on': None # Track objects that this object has been placed on
                }
                continue

            prev = self.prev_states[obj_name]

            # PLACE: Object becomes OnTop/Inside AND is in ContactBodies. Then, triggers the next step the gripper opening
            # All three criteria must be satisfied simultaneously (gripper check is optional)
            was_supported = prev['on_top_of'] is not None or prev['inside_of'] is not None
            is_supported = curr_on_top is not None or curr_inside is not None
            gripper_open_condition = not self.gripper_action_is_closing(gripper_action) if self.check_gripper_open_during_place else True
            if not was_supported and is_supported and gripper_open_condition:
                target_obj = curr_on_top if curr_on_top is not None else curr_inside

                # Check if object was already placed on this object
                if self.prev_states[obj_name]["placed_on"] == target_obj:
                    # Actively already on this object, skip
                    continue

                signal = PlaceSignal(
                    release_frame_idx=frame_idx,  # Same as contact for compatibility
                    contact_frame_idx=frame_idx,
                    released_object_name=obj.name,
                    released_object_category=obj.category,
                    # released_object_model=obj.model,
                    target_object_name=target_obj.name,
                    target_object_category=target_obj.category,
                    # target_object_model=target_obj.model,
                    target_contact_link=""  # Not tracked in state-based approach
                )
                self.signals.append(asdict(signal))
                self.pick_is_active = False
                self.prev_states[obj_name]["placed_on"] = target_obj
                print(f"  Detected PLACE at frame {frame_idx}: "
                        f"{obj.category} ({obj.name}) onto {target_obj.category} ({target_obj.name})")

            # OPEN/CLOSE: Open state changed
            if curr_open is not None and prev['open'] is not None:
                if curr_open != prev['open']:
                    # Query which link is in contact with robot grippers
                    contact_link = self._get_contacted_link(obj)
                    if contact_link is None:
                        # No contact detected, skip this signal
                        print(f"  Warning: Open/Close state changed for {obj.category} ({obj.name}) but no gripper contact detected")

                    if curr_open and contact_link:
                        # Opened
                        signal = OpenSignal(
                            contact_frame_idx=frame_idx,  # For compatibility
                            release_frame_idx=frame_idx,
                            object_name=obj.name,
                            object_category=obj.category,
                            # object_model=obj.model,
                            contact_link=contact_link
                        )
                        self.signals.append(asdict(signal))
                        print(f"  Detected OPEN at frame {frame_idx}: {obj.category} ({obj.name}) link={contact_link}")
                    elif not curr_open and contact_link:
                        # Closed
                        signal = CloseSignal(
                            contact_frame_idx=frame_idx,  # For compatibility
                            release_frame_idx=frame_idx,
                            object_name=obj.name,
                            object_category=obj.category,
                            # object_model=obj.model,
                            contact_link=contact_link
                        )
                        self.signals.append(asdict(signal))
                        print(f"  Detected CLOSE at frame {frame_idx}: {obj.category} ({obj.name}) link={contact_link}")

            # Update previous state
            self.prev_states[obj_name] = {
                'on_top_of': curr_on_top,
                'inside_of': curr_inside,
                'open': curr_open,
                'picked': prev.get('picked', False),
                'placed_on': prev.get('placed_on', None),
            }

    def _get_object_on_top_of(self, obj):
        """Get what object this object is on top of"""
        if OnTop not in obj.states:
            return None

        # Check all other objects to see if obj is on top of them
        for other_obj in self.env.scene.objects:
            if other_obj == obj or other_obj == self.robot:
                continue
            try:
                if obj.states[OnTop].get_value(other_obj):
                    return other_obj
            except:
                pass

        return None

    def _get_object_inside_of(self, obj):
        """Get what object this object is inside of"""
        if Inside not in obj.states:
            return None

        # Check all other objects to see if obj is inside them
        for other_obj in self.env.scene.objects:
            if other_obj == obj or other_obj == self.robot:
                continue
            try:
                if obj.states[Inside].get_value(other_obj):
                    return other_obj
            except:
                pass

        return None

    def _detect_pick_from_contacts(self):
        """
        Detect pick event using contact-based heuristic.

        Pick happens when:
        1. Both gripper finger links are in contact with an object
        2. The contact is with the object's base/root link (to rule out articulated objects)

        Returns:
            Object or None: The object being picked, or None if no valid pick detected
        """
        # Get gripper contacts
        # contact_prims: set of prim_paths in contact with gripper fingers
        # robot_contact_links: dict mapping contact prim_path -> set of robot link prim_paths touching it
        arm = self.robot.default_arm
        contact_prims, robot_contact_links = self.robot._find_gripper_contacts(arm=arm)

        if not contact_prims:
            return None

        # Get finger link prim paths for checking
        finger_link_paths = {link.prim_path for link in self.robot.finger_links[arm]}

        # Check each contacted object
        for contact_prim_path in contact_prims:
            # Parse object and link from prim path
            # Format: /World/.../object_name/link_name
            try:
                # Find the object that this prim path belongs to
                # Format: /World/scene_X/object_name/...
                contact_obj_prim_path = "/".join(contact_prim_path.split('/')[:4])
                obj = self.env.scene.object_registry("prim_path", contact_obj_prim_path, None)

                if obj is None:
                    continue

                # Check if we've already detected a pick for this object
                if obj.name in self.prev_states and self.prev_states[obj.name].get('picked', False):
                    continue

                # Extract the link name from the contact prim path
                # contact_prim_path is like: /World/.../object_name/link_name
                link_name = contact_prim_path.split('/')[-1]

                # Check if this is the root/base link
                if link_name != obj.root_link_name:
                    # Contact is with a non-base link (e.g., articulated part), skip
                    continue

                # Check if both fingers are in contact with this object
                robot_links_in_contact = robot_contact_links[contact_prim_path]
                fingers_in_contact = finger_link_paths.intersection(robot_links_in_contact)

                if len(fingers_in_contact) >= 1:
                    # At least one finger is touching the base link of this object - this is a pick!
                    # Mark this object as picked to avoid duplicate detections
                    if obj.name not in self.prev_states:
                        self.prev_states[obj.name] = {
                            'on_top_of': None,
                            'inside_of': None,
                            'open': None,
                            'picked': True
                        }
                    else:
                        self.prev_states[obj.name]['picked'] = True

                    return obj

            except Exception as e:
                # If there's any error parsing the prim path, skip this contact
                print(f"  Warning: Error processing contact prim path {contact_prim_path}: {e}")
                continue

        return None

    def _robot_in_contact_with(self, obj):
        """Check if robot gripper is in contact with this object"""
        # Use robot's built-in gripper contact detection
        arm = self.robot.default_arm
        contact_prims, _ = self.robot._find_gripper_contacts(arm=arm)

        # Check if any contact prim belongs to this object
        for contact_prim_path in contact_prims:
            if contact_prim_path.startswith(obj.prim_path):
                return True

        return False

    def _get_contacted_link(self, obj):
        """
        Get the link name of an object being manipulated by the robot

        This uses joint velocity to determine which link is being moved, which is
        more robust than contact detection (especially for closing actions where
        the EEF may lose contact quickly).

        Fallback to contact detection if velocity-based approach fails.

        Args:
            obj: Object to check for manipulation

        Returns:
            str or None: Link name if being manipulated, None otherwise
        """
        # Primary method: Check which joint has the highest velocity
        # This works for both opening (grasping handle) and closing (pushing with EEF)
        if obj.n_joints > 0:
            max_vel = 0.0
            max_vel_joint = None

            # Iterate through all joints and check their velocities
            for joint in obj.joints.values():
                # Get joint state (pos, vel, effort)
                _, vel, _ = joint.get_state()

                # Calculate absolute velocity
                if vel.ndim == 0:
                    abs_vel = abs(vel.item())
                else:
                    abs_vel = th.max(th.abs(vel)).item()

                # Track joint with highest velocity
                if abs_vel > max_vel:
                    max_vel = abs_vel
                    max_vel_joint = joint

            # Only consider if velocity is significant (> 0.01 rad/s or m/s)
            if max_vel > 0.01 and max_vel_joint is not None:
                # Get the child link (body1) attached to this joint
                body1_prim_path = max_vel_joint.body1
                if body1_prim_path:
                    # Extract link name from prim path
                    # Format: /World/.../object_name/link_name
                    link_name = body1_prim_path.split('/')[-1]
                    return link_name

        return None

    def get_annotations(self) -> list:
        """Get all annotations for current episode"""
        return self.signals

    def episode_start_callback(self, episode_id: int, env: Any):
        """Callback for episode start"""
        self.reset()


class MinimalPlaybackWrapper:
    """
    Minimal wrapper for replaying HDF5 data without saving outputs

    This is a lightweight alternative to DataPlaybackWrapper that only
    replays trajectories without writing to disk.

    Supports optional step callbacks for custom processing during playback.
    """

    def __init__(self, env, input_hdf5, step_callback=None, episode_start_callback=None):
        """
        Args:
            env: OmniGibson environment
            input_hdf5: h5py.File handle to input HDF5 file
            step_callback: Optional callable(frame_idx, action, env) called at each step
            episode_start_callback: Optional callable(episode_id, env) called at episode start
        """
        self.env = env
        self.input_hdf5 = input_hdf5
        self.scene_file = json.loads(input_hdf5["data"].attrs["scene_file"])
        self.step_callback = step_callback
        self.episode_start_callback = episode_start_callback

        # with macros.unlocked():
        #     macros.prims.entity_prim.DEFAULT_SLEEP_THRESHOLD = 0.0

    def playback_episode(self, episode_id: int) -> Any:
        """
        Playback a single episode with optional callbacks

        Args:
            episode_id: Episode index to replay

        Returns:
            None (callbacks handle their own data collection)
        """
        data_grp = self.input_hdf5["data"]
        assert f"demo_{episode_id}" in data_grp, f"No valid episode with ID {episode_id} found!"
        traj_grp = data_grp[f"demo_{episode_id}"]

        print(f"\nReplaying episode {episode_id}...")

        # Load episode data
        transitions = json.loads(traj_grp.attrs["transitions"])
        traj_grp = h5py_group_to_torch(traj_grp)
        init_metadata = traj_grp["init_metadata"]
        action = traj_grp["action"]
        state = traj_grp["state"]
        state_size = traj_grp["state_size"]

        # Reset environment
        self.env.scene.restore(self.scene_file, update_initial_file=True)

        # Reset object attributes from metadata
        with og.sim.stopped():
            for attr, vals in init_metadata.items():
                assert len(vals) == self.env.scene.n_objects
            for i, obj in enumerate(self.env.scene.objects):
                for attr, vals in init_metadata.items():
                    val = vals[i]
                    setattr(obj, attr, val.item() if val.ndim == 0 else val)

        self.env.reset()

        # Disable robot control during playback
        for robot in self.env.robots:
            robot.control_enabled = False

        # Restore initial state
        og.sim.load_state(state[0, : int(state_size[0])], serialized=True)

        # Call episode start callback if provided
        if self.episode_start_callback:
            self.episode_start_callback(episode_id, self.env)

        # Step through environment
        for i, (a, s, ss) in enumerate(zip(action, state[1:], state_size[1:])):
            # Execute any transitions at this step
            if str(i) in transitions:
                # TODO: Handle transitions if needed
                pass

            # Restore sim state
            og.sim.load_state(s[: int(ss)], serialized=True)

            # Wake up all objects to avoid contact check failure
            for obj in self.env.scene.objects:
                obj.wake() # TODO: properly disable sleeping

            # Call step callback if provided
            if self.step_callback:
                self.step_callback(frame_idx=i, action=a, env=self.env)

            # Step simulation for rendering
            og.sim.step()

        return None


class WaypointExtractor:
    """
    Extracts object-centric waypoint trajectories around signal events.

    For each signal (pick/place), extracts a trajectory segment in the reference
    object's frame, including gripper actions.
    """

    def __init__(
        self,
        env,
        signals: List[Dict],
        pick_min_start_distance: float,
        pick_max_distance: float,
        place_min_start_distance: float,
        place_max_distance: float,
        open_min_start_distance: float = 0.15,
        open_max_distance: float = 0.50,
        close_min_start_distance: float = 0.15,
        close_max_distance: float = 0.50,
        end_z_threshold: float = 0.05,
        max_frames: Optional[int] = None,
        robot_id: int = 0,
        debug: bool = False,
        eef_z_offset: float = 0.0,
        merge_sequences: bool = False,
        use_open_grasp_signal: bool = False,
        open_grasp_max_distance: float = 0.15,
    ):
        """
        Args:
            env: OmniGibson environment
            signals: List of signal dictionaries from annotation stage
            pick_min_start_distance: Min distance from object to start extracting pick trajectory (meters)
            pick_max_distance: Max distance from object to stop extracting (meters)
            place_min_start_distance: Min distance from target to start extracting place trajectory (meters)
            place_max_distance: Max distance from target to stop extracting (meters)
            open_min_start_distance: Min distance from object to start extracting open trajectory (meters, default 0.15)
            open_max_distance: Max distance from object to stop extracting open trajectory (meters, default 0.50)
            close_min_start_distance: Min distance from object to start extracting close trajectory (meters, default 0.15)
            close_max_distance: Max distance from object to stop extracting close trajectory (meters, default 0.50)
            end_z_threshold: Max distance to keep extracting (euclidean from signal EEF for pick/place, relative vertical to link for open/close, default 0.05)
            max_frames: Maximum frame index in episode (for boundary checking)
            robot_id: Robot ID to extract waypoints for
            debug: Whether to print debug information
            eef_z_offset: Z offset in local EEF frame to apply when tracking EEF pose (meters)
            merge_sequences: If True, merge adjacent subtask sequences that share the same reference
                object frame, filling in gap frames from the frame buffer so the full trajectory
                between them is preserved without requiring freespace interpolation
            use_open_grasp_signal: If True, for "open" signals, verify gripper is closed at
                the signal frame and use gripper open / L2 distance from release position as the
                stopping condition instead of the standard distance-based thresholds
            open_grasp_max_distance: When use_open_grasp_signal is True, the L2 distance threshold
                from the EEF position where the gripper first opens. Waypoints continue until
                this distance is exceeded (meters)
        """
        self.env = env
        self.robot = env.robots[robot_id]
        self.robot_action_idx_offset = 0
        for robot in env.robots[:robot_id]:
            self.robot_action_idx_offset += robot.action_dim
        self.signals = signals
        self.pick_min_start_distance = pick_min_start_distance
        self.pick_max_distance = pick_max_distance
        self.place_min_start_distance = place_min_start_distance
        self.place_max_distance = place_max_distance
        self.open_min_start_distance = open_min_start_distance
        self.open_max_distance = open_max_distance
        self.close_min_start_distance = close_min_start_distance
        self.close_max_distance = close_max_distance
        self.z_end_threshold = end_z_threshold
        self.max_frames = max_frames
        self.debug = debug
        self.eef_z_offset = eef_z_offset
        self.merge_sequences = merge_sequences
        self.use_open_grasp_signal = use_open_grasp_signal
        self.open_grasp_max_distance = open_grasp_max_distance

        # Storage for extracted waypoints
        self.waypoints = []

        # Current frame buffer for extracting segments
        self.frame_buffer = []
        self.current_frame_idx = 0

        # Track which signals we've processed
        self.signal_idx = 0

    def reset(self):
        """Reset for new episode"""
        self.waypoints = []
        self.frame_buffer = []
        self.current_frame_idx = 0
        self.signal_idx = 0

    def step(self, frame_idx: int, action: th.Tensor, env: Any):
        """
        Process one frame and extract waypoints if needed

        Args:
            frame_idx: Current frame index
            action: Action tensor [7] (delta_pos, delta_aa, gripper)
            env: Environment
        """
        self.current_frame_idx = frame_idx

        # Get current EEF pose in robot frame
        eef_pos_robot, eef_quat_robot = self.robot.get_relative_eef_pose(arm="default")

        # Convert to tensors if needed
        if not isinstance(eef_pos_robot, th.Tensor):
            eef_pos_robot = th.tensor(eef_pos_robot, dtype=th.float32)
        if not isinstance(eef_quat_robot, th.Tensor):
            eef_quat_robot = th.tensor(eef_quat_robot, dtype=th.float32)

        # Apply EEF z-offset in local EEF frame if specified
        if self.eef_z_offset != 0.0:
            eef_pos_robot, eef_quat_robot = self._apply_eef_z_offset(
                eef_pos_robot, eef_quat_robot, self.eef_z_offset
            )

        # Robot world pose at this frame
        robot_pos_world, robot_quat_world = self.robot.get_position_orientation()

        # Get gripper action (last element of action)
        gripper_action = action[self.robot_action_idx_offset + self.robot.gripper_action_idx[self.robot.default_arm]].item()

        # Store all object poses at this frame (for accurate waypoint extraction)
        object_poses = {}
        link_poses = {}  # Store poses of specific links for articulated objects
        for obj in env.scene.objects:
            if obj != self.robot:
                pos, quat = obj.get_position_orientation()
                if not isinstance(pos, th.Tensor):
                    pos = th.tensor(pos, dtype=th.float32)
                if not isinstance(quat, th.Tensor):
                    quat = th.tensor(quat, dtype=th.float32)
                object_poses[obj.name] = {
                    'pos': pos.clone(),
                    'quat': quat.clone()
                }

                # For articulated objects, also store poses of all links
                # (we'll use these for open/close waypoint extraction)
                if Open in obj.states:
                    for link_name, link in obj.links.items():
                        link_pos, link_quat = link.get_position_orientation()
                        if not isinstance(link_pos, th.Tensor):
                            link_pos = th.tensor(link_pos, dtype=th.float32)
                        if not isinstance(link_quat, th.Tensor):
                            link_quat = th.tensor(link_quat, dtype=th.float32)
                        link_poses[(obj.name, link_name)] = {
                            'pos': link_pos.clone(),
                            'quat': link_quat.clone()
                        }

        # Store frame data in buffer
        self.frame_buffer.append({
            'frame_idx': frame_idx,
            'eef_pos_robot': eef_pos_robot.clone(),
            'eef_quat_robot': eef_quat_robot.clone(),
            'robot_pos_world': robot_pos_world.clone(),
            'robot_quat_world': robot_quat_world.clone(),
            'gripper_action': gripper_action,
            'object_poses': object_poses,  # Store all object poses
            'link_poses': link_poses,  # Store all link poses for articulated objects
        })

        # Check if we need to extract waypoints for any signals
        self._check_and_extract_waypoints()

    def _check_and_extract_waypoints(self):
        """Check if current frame triggers waypoint extraction for any signal"""
        while self.signal_idx < len(self.signals):
            signal = self.signals[self.signal_idx]

            # Determine the signal frame and reference details based on signal type
            if signal['type'] == 'pick':
                signal_frame = signal['frame_idx']
                max_distance = self.pick_max_distance
                ref_obj_name = signal['object_name']
            elif signal['type'] == 'place':
                # For place, use the contact frame (when object contacts target)
                signal_frame = signal['contact_frame_idx']
                max_distance = self.place_max_distance
                ref_obj_name = signal['target_object_name']
            elif signal['type'] == 'open':
                # For open, use the release frame (when robot releases the handle)
                signal_frame = signal['release_frame_idx']
                max_distance = self.open_max_distance
                ref_obj_name = signal['object_name']
            elif signal['type'] == 'close':
                # For close, use the release frame (when robot releases the handle)
                signal_frame = signal['release_frame_idx']
                max_distance = self.close_max_distance
                ref_obj_name = signal['object_name']
            else:
                self.signal_idx += 1
                continue

            # Wait until we've at least reached the signal frame
            if self.current_frame_idx < signal_frame:
                break

            # Need the signal frame in the buffer to compute relative z offset
            signal_frame_data = next((f for f in self.frame_buffer if f['frame_idx'] == signal_frame), None)
            if signal_frame_data is None:
                break

            # For open/close, pass the link name
            ref_link_name = signal.get('contact_link', None) if signal['type'] in ['open', 'close'] else None
            end_reached = self._end_condition_met(ref_obj_name, max_distance, signal_frame_data, signal['type'], ref_link_name)
            at_episode_end = self.max_frames is not None and self.current_frame_idx >= self.max_frames - 1

            # If we haven't satisfied end conditions and we're not at the end of the episode yet, keep collecting frames
            if not end_reached and not at_episode_end:
                break

            # Extract waypoints for this signal
            self._extract_waypoints_for_signal(signal, signal_frame, available_end_frame=self.current_frame_idx)
            self.signal_idx += 1

    def _apply_eef_z_offset(
        self,
        eef_pos_robot: th.Tensor,
        eef_quat_robot: th.Tensor,
        z_offset: float
    ) -> tuple:
        """
        Apply a z-offset in the local EEF frame to the EEF pose.

        Args:
            eef_pos_robot: EEF position in robot frame [3]
            eef_quat_robot: EEF quaternion in robot frame [4]
            z_offset: Z offset in local EEF frame (meters)

        Returns:
            tuple: (new_eef_pos_robot, eef_quat_robot) - Updated EEF pose with offset applied
        """
        # Create the offset in local EEF frame (z-direction)
        offset_local = th.tensor([0.0, 0.0, z_offset], dtype=th.float32)

        # Transform offset from local EEF frame to robot frame using the EEF orientation
        eef_rot_mat = T.quat2mat(eef_quat_robot)
        offset_robot = eef_rot_mat @ offset_local

        # Apply offset to position
        new_eef_pos_robot = eef_pos_robot + offset_robot

        return new_eef_pos_robot, eef_quat_robot

    def _eef_world_from_frame(self, frame_data: Dict) -> th.Tensor:
        """Compute EEF world position for a stored frame."""
        robot_pos = frame_data['robot_pos_world']
        robot_quat = frame_data['robot_quat_world']
        if not isinstance(robot_pos, th.Tensor):
            robot_pos = th.tensor(robot_pos, dtype=th.float32)
        if not isinstance(robot_quat, th.Tensor):
            robot_quat = th.tensor(robot_quat, dtype=th.float32)

        eef_pos_robot = frame_data['eef_pos_robot']
        eef_quat_robot = frame_data['eef_quat_robot']
        # # Apply z-offset to eef_pos_robot
        # eef_pos_robot, _ = self._apply_eef_z_offset(eef_pos_robot, eef_quat_robot, self.eef_z_offset)
        if not isinstance(eef_pos_robot, th.Tensor):
            eef_pos_robot = th.tensor(eef_pos_robot, dtype=th.float32)
        if not isinstance(eef_quat_robot, th.Tensor):
            eef_quat_robot = th.tensor(eef_quat_robot, dtype=th.float32)

        robot_T_world = T.pose2mat((robot_pos, robot_quat))
        eef_T_robot = T.pose2mat((eef_pos_robot, eef_quat_robot))
        eef_T_world = robot_T_world @ eef_T_robot
        eef_pos_world, _ = T.mat2pose(eef_T_world)
        return eef_pos_world

    def _end_condition_met(self, ref_obj_name: str, max_distance: float, signal_frame_data: Dict, signal_type: str, ref_link_name: str = None) -> bool:
        """
        Check whether the current frame satisfies any end condition for waypoint extraction.

        Args:
            ref_obj_name: Reference object for this segment
            max_distance: Maximum allowed Euclidean distance to continue extracting
            signal_frame_data: Stored frame data for the signal frame (used for distance threshold)
            signal_type: Type of signal (pick/place/open/close)
            ref_link_name: For open/close, the articulated link name

        Returns:
            bool: True if an end condition is met
        """
        if not self.frame_buffer or signal_frame_data is None:
            return False

        if signal_type == 'open' and self.use_open_grasp_signal:
            return self._open_grasp_end_condition_met(signal_frame_data)

        latest_frame = self.frame_buffer[-1]
        obj_pose = latest_frame['object_poses'].get(ref_obj_name, None)
        if obj_pose is None:
            return False

        obj_pos = obj_pose['pos']
        if not isinstance(obj_pos, th.Tensor):
            obj_pos = th.tensor(obj_pos, dtype=th.float32)

        eef_pos_world = self._eef_world_from_frame(latest_frame)
        distance = th.linalg.norm(eef_pos_world - obj_pos).item()

        signal_eef_world = self._eef_world_from_frame(signal_frame_data)
        # For pick/place: use full euclidean distance from signal frame EEF
        # For open/close: use relative vertical distance to link
        if signal_type in ['pick', 'place']:
            distance_from_signal = th.linalg.norm(eef_pos_world - signal_eef_world).item()
        else:  # open/close
            if ref_link_name:
                link_key = (ref_obj_name, ref_link_name)
                link_pos_signal = signal_frame_data['link_poses'].get(link_key, {}).get('pos', None)
                link_pos_current = latest_frame['link_poses'].get(link_key, {}).get('pos', None)
                if link_pos_signal is not None and link_pos_current is not None:
                    link_z_distance_at_signal = abs(signal_eef_world[2].item() - link_pos_signal[2].item())
                    current_link_z_distance = abs(eef_pos_world[2].item() - link_pos_current[2].item())
                    distance_from_signal = current_link_z_distance - link_z_distance_at_signal
                else:
                    distance_from_signal = 0.0
            else:
                distance_from_signal = 0.0

        return distance > max_distance or distance_from_signal >= self.z_end_threshold

    def _open_grasp_end_condition_met(self, signal_frame_data: Dict) -> bool:
        """
        Check end condition for open signals using grasp signal mode.

        Waits for the gripper to open after the signal frame, then checks if the
        EEF has moved far enough from the release position.
        """
        signal_frame = signal_frame_data['frame_idx']

        release_eef_pos = None
        for frame_data in sorted(self.frame_buffer, key=lambda x: x['frame_idx']):
            if frame_data['frame_idx'] <= signal_frame:
                continue
            if frame_data['gripper_action'] >= 0.5:
                release_eef_pos = self._eef_world_from_frame(frame_data)
                break

        if release_eef_pos is None:
            return False

        latest_frame = self.frame_buffer[-1]
        current_eef_pos = self._eef_world_from_frame(latest_frame)
        distance = th.linalg.norm(current_eef_pos - release_eef_pos).item()

        return distance > self.open_grasp_max_distance

    def _extract_waypoints_for_signal(self, signal: Dict, signal_frame: int, available_end_frame: Optional[int] = None):
        """
        Extract waypoints for a single signal

        Args:
            signal: Signal dictionary
            signal_frame: Frame index of the signal event
            available_end_frame: Last frame index currently available for extraction
        """
        # Get appropriate distance thresholds based on signal type
        if signal['type'] == 'pick':
            min_start_distance = self.pick_min_start_distance
            max_distance = self.pick_max_distance
        elif signal['type'] == 'place':
            min_start_distance = self.place_min_start_distance
            max_distance = self.place_max_distance
        elif signal['type'] == 'open':
            min_start_distance = self.open_min_start_distance
            max_distance = self.open_max_distance
        elif signal['type'] == 'close':
            min_start_distance = self.close_min_start_distance
            max_distance = self.close_max_distance
        else:
            print(f"Warning: Unknown signal type '{signal['type']}', skipping")
            return

        # Get reference object based on signal type
        ref_link_name = None  # For open/close, we'll use a specific link
        if signal['type'] == 'pick':
            # For pick, reference is the object being grasped
            ref_obj_name = signal['object_name']
        elif signal['type'] == 'place':
            # For place, reference is the object being placed onto (target)
            ref_obj_name = signal['target_object_name']
        elif signal['type'] in ['open', 'close']:
            # For open/close, reference is the specific articulated link
            ref_obj_name = signal['object_name']
            ref_link_name = signal.get('contact_link', '')
            if not ref_link_name:
                print(f"Warning: No contact_link specified for {signal['type']} signal, skipping")
                return
        else:
            print(f"Warning: Unknown signal type '{signal['type']}', skipping")
            return

        # Find reference object in scene
        ref_obj = self.env.scene.object_registry("name", ref_obj_name, None)
        if ref_obj is None:
            print(f"Warning: Reference object '{ref_obj_name}' not found in scene, skipping")
            return

        # Get initial pose (at signal frame) - either object or link
        # Find the frame in buffer corresponding to signal_frame
        signal_frame_data = None
        for frame_data in self.frame_buffer:
            if frame_data['frame_idx'] == signal_frame:
                signal_frame_data = frame_data
                break

        if signal_frame_data is None:
            print(f"Warning: Signal frame {signal_frame} not in buffer, skipping")
            return

        # Get reference pose at signal frame
        # For all action types (pick/place/open/close), use the object base pose
        # This ensures a stable reference frame that doesn't move (unlike articulated links)
        if ref_obj_name not in signal_frame_data['object_poses']:
            print(f"Warning: Object '{ref_obj_name}' pose not found at signal frame {signal_frame}, skipping")
            return
        obj_pos_signal = signal_frame_data['object_poses'][ref_obj_name]['pos']
        obj_quat_signal = signal_frame_data['object_poses'][ref_obj_name]['quat']

        # Debug: print reference frame being used
        print(f"  Using reference frame: {ref_obj_name} base at signal frame {signal_frame}")
        print(f"    Object pose: pos={obj_pos_signal.cpu().numpy()}, quat={obj_quat_signal.cpu().numpy()}")

        eef_pos_signal_world = self._eef_world_from_frame(signal_frame_data)

        # For open/close: compute the distance from EEF to link at signal frame (reference distance)
        link_distance_at_signal = None
        link_z_distance_at_signal = None
        if signal['type'] in ['open', 'close']:
            link_key = (ref_obj_name, ref_link_name)
            link_pos_signal = signal_frame_data['link_poses'].get(link_key, {}).get('pos', None)
            if link_pos_signal is not None:
                link_distance_at_signal = th.linalg.norm(eef_pos_signal_world - link_pos_signal).item()
                # Also compute vertical distance for the secondary threshold
                link_z_distance_at_signal = abs(eef_pos_signal_world[2].item() - link_pos_signal[2].item())

        # DISTANCE-BASED EXTRACTION:
        # Pick/Place: Distance relative to signal frame EEF position
        # Open/Close: Distance relative to the articulated link distance at signal frame
        # 1. Go backwards from signal_frame to find start frame
        # 2. Extract forward until stopping condition met

        # Step 1: Find start frame by going backwards
        start_frame = None
        candidate_frames = [f for f in self.frame_buffer if f['frame_idx'] <= signal_frame]
        earliest_frame_available = min([f['frame_idx'] for f in candidate_frames]) if candidate_frames else signal_frame

        # Sort frames in reverse order from signal_frame
        sorted_frames = sorted(candidate_frames, key=lambda x: x['frame_idx'], reverse=True)

        for frame_data in sorted_frames:
            # Get EEF position in world frame at this frame
            eef_pos_world = self._eef_world_from_frame(frame_data)

            # Compute distance based on signal type
            if signal['type'] in ['open', 'close']:
                # For open/close: compute distance from EEF to link at this frame,
                # then compare against signal frame distance (relative comparison)
                link_key = (ref_obj_name, ref_link_name)
                link_pos = frame_data['link_poses'].get(link_key, {}).get('pos', None)
                if link_pos is None or link_distance_at_signal is None:
                    continue
                current_link_distance = th.linalg.norm(eef_pos_world - link_pos).item()
                # Distance threshold is relative to signal frame distance
                distance = current_link_distance - link_distance_at_signal
            else:
                # For pick/place: distance from EEF at this frame to EEF at signal frame
                distance = th.linalg.norm(eef_pos_world - eef_pos_signal_world).item()

            if distance >= min_start_distance:
                start_frame = frame_data['frame_idx']
                if signal['type'] in ['open', 'close']:
                    ref_desc = f"articulated link (relative to signal frame: {link_distance_at_signal:.3f}m)"
                else:
                    ref_desc = "signal frame EEF"
                print(f"  Found start frame {start_frame} at distance {distance:.3f}m from {ref_desc}")
                break

        if start_frame is None:
            print(f"  WARNING: Could not find start frame with distance >= {min_start_distance}m")
            print(f"  Searched up to signal frame. Using earliest available frame {earliest_frame_available}.")
            start_frame = earliest_frame_available

        # Step 2: Extract waypoints from start_frame forward
        if available_end_frame is None:
            if self.frame_buffer:
                available_end_frame = self.frame_buffer[-1]['frame_idx']
            else:
                available_end_frame = signal_frame

        end_frame = available_end_frame
        if self.max_frames is not None:
            end_frame = min(end_frame, self.max_frames - 1)

        waypoints_list = []
        eef_world_positions = []  # For debug drawing
        extraction_stopped_by_threshold = False
        stop_reason = ""
        open_grasp_release_eef_pos = None

        if signal['type'] == 'open' and self.use_open_grasp_signal:
            signal_gripper = signal_frame_data['gripper_action']
            if signal_gripper >= 0.5:
                print(f"  WARNING: use_open_grasp_signal is enabled but gripper is not closed "
                      f"at open signal frame {signal_frame} (gripper_action={signal_gripper:.3f})")

        for frame_data in sorted(self.frame_buffer, key=lambda x: x['frame_idx']):
            if not (start_frame <= frame_data['frame_idx'] <= end_frame):
                continue

            eef_pos_world = self._eef_world_from_frame(frame_data)

            # Compute distance based on signal type
            if signal['type'] in ['open', 'close']:
                # For open/close: compute distance from EEF to link at this frame,
                # then compare against signal frame distance (relative comparison)
                link_key = (ref_obj_name, ref_link_name)
                link_pos = frame_data['link_poses'].get(link_key, {}).get('pos', None)
                if link_pos is None or link_distance_at_signal is None:
                    # Skip this frame if link pose not available
                    continue
                current_link_distance = th.linalg.norm(eef_pos_world - link_pos).item()
                # Distance threshold is relative to signal frame distance
                distance = current_link_distance - link_distance_at_signal
            else:
                # For pick/place: distance from EEF at this frame to EEF at signal frame
                distance = th.linalg.norm(eef_pos_world - eef_pos_signal_world).item()

            # Distance from signal frame EEF pose
            # For pick/place: use full euclidean distance from signal frame EEF
            # For open/close: use relative vertical distance to link
            if signal['type'] in ['pick', 'place']:
                distance_from_signal = th.linalg.norm(eef_pos_world - eef_pos_signal_world).item()
            else:  # open/close
                # Compute current vertical distance to link, subtract signal frame vertical distance
                if link_pos is not None and link_z_distance_at_signal is not None:
                    current_link_z_distance = abs(eef_pos_world[2].item() - link_pos[2].item())
                    distance_from_signal = current_link_z_distance - link_z_distance_at_signal
                else:
                    distance_from_signal = 0.0

            # Transform EEF pose from robot frame to object frame (use signal frame object pose)
            eef_pos_obj, eef_quat_obj = self._transform_eef_to_object_frame(
                frame_data['eef_pos_robot'],
                frame_data['eef_quat_robot'],
                obj_pos_signal,
                obj_quat_signal,
                frame_data['robot_pos_world'],
                frame_data['robot_quat_world']
            )

            waypoints_list.append({
                'frame_idx': frame_data['frame_idx'],
                'eef_pos_obj': eef_pos_obj.cpu().numpy().tolist(),
                'eef_quat_obj': eef_quat_obj.cpu().numpy().tolist(),
                'gripper_action': frame_data['gripper_action'],
            })

            # Store EEF position in world frame for debug visualization
            eef_world_positions.append(eef_pos_world.cpu().numpy().tolist())

            # Debug: for first and last waypoint, print both world and object-frame positions
            if len(waypoints_list) == 1 or len(waypoints_list) == 10:
                print(f"    Waypoint {len(waypoints_list)}: eef_world={eef_pos_world.cpu().numpy()}, eef_obj={eef_pos_obj.cpu().numpy()}")

            # Evaluate stopping conditions after including this frame
            # Only check stopping conditions AFTER the signal frame (we want to extract at least up to the signal)
            if frame_data['frame_idx'] > signal_frame:
                if signal['type'] == 'open' and self.use_open_grasp_signal:
                    if frame_data['gripper_action'] >= 0.5 and open_grasp_release_eef_pos is None:
                        open_grasp_release_eef_pos = eef_pos_world.clone()
                        print(f"    Gripper opened at frame {frame_data['frame_idx']}, "
                              f"recording release EEF position: {open_grasp_release_eef_pos.cpu().numpy()}")

                    if open_grasp_release_eef_pos is not None:
                        release_distance = th.linalg.norm(eef_pos_world - open_grasp_release_eef_pos).item()
                        if release_distance > self.open_grasp_max_distance:
                            extraction_stopped_by_threshold = True
                            stop_reason = (f"open grasp: L2 from release position "
                                           f"{release_distance:.3f}m > {self.open_grasp_max_distance}m")
                            print(f"  Stopped extraction at frame {frame_data['frame_idx']}: {stop_reason}")
                            break
                else:
                    stop_checks = []
                    if distance > max_distance:
                        if signal['type'] in ['open', 'close']:
                            ref_desc = f"link (relative to signal: {link_distance_at_signal:.3f}m)"
                        else:
                            ref_desc = "signal EEF"
                        stop_checks.append(f"distance from {ref_desc} {distance:.3f}m > max {max_distance}m")
                    if distance_from_signal >= self.z_end_threshold:
                        if signal['type'] in ['pick', 'place']:
                            stop_checks.append(f"euclidean distance from signal EEF {distance_from_signal:.3f}m >= threshold {self.z_end_threshold}m")
                        else:  # open/close
                            stop_checks.append(f"relative vertical distance to link {distance_from_signal:.3f}m >= threshold {self.z_end_threshold}m")

                    if stop_checks:
                        extraction_stopped_by_threshold = True
                        stop_reason = "; ".join(stop_checks)
                        print(f"  Stopped extraction at frame {frame_data['frame_idx']}: {stop_reason}")
                        break

        # Debug visualization: draw waypoints for this segment with gradient
        robot_T_world = T.pose2mat((frame_data['robot_pos_world'], frame_data['robot_quat_world']))
        if len(eef_world_positions) > 0:
            eef_world_positions_vis = th.tensor(eef_world_positions)
            draw_trajectory_gradient(
                (th.cat([eef_world_positions_vis, th.ones(eef_world_positions_vis.shape[0], 1)], dim=1) @ robot_T_world.T)[:, :3],
                color_scheme="green_to_red",
                point_size=10,
                clear_previous=True
            )

        actual_before = signal_frame - start_frame
        actual_after = max(0, (waypoints_list[-1]['frame_idx'] if waypoints_list else signal_frame) - signal_frame)

        # Store extracted waypoints
        subtask_data = {
            'type': signal['type'],
            'signal_frame_idx': signal_frame,
            'reference_object': {
                'name': ref_obj_name,
                'category': signal.get('object_category' if signal['type'] == 'pick' else 'target_object_category', ''),
                # 'model': signal.get('object_model' if signal['type'] == 'pick' else 'target_object_model', ''),
                'pos': obj_pos_signal.cpu().numpy().tolist(),
                'quat': obj_quat_signal.cpu().numpy().tolist(),
            },
            'extraction_params': {
                'min_start_distance': min_start_distance,
                'max_distance': max_distance,
                'z_end_threshold': self.z_end_threshold,
            },
            'actual_frames_before': actual_before,
            'actual_frames_after': actual_after,
            'waypoints': waypoints_list,
        }

        # For place actions, store the object being placed
        if signal['type'] == 'place':
            subtask_data['placed_object'] = {
                'name': signal.get('released_object_name', ''),
                'category': signal.get('released_object_category', ''),
                # 'model': signal.get('released_object_model', ''),
            }

        self.waypoints.append(subtask_data)

        # Format the distance threshold description based on signal type
        if signal['type'] in ['pick', 'place']:
            threshold_desc = f"euclidean_dist<{self.z_end_threshold}m"
        else:  # open/close
            threshold_desc = f"rel_z_to_link<{self.z_end_threshold}m"

        print(f"  Extracted {len(waypoints_list)} waypoints for {signal['type']} signal at frame {signal_frame} "
              f"(distance-based: start>={min_start_distance}m, end<=({max_distance}m, {threshold_desc}){' [' + stop_reason + ']' if extraction_stopped_by_threshold else ''}, "
              f"actual frames: -{actual_before}/+{actual_after})")

    def _transform_eef_to_object_frame(
        self,
        eef_pos_robot: th.Tensor,
        eef_quat_robot: th.Tensor,
        obj_pos: th.Tensor,
        obj_quat: th.Tensor,
        robot_pos_world: th.Tensor,
        robot_quat_world: th.Tensor
    ):
        """
        Transform EEF pose from robot frame to object frame

        Args:
            eef_pos_robot: EEF position in robot frame [3]
            eef_quat_robot: EEF quaternion in robot frame [4]
            obj_pos: Object position in world frame [3]
            obj_quat: Object quaternion in world frame [4]
            robot_pos_world: Robot base position in world frame [3]
            robot_quat_world: Robot base quaternion in world frame [4]

        Returns:
            tuple: (eef_pos_obj, eef_quat_obj) - EEF pose in object frame
        """
        # Convert to tensors if needed
        if not isinstance(robot_pos_world, th.Tensor):
            robot_pos_world = th.tensor(robot_pos_world, dtype=th.float32)
        if not isinstance(robot_quat_world, th.Tensor):
            robot_quat_world = th.tensor(robot_quat_world, dtype=th.float32)
        if not isinstance(obj_pos, th.Tensor):
            obj_pos = th.tensor(obj_pos, dtype=th.float32)
        if not isinstance(obj_quat, th.Tensor):
            obj_quat = th.tensor(obj_quat, dtype=th.float32)

        # Build transformation matrices
        # Robot frame → World frame
        robot_T_world = T.pose2mat((robot_pos_world, robot_quat_world))

        # EEF in robot frame → EEF in world frame
        eef_T_robot = T.pose2mat((eef_pos_robot, eef_quat_robot))
        eef_T_world = robot_T_world @ eef_T_robot

        # World frame → Object frame (inverse of object_T_world)
        obj_T_world = T.pose2mat((obj_pos, obj_quat))
        world_T_obj = T.pose_inv(obj_T_world)

        # EEF in world frame → EEF in object frame
        eef_T_obj = world_T_obj @ eef_T_world

        # Extract position and quaternion
        eef_pos_obj, eef_quat_obj = T.mat2pose(eef_T_obj)

        return eef_pos_obj, eef_quat_obj

    def _merge_adjacent_subtasks(self):
        """
        Merge adjacent subtask sequences that share the same reference object frame.

        Uses the frame buffer to fill in the gap frames between adjacent subtasks,
        transforming them into the first subtask's reference object frame. This produces
        a single continuous trajectory that doesn't require freespace interpolation.

        For each merge, the gap frames (between the end of subtask N and the start of
        subtask N+1) and the second subtask's frames are all (re-)extracted from the
        frame buffer using the first subtask's reference object pose, ensuring a
        consistent coordinate frame across the merged trajectory.
        """
        if len(self.waypoints) <= 1:
            return

        frame_lookup = {f['frame_idx']: f for f in self.frame_buffer}

        merged = [self.waypoints[0]]

        for i in range(1, len(self.waypoints)):
            prev = merged[-1]
            curr = self.waypoints[i]

            prev_ref = prev['reference_object']['name']
            curr_ref = curr['reference_object']['name']

            if prev_ref != curr_ref:
                merged.append(curr)
                continue

            ref_obj_name = prev_ref

            # Use the first (or already-merged-first) subtask's signal frame as the
            # canonical reference pose, ensuring a single consistent object frame.
            ref_signal_frame = prev['signal_frame_idx']
            ref_frame_data = frame_lookup.get(ref_signal_frame)
            if ref_frame_data is None or ref_obj_name not in ref_frame_data['object_poses']:
                print(f"  WARNING: Cannot merge subtasks {i-1} and {i} — reference frame "
                      f"data for '{ref_obj_name}' at frame {ref_signal_frame} not available")
                merged.append(curr)
                continue

            obj_pos_ref = ref_frame_data['object_poses'][ref_obj_name]['pos']
            obj_quat_ref = ref_frame_data['object_poses'][ref_obj_name]['quat']

            # Frame ranges
            prev_end_frame = prev['waypoints'][-1]['frame_idx']
            curr_start_frame = curr['waypoints'][0]['frame_idx']
            curr_end_frame = curr['waypoints'][-1]['frame_idx']

            n_gap_frames = max(0, curr_start_frame - prev_end_frame - 1)
            print(f"  Merging subtask {i-1} ({prev['type']}) and subtask {i} ({curr['type']}) "
                  f"— same reference object '{ref_obj_name}', filling {n_gap_frames} gap frames "
                  f"(frames {prev_end_frame + 1}..{curr_start_frame - 1})")

            # Extract gap frames + second subtask's frames from the frame buffer,
            # all transformed into the first subtask's reference object frame.
            new_waypoints = []
            for frame_idx in range(prev_end_frame + 1, curr_end_frame + 1):
                frame_data = frame_lookup.get(frame_idx)
                if frame_data is None:
                    continue

                eef_pos_obj, eef_quat_obj = self._transform_eef_to_object_frame(
                    frame_data['eef_pos_robot'],
                    frame_data['eef_quat_robot'],
                    obj_pos_ref,
                    obj_quat_ref,
                    frame_data['robot_pos_world'],
                    frame_data['robot_quat_world']
                )

                new_waypoints.append({
                    'frame_idx': frame_data['frame_idx'],
                    'eef_pos_obj': eef_pos_obj.cpu().numpy().tolist(),
                    'eef_quat_obj': eef_quat_obj.cpu().numpy().tolist(),
                    'gripper_action': frame_data['gripper_action'],
                })

            # Combine: previous waypoints (already in correct frame) + gap + re-extracted current
            combined_waypoints = prev['waypoints'] + new_waypoints
            combined_type = f"{prev['type']}+{curr['type']}"

            last_wp_frame = combined_waypoints[-1]['frame_idx'] if combined_waypoints else prev['signal_frame_idx']
            merged_subtask = {
                'type': combined_type,
                'signal_frame_idx': prev['signal_frame_idx'],
                'reference_object': prev['reference_object'],
                'extraction_params': prev['extraction_params'],
                'actual_frames_before': prev['actual_frames_before'],
                'actual_frames_after': max(0, last_wp_frame - prev['signal_frame_idx']),
                'waypoints': combined_waypoints,
            }

            if 'placed_object' in prev:
                merged_subtask['placed_object'] = prev['placed_object']
            if 'placed_object' in curr:
                merged_subtask['placed_object'] = curr['placed_object']

            merged[-1] = merged_subtask

        n_before = len(self.waypoints)
        self.waypoints = merged
        n_after = len(self.waypoints)
        if n_after < n_before:
            print(f"  Merged {n_before} subtasks into {n_after} (merge_sequences=True)")

    def finalize(self):
        """
        Extract any remaining signals that haven't been processed yet.
        Call this at the end of an episode to handle signals near the end.
        """
        while self.signal_idx < len(self.signals):
            signal = self.signals[self.signal_idx]

            # Determine signal frame
            if signal['type'] == 'pick':
                signal_frame = signal['frame_idx']
            elif signal['type'] == 'place':
                signal_frame = signal['contact_frame_idx']
            elif signal['type'] in ['open', 'close']:
                signal_frame = signal['release_frame_idx']
            else:
                self.signal_idx += 1
                continue

            # Extract waypoints for this signal (will be clamped to available frames)
            last_frame_idx = self.frame_buffer[-1]['frame_idx'] if self.frame_buffer else None
            self._extract_waypoints_for_signal(signal, signal_frame, available_end_frame=last_frame_idx)
            self.signal_idx += 1

        if self.merge_sequences:
            self._merge_adjacent_subtasks()

    def get_waypoints(self) -> List[Dict]:
        """Get all extracted waypoints"""
        return self.waypoints

    def episode_start_callback(self, episode_id: int, env: Any):
        """Callback for episode start"""
        clear_trajectory_visualization()
        self.reset()
