# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
YAM JoyLo Environment Server

An OmniGibson environment server that communicates with the JoyLo system
to read joint angles and apply them to control two YAM robot arms for data collection.

This is a simplified version of the logic found in deps/BEHAVIOR-1K/joylo/gello/robots/sim_robot/og_sim.py,
adapted for controlling two separate YAM arms instead of a single bimanual R1Pro robot.

Usage:
    python yam_joylo_env_server.py --scene-json <scene_json_name> --task <task_name> --recording-path <path_to_save_data>
"""

import os
import time
import torch as th
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import signal

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.envs import HDF5CollectionWrapper
from omnigibson.robots import Yam
from omnigibson.utils.config_utils import parse_config

from digital_cousins import CFG_DIR as CDC_CFG_DIR, ASSET_DIR as CDC_ASSET_DIR
from digital_cousins.tasks.pick_place_task import PickPlaceTask
from digital_cousins.utils.scene_utils import load_json_with_absolute_usd_paths
from digital_cousins.utils.og_utils import (
    apply_teleop_omnigibson_macros,
    setup_task_status_ui,
    update_task_status,
    update_demo_success_counter,
    setup_robot_visualizers,
    update_in_hand_status,
    update_grasp_status,
)


# Configuration constants
DEFAULT_RESET_DELTA_SPEED = 10.0  # deg / sec
N_COOLDOWN_SECS = 1.5
RESOLUTION = [1080, 1080]  # [H, W]


class SignalChangeDetector:
    """
    Detects changes in a signal with debounce logic.
    """
    def __init__(self, debounce_time=0.5):
        self.debounce_time = debounce_time
        self.last_change_time = 0.0
        self.last_value = None
    
    def process_sample(self, value):
        """
        Process a sample and return True if a state change was detected.
        
        Args:
            value: Current signal value
            
        Returns:
            bool: True if state changed and debounce time has passed
        """
        current_time = time.time()
        changed = False
        
        if self.last_value is None:
            self.last_value = value
        elif value != self.last_value:
            if current_time - self.last_change_time > self.debounce_time:
                changed = True
                self.last_change_time = current_time
            self.last_value = value
            
        return changed
    
    def reset(self):
        """Reset the detector state."""
        self.last_change_time = 0.0
        self.last_value = None


class YamJoyLoEnvServer:
    """
    An environment server that controls two YAM robot arms via JoyLo for teleoperation data collection.
    
    This server:
    - Loads an OmniGibson environment with two YAM arms
    - Communicates with JoyLo via ZMQ to receive joint commands
    - Collects demonstration data using HDF5CollectionWrapper
    - Supports PickPlaceTask for task-based data collection
    """
    
    def __init__(
        self,
        scene_json_name: str,
        task_name: str,
        host: str = "127.0.0.1",
        port: int = 5556,
        use_floor_plane: bool = True,
        max_steps: int = 500,
        external_sensors_cfg: str = "sim_yam_teleop_joylo",
        action_freq: int = 15,
        recording_path: Optional[str] = None,
        only_successes: bool = True,
        flush_every_n_traj: int = 1,
        viewport_camera_path: Optional[str] = None,
        overwrite: bool = False,
        instance_id: Optional[int] = None,
    ):
        """
        Initialize the YAM JoyLo environment server.
        
        Args:
            scene_json_name: Name of the scene JSON to load (located in CDC_ASSET_DIR/scenes, without json extension)
            task_name: Name of the task configuration (located in CDC_CFG_DIR/task, without yaml extension)
            host: ZMQ server host address
            port: ZMQ server port for communication (handles both arms as 12 total motors)
            use_floor_plane: Whether to use a floor plane in the scene
            max_steps: Maximum steps per episode before termination
            external_sensors_cfg: Name of external sensors config (located in CDC_CFG_DIR/external_sensors, without yaml extension)
            action_freq: Action/rendering frequency in Hz
            recording_path: Path to save HDF5 recording data
            only_successes: Whether to only save successful demonstrations
            flush_every_n_traj: How often to flush data to disk
            viewport_camera_path: Path to viewport camera for HDF5 recording (None uses default viewer camera)
            overwrite: Whether to overwrite existing HDF5 file
            instance_id: Optional instance ID for UI display
        """
        self.task_name = task_name
        self._port = port
        self._instance_id = instance_id
        
        # Apply OmniGibson macros for teleop
        apply_teleop_omnigibson_macros()
        
        # Build configuration programmatically
        cfg = self._build_config(
            scene_json_name=scene_json_name,
            task_name=task_name,
            use_floor_plane=use_floor_plane,
            max_steps=max_steps,
            external_sensors_cfg=external_sensors_cfg,
            action_freq=action_freq,
        )
        
        # Create environment
        self.env = og.Environment(configs=cfg)
        
        # Validate robots
        assert len(self.env.robots) == 2, f"Expected 2 robots, got {len(self.env.robots)}"
        for robot in self.env.robots:
            assert isinstance(robot, Yam), f"Expected Yam robot, got {type(robot)}"
        
        # Get robot references - assume convention of left/right based on naming or order
        self.robots = self.env.robots
        self.left_robot = self.robots[0]
        self.right_robot = self.robots[1]
        
        # Validate task is PickPlaceTask
        if self.task_name is not None:
            assert isinstance(self.env.task, PickPlaceTask), \
                f"Expected PickPlaceTask, got {type(self.env.task)}"
        
        # Setup cameras and visualizations
        self._setup_teleop_support()
        
        # Set variables for control
        self._reset_max_arm_delta = DEFAULT_RESET_DELTA_SPEED * (np.pi / 180) * og.sim.get_sim_step_dt()
        self._resume_cooldown_time = None
        self._in_cooldown = False
        self._joint_state = {
            "left": None,
            "right": None,
        }
        self._joint_cmd = {
            "left_arm": None,
            "right_arm": None,
            "left_gripper": th.ones(1),
            "right_gripper": th.ones(1),
            "button_x": th.zeros(1),
            "button_y": th.zeros(1),
            "button_a": th.zeros(1),
            "button_b": th.zeros(1),
            "button_home": th.zeros(1),
        }
        self._waiting_to_resume = True
        self._grasp_action = {"left": 1, "right": 1}
        
        # Recording configuration
        # We always record (is_recording=True), but use filter_current_frame to control
        # which frames are included in the final dataset. When filter_current_frame=True,
        # frames are marked with filter_mask=1 (to be filtered out).
        self._recording_path = recording_path
        self._is_filtering_frames = True  # Start with frame filtering enabled (frames excluded)
        if self._recording_path is not None:
            # Use provided viewport camera path or fall back to default viewer camera
            cam_path = viewport_camera_path if viewport_camera_path is not None else og.sim.viewer_camera.active_camera_path
            self.env = HDF5CollectionWrapper(
                env=self.env,
                output_path=self._recording_path,
                viewport_camera_path=cam_path,
                only_successes=only_successes,
                flush_every_n_traj=flush_every_n_traj,
                overwrite=overwrite,
                enable_dump_filters=False,
            )
            # Always record, but start with frame filtering enabled
            self.env.is_recording = True
            self.env.filter_current_frame = True
        
        # Status tracking
        self._prev_grasp_status = {"left": False, "right": False}
        self._prev_in_hand_status = {"left": False, "right": False}
        self._frame_counter = 0
        self._button_toggled_state = {
            "x": False,
            "y": False,
            "a": False,
            "b": False,
            "home": False,
        }
        self._gripper_action_signal_detectors = {
            "left": SignalChangeDetector(debounce_time=0.5),
            "right": SignalChangeDetector(debounce_time=0.5),
        }
        
        self.obs = {}
        
        # Cached observations for thread-safe access from ZMQ server
        # These are updated in the main thread via _update_observations()
        self._cached_obs = {}
        
        # Cache joint limits for each robot
        self._arm_joint_limits = {}
        for arm, robot in [("left", self.left_robot), ("right", self.right_robot)]:
            qpos_min, qpos_max = robot.joint_lower_limits, robot.joint_upper_limits
            arm_name = robot.default_arm
            self._arm_joint_limits[arm] = {
                "lower": qpos_min[robot.arm_control_idx[arm_name]],
                "upper": qpos_max[robot.arm_control_idx[arm_name]],
            }
        
        # Reset environment to initialize
        self.reset(increment_instance=False)
        
        # Take a single step
        action = self.get_action()
        self.env.step(action)
        
        # Set up keyboard handlers
        self._setup_keyboard_handlers()
        
        # Initialize demo success counter
        self.demo_success_count = 0
        self._is_success = False
        
        # Create ZMQ server for communication (handles both arms as 12 total motors)
        from gello.robots.sim_robot.zmq_server import ZMQRobotServer, ZMQServerThread
        
        self._zmq_server = ZMQRobotServer(
            robot=self,
            host=host,
            port=port,
            verbose=False,
        )
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)
        
        # Register shutdown handler
        signal.signal(signal.SIGINT, self._shutdown_handler)
    
    def _build_config(
        self,
        scene_json_name: str,
        task_name: str,
        use_floor_plane: bool,
        max_steps: int,
        external_sensors_cfg: str,
        action_freq: int,
    ) -> Dict:
        """
        Build the OmniGibson environment configuration programmatically.
        
        This method loads the scene JSON, modifies robot controller configs to use
        JointController for arms and smooth gripper control, then passes the modified
        dict to the environment.
        
        Args:
            scene_json_name: Name of the scene JSON to load
            task_name: Name of the task configuration
            use_floor_plane: Whether to use a floor plane
            max_steps: Maximum steps per episode
            external_sensors_cfg: Name of external sensors config
            action_freq: Action/rendering frequency in Hz
            
        Returns:
            dict: Complete OmniGibson environment configuration
        """
        import json
        
        # Build scene file path
        og_scene_json_path = f"{CDC_ASSET_DIR}/scenes/{scene_json_name}/{scene_json_name}_scene_state_latest.json"
        if not os.path.exists(og_scene_json_path):
            # Try alternate path format
            og_scene_json_path = f"{CDC_ASSET_DIR}/scenes/{scene_json_name}.json"
            if not os.path.exists(og_scene_json_path):
                raise FileNotFoundError(f"Scene JSON not found at expected paths for: {scene_json_name}")
        
        # Load scene JSON as dict
        scene_json_dict = load_json_with_absolute_usd_paths(og_scene_json_path)
        
        # Modify robot controller configs in the scene dict
        self._modify_robot_controllers_in_scene(scene_json_dict)
        
        # Build scene config - pass the modified dict instead of the path
        scene_cfg = {
            "type": "Scene",
            "scene_file": scene_json_dict,  # Pass dict directly
            "use_floor_plane": use_floor_plane,
            "floor_plane_visible": use_floor_plane,
            "use_skybox": True,
            "include_robots": True,
        }
        
        # Load task config from YAML
        og_task_cfg_path = f"{CDC_CFG_DIR}/task/{task_name}.yaml"
        if not os.path.exists(og_task_cfg_path):
            raise FileNotFoundError(f"Task config not found at: {og_task_cfg_path}")
        task_cfg = parse_config(og_task_cfg_path)["og_task_config"]
        task_cfg["termination_config"]["max_steps"] = max_steps
        
        # Load external sensors config
        external_sensors_cfg_path = f"{CDC_CFG_DIR}/external_sensors/{external_sensors_cfg}.yaml"
        if not os.path.exists(external_sensors_cfg_path):
            raise FileNotFoundError(f"External sensors config not found at: {external_sensors_cfg_path}")
        external_sensors = parse_config(external_sensors_cfg_path).get("external_sensors", [])
        
        # Build environment config
        env_cfg = {
            "external_sensors": external_sensors,
            "action_frequency": action_freq,
            "rendering_frequency": action_freq,
            "physics_frequency": 120,
        }
        
        # Combine all configs
        og_cfg = {
            "env": env_cfg,
            "scene": scene_cfg,
            "task": task_cfg,
        }
        
        return og_cfg
    
    def _modify_robot_controllers_in_scene(self, scene_dict: Dict) -> None:
        """
        Modify robot controller configs in the scene dict to use JointController for arms
        and smooth gripper control. Also clears the controller state entries in the state
        section to prevent stale state from causing issues.
        
        Args:
            scene_dict: The loaded scene JSON dict to modify in-place
        """
        # Yam-specific gripper configuration
        open_qpos = [-0.045, 0.045]
        closed_qpos = [0.0, 0.0]
        inverted = [False, True]
        
        # Get init_info from scene dict (for controller config)
        init_info = scene_dict.get("objects_info", {}).get("init_info", {})
        
        # Get object registry from state (for controller state)
        object_registry = scene_dict.get("state", {}).get("registry", {}).get("object_registry", {})
        
        # Default cleared controller state
        cleared_controller_state = {
            "goal_is_valid": False,
            "goal": None,
        }
        
        for obj_name, obj_info in init_info.items():
            # Check if this is a Yam robot
            if obj_info.get("class_name") == "Yam":
                args = obj_info.get("args", {})
                
                # Build new controller config for this robot
                # Yam has default_arm = "0", so controller keys are "arm_0" and "gripper_0"
                new_controller_config = {
                    "arm_0": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "command_output_limits": None,
                        "use_impedances": False,
                        "use_delta_commands": False,
                    },
                    "gripper_0": {
                        "name": "MultiFingerGripperController",
                        "command_input_limits": [0.0, 1.0],
                        "mode": "smooth",
                        "open_qpos": open_qpos,
                        "closed_qpos": closed_qpos,
                        "inverted": inverted,
                    },
                }
                
                # Update the controller config in args
                args["controller_config"] = new_controller_config
                
                # Clear the controller state in the state section
                if obj_name in object_registry:
                    robot_state = object_registry[obj_name]
                    if "controllers" in robot_state:
                        # Replace all controller states with cleared state
                        for controller_name in robot_state["controllers"]:
                            robot_state["controllers"][controller_name] = cleared_controller_state.copy()
                        print(f"Cleared controller states for robot: {obj_name}")
                
                print(f"Updated controller config for robot: {obj_name}")
    
    def _setup_teleop_support(self):
        """Set up cameras, visualizations, and UI elements."""
        # Setup visualizers for each robot
        self.vis_elements = {}
        with og.sim.stopped():
            for arm, robot in [("left", self.left_robot), ("right", self.right_robot)]:
                # Yam-specific offset for visualization
                offset = th.tensor([0.0, -0.041, 0.13])
                self.vis_elements[arm] = setup_robot_visualizers(
                    robot=robot,
                    scene=self.env.scene,
                    offset=offset,
                )
        
        # Setup task-related UI if task is specified
        if self.task_name is not None:
            (
                self.overlay_window,
                self.text_labels,
                self.instance_id_label,
                self.demo_count_label,
                self._prev_status,
            ) = setup_task_status_ui(
                task_name=self.task_name,
                env=self.env,
                instance_id=self._instance_id,
            )
        else:
            self.overlay_window = None
            self.text_labels = None
            self.instance_id_label = None
            self.demo_count_label = None
            self._prev_status = {}
    
    def _setup_keyboard_handlers(self):
        """Set up keyboard event handlers."""
        def keyboard_event_handler(event, *args, **kwargs):
            if (
                event.type == lazy.carb.input.KeyboardEventType.KEY_PRESS
                or event.type == lazy.carb.input.KeyboardEventType.KEY_REPEAT
            ):
                if event.input == lazy.carb.input.KeyboardInput.R:
                    self.reset()
                elif event.input == lazy.carb.input.KeyboardInput.P:
                    self.pause()
                elif event.input == lazy.carb.input.KeyboardInput.X:
                    self.resume_control()
                elif event.input == lazy.carb.input.KeyboardInput.T:
                    self._toggle_recording()
                elif event.input == lazy.carb.input.KeyboardInput.ESCAPE:
                    self.stop()
            
            return True
        
        appwindow = lazy.omni.appwindow.get_default_app_window()
        input_interface = lazy.carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()
        self.sub_keyboard = input_interface.subscribe_to_keyboard_events(keyboard, keyboard_event_handler)
    
    def _toggle_recording(self):
        """
        Toggle frame filtering state.
        
        When filter_current_frame=False, frames are included in the dataset (recording active).
        When filter_current_frame=True, frames are marked to be filtered out (recording paused).
        """
        self._is_filtering_frames = not self._is_filtering_frames
        if hasattr(self.env, 'filter_current_frame'):
            self.env.filter_current_frame = self._is_filtering_frames
        # Display status: when NOT filtering, we're actively recording
        status = "PAUSED" if self._is_filtering_frames else "RECORDING"
        print(f"\n*** Recording {status} ***\n")
    
    def num_dofs(self) -> int:
        """Return the total number of degrees of freedom (12 arm joints for 2 YAM arms)."""
        # 6 DOF per arm * 2 arms = 12 total arm joints
        return 12
    
    def get_joint_state(self) -> th.Tensor:
        """
        Get the current joint state for both arms.
        
        Returns:
            torch.Tensor: Concatenated joint positions [6 left arm, 6 right arm]
        """
        left_state = self._joint_state["left"]
        right_state = self._joint_state["right"]
        
        if left_state is None or right_state is None:
            # Return zeros if not yet initialized
            return th.zeros(12)
        
        return th.cat([left_state, right_state])
    
    def command_joint_state(self, joint_state: th.Tensor) -> None:
        """
        Command both robot arms to joint states.
        
        Args:
            joint_state: Target joint state tensor
                Format: [6 left arm, 6 right arm, 1 left gripper, 1 right gripper, 
                         button_x, button_y, button_home]
        """
        state = joint_state.clone()
        
        # Parse joint commands - expected format for 2 YAM arms:
        # [6 left arm, 6 right arm, 1 left gripper, 1 right gripper, button_x, button_y, button_a, button_b, button_home]
        start_idx = 0
        for component, dim in [
            ("left_arm", 6),
            ("right_arm", 6),
            ("left_gripper", 1),
            ("right_gripper", 1),
            ("button_x", 1),
            ("button_y", 1),
            ("button_a", 1),
            ("button_b", 1),
            ("button_home", 1),
        ]:
            if start_idx >= len(state):
                break
            self._joint_cmd[component] = state[start_idx:start_idx + dim]
            start_idx += dim
    
    def get_observations(self) -> Dict[str, th.Tensor]:
        """
        Get the current observations for both arms.
        
        NOTE: This method is called from the ZMQ server thread. To avoid PhysX
        threading conflicts (overlapping read/write), we return cached observations
        that are updated in the main thread via _update_observations().
        
        Returns:
            dict: Observations including joint positions, velocities, and status flags
        """
        return self._cached_obs
    
    def _update_observations(self):
        """
        Update observations for both arms.
        
        This method is called from the main thread and updates the cached observations
        that are returned by get_observations() (called from ZMQ server thread).
        This design avoids PhysX threading conflicts.
        """
        obs = {}
        
        obs["in_cooldown"] = self._in_cooldown
        obs["waiting_to_resume"] = self._waiting_to_resume
        obs["is_recording"] = not self._is_filtering_frames  # Recording when NOT filtering frames
        
        for arm, robot in [("left", self.left_robot), ("right", self.right_robot)]:
            arm_name = robot.default_arm
            joint_pos = robot.get_joint_positions()
            joint_vel = robot.get_joint_velocities()
            
            arm_control_idx = robot.arm_control_idx[arm_name]
            obs[f"arm_{arm}_joint_positions"] = joint_pos[arm_control_idx]
            obs[f"arm_{arm}_joint_velocities"] = joint_vel[arm_control_idx]
            obs[f"arm_{arm}_gripper_positions"] = joint_pos[robot.gripper_control_idx[arm_name]]
            obs[f"arm_{arm}_ee_pos_quat"] = th.concatenate(robot.eef_links[arm_name].get_position_orientation())
            
            # Update internal joint state cache
            self._joint_state[arm] = joint_pos[robot.arm_control_idx[arm_name]]
        
        # Update cached observations for thread-safe access
        self._cached_obs = obs
    
    def resume_control(self):
        """Resume control after waiting."""
        if self._waiting_to_resume:
            self._waiting_to_resume = False
            self._resume_cooldown_time = time.time() + N_COOLDOWN_SECS
            self._in_cooldown = True
            print("Control resumed, cooling down...")
    
    def serve(self) -> None:
        """Main serving loop."""
        # Initialize cached observations before starting ZMQ server
        # This ensures the client won't receive empty observations on first request
        self._update_observations()
        
        # Start the ZMQ server thread
        self._zmq_server_thread.start()
        
        print("\n" + "=" * 60)
        print("YAM JoyLo Environment Server Started")
        print("=" * 60)
        print(f"ZMQ server running on port {self._port} (12 total motors: 6 left + 6 right)")
        print("\nKeyboard shortcuts:")
        print("  R: Reset environment")
        print("  P: Pause simulation")
        print("  X: Resume control")
        print("  T: Toggle recording")
        print("  ESC: Stop and exit")
        print("=" * 60 + "\n")
        
        while True:
            t1 = time.time()
            self._update_observations()
            
            # Process button inputs
            self._process_button_inputs()
            
            # Only decrement cooldown if we're not waiting to resume
            if not self._waiting_to_resume:
                if self._in_cooldown:
                    print(f"\rIn cooldown!{' ' * 40}", end="", flush=True)
                    self._in_cooldown = time.time() < self._resume_cooldown_time
                else:
                    # Display recording status: when NOT filtering frames, we're actively recording
                    recording_status = "[---]" if self._is_filtering_frames else "[REC]"
                    print(f"\rRunning! {recording_status}{' ' * 30}", end="", flush=True)
            
            # If waiting to resume, simply render without updating action
            if self._waiting_to_resume:
                og.sim.render()
                print(f"\rPress X (keyboard) to resume sim!{' ' * 30}", end="", flush=True)
            else:
                # Generate action and deploy
                action = self.get_action()
                for obj in self.env.scene.objects:
                    obj.wake() # wake up objects to avoid contact check failure
                while (time.time() - t1) < 1 / self.env.env_config["action_frequency"]:
                    og.sim.render()
                _, _, _, _, info = self.env.step(action)
                
                # Update visualizations and status
                self._update_visualization_and_status(info)
    
    def _process_button_inputs(self):
        """
        Process button inputs from JoyLo controllers.
        
        Button mappings:
        - X: Resume control (when waiting to resume)
        - Y: Toggle recording
        - A: Reset environment (saves episode if success)
        - B: Reset environment (discards episode, regardless of success)
        - Home: Reset environment (saves episode if success)
        """
        # X button: Resume control
        button_x_state = self._joint_cmd["button_x"].item() != 0.0
        if button_x_state and not self._button_toggled_state["x"]:
            if self._waiting_to_resume:
                self.resume_control()
        self._button_toggled_state["x"] = button_x_state
        
        # Y button: Toggle recording
        button_y_state = self._joint_cmd["button_y"].item() != 0.0
        if button_y_state and not self._button_toggled_state["y"]:
            self._toggle_recording()
        self._button_toggled_state["y"] = button_y_state
        
        # A button: Reset environment (save episode if success)
        button_a_state = self._joint_cmd["button_a"].item() != 0.0
        if button_a_state and not self._button_toggled_state["a"]:
            if not self._in_cooldown:
                self.reset(save_on_success=True)
        self._button_toggled_state["a"] = button_a_state
        
        # B button: Reset environment (discard episode, regardless of success)
        button_b_state = self._joint_cmd["button_b"].item() != 0.0
        if button_b_state and not self._button_toggled_state["b"]:
            if not self._in_cooldown:
                self.reset(save_on_success=False)
        self._button_toggled_state["b"] = button_b_state
        
        # Home button: Reset environment (save episode if success)
        button_home_state = self._joint_cmd["button_home"].item() != 0.0
        if button_home_state and not self._button_toggled_state["home"]:
            if not self._in_cooldown:
                self.reset(save_on_success=True)
        self._button_toggled_state["home"] = button_home_state
    
    def _update_visualization_and_status(self, info):
        """Update visualization and status based on new information."""
        # Update task goal status if task is active
        if self.task_name is not None:
            status = {"success": self.env.task.success}
            
            # Track demo successes
            if status["success"] and not self._prev_status.get("success", False):
                self._is_success = True
                print(f"\nDemo success #{self.demo_success_count} achieved!")
            
            self._prev_status = update_task_status(
                text_labels=self.text_labels,
                goal_status=status,
                prev_goal_status=self._prev_status,
                env=self.env,
            )
        
        # Update visualization elements for both arms
        for arm, robot in [("left", self.left_robot), ("right", self.right_robot)]:
            vis_elem = self.vis_elements[arm]
            
            # Map to the proper arm name key for the robot's internal naming
            arm_name = robot.default_arm
            
            self._prev_in_hand_status[arm] = update_in_hand_status(
                robot,
                vis_elem["vis_mats"],
                {arm_name: self._prev_in_hand_status[arm]},
            ).get(arm_name, self._prev_in_hand_status[arm])
            
            self._prev_grasp_status[arm] = update_grasp_status(
                robot,
                vis_elem["eef_cylinder_geoms"],
                {arm_name: self._prev_grasp_status[arm]},
            ).get(arm_name, self._prev_grasp_status[arm])
        
        self._frame_counter += 1
    
    def get_action(self) -> th.Tensor:
        """
        Generate action based on current joint commands.
        
        Returns:
            torch.Tensor: Concatenated action for both robots [left_robot_action, right_robot_action]
        """
        actions = []
        
        for arm, robot in [("left", self.left_robot), ("right", self.right_robot)]:
            arm_name = robot.default_arm
            action = th.zeros(robot.action_dim)
            
            # Get arm command
            arm_cmd = self._joint_cmd.get(f"{arm}_arm")
            if arm_cmd is not None:
                arm_act = arm_cmd.clone().clip(
                    self._arm_joint_limits[arm]["lower"],
                    self._arm_joint_limits[arm]["upper"],
                )
                
                # If we're in cooldown, clip values based on max delta value
                if self._in_cooldown:
                    robot_pos = robot.get_joint_positions()
                    robot_arm_pos = robot_pos[robot.arm_control_idx[arm_name]]
                    robot_delta = arm_act - robot_arm_pos
                    arm_act = robot_arm_pos + robot_delta.clip(
                        -self._reset_max_arm_delta,
                        self._reset_max_arm_delta,
                    )
                
                action[robot.arm_action_idx[arm_name]] = arm_act
            
            # Apply gripper action
            gripper_signal = self._joint_cmd.get(f"{arm}_gripper", th.ones(1)).item()
            gripper_changed = self._gripper_action_signal_detectors[arm].process_sample(gripper_signal)
            if gripper_changed:
                self._grasp_action[arm] = -self._grasp_action[arm]
            action[robot.gripper_action_idx[arm_name]] = 0 if self._grasp_action[arm] <= 0 else 1
            
            actions.append(action)
        
        return th.cat(actions)
    
    def pause(self):
        """Pause the simulation and wait for resume."""
        self._waiting_to_resume = True
        for detector in self._gripper_action_signal_detectors.values():
            detector.reset()
    
    def reset(self, increment_instance: bool = True, save_on_success: bool = True):
        """
        Reset the environment and robot state.
        
        Args:
            increment_instance: Whether to track demo successes (used for multi-instance collection)
            save_on_success: If True, save the episode if it was successful. If False, discard
                the episode regardless of success status.
        """
        print("\nResetting environment...")
        
        # Handle episode saving/discarding based on save_on_success flag
        if not save_on_success:
            self.env.is_recording = False
            print(f"Episode discarding ({len(self.env.current_traj_history)} steps).")
            self.env.flush_current_traj()
            assert len(self.env.current_traj_history) == 0, "Current trajectory history should be empty after flushing"
            self.env.is_recording = True
        
        # Track success on reset (only if save_on_success is True)
        if increment_instance and self._is_success and save_on_success:
            self.demo_success_count += 1
            if self.demo_count_label is not None:
                update_demo_success_counter(self.demo_count_label, self.demo_success_count)
        
        # Reset internal variables
        self._resume_cooldown_time = time.time() + N_COOLDOWN_SECS
        self._in_cooldown = True
        self._waiting_to_resume = True
        self._grasp_action = {"left": 1, "right": 1}
        self._is_success = False

        
        # At the start of every new episode, enable frame filtering (frames excluded until Y is pressed)
        self._is_filtering_frames = True
        if hasattr(self.env, 'filter_current_frame'):
            self.env.filter_current_frame = True
        
        for detector in self._gripper_action_signal_detectors.values():
            detector.reset()
        
        # Initialize joint commands from reset positions
        for arm, robot in [("left", self.left_robot), ("right", self.right_robot)]:
            arm_name = robot.default_arm
            reset_pos = robot.reset_joint_pos
            self._joint_cmd[f"{arm}_arm"] = reset_pos[robot.arm_control_idx[arm_name]]
            self._joint_cmd[f"{arm}_gripper"] = th.ones(1)
        
        self._joint_cmd["button_x"] = th.zeros(1)
        self._joint_cmd["button_y"] = th.zeros(1)
        self._joint_cmd["button_a"] = th.zeros(1)
        self._joint_cmd["button_b"] = th.zeros(1)
        self._joint_cmd["button_home"] = th.zeros(1)
        
        # Reset env
        self.env.reset()
        
        # Update task status display
        if self.text_labels is not None:
            self._prev_status = update_task_status(
                text_labels=self.text_labels,
                goal_status={"success": False},
                prev_goal_status={"success": True},
                env=self.env,
            )
        
        print("Environment reset complete.")
    
    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signal gracefully."""
        print("\n\nShutdown signal received, saving data...")
        self.stop()
    
    def stop(self) -> None:
        """Stop the server and clean up resources."""
        print("\nStopping server...")
        
        self._zmq_server_thread.terminate()
        self._zmq_server_thread.join()
        
        if self._recording_path is not None and hasattr(self.env, 'save_data'):
            # Flush any remaining trajectory data
            if hasattr(self.env, 'current_traj_history') and len(self.env.current_traj_history) > 0:
                self.env.flush_current_traj()
            self.env.save_data()
            print(f"Data saved to {self._recording_path}")
        
        og.shutdown()
        print("Server stopped.")
    
    def __del__(self) -> None:
        """Clean up when object is deleted."""
        try:
            self.stop()
        except Exception:
            pass


@dataclass
class YamJoyLoServerArgs:
    """Command-line arguments for YAM JoyLo Environment Server."""
    
    # Scene and task configuration
    scene_json: str
    """Name of scene JSON to load (located in CDC_ASSET_DIR/scenes, without json extension)."""
    
    task: str
    """Task name (located in CDC_CFG_DIR/task, without yaml extension)."""
    
    use_floor_plane: bool = True
    """Whether to use a floor plane in the scene."""
    
    max_steps: int = 500
    """Maximum steps per episode before termination."""
    
    external_sensors_cfg: str = "sim_yam_teleop_joylo"
    """External sensors config name (located in CDC_CFG_DIR/external_sensors, without yaml extension)."""
    
    action_freq: int = 15
    """Action/rendering frequency in Hz."""
    
    # ZMQ server configuration
    host: str = "127.0.0.1"
    """ZMQ server host."""
    
    port: int = 5556
    """ZMQ server port (handles both arms as 12 total motors)."""
    
    # Recording configuration
    recording_path: Optional[str] = None
    """Path to save HDF5 recordings."""
    
    only_successes: bool = True
    """Only save successful demonstrations."""
    
    flush_every_n_traj: int = 1
    """Flush data every N trajectories."""
    
    viewport_camera_path: Optional[str] = None
    """Path to viewport camera for HDF5 recording (None uses default viewer camera)."""
    
    overwrite: bool = False
    """Whether to overwrite existing HDF5 file."""
    
    # UI configuration
    instance_id: Optional[int] = None
    """Optional instance ID for UI display."""


def main(args: YamJoyLoServerArgs):
    """Main entry point for YAM JoyLo Environment Server."""
    server = YamJoyLoEnvServer(
        scene_json_name=args.scene_json,
        task_name=args.task,
        host=args.host,
        port=args.port,
        use_floor_plane=args.use_floor_plane,
        max_steps=args.max_steps,
        external_sensors_cfg=args.external_sensors_cfg,
        action_freq=args.action_freq,
        recording_path=args.recording_path,
        only_successes=args.only_successes,
        flush_every_n_traj=args.flush_every_n_traj,
        viewport_camera_path=args.viewport_camera_path,
        overwrite=args.overwrite,
        instance_id=args.instance_id,
    )
    
    server.serve()


if __name__ == "__main__":
    import tyro
    
    main(tyro.cli(YamJoyLoServerArgs))

