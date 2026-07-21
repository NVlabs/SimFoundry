"""
YAM Gello Agent

Agent for controlling two YAM 6DOF robot arms via GELLO hardware.
This is adapted from the R1Pro agent for use with separate YAM arms.

The YAM setup:
- 6 DOF per arm (no redundant motors unlike R1/R1Pro)
- 12 total arm motors (6 left + 6 right)
- No trunk or base control
"""

from typing import Dict, Optional

import torch as th
import numpy as np

from gello.agents.joycon_agent import JoyconAgent
from gello.agents.gello_agent import DynamixelRobotConfig, GelloAgent, MotorFeedbackConfig
from gello.dynamixel.driver import OperatingMode


class YamGelloAgent(GelloAgent):
    """
    GELLO Agent for controlling two YAM 6DOF robot arms.
    
    This agent:
    - Handles 12 arm motors (6 per arm) with no redundancy
    - Supports optional JoyCon integration for gripper control and joint locking
    - Provides haptic feedback based on arm contact states
    """
    
    def __init__(
        self,
        port: str,
        dynamixel_config: Optional[DynamixelRobotConfig] = None,
        start_joints: Optional[np.ndarray] = None,
        default_joints: Optional[np.ndarray] = None,
        damping_motor_kp: Optional[float] = 0.0,
        motor_feedback_type: MotorFeedbackConfig = MotorFeedbackConfig.NONE,
        enable_locking_joints: bool = True,
        joycon_agent: Optional[JoyconAgent] = None,
        use_feedback_during_reset: bool = True,
    ):
        """
        Initialize the YAM GELLO agent.
        
        Args:
            port: Serial port for the Dynamixel motors
            dynamixel_config: Configuration for the Dynamixel robot
            start_joints: Starting joint positions
            default_joints: Default joint positions for reset
            damping_motor_kp: Damping gain for motor feedback
            motor_feedback_type: Type of motor feedback to use
            enable_locking_joints: Whether to enable joint locking via JoyCon
            joycon_agent: Optional JoyCon agent for additional control
            use_feedback_during_reset: Whether to use feedback during reset or not (i.e.: motors will try to converge to reset qpos)
        """
        # YAM has 6 joints per arm, 12 total
        self.motors_per_arm = 6
        self.total_motors = self.motors_per_arm * 2
        
        # Create arm info tracking structures
        self.arm_info = {
            "left": {
                "locked": {
                    "upper": False,
                    "lower": False,
                    "all": False,
                },
                "gello_ids": np.arange(self.motors_per_arm),
                "locked_wrist_angle": None,
                "colliding": False
            },
            "right": {
                "locked": {
                    "upper": False,
                    "lower": False,
                    "all": False,
                },
                "gello_ids": np.arange(self.motors_per_arm) + self.motors_per_arm,
                "locked_wrist_angle": None,
                "colliding": False
            },
        }
        
        self.joycon_agent = joycon_agent
        self._motor_feedback_type = motor_feedback_type
        self.use_feedback_during_reset = use_feedback_during_reset
        if self.use_feedback_during_reset:
            assert start_joints is None, "start_joints must be None if use_feedback_during_reset is True"
            assert default_joints is None, "default_joints must be None if use_feedback_during_reset is True"
        
        # Can only enable locking joints if we have a valid joycon agent
        self.enable_locking_joints = enable_locking_joints and self.joycon_agent is not None
        
        # Stores joint offsets to apply dynamically
        self.joint_offsets = np.zeros(self.total_motors)
        
        # Whether we're waiting to resume or not
        self._waiting_to_resume = False
        
        # No feedback by default
        self.default_operation_modes = np.array([OperatingMode.NONE for _ in range(self.total_motors)])
        
        # Run super
        super().__init__(
            port=port,
            dynamixel_config=dynamixel_config,
            start_joints=start_joints,
            default_joints=default_joints,
            damping_motor_kp=damping_motor_kp,
        )
    
    def _gello_joints_to_obs_joints(self, joints: np.ndarray) -> np.ndarray:
        """
        Convert GELLO joint configuration to observation joint configuration.
        
        For YAM, there's a direct 1:1 mapping (no redundant motors).
        
        Args:
            joints: GELLO joint positions (12 values)
            
        Returns:
            Observation joint positions (12 values: 6 left + 6 right)
        """
        # Apply joint offsets and return directly (no redundancy filtering needed)
        return joints - self.joint_offsets
    
    def _obs_joints_to_gello_joints(self, obs: Dict) -> np.ndarray:
        """
        Maps observed joints from the environment into equivalent GELLO joint configuration.
        
        Args:
            obs: Dictionary of observations from the environment
            
        Returns:
            GELLO joint positions (12 values)
        """
        # Convert [left, right] arm qpos into single array
        obs_jnts = np.concatenate([
            obs[f"arm_{arm}_joint_positions"].detach().cpu().numpy() 
            for arm in ["left", "right"]
        ])
        # Add joint offsets
        return obs_jnts + self.joint_offsets
    
    def _obs_to_gello_form(self, obs_jnts: np.ndarray) -> np.ndarray:
        """
        Convert observation joints to GELLO form.
        
        For YAM, this is a direct mapping (no motor duplication).
        
        Args:
            obs_jnts: Observation joint positions
            
        Returns:
            GELLO joint positions
        """
        return obs_jnts
    
    def _gello_to_obs_form(self, gello_jnts: np.ndarray) -> th.Tensor:
        """
        Convert GELLO joints to observation form.
        
        For YAM, this is a direct mapping.
        
        Args:
            gello_jnts: GELLO joint positions
            
        Returns:
            Observation joint positions as torch tensor
        """
        return th.from_numpy(gello_jnts.astype(np.float32))
    
    def compute_feedback_currents(
        self, 
        joint_error: np.ndarray, 
        joint_vel: np.ndarray, 
        obs: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute the feedback currents (i.e. motor torques) given the current discrepancy between
        the sim joints and gello joints.
        
        Args:
            joint_error: Array of joint errors (in degrees), in GELLO format
            joint_vel: Array of joint velocities, in GELLO format
            obs: Dictionary of observations from the sim
            
        Returns:
            Array of feedback currents, one for each GELLO joint
        """
        if self._motor_feedback_type == MotorFeedbackConfig.NONE:
            # No feedback
            current = np.zeros(joint_error.shape)
        
        elif self._motor_feedback_type == MotorFeedbackConfig.JOINT_SPACE:
            # Joint space feedback
            current = (
                self._damping_motor_kp * 0.2 * (joint_error ** 2) * np.sign(joint_error) 
                - self._damping_motor_kv * joint_vel * 0.1
            )
        
        elif self._motor_feedback_type == MotorFeedbackConfig.OPERATIONAL_SPACE:
            # For YAM, we don't have pre-computed Jacobians in observations
            # Fall back to joint space feedback
            current = (
                self._damping_motor_kp * 0.2 * (joint_error ** 2) * np.sign(joint_error) 
                - self._damping_motor_kv * joint_vel * 0.1
            )
        
        else:
            raise ValueError(f"Unexpected joint feedback type: {self._motor_feedback_type}!")
        
        return current
    
    def start(self):
        """Start the agent and set operating modes."""
        super().start()
        
        # Set all joints to default operating modes
        self._robot.set_operating_mode(self.default_operation_modes)
    
    def act(self, obs: Dict) -> th.Tensor:
        """
        Generate action based on current observations.
        
        Args:
            obs: Dictionary of observations from the environment
            
        Returns:
            Action tensor containing joint positions and auxiliary signals
        """
        # Run super first to get joint positions
        jnts = super().act(obs=obs)
        
        # Convert back to gello form
        gello_jnts = self._obs_to_gello_form(jnts.numpy())
        
        # If we see that we're waiting to resume from the sim, reset the joints to the observed values
        if obs["waiting_to_resume"] and not self._waiting_to_resume:
            # Up signal -- track the current pose from the robot and reset to that qpos
            reset_jnts = self._obs_joints_to_gello_joints(obs=obs)
            if self.use_feedback_during_reset:
                self.set_reset_qpos(qpos=reset_jnts)
            self.reset()
            print("Waiting to resume from sim...")
            self._waiting_to_resume = True
        elif not obs["waiting_to_resume"] and self._waiting_to_resume:
            # Down signal
            self.start()
            self._waiting_to_resume = False
        
        # Only compute action if we're not waiting to resume
        if self._waiting_to_resume:
            action = jnts
        else:
            active_operating_mode_idxs = np.array([], dtype=int)
            operating_modes = np.zeros(len(gello_jnts), dtype=int)
            active_commanded_jnt_idxs = np.array([], dtype=int)
            commanded_jnts = gello_jnts + self.joint_offsets
            
            # If we have a joycon agent, we possibly provide additional constraints to GELLO
            if self.enable_locking_joints:
                self._handle_joint_locking(
                    obs,
                    gello_jnts,
                    operating_modes,
                    active_operating_mode_idxs,
                    active_commanded_jnt_idxs,
                    commanded_jnts
                )
            
            # Convert back to observation form for the final action
            action = self._gello_to_obs_form(gello_jnts)
        
        # Get gripper and button values from JoyCon if available
        if self.joycon_agent is not None:
            joycon_action = self.joycon_agent.act(obs)
            # Extract gripper values (left, right) and buttons
            # JoyCon returns: [base_x, base_y, base_r, trunk_translate, trunk_tilt, 
            #                  gripper_l, gripper_r, -, +, X, Y, B, A, capture, home, left, right]
            left_gripper = joycon_action[5]
            right_gripper = joycon_action[6]
            button_x = joycon_action[9]
            button_y = joycon_action[10]
            button_b = joycon_action[11]
            button_a = joycon_action[12]
            button_home = joycon_action[14]
            
            # Compose final action: [6 left arm, 6 right arm, left_gripper, right_gripper, button_x, button_y, button_home]
            action = th.cat([
                action,
                th.tensor([left_gripper, right_gripper, button_x, button_y, button_a, button_b, button_home])
            ])
        else:
            # No JoyCon - append default gripper/button values
            action = th.cat([
                action,
                th.tensor([1.0, 1.0, 0.0, 0.0, 0.0])  # grippers open, no buttons
            ])
        
        return action
    
    def _handle_joint_locking(
        self,
        obs: Dict,
        gello_jnts: np.ndarray,
        operating_modes: np.ndarray,
        active_operating_mode_idxs: np.ndarray,
        active_commanded_jnt_idxs: np.ndarray,
        commanded_jnts: np.ndarray
    ):
        """
        Handle joint locking based on JoyCon input.
        
        For YAM:
        - '-' button locks all left arm joints
        - '+' button locks all right arm joints
        - 'L' button locks lower (wrist) joints of left arm
        - 'R' button locks lower (wrist) joints of right arm
        """
        if self.joycon_agent is None:
            return
        
        for arm, (lock_all, lock_lower) in zip(
            ("left", "right"),
            (
                (self.joycon_agent.gripper_info["-"]["status"], self.joycon_agent.jc_left.get_button_l()),
                (self.joycon_agent.gripper_info["+"]["status"], self.joycon_agent.jc_right.get_button_r()),
            ),
        ):
            arm_info = self.arm_info[arm]
            
            # Handle entire arm locking (triggered by -/+ buttons in toggle mode)
            operating_modes, active_operating_mode_idxs, active_commanded_jnt_idxs = self._handle_all_arm_locking(
                arm_info,
                lock_all == -1,
                gello_jnts,
                operating_modes,
                active_operating_mode_idxs,
                active_commanded_jnt_idxs
            )
            
            # Handle lower arm (wrist) locking (triggered by L/R buttons)
            operating_modes, active_operating_mode_idxs, active_commanded_jnt_idxs = self._handle_lower_arm_locking(
                arm_info,
                lock_lower,
                operating_modes,
                active_operating_mode_idxs,
                active_commanded_jnt_idxs
            )
        
        # Update operating mode if requested
        if len(active_operating_mode_idxs) > 0:
            self._robot.set_operating_mode(operating_modes[active_operating_mode_idxs], idxs=active_operating_mode_idxs)
        
        # Command joints if requested
        if len(active_commanded_jnt_idxs) > 0:
            self._robot.command_joint_state(commanded_jnts[active_commanded_jnt_idxs], idxs=active_commanded_jnt_idxs)
    
    def _handle_all_arm_locking(
        self,
        arm_info: Dict,
        lock_all: bool,
        gello_jnts: np.ndarray,
        operating_modes: np.ndarray,
        active_operating_mode_idxs: np.ndarray,
        active_commanded_jnt_idxs: np.ndarray
    ):
        """
        Handle all arm locking for YAM.
        
        When locked, all 6 joints of the arm are held in position.
        """
        all_currently_locked = arm_info["locked"]["all"]
        
        if lock_all:
            if not all_currently_locked:
                # Just became locked - set all joints to position mode
                operating_modes[arm_info["gello_ids"]] = [OperatingMode.EXTENDED_POSITION] * 6
                active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])
                
                # Add all joints to commanded set
                active_commanded_jnt_idxs = np.concatenate([active_commanded_jnt_idxs, arm_info["gello_ids"]])
                
                # Update lock state
                arm_info["locked"]["all"] = True
        else:
            if all_currently_locked:
                # Just became unlocked - restore default operating modes
                operating_modes[arm_info["gello_ids"]] = self.default_operation_modes[arm_info["gello_ids"]]
                active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])
                
                # Update lock state
                arm_info["locked"]["all"] = False
        
        return operating_modes, active_operating_mode_idxs, active_commanded_jnt_idxs
    
    def _handle_lower_arm_locking(
        self,
        arm_info: Dict,
        lock_lower: bool,
        operating_modes: np.ndarray,
        active_operating_mode_idxs: np.ndarray,
        active_commanded_jnt_idxs: np.ndarray
    ):
        """
        Handle lower arm (wrist) locking for YAM.
        
        For YAM 6DOF, the last 3 joints are considered the wrist/lower arm.
        """
        lower_currently_locked = arm_info["locked"]["lower"]
        
        if lock_lower:
            if not lower_currently_locked:
                # Just became locked - set last 3 joints to position mode
                operating_modes[arm_info["gello_ids"]] = (
                    [OperatingMode.NONE] * 3 + [OperatingMode.EXTENDED_POSITION] * 3
                )
                active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])
                
                # Add lower joints to commanded set
                active_commanded_jnt_idxs = np.concatenate([active_commanded_jnt_idxs, arm_info["gello_ids"][-3:]])
                
                # Update lock state
                arm_info["locked"]["lower"] = True
        else:
            if lower_currently_locked:
                # Just became unlocked - restore default operating modes
                operating_modes[arm_info["gello_ids"]] = self.default_operation_modes[arm_info["gello_ids"]]
                active_operating_mode_idxs = np.concatenate([active_operating_mode_idxs, arm_info["gello_ids"]])
                
                # Update lock state
                arm_info["locked"]["lower"] = False
        
        return operating_modes, active_operating_mode_idxs, active_commanded_jnt_idxs

