# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Run YAM JoyLo Teleoperation

A stripped-down version of run_joylo.py specifically for controlling two YAM 6DOF robot arms
via GELLO hardware for teleoperation and data collection.

Usage:
    python run_yam_joylo.py --robot-port 5556 --hostname 127.0.0.1

The YAM JoyLo environment server must be running before starting this script.
"""

import glob
import time
import yaml
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tyro

from gello.agents.agent import Agent
from gello.agents.gello_agent import DynamixelRobotConfig, MotorFeedbackConfig
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot
from gello import REPO_DIR

from digital_cousins.envs.omnigibson.joylo.yam_gello_agent import YamGelloAgent


def print_color(*args, color=None, attrs=(), **kwargs):
    """Print text with color in the terminal."""
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class Args:
    """Command-line arguments for YAM JoyLo teleoperation."""
    
    # ZMQ connection settings
    robot_port: int = 5556
    """Port for ZMQ communication with the YAM JoyLo environment server."""
    
    hostname: str = "127.0.0.1"
    """Hostname for ZMQ communication."""
    
    # Control settings
    hz: int = 100
    """Control frequency in Hz."""
    
    start_joints: Optional[Tuple[float, ...]] = None
    """Starting joint positions (12 values: 6 left + 6 right). If None, uses default."""
    
    # GELLO settings
    joint_config_file: str = "joint_config_black_yam.yaml"
    """YAML file containing joint offsets and signs for GELLO calibration."""
    
    gello_port: Optional[str] = None
    """Serial port for GELLO. If None, auto-detects."""
    
    mock: bool = False
    """Use mock robot for testing without hardware."""
    
    damping_motor_kp: float = 0.0
    """Motor damping gain for feedback. Set to 0 for no feedback."""
    
    motor_feedback_type: str = "NONE"
    """Motor feedback type: NONE, JOINT_SPACE, or OPERATIONAL_SPACE."""
    
    # JoyCon settings
    use_joycons: bool = True
    """Whether to use JoyCons for gripper control and additional inputs."""
    
    # Debug settings
    verbose: bool = False
    """Enable verbose output."""
    
    debug_joints: bool = False
    """Print real-time GELLO joint positions (in degrees) for calibration debugging."""
    
    use_feedback_during_reset: bool = True
    """Whether to use feedback during reset or not (i.e.: motors will try to converge to reset qpos)."""

def main(args: Args):
    """Main entry point for YAM JoyLo teleoperation."""
    
    # Create robot client
    if args.mock:
        robot_client = PrintRobot(12, dont_print=True)
    else:
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    
    # Create environment
    env = RobotEnv(
        robot_client, 
        control_rate_hz=args.hz, 
        camera_dict={},
        active_arm="right",  # Not used for YAM but required by RobotEnv
    )
    
    # Auto-detect GELLO port if not specified
    gello_port = args.gello_port
    if gello_port is None:
        import platform

        if platform.system().lower() == "linux":
            usb_ports = glob.glob("/dev/serial/by-id/*")
        elif platform.system().lower() == "darwin":
            usb_ports = glob.glob("/dev/cu.usbserial-*")
        else:
            raise ValueError(f"Unsupported platform {platform.system()}")
        
        print(f"Found {len(usb_ports)} USB ports")
        if len(usb_ports) > 0:
            gello_port = usb_ports[0]
            print(f"Using GELLO port: {gello_port}")
        else:
            raise ValueError(
                "No GELLO port found. Please specify one with --gello-port or plug in GELLO hardware."
            )
    
    # Load joint configuration from YAML
    try:
        with open(args.joint_config_file, "r") as file:
            joint_config = yaml.load(file, Loader=yaml.SafeLoader)
    except FileNotFoundError:
        raise ValueError(f"Joint config file not found at {args.joint_config_file}")
    
    # YAM has 12 motors total (6 per arm)
    num_motors = 12
    
    # Create Dynamixel configuration
    dynamixel_config = DynamixelRobotConfig(
        joint_ids=tuple(np.arange(num_motors).tolist()),
        joint_offsets=[np.deg2rad(x) for x in joint_config['joints']['offsets']],
        joint_signs=joint_config['joints']['signs'],
        gripper_config=None,
    )
    
    # Set default start joints if not specified
    start_joints = args.start_joints
    if start_joints is None:
        # Default neutral pose for YAM arms
        # Format: [6 left arm joints, 6 right arm joints]
        start_joints = np.array([
            0.0, 1.047, 1.047, 0.0, 0.0, 0.0,  # Left arm (matches yam.py default)
            0.0, 1.047, 1.047, 0.0, 0.0, 0.0,  # Right arm (mirrored)
        ])
    else:
        start_joints = np.array(start_joints)
    
    # Create JoyCon agent if enabled
    joycon_agent = None
    if args.use_joycons:
        from gello.agents.joycon_agent import JoyconAgent
        joycon_agent = JoyconAgent(
            calibration_dir=f"{REPO_DIR}/configs",
            deadzone_threshold=0.2,
            max_translation=0.35,
            max_rotation=0.3,
            max_trunk_translate=0.1,
            max_trunk_tilt=0.05,
            enable_rumble=False,
        )
        print_color("JoyCons connected successfully!", color="green")
    
    # Create YAM GELLO agent
    agent = YamGelloAgent(
        port=gello_port,
        dynamixel_config=dynamixel_config,
        start_joints=start_joints if args.use_feedback_during_reset else None,       # If set, will apply motor gains to start position
        default_joints=None,
        damping_motor_kp=args.damping_motor_kp,
        motor_feedback_type=MotorFeedbackConfig[args.motor_feedback_type],
        enable_locking_joints=True,
        joycon_agent=joycon_agent,
        use_feedback_during_reset=args.use_feedback_during_reset,
    )
    
    # Start the agent
    agent.start()
    
    # Move to start position
    print_color("Moving to start position...", color="cyan")
    agent.reset()
    
    # Print welcome message and controls
    print_color("\n" + "=" * 60, color="magenta", attrs=("bold",))
    print_color("Welcome to YAM JoyLo Teleoperation!", color="magenta", attrs=("bold",))
    print_color("=" * 60, color="magenta", attrs=("bold",))
    print_color("\nControls:", color="magenta", attrs=("bold",))
    print_color("  GELLO Arms: Move the physical GELLO arms to control the robot", color="white")
    
    if joycon_agent is not None:
        print_color("\nJoyCon Controls:", color="cyan", attrs=("bold",))
        print_color("  ZL / ZR: Toggle gripper (left / right)", color="white")
        print_color("  L / R: Lock lower 3 wrist joints", color="white")
        print_color("  - / +: Lock all arm joints", color="white")
        print_color("  X: Toggle recording", color="white")
        print_color("  Y: Move robot towards reset pose", color="white")
        print_color("  Home: Reset environment", color="white")
    else:
        print_color("\nNote: JoyCons not connected. Grippers default to open.", color="yellow")
    
    print_color("\nKeyboard Controls (in OmniGibson window):", color="cyan", attrs=("bold",))
    print_color("  R: Reset environment", color="white")
    print_color("  P: Pause simulation", color="white")
    print_color("  X: Resume control", color="white")
    print_color("  T: Toggle recording", color="white")
    print_color("  ESC: Stop and exit", color="white")
    
    print_color("\n" + "=" * 60, color="magenta", attrs=("bold",))
    print_color("Starting teleoperation... 🚀", color="green", attrs=("bold",))
    print_color("=" * 60 + "\n", color="magenta", attrs=("bold",))
    
    if args.debug_joints:
        while True:
            # Get raw joint positions from the GELLO agent (in radians)
            raw_joints = agent._robot.get_joint_state()
            raw_joints_deg = np.rad2deg(raw_joints)
            left_joints = raw_joints_deg[:6]
            right_joints = raw_joints_deg[6:12]
            left_str = " ".join([f"{j:7.2f}" for j in left_joints])
            right_str = " ".join([f"{j:7.2f}" for j in right_joints])
            print(f"\rL: [{left_str}]  R: [{right_str}]", end="", flush=True)
    
    else:
        # Main control loop
        obs = env.get_obs()
        start_time = time.time()
        
        try:
            while True:
                # Display elapsed time
                elapsed = time.time() - start_time
                # Get action from agent and step environment
                action = agent.act(obs)
                msg = f"\rTime: {elapsed:.2f}s | Recording: {obs.get('is_recording', False)}" if args.verbose else f"\rTime: {elapsed:.2f}s"
                print_color(msg, color="white", attrs=("bold",), end="", flush=True)
                obs = env.step(action)
            
        except KeyboardInterrupt:
            print_color("\n\nStopping teleoperation...", color="yellow")
    
    print_color("YAM JoyLo teleoperation ended.", color="cyan")


if __name__ == "__main__":
    main(tyro.cli(Args))

