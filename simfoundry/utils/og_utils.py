# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import omnigibson as og

_OG_REQUIRED = "3.8.0"
if og.__version__ != _OG_REQUIRED:
    raise ImportError(
        f"OmniGibson {_OG_REQUIRED} is required but {og.__version__} is installed. "
        f"Run the SimFoundry installer to get the correct version (BEHAVIOR-1K commit d89aae4e)."
    )

from omnigibson.macros import gm
import omnigibson.lazy as lazy
from omnigibson.scenes import Scene
from omnigibson.robots import BaseRobot
from omnigibson.prims import VisualGeomPrim
from omnigibson.prims.material_prim import OmniPBRMaterialPrim
from omnigibson.utils.usd_utils import create_primitive_mesh, absolute_prim_path_to_scene_relative
from omnigibson.utils import transform_utils as T
import torch as th
import numpy as np
from typing import Tuple


# OmniGibson simulator settings
USE_FLUID = False
USE_CLOTH = False
OMNIGIBSON_MACROS = {
    "USE_NUMPY_CONTROLLER_BACKEND": True,
    "USE_GPU_DYNAMICS": (USE_FLUID or USE_CLOTH),
    "ENABLE_FLATCACHE": True,
    "ENABLE_OBJECT_STATES": True,  # True (FOR TASKS!)
    "ENABLE_TRANSITION_RULES": True,
    "ENABLE_CCD": True,
    "ENABLE_HQ_RENDERING": USE_FLUID,
    "GUI_VIEWPORT_ONLY": True,
}

UI_SETTINGS = {
    "goal_satisfied_color": 0xFF00FF00,  # Green (ABGR)
    "goal_unsatisfied_color": 0xFF0000FF,  # Red (ABGR)
    "font_size": 25,
    "top_margin": 50,
    "left_margin": 50,
}

# Visualization cylinder configurations
VIS_CYLINDER_CONFIG = {
    "width": 0.01,
    "lengths": [0.25, 0.25, 0.5],            # x,y,z
    "proportion_offsets": [0.0, 0.0, 0.5],   # x,y,z
    "quat_offsets": [
        T.euler2quat(th.tensor([0.0, th.pi / 2, 0.0])),
        T.euler2quat(th.tensor([-th.pi / 2, 0.0, 0.0])),
        T.euler2quat(th.tensor([0.0, 0.0, 0.0])),
    ]
}

# Visualization cylinder configs
VIS_GEOM_COLORS = {
    False: [th.tensor([1.0, 0, 0]),  # Red
            th.tensor([0, 1.0, 0]),  # Green
            th.tensor([0, 0, 1.0]),  # Blue
    ],
    True: [th.tensor([1.0, 0.5, 0.5]),  # Light Red
           th.tensor([0.5, 1.0, 0.5]),  # Light Green
           th.tensor([0.5, 0.5, 1.0]),  # Light Blue
    ]
}


def set_obj_materials(objs: dict, obj_frictions: dict) -> dict:
    # Set physical materials for each object
    obj_materials = dict()
    for obj_name, obj in objs.items():
        if isinstance(obj, BaseRobot):
            continue

        # Create physical material for object
        obj_mat = lazy.isaacsim.core.api.materials.PhysicsMaterial(
            prim_path=f"{obj.prim_path}/Looks/mat",
            name=f"{obj_name}_mat",
            static_friction=obj_frictions[obj_name],
            # dynamic_friction=1.0,
            restitution=None,
        )
        # for link in obj.links.values():
        #     # TODO: Why is inertia used (not ignored during import)? And also why is it bad?
        #     link.set_property("physics:diagonalInertia", lazy.pxr.Gf.Vec3f(0, 0, 0))
        #     # for msh in link.collision_meshes.values():
        #     #     msh.apply_physics_material(obj_mat)
        obj_materials[obj_name] = obj_mat

        objs[obj_name] = obj

    return obj_materials


def setup_task_status_ui(task_name, env, instance_id=None, show_demo_count=True, show_reward=False):
    """
    Set up UI for displaying task instructions and goal status

    Args:
        task_name (str): Name of the task
        env: Environment object
        instance_id (int, optional): Instance ID to display in the top right corner
        show_demo_count (bool, optional): Whether to show demo success counter
        show_reward (bool, optional): Whether to show reward counter

    Returns:
        tuple: (overlay_window, text_labels, instance_id_label, demo_count_label, reward_label, bddl_goal_conditions)
    """
    if task_name is None:
        return None, None, None, None, None, None

    status = {
        "success": env.task.success,
    }

    # Setup overlay window
    main_viewport = og.sim.viewer_camera._viewport
    main_viewport.dock_tab_bar_visible = False
    og.sim.render()

    overlay_window = lazy.omni.ui.Window(
        main_viewport.name,
        width=0,
        height=0,
        flags=lazy.omni.ui.WINDOW_FLAGS_NO_TITLE_BAR |
              lazy.omni.ui.WINDOW_FLAGS_NO_SCROLLBAR |
              lazy.omni.ui.WINDOW_FLAGS_NO_RESIZE
    )
    og.sim.render()

    text_labels = dict()
    instance_id_label = None
    demo_count_label = None
    reward_label = None

    with overlay_window.frame:
        with lazy.omni.ui.ZStack():
            # Bottom layer - transparent spacer
            lazy.omni.ui.Spacer()

            # Create a container for the instance ID in the top right corner
            if instance_id is not None:
                with lazy.omni.ui.VStack(alignment=lazy.omni.ui.Alignment.RIGHT_TOP, spacing=0):
                    lazy.omni.ui.Spacer(height=UI_SETTINGS["top_margin"])  # Top margin
                    with lazy.omni.ui.HStack(height=20):
                        instance_id_label = lazy.omni.ui.Label(
                            f"Instance ID: {instance_id}",
                            alignment=lazy.omni.ui.Alignment.RIGHT_CENTER,
                            style={
                                "color": 0xFFFFFFFF,  # White color (ABGR)
                                "font_size": UI_SETTINGS["font_size"],
                                "margin": 0,
                                "padding": 0
                            }
                        )
                        lazy.omni.ui.Spacer(width=UI_SETTINGS["left_margin"])  # Right margin

            # Text container at top left
            with lazy.omni.ui.VStack(alignment=lazy.omni.ui.Alignment.LEFT_TOP, spacing=0):
                lazy.omni.ui.Spacer(height=UI_SETTINGS["top_margin"])  # Top margin

                # Add demo success counter if enabled
                if show_demo_count:
                    with lazy.omni.ui.HStack(height=20):
                        lazy.omni.ui.Spacer(width=UI_SETTINGS["left_margin"])  # Left margin
                        demo_count_label = lazy.omni.ui.Label(
                            "Demo Successes: 0",
                            alignment=lazy.omni.ui.Alignment.LEFT_CENTER,
                            style={
                                "color": 0xFFFFFF00,  # Cyan (ABGR)
                                "font_size": UI_SETTINGS["font_size"],
                                "margin": 0,
                                "padding": 0
                            }
                        )

                # Add reward counter if enabled
                if show_reward:
                    with lazy.omni.ui.HStack(height=20):
                        lazy.omni.ui.Spacer(width=UI_SETTINGS["left_margin"])  # Left margin
                        reward_label = lazy.omni.ui.Label(
                            "Reward: 0.000",
                            alignment=lazy.omni.ui.Alignment.LEFT_CENTER,
                            style={
                                "color": 0xFFFFAA00,  # Orange (ABGR)
                                "font_size": UI_SETTINGS["font_size"],
                                "margin": 0,
                                "padding": 0
                            }
                        )

                # Create labels for each goal condition
                for k, v in status.items():
                    with lazy.omni.ui.HStack(height=20):
                        lazy.omni.ui.Spacer(width=UI_SETTINGS["left_margin"])  # Left margin
                        label = lazy.omni.ui.Label(
                            f"{k}: {v}",
                            alignment=lazy.omni.ui.Alignment.LEFT_CENTER,
                            style={
                                "color": UI_SETTINGS["goal_unsatisfied_color"],  # Red color (ABGR)
                                "font_size": UI_SETTINGS["font_size"],
                                "margin": 0,
                                "padding": 0
                            }
                        )
                        text_labels[k] = label

    # Force render to update the overlay
    og.sim.render()

    return overlay_window, text_labels, instance_id_label, demo_count_label, reward_label, status


def update_task_status(text_labels, goal_status, prev_goal_status, env):
    """
    Update the UI based on goal status changes

    Args:
        text_labels: List of UI text labels
        goal_status: Current goal status
        prev_goal_status: Previous goal status
        env: Environment object

    Returns:
        dict: Updated previous goal status
    """
    if text_labels is None:
        return prev_goal_status

    # Check if status has changed
    status_changed = any(v != prev_goal_status[k] for k, v in goal_status.items())

    if status_changed:
        # Differentiate based on color
        for k, v in goal_status.items():
            current_style = text_labels[k].style
            current_style.update({"color": UI_SETTINGS["goal_satisfied_color" if bool(v) else "goal_unsatisfied_color"]})
            text_labels[k].set_style(current_style)
            text_labels[k].text = f"{k}: {v}"

        # # Render to force change
        # og.sim.render()

        # Return the updated status
        return goal_status.copy()

    return prev_goal_status


def update_demo_success_counter(demo_count_label, success_count):
    """
    Update the demo success counter display

    Args:
        demo_count_label: UI label for demo success counter
        success_count (int): Current number of successful demos
    """
    if demo_count_label is not None:
        demo_count_label.text = f"Demo Successes: {success_count}"


def update_reward_ui(reward_label, reward):
    """
    Update the reward display in the UI

    Args:
        reward_label: UI label for reward display
        reward (float): Current cumulative reward value
    """
    if reward_label is not None:
        reward_label.text = f"Reward: {reward:.3f}"


def setup_robot_visualizers(robot, scene, offset=None):
    """
    Set up visualization elements for teleop

    Args:
        robot: The robot object
        scene: The scene object
        offset: Offset to apply to the visualization cylinders
    Returns:
        dict: Dictionary of visualization elements
    """
    assert og.sim.is_stopped(), "Sim must be stopped to setup robot visualizers!"
    vis_elements = {
        "eef_cylinder_geoms": {},
        "vis_mats": {},
        "vertical_visualizers": {},
        "reachability_visualizers": {}
    }

    if offset is None:
        offset = th.tensor([0.0, 0.0, 0.0])

    # Create materials for visualization cylinders
    for arm in robot.arm_names:
        vis_elements["vis_mats"][arm] = []
        for axis, color in zip(("x", "y", "z"), VIS_GEOM_COLORS[False]):
            mat_prim_path = f"{robot.prim_path}/Looks/vis_cylinder_{arm}_{axis}_mat"
            mat = OmniPBRMaterialPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(scene, mat_prim_path),
                name=f"{robot.name}:vis_cylinder_{arm}_{axis}_mat",
            )
            mat.load(scene)
            mat.diffuse_color_constant = color
            vis_elements["vis_mats"][arm].append(mat)

    # Extract visualization cylinder settings
    vis_geom_width = VIS_CYLINDER_CONFIG["width"]
    vis_geom_lengths = VIS_CYLINDER_CONFIG["lengths"]
    vis_geom_proportion_offsets = VIS_CYLINDER_CONFIG["proportion_offsets"]
    vis_geom_quat_offsets = VIS_CYLINDER_CONFIG["quat_offsets"]

    # Create visualization cylinders for each arm
    for arm in robot.arm_names:
        hand_link = robot.eef_links[arm]
        vis_elements["eef_cylinder_geoms"][arm] = []
        for axis, length, mat, prop_offset, quat_offset in zip(
                ("x", "y", "z"),
                vis_geom_lengths,
                vis_elements["vis_mats"][arm],
                vis_geom_proportion_offsets,
                vis_geom_quat_offsets,
        ):
            vis_prim_path = f"{hand_link.prim_path}/visuals/vis_cylinder_{axis}"
            vis_prim = create_primitive_mesh(
                vis_prim_path,
                "Cylinder",
                extents=1.0
            )
            vis_geom = VisualGeomPrim(
                relative_prim_path=absolute_prim_path_to_scene_relative(scene, vis_prim_path),
                name=f"{robot.name}:arm_{arm}:vis_cylinder_{axis}"
            )
            vis_geom.load(scene)

            # Attach a material to this prim
            vis_geom.material = mat

            vis_geom.scale = th.tensor([vis_geom_width, vis_geom_width, length])
            vis_geom.set_position_orientation(
                position=offset + th.tensor([0, 0, length * prop_offset]),
                orientation=quat_offset,
                frame="parent"
            )
            vis_elements["eef_cylinder_geoms"][arm].append(vis_geom)

    return vis_elements


def update_grasp_status(robot, eef_cylinder_geoms, prev_grasp_status):
    """
    Update the visualization based on whether robot is grasping

    Args:
        robot: Robot object
        eef_cylinder_geoms: End effector cylinder geometries
        prev_grasp_status: Previous grasp status

    Returns:
        dict: Updated grasp status
    """
    updated_status = prev_grasp_status.copy()

    for arm in robot.arm_names:
        is_grasping = robot.is_grasping(arm) > 0
        if is_grasping != prev_grasp_status[arm]:
            updated_status[arm] = is_grasping
            for cylinder in eef_cylinder_geoms[arm]:
                cylinder.visible = not is_grasping

    return updated_status


def update_in_hand_status(robot, vis_mats, prev_in_hand_status):
    """
    Update the visualization based on whether objects are in hand

    Args:
        robot: Robot object
        vis_mats: Visualization materials dictionary
        prev_in_hand_status: Previous in-hand status

    Returns:
        dict: Updated in-hand status
    """
    updated_status = prev_in_hand_status.copy()

    # Update the in-hand status of the robot's arms
    for arm in robot.arm_names:
        in_hand = len(robot._find_gripper_raycast_collisions(arm)) != 0
        if in_hand != prev_in_hand_status[arm]:
            updated_status[arm] = in_hand
            for idx, mat in enumerate(vis_mats[arm]):
                mat.diffuse_color_constant = VIS_GEOM_COLORS[in_hand][idx]

    return updated_status


def apply_teleop_omnigibson_macros(enable_tr=True):
    """Apply global OmniGibson settings"""
    for key, value in OMNIGIBSON_MACROS.items():
        if key == "ENABLE_TRANSITION_RULES":
            value = enable_tr
        setattr(gm, key, value)


def setup_wrist_camera_viewport(env, robot_id=0, width=256, height=256):
    """
    Configure the wrist camera viewport size for debugging purposes.

    Args:
        env: OmniGibson environment
        robot_id: Robot ID to setup viewport for
        width: Viewport width in pixels (default: 256)
        height: Viewport height in pixels (default: 256)
    """
    assert len(env.robots[robot_id].sensors) == 1, "Expected exactly one sensor on robot"

    # Adjust wrist camera viewport size
    wrist_cam = next(iter(env.robots[robot_id].sensors.values()))
    wrist_cam._viewport.width = width
    wrist_cam._viewport.height = height

    print(f"  Configured wrist camera viewport: {width}x{height}")


def setup_debug_viewports(env, wrist_width=256, wrist_height=256):
    """
    Configure debug viewports for visualization.

    This function sets up the wrist camera viewport for debugging
    and can be extended to configure additional viewports as needed.

    Args:
        env: OmniGibson environment
        wrist_width: Wrist camera viewport width (default: 256)
        wrist_height: Wrist camera viewport height (default: 256)
    """
    # Setup wrist camera viewport
    setup_wrist_camera_viewport(env, width=wrist_width, height=wrist_height)


def draw_trajectory_gradient(positions, color_scheme="green_to_red", point_size=10, clear_previous=True):
    """
    Draw a trajectory with gradient coloring to show progression.

    Args:
        positions: List or array of (x, y, z) positions in world frame
        color_scheme: Color scheme to use. Options:
            - "green_to_red": Green → Yellow → Red (progress/heat indicator)
            - "blue_to_red": Blue → Purple → Red (cool to hot)
            - "cyan_to_yellow": Cyan → Green → Yellow (viridis-style)
        point_size: Size of the points to draw
        clear_previous: Whether to clear previously drawn points
    """
    if len(positions) == 0:
        return

    debug_draw = lazy.isaacsim.util.debug_draw._debug_draw.acquire_debug_draw_interface()

    if clear_previous:
        debug_draw.clear_points()

    # Generate gradient colors based on scheme
    n_points = len(positions)
    colors = []

    for i in range(n_points):
        # Interpolation factor from 0 to 1
        t = i / (n_points - 1) if n_points > 1 else 0

        if color_scheme == "green_to_red":
            # Green (0,1,0) → Yellow (1,1,0) → Red (1,0,0)
            if t < 0.5:
                # Green to Yellow: increase red from 0 to 1
                r = 2 * t
                g = 1.0
                b = 0.0
            else:
                # Yellow to Red: decrease green from 1 to 0
                r = 1.0
                g = 2 * (1 - t)
                b = 0.0
            colors.append((r, g, b, 1.0))

        elif color_scheme == "blue_to_red":
            # Blue (0,0,1) → Purple → Red (1,0,0)
            r = t
            g = 0.0
            b = 1.0 - t
            colors.append((r, g, b, 1.0))

        elif color_scheme == "cyan_to_yellow":
            # Cyan (0,1,1) → Green (0,1,0) → Yellow (1,1,0)
            if t < 0.5:
                # Cyan to Green: decrease blue from 1 to 0
                r = 0.0
                g = 1.0
                b = 1.0 - 2 * t
            else:
                # Green to Yellow: increase red from 0 to 1
                r = 2 * (t - 0.5)
                g = 1.0
                b = 0.0
            colors.append((r, g, b, 1.0))

        else:
            raise ValueError(f"Unknown color scheme: {color_scheme}")

    # Convert positions to list format if needed
    if not isinstance(positions, list):
        positions = [pos.tolist() if hasattr(pos, 'tolist') else list(pos) for pos in positions]

    # Draw all points
    debug_draw.draw_points(
        positions,
        colors,
        [point_size] * n_points
    )

    # Render to make points visible
    og.sim.step()
    for _ in range(5):
        og.sim.render()


def clear_trajectory_visualization():
    """Clear all trajectory visualization points."""
    debug_draw = lazy.isaacsim.util.debug_draw._debug_draw.acquire_debug_draw_interface()
    debug_draw.clear_points()
    og.sim.step()
    og.sim.render()

