# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Interactive Scene Editor for OmniGibson

Should be run from b1k environment.

This script allows you to:
- Load a 3DGS background from a specified filepath
- Load objects from specified USD file path(s)
- Interactively move/rotate/scale objects using keyboard inputs
- Save the scene state with a button press

Key Controls:
    Object Selection:
        TAB         - Cycle through objects (forward)
        GRAVE (`)   - Cycle through objects (backward) 
        
    Translation (in world frame):
        Arrow keys  - Move in XY plane (UP/DOWN = +/-X, LEFT/RIGHT = +/-Y)
        W/S  - Move up/down (Z axis)
        
    Rotation:
        N/M         - Rotate around X axis (pitch)
        ,/<         - Rotate around Y axis (roll) 
        ./> and /   - Rotate around Z axis (yaw)
        
    Scale:
        +/-         - Uniform scale up/down
        [/]         - Scale up/down (alternate keys)
        
    State Management:
        ENTER       - Save scene state
        BACKSPACE   - Reset selected object to initial pose
        U           - Undo last operation
        F1          - Print current object pose
        
    Camera:
        The viewer camera can be controlled with mouse. Click and drag to rotate,
        scroll to zoom, right-click drag to pan.
        
    System:
        ESC         - Exit

Requires installing:
    - BEHAVIOR-1K, see https://github.com/StanfordVL/BEHAVIOR-1K
"""

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.lazy as lazy
from omnigibson.scenes import Scene
from omnigibson.objects import USDObject, DatasetObject
from omnigibson.prims import XFormPrim
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.sensors import create_sensor
from omnigibson.utils.ui_utils import KeyboardEventHandler
from omnigibson.utils.config_utils import parse_config
import omnigibson.utils.transform_utils as T
from pathlib import Path
from copy import deepcopy
import torch as th
import json
import argparse
import os
import shutil
import numpy as np
import math
from datetime import datetime
from simfoundry import ASSET_DIR
from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths


def resolve_prim(stage, path, lazy):
    """Return the prim at the given path if it exists and is valid, else None."""
    prim = stage.GetPrimAtPath(path)
    return prim if prim is not None and prim.IsValid() else None


def find_first_mesh(root_prim, lazy):
    """Return the first UsdGeom.Mesh prim under root_prim (inclusive), else None."""
    for prim in lazy.pxr.Usd.PrimRange(root_prim):
        if prim.IsA(lazy.pxr.UsdGeom.Mesh):
            return prim
    # Fall back to any Gprim (e.g. a Plane) if no Mesh exists.
    for prim in lazy.pxr.Usd.PrimRange(root_prim):
        if prim.IsA(lazy.pxr.UsdGeom.Gprim):
            return prim
    return None


def find_prims_by_name(root_prim, name, lazy):
    """Return all prims under root_prim (inclusive) whose name matches exactly."""
    return [prim for prim in lazy.pxr.Usd.PrimRange(root_prim) if prim.GetName() == name]


def set_matte(prim, lazy):
    """Set primvars:isMatteObject=True on the prim (creating the attribute if needed)."""
    matte_attr_name = "primvars:isMatteObject"
    if not prim.HasProperty(matte_attr_name):
        prim.CreateAttribute(matte_attr_name, lazy.pxr.Sdf.ValueTypeNames.Bool)
    prim.GetProperty(matte_attr_name).Set(True)


def apply_realistic_render_settings(og, lazy, gs_pending=False) -> None:
    """gauss.proxy → floor mesh; floor mesh matte + visible; auto-exposure + whitepoint=3.7.

    Uses the explicit prim paths the GUI exposes:
      - gauss Volume: /World/scene_0/gs_background/gauss/gauss
      - floor Mesh:   /World/ground_plane/geom
    Falls back to a name/type search if those exact paths aren't found.

    gs_pending: set True when a GS background will be loaded after this call — suppresses
    the non-GS histogram/rendermode settings that would disturb the NuRec compositor init.
    """
    stage = lazy.isaacsim.core.utils.stage.get_current_stage()

    # --- Floor mesh: prefer explicit /World/ground_plane/geom ---
    floor_mesh = resolve_prim(stage, "/World/ground_plane/geom", lazy)
    if floor_mesh is None and og.sim.floor_plane is not None:
        print("[render] /World/ground_plane/geom not found; falling back to first Mesh under floor_plane")
        floor_mesh = find_first_mesh(og.sim.floor_plane.prim, lazy)

    # --- Gauss prim: prefer explicit /World/scene_0/gs_background/gauss/gauss ---
    gauss_prim = resolve_prim(stage, "/World/scene_0/gs_background/gauss/gauss", lazy)
    if gauss_prim is None:
        # Fall back: find the deepest prim named 'gauss' under any gs_background object.
        for obj in og.sim.scenes[0].objects:
            if "gs_background" in obj.name:
                matches = find_prims_by_name(obj.root_link.prim, "gauss", lazy)
                if matches:
                    # Deepest one is the inner Volume; outer Xforms have shallower paths.
                    gauss_prim = max(matches, key=lambda p: len(p.GetPath().pathString))
                    print(f"[render] /World/scene_0/gs_background/gauss/gauss not found; "
                          f"using deepest 'gauss' match {gauss_prim.GetPrimPath()}")
                break

    # Determine whether a GS background is active or about to be loaded.
    has_gs = gauss_prim is not None or gs_pending

    if floor_mesh is None:
        print("[render] no floor mesh found — skipping matte + proxy setup")
    else:
        # Matte floor: only when NO GS active.
        # NuRec registered compositing runs as a post-process AFTER the RTX render.
        # Matte objects show the pre-compositor background (black), NOT the NuRec GS
        # output.  Enabling matte with GS → floor area appears black instead of GS.
        # When GS is active, render the floor normally so GS is composited behind it.
        if not has_gs:
            set_matte(floor_mesh, lazy)
            print(f"[render] floor mesh {floor_mesh.GetPrimPath()} → isMatteObject=True, visible")
        else:
            print(f"[render] floor mesh {floor_mesh.GetPrimPath()} → normal (no matte; GS active)")
        # Make the mesh itself visible (user explicitly wants it visible, not just inherited).
        vis_attr = floor_mesh.GetAttribute("visibility")
        if vis_attr is not None:
            vis_attr.Set(lazy.pxr.UsdGeom.Tokens.inherited)
        # And make sure the parent chain is inherited too (so the leaf visibility actually
        # propagates from "inherited" → "inherited" → "visible").
        parent = floor_mesh.GetParent()
        while parent and parent.IsValid() and parent.GetPath() != lazy.pxr.Sdf.Path("/"):
            v = parent.GetAttribute("visibility")
            if v is not None:
                v.Set(lazy.pxr.UsdGeom.Tokens.inherited)
            parent = parent.GetParent()

    if gauss_prim is None:
        print("[render] no gauss prim found — skipping proxy setup")
    elif floor_mesh is None:
        print("[render] gauss proxy not set (no floor mesh to point at)")
    else:
        proxy_rel = gauss_prim.GetRelationship("proxy")
        if proxy_rel is None or not proxy_rel.IsValid():
            proxy_rel = gauss_prim.CreateRelationship("proxy")
        current_targets = proxy_rel.GetTargets() if proxy_rel and proxy_rel.IsValid() else []
        if not current_targets:
            # Only set if not already set from load_background — avoid a Hydra resync of
            # the NuRec Volume prim after the compositor has already initialized.
            proxy_rel.SetTargets([floor_mesh.GetPrimPath()])
            print(f"[render] {gauss_prim.GetPrimPath()}.proxy → {floor_mesh.GetPrimPath()}")
        else:
            print(f"[render] {gauss_prim.GetPrimPath()}.proxy already set ({current_targets}), skipping")

    # When GS is active (or about to load), skip pipeline setting changes entirely.
    # Any og.app.set_setting call that touches rendermode/histogram after NuRec has
    # committed its first render (or before it's had a chance to init cleanly) will
    # destroy the compositor.  Without GS, apply standard auto-exposure settings.
    if not has_gs:
        settings = [
            ("/rtx/rendermode", "RaytracedLighting"),
            ("/rtx/matteObject/visibility/secondaryRays", True),
            ("/rtx/post/histogram/enabled", True),
            ("/rtx/post/histogram/whiteScale", 3.7),
        ]
        for key, value in settings:
            try:
                og.app.set_setting(key, value)
                print(f"[render] {key} = {value}")
            except Exception as e:
                print(f"[render] failed to set {key}={value}: {e}")


class InteractiveSceneEditor:
    """
    Interactive scene editor that allows manipulation of objects via keyboard.
    """
    
    def __init__(
        self,
        scene_name,
        gs_background_path=None,
        mesh_background_path=None,
        hdr_background_path=None,
        object_paths=None,
        object_poses=None,
        dataset_objects=None,
        usd_objects=None,
        scene_objects_info_path=None,
        pb_scene_poses_path=None,
        scene_objects_categories=None,
        cam2world_tf=None,
        asset_dir=ASSET_DIR,
        translation_delta=0.01,
        rotation_delta=0.05,
        scale_delta=0.05,
        robot_configs=None,
        arm_controller="ik",
        ground_plane=False,
        external_sensors_config=None,
        skybox_rot_x_deg=0.0,
        skybox_rot_y_deg=0.0,
        skybox_yaw_deg=0.0,
        skybox_intensity=1000.0,
        skybox_yaw_step_deg=2.0,
        skybox_intensity_scale=1.1,
        load_scene_json=None,
    ):
        """
        Initialize the interactive scene editor.
        
        Args:
            scene_name (str): Name of the scene. Used for output directory and file naming.
            gs_background_path (str): Path to 3DGS background USDZ file
            mesh_background_path (str): Path to mesh background USD file (not copied on save)
            hdr_background_path (str): Path to HDR background .exr file for skybox texture
            object_paths (list): List of paths to object USD files
            object_poses (list): List of (position, orientation) tuples for initial poses
            dataset_objects (list): List of dataset object specs as dicts with keys:
                'dataset_name', 'category', 'model', and optionally 'name'
            usd_objects (list): List of USD object specs as dicts with keys:
                'category', 'usd_path', and optionally 'name'
            scene_objects_info_path (str): Path to scene_objects_info.json from pipeline stage 10
            pb_scene_poses_path (str): Path to pb_scene_poses.json from pipeline stage 11 (s11_physics)
            scene_objects_categories (list): List of category names to filter when loading from 
                scene_objects_info. If None, all categories are loaded.
            cam2world_tf (th.Tensor): 4x4 camera-to-world transform for positioning background
            asset_dir (str): Base assets directory. Defaults to <repo>/assets
            translation_delta (float): Step size for translation movements (default: 0.01)
            rotation_delta (float): Step size for rotation movements (radians) (default: 0.05)
            scale_delta (float): Step size for scale changes (default: 0.05)
            robot_configs (list): List of robot configuration dicts with keys like 'type', 'end_effector', 
                'position', 'orientation', 'count'. If None, no robots are spawned.
            arm_controller (str): Type of arm controller to use. Options: 'ik' for InverseKinematicsController,
                'joint_pos' for JointController with position control. Default: 'ik'.
            ground_plane (bool): Whether to include a ground plane in the scene. Default: False.
            external_sensors_config (str): Path to a .yaml file containing external sensor configurations.
            skybox_rot_x_deg (float): Initial skybox rotation around X axis in degrees.
            skybox_rot_y_deg (float): Initial skybox rotation around Y axis in degrees.
            skybox_yaw_deg (float): Initial skybox yaw in degrees for HDR lighting.
            skybox_intensity (float): Initial skybox intensity.
            skybox_yaw_step_deg (float): Per-key skybox yaw step in degrees.
            skybox_intensity_scale (float): Per-key multiplicative intensity scale.
            load_scene_json (str): Path to a pre-saved scene JSON file to load. If provided, the scene will be restored from this file instead of creating a new one.
        """
        self.scene_name = scene_name
        self.gs_background_path = gs_background_path
        self.mesh_background_path = mesh_background_path
        self.hdr_background_path = hdr_background_path
        self.object_paths = object_paths or []
        self.object_poses = object_poses or []
        self.dataset_objects = dataset_objects or []
        self.usd_objects = usd_objects or []
        self.scene_objects_info_path = scene_objects_info_path
        self.pb_scene_poses_path = pb_scene_poses_path
        self.scene_objects_categories = set(scene_objects_categories) if scene_objects_categories else None
        self.cam2world_tf = cam2world_tf
        
        # Setup output directory: <asset_dir>/scenes/<scene_name>
        self.asset_dir = Path(asset_dir if asset_dir else ASSET_DIR)
        self.output_dir = self.asset_dir / "scenes" / scene_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Objects subdirectory for copied USD files
        self.objects_dir = self.output_dir / "objects"
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        
        # Control parameters
        self.translation_delta = translation_delta
        self.rotation_delta = rotation_delta
        self.scale_delta = scale_delta
        
        # Robot configuration
        self.robot_configs = robot_configs or []
        self.arm_controller = arm_controller
        self.robots = []  # List of loaded robots
        self.robot = None  # Reference to first robot (for backward compatibility)
        
        # Scene options
        self.ground_plane = ground_plane
        self.external_sensors_config = external_sensors_config
        self.external_sensors = {}  # Dict to store loaded sensors

        # Skybox lighting controls
        self.skybox_rot_x = float(skybox_rot_x_deg) * (math.pi / 180.0)
        self.skybox_rot_y = float(skybox_rot_y_deg) * (math.pi / 180.0)
        self.skybox_rot_z = float(skybox_yaw_deg) * (math.pi / 180.0)
        self.skybox_intensity = max(0.0, float(skybox_intensity))
        self.skybox_rot_delta = max(0.1, float(skybox_yaw_step_deg)) * (math.pi / 180.0)
        self.skybox_intensity_scale = max(1.01, float(skybox_intensity_scale))
        
        # Scene loading
        self.load_scene_json = load_scene_json
        
        # State tracking
        self.objects = {}
        self.object_names = []
        self.selected_idx = 0
        self.initial_poses = {}
        self.initial_scales = {}
        self.gs_background = None
        self.mesh_background = None
        self.scene = None
        self.shift_pressed = False
        self.mesh_background_state_to_restore = None
        
        # Track original USD paths for copying during save
        self.usd_object_paths = {}  # obj_name -> original usd_path
        
        # Undo stack: list of snapshots, each is a dict mapping obj_name -> (pos, ori, scale)
        self.undo_stack = []
        self.max_undo = 100
        
        # Soft-deleted objects: hidden but still in scene, excluded from save
        self.soft_deleted = set()
        
        # Group mode: when enabled, operations apply to all scene_objects_info objects
        self.group_mode = False
        self.scene_objects_info_names = []  # Track objects loaded from scene_objects_info
        
    def setup_scene(self):
        """Create and setup the OmniGibson scene."""
        # Determine whether to show floor/skybox based on background presence
        include_gs = self.gs_background_path is not None
        include_mesh_bg = self.mesh_background_path is not None
        include_any_background = include_gs or include_mesh_bg
        use_gs_shadow_catcher = include_gs
        
        # Always use skybox: it provides ambient lighting for geometry even when the GS
        # background replaces the visual sky.  Without it the scene is pitch-black and
        # NuRec's compositor has nothing to work with.  If an explicit HDR path is given
        # it overrides the default skybox texture; otherwise the default dome is used.
        use_skybox = True
        
        scene_cfg = {
            "type": "Scene",
            # For GS compositing we still need a floor plane as shadow catcher proxy.
            "use_floor_plane": self.ground_plane or use_gs_shadow_catcher,
            # Keep floor visible when used as GS shadow catcher; matte will hide base color.
            "floor_plane_visible": (self.ground_plane and (not include_any_background)) or use_gs_shadow_catcher,
            "use_skybox": use_skybox,
        }
        
        og_cfg = dict(scene=scene_cfg)
        
        # Create environment
        env = og.Environment(configs=og_cfg)
        env.reset()
        
        self.scene = env.scene
        
        # Play simulation briefly for initialization
        og.sim.play()
        for _ in range(10):
            og.sim.step()
        
        # Setup HDR background if specified
        if self.hdr_background_path is not None:
            self._setup_hdr_background()
        
        # Stop simulation so user can position objects without physics
        og.sim.stop()
        print("Simulation started in STOPPED state - position objects before playing")
        
        return env
    
    def _setup_hdr_background(self):
        """Setup HDR background texture for the skybox."""
        if not os.path.exists(self.hdr_background_path):
            print(f"Warning: HDR background path does not exist: {self.hdr_background_path}")
            return
        
        print(f"Setting up HDR background from: {self.hdr_background_path}")
        
        # Ensure a skybox exists (it may not if the loaded scene didn't have use_skybox=True)
        if og.sim.skybox is None:
            og.sim.add_skybox()
        
        # Set skybox texture
        og.sim.skybox.texture_file_path = self.hdr_background_path
        self._apply_skybox_lighting()
        
        # If we have a GS background, configure floor as a shadow catcher.
        if self.gs_background_path is not None:
            self._configure_floor_as_gs_shadow_catcher()
        
        print("HDR background setup complete")

    def _configure_floor_as_gs_shadow_catcher(self):
        """
        Configure floor plane geometry as matte shadow catcher for GS compositing.

        Returns:
            Sdf.Path | None: Proxy target prim path for the gauss proxy relationship.
        """
        if og.sim.floor_plane is None:
            return None

        floor_prim = og.sim.floor_plane.prim
        gprims = []
        for prim in lazy.pxr.Usd.PrimRange(floor_prim):
            if prim.IsA(lazy.pxr.UsdGeom.Gprim):
                gprims.append(prim)

        if not gprims:
            return None

        matte_attr_name = "primvars:isMatteObject"
        for gprim in gprims:
            if not gprim.HasProperty(matte_attr_name):
                gprim.CreateAttribute(matte_attr_name, lazy.pxr.Sdf.ValueTypeNames.Bool)
            gprim.GetProperty(matte_attr_name).Set(True)

        # Ensure parent is visible; matte should suppress direct surface visibility while keeping shadows.
        vis_attr = floor_prim.GetAttribute("visibility")
        if vis_attr is not None:
            vis_attr.Set(lazy.pxr.UsdGeom.Tokens.inherited)

        return gprims[0].GetPrimPath()

    def _has_skybox(self):
        return hasattr(og.sim, "skybox") and og.sim.skybox is not None

    def _apply_skybox_lighting(self):
        """Apply current skybox yaw / intensity settings."""
        if not self._has_skybox():
            return False
        orientation = T.euler2quat(
            th.tensor([self.skybox_rot_x, self.skybox_rot_y, self.skybox_rot_z], dtype=th.float32)
        )
        og.sim.skybox.set_position_orientation(orientation=orientation)
        og.sim.skybox.color = [1.0, 1.0, 1.0]
        og.sim.skybox.intensity = self.skybox_intensity
        og.sim.render()
        return True

    def _rotate_skybox_axis(self, axis, delta):
        """Rotate skybox around one Euler axis: x / y / z."""
        if not self._has_skybox():
            print("No skybox available. Enable --hdr_background or scene skybox first.")
            return
        if axis == "x":
            self.skybox_rot_x = (self.skybox_rot_x + delta) % (2.0 * math.pi)
        elif axis == "y":
            self.skybox_rot_y = (self.skybox_rot_y + delta) % (2.0 * math.pi)
        elif axis == "z":
            self.skybox_rot_z = (self.skybox_rot_z + delta) % (2.0 * math.pi)
        else:
            print(f"Unknown skybox axis '{axis}'")
            return
        self._apply_skybox_lighting()
        print(
            "Skybox rot (deg): "
            f"x={self.skybox_rot_x * 180.0 / math.pi:.1f}, "
            f"y={self.skybox_rot_y * 180.0 / math.pi:.1f}, "
            f"z={self.skybox_rot_z * 180.0 / math.pi:.1f}"
        )

    def increase_skybox_intensity(self):
        """Increase skybox intensity multiplicatively."""
        if not self._has_skybox():
            print("No skybox available. Enable --hdr_background or scene skybox first.")
            return
        self.skybox_intensity *= self.skybox_intensity_scale
        self._apply_skybox_lighting()
        print(f"Skybox intensity: {self.skybox_intensity:.2f}")

    def decrease_skybox_intensity(self):
        """Decrease skybox intensity multiplicatively."""
        if not self._has_skybox():
            print("No skybox available. Enable --hdr_background or scene skybox first.")
            return
        self.skybox_intensity /= self.skybox_intensity_scale
        self._apply_skybox_lighting()
        print(f"Skybox intensity: {self.skybox_intensity:.2f}")
    
    def load_from_scene_json(self):
        """
        Load a scene from a pre-saved JSON file and populate object tracking.
        
        Returns:
            bool: True if scene was loaded successfully, False otherwise
        """
        if self.load_scene_json is None:
            return False
        
        if not os.path.exists(self.load_scene_json):
            print(f"Warning: Scene JSON file does not exist: {self.load_scene_json}")
            return False
        
        print(f"Loading scene from: {self.load_scene_json}")

        # Load json as dict
        scene_json_dict = load_json_with_absolute_usd_paths(self.load_scene_json)
        saved_viewer_camera_state = scene_json_dict.get("viewer_camera_state")
        saved_lighting_state = scene_json_dict.get("lighting_state")
        saved_mesh_background_state = scene_json_dict.get("mesh_background_state")
        self._restore_lighting_state(saved_lighting_state)
        if self.mesh_background_path is None and saved_mesh_background_state is not None:
            saved_mesh_path = saved_mesh_background_state.get("usd_path")
            if saved_mesh_path:
                saved_mesh_path = Path(saved_mesh_path)
                if not saved_mesh_path.is_absolute():
                    saved_mesh_path = (Path(self.load_scene_json).parent / saved_mesh_path).resolve()
                self.mesh_background_path = str(saved_mesh_path)
                self.mesh_background_state_to_restore = dict(saved_mesh_background_state)
                self.mesh_background_state_to_restore["usd_path"] = self.mesh_background_path
                print(f"Will restore mesh background from scene JSON metadata: {self.mesh_background_path}")
        
        # If a new gs_background is specified, prune the existing one from the JSON
        # so we don't need to remove it at runtime
        if self.gs_background_path is not None:
            if "objects_info" in scene_json_dict and "init_info" in scene_json_dict["objects_info"]:
                if "gs_background" in scene_json_dict["objects_info"]["init_info"]:
                    print("Pruning existing gs_background from loaded scene JSON (will be replaced with new one)")
                    del scene_json_dict["objects_info"]["init_info"]["gs_background"]
                # Also prune any mesh_background since the GS background replaces it
                mesh_bg_keys = [k for k in scene_json_dict["objects_info"]["init_info"] if "mesh_background" in k]
                for k in mesh_bg_keys:
                    print(f"Pruning '{k}' from loaded scene JSON (replaced by GS background)")
                    del scene_json_dict["objects_info"]["init_info"][k]
                    # Also remove from state registry if present
                    obj_reg = scene_json_dict.get("state", {}).get("registry", {}).get("object_registry", {})
                    if k in obj_reg:
                        del obj_reg[k]
        
        # If a new mesh_background is specified, prune the existing one from the JSON
        # so we don't need to remove it at runtime
        if self.mesh_background_path is not None:
            if "objects_info" in scene_json_dict and "init_info" in scene_json_dict["objects_info"]:
                if "mesh_background" in scene_json_dict["objects_info"]["init_info"]:
                    print("Pruning existing mesh_background from loaded scene JSON (will be replaced with new one)")
                    del scene_json_dict["objects_info"]["init_info"]["mesh_background"]
        
        # Write the modified dict to a temp JSON file and pass the file path
        # to og.sim.restore. Passing the dict directly causes a circular reference
        # because restore() sets init_info["args"]["scene_file"] = scene_file,
        # which points back to the same dict that contains init_info.
        import tempfile
        tmp_scene_fd, tmp_scene_path = tempfile.mkstemp(suffix=".json", prefix="scene_")
        os.close(tmp_scene_fd)
        with open(tmp_scene_path, "w") as f:
            json.dump(scene_json_dict, f, indent=4)
        
        # Use og.sim.restore to load the scene
        og.launch()
        og.sim.restore(scene_files=[tmp_scene_path])
        
        # Get the loaded scene
        self.scene = og.sim.scenes[0]
        
        # Populate our object tracking from the loaded scene
        for obj in self.scene.objects:
            obj_name = obj.name
            self.objects[obj_name] = obj
            self.object_names.append(obj_name)
            
            # Store initial pose and scale
            pos, ori = obj.get_position_orientation()
            self.initial_poses[obj_name] = (pos.clone(), ori.clone())
            self.initial_scales[obj_name] = obj.scale.clone()
            
            # Check if this is a robot
            if obj_name.startswith("robot") and self.robot is None:
                self.robot = obj
                print(f"Found robot: {obj_name}")
                # Make sure controller config is unified
                self.robot.reload_controllers(self.get_controller_config())
            if obj_name == "gs_background" or "gs_background" in obj_name:
                self.gs_background = obj
            if obj_name == "mesh_background" or "mesh_background" in obj_name:
                self.mesh_background = obj
        
        print(f"Loaded {len(self.object_names)} objects from scene")
        
        # Step simulation briefly for initialization
        og.sim.play()
        for _ in range(10):
            og.sim.step()
        
        # Setup / restore skybox lighting while sim is playing
        # (og.sim.skybox can be None after og.sim.stop())
        if self.hdr_background_path is not None:
            self._setup_hdr_background()
        elif saved_lighting_state is not None:
            self._apply_skybox_lighting()
        elif self.gs_background_path is not None and not self._has_skybox():
            # GS background needs ambient lighting from the skybox to composite correctly.
            # The scene JSON may have been saved with use_skybox=False (e.g. stage 13 runs
            # with include_gs=True disable the skybox). Force-add one so NuRec has a
            # lit scene to composite against — without it GS renders at ~4% brightness.
            print("[load_scene] Adding skybox for GS ambient lighting (scene JSON had use_skybox=False)")
            og.sim.add_skybox()
            self._apply_skybox_lighting()

        # Stop simulation so user can position objects without physics
        og.sim.stop()
        print("Simulation started in STOPPED state - position objects before playing")
        
        # Apply ground plane position if saved in scene JSON
        if "ground_plane_info" in scene_json_dict and og.sim.floor_plane is not None:
            ground_plane_info = scene_json_dict["ground_plane_info"]
            floor_pos = th.tensor(ground_plane_info["position"], dtype=th.float32)
            floor_ori = th.tensor(ground_plane_info["orientation"], dtype=th.float32)
            og.sim.floor_plane.set_position_orientation(position=floor_pos, orientation=floor_ori)
            print(f"Applied ground plane position from scene JSON: z={floor_pos[2]:.4f}m")
        
        # Setup HDR background if specified (even when loading from saved scene)
        if self.hdr_background_path is not None:
            self._setup_hdr_background()
        elif self.gs_background_path is not None and not self._has_skybox():
            # og.sim.skybox can become None after og.sim.stop() — re-add the skybox
            # so ambient lighting is available for the NuRec GS warmup renders.
            og.sim.add_skybox()
            self._apply_skybox_lighting()
        # Restore viewer camera state from scene JSON if present.
        self._restore_viewer_camera_state(saved_viewer_camera_state)
        
        return True
    
    def load_background(self):
        """Load the 3DGS background if specified."""
        # If GS background already exists (e.g. loaded by --load_scene), reuse it.
        if self.gs_background is None:
            self.gs_background = self.objects.get("gs_background")
        if self.gs_background is None:
            for name, obj in self.objects.items():
                if "gs_background" in name:
                    self.gs_background = obj
                    break
        if self.gs_background is not None:
            if hasattr(self.gs_background, "usd_path") and self.gs_background.usd_path is not None:
                self.gs_background_path = self.gs_background.usd_path
            return

        if self.gs_background_path is None:
            return
            
        if not os.path.exists(self.gs_background_path):
            print(f"Warning: Background path does not exist: {self.gs_background_path}")
            return
        
        # Track original path for copying during save
        self.usd_object_paths["gs_background"] = self.gs_background_path
            
        self.gs_background = USDObject(
            name="gs_background",
            usd_path=self.gs_background_path,
            fixed_base=True,
            visual_only=True,
        )
        
        self.scene.add_object(self.gs_background)
        og.sim.step()

        # Position background using camera transform if provided
        if self.cam2world_tf is not None:
            self.gs_background.set_position_orientation(*T.mat2pose(self.cam2world_tf))

        og.sim.step()

        # Add background to manipulable objects list
        bg_name = "gs_background"
        self.objects[bg_name] = self.gs_background
        self.object_names.append(bg_name)

        # Store initial pose and scale
        pos, ori = self.gs_background.get_position_orientation()
        self.initial_poses[bg_name] = (pos.clone(), ori.clone())
        self.initial_scales[bg_name] = self.gs_background.scale.clone()

        # Set up proxy BEFORE the first render so the NuRec compositor initializes
        # with the proxy relationship already authored — setting it AFTER the first
        # render triggers a Hydra resync of the Volume prim that kills the compositor.
        floor_proxy_path = self._configure_floor_as_gs_shadow_catcher()
        if floor_proxy_path is not None:
            gauss = self.gs_background.root_link.prim.GetChildren()[0]
            proxy_rel = gauss.GetRelationship("proxy")
            if not proxy_rel or not proxy_rel.IsValid():
                proxy_rel = gauss.CreateRelationship("proxy")
            proxy_rel.SetTargets([floor_proxy_path])
        else:
            print("Warning: Failed to configure floor shadow catcher proxy for GS.")

        og.sim.render()

        print(f"Loaded 3DGS background from: {self.gs_background_path}")
    
    def load_mesh_background(self):
        """Load the mesh background if specified. This background is not copied on save."""
        # If mesh background already exists (e.g. loaded by --load_scene), reuse and optionally restore metadata pose.
        if "mesh_background" in self.objects:
            self.mesh_background = self.objects["mesh_background"]
            if hasattr(self.mesh_background, "usd_path") and self.mesh_background.usd_path is not None:
                self.mesh_background_path = self.mesh_background.usd_path
            self._apply_pending_mesh_background_state()
            return

        if self.mesh_background_path is None:
            return
            
        if not os.path.exists(self.mesh_background_path):
            print(f"Warning: Mesh background path does not exist: {self.mesh_background_path}")
            return
        
        # NOTE: We intentionally do NOT track mesh_background in usd_object_paths
        # so that it is not copied during save
        
        self.mesh_background = USDObject(
            name="mesh_background",
            usd_path=self.mesh_background_path,
            fixed_base=True,
            visual_only=True,
        )

        try:
            self.scene.add_object(self.mesh_background)
            og.sim.step()
        except AssertionError as e:
            # Some USD / USDZ scan meshes are scene references without OG-style rigid links.
            # Keep the original USDObject path first, then gracefully fall back for mesh-only assets.
            if "Exactly one single root link should have been found" not in str(e):
                raise

            print(
                "Warning: mesh_background is not a link-structured USDObject. "
                "Falling back to stage Xform reference loading."
            )

            # Clean up partially loaded prim from failed USDObject path, if any.
            failed_bg = self.mesh_background
            self.mesh_background = None
            try:
                if failed_bg is not None and getattr(failed_bg, "loaded", False):
                    failed_bg.remove()
            except Exception as cleanup_err:
                print(f"Warning: Failed to cleanup partially loaded mesh_background: {cleanup_err}")

            self.mesh_background = XFormPrim(
                relative_prim_path="/mesh_background_ref",
                name="mesh_background",
            )
            self.mesh_background.load(self.scene)
            added_ref = self.mesh_background.prim.GetReferences().AddReference(self.mesh_background_path)
            if not added_ref:
                raise RuntimeError(f"Failed to add USD reference for mesh background: {self.mesh_background_path}")
            og.sim.step()
        
        # Position background using camera transform if provided
        if self.cam2world_tf is not None:
            self.mesh_background.set_position_orientation(*T.mat2pose(self.cam2world_tf))

        # Saved scene metadata should override default placement if available.
        self._apply_pending_mesh_background_state()
        
        og.sim.step()
        og.sim.render()
        
        # Add background to manipulable objects list
        bg_name = "mesh_background"
        self.objects[bg_name] = self.mesh_background
        self.object_names.append(bg_name)
        
        # Store initial pose and scale
        pos, ori = self.mesh_background.get_position_orientation()
        self.initial_poses[bg_name] = (pos.clone(), ori.clone())
        self.initial_scales[bg_name] = self.mesh_background.scale.clone()
        
        print(f"Loaded mesh background from: {self.mesh_background_path}")

    def _apply_pending_mesh_background_state(self):
        """Apply mesh background pose / scale from saved JSON metadata if available."""
        if self.mesh_background is None or self.mesh_background_state_to_restore is None:
            return

        state = self.mesh_background_state_to_restore
        try:
            pos = state.get("position")
            ori = state.get("orientation")
            scale = state.get("scale")
            if pos is not None and ori is not None:
                self.mesh_background.set_position_orientation(
                    position=th.tensor(pos, dtype=th.float32),
                    orientation=th.tensor(ori, dtype=th.float32),
                )
            if scale is not None:
                self.mesh_background.scale = th.tensor(scale, dtype=th.float32)
            print("Restored mesh background pose/scale from saved scene JSON.")
        except Exception as e:
            print(f"Warning: Failed to restore mesh background pose/scale: {e}")
        finally:
            self.mesh_background_state_to_restore = None
    
    def load_objects(self):
        """Load all specified objects into the scene."""
        for idx, obj_path in enumerate(self.object_paths):
            if not os.path.exists(obj_path):
                print(f"Warning: Object path does not exist: {obj_path}")
                continue
            
            obj_name = f"object_{idx}"
            
            # Track original path for copying during save
            self.usd_object_paths[obj_name] = obj_path
            
            obj = USDObject(
                name=obj_name,
                usd_path=obj_path,
            )
            
            self.scene.add_object(obj)
            og.sim.step()
            
            # Set initial pose if provided
            if idx < len(self.object_poses) and self.object_poses[idx] is not None:
                pos, ori = self.object_poses[idx]
                obj.set_position_orientation(
                    position=th.tensor(pos, dtype=th.float32),
                    orientation=th.tensor(ori, dtype=th.float32)
                )
            
            self.objects[obj_name] = obj
            self.object_names.append(obj_name)
            
            # Store initial pose and scale
            pos, ori = obj.get_position_orientation()
            self.initial_poses[obj_name] = (pos.clone(), ori.clone())
            self.initial_scales[obj_name] = obj.scale.clone()
            
            print(f"Loaded object '{obj_name}' from: {obj_path}")
            og.sim.step()
        
        if self.object_names:
            print(f"\nTotal USD objects loaded: {len(self.object_names)}")
            print(f"Selected object: {self.object_names[self.selected_idx]}")
    
    def load_dataset_objects(self):
        """
        Load dataset objects (from behavior-1k-assets, real2sim-assets, etc.) as USDObjects.
        
        This extracts the USD path from the dataset and loads as USDObject for consistent 
        saving/copying behavior.
        """
        for idx, obj_spec in enumerate(self.dataset_objects):
            dataset_name = obj_spec.get("dataset_name")
            category = obj_spec.get("category")
            model = obj_spec.get("model")
            fixed_base = obj_spec.get("fixed_base", False)
            obj_name = obj_spec.get("name", f"{category}_{model}_{idx}")
            
            if not all([dataset_name, category, model]):
                print(f"Warning: Invalid dataset object spec: {obj_spec}")
                continue
            
            try:
                # Get the USD path using DatasetObject's class method
                usd_path = DatasetObject.get_usd_path(
                    category=category,
                    model=model,
                    dataset_name=dataset_name
                )
                
                if not os.path.exists(usd_path):
                    print(f"Warning: USD path does not exist: {usd_path}")
                    continue
                
                # Load as USDObject for consistent saving/copying
                obj = USDObject(
                    name=obj_name,
                    usd_path=usd_path,
                    category=category,
                    fixed_base=fixed_base,
                )
                
                self.scene.add_object(obj)
                og.sim.step()
                
                self.objects[obj_name] = obj
                self.object_names.append(obj_name)
                
                # Store initial pose and scale
                pos, ori = obj.get_position_orientation()
                self.initial_poses[obj_name] = (pos.clone(), ori.clone())
                self.initial_scales[obj_name] = obj.scale.clone()
                
                # Store original USD path for copying during save
                self.usd_object_paths[obj_name] = usd_path
                
                fixed_str = " (fixed_base)" if fixed_base else ""
                print(f"Loaded '{obj_name}': {dataset_name}/{category}/{model}{fixed_str}")
                og.sim.step()
                
            except Exception as e:
                print(f"Warning: Failed to load dataset object {obj_spec}: {e}")
        
        if self.dataset_objects:
            print(f"\nTotal dataset objects loaded: {len(self.dataset_objects)}")

    def load_usd_objects(self):
        """Load arbitrary USD objects from specified file paths."""
        for idx, obj_spec in enumerate(self.usd_objects):
            category = obj_spec.get("category")
            usd_path = obj_spec.get("usd_path")
            fixed_base = obj_spec.get("fixed_base", False)
            obj_name = obj_spec.get("name", f"{category}_{idx}")
            
            if not all([category, usd_path]):
                print(f"Warning: Invalid USD object spec: {obj_spec}")
                continue
            
            if not os.path.exists(usd_path):
                print(f"Warning: USD path does not exist: {usd_path}")
                continue
            
            try:
                obj = USDObject(
                    name=obj_name,
                    usd_path=usd_path,
                    category=category,
                    fixed_base=fixed_base,
                )
                
                self.scene.add_object(obj)
                og.sim.step()
                
                self.objects[obj_name] = obj
                self.object_names.append(obj_name)
                
                # Store initial pose and scale
                pos, ori = obj.get_position_orientation()
                self.initial_poses[obj_name] = (pos.clone(), ori.clone())
                self.initial_scales[obj_name] = obj.scale.clone()
                
                # Store original USD path for copying during save
                self.usd_object_paths[obj_name] = usd_path
                
                fixed_str = " (fixed_base)" if fixed_base else ""
                print(f"Loaded USD object '{obj_name}': {category} from {usd_path}{fixed_str}")
                og.sim.step()
                
            except Exception as e:
                print(f"Warning: Failed to load USD object {obj_spec}: {e}")
        
        if self.usd_objects:
            print(f"\nTotal USD objects loaded: {len(self.usd_objects)}")

    def load_scene_objects_info(self):
        """
        Load objects from scene_objects_info.json and pb_scene_poses.json files.
        
        This method loads objects using the same format as 13_create_og_scene.py,
        but creates USDObjects (not DatasetObjects) so they can be saved/copied properly.
        """
        if not self.scene_objects_info_path or not self.pb_scene_poses_path:
            return
        
        # Load scene objects info
        if not os.path.exists(self.scene_objects_info_path):
            print(f"Warning: scene_objects_info file not found: {self.scene_objects_info_path}")
            return
        
        with open(self.scene_objects_info_path, "r") as f:
            scene_objects_info = json.load(f)
        
        # Load pose info
        if not os.path.exists(self.pb_scene_poses_path):
            print(f"Warning: pb_scene_poses file not found: {self.pb_scene_poses_path}")
            return
        
        with open(self.pb_scene_poses_path, "r") as f:
            obj_poses = json.load(f)

        # Filter by categories if specified
        if self.scene_objects_categories:
            filtered_info = {
                k: v for k, v in scene_objects_info.items() 
                if v["category"] in self.scene_objects_categories
            }
            print(f"Filtering to categories: {self.scene_objects_categories}")
            print(f"Loading {len(filtered_info)} objects (filtered from {len(scene_objects_info)}) from scene_objects_info...")
            scene_objects_info = filtered_info
        else:
            print(f"Loading {len(scene_objects_info)} objects from scene_objects_info...")


        with og.sim.stopped():
            for idx, obj_info in reversed(list(scene_objects_info.items())):
                obj_category = obj_info["category"]
                obj_model = obj_info["model"]
                original_obj_name = obj_info["name"]
                obj_name = lazy.pxr.Tf.MakeValidIdentifier(f"{obj_category}_{obj_model}_{idx}")   # convert to valid string

                # Get the USD path using DatasetObject's method
                # Default to custom-assets dataset
                dataset_name = obj_info.get("dataset_name", "custom-assets")
                usd_path = DatasetObject.get_usd_path(
                    category=obj_category,
                    model=obj_model,
                    dataset_name=dataset_name
                )

                if not os.path.exists(usd_path):
                    print(f"Warning: USD path does not exist: {usd_path}")
                    continue

                try:
                    # Create as USDObject so it can be saved/copied properly
                    obj = USDObject(
                        name=obj_name,
                        usd_path=usd_path,
                        category=obj_category,
                    )

                    self.scene.add_object(obj)
                    og.sim.step()

                    # Get original pose from pb_scene_poses
                    if original_obj_name in obj_poses:
                        original_pos = th.tensor(obj_poses[original_obj_name][0])
                        original_ori = th.tensor(obj_poses[original_obj_name][1])
                        obj.set_position_orientation(position=original_pos, orientation=original_ori)
                        og.sim.step()

                    self.objects[obj_name] = obj
                    self.object_names.append(obj_name)

                    # Track this object as part of scene_objects_info group
                    self.scene_objects_info_names.append(obj_name)

                    # Store initial pose and scale
                    pos, ori = obj.get_position_orientation()
                    self.initial_poses[obj_name] = (pos.clone(), ori.clone())
                    self.initial_scales[obj_name] = obj.scale.clone()

                    # Store original USD path for copying during save
                    self.usd_object_paths[obj_name] = usd_path

                    print(f"Loaded '{obj_name}': {dataset_name}/{obj_category}/{obj_model}")

                except Exception as e:
                    print(f"Warning: Failed to load object {obj_name}: {e}")

        print(f"\nTotal objects loaded from scene_objects_info: {len(self.scene_objects_info_names)}")
        for i in range(200):
            og.sim.step()
            if (i + 1) % 20 == 0:
                print(f"  Settling step {i + 1}/{200}...")
        if self.object_names:
            print(f"Selected object: {self.object_names[self.selected_idx]}")

    def get_controller_config(self):
        arms = ["0"]        # TODO: Support bimanual robots
        controller_config = {}
        for arm in arms:
            if self.arm_controller == "joint_pos":
                controller_config[f"arm_{arm}"] = {
                    "name": "JointController",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                }
            else:  # Default to IK controller
                controller_config[f"arm_{arm}"] = {
                    "name": "InverseKinematicsController",
                    "command_input_limits": None,
                    "mode": "pose_delta_ori",
                    "smoothing_filter_size": 5,
                }
            controller_config[f"gripper_{arm}"] = {
                "name": "MultiFingerGripperController",
                "command_input_limits": (0.0, 1.0),
                "mode": "smooth",
            }
        return controller_config
    
    def load_robots(self):
        """Load robots into the scene based on robot_configs."""
        if not self.robot_configs:
            return
        
        # Default spacing between robots (in meters)
        robot_spacing = 1.0
        
        # Track total robot index for naming
        total_robot_idx = 0
        
        for config in self.robot_configs:
            robot_type = config.get("type", "FrankaPanda")
            count = config.get("count", 1)
            
            # Check if robot type exists
            if robot_type not in REGISTERED_ROBOTS:
                available = list(REGISTERED_ROBOTS.keys())
                print(f"Warning: Robot type '{robot_type}' not found. Available: {available}")
                continue
            
            robot_cls = REGISTERED_ROBOTS[robot_type]
            
            for i in range(count):
                robot_name = f"robot{total_robot_idx}"
                
                # Build robot kwargs from config
                robot_kwargs = {
                    "name": robot_name,
                    "self_collisions": False,
                    "obs_modalities": ["rgb", "depth_linear", "proprio"],
                    "action_normalize": False,
                    "grasping_mode": "assisted",
                }
                
                # Add end_effector if specified (for FrankaPanda)
                if config.get("end_effector"):
                    robot_kwargs["end_effector"] = config["end_effector"]
                
                # Add other optional parameters
                if "fixed_base" in config:
                    robot_kwargs["fixed_base"] = config["fixed_base"]
                if "grasping_mode" in config:
                    robot_kwargs["grasping_mode"] = config["grasping_mode"]

                robot_kwargs["controller_config"] = self.get_controller_config()
                
                # Create the robot
                robot = robot_cls(**robot_kwargs)
                
                # Determine position
                if "position" in config and i == 0:
                    # Use specified position for first robot
                    position = config["position"]
                else:
                    # Auto-space robots along Y axis
                    # Center them around origin based on count
                    offset = (total_robot_idx - (len(self.robots) + count - 1) / 2) * robot_spacing
                    position = [0, offset, 0]
                
                orientation = config.get("orientation", [0, 0, 0, 1])
                
                # Add to scene
                with og.sim.stopped():
                    self.scene.add_object(robot)
                    og.sim.step()
                    
                    robot.set_position_orientation(
                        position=th.tensor(position, dtype=th.float32),
                        orientation=th.tensor(orientation, dtype=th.float32)
                    )
                
                # Reset robot to default pose
                with og.sim.playing():
                    robot.reset()
                    robot.keep_still()
                
                # Track this robot
                self.robots.append(robot)
                if self.robot is None:
                    self.robot = robot  # First robot for backward compatibility
                
                # Also add robot to objects dict so it can be manipulated
                self.objects[robot_name] = robot
                self.object_names.append(robot_name)
                
                # Store initial pose and scale
                pos, ori = robot.get_position_orientation()
                self.initial_poses[robot_name] = (pos.clone(), ori.clone())
                self.initial_scales[robot_name] = robot.scale.clone()
                
                print(f"Loaded robot '{robot_name}' of type '{robot_type}'")
                if config.get("end_effector"):
                    print(f"  End effector: {config['end_effector']}")
                print(f"  Position: {position}")
                print(f"  Orientation: {orientation}")
                
                total_robot_idx += 1
        
        print(f"\nTotal robots loaded: {len(self.robots)}")
    
    def load_external_sensors(self):
        """
        Load external sensors from the external_sensors_config yaml file.
        
        Uses the same loading logic as OmniGibson's Environment._load_external_sensors.
        """
        if self.external_sensors_config is None:
            return
        
        if not os.path.exists(self.external_sensors_config):
            print(f"Warning: External sensors config file not found: {self.external_sensors_config}")
            return
        
        print(f"Loading external sensors from: {self.external_sensors_config}")
        
        # Parse the yaml config
        config = parse_config(self.external_sensors_config)
        sensors_config = config.get("external_sensors", [])
        
        if not sensors_config:
            print("Warning: No external_sensors found in config file")
            return
        
        for i, sensor_config in enumerate(sensors_config):
            # Add a name for the sensor if necessary
            if "name" not in sensor_config:
                sensor_config["name"] = f"external_sensor{i}"
            
            # Determine prim path if not specified
            if "relative_prim_path" not in sensor_config:
                sensor_config["relative_prim_path"] = f"/{sensor_config['name']}"
            
            # Make a copy to avoid modifying original
            sensor_config = deepcopy(sensor_config)
            
            # Pop position, orientation, and pose_frame
            position = sensor_config.pop("position", None)
            orientation = sensor_config.pop("orientation", None)
            pose_frame = sensor_config.pop("pose_frame", "scene")
            
            # Pop include_in_obs (not used here but needs to be removed)
            sensor_config.pop("include_in_obs", None)
            
            try:
                # Create the sensor
                sensor = create_sensor(**sensor_config)
                
                # Load and initialize the sensor
                sensor.load(self.scene)
                sensor.initialize()
                
                # Set position and orientation
                if position is not None or orientation is not None:
                    sensor.set_position_orientation(
                        position=position, 
                        orientation=orientation, 
                        frame=pose_frame
                    )
                
                # Store the sensor
                self.external_sensors[sensor.name] = sensor
                
                print(f"Loaded external sensor '{sensor.name}': {sensor_config.get('sensor_type', 'unknown')}")
                
            except Exception as e:
                print(f"Warning: Failed to load external sensor {sensor_config.get('name', i)}: {e}")
        
        print(f"\nTotal external sensors loaded: {len(self.external_sensors)}")
    
    def setup_keyboard_controls(self):
        """Setup keyboard event handlers for object manipulation."""
        KeyboardEventHandler.initialize()
        
        # Object selection
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.LEFT_BRACKET,
            callback_fn=self.cycle_object_forward
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.RIGHT_BRACKET,
            callback_fn=self.cycle_object_backward
        )
        
        # Translation controls (Arrow keys + Page Up/Down)
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.UP,
            callback_fn=lambda: self.translate_object(th.tensor([self.translation_delta, 0, 0]))
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.DOWN,
            callback_fn=lambda: self.translate_object(th.tensor([-self.translation_delta, 0, 0]))
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.LEFT,
            callback_fn=lambda: self.translate_object(th.tensor([0, self.translation_delta, 0]))
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.RIGHT,
            callback_fn=lambda: self.translate_object(th.tensor([0, -self.translation_delta, 0]))
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.W,
            callback_fn=lambda: self.translate_object(th.tensor([0, 0, self.translation_delta]))
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.Q,
            callback_fn=lambda: self.translate_object(th.tensor([0, 0, -self.translation_delta]))
        )
        
        # Rotation controls (N/M, Comma/Period, Slash)
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.N,
            callback_fn=lambda: self.rotate_object(th.tensor([self.rotation_delta, 0, 0]))
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.M,
            callback_fn=lambda: self.rotate_object(th.tensor([-self.rotation_delta, 0, 0]))
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.C,
            callback_fn=lambda: self.rotate_object(th.tensor([0, self.rotation_delta, 0]))
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.V,
            callback_fn=lambda: self.rotate_object(th.tensor([0, -self.rotation_delta, 0]))
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.SLASH,
            callback_fn=lambda: self.rotate_object(th.tensor([0, 0, self.rotation_delta]))
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.APOSTROPHE,
            callback_fn=lambda: self.rotate_object(th.tensor([0, 0, -self.rotation_delta]))
        )
        
        # Global Z rotation controls (Z/X keys)
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.Z,
            callback_fn=lambda: self.rotate_object_global_z(self.rotation_delta)
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.X,
            callback_fn=lambda: self.rotate_object_global_z(-self.rotation_delta)
        )
        
        # Global X rotation controls (1/2 keys)
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.KEY_1,
            callback_fn=lambda: self.rotate_object_global_x(self.rotation_delta)
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.KEY_2,
            callback_fn=lambda: self.rotate_object_global_x(-self.rotation_delta)
        )
        
        # Global Y rotation controls (3/4 keys)
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.KEY_3,
            callback_fn=lambda: self.rotate_object_global_y(self.rotation_delta)
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.KEY_4,
            callback_fn=lambda: self.rotate_object_global_y(-self.rotation_delta)
        )
        
        # Scale controls
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_ADD,
            callback_fn=lambda: self.scale_object(1.0 + self.scale_delta)
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.NUMPAD_SUBTRACT,
            callback_fn=lambda: self.scale_object(1.0 - self.scale_delta)
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.EQUAL,  # + key (Shift+=)
            callback_fn=lambda: self.scale_object(1.0 + self.scale_delta)
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.MINUS,
            callback_fn=lambda: self.scale_object(1.0 - self.scale_delta)
        )
        # KeyboardEventHandler.add_keyboard_callback(
        #     key=lazy.carb.input.KeyboardInput.PERIOD,
        #     callback_fn=lambda: self.scale_object(1.0 + self.scale_delta)
        # )
        # KeyboardEventHandler.add_keyboard_callback(
        #     key=lazy.carb.input.KeyboardInput.COMMA,
        #     callback_fn=lambda: self.scale_object(1.0 - self.scale_delta)
        # )
        
        # State management
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.ENTER,
            callback_fn=self.save_scene_state
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.BACKSPACE,
            callback_fn=self.reset_selected_object
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.F1,
            callback_fn=self.print_object_pose
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.U,
            callback_fn=self.undo
        )
        
        # Simulation control
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.SPACE,
            callback_fn=self.toggle_simulation
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.O,
            callback_fn=self.pause_simulation
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.P,
            callback_fn=self.play_simulation
        )
        
        # Delta adjustment controls (F2/F3 for translation, F4/F5 for rotation, F6/F7 for scale)
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.KEY_5,
            callback_fn=self.increase_translation_delta
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.KEY_6,
            callback_fn=self.decrease_translation_delta
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.KEY_7,
            callback_fn=self.increase_rotation_delta
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.KEY_8,
            callback_fn=self.decrease_rotation_delta
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.KEY_9,
            callback_fn=self.increase_scale_delta
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.KEY_0,
            callback_fn=self.decrease_scale_delta
        )

        # Skybox lighting controls
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.R,
            callback_fn=lambda: self._rotate_skybox_axis("x", self.skybox_rot_delta)
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.F,
            callback_fn=lambda: self._rotate_skybox_axis("x", -self.skybox_rot_delta)
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.Y,
            callback_fn=lambda: self._rotate_skybox_axis("y", self.skybox_rot_delta)
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.H,
            callback_fn=lambda: self._rotate_skybox_axis("y", -self.skybox_rot_delta)
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.J,
            callback_fn=lambda: self._rotate_skybox_axis("z", self.skybox_rot_delta)
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.L,
            callback_fn=lambda: self._rotate_skybox_axis("z", -self.skybox_rot_delta)
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.I,
            callback_fn=self.increase_skybox_intensity
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.K,
            callback_fn=self.decrease_skybox_intensity
        )
        
        # Debug - drop into IPython shell
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.B,
            callback_fn=self.debug_shell
        )
        
        # Group mode toggle (for scene_objects_info objects)
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.G,
            callback_fn=self.toggle_group_mode
        )
        
        # Delete selected object
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.D,
            callback_fn=self.delete_selected_object
        )
        
        # Set exact scale
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.S,
            callback_fn=self.set_object_scale
        )
        
        # Exit
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.ESCAPE,
            callback_fn=self.exit_editor
        )
    
    def get_selected_object(self):
        """Get the currently selected object."""
        if not self.object_names:
            return None
        return self.objects[self.object_names[self.selected_idx]]
    
    def cycle_object_forward(self):
        """Cycle to the next object."""
        if not self.object_names:
            print("No objects loaded.")
            return
        self.selected_idx = (self.selected_idx + 1) % len(self.object_names)
        print(f"Selected object: {self.object_names[self.selected_idx]}")
    
    def cycle_object_backward(self):
        """Cycle to the previous object."""
        if not self.object_names:
            print("No objects loaded.")
            return
        self.selected_idx = (self.selected_idx - 1) % len(self.object_names)
        print(f"Selected object: {self.object_names[self.selected_idx]}")
    
    def get_robot_base_rotation_matrix(self):
        """
        Get the rotation matrix of the robot's base frame.
        
        Returns:
            th.Tensor: 3x3 rotation matrix, or identity if no robot loaded
        """
        if self.robot is None:
            return th.eye(3)
        
        _, robot_ori = self.robot.get_position_orientation()
        return T.quat2mat(robot_ori)
    
    def _get_target_objects(self):
        """
        Get the list of objects to apply operations to.
        
        Returns a list of objects: if group mode is enabled and scene_objects_info
        objects exist, returns all those objects. Otherwise, returns just the selected object.
        """
        if self.group_mode and self.scene_objects_info_names:
            return [self.objects[name] for name in self.scene_objects_info_names if name in self.objects]
        else:
            obj = self.get_selected_object()
            return [obj] if obj is not None else []
    
    def _get_group_center(self, objects):
        """
        Compute the center of the bounding box encompassing all objects.
        
        Args:
            objects (list): List of objects to compute center for
            
        Returns:
            th.Tensor: 3D position of the group center
        """
        if not objects:
            return th.zeros(3)
        
        # Collect all object positions
        positions = th.stack([obj.get_position_orientation()[0] for obj in objects])
        
        # Compute the center as the mean of all positions
        # (This approximates the bounding box center)
        center = positions.mean(dim=0)
        return center
    
    def _push_undo(self, objects=None):
        """Snapshot the pose and scale of the given objects onto the undo stack.

        Args:
            objects: list of objects to snapshot, or None to use _get_target_objects().
                     Also captures the floor plane position if it exists.
        """
        if objects is None:
            objects = self._get_target_objects()
        if not objects:
            return

        snapshot = {"type": "transform", "states": {}}
        for obj in objects:
            pos, ori = obj.get_position_orientation()
            snapshot["states"][obj.name] = (pos.clone(), ori.clone(), obj.scale.clone())
        if og.sim.floor_plane is not None:
            fp_pos, fp_ori = og.sim.floor_plane.get_position_orientation()
            snapshot["states"]["__floor_plane__"] = (fp_pos.clone(), fp_ori.clone(), None)

        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

    def _push_undo_delete(self, obj_name):
        """Push a delete-specific undo entry that records which object was soft-deleted
        and where it was in the selection list."""
        idx = self.object_names.index(obj_name)
        snapshot = {"type": "delete", "obj_name": obj_name, "list_index": idx}
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

    def undo(self):
        """Restore the last saved state from the undo stack."""
        if not self.undo_stack:
            print("Nothing to undo.")
            return

        snapshot = self.undo_stack.pop()

        if snapshot["type"] == "delete":
            obj_name = snapshot["obj_name"]
            obj = self.objects.get(obj_name)
            if obj is not None:
                self._set_object_visibility(obj, True)
                self.soft_deleted.discard(obj_name)
                idx = min(snapshot["list_index"], len(self.object_names))
                self.object_names.insert(idx, obj_name)
                self.selected_idx = idx
                print(f"Undo delete: restored '{obj_name}'")
            return

        for name, (pos, ori, scale) in snapshot["states"].items():
            if name == "__floor_plane__":
                if og.sim.floor_plane is not None:
                    og.sim.floor_plane.set_position_orientation(position=pos, orientation=ori)
                continue
            obj = self.objects.get(name)
            if obj is None:
                continue
            obj.set_position_orientation(position=pos, orientation=ori)
            if scale is not None:
                obj.scale = scale
        print("Undo")

    def translate_object(self, delta):
        """
        Translate object(s) by delta relative to the robot's base frame.
        If no robot is loaded, uses world frame.
        If group mode is enabled, translates all scene_objects_info objects.
        
        When translating the mesh_background, the ground plane Z position is also
        adjusted by the same amount to keep them in sync.
        
        Args:
            delta (th.Tensor): 3D translation vector in robot base frame
        """
        target_objects = self._get_target_objects()
        if not target_objects:
            return
        self._push_undo(target_objects)
        
        # Transform delta from robot base frame to world frame
        robot_rot = self.get_robot_base_rotation_matrix()
        world_delta = robot_rot @ delta
        
        # Track if mesh_background is being moved
        moving_mesh_background = False
        
        for obj in target_objects:
            pos, ori = obj.get_position_orientation()
            new_pos = pos + world_delta
            obj.set_position_orientation(position=new_pos, orientation=ori)
            
            # Check if this is the mesh_background
            if obj.name == "mesh_background_0":
                moving_mesh_background = True
        
        # If mesh_background was moved, also move the ground plane by the same Z delta
        if moving_mesh_background and og.sim.floor_plane is not None:
            z_delta = world_delta[2].item() if hasattr(world_delta[2], 'item') else world_delta[2]
            if abs(z_delta) > 1e-6:  # Only move if there's meaningful Z change
                floor_pos, floor_ori = og.sim.floor_plane.get_position_orientation()
                new_floor_pos = floor_pos.clone()
                new_floor_pos[2] = floor_pos[2] + z_delta
                og.sim.floor_plane.set_position_orientation(position=new_floor_pos, orientation=floor_ori)
                print(f"Ground plane Z synced with mesh_background (delta: {z_delta:.4f}m)")
    
    def rotate_object(self, euler_delta):
        """
        Rotate object(s) by euler angles (in radians) relative to robot's base frame.
        If no robot is loaded, uses world frame for the rotation axes.
        If group mode is enabled, rotates the group around its collective center.
        
        Args:
            euler_delta (th.Tensor): 3D euler angle delta (roll, pitch, yaw) in robot base frame
        """
        target_objects = self._get_target_objects()
        if not target_objects:
            return
        self._push_undo(target_objects)
        
        # Get the robot's base rotation to define the rotation axes
        robot_rot = self.get_robot_base_rotation_matrix()
        
        # Create rotation in robot frame, then transform to world frame
        delta_mat_robot = T.euler2mat(euler_delta)
        # Transform: world_rotation = robot_rot @ delta_mat_robot @ robot_rot.T
        delta_mat_world = robot_rot @ delta_mat_robot @ robot_rot.T
        
        # In group mode, rotate around the group center
        if self.group_mode and len(target_objects) > 1:
            group_center = self._get_group_center(target_objects)
            
            for obj in target_objects:
                pos, ori = obj.get_position_orientation()
                
                # Rotate position around group center
                relative_pos = pos - group_center
                new_relative_pos = delta_mat_world @ relative_pos
                new_pos = group_center + new_relative_pos
                
                # Rotate orientation
                current_mat = T.quat2mat(ori)
                new_mat = delta_mat_world @ current_mat
                new_ori = T.mat2quat(new_mat)
                
                obj.set_position_orientation(position=new_pos, orientation=new_ori)
        else:
            for obj in target_objects:
                pos, ori = obj.get_position_orientation()
                
                # Apply the world-frame rotation to the object
                current_mat = T.quat2mat(ori)
                new_mat = delta_mat_world @ current_mat
                new_ori = T.mat2quat(new_mat)
                
                obj.set_position_orientation(position=pos, orientation=new_ori)
    
    def rotate_object_global_z(self, angle):
        """
        Rotate object(s) around the global Z axis.
        If group mode is enabled, rotates the group around its collective center.
        
        Args:
            angle (float): Rotation angle in radians
        """
        target_objects = self._get_target_objects()
        if not target_objects:
            return
        self._push_undo(target_objects)
        
        # Create rotation matrix around global Z axis
        global_z_rotation = T.euler2mat(th.tensor([0, 0, angle]))
        
        # In group mode, rotate around the group center
        if self.group_mode and len(target_objects) > 1:
            group_center = self._get_group_center(target_objects)
            
            for obj in target_objects:
                pos, ori = obj.get_position_orientation()
                
                # Rotate position around group center
                relative_pos = pos - group_center
                new_relative_pos = global_z_rotation @ relative_pos
                new_pos = group_center + new_relative_pos
                
                # Rotate orientation
                current_mat = T.quat2mat(ori)
                new_mat = global_z_rotation @ current_mat
                new_ori = T.mat2quat(new_mat)
                
                obj.set_position_orientation(position=new_pos, orientation=new_ori)
        else:
            for obj in target_objects:
                pos, ori = obj.get_position_orientation()
                
                # Apply global rotation: new_ori = global_rotation @ current_ori
                current_mat = T.quat2mat(ori)
                new_mat = global_z_rotation @ current_mat
                new_ori = T.mat2quat(new_mat)
                
                obj.set_position_orientation(position=pos, orientation=new_ori)
    
    def rotate_object_global_x(self, angle):
        """
        Rotate object(s) around the global X axis.
        If group mode is enabled, rotates the group around its collective center.
        
        Args:
            angle (float): Rotation angle in radians
        """
        target_objects = self._get_target_objects()
        if not target_objects:
            return
        self._push_undo(target_objects)
        
        # Create rotation matrix around global X axis
        global_x_rotation = T.euler2mat(th.tensor([angle, 0, 0]))
        
        # In group mode, rotate around the group center
        if self.group_mode and len(target_objects) > 1:
            group_center = self._get_group_center(target_objects)
            
            for obj in target_objects:
                pos, ori = obj.get_position_orientation()
                
                # Rotate position around group center
                relative_pos = pos - group_center
                new_relative_pos = global_x_rotation @ relative_pos
                new_pos = group_center + new_relative_pos
                
                # Rotate orientation
                current_mat = T.quat2mat(ori)
                new_mat = global_x_rotation @ current_mat
                new_ori = T.mat2quat(new_mat)
                
                obj.set_position_orientation(position=new_pos, orientation=new_ori)
        else:
            for obj in target_objects:
                pos, ori = obj.get_position_orientation()
                
                # Apply global rotation: new_ori = global_rotation @ current_ori
                current_mat = T.quat2mat(ori)
                new_mat = global_x_rotation @ current_mat
                new_ori = T.mat2quat(new_mat)
                
                obj.set_position_orientation(position=pos, orientation=new_ori)
    
    def rotate_object_global_y(self, angle):
        """
        Rotate object(s) around the global Y axis.
        If group mode is enabled, rotates the group around its collective center.
        
        Args:
            angle (float): Rotation angle in radians
        """
        target_objects = self._get_target_objects()
        if not target_objects:
            return
        self._push_undo(target_objects)
        
        # Create rotation matrix around global Y axis
        global_y_rotation = T.euler2mat(th.tensor([0, angle, 0]))
        
        # In group mode, rotate around the group center
        if self.group_mode and len(target_objects) > 1:
            group_center = self._get_group_center(target_objects)
            
            for obj in target_objects:
                pos, ori = obj.get_position_orientation()
                
                # Rotate position around group center
                relative_pos = pos - group_center
                new_relative_pos = global_y_rotation @ relative_pos
                new_pos = group_center + new_relative_pos
                
                # Rotate orientation
                current_mat = T.quat2mat(ori)
                new_mat = global_y_rotation @ current_mat
                new_ori = T.mat2quat(new_mat)
                
                obj.set_position_orientation(position=new_pos, orientation=new_ori)
        else:
            for obj in target_objects:
                pos, ori = obj.get_position_orientation()
                
                # Apply global rotation: new_ori = global_rotation @ current_ori
                current_mat = T.quat2mat(ori)
                new_mat = global_y_rotation @ current_mat
                new_ori = T.mat2quat(new_mat)
                
                obj.set_position_orientation(position=pos, orientation=new_ori)
    
    def scale_object(self, scale_factor):
        """
        Scale object(s) uniformly.
        If group mode is enabled, scales all scene_objects_info objects.
        
        Args:
            scale_factor (float): Multiplicative scale factor
        """
        target_objects = self._get_target_objects()
        if not target_objects:
            return
        self._push_undo(target_objects)
        
        for obj in target_objects:
            current_scale = obj.scale
            new_scale = current_scale * scale_factor
            obj.scale = new_scale
        
        if self.group_mode and len(target_objects) > 1:
            print(f"Scaled {len(target_objects)} objects by factor {scale_factor}")
        else:
            obj = self.get_selected_object()
            if obj is not None:
                print(f"Scaled {self.object_names[self.selected_idx]} to {obj.scale.tolist()}")
    
    def set_object_scale(self):
        """
        Prompt user for exact scale values and apply to the selected object.
        
        User enters 3 comma-separated numbers for X, Y, Z scale.
        """
        obj = self.get_selected_object()
        if obj is None:
            print("No object selected.")
            return
        
        obj_name = self.object_names[self.selected_idx]
        current_scale = obj.scale
        
        print(f"\nCurrent scale of '{obj_name}': {current_scale.tolist()}")
        user_input = input("Enter new scale (x,y,z): ").strip()
        
        if not user_input:
            print("Scale change cancelled.")
            return
        
        try:
            parts = [float(p.strip()) for p in user_input.split(",")]
            if len(parts) != 3:
                print("Error: Please enter exactly 3 comma-separated numbers (x,y,z).")
                return
            
            self._push_undo([obj])
            new_scale = th.tensor(parts, dtype=th.float32)
            obj.scale = new_scale
            print(f"Set scale of '{obj_name}' to {new_scale.tolist()}")
            
        except ValueError as e:
            print(f"Error parsing scale values: {e}")
    
    def reset_selected_object(self):
        """Reset the selected object to its initial pose and scale."""
        if not self.object_names:
            print("No objects loaded.")
            return
        
        obj_name = self.object_names[self.selected_idx]
        obj = self.objects[obj_name]
        
        self._push_undo([obj])
        pos, ori = self.initial_poses[obj_name]
        scale = self.initial_scales[obj_name]
        
        obj.set_position_orientation(position=pos.clone(), orientation=ori.clone())
        obj.scale = scale.clone()
        
        print(f"Reset {obj_name} to initial pose and scale")
    
    def _set_object_visibility(self, obj, visible):
        """Toggle USD visibility on an object's prim."""
        token = lazy.pxr.UsdGeom.Tokens.inherited if visible else lazy.pxr.UsdGeom.Tokens.invisible
        imageable = lazy.pxr.UsdGeom.Imageable(obj.prim)
        imageable.GetVisibilityAttr().Set(token)

    def delete_selected_object(self):
        """
        Soft-delete the currently selected object.

        The object is made invisible and removed from the selectable list,
        but stays in the scene so undo can restore it. On save, soft-deleted
        objects are stripped from the output JSON.
        """
        if not self.object_names:
            print("No objects loaded.")
            return
        
        obj_name = self.object_names[self.selected_idx]
        obj = self.objects.get(obj_name)
        
        if obj is None:
            print(f"Object {obj_name} not found.")
            return
        
        # Check if the object is a robot
        if hasattr(obj, 'is_robot') and obj.is_robot:
            print(f"Cannot delete robot: {obj_name}")
            return
        
        if obj_name in [r.name for r in self.robots]:
            print(f"Cannot delete robot: {obj_name}")
            return
        
        # Push undo snapshot before soft-deleting
        self._push_undo_delete(obj_name)
        
        # Hide the object
        self._set_object_visibility(obj, False)
        self.soft_deleted.add(obj_name)
        
        # Remove from selectable list (but keep in self.objects for undo)
        self.object_names.remove(obj_name)
        
        # Update selected index
        if self.selected_idx >= len(self.object_names):
            self.selected_idx = max(0, len(self.object_names) - 1)
        
        print(f"Deleted object: {obj_name} (press U to undo)")
        if self.object_names:
            print(f"Now selected: {self.object_names[self.selected_idx]}")
        else:
            print("No objects remaining.")
    
    def print_object_pose(self):
        """Print the current pose of the selected object."""
        obj = self.get_selected_object()
        if obj is None:
            print("No object selected.")
            return
        
        pos, ori = obj.get_position_orientation()
        scale = obj.scale
        
        print(f"\n{'='*50}")
        print(f"Object: {self.object_names[self.selected_idx]}")
        print(f"Position: {pos.tolist()}")
        print(f"Orientation (xyzw): {ori.tolist()}")
        print(f"Scale: {scale.tolist()}")
        print(f"{'='*50}\n")
    
    def _get_category_folder(self, usd_path):
        """
        Get the 3rd parent folder (category folder) from a USD path.
        
        For path like: "some/path/yellow_banana/sscmmv/usd/sscmmv.usd"
        Returns: Path to "some/path/yellow_banana"
        
        Args:
            usd_path (str): Path to the USD file
            
        Returns:
            Path: Path to the category folder (3rd parent)
        """
        path = Path(usd_path)
        # Go up 3 levels: file -> usd folder -> model folder -> category folder
        return path.parent.parent.parent
    
    def _copy_usd_object_assets(self, usd_paths=None):
        """
        Copy USD object assets to the scene output directory.
        
        Args:
            usd_paths (dict, optional): Mapping of object names to their original USD paths.
                If None, uses self.usd_object_paths.
        
        Returns:
            dict: Mapping of object names to their new USD paths
        """
        if usd_paths is None:
            usd_paths = self.usd_object_paths
            
        new_paths = {}
        
        for obj_name, original_path in usd_paths.items():
            original_path = Path(original_path)
            
            if obj_name == "gs_background":
                # For background, copy the .usdz file directly
                new_filename = f"{self.scene_name}_gs_background.usdz"
                new_path = self.output_dir / new_filename
                
                if not new_path.exists() or not new_path.samefile(original_path):
                    shutil.copy2(original_path, new_path)
                    print(f"Copied background to: {new_path}")
                
                new_paths[obj_name] = str(new_path)
            else:
                # For other USD objects, copy the 3rd parent folder (category folder)
                category_folder = self._get_category_folder(original_path)
                category_name = category_folder.name
                
                # Destination is objects_dir / category_name
                dest_folder = self.objects_dir / category_name
                
                if not dest_folder.exists():
                    shutil.copytree(category_folder, dest_folder)
                    print(f"Copied object folder '{category_name}' to: {dest_folder}")
                
                # Compute new USD path relative to copied location
                # Original: category/model/usd/file.usd
                # New: objects_dir/category/model/usd/file.usd
                relative_path = original_path.relative_to(category_folder)
                new_path = dest_folder / relative_path
                
                new_paths[obj_name] = str(new_path)
        
        return new_paths
    
    def _update_scene_json_paths(self, json_path, new_usd_paths):
        """
        Update the scene JSON to use the new copied USD paths and convert all USD paths to absolute.
        
        Args:
            json_path (Path): Path to the scene JSON file
            new_usd_paths (dict): Mapping of object names to their new USD paths
        """
        json_path = Path(json_path)
        json_dir = json_path.parent
        
        with open(json_path, "r") as f:
            scene_data = json.load(f)
        
        # Update USD paths in the init_info for each object
        if "objects_info" in scene_data and "init_info" in scene_data["objects_info"]:
            for obj_name, obj_info in scene_data["objects_info"]["init_info"].items():
                if "args" in obj_info and "usd_path" in obj_info["args"]:
                    # First, update with new path if available
                    if obj_name in new_usd_paths:
                        obj_info["args"]["usd_path"] = new_usd_paths[obj_name]
                    
                    # Convert to absolute path
                    usd_path = obj_info["args"]["usd_path"]
                    if usd_path:
                        usd_path = Path(usd_path)
                        if usd_path.is_absolute():
                            obj_info["args"]["usd_path"] = str(usd_path)
                        else:
                            obj_info["args"]["usd_path"] = str((json_dir / usd_path).resolve())
        
        # Write back the updated JSON
        with open(json_path, "w") as f:
            json.dump(scene_data, f, indent=2)

    def _get_viewer_camera_state(self):
        """
        Get current viewer camera state for persistence.

        Returns:
            dict | None: Camera state with position / orientation / cam2world_tf, or None if unavailable.
        """
        if not hasattr(og.sim, "viewer_camera") or og.sim.viewer_camera is None:
            return None

        try:
            cam_pos, cam_ori = og.sim.viewer_camera.get_position_orientation()
        except Exception as e:
            print(f"Warning: Failed to read viewer camera pose: {e}")
            return None

        camera_state = {
            "position": cam_pos.detach().cpu().tolist() if th.is_tensor(cam_pos) else list(cam_pos),
            "orientation": cam_ori.detach().cpu().tolist() if th.is_tensor(cam_ori) else list(cam_ori),
        }

        try:
            cam2world_tf = T.pose2mat((cam_pos, cam_ori))
            camera_state["cam2world_tf"] = (
                cam2world_tf.detach().cpu().tolist() if th.is_tensor(cam2world_tf) else np.asarray(cam2world_tf).tolist()
            )
        except Exception:
            # Position + quaternion are enough to restore the viewer camera.
            pass

        return camera_state

    def _write_viewer_camera_state_to_scene_json(self, json_path):
        """
        Write current viewer camera state into saved scene JSON.

        Args:
            json_path (Path | str): Path to scene state JSON.
        """
        camera_state = self._get_viewer_camera_state()
        if camera_state is None:
            print("Warning: Viewer camera state unavailable; skipping camera save.")
            return

        with open(json_path, "r") as f:
            scene_data = json.load(f)

        scene_data["viewer_camera_state"] = camera_state

        with open(json_path, "w") as f:
            json.dump(scene_data, f, indent=2)

    def _get_lighting_state(self):
        """
        Get current lighting state for persistence.

        Returns:
            dict: Lighting state including skybox controls and HDR texture path.
        """
        hdr_path = self.hdr_background_path
        if hdr_path is not None:
            hdr_path = str(Path(hdr_path).resolve())

        return {
            "hdr_background_path": hdr_path,
            "skybox_rot_x_deg": self.skybox_rot_x * 180.0 / math.pi,
            "skybox_rot_y_deg": self.skybox_rot_y * 180.0 / math.pi,
            "skybox_rot_z_deg": self.skybox_rot_z * 180.0 / math.pi,
            "skybox_intensity": float(self.skybox_intensity),
            "skybox_rot_step_deg": self.skybox_rot_delta * 180.0 / math.pi,
            "skybox_intensity_scale": float(self.skybox_intensity_scale),
            # Backward-compatible aliases for older JSON readers
            "skybox_yaw_deg": self.skybox_rot_z * 180.0 / math.pi,
            "skybox_yaw_step_deg": self.skybox_rot_delta * 180.0 / math.pi,
        }

    def _write_lighting_state_to_scene_json(self, json_path):
        """
        Write current lighting state into saved scene JSON.

        Args:
            json_path (Path | str): Path to scene state JSON.
        """
        with open(json_path, "r") as f:
            scene_data = json.load(f)

        scene_data["lighting_state"] = self._get_lighting_state()

        with open(json_path, "w") as f:
            json.dump(scene_data, f, indent=2)

    def _get_mesh_background_state(self):
        """
        Get current mesh background state for persistence.

        Returns:
            dict | None: Mesh background state with usd_path / pose / scale, or None if unavailable.
        """
        if self.mesh_background is None:
            return None

        try:
            pos, ori = self.mesh_background.get_position_orientation()
            scale = self.mesh_background.scale
        except Exception as e:
            print(f"Warning: Failed to read mesh background pose/scale: {e}")
            return None

        mesh_path = None
        if hasattr(self.mesh_background, "usd_path") and self.mesh_background.usd_path is not None:
            mesh_path = str(Path(self.mesh_background.usd_path).resolve())
        elif self.mesh_background_path is not None:
            mesh_path = str(Path(self.mesh_background_path).resolve())

        return {
            "usd_path": mesh_path,
            "position": pos.detach().cpu().tolist() if th.is_tensor(pos) else list(pos),
            "orientation": ori.detach().cpu().tolist() if th.is_tensor(ori) else list(ori),
            "scale": scale.detach().cpu().tolist() if th.is_tensor(scale) else list(scale),
        }

    def _write_mesh_background_state_to_scene_json(self, json_path):
        """
        Write current mesh background state into saved scene JSON.

        Args:
            json_path (Path | str): Path to scene state JSON.
        """
        with open(json_path, "r") as f:
            scene_data = json.load(f)

        mesh_bg_state = self._get_mesh_background_state()
        if mesh_bg_state is None:
            scene_data.pop("mesh_background_state", None)
        else:
            scene_data["mesh_background_state"] = mesh_bg_state

        with open(json_path, "w") as f:
            json.dump(scene_data, f, indent=2)

    def _restore_lighting_state(self, lighting_state):
        """
        Restore lighting parameters from saved JSON data.

        Args:
            lighting_state (dict | None): Saved lighting state.
        """
        if not lighting_state:
            return

        try:
            if "skybox_rot_x_deg" in lighting_state:
                self.skybox_rot_x = float(lighting_state["skybox_rot_x_deg"]) * (math.pi / 180.0)
            if "skybox_rot_y_deg" in lighting_state:
                self.skybox_rot_y = float(lighting_state["skybox_rot_y_deg"]) * (math.pi / 180.0)
            if "skybox_rot_z_deg" in lighting_state:
                self.skybox_rot_z = float(lighting_state["skybox_rot_z_deg"]) * (math.pi / 180.0)
            elif "skybox_yaw_deg" in lighting_state:
                self.skybox_rot_z = float(lighting_state["skybox_yaw_deg"]) * (math.pi / 180.0)
            if "skybox_intensity" in lighting_state:
                self.skybox_intensity = max(0.0, float(lighting_state["skybox_intensity"]))
            if "skybox_rot_step_deg" in lighting_state:
                self.skybox_rot_delta = max(0.1, float(lighting_state["skybox_rot_step_deg"])) * (math.pi / 180.0)
            elif "skybox_yaw_step_deg" in lighting_state:
                self.skybox_rot_delta = max(0.1, float(lighting_state["skybox_yaw_step_deg"])) * (math.pi / 180.0)
            if "skybox_intensity_scale" in lighting_state:
                self.skybox_intensity_scale = max(1.01, float(lighting_state["skybox_intensity_scale"]))

            saved_hdr_path = lighting_state.get("hdr_background_path")
            if self.hdr_background_path is None and saved_hdr_path:
                self.hdr_background_path = str(saved_hdr_path)

            print("Restored lighting settings from saved scene JSON.")
        except Exception as e:
            print(f"Warning: Failed to restore lighting settings: {e}")

    def _restore_viewer_camera_state(self, camera_state):
        """
        Restore viewer camera state from saved JSON data.

        Args:
            camera_state (dict | None): Saved camera state.
        """
        if not camera_state:
            return
        if not hasattr(og.sim, "viewer_camera") or og.sim.viewer_camera is None:
            print("Warning: Viewer camera unavailable; cannot restore saved camera state.")
            return
        if "position" not in camera_state or "orientation" not in camera_state:
            print("Warning: Invalid viewer_camera_state in scene JSON; skipping camera restore.")
            return

        try:
            cam_pos = th.tensor(camera_state["position"], dtype=th.float32)
            cam_ori = th.tensor(camera_state["orientation"], dtype=th.float32)
            og.sim.viewer_camera.set_position_orientation(cam_pos, cam_ori)
            og.sim.render()
            print("Restored viewer camera pose from saved scene JSON.")
        except Exception as e:
            print(f"Warning: Failed to restore viewer camera pose: {e}")
    
    def save_scene_state(self):
        """
        Save the current scene state to a JSON file.
        
        - Saves to <asset_dir>/scenes/<scene_name>/<scene_name>_scene_state_<timestamp>.json
        - Copies USD object folders to <asset_dir>/scenes/<scene_name>/objects/
        - Copies GS background to <asset_dir>/scenes/<scene_name>/<scene_name>_gs_background.usdz
        - Updates the scene JSON paths to point to copied locations
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"{self.scene_name}_scene_state_{timestamp}.json"
        json_latest_path = self.output_dir / f"{self.scene_name}_scene_state_latest.json"
        
        # Use og.sim.save() to save the scene
        self.scene.update_initial_file()
        og.sim.save(json_paths=[str(json_path)])
        self._strip_nested_init_info(json_path)
        self._strip_soft_deleted_objects(json_path)
        self._write_viewer_camera_state_to_scene_json(json_path)
        self._write_lighting_state_to_scene_json(json_path)
        self._write_mesh_background_state_to_scene_json(json_path)
        
        # Build a comprehensive list of all USD paths from scene objects
        # This includes objects loaded from --load_scene that might not be in self.usd_object_paths
        all_usd_paths = dict(self.usd_object_paths)  # Start with explicitly tracked paths
        
        # Remove any background objects from copying - they should not be copied
        # Check for names containing "gs_background" or "mesh_background"
        background_keys_to_remove = [
            key for key in all_usd_paths 
            if "gs_background" in key or "mesh_background" in key
        ]
        for key in background_keys_to_remove:
            del all_usd_paths[key]
        
        for obj in self.scene.objects:
            obj_name = obj.name
            # Skip if already tracked
            if obj_name in all_usd_paths:
                continue
            
            # Skip objects with gs_background or mesh_background in their name - don't copy them
            if "gs_background" in obj_name or "mesh_background" in obj_name:
                continue
            
            # Check if the object has a usd_path property (USDObject and subclasses)
            if hasattr(obj, 'usd_path') and obj.usd_path is not None:
                usd_path = obj.usd_path
                # Skip if it looks like a robot (robots have their own asset structure)
                if hasattr(obj, 'is_robot') and obj.is_robot:
                    continue
                # Skip if it's an internal OmniGibson asset (e.g., ground plane)
                if 'omnigibson' in str(usd_path).lower() and 'data' in str(usd_path).lower():
                    continue
                all_usd_paths[obj_name] = usd_path
                print(f"Found additional object to copy: {obj_name} -> {usd_path}")
        
        # Copy USD assets and get new paths
        print(f"all_usd_paths: {all_usd_paths}")
        if all_usd_paths:
            print("\nCopying USD assets...")
            new_usd_paths = self._copy_usd_object_assets(all_usd_paths)
            
            # Update the scene JSON with new paths
            self._update_scene_json_paths(json_path, new_usd_paths)
            print("Updated scene JSON with new asset paths.")
        
        # Save ground plane position to scene JSON if it exists
        if og.sim.floor_plane is not None:
            floor_pos, floor_ori = og.sim.floor_plane.get_position_orientation()
            with open(json_path, "r") as f:
                scene_data = json.load(f)
            
            # Add ground plane info to scene JSON
            scene_data["ground_plane_info"] = {
                "position": floor_pos.tolist(),
                "orientation": floor_ori.tolist(),
            }
            
            with open(json_path, "w") as f:
                json.dump(scene_data, f, indent=2)
            print(f"Saved ground plane position: z={floor_pos[2]:.4f}m")
        
        print(f"\n{'='*50}")
        print(f"Scene state saved to: {json_path}")
        # Copy to latest
        shutil.copy(json_path, json_latest_path)
        print(f"Copied scene state to latest: {json_latest_path}")
        print(f"Assets copied to: {self.output_dir}")
        print(f"{'='*50}\n")
    
    @staticmethod
    def _strip_nested_init_info(json_path):
        """Remove recursive init_info nesting from a saved scene JSON.

        og.sim.save() embeds the full previous scene file inside
        init_info.args.scene_file, which itself contains its own init_info
        from the save before that. This strips the nested init_info to
        prevent unbounded growth across successive saves.
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        scene_file = data.get("init_info", {}).get("args", {}).get("scene_file")
        if isinstance(scene_file, dict) and "init_info" in scene_file:
            del scene_file["init_info"]
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

    def _strip_soft_deleted_objects(self, json_path):
        """Remove soft-deleted objects from a saved scene JSON."""
        if not self.soft_deleted:
            return

        with open(json_path, "r") as f:
            data = json.load(f)

        obj_reg = data.get("state", {}).get("registry", {}).get("object_registry", {})
        init_info = data.get("objects_info", {}).get("init_info", {})

        for name in self.soft_deleted:
            if name in obj_reg:
                del obj_reg[name]
            if name in init_info:
                del init_info[name]

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Stripped {len(self.soft_deleted)} soft-deleted object(s) from saved JSON")

    def toggle_simulation(self):
        """Toggle simulation between play and stop states."""
        if og.sim.is_playing():
            og.sim.stop()
            print("Simulation STOPPED")
        else:
            # Update initial file before playing so physics starts from current state
            og.sim.step_physics()
            og.sim.play()
            self.scene.update_initial_file()
            print("Simulation PLAYING (initial state updated)")
    
    def pause_simulation(self):
        """Pause/stop the simulation."""
        if og.sim.is_playing():
            og.sim.stop()
            print("Simulation STOPPED")
    
    def play_simulation(self):
        """Play/resume the simulation."""
        if not og.sim.is_playing():
            # Update initial file before playing so physics starts from current state
            og.sim.step_physics()
            og.sim.play()
            self.scene.update_initial_file()
            print("Simulation PLAYING (initial state updated)")
    
    def increase_translation_delta(self):
        """Increase translation delta by 0.001m."""
        self.translation_delta += 0.001
        print(f"Translation delta: {self.translation_delta:.4f} m")
    
    def decrease_translation_delta(self):
        """Decrease translation delta by 0.001m (minimum 0.001m)."""
        self.translation_delta = max(0.001, self.translation_delta - 0.001)
        print(f"Translation delta: {self.translation_delta:.4f} m")
    
    def increase_rotation_delta(self):
        """Increase rotation delta by 0.5 degrees."""
        self.rotation_delta += 0.5 * (3.14159265359 / 180.0)  # 0.5 degrees in radians
        print(f"Rotation delta: {self.rotation_delta * 180.0 / 3.14159265359:.2f} deg")
    
    def decrease_rotation_delta(self):
        """Decrease rotation delta by 0.5 degrees (minimum 0.5 deg)."""
        min_delta = 0.5 * (3.14159265359 / 180.0)  # 0.5 degrees in radians
        self.rotation_delta = max(min_delta, self.rotation_delta - min_delta)
        print(f"Rotation delta: {self.rotation_delta * 180.0 / 3.14159265359:.2f} deg")
    
    def increase_scale_delta(self):
        """Increase scale delta by 0.01."""
        self.scale_delta += 0.01
        print(f"Scale delta: {self.scale_delta:.3f}")
    
    def decrease_scale_delta(self):
        """Decrease scale delta by 0.01 (minimum 0.01)."""
        self.scale_delta = max(0.01, self.scale_delta - 0.01)
        print(f"Scale delta: {self.scale_delta:.3f}")
    
    def toggle_group_mode(self):
        """
        Toggle group mode on/off.
        
        When group mode is enabled and scene_objects_info was used to load objects,
        translation/rotation/scale operations are applied to all objects in that group
        instead of just the selected object.
        """
        if not self.scene_objects_info_names:
            print("Group mode not available: No objects loaded from scene_objects_info")
            return
        
        self.group_mode = not self.group_mode
        status = "ENABLED" if self.group_mode else "DISABLED"
        print(f"Group mode {status} - Operations will apply to {len(self.scene_objects_info_names)} objects")
    
    def debug_shell(self):
        """Drop into an IPython shell for debugging with access to editor state."""
        print("\n" + "="*60)
        print("ENTERING DEBUG SHELL (IPython)")
        print("="*60)
        print("Available variables:")
        print("  self        - The InteractiveSceneEditor instance")
        print("  og          - OmniGibson module")
        print("  th          - PyTorch (torch)")
        print("  T           - Transform utils (omnigibson.utils.transform_utils)")
        print("  self.scene  - The current scene")
        print("  self.objects - Dict of all objects")
        print("  self.robots - List of loaded robots")
        print("  self.robot  - First robot (if any)")
        print("Type 'exit' or Ctrl+D to return to the editor")
        print("="*60 + "\n")
        
        from IPython import embed
        embed()
        
        print("\nReturning to interactive editor...")
    
    def exit_editor(self):
        """Clean up and exit the editor."""
        print("\nExiting interactive scene editor...")
        og.clear()
        og.shutdown()
    
    def print_controls(self):
        """Print the keyboard control reference."""
        print("\n" + "="*60)
        print("INTERACTIVE SCENE EDITOR - KEYBOARD CONTROLS")
        print("="*60)
        print("\nObject Selection (includes background if loaded):")
        print("  [           - Cycle through objects (forward)")
        print("  ]           - Cycle through objects (backward)")
        print("\nTranslation (relative to robot base frame, or world if no robot):")
        print("  UP/DOWN     - Move along X axis")
        print("  LEFT/RIGHT  - Move along Y axis")
        print("  W/Q         - Move along Z axis")
        print("\nRotation (relative to robot base frame, or world if no robot):")
        print("  N/M         - Rotate around X axis (pitch)")
        print("  C/V         - Rotate around Y axis (roll)")
        print("  / and '     - Rotate around Z axis (yaw)")
        print("\nRotation (global frame):")
        print("  1/2         - Rotate around global X axis")
        print("  3/4         - Rotate around global Y axis")
        print("  Z/X         - Rotate around global Z axis")
        print("\nScale:")
        print("  +/-         - Uniform scale up/down")
        print("\nState Management:")
        print("  ENTER       - Save scene state")
        print("  BACKSPACE   - Reset selected object to initial pose")
        print("  U           - Undo last operation")
        print("  F1          - Print current object pose")
        print("\nSimulation Control:")
        print("  SPACE       - Toggle simulation play/stop")
        print("  O           - Stop simulation")
        print("  P           - Play simulation")
        print("\nDelta Adjustment:")
        print("  KEY_5/KEY_6 - Increase/decrease translation delta (±0.001m)")
        print("  KEY_7/KEY_8 - Increase/decrease rotation delta (±0.5°)")
        print("  KEY_9/KEY_0 - Increase/decrease scale delta (±0.01)")
        print("\nLighting (skybox):")
        print("  R/F         - Rotate HDR light direction around X axis")
        print("  Y/H         - Rotate HDR light direction around Y axis")
        print("  J/L         - Rotate HDR light direction around Z axis")
        print("  I/K         - Increase/decrease skybox intensity")
        print("\nGroup Mode (for scene_objects_info objects):")
        print("  G           - Toggle group mode (apply operations to all scene_objects_info objects)")
        print("\nObject Management:")
        print("  D           - Delete selected object (cannot delete robots, stops sim)")
        print("  S           - Set exact scale (enter x,y,z values)")
        print("\nSystem:")
        print("  B           - Debug shell (IPython)")
        print("  ESC         - Exit")
        print("="*60 + "\n")
        print(f"Current deltas: translation={self.translation_delta:.4f}m, "
              f"rotation={self.rotation_delta * 180.0 / 3.14159265359:.2f}°, "
              f"scale={self.scale_delta:.3f}")
        if self._has_skybox():
            print(f"Skybox lighting: rot_x={self.skybox_rot_x * 180.0 / math.pi:.1f}°, "
                  f"rot_y={self.skybox_rot_y * 180.0 / math.pi:.1f}°, "
                  f"rot_z={self.skybox_rot_z * 180.0 / math.pi:.1f}°, "
                  f"intensity={self.skybox_intensity:.2f}, "
                  f"rot_step={self.skybox_rot_delta * 180.0 / math.pi:.1f}°")
        if self.scene_objects_info_names:
            group_status = "ENABLED" if self.group_mode else "DISABLED"
            print(f"Group mode: {group_status} ({len(self.scene_objects_info_names)} objects in group)")
    
    def run(self):
        """Main run loop for the interactive editor."""
        # Check if we're loading from a saved scene
        scene_loaded_from_json = False
        if self.load_scene_json is not None:
            print("Loading from saved scene JSON...")
            if self.load_from_scene_json():
                print("Scene loaded successfully from JSON.")
                scene_loaded_from_json = True
            else:
                print("Failed to load scene from JSON. Creating new scene...")
        
        if not scene_loaded_from_json:
            # Setup scene from scratch
            print("Setting up scene...")
            self.setup_scene()
            
        # Pre-configure NuRec render settings BEFORE loading the GS so the compositor
        # initializes into the correct pipeline state.
        # invertToneMap=True is the 3dgrut default: NuRec outputs in linear space, so
        # the compositor must invert the scene's tone-map before compositing and then
        # re-apply it.  With False, the linear GS is composited onto an already
        # tonemapped scene → ACES exposure adaptation crushes the image to near-black.
        has_gs_background = self.gs_background_path is not None or self.gs_background is not None
        if has_gs_background:
            print("Pre-configuring render pipeline for NuRec GS...")
            # Apply compositing settings BEFORE loading the GS so NuRec initializes
            # into the correct pipeline state.  Do NOT set rendermode here — NuRec
            # works in Real-Time mode and every set_setting call fires
            # renderSettingsChanged which restarts the NuRec compositor (causing a
            # ~20-frame blackout).  The USDA's customLayerData handles rendermode.
            for key, value in [
                ("/rtx/post/histogram/enabled", False),
                ("/rtx/post/registeredCompositing/invertToneMap", True),
                ("/rtx/post/registeredCompositing/invertColorCorrection", True),
                ("/rtx/matteObject/visibility/secondaryRays", True),
                ("/rtx/post/tonemap/op", 2),
                ("/rtx/material/enableRefraction", False),
                ("/rtx/raytracing/fractionalCutoutOpacity", False),
            ]:
                try:
                    og.app.set_setting(key, value)
                    print(f"  [pre-render] {key} = {value}")
                except Exception as e:
                    print(f"  [pre-render] failed {key}={value}: {e}")
            og.sim.render()
            og.sim.render()

        print("Loading background...")
        self.load_background()

        print("Loading mesh background...")
        self.load_mesh_background()

        print("Loading USD objects...")
        self.load_objects()

        print("Loading dataset objects...")
        self.load_dataset_objects()

        print("Loading USD objects...")
        self.load_usd_objects()

        print("Loading objects from scene_objects_info...")
        self.load_scene_objects_info()

        print("Loading robots...")
        self.load_robots()

        print("Loading external sensors...")
        self.load_external_sensors()

        # Apply floor-matte and proxy. With GS already loaded, has_gs=True so the
        # histogram/rendermode set_setting calls are suppressed automatically.
        print("Applying realistic render settings...")
        apply_realistic_render_settings(og, lazy)

        # No enforce block here: applying set_setting() AFTER the GS is loaded
        # triggers renderSettingsChanged → NuRec compositor restarts → 20-frame
        # blackout.  Pre-config (above) sets all compositing settings before GS
        # load so NuRec initializes correctly without any restart.

        # Screenshot helper for diagnostic warmup frames
        import pathlib as _pl
        _dbg = _pl.Path(__file__).resolve().parents[2] / "Data" / "output2" / "debug_editor"
        if has_gs_background and self.gs_background is not None:
            _dbg.mkdir(parents=True, exist_ok=True)
            def _snap(label):
                try:
                    import omni.renderer_capture as _rc
                    _rc.acquire_renderer_capture_interface().capture_next_frame_swapchain_to_file(
                        str(_dbg / f"{label}.png"))
                    og.sim.render()
                except Exception:
                    pass

            # Warm up long enough for NuRec to converge (~20 frames from load).
            # load_background() already gave NuRec a few frames; 35 frames here
            # ensures the GS is visible and stable before the user interacts.
            print("[nurec] Warming up NuRec compositor (35 frames)...")
            for i in range(1, 36):
                og.sim.render()
                if i in (10, 15, 20, 25, 30, 35):
                    _snap(f"warmup_{i:02d}")
            print("[nurec] Warmup complete.")
        else:
            _snap = None

        # Setup controls
        print("Setting up keyboard controls...")
        self.setup_keyboard_controls()
        if _snap is not None:
            _snap("after_keyboard_setup")
        
        # Enable camera teleoperation
        # og.sim.enable_viewer_camera_teleoperation()
        
        # Print controls
        self.print_controls()
        
        if self.object_names:
            print(f"Selected object: {self.object_names[self.selected_idx]}")
        else:
            print("No objects loaded. Add objects using --objects argument.")
        
        print("\nInteractive editor running. Use keyboard to manipulate objects.")
        print("Press ESC to exit.\n")
        
        # Main render loop.
        #
        # NuRec GS: do NOT call set_setting() after the GS is loaded.  Every
        # set_setting call fires renderSettingsChanged → NuRec restarts.  Settings
        # are locked in by the pre-config block above (applied before GS load).
        # render() calls app.update() so keyboard events are processed without step().
        try:
            while True:
                if og.sim.is_playing():
                    # Physics active: full step needed for robot/object dynamics.
                    if getattr(og.sim, "physics_sim_view", None) is None:
                        og.sim.update_handles()
                    og.sim.step()
                og.sim.render()
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            self.exit_editor()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive Scene Editor for OmniGibson",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--scene_name",
        type=str,
        required=True,
        help="Name of the scene. Used for output directory and file naming. "
             "Outputs will be saved to <asset_dir>/scenes/<scene_name>/"
    )
    
    parser.add_argument(
        "--asset_dir",
        type=str,
        default=None,
        help="Base assets directory. Defaults to <repo>/assets"
    )
    
    parser.add_argument(
        "--background",
        type=str,
        default=None,
        help="Path to 3DGS background USDZ file"
    )
    
    parser.add_argument(
        "--mesh_background",
        type=str,
        default=None,
        help="Path to mesh background USD file. Unlike 3DGS background, this is not copied on save."
    )
    
    parser.add_argument(
        "--hdr_background",
        type=str,
        default=None,
        help="Path to HDR background .exr file for skybox texture"
    )
    
    parser.add_argument(
        "--cam2world",
        type=str,
        default=None,
        help="Path to cam2world transform .npy file (for positioning background)"
    )
    
    parser.add_argument(
        "--objects",
        type=str,
        nargs="+",
        default=[],
        help="Paths to object USD files to load"
    )
    
    parser.add_argument(
        "--poses_file",
        type=str,
        default=None,
        help="Path to JSON file containing initial poses for objects"
    )
    
    parser.add_argument(
        "--translation_delta",
        type=float,
        default=0.01,
        help="Step size for translation (default: 0.01)"
    )
    
    parser.add_argument(
        "--rotation_delta",
        type=float,
        default=0.05,
        help="Step size for rotation in radians (default: 0.05)"
    )
    
    parser.add_argument(
        "--scale_delta",
        type=float,
        default=0.05,
        help="Step size for scale changes (default: 0.05)"
    )
    
    parser.add_argument(
        "--robot",
        type=str,
        nargs="+",
        default=[],
        help="Robot(s) to spawn. Format: 'ClassName', 'ClassName:end_effector', or 'ClassName:end_effector:count'. "
             "Use '::count' for robots without end_effector option. Can specify multiple. "
             "Examples: 'FrankaPanda', 'FrankaPanda:robotiq:2', 'Yam::2', 'Fetch'"
    )
    
    parser.add_argument(
        "--robot_position",
        type=float,
        nargs=3,
        default=[0, 0, 0],
        help="Position for the first robot as x y z (default: 0 0 0). Additional robots are auto-spaced."
    )
    
    parser.add_argument(
        "--robot_orientation",
        type=float,
        nargs=4,
        default=[0, 0, 0, 1],
        help="Orientation for the first robot as quaternion x y z w (default: 0 0 0 1)"
    )
    
    parser.add_argument(
        "--arm_controller",
        type=str,
        choices=["ik", "joint_pos"],
        default="ik",
        help="Type of arm controller to use. 'ik' for InverseKinematicsController (default), "
             "'joint_pos' for JointController with position control"
    )
    
    parser.add_argument(
        "--ground_plane",
        action="store_true",
        help="Include a ground plane in the scene. If not set, no ground plane is loaded."
    )
    
    parser.add_argument(
        "--external_sensors",
        type=str,
        default=None,
        help="Path to a .yaml file containing external sensor configurations. "
             "See OmniGibson external_sensors config format."
    )

    parser.add_argument(
        "--skybox_rot_x_deg",
        type=float,
        default=0.0,
        help="Initial skybox rotation around X axis in degrees (default: 0.0)."
    )

    parser.add_argument(
        "--skybox_rot_y_deg",
        type=float,
        default=0.0,
        help="Initial skybox rotation around Y axis in degrees (default: 0.0)."
    )

    parser.add_argument(
        "--skybox_yaw_deg",
        type=float,
        default=0.0,
        help="Initial skybox rotation around Z axis in degrees (default: 0.0)."
    )

    parser.add_argument(
        "--skybox_intensity",
        type=float,
        default=1000.0,
        help="Initial skybox intensity (default: 1000.0)."
    )

    parser.add_argument(
        "--skybox_yaw_step_deg",
        type=float,
        default=2.0,
        help="Per-key skybox yaw step in degrees for J/L controls (default: 2.0)."
    )

    parser.add_argument(
        "--skybox_intensity_scale",
        type=float,
        default=1.1,
        help="Per-key multiplicative skybox intensity scale for I/K (default: 1.1)."
    )
    
    parser.add_argument(
        "--dataset_objects",
        type=str,
        nargs="+",
        default=[],
        help="Dataset objects to spawn. Format: 'dataset:category:model[:name][:fixed_base]'. "
             "Append ':fixed_base' to make object static. "
             "Examples: 'real2sim-assets:yellow_banana:sscmmv', 'real2sim-assets:table:abc123:fixed_base'"
    )
    
    parser.add_argument(
        "--usd_objects",
        type=str,
        nargs="+",
        default=[],
        help="USD objects to spawn from arbitrary file paths. Format: 'usd:<category>:<path>[:name][:fixed_base]'. "
             "Append ':fixed_base' to make object static. "
             "Examples: 'usd:banana:/path/to/banana.usd', 'usd:table:/path/table.usda:fixed_base'"
    )
    
    parser.add_argument(
        "--scene_objects_info",
        type=str,
        default=None,
        help="Path to scene_objects_info.json file from pipeline stage 10 (s10_sim). "
             "Use together with --pb_scene_poses to load objects with their poses."
    )
    
    parser.add_argument(
        "--pb_scene_poses",
        type=str,
        default=None,
        help="Path to pb_scene_poses.json file from pipeline stage 11 (s11_physics). "
             "Use together with --scene_objects_info to load objects with their poses."
    )
    
    parser.add_argument(
        "--scene_objects_categories",
        type=str,
        nargs="+",
        default=None,
        help="Filter categories to load from scene_objects_info. Only objects with matching "
             "categories will be loaded. Example: --scene_objects_categories yellow_banana apple"
    )
    
    parser.add_argument(
        "--load_scene",
        type=str,
        default=None,
        help="Path to a pre-saved scene JSON file to load. When specified, the scene will be "
             "restored from this file and you can continue editing. Other object/robot arguments "
             "are ignored when loading from a saved scene."
    )
    
    return parser.parse_args()


def parse_robot_arg(robot_arg):
    """
    Parse robot argument string into robot config dict.
    
    Format: 'ClassName', 'ClassName:end_effector', 'ClassName::count', or 'ClassName:end_effector:count'
    Examples: 
        'FrankaPanda' -> {'type': 'FrankaPanda', 'count': 1}
        'FrankaPanda:robotiq' -> {'type': 'FrankaPanda', 'end_effector': 'robotiq', 'count': 1}
        'FrankaPanda:robotiq:2' -> {'type': 'FrankaPanda', 'end_effector': 'robotiq', 'count': 2}
        'Yam::2' -> {'type': 'Yam', 'count': 2}
        'FrankaMounted' -> {'type': 'FrankaMounted', 'count': 1}
    
    Args:
        robot_arg (str): Robot specification string
        
    Returns:
        dict: Robot configuration dictionary
    """
    if robot_arg is None:
        return None
    
    parts = robot_arg.split(":")
    robot_config = {"type": parts[0], "count": 1}
    
    if len(parts) > 1 and parts[1]:
        # Non-empty end_effector
        robot_config["end_effector"] = parts[1]
    
    if len(parts) > 2 and parts[2]:
        # Count specified
        try:
            robot_config["count"] = int(parts[2])
        except ValueError:
            print(f"Warning: Invalid robot count '{parts[2]}', using 1")
            robot_config["count"] = 1
    
    return robot_config


def parse_dataset_object_arg(obj_arg):
    """
    Parse dataset object argument string into object spec dict.
    
    Format: 'dataset:category:model', 'dataset:category:model:fixed_base', or 'dataset:category:model:name:fixed_base'
    Examples:
        'real2sim-assets:yellow_banana:sscmmv' -> 
            {'dataset_name': 'real2sim-assets', 'category': 'yellow_banana', 'model': 'sscmmv'}
        'real2sim-assets:yellow_banana:sscmmv:fixed_base' ->
            {'dataset_name': 'real2sim-assets', 'category': 'yellow_banana', 'model': 'sscmmv', 'fixed_base': True}
        'behavior-1k-assets:apple:agvzbp:my_apple' ->
            {'dataset_name': 'behavior-1k-assets', 'category': 'apple', 'model': 'agvzbp', 'name': 'my_apple'}
        'behavior-1k-assets:apple:agvzbp:my_apple:fixed_base' ->
            {'dataset_name': 'behavior-1k-assets', 'category': 'apple', 'model': 'agvzbp', 'name': 'my_apple', 'fixed_base': True}
    
    Args:
        obj_arg (str): Dataset object specification string
        
    Returns:
        dict: Object specification dictionary, or None if invalid
    """
    parts = obj_arg.split(":")
    if len(parts) < 3:
        print(f"Warning: Invalid dataset object format '{obj_arg}'. Expected 'dataset:category:model'")
        return None
    
    obj_spec = {
        "dataset_name": parts[0],
        "category": parts[1],
        "model": parts[2],
        "fixed_base": False,
    }
    
    # Check remaining parts for name and/or fixed_base
    remaining_parts = parts[3:]
    for part in remaining_parts:
        if part.lower() == "fixed_base":
            obj_spec["fixed_base"] = True
        elif part and "name" not in obj_spec:
            obj_spec["name"] = part
    
    return obj_spec


def parse_usd_object_arg(obj_arg):
    """
    Parse USD object argument string into object spec dict.
    
    Format: 'usd:<category>:<path>', 'usd:<category>:<path>:fixed_base', or 'usd:<category>:<path>:<name>:fixed_base'
    Examples:
        'usd:banana:/path/to/banana.usd' -> 
            {'category': 'banana', 'usd_path': '/path/to/banana.usd'}
        'usd:banana:/path/to/banana.usd:fixed_base' -> 
            {'category': 'banana', 'usd_path': '/path/to/banana.usd', 'fixed_base': True}
        'usd:table:/data/models/table.usda:my_table' ->
            {'category': 'table', 'usd_path': '/data/models/table.usda', 'name': 'my_table'}
        'usd:table:/data/models/table.usda:my_table:fixed_base' ->
            {'category': 'table', 'usd_path': '/data/models/table.usda', 'name': 'my_table', 'fixed_base': True}
    
    Args:
        obj_arg (str): USD object specification string
        
    Returns:
        dict: Object specification dictionary, or None if invalid
    """
    parts = obj_arg.split(":")
    
    # Handle paths that may contain colons (e.g., Windows paths like C:\path\to\file.usd)
    if len(parts) < 3:
        print(f"Warning: Invalid USD object format '{obj_arg}'. Expected 'usd:<category>:<path>'")
        return None
    
    if parts[0].lower() != "usd":
        print(f"Warning: USD object spec must start with 'usd:'. Got: '{obj_arg}'")
        return None
    
    category = parts[1]
    
    # Reconstruct the path (handles cases with colons in path, e.g. Windows absolute paths)
    # Check if the third part looks like a Windows drive letter (single letter)
    if len(parts) >= 4 and len(parts[2]) == 1 and parts[2].isalpha():
        # Windows path like C:\path\to\file.usd
        usd_path = parts[2] + ":" + parts[3]
        remaining_parts = parts[4:]
    else:
        usd_path = parts[2]
        remaining_parts = parts[3:]
    
    obj_spec = {
        "category": category,
        "usd_path": usd_path,
        "fixed_base": False,
    }
    
    # Check remaining parts for name and/or fixed_base
    for part in remaining_parts:
        if part.lower() == "fixed_base":
            obj_spec["fixed_base"] = True
        elif part and "name" not in obj_spec:
            obj_spec["name"] = part
    
    return obj_spec


def main():
    args = parse_args()
    
    # Load cam2world transform if provided
    cam2world_tf = None
    if args.cam2world is not None and os.path.exists(args.cam2world):
        cam2world_tf = th.from_numpy(np.load(args.cam2world)).float()
    
    # Load initial poses if provided
    object_poses = None
    if args.poses_file is not None and os.path.exists(args.poses_file):
        with open(args.poses_file, "r") as f:
            poses_data = json.load(f)
        # Convert to list of (position, orientation) tuples
        object_poses = []
        for i in range(len(args.objects)):
            key = f"object_{i}"
            if key in poses_data:
                object_poses.append((
                    poses_data[key]["position"],
                    poses_data[key]["orientation"]
                ))
            else:
                object_poses.append(None)
    
    # Parse robot configurations (can be multiple robots)
    robot_configs = []
    if args.robot:
        for robot_arg in args.robot:
            robot_config = parse_robot_arg(robot_arg)
            if robot_config is not None:
                # Apply position/orientation to first robot only
                if len(robot_configs) == 0:
                    robot_config["position"] = args.robot_position
                    robot_config["orientation"] = args.robot_orientation
                robot_configs.append(robot_config)
    
    # Parse dataset objects
    dataset_objects = []
    for obj_arg in args.dataset_objects:
        obj_spec = parse_dataset_object_arg(obj_arg)
        if obj_spec is not None:
            dataset_objects.append(obj_spec)
    
    # Parse USD objects
    usd_objects = []
    for obj_arg in args.usd_objects:
        obj_spec = parse_usd_object_arg(obj_arg)
        if obj_spec is not None:
            usd_objects.append(obj_spec)
    
    # Create and run editor
    editor = InteractiveSceneEditor(
        scene_name=args.scene_name,
        gs_background_path=args.background,
        mesh_background_path=args.mesh_background,
        hdr_background_path=args.hdr_background,
        object_paths=args.objects,
        object_poses=object_poses,
        dataset_objects=dataset_objects,
        usd_objects=usd_objects,
        scene_objects_info_path=args.scene_objects_info,
        pb_scene_poses_path=args.pb_scene_poses,
        scene_objects_categories=args.scene_objects_categories,
        cam2world_tf=cam2world_tf,
        asset_dir=args.asset_dir,
        translation_delta=args.translation_delta,
        rotation_delta=args.rotation_delta,
        scale_delta=args.scale_delta,
        robot_configs=robot_configs,
        arm_controller=args.arm_controller,
        ground_plane=args.ground_plane,
        external_sensors_config=args.external_sensors,
        skybox_rot_x_deg=args.skybox_rot_x_deg,
        skybox_rot_y_deg=args.skybox_rot_y_deg,
        skybox_yaw_deg=args.skybox_yaw_deg,
        skybox_intensity=args.skybox_intensity,
        skybox_yaw_step_deg=args.skybox_yaw_step_deg,
        skybox_intensity_scale=args.skybox_intensity_scale,
        load_scene_json=args.load_scene,
    )
    
    editor.run()


if __name__ == "__main__":
    main()
