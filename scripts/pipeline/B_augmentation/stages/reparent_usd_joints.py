#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Reparent joints in a USD file to be children of their parent link prims.

This is necessary because Isaac Sim's URDF importer places joints in a separate
/joints scope, but OmniGibson expects them to be children of the parent link Xforms.

Usage:
    python reparent_usd_joints.py <usd_path>
"""
import os
import site
import sys
from pathlib import Path

# Set headless mode before any Isaac Sim imports
os.environ["OMNIGIBSON_HEADLESS"] = "1"
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

_PXR_BOOTSTRAP_ENV = "SIMFOUNDRY_PXR_BOOTSTRAPPED"


def _prepend_env_path(env: dict[str, str], key: str, value: str) -> None:
    current = env.get(key)
    paths = [] if not current else current.split(os.pathsep)
    if value not in paths:
        env[key] = value if not current else value + os.pathsep + current


def _find_usd_extension_root() -> Path:
    roots: list[Path] = []
    for site_root in site.getsitepackages():
        roots.extend(Path(site_root).glob("isaacsim/extscache/omni.usd.libs-*"))
    user_site = site.getusersitepackages()
    if user_site:
        roots.extend(Path(user_site).glob("isaacsim/extscache/omni.usd.libs-*"))

    valid_roots = sorted(root for root in roots if (root / "pxr").is_dir() and (root / "bin").is_dir())
    if not valid_roots:
        raise RuntimeError("Could not locate Isaac Sim omni.usd.libs extension containing pxr bindings.")
    return valid_roots[-1]


def _bootstrap_pxr():
    try:
        from pxr import Sdf, Usd, UsdPhysics

        return Sdf, Usd, UsdPhysics
    except (ImportError, ModuleNotFoundError):
        if os.environ.get(_PXR_BOOTSTRAP_ENV) == "1":
            raise

        usd_root = _find_usd_extension_root()
        env = os.environ.copy()
        _prepend_env_path(env, "PYTHONPATH", str(usd_root))
        _prepend_env_path(env, "LD_LIBRARY_PATH", str(Path(sys.prefix) / "lib"))
        _prepend_env_path(env, "LD_LIBRARY_PATH", str(usd_root / "bin"))
        env[_PXR_BOOTSTRAP_ENV] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)
        raise RuntimeError("Failed to re-exec with USD bindings enabled.")


def reparent_joints(usd_path: str) -> bool:
    """
    Post-processes a USD file to reparent joints from the /joints scope to be 
    children of their respective parent link prims.
    """
    Sdf, Usd, UsdPhysics = _bootstrap_pxr()
    
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"Error: Could not open USD stage at {usd_path}")
        return False
    
    root_layer = stage.GetRootLayer()
    joints_reparented = False
    
    # Find all joint prims in the stage
    joints_to_move = []
    for prim in stage.Traverse():
        prim_type = prim.GetTypeName().lower()
        if "joint" in prim_type and "fixed" not in prim_type:
            # Get the parent link from physics:body0
            body0_rel = prim.GetRelationship("physics:body0")
            if body0_rel and body0_rel.GetTargets():
                parent_link_path = body0_rel.GetTargets()[0]
                current_path = prim.GetPath()
                
                # Check if joint is NOT already a child of the parent link
                parent_link_path_str = str(parent_link_path)
                current_path_str = str(current_path)
                
                if not current_path_str.startswith(parent_link_path_str + "/"):
                    joints_to_move.append({
                        "current_path": current_path,
                        "parent_link_path": parent_link_path,
                        "joint_name": prim.GetName(),
                    })
    
    # Move joints to be children of their parent links
    for joint_info in joints_to_move:
        current_path = joint_info["current_path"]
        parent_link_path = joint_info["parent_link_path"]
        joint_name = joint_info["joint_name"]
        
        # New path for the joint: parent_link / joint_name
        new_path = Sdf.Path(str(parent_link_path) + "/" + joint_name)
        
        print(f"Moving joint from {current_path} to {new_path}")
        
        # Copy the prim to the new location
        if Sdf.CopySpec(root_layer, current_path, root_layer, new_path):
            # Delete from the old location
            parent_path = current_path.GetParentPath()
            if parent_path and str(parent_path) in root_layer.rootPrims:
                # Joint is a direct child of root, delete differently
                del root_layer.rootPrims[current_path.name]
            else:
                # Joint is nested (e.g., in /joints scope)
                parent_spec = root_layer.GetPrimAtPath(parent_path)
                if parent_spec and current_path.name in parent_spec.nameChildren:
                    del parent_spec.nameChildren[current_path.name]
            
            joints_reparented = True
        else:
            print(f"Warning: Failed to copy joint from {current_path} to {new_path}")
    
    # Clean up empty /joints scope if it exists
    joints_scope_path = Sdf.Path("/joints")
    joints_scope = stage.GetPrimAtPath(joints_scope_path)
    if joints_scope and not list(joints_scope.GetChildren()):
        # Scope is empty, remove it
        if "joints" in root_layer.rootPrims:
            del root_layer.rootPrims["joints"]
            print("Removed empty /joints scope")
    
    # Also check for joints scope under the default prim
    default_prim = stage.GetDefaultPrim()
    if default_prim:
        joints_scope_path = Sdf.Path(str(default_prim.GetPath()) + "/joints")
        joints_scope = stage.GetPrimAtPath(joints_scope_path)
        if joints_scope and not list(joints_scope.GetChildren()):
            parent_spec = root_layer.GetPrimAtPath(default_prim.GetPath())
            if parent_spec and "joints" in parent_spec.nameChildren:
                del parent_spec.nameChildren["joints"]
                print("Removed empty joints scope under default prim")
    
    if joints_reparented:
        print(f"Successfully reparented joints in {usd_path}")
    else:
        print(f"No joints needed reparenting in {usd_path}")
    
    # Set stiffness to 0 for all joint drives. This prevents spring-loaded joints
    # from moving unexpectedly when OmniGibson loads the object.
    joints_modified = False
    for prim in stage.Traverse():
        prim_type = prim.GetTypeName().lower()
        if "joint" in prim_type and "fixed" not in prim_type:
            # Check for drive API (angular for revolute, linear for prismatic)
            if prim.HasAPI(UsdPhysics.DriveAPI):
                # Try angular drive (for revolute joints)
                angular_drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                if angular_drive:
                    stiffness_attr = angular_drive.GetStiffnessAttr()
                    if stiffness_attr:
                        stiffness_attr.Set(0.0)
                        joints_modified = True
                        print(f"Set angular stiffness=0 for {prim.GetPath()}")
                
                # Try linear drive (for prismatic joints)
                linear_drive = UsdPhysics.DriveAPI.Get(prim, "linear")
                if linear_drive:
                    stiffness_attr = linear_drive.GetStiffnessAttr()
                    if stiffness_attr:
                        stiffness_attr.Set(0.0)
                        joints_modified = True
                        print(f"Set linear stiffness=0 for {prim.GetPath()}")
    
    if joints_reparented or joints_modified:
        stage.Save()
    
    return joints_reparented or joints_modified


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <usd_path>")
        sys.exit(1)
    
    usd_path = sys.argv[1]
    if not os.path.exists(usd_path):
        print(f"Error: USD file not found: {usd_path}")
        sys.exit(1)
    
    reparent_joints(usd_path)
    sys.exit(0)
