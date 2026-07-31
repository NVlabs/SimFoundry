# Copyright (c) 2023 Stanford Vision and Learning Group
# Licensed under the MIT License. Full text in THIRD_PARTY_LICENSES.md.
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This file originates from OmniGibson within the BEHAVIOR-1K repository
# (OmniGibson/omnigibson/utils/asset_conversion_utils.py), modified by NVIDIA to be
# OmniGibson / Isaac Sim agnostic. NVIDIA modifications are licensed under Apache-2.0;
# the adapted upstream portions remain subject to the MIT terms above, whose notice
# MIT requires be retained.

"""
Source code originally from BEHAVIOR-1K repository (https://github.com/StanfordVL/BEHAVIOR-1K),
slightly modified to be OmniGibson / IsaacSim agnostic
"""

import io
import json
import math
import os
import pathlib
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from os.path import exists
from pathlib import Path
from xml.dom import minidom
from scipy.spatial.transform import Rotation as R
from simfoundry.utils.urdfpy_utils import URDF
import numpy as np

import click
import pymeshlab
import torch as th
import trimesh
from typing import Literal, Optional, Union


# TODO: MOVE ALL THESE CONSTANTS TO A CONFIG FILE
_SPLIT_COLLISION_MESHES = False

REVOLUTE_JOINT_FRIC = 0.01
PRISMATIC_JOINT_FRIC = 0.4



def _split_obj_file_into_connected_components(obj_fpath):
    """
    Splits an OBJ file into individual OBJ files, each containing a single connected mesh.

    Args:
        obj_fpath (str): The file path to the input OBJ file.

    Returns:
        int: The number of individual connected mesh files created.

    The function performs the following steps:
    1. Loads the OBJ file using trimesh.
    2. Splits the loaded mesh into individual connected components.
    3. Saves each connected component as a separate OBJ file in the same directory as the input file.
    """
    # Open file in trimesh
    obj = trimesh.load(obj_fpath, file_type="obj", force="mesh")

    # Split to grab all individual bodies
    obj_bodies = obj.split(only_watertight=False)

    # Procedurally create new files in the same folder as obj_fpath
    out_fpath = os.path.dirname(obj_fpath)
    out_fname_root = os.path.splitext(os.path.basename(obj_fpath))[0]

    for i, obj_body in enumerate(obj_bodies):
        # Write to a new file
        obj_body.export(f"{out_fpath}/{out_fname_root}_{i}.obj", "obj")

    # We return the number of splits we had
    return len(obj_bodies)


def _split_all_objs_in_urdf(urdf_fpath, name_suffix="split", mesh_fpath_offset="."):
    """
    Splits the OBJ references in the given URDF file into separate files for each connected component.

    This function parses a URDF file, finds all collision mesh references, splits the referenced OBJ files into
    connected components, and updates the URDF file to reference these new OBJ files. The updated URDF file is
    saved with a new name.

    Args:
        urdf_fpath (str): The file path to the URDF file to be processed.
        name_suffix (str, optional): Suffix to append to the output URDF file name. Defaults to "split".
        mesh_fpath_offset (str, optional): Offset path to the directory containing the mesh files. Defaults to ".".

    Returns:
        str: The file path to the newly created URDF file with split OBJ references.
    """
    tree = ET.parse(urdf_fpath)
    root = tree.getroot()
    urdf_dir = os.path.dirname(urdf_fpath)
    out_fname_root = os.path.splitext(os.path.basename(urdf_fpath))[0]

    def recursively_find_collision_meshes(ele):
        # Finds all collision meshes starting at @ele
        cols = []
        for child in ele:
            if child.tag == "collision":
                # If the nested geom type is a mesh, add this to our running list along with its parent node
                if child.find("./geometry/mesh") is not None:
                    cols.append((child, ele))
            elif child.tag == "visual":
                # There will be no collision mesh internally here so we simply pass
                continue
            else:
                # Recurisvely look through all children of the child
                cols += recursively_find_collision_meshes(ele=child)

        return cols

    # Iterate over the tree and find all collision entries
    col_elements = recursively_find_collision_meshes(ele=root)

    # For each collision element and its parent, we remove the original one and create a set of new ones with their
    # filename references changed
    for col, parent in col_elements:
        # Don't change the original
        col_copy = deepcopy(col)
        # Delete the original
        parent.remove(col)
        # Create new objs first so we know how many we need to create in the URDF
        obj_fpath = col_copy.find("./geometry/mesh").attrib["filename"]
        n_new_objs = _split_obj_file_into_connected_components(obj_fpath=f"{urdf_dir}/{mesh_fpath_offset}/{obj_fpath}")
        # Create the new objs in the URDF
        for i in range(n_new_objs):
            # Copy collision again
            col_copy_copy = deepcopy(col_copy)
            # Modify the filename
            fname = col_copy_copy.find("./geometry/mesh").attrib["filename"]
            fname = fname.split(".obj")[0] + f"_{i}.obj"
            col_copy_copy.find("./geometry/mesh").attrib["filename"] = fname
            # Add to parent
            parent.append(col_copy_copy)

    # Finally, write this to a new file
    urdf_out_path = f"{urdf_dir}/{out_fname_root}_{name_suffix}.urdf"
    tree.write(urdf_out_path)

    # Return the urdf it wrote to
    return urdf_out_path


def _get_visual_objs_from_urdf(urdf_path):
    """
    Extracts visual objects from a URDF file.

    Args:
        urdf_path (str): Path to the URDF file.

    Returns:
        OrderedDict: A dictionary mapping link names to dictionaries of visual meshes. Each link name (e.g., 'base_link')
                     maps to another dictionary where the keys are visual mesh names and the values are the corresponding
                     visual object file paths. If no visual object file is found for a mesh, the value will be None.
    """
    visual_objs = OrderedDict()
    # Parse URDF
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for ele in root:
        if ele.tag == "link":
            name = ele.get("name").replace("-", "_")
            visual_objs[name] = OrderedDict()
            for sub_ele in ele:
                if sub_ele.tag == "visual":
                    visual_mesh_name = sub_ele.get("name", "visuals").replace("-", "_")
                    obj_file = None if sub_ele.find(".//mesh") is None else sub_ele.find(".//mesh").get("filename")
                    if obj_file is None:
                        print(f"Warning: No obj file found associated with {name}/{visual_mesh_name}!")
                    visual_objs[name][visual_mesh_name] = obj_file

    return visual_objs


def _pretty_print_xml(current, parent=None, index=-1, depth=0, use_tabs=False):
    """
    Recursively formats an XML element tree to be pretty-printed with indentation.

    Args:
        current (xml.etree.ElementTree.Element): The current XML element to format.
        parent (xml.etree.ElementTree.Element, optional): The parent XML element. Defaults to None.
        index (int, optional): The index of the current element in the parent's children. Defaults to -1.
        depth (int, optional): The current depth in the XML tree, used for indentation. Defaults to 0.
        use_tabs (bool, optional): If True, use tabs for indentation; otherwise, use spaces. Defaults to False.

    Returns:
        None
    """
    space = "\t" if use_tabs else " " * 4
    for i, node in enumerate(current):
        _pretty_print_xml(node, current, i, depth + 1)
    if parent is not None:
        if index == 0:
            parent.text = "\n" + (space * depth)
        else:
            parent[index - 1].tail = "\n" + (space * depth)
        if index == len(parent) - 1:
            current.tail = "\n" + (space * (depth - 1))


def _space_string_to_tensor(string):
    """
    Converts a space-separated string of numbers into a PyTorch tensor.

    Examples:
        "0 1 2" => tensor([0., 1., 2.])

    Args:
        string (str): Space-separated string of numbers to convert.

    Returns:
        torch.Tensor: Tensor containing the numerical values from the input string.
    """
    return th.tensor([float(x) for x in string.split(" ")])


def _tensor_to_space_script(array):
    """
    Converts a numeric array into the string format in mujoco.

    Examples:
        [0, 1, 2] => "0 1 2"

    Args:
        array (th.Tensor): Array to convert to a string

    Returns:
        str: String equivalent of @array
    """
    return " ".join(["{}".format(x) for x in array.tolist()])

def _convert_to_xml_string(inp):
    """
    Converts any type of {bool, int, float, list, tuple, array, string, th.Tensor} into a URDF-compatible string.
    Note that an input string / th.Tensor results in a no-op action.

    Args:
        inp: Input to convert to string

    Returns:
        str: String equivalent of @inp

    Raises:
        ValueError: If the input type is unsupported.
    """
    if type(inp) in {list, tuple, th.Tensor}:
        return _tensor_to_space_script(inp)
    elif type(inp) in {int, float, bool, th.float32, th.float64, th.int32, th.int64}:
        return str(inp).lower()
    elif type(inp) in {str}:
        return inp
    else:
        raise ValueError("Unsupported type received: got {}".format(type(inp)))


def _create_urdf_joint(
    name,
    parent,
    child,
    pos=(0, 0, 0),
    rpy=(0, 0, 0),
    joint_type="fixed",
    axis=None,
    damping=None,
    friction=None,
    limits=None,
):
    """
    Generates URDF joint
    Args:
        name (str): Name of this joint
        parent (str or ET.Element): Name of parent link or parent link element itself for this joint
        child (str or ET.Element): Name of child link or child link itself for this joint
        pos (list or tuple or th.Tensor): (x,y,z) offset pos values when creating the collision body
        rpy (list or tuple or th.Tensor): (r,p,y) offset rot values when creating the joint
        joint_type (str): What type of joint to create. Must be one of {fixed, revolute, prismatic}
        axis (None or 3-tuple): If specified, should be (x,y,z) axis corresponding to DOF
        damping (None or float): If specified, should be damping value to apply to joint
        friction (None or float): If specified, should be friction value to apply to joint
        limits (None or 2-tuple): If specified, should be min / max limits to the applied joint
    Returns:
        ET.Element: Generated joint element
    """
    # Create the initial joint
    jnt = ET.Element("joint", name=name, type=joint_type)
    # Create origin subtag
    ET.SubElement(
        jnt,
        "origin",
        attrib={"rpy": _convert_to_xml_string(rpy), "xyz": _convert_to_xml_string(pos)},
    )
    # Make sure parent and child are both names (str) -- if they're not str already, we assume it's the element ref
    if not isinstance(parent, str):
        parent = parent.get("name")
    if not isinstance(child, str):
        child = child.get("name")
    # Create parent and child subtags
    parent = ET.SubElement(jnt, "parent", link=parent)
    child = ET.SubElement(jnt, "child", link=child)
    # Add additional parameters if specified
    if axis is not None:
        ET.SubElement(jnt, "axis", xyz=_convert_to_xml_string(axis))
    dynamic_params = {}
    if damping is not None:
        dynamic_params["damping"] = _convert_to_xml_string(damping)
    if friction is not None:
        dynamic_params["friction"] = _convert_to_xml_string(friction)
    if dynamic_params:
        ET.SubElement(jnt, "dynamics", **dynamic_params)
    if limits is not None:
        ET.SubElement(jnt, "limit", lower=limits[0], upper=limits[1])

    # Return this element
    return jnt


def _create_urdf_link(name, subelements=None, mass=None, inertia=None):
    """
    Generates URDF link element
    Args:
        name (str): Name of this link
        subelements (None or list): If specified, specifies all nested elements that should belong to this link
            (e.g.: visual, collision body elements)
        mass (None or float): If specified, will add an inertial tag with specified mass value
        inertia (None or 6-array): If specified, will add an inertial tag with specified inertia value
            Value should be (ixx, iyy, izz, ixy, ixz, iyz)
    Returns:
        ET.Element: Generated link
    """
    # Create the initial link
    link = ET.Element("link", name=name)
    # Add all subelements if specified
    if subelements is not None:
        for ele in subelements:
            link.append(ele)
    # Add mass subelement if requested
    if mass is not None or inertia is not None:
        inertial = ET.SubElement(link, "inertial")
    if mass is not None:
        ET.SubElement(inertial, "mass", value=_convert_to_xml_string(mass))
    if inertia is not None:
        axes = ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"]
        inertia_vals = {ax: str(i) for ax, i in zip(axes, inertia)}
        ET.SubElement(inertial, "inertia", **inertia_vals)

    # Return this element
    return link


def _create_urdf_meta_link(
    root_element,
    meta_link_name,
    parent_link_name="base_link",
    pos=(0, 0, 0),
    rpy=(0, 0, 0),
):
    """
    Creates the appropriate URDF joint and link for a meta link and appends it to the root element.

    Args:
        root_element (Element): The root XML element to which the meta link will be appended.
        meta_link_name (str): The name of the meta link to be created.
        parent_link_name (str, optional): The name of the parent link. Defaults to "base_link".
        pos (tuple, optional): The position of the joint in the form (x, y, z). Defaults to (0, 0, 0).
        rpy (tuple, optional): The roll, pitch, and yaw of the joint in the form (r, p, y). Defaults to (0, 0, 0).

    Returns:
        None
    """
    # Create joint
    jnt = _create_urdf_joint(
        name=f"{meta_link_name}_joint",
        parent=parent_link_name,
        child=f"{meta_link_name}_link",
        pos=pos,
        rpy=rpy,
        joint_type="fixed",
    )
    # Create child link
    link = _create_urdf_link(
        name=f"{meta_link_name}_link",
        mass=0.0001,
        inertia=[0.00001, 0.00001, 0.00001, 0, 0, 0],
    )

    # Add to root element
    root_element.append(jnt)
    root_element.append(link)


def _save_xmltree_as_urdf(root_element, name, dirpath, unique_urdf=False):
    """
    Generates a URDF file corresponding to @xmltree at @dirpath with name @name.urdf.
    Args:
        root_element (ET.Element): Element tree that compose the URDF
        name (str): Name of this file (name assigned to robot tag)
        dirpath (str): Absolute path to the location / filename for the generated URDF
        unique_urdf (bool): Whether to use a unique identifier when naming urdf (uses current datetime)
    Returns:
        str: Path to newly created urdf (fpath/<name>.urdf)
    """
    # Write to fpath, making sure the directory exists (if not, create it)
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    # Get file
    date = datetime.now().isoformat(timespec="microseconds").replace(".", "_").replace(":", "_").replace("-", "_")
    fname = f"{name}_{date}.urdf" if unique_urdf else f"{name}.urdf"
    fpath = os.path.join(dirpath, fname)
    with open(fpath, "w") as f:
        # Write top level header line first
        f.write('<?xml version="1.0" ?>\n')
        # Convert xml to string form and write to file
        _pretty_print_xml(current=root_element)
        xml_str = ET.tostring(root_element, encoding="unicode")
        f.write(xml_str)

    # Return path to file
    return fpath


def _get_objects_config_from_scene_urdf(urdf):
    """
    Parses a URDF file to extract object configuration information.

    Args:
        urdf (str): Path to the URDF file.

    Returns:
        dict: A dictionary containing the configuration of objects extracted from the URDF file.
    """
    tree = ET.parse(urdf)
    root = tree.getroot()
    objects_cfg = dict()
    _get_objects_config_from_element(root, model_pose_info=objects_cfg)
    return objects_cfg


def _get_objects_config_from_element(element, model_pose_info):
    """
    Extracts and populates object configuration information from an URDF element.

    This function processes an URDF element to extract joint and link information,
    populating the provided `model_pose_info` dictionary with the relevant data.

    Args:
        element (xml.etree.ElementTree.Element): The URDF element containing object configuration data.
        model_pose_info (dict): A dictionary to be populated with the extracted configuration information.

    The function performs two passes through the URDF element:
    1. In the first pass, it extracts joint information and populates `model_pose_info` with joint pose data.
    2. In the second pass, it extracts link information, imports object models, and updates `model_pose_info` with
       additional configuration details such as category, model, bounding box, rooms, scale, and object scope.

    The function also handles nested elements by recursively calling itself for child elements.

    Note:
        - Joint names with hyphens are replaced with underscores.
        - The function asserts that each link name (except "world") is present in `model_pose_info` after the first pass.
    """
    # First pass through, populate the joint pose info
    for ele in element:
        if ele.tag == "joint":
            name, pos, quat, fixed_jnt = _get_joint_info(ele)
            name = name.replace("-", "_")
            model_pose_info[name] = {
                "bbox_pos": pos,
                "bbox_quat": quat,
                "cfg": {
                    "fixed_base": fixed_jnt,
                },
            }

    # Second pass through, import object models
    for ele in element:
        if ele.tag == "link":
            # This is a valid object, import the model
            name = ele.get("name").replace("-", "_")
            if name == "world":
                # Skip this
                pass
            else:
                print(f"Grabbing obj config for ele name: {name}")
                assert name in model_pose_info, f"Did not find {name} in current model pose info!"
                model_pose_info[name]["cfg"]["category"] = ele.get("category")
                model_pose_info[name]["cfg"]["visual_only"] = False #ele.get("category") in _VISUAL_ONLY_CATEGORIES
                model_pose_info[name]["cfg"]["model"] = ele.get("model")
                model_pose_info[name]["cfg"]["bounding_box"] = (
                    _space_string_to_tensor(ele.get("bounding_box")) if "bounding_box" in ele.keys() else None
                )
                in_rooms = ele.get("rooms", "")
                if in_rooms:
                    in_rooms = in_rooms.split(",")
                model_pose_info[name]["cfg"]["in_rooms"] = in_rooms
                model_pose_info[name]["cfg"]["scale"] = (
                    _space_string_to_tensor(ele.get("scale")) if "scale" in ele.keys() else None
                )
                model_pose_info[name]["cfg"]["bddl_object_scope"] = ele.get("object_scope", None)

        # If there's children nodes, we iterate over those
        for child in ele:
            _get_objects_config_from_element(child, model_pose_info=model_pose_info)



def _get_joint_info(joint_element):
    """
    Extracts joint information from an URDF element.

    Args:
        joint_element (xml.etree.ElementTree.Element): The URDF element containing joint information.

    Returns:
        tuple: A tuple containing:
            - child (str or None): The name of the child link, or None if not specified.
            - pos (numpy.ndarray or None): The position as a tensor, or None if not specified.
            - quat (numpy.ndarray or None): The orientation as a quaternion, or None if not specified.
            - fixed_jnt (bool): True if the joint is fixed, False otherwise.
    """
    child, pos, quat, fixed_jnt = None, None, None, None
    fixed_jnt = joint_element.get("type") == "fixed"
    for ele in joint_element:
        if ele.tag == "origin":
            quat = R.from_euler("xyz", _space_string_to_tensor(ele.get("rpy"))).to_quat()
            pos = _space_string_to_tensor(ele.get("xyz"))
        elif ele.tag == "child":
            child = ele.get("link")
    return child, pos, quat, fixed_jnt


def make_mesh_positive(mesh_fpath, scale, output_suffix="mirror"):
    assert "." not in mesh_fpath
    for sc, letter in zip(scale, "xyz"):
        if sc < 0:
            output_suffix += f"_{letter}"
    for filetype in [".obj", ".stl", ".dae"]:
        fpath = f"{mesh_fpath}{filetype}"
        out_fpath = f"{mesh_fpath}_{output_suffix}{filetype}"
        kwargs = dict()
        if filetype == ".dae":
            kwargs["force"] = "mesh"
        if os.path.exists(fpath):
            try:
                tm = trimesh.load(fpath, **kwargs)
                tm.apply_scale(scale)
                tm.export(out_fpath)
                if filetype == ".obj":
                    # Update header lines
                    lines = []
                    with open(fpath, "r") as f:
                        for line in f.readlines():
                            if line.startswith("v "):
                                break
                            lines.append(line)
                    start = False
                    with open(out_fpath, "r") as f:
                        for line in f.readlines():
                            if line.startswith("v "):
                                start = True
                            if start:
                                lines.append(line)
                    with open(out_fpath, "w+") as f:
                        f.writelines(lines)
            except KeyError:
                # Degenerate mesh, so immediately return
                return None
    return output_suffix


def make_asset_positive(urdf_fpath, output_suffix="mirror"):
    assert urdf_fpath.endswith(".urdf")
    out_lines = []

    with open(urdf_fpath, "r") as f:
        for line in f.readlines():
            # print(line)
            out_line = line
            if "<mesh " in line and "scale=" in line:
                # Grab the scale, and possibly convert negative values
                scale_str = line.split("scale=")[1].split('"')[1]
                scale = _space_string_to_tensor(scale_str)
                if th.any(scale < 0).item():
                    mesh_rel_fpath = line.split("filename=")[1].split('"')[1]
                    base_fpath = f"{os.path.dirname(urdf_fpath)}/"
                    mesh_abs_fpath = f"{base_fpath}{mesh_rel_fpath}"
                    filetype = mesh_abs_fpath.split(".")[-1]
                    mesh_output_suffix = make_mesh_positive(
                        mesh_abs_fpath.split(".")[0], scale.cpu().numpy(), output_suffix
                    )
                    new_mesh_abs_fpath = mesh_abs_fpath.replace(f".{filetype}", f"_{mesh_output_suffix}.{filetype}")
                    new_mesh_rel_fpath = new_mesh_abs_fpath.split(base_fpath)[1]
                    out_line = line.replace(mesh_rel_fpath, new_mesh_rel_fpath).replace(scale_str, "1 1 1")
            out_lines.append(out_line)

    # Write to output file
    out_file = urdf_fpath.replace(".urdf", f"_{output_suffix}.urdf")
    with open(out_file, "w+") as f:
        f.writelines(out_lines)

    return out_file


def simplify_convex_hull(tm, max_vertices=60, max_faces=128):
    """
    Simplifies a convex hull mesh by using quadric edge collapse to reduce the number of faces

    Args:
        tm (Trimesh): Trimesh mesh to simply. Should be convex hull
        max_vertices (int): Maximum number of vertices to generate
    """
    # If number of faces is less than or equal to @max_faces, simply return directly
    if len(tm.vertices) <= max_vertices:
        return tm

    # Use pymeshlab to reduce
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=tm.vertices, face_matrix=tm.faces, v_normals_matrix=tm.vertex_normals))
    while len(ms.current_mesh().vertex_matrix()) > max_vertices:
        ms.apply_filter("meshing_decimation_quadric_edge_collapse", targetfacenum=max_faces)
        max_faces -= 2
    vertices_reduced = ms.current_mesh().vertex_matrix()
    faces_reduced = ms.current_mesh().face_matrix()
    vertex_normals_reduced = ms.current_mesh().vertex_normal_matrix()
    return trimesh.Trimesh(
        vertices=vertices_reduced,
        faces=faces_reduced,
        vertex_normals=vertex_normals_reduced,
    ).convex_hull


def generate_collision_meshes(
    trimesh_mesh, method="coacd", hull_count=32, discard_not_volume=True, error_handling=False
):
    """
    Generates a set of collision meshes from a trimesh mesh using CoACD.

    Args:
        trimesh_mesh (trimesh.Trimesh): The trimesh mesh to generate the collision mesh from.
        method (str): Method to generate collision meshes. Valid options are {"coacd", "convex"}
        hull_count (int): If @method="coacd", this sets the max number of hulls to generate
        discard_not_volume (bool): If @method="coacd" and set to True, this discards any generated hulls
            that are not proper volumes
        error_handling: If true, will run coacd_runner.py and handle the coacd assertion fault by using convex hull instead

    Returns:
        List[trimesh.Trimesh]: The collision meshes.
    """
    # error_handling = False
    # If the mesh is convex or the mesh is a proper volume and similar to its convex hull, simply return that directly
    if trimesh_mesh.is_convex or (
        trimesh_mesh.is_volume and (trimesh_mesh.volume / trimesh_mesh.convex_hull.volume) > 0.90
    ):
        hulls = [trimesh_mesh.convex_hull]

    elif method == "coacd":
        if error_handling:
            # Run CoACD with error handling
            import subprocess
            import sys
            import tempfile
            import pickle
            import os

            # Create separate temp files with proper extensions
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                data_path = f.name
                pickle.dump((trimesh_mesh.vertices, trimesh_mesh.faces, hull_count), f)

            script_path = tempfile.mktemp(suffix=".py")
            result_path = tempfile.mktemp(suffix=".pkl")

            # Run subprocess with clean file paths
            success = (
                subprocess.call(
                    [sys.executable, os.path.join(os.path.dirname(__file__), "_coacd_runner.py"), data_path, result_path]
                )
                == 0
            )

            # Process results or fallback
            if success and os.path.exists(result_path):
                with open(result_path, "rb") as f:
                    result = pickle.load(f)

                # Process results as before
                hulls = []
                coacd_vol = 0.0
                for vs, fs in result:
                    hull = trimesh.Trimesh(vertices=vs, faces=fs, process=False)
                    if discard_not_volume and not hull.is_volume:
                        continue
                    hulls.append(hull)
                    coacd_vol += hull.convex_hull.volume

                # Check if we found any valid hulls
                if len(hulls) == 0:
                    print("No valid collision meshes generated, falling back to convex hull")
                    hulls = [trimesh_mesh.convex_hull]
                else:
                    # Compare volume ratios as in original code
                    vol_ratio = coacd_vol / trimesh_mesh.convex_hull.volume
                    if 0.95 < vol_ratio < 1.05:
                        print("MINIMAL CHANGE -- USING CONVEX HULL INSTEAD")
                        hulls = [trimesh_mesh.convex_hull]
            else:
                print("CoACD processing failed, falling back to convex hull")
                hulls = [trimesh_mesh.convex_hull]

            # Clean up temp files
            for path in [data_path, script_path, result_path]:
                if os.path.exists(path):
                    os.remove(path)
        else:
            try:
                import coacd
            except ImportError:
                raise ImportError("Please install the `coacd` package to use this function.")

            # Get the vertices and faces
            coacd_mesh = coacd.Mesh(trimesh_mesh.vertices, trimesh_mesh.faces)

            # Run CoACD with the hull count
            result = coacd.run_coacd(
                coacd_mesh,
                max_convex_hull=hull_count,
                decimate=True,
                max_ch_vertex=60,
            )

            # Convert the returned vertices and faces to trimesh meshes
            # and assert that they are volumes (and if not, discard them if required)
            hulls = []
            coacd_vol = 0.0
            for vs, fs in result:
                hull = trimesh.Trimesh(vertices=vs, faces=fs, process=False)
                if discard_not_volume and not hull.is_volume:
                    continue
                hulls.append(hull)
                coacd_vol += hull.convex_hull.volume

            # Assert that we got _some_ collision meshes
            assert len(hulls) > 0, "No collision meshes generated!"

            # Compare coacd's generation compared to the original mesh's convex hull
            # If the difference is small (<10% volume difference), simply keep the convex hull
            vol_ratio = coacd_vol / trimesh_mesh.convex_hull.volume
            if 0.95 < vol_ratio < 1.05:
                print("MINIMAL CHANGE -- USING CONVEX HULL INSTEAD")
                hulls = [trimesh_mesh.convex_hull]

    elif method == "convex":
        hulls = [trimesh_mesh.convex_hull]

    else:
        raise ValueError(f"Invalid collision mesh generation method specified: {method}")

    # Sanity check all convex hulls
    # For whatever reason, some convex hulls are not true volumes, so we take the convex hull again
    # See https://github.com/mikedh/trimesh/issues/535
    hulls = [hull.convex_hull if not hull.is_volume else hull for hull in hulls]

    # For each hull, simplify so that the complexity is guaranteed to be Omniverse-GPU compatible
    # See https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/rigid-bodies.html#collision-settings
    simplified_hulls = [simplify_convex_hull(hull) for hull in hulls]

    return simplified_hulls


def generate_inertia_frames(urdf_path):
    # NOTE: This will override all inertias found in the URDF!

    # Open URDF
    obj = URDF.load(urdf_path)
    for link in obj.links:
        link_mass = link.inertial.mass
        assert link_mass is not None
        # Skip the link if there's no collisions
        if len(link.collisions) == 0:
            continue
        # Compile collisions to compute inertia
        link_tm_scene = trimesh.scene.Scene()
        total_local_mass = 0
        for col in link.collisions:
            col_tf = col.origin
            # Can't handle anything else other than mesh for now (not implemented),
            # so we explicitly break to avoid silent errors
            if col.geometry.box is not None:
                raise NotImplementedError("Inertia inference from collision boxes not yet implemented!")
            if col.geometry.cylinder is not None:
                raise NotImplementedError("Inertia inference from collision cylinders not yet implemented!")
            if col.geometry.sphere is not None:
                raise NotImplementedError("Inertia inference from collision spheres not yet implemented!")
            if col.geometry.mesh is not None:
                scale = col.geometry.mesh.scale
                for mesh in col.geometry.mesh.meshes:
                    # Scale this mesh and transform it
                    tmp_mesh = deepcopy(mesh)
                    tmp_mesh.apply_scale(scale)
                    tmp_mesh.apply_transform(col_tf)
                    # raw_inertia_tensor = tm_mesh.moment_inertia
                    # # Grab the inertial frame, where the inertia is diagonal
                    link_tm_scene.add_geometry(tmp_mesh)
                    total_local_mass += tmp_mesh.mass
        # If we didn't get any collisions, skip
        if len(link_tm_scene.geometry) == 0:
            continue
        # Update individual densities based on overall mass proportion
        for geo in link_tm_scene.geometry.values():
            col_mass = geo.mass
            prop = geo.mass / total_local_mass
            expected_mass = link_mass * prop
            geo.density = geo.density * expected_mass / geo.mass    # Increase density to give us the expected mass
        # Calculate inertia frame
        link_inertia_tensor = link_tm_scene.moment_inertia
        # Get the orientation corresponding to the isolated components
        components, principal_axes = trimesh.inertia.principal_axis(link_inertia_tensor)
        # Flip principal axes if not valid orientation (det == -1 instead of +1)
        if np.linalg.det(principal_axes) < 0:
            principal_axes *= -1
        # Convert orientation from matrix -> euler (rpy) form
        inertial_frame = np.eye(4)
        inertial_frame[:3, :3] = principal_axes
        inertial_frame[:3, 3] =  link_tm_scene.center_mass
        link.inertial.origin = inertial_frame
        link.inertial.inertia = np.eye(3) * components

    # Save in place
    obj.save(urdf_path)

def get_collision_approximation_for_urdf(
    urdf_path,
    collision_method="coacd",
    hull_count=32,
    coacd_links=None,
    convex_links=None,
    no_decompose_links=None,
    visual_only_links=None,
    ignore_links=None,
):
    """
    Computes collision approximation for all collision meshes (which are assumed to be non-convex) in
    the given URDF.

    NOTE: This is an in-place operation! It will overwrite @urdf_path

    Args:
        urdf_path (str): Absolute path to the URDF to decompose
        collision_method (str): Default collision method to use. Valid options are: {"coacd", "convex"}
        hull_count (int): Maximum number of convex hulls to decompose individual visual meshes into.
            Only relevant if @collision_method is "coacd"
        coacd_links (None or list of str): If specified, links that should use CoACD to decompose collision meshes
        convex_links (None or list of str): If specified, links that should use convex hull to decompose collision meshes
        no_decompose_links (None or list of str): If specified, links that should not have any special collision
            decomposition applied. This will only use the convex hull
        visual_only_links (None or list of str): If specified, link names corresponding to links that should have
            no collision associated with them (so any pre-existing collisions will be removed!)
        ignore_links (None or list of str): If specified, link names corresponding to links that should be skipped
            during collision generation process
    """
    # Load URDF
    urdf_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Next, iterate over each visual mesh and define collision meshes for them
    coacd_links = set() if coacd_links is None else set(coacd_links)
    convex_links = set() if convex_links is None else set(convex_links)
    no_decompose_links = set() if no_decompose_links is None else set(no_decompose_links)
    visual_only_links = set() if visual_only_links is None else set(visual_only_links)
    ignore_links = set() if ignore_links is None else set(ignore_links)
    col_mesh_rel_folder = "meshes/collision"
    col_mesh_folder = pathlib.Path(urdf_dir) / col_mesh_rel_folder
    col_mesh_folder.mkdir(exist_ok=True, parents=True)
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        old_cols = link.findall("collision")
        # Completely skip this link if this a link to explicitly skip or we have no collision tags
        if link_name in ignore_links or len(old_cols) == 0:
            continue

        print(f"Generating collision approximation for link {link_name}...")
        generated_new_col = False
        idx = 0
        if link_name not in visual_only_links:
            for vis in link.findall("visual"):
                # Get origin
                origin = vis.find("origin")
                # Check all geometries
                geoms = vis.findall("geometry/*")
                # We should only have a single geom, so assert here
                assert len(geoms) == 1
                # Check whether we actually need to generate a collision approximation
                # No need if the geom type is not a mesh (i.e.: it's a primitive -- so we assume if a collision is already
                # specified, it's that same primitive)
                geom = geoms[0]
                if geom.tag != "mesh":
                    continue
                mesh_path = os.path.join(os.path.dirname(urdf_path), geom.attrib["filename"])
                tm = trimesh.load(mesh_path, force="mesh", process=False)

                if link_name in coacd_links:
                    method = "coacd"
                elif link_name in convex_links:
                    method = "convex"
                elif link_name in no_decompose_links:
                    # Output will just be ignored, so skip
                    continue
                else:
                    method = collision_method
                collision_meshes = generate_collision_meshes(
                    trimesh_mesh=tm,
                    method=method,
                    hull_count=hull_count,
                )
                # Save and merge precomputed collision mesh
                collision_filenames_and_scales = []
                for i, collision_mesh in enumerate(collision_meshes):
                    processed_collision_mesh = collision_mesh.copy()
                    processed_collision_mesh._cache.cache["vertex_normals"] = processed_collision_mesh.vertex_normals
                    collision_filename = f"{link_name}_col_{idx}.obj"

                    # OmniGibson requires unit-bbox collision meshes, so here we do that scaling
                    bounding_box = processed_collision_mesh.bounding_box.extents
                    assert all(
                        x > 0 for x in bounding_box
                    ), f"Bounding box extents are not all positive: {bounding_box}"
                    collision_scale = 1.0 / bounding_box
                    collision_scale_matrix = th.eye(4)
                    collision_scale_matrix[:3, :3] = th.diag(th.as_tensor(collision_scale))
                    processed_collision_mesh.apply_transform(collision_scale_matrix.numpy())
                    processed_collision_mesh.export(col_mesh_folder / collision_filename, file_type="obj")
                    collision_filenames_and_scales.append((collision_filename, 1 / collision_scale))

                    idx += 1

                for collision_filename, collision_scale in collision_filenames_and_scales:
                    collision_xml = ET.SubElement(link, "collision")
                    collision_xml.attrib = {"name": collision_filename.replace(".obj", "")}
                    # Add origin info if defined
                    if origin is not None:
                        collision_xml.append(deepcopy(origin))
                    collision_geometry_xml = ET.SubElement(collision_xml, "geometry")
                    collision_mesh_xml = ET.SubElement(collision_geometry_xml, "mesh")
                    collision_mesh_xml.attrib = {
                        "filename": os.path.join(col_mesh_rel_folder, collision_filename),
                        "scale": " ".join([str(item) for item in collision_scale]),
                    }

                if link_name not in no_decompose_links:
                    generated_new_col = True

        # If we generated a new set of collision meshes, remove the old ones
        if generated_new_col or link_name in visual_only_links:
            for col in old_cols:
                link.remove(col)

    # Save the URDF file
    _save_xmltree_as_urdf(
        root_element=root,
        name=os.path.splitext(os.path.basename(urdf_path))[0],
        dirpath=os.path.dirname(urdf_path),
        unique_urdf=False,
    )


def copy_urdf_to_dataset(
    urdf_path,
    category,
    mdl,
    dataset_root,
    urdf_dep_paths=None,
    suffix="original",
    overwrite=False,
):
    """
    Copies a URDF file and its dependencies to a structured dataset directory.

    Parameters:
        urdf_path (str): Path to the source URDF file.
        category (str): Category name for organizing the model in the dataset.
        mdl (str): Model identifier/name.
        dataset_root (str): Root directory of the dataset.
        urdf_dep_paths (list, optional): List of relative paths to URDF dependencies.
            If None, dependencies will be automatically detected. Defaults to None.
        suffix (str, optional): Suffix to append to the model name in the new URDF.
            Defaults to "original".
        overwrite (bool, optional): Whether to overwrite existing directories.
            If False, raises an assertion error if target directory exists.
            Defaults to False.

    Returns:
        str: Path to the newly created URDF file in the dataset.

    Raises:
        AssertionError: If the target directory already exists and overwrite is False.
    """
    # Create a directory for the object
    obj_dir = pathlib.Path(dataset_root) / "objects" / category / mdl / "urdf"
    if not overwrite:
        assert not obj_dir.exists(), f"Object directory {obj_dir} already exists!"
    obj_dir.mkdir(parents=True, exist_ok=True)

    # Copy over all relevant meshes to new obj directory
    old_urdf_dir = pathlib.Path(os.path.dirname(urdf_path))

    # Load urdf
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Find all mesh paths, and replace them with new obj directory
    # urdf_dep_paths should be relative paths wrt the original URDF path
    new_dirs = set() if urdf_dep_paths is None else set(urdf_dep_paths)
    for mesh_type in ["visual", "collision"]:
        for mesh_element in root.findall(f"link/{mesh_type}/geometry/mesh"):
            mesh_root_dir = mesh_element.attrib["filename"].split("/")[0]
            new_dirs.add(mesh_root_dir)
    for new_dir in new_dirs:
        shutil.copytree(old_urdf_dir / new_dir, obj_dir / new_dir, dirs_exist_ok=overwrite)

    # Export this URDF
    return _save_xmltree_as_urdf(
        root_element=root,
        name=f"{mdl}_{suffix}",
        dirpath=obj_dir,
        unique_urdf=False,
    )


def generate_urdf_for_mesh(
    asset_path,
    obj_dir,
    category,
    mdl,
    collision_method=None,
    hull_count=32,
    up_axis="z",
    scale=1.0,
    check_scale=False,
    rescale=False,
    dataset_root=None,
    overwrite=False,
    n_submesh=10,
):
    """
    Generate URDF file for either single mesh or articulated files.
    Each submesh in articulated files (glb, gltf) will be extracted as a separate link.

    Args:
        asset_path: Path to the input mesh file (.obj, .glb, .gltf)
        obj_dir: Output directory
        category: Category name for the object
        mdl: Model name
        collision_method: Method for generating collision meshes ("convex", "coacd", or None)
        hull_count: Maximum number of convex hulls for COACD method
        up_axis: Up axis for the model ("y" or "z")
        scale: User choice scale, will be overwritten if check_scale and rescale
        check_scale: Whether to check mesh size based on heuristic
        rescale: Whether to rescale mesh if check_scale
        dataset_root: Root directory for the dataset
        overwrite: Whether to overwrite existing files
        n_submesh: If submesh number is more than n_submesh, will not convert and skip
    """

    # Validate file format
    valid_formats = trimesh.available_formats()
    mesh_format = pathlib.Path(asset_path).suffix[1:]  # Remove the dot
    assert mesh_format in valid_formats, f"Invalid mesh format: {mesh_format}. Valid formats: {valid_formats}"
    assert mesh_format in [
        "obj",
        "glb",
        "gltf",
    ], "Not obj, glb, gltf file, can only deal with these file types"
    # assert isinstance(scale, float), f"Scale must be a single float number, but got: {scale}"

    # Convert obj_dir to Path object
    if isinstance(obj_dir, str):
        obj_dir = pathlib.Path(obj_dir)

    # Create directory structure
    if not overwrite:
        assert not obj_dir.exists(), f"Object directory {obj_dir} already exists!"
    obj_dir.mkdir(parents=True, exist_ok=True)

    obj_name = "_".join([category, mdl])

    # Dictionary to store links with their visual and collision meshes
    links = {}

    # Load and process based on file type
    if mesh_format == "obj":
        # Handle single mesh files with original loading method
        visual_mesh = trimesh.load(asset_path, force="mesh", process=False)
        if isinstance(visual_mesh, list):
            visual_mesh = visual_mesh[0]  # Take first mesh if multiple

        # Generate collision meshes if requested
        collision_meshes = []
        if collision_method is not None:
            collision_meshes = generate_collision_meshes(
                visual_mesh, method=collision_method, hull_count=hull_count, error_handling=True
            )

        # Add to links dictionary as a single link named "base_link"
        links["base_link"] = {"visual_mesh": visual_mesh, "collision_meshes": collision_meshes, "transform": th.eye(4)}

    elif mesh_format in ["glb", "gltf"]:
        # Handle articulated files
        scene = trimesh.load(asset_path)
        # Count geometries (submeshes)
        submesh_count = len(scene.geometry)
        if submesh_count > n_submesh:
            print(f"❌ Submesh count: {submesh_count} > {n_submesh}, skipping")
            return None

        # Get transforms from graph and extract each geometry as a separate link
        link_index = 0
        for node_name in scene.graph.nodes_geometry:
            geometry_name = scene.graph[node_name][1]
            if not isinstance(geometry_name, str):
                print(f"Warning: Skipping node {node_name} with non-string geometry name: {geometry_name}")
                continue

            # Get the geometry and transform
            geometry = scene.geometry[geometry_name]

            transform, _ = scene.graph.get(frame_to=node_name, frame_from=scene.graph.base_frame)
            transform_tensor = th.from_numpy(transform.copy()).float()

            # Process the geometry based on its type
            if isinstance(geometry, trimesh.Trimesh):
                # Create a link name based on the node name or index
                link_name = f"link_{link_index}"
                if node_name and isinstance(node_name, str):
                    # Clean up node name to make it a valid link name
                    link_name = "link_" + "".join(c if c.isalnum() or c == "_" else "_" for c in node_name)

                # Create a copy of the geometry
                visual_mesh = geometry.copy()

                # Generate collision meshes if requested
                collision_meshes = []
                if collision_method is not None:
                    # Create collision meshes based on the original geometry
                    # (not transformed yet - we'll handle transforms at the URDF level)
                    collision_meshes = generate_collision_meshes(
                        geometry,
                        method=collision_method,
                        hull_count=hull_count,
                        discard_not_volume=True,
                        error_handling=True,
                    )

                # Add to links dictionary with original transform
                links[link_name] = {
                    "visual_mesh": visual_mesh,
                    "collision_meshes": collision_meshes,
                    "transform": transform_tensor,
                    "node_name": node_name,
                }
                link_index += 1

            elif isinstance(geometry, (list, tuple)):
                # Handle cases where geometry is a list of meshes
                for i, submesh in enumerate(geometry):
                    if isinstance(submesh, trimesh.Trimesh):
                        # Create a link name
                        link_name = f"link_{link_index}"
                        if node_name and isinstance(node_name, str):
                            link_name = f"link_{node_name}_{i}"

                        # Create a copy of the submesh
                        visual_mesh = submesh.copy()

                        # Generate collision meshes if requested
                        collision_meshes = []

                        if collision_method is not None:
                            # Create collision meshes based on the original geometry
                            collision_meshes = generate_collision_meshes(
                                submesh,
                                method=collision_method,
                                hull_count=hull_count,
                                discard_not_volume=True,
                                error_handling=True,
                            )

                        # Add to links dictionary with original transform
                        links[link_name] = {
                            "visual_mesh": visual_mesh,
                            "collision_meshes": collision_meshes,
                            "transform": transform_tensor,
                            "node_name": f"{node_name}_{i}",
                        }
                        link_index += 1

        if not links:
            print("Warning: No valid meshes found in the scene!")
            print("Scene contents:")
            print(f"Geometries: {scene.geometry}")
            print(f"Graph: {scene.graph}")
            raise ValueError("No valid meshes found in the input file")
    else:
        raise ValueError(f"Unsupported file format: {mesh_format}")

    # Handle rotation for up_axis if needed
    if up_axis == "y":
        rotation_matrix = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
        rotation_tensor = th.from_numpy(rotation_matrix).float()

        for link_name, link_data in links.items():
            # Update the transform - we'll apply the actual transforms later
            link_data["transform"] = th.matmul(rotation_tensor, link_data["transform"])

    # Compute new scale if check_scale = True
    new_scale = 1.0
    if check_scale:
        if links:
            # Find the link with the biggest bounding box
            max_bbox_size = [0, 0, 0]
            max_bbox_link = None

            for link_name, link_data in links.items():
                # Apply the transform to get the correct size
                temp_mesh = link_data["visual_mesh"].copy()
                temp_mesh.apply_transform(link_data["transform"].numpy())
                bbox_size = temp_mesh.bounding_box.extents

                # Check if this link has a bigger dimension than the current max
                if any(s > max_s for s, max_s in zip(bbox_size, max_bbox_size)):
                    max_bbox_size = bbox_size
                    max_bbox_link = link_name

            click.echo(f"Largest visual mesh bounding box size: {max_bbox_size} (link: {max_bbox_link})")

            # Check if any dimension is too large (> 100)
            if any(size > 5.0 for size in max_bbox_size):
                if any(size > 50.0 for size in max_bbox_size):
                    if any(size > 500.0 for size in max_bbox_size):
                        new_scale = 0.001
                    else:
                        new_scale = 0.01
                else:
                    new_scale = 0.1

                click.echo(
                    "Warning: The bounding box sounds a bit large. "
                    "We just wanted to confirm this is intentional. You can skip this check by passing check_scale = False."
                )

            # Check if any dimension is too small (< 0.01)
            elif all(size < 0.005 for size in max_bbox_size):
                new_scale = 1000.0
                click.echo(
                    "Warning: The bounding box sounds a bit small. "
                    "We just wanted to confirm this is intentional. You can skip this check by passing check_scale = False."
                )

            else:
                click.echo("Size is reasonable, no scaling")

        else:
            click.echo("Warning: No links found in the file!")
            return None

    # Rescale mesh if rescale= True, else scale based on function input scale
    if rescale:
        click.echo(f"Original scale {scale} be overwrtten to {new_scale}")
        scale = new_scale

    if scale != 1.0 and np.any(np.array(scale) != 1.0):
        click.echo(f"Adjusting scale to {scale}")
        scale_transform = np.eye(4)
        scale_transform[:3, :3] *= scale
        scale_tensor = th.from_numpy(scale_transform).float()

        for link_name, link_data in links.items():
            # Update the transform - we'll apply the actual transforms later
            link_data["transform"] = th.matmul(scale_tensor, link_data["transform"])

    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)

        # Create directory structure for the output
        obj_link_mesh_folder = obj_dir / "shape"
        obj_link_mesh_folder.mkdir(exist_ok=True)
        obj_link_visual_mesh_folder = obj_link_mesh_folder / "visual"
        obj_link_visual_mesh_folder.mkdir(exist_ok=True)
        obj_link_collision_mesh_folder = obj_link_mesh_folder / "collision"
        obj_link_collision_mesh_folder.mkdir(exist_ok=True)
        obj_link_material_folder = obj_dir / "material"
        obj_link_material_folder.mkdir(exist_ok=True)

        # Dictionary to store information for URDF generation
        urdf_links = {}

        # Process each link
        for link_name, link_data in links.items():
            visual_mesh = link_data["visual_mesh"].copy()  # Create a copy to avoid modifying original
            collision_meshes = [mesh.copy() for mesh in link_data["collision_meshes"]]  # Copy all collision meshes
            transform = link_data["transform"]

            # Apply transform to visual mesh before exporting
            visual_mesh.apply_transform(transform.numpy())

            # Export the transformed mesh
            visual_filename = f"{obj_name}_{link_name}.obj"
            visual_temp_path = temp_dir_path / visual_filename
            visual_mesh.export(visual_temp_path, file_type="obj")

            # Check for material files
            material_files = [x for x in temp_dir_path.iterdir() if x.suffix == ".mtl"]
            material_filename = None

            if material_files:
                # Process material file if exists
                material_file = material_files[0]
                material_filename = f"{obj_name}_{link_name}.mtl"

                # Process MTL file (similar to original code)
                with open(visual_temp_path, "r") as f:
                    new_lines = []
                    for line in f.readlines():
                        if f"mtllib {material_file.name}" in line:
                            line = f"mtllib {material_filename}\n"
                        new_lines.append(line)

                with open(visual_temp_path, "w") as f:
                    for line in new_lines:
                        f.write(line)

                # Process texture references in MTL file
                with open(material_file, "r") as f:
                    new_lines = []
                    for line in f.readlines():
                        if "map_" in line:
                            parts = line.split(" ", 1)
                            if len(parts) > 1:
                                map_kind, texture_filename = parts
                                texture_filename = texture_filename.strip()
                                map_kind = map_kind.strip().replace("map_", "")
                                new_filename = f"../../material/{obj_name}_{link_name}_{map_kind}.png"

                                # Copy texture file
                                texture_from_path = temp_dir_path / texture_filename
                                if texture_from_path.exists():
                                    texture_to_path = (
                                        obj_link_material_folder / f"{obj_name}_{link_name}_{map_kind}.png"
                                    )
                                    if not overwrite and texture_to_path.exists():
                                        print(f"Warning: Texture file {texture_to_path} already exists!")
                                    else:
                                        shutil.copy2(texture_from_path, texture_to_path)

                                # Update line
                                line = f"{parts[0]} {new_filename}\n"
                        new_lines.append(line)

                # Write updated MTL file
                with open(obj_link_visual_mesh_folder / material_filename, "w") as f:
                    for line in new_lines:
                        f.write(line)

            # Copy visual mesh to final location
            visual_final_path = obj_link_visual_mesh_folder / visual_filename
            shutil.copy2(visual_temp_path, visual_final_path)

            # Process collision meshes
            collision_info = []
            for i, collision_mesh in enumerate(collision_meshes):
                # Apply transform to collision mesh before exporting
                collision_mesh.apply_transform(transform.numpy())

                # Export collision mesh filename
                collision_filename = visual_filename.replace(".obj", f"_collision_{i}.obj")

                # Scale collision mesh to unit bbox if needed
                bounding_box = collision_mesh.bounding_box.extents
                if all(x > 0 for x in bounding_box):
                    collision_scale = 1.0 / bounding_box
                    collision_scale_matrix = th.eye(4)
                    collision_scale_matrix[:3, :3] = th.diag(th.as_tensor(collision_scale))

                    # Create a copy to avoid modifying the original
                    scaled_collision_mesh = collision_mesh.copy()
                    scaled_collision_mesh.apply_transform(collision_scale_matrix.numpy())

                    # Export collision mesh
                    collision_path = obj_link_collision_mesh_folder / collision_filename
                    scaled_collision_mesh.export(collision_path, file_type="obj")

                    # Since we've already applied the transform, scale includes only the sizing adjustment
                    collision_info.append({"filename": collision_filename, "scale": 1.0 / collision_scale})
                else:
                    print(f"Warning: Skipping collision mesh with invalid bounding box: {bounding_box}")

            # Store information for URDF generation - now without transform since it's been applied
            urdf_links[link_name] = {
                "visual_filename": visual_filename,
                "collision_info": collision_info,
                "transform": th.eye(4),  # Identity transform since we've already applied it to the meshes
            }

    if mesh_format == "obj":
        # Change the link name from "base_link" to "obj_link"
        if "base_link" in urdf_links:
            urdf_links["obj_link"] = urdf_links.pop("base_link")

    # Generate URDF XML
    tree_root = ET.Element("robot")
    tree_root.attrib = {"name": mdl}

    # # Create a base_link as the root
    # base_link = ET.SubElement(tree_root, "link")
    # base_link.attrib = {"name": "base_link"}

    # Add all other links and joints to connect them to the base_link
    for link_name, link_info in urdf_links.items():
        # Create link element
        link_xml = ET.SubElement(tree_root, "link")
        link_xml.attrib = {"name": link_name}

        # Add visual geometry
        visual_xml = ET.SubElement(link_xml, "visual")
        visual_origin_xml = ET.SubElement(visual_xml, "origin")
        visual_origin_xml.attrib = {"xyz": "0 0 0", "rpy": "0 0 0"}  # Zero transform since already applied
        visual_geometry_xml = ET.SubElement(visual_xml, "geometry")
        visual_mesh_xml = ET.SubElement(visual_geometry_xml, "mesh")
        visual_mesh_xml.attrib = {
            "filename": os.path.join("shape", "visual", link_info["visual_filename"]).replace("\\", "/"),
            "scale": "1 1 1",  # Using 1.0 scale since transform already applied
        }

        # Add collision geometries
        for i, collision in enumerate(link_info["collision_info"]):
            collision_xml = ET.SubElement(link_xml, "collision")
            collision_xml.attrib = {"name": f"{link_name}_collision_{i}"}
            collision_origin_xml = ET.SubElement(collision_xml, "origin")
            collision_origin_xml.attrib = {"xyz": "0 0 0", "rpy": "0 0 0"}  # Zero transform since already applied
            collision_geometry_xml = ET.SubElement(collision_xml, "geometry")
            collision_mesh_xml = ET.SubElement(collision_geometry_xml, "mesh")
            collision_mesh_xml.attrib = {
                "filename": os.path.join("shape", "collision", collision["filename"]).replace("\\", "/"),
                "scale": " ".join(str(item) for item in collision["scale"]),
            }

        # # Create a joint to connect this link to the base_link
        # joint_xml = ET.SubElement(tree_root, "joint")
        # joint_xml.attrib = {"name": f"{link_name}_joint", "type": "fixed"}
        #
        # # Set parent and child links
        # parent_xml = ET.SubElement(joint_xml, "parent")
        # parent_xml.attrib = {"link": "base_link"}
        # child_xml = ET.SubElement(joint_xml, "child")
        # child_xml.attrib = {"link": link_name}
        #
        # # Set origin for the joint with zeros since transform was applied to meshes
        # joint_origin_xml = ET.SubElement(joint_xml, "origin")
        # joint_origin_xml.attrib = {"xyz": "0 0 0", "rpy": "0 0 0"}

    # Save URDF file
    xmlstr = minidom.parseString(ET.tostring(tree_root)).toprettyxml(indent="   ")
    xmlio = io.StringIO(xmlstr)
    tree = ET.parse(xmlio)

    urdf_path = obj_dir / f"{mdl}.urdf"
    with open(urdf_path, "wb") as f:
        tree.write(f, xml_declaration=True)

    return str(urdf_path)


def record_obj_metadata_from_urdf(urdf_path, obj_dir, joint_setting="zero", openable_joint_ids=None, overwrite=False):
    """
    Records object metadata and writes it to misc/metadata.json within the object directory.

    Args:
        urdf_path (str): Path to object URDF
        obj_dir (str): Absolute path to the object's root directory
        joint_setting (str): Setting for joints when calculating canonical metadata. Valid options
            are {"low", "zero", "high"} (i.e.: lower joint limit, all 0 values, or upper joint limit)
        openable_joint_ids (list or None): Optional list of joint metadata in BEHAVIOR-1K format.
            Each element is [joint_index, joint_name, direction] where direction is 1 for positive 
            opening (increasing angle = open) or -1 for negative opening (decreasing angle = open).
            Format: [[0, "joint_name", -1], [1, "another_joint", 1]]
            If not provided in metadata.json, will use empty list.
        overwrite (bool): Whether to overwrite any pre-existing data
    """
    # Load the URDF file into urdfpy
    robot = URDF.load(urdf_path)

    # Do FK with everything at desired configuration
    if joint_setting == "zero":
        val = lambda jnt: 0.0
    elif joint_setting == "low":
        val = lambda jnt: jnt.limit.lower
    elif joint_setting == "high":
        val = lambda jnt: jnt.limit.upper
    else:
        raise ValueError(f"Got invalid joint_setting: {joint_setting}! Valid options are ['low', 'zero', 'high']")
    joint_cfg = {joint.name: val(joint) for joint in robot.joints if joint.joint_type in ("prismatic", "revolute")}
    vfk = robot.visual_trimesh_fk(cfg=joint_cfg)

    scene = trimesh.Scene()
    for mesh, transform in vfk.items():
        scene.add_geometry(geometry=mesh, transform=transform)

    # Calculate relevant metadata

    # Base link offset is pos offset from robot root link -> overall AABB center
    # Since robot is placed at origin, this is simply the AABB centroid
    base_link_offset = scene.bounding_box.centroid

    # BBox size is simply the extent of the overall AABB
    bbox_size = scene.bounding_box.extents

    # Save metadata json
    out_metadata = {
        "meta_links": {},
        "link_tags": {},
        "object_parts": [],
        "base_link_offset": base_link_offset.tolist(),
        "bbox_size": bbox_size.tolist(),
        "orientations": [],
    }
    
    # Add openable joint IDs if provided
    if openable_joint_ids is not None:
        out_metadata["openable_joint_ids"] = openable_joint_ids
    
    misc_dir = pathlib.Path(obj_dir) / "misc"
    misc_dir.mkdir(exist_ok=overwrite)
    with open(misc_dir / "metadata.json", "w") as f:
        json.dump(out_metadata, f)


def process_urdf(
    category,
    model,
    dataset_root,
    urdf_path=None,
    urdf_dep_paths=None,
    collision_method="coacd",
    coacd_links=None,
    convex_links=None,
    no_decompose_links=None,
    visual_only_links=None,
    split_collision_meshes=False,
    hull_count=32,
    openable_joint_ids=None,
    overwrite=False,
):
    """
    Imports an asset from URDF format into OmniGibson-compatible USD format. This will write the new USD
    (and copy the URDF if it does not already exist within @dataset_root) to @dataset_root

    Args:
        category (str): Category to assign to imported asset
        model (str): Model name to assign to imported asset
        dataset_root (str): Path to dataset to write to
        urdf_path (None or str): If specified, external URDF that should be copied into the dataset first before
            converting into USD format. Otherwise, assumes that the urdf file already exists within @dataset_root dir
        urdf_dep_paths (None or list of str): If specified, relative paths to the @urdf_path directory that should be copied
            over to the custom dataset, e.g., relevant material directories
        collision_method (None or str): If specified, collision decomposition method to use to generate
            OmniGibson-compatible collision meshes. Valid options are {"coacd", "convex"}
        coacd_links (None or list of str): If specified, links that should use CoACD to decompose collision meshes
        convex_links (None or list of str): If specified, links that should use convex hull to decompose collision meshes
        no_decompose_links (None or list of str): If specified, links that should not have any special collision
            decomposition applied. This will only use the convex hull
        visual_only_links (None or list of str): If specified, links that should have no colliders associated with it
        split_collision_meshes (bool): Whether to split collision meshes into individual submeshes or not
        hull_count (int): Maximum number of convex hulls to decompose individual visual meshes into.
            Only relevant if @collision_method is "coacd"
        overwrite (bool): If set, will overwrite any pre-existing files

    Returns:
        str: Absolute path to post-processed URDF file
    """
    # If URDF already exists, write it to the dataset
    if urdf_path is not None:
        print(f"Copying URDF to dataset root {dataset_root}...")
        urdf_path = copy_urdf_to_dataset(
            urdf_path=urdf_path,
            category=category,
            mdl=model,
            urdf_dep_paths=urdf_dep_paths,
            dataset_root=dataset_root,
            suffix="original",
            overwrite=overwrite,
        )
    else:
        # Verify that the object exists at the expected location
        # This is <dataset_root>/objects/<category>/<model>/urdf/<model>_original.urdf
        urdf_path = os.path.join(dataset_root, "objects", category, model, "urdf", f"{model}_original.urdf")
        assert os.path.exists(urdf_path), f"Expected urdf at dataset location {urdf_path}, but none was found!"

    # Make sure all scaling is positive
    model_dir = os.path.join(dataset_root, "objects", category, model)
    urdf_path = make_asset_positive(urdf_fpath=urdf_path)

    # Update collisions if requested
    if collision_method is not None:
        print("Generating collision approximation for URDF...")
        get_collision_approximation_for_urdf(
            urdf_path=urdf_path,
            collision_method=collision_method,
            hull_count=hull_count,
            coacd_links=coacd_links,
            convex_links=convex_links,
            no_decompose_links=no_decompose_links,
            visual_only_links=visual_only_links,
        )

    # Generate metadata
    print("Recording object metadata from URDF...")
    record_obj_metadata_from_urdf(
        urdf_path=urdf_path,
        obj_dir=model_dir,
        joint_setting="zero",
        openable_joint_ids=openable_joint_ids,
        overwrite=overwrite,
    )

    # Split collision meshes if requested
    if split_collision_meshes:
        print(f"Converting collision meshes from {category}, {model}...")
        urdf_path = _split_all_objs_in_urdf(urdf_fpath=urdf_path, name_suffix="split")

    print(f"\nProcessing URDF complete! Final URDF located at:\n\n{urdf_path}\n")

    return urdf_path



def import_custom_object(
    asset_path: str,
    category: str,
    model: str,
    dataset_root: str,
    collision_method: Literal["coacd", "convex", "none"],
    hull_count: int,
    up_axis: Literal["z", "y"],
    scale: Union[np.ndarray, int],
    check_scale: bool,
    rescale: bool,
    overwrite: bool,
    n_submesh: int,
    mass: Optional[float] = None,
):
    """
    Imports a custom-defined object asset into an OmniGibson-compatible USD format and saves the imported asset
    files to the custom dataset directory (gm.CUSTOM_DATASET_PATH)
    """

    assert len(model) == 6 and model.isalpha(), "Model name must be 6 characters long and contain only letters."
    collision_method = None if collision_method == "none" else collision_method

    # Sanity check mesh type
    mesh_format = asset_path.split(".")[-1]

    # If we're not a URDF, import the mesh directly first
    urdf_dep_paths = None
    temp_dirs = []
    if mesh_format != "urdf":
        temp_urdf_dir = tempfile.mkdtemp()
        temp_dirs.append(temp_urdf_dir)

        # Try to generate URDF, may raise ValueError if too many submeshes
        urdf_path = generate_urdf_for_mesh(
            asset_path,
            temp_urdf_dir,
            category,
            model,
            collision_method,
            hull_count,
            up_axis,
            scale=scale,
            check_scale=check_scale,
            rescale=rescale,
            overwrite=overwrite,
            n_submesh=n_submesh,
        )
        if urdf_path is not None:
            click.echo("URDF generation complete!")
            urdf_dep_paths = ["material"]
            # Collision was already decomposed, so no need to repeat the process again
            collision_method = None
        else:
            # Clean up temp directories before exiting
            for tmp_dir in temp_dirs:
                shutil.rmtree(tmp_dir)
            click.echo("Error during URDF generation")
            raise RuntimeError
    else:
        urdf_path = asset_path

    try:
        # Process URDF
        urdf_path = process_urdf(
            category=category,
            model=model,
            dataset_root=dataset_root,
            urdf_path=urdf_path,
            urdf_dep_paths=urdf_dep_paths,
            collision_method=collision_method,
            split_collision_meshes=False,
            hull_count=hull_count,
            overwrite=overwrite,
        )

        # If the final urdf path is "_original_mirror.urdf", copy to simply ".urdf"
        if urdf_path.endswith("_original_mirror.urdf"):
            new_urdf_path = urdf_path.replace("_original_mirror.urdf", ".urdf")
            shutil.copy2(urdf_path, new_urdf_path)
            print(f"Copied final URDF at {urdf_path} to {new_urdf_path}")
            urdf_path = new_urdf_path

        # Potentially annotate the URDF with mass and friction
        if mass is not None:
            # Load URDF
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            for link in root.findall("link"):
                # Check if link has any collision bodies, if so, add mass / friction
                collision_xmls = link.findall("collision")
                if len(collision_xmls) > 0:
                    # Check for inertia flag
                    inertials = link.findall("inertial")
                    if len(inertials) == 0:
                        # Create the new tag
                        inertial_xml = ET.SubElement(link, "inertial")
                    else:
                        inertial_xml = inertials[0]
                    # Add mass
                    mass_xml = ET.SubElement(inertial_xml, "mass")
                    mass_xml.attrib["value"] = str(mass)
                    # Add origin
                    origin_xml = ET.SubElement(inertial_xml, "origin")
                    origin_xml.attrib["xyz"] = "0 0 0"
                    origin_xml.attrib["rpy"] = "0 0 0"
                    # Add inertia
                    inertia_xml = ET.SubElement(inertial_xml, "inertia")
                    inertia_xml.attrib["ixx"] = "1.0"
                    inertia_xml.attrib["ixy"] = "0.0"
                    inertia_xml.attrib["ixz"] = "0.0"
                    inertia_xml.attrib["iyy"] = "1.0"
                    inertia_xml.attrib["iyz"] = "0.0"
                    inertia_xml.attrib["izz"] = "1.0"

            _save_xmltree_as_urdf(
                root_element=root,
                name=os.path.splitext(os.path.basename(urdf_path))[0],
                dirpath=os.path.dirname(urdf_path),
                unique_urdf=False,
            )

            # Compute proper inertias; update in place
            generate_inertia_frames(urdf_path=urdf_path)

            print(f"Computing inertias for given mass: {mass} kg")

    except Exception as e:
        click.echo(f"Error during URDF conversion: {str(e)}")
        # Clean up temp directories before exiting
        for tmp_dir in temp_dirs:
            shutil.rmtree(tmp_dir)
        raise click.Abort()

    # Clean up temp directories
    for tmp_dir in temp_dirs:
        shutil.rmtree(tmp_dir)


def import_articulated_object(
    urdf_path: str,
    mesh_parts_dir: str,
    parts_properties: list,
    category: str,
    model: str,
    dataset_root: str,
    scale: float = 1.0,
    collision_method: Literal["coacd", "convex", "none"] = "coacd",
    hull_count: int = 32,
    up_axis: Literal["z", "y"] = "z",
    apply_base_rotation: bool = True,
    overwrite: bool = True,
) -> str:
    """
    Imports an articulated object from an existing URDF (e.g., from articulation pipeline)
    by adding physical properties and generating collision meshes.
    
    Args:
        urdf_path: Path to existing URDF from articulation step (e.g., mobility.urdf)
        mesh_parts_dir: Directory containing the mesh parts referenced by the URDF
        parts_properties: List of dicts with per-part properties:
            [{"name": "link_name", "mass_kg": 1.0, "friction": 0.5, "joint_damping": 0.1}, ...]
        category: Category name for the object
        model: Model identifier (6 lowercase letters)
        dataset_root: Root directory of the dataset
        scale: Scale factor to apply to meshes
        collision_method: Method for generating collision meshes ("coacd", "convex", or "none")
        hull_count: Maximum number of convex hulls for COACD
        up_axis: Up axis for the input model ("y" or "z"). If "y", will rotate to Z-up.
        apply_base_rotation: Whether to apply the rotation from the dummy base joint to meshes
            after removing it. The base link is always removed; this controls whether its
            rotation is baked into the meshes and joint origins.

        overwrite: Whether to overwrite existing files
    
    Returns:
        Path to the processed URDF in the dataset
    """
    assert len(model) == 6 and model.isalpha(), "Model name must be 6 characters long and contain only letters."
    collision_method_actual = None if collision_method == "none" else collision_method
    
   
    if up_axis == "y":
        y_to_z_rotation = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
    else:
        y_to_z_rotation = None
    
   
    obj_dir = pathlib.Path(dataset_root) / "objects" / category / model
    urdf_dir = obj_dir / "urdf"
    urdf_dir.mkdir(parents=True, exist_ok=True)
    
    # Mesh dirs
    shape_dir = urdf_dir / "shape"
    visual_dir = shape_dir / "visual"
    collision_dir = shape_dir / "collision"
    visual_dir.mkdir(parents=True, exist_ok=True)
    collision_dir.mkdir(parents=True, exist_ok=True)
    

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    props_by_name = {p['name']: p for p in parts_properties}
    
    # Track processed links for joint dynamics
    link_to_props = {}
    
    # Handle the dummy "base" world frame link if present (artifact of the articulation pipeline)
    base_link = None
    base_to_root_rotation = None 
    
    for link in root.findall("link"):
        if link.attrib["name"] == "base":
            # Verify it's empty (no visuals)
            if link.find(".//visual") is None:
                base_link = link
                break
    
    if base_link is not None:
        root.remove(base_link)
        print("Removed empty 'base' world frame link")
        
        # Remove the single joint from base to the actual root link
        # Optionally capture its rotation to apply to meshes
        for joint in root.findall("joint"):
            parent_elem = joint.find("parent")
            if parent_elem is not None and parent_elem.attrib.get("link") == "base":
                child_elem = joint.find("child")
                child_name = child_elem.attrib.get("link", "unknown") if child_elem is not None else "unknown"
                
                # Extract rotation from joint origin (only if apply_base_rotation is True)
                if apply_base_rotation:
                    joint_origin = joint.find("origin")
                    if joint_origin is not None:
                        rpy_str = joint_origin.attrib.get("rpy", "0 0 0")
                        rpy_values = [float(v) for v in rpy_str.split()]
                        if any(abs(v) > 1e-6 for v in rpy_values):
                            # Build rotation matrix from RPY (roll, pitch, yaw)
                            from scipy.spatial.transform import Rotation as R
                            base_to_root_rotation = R.from_euler('xyz', rpy_values).as_matrix()
                            # Convert to 4x4 transformation matrix for trimesh
                            base_to_root_tf = np.eye(4)
                            base_to_root_tf[:3, :3] = base_to_root_rotation
                            base_to_root_rotation = base_to_root_tf
                            print(f"Captured rotation from base joint: rpy={rpy_values}")
                else:
                    print("Skipping base joint rotation (apply_base_rotation=False)")
                
                root.remove(joint)
                print(f"Removed fixed joint from 'base' to '{child_name}' (now root link)")
                break
    
    # Second pass: process remaining links with visuals
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        
        # Find first visual element to get both mesh path and origin
        first_visual = link.find(".//visual")
        if first_visual is None:
            continue
        visual_elem = first_visual.find(".//geometry/mesh")
        if visual_elem is None:
            continue
        
        # Capture original visual origin BEFORE removing visual elements
        original_origin = first_visual.find("origin")
        if original_origin is not None:
            original_xyz = original_origin.attrib.get("xyz", "0 0 0")
            original_rpy = original_origin.attrib.get("rpy", "0 0 0")
        else:
            original_xyz = "0 0 0"
            original_rpy = "0 0 0"
        
        # Parse xyz coordinates for potential scaling
        xyz_values = [float(v) for v in original_xyz.split()]
        
        # Link properties
        props = props_by_name.get(link_name, {})
        mass = props.get("mass_kg", 1.0)  
        link_to_props[link_name] = props
        
        # Get original mesh path
        original_mesh_path = visual_elem.attrib.get("filename", "")
        if not os.path.isabs(original_mesh_path):
            # Try to resolve relative to mesh_parts_dir
            mesh_path = os.path.join(mesh_parts_dir, os.path.basename(original_mesh_path))
        else:
            mesh_path = original_mesh_path
        
        if not os.path.exists(mesh_path):
            print(f"Warning: Mesh not found for link {link_name}: {mesh_path}")
            continue
        
        # Remove all existing visual elements (there may be multiple from the original URDF)
        for old_visual in link.findall("visual"):
            link.remove(old_visual)

        tm = trimesh.load(mesh_path, force="mesh", process=True)
        if isinstance(tm, trimesh.Scene):
            tm = tm.dump(concatenate=True) if hasattr(tm, 'dump') else tm.to_geometry()
        

        if base_to_root_rotation is not None:
            tm.apply_transform(base_to_root_rotation)
            # Rotate the origin xyz coordinates using the same rotation
            xyz_arr = np.array(xyz_values)
            xyz_arr = base_to_root_rotation[:3, :3] @ xyz_arr
            xyz_values = xyz_arr.tolist()
        # Fallback: Apply explicit Y-up to Z-up rotation if specified and no base rotation was found
        elif y_to_z_rotation is not None:
            tm.apply_transform(y_to_z_rotation)
            # Rotate origin coordinates for Y-to-Z rotation: (x, y, z) -> (x, -z, y)
            xyz_values = [xyz_values[0], -xyz_values[2], xyz_values[1]]
  
        if scale != 1.0:
            tm.apply_scale(scale)
            # Scale the origin xyz coordinates
            xyz_values = [v * scale for v in xyz_values]
        

        scaled_xyz = " ".join(str(v) for v in xyz_values)

        visual_filename = f"{link_name}.obj"
        visual_out_path = visual_dir / visual_filename
        tm.export(str(visual_out_path), file_type="obj")
        
        # Create new visual element with correct path and origin
        new_visual = ET.SubElement(link, "visual")
        new_geometry = ET.SubElement(new_visual, "geometry")
        new_mesh = ET.SubElement(new_geometry, "mesh")
        new_mesh.attrib["filename"] = f"shape/visual/{visual_filename}"
        new_mesh.attrib["scale"] = "1 1 1"
        new_origin = ET.SubElement(new_visual, "origin")
        new_origin.attrib["rpy"] = original_rpy  # Preserve original rotation
        new_origin.attrib["xyz"] = scaled_xyz    # Use scaled position

        # Remove all existing collision elements
        for old_col in link.findall("collision"):
            link.remove(old_col)
        
        # Generate collision meshes
        if collision_method_actual is not None:
            collision_meshes = generate_collision_meshes(
                trimesh_mesh=tm,
                method=collision_method_actual,
                hull_count=hull_count,
                error_handling=True,
            )
            
            # Add collision elements
            for i, col_mesh in enumerate(collision_meshes):
                # Scale collision mesh to unit bbox (OmniGibson requirement)
                bounding_box = col_mesh.bounding_box.extents
                if all(x > 0 for x in bounding_box):
                    collision_scale = 1.0 / bounding_box
                    scaled_col_mesh = col_mesh.copy()
                    scale_matrix = np.eye(4)
                    scale_matrix[:3, :3] *= collision_scale
                    scaled_col_mesh.apply_transform(scale_matrix)
                    
                    # Export collision mesh
                    col_filename = f"{link_name}_col_{i}.obj"
                    col_out_path = collision_dir / col_filename
                    scaled_col_mesh.export(str(col_out_path), file_type="obj")
                    
                    # Create collision element in URDF
                    collision_xml = ET.SubElement(link, "collision")
                    collision_xml.attrib["name"] = f"{link_name}_collision_{i}"
                   
                    visual_origin = link.find(".//visual/origin")
                    if visual_origin is not None:
                        col_origin = ET.SubElement(collision_xml, "origin")
                        col_origin.attrib = dict(visual_origin.attrib)
                    else:
                        col_origin = ET.SubElement(collision_xml, "origin")
                        col_origin.attrib = {"xyz": "0 0 0", "rpy": "0 0 0"}
                    
                    col_geom = ET.SubElement(collision_xml, "geometry")
                    col_mesh_elem = ET.SubElement(col_geom, "mesh")
                    col_mesh_elem.attrib = {
                        "filename": f"shape/collision/{col_filename}",
                        "scale": " ".join(str(1.0 / s) for s in collision_scale),
                    }
        
        #  inertial element
        inertial = link.find("inertial")
        if inertial is not None:
            link.remove(inertial)
        
        inertial = ET.SubElement(link, "inertial")
        
        mass_elem = ET.SubElement(inertial, "mass")
        mass_elem.attrib["value"] = str(mass)
        
        
        if tm.is_volume:
            tm.density = mass / tm.volume if tm.volume > 0 else 1000.0
            com = tm.center_mass
            inertia_tensor = tm.moment_inertia
        else:
            # Fallback for non-volume meshes
            com = tm.centroid
            # Use bounding box approximation for inertia
            extents = tm.bounding_box.extents
            # Approximate as solid box
            ixx = (mass / 12.0) * (extents[1]**2 + extents[2]**2)
            iyy = (mass / 12.0) * (extents[0]**2 + extents[2]**2)
            izz = (mass / 12.0) * (extents[0]**2 + extents[1]**2)
            inertia_tensor = np.diag([ixx, iyy, izz])
        
        # Add the visual origin offset to the center of mass
        com_with_offset = np.array(com) + np.array(xyz_values)
        
        origin_elem = ET.SubElement(inertial, "origin")
        origin_elem.attrib = {
            "xyz": f"{com_with_offset[0]} {com_with_offset[1]} {com_with_offset[2]}",
            "rpy": "0 0 0"
        }
        
        # Add inertia tensor
        inertia_elem = ET.SubElement(inertial, "inertia")
        inertia_elem.attrib = {
            "ixx": str(inertia_tensor[0, 0]),
            "ixy": str(inertia_tensor[0, 1]) if inertia_tensor.shape == (3, 3) else "0",
            "ixz": str(inertia_tensor[0, 2]) if inertia_tensor.shape == (3, 3) else "0",
            "iyy": str(inertia_tensor[1, 1]),
            "iyz": str(inertia_tensor[1, 2]) if inertia_tensor.shape == (3, 3) else "0",
            "izz": str(inertia_tensor[2, 2]),
        }
    
    # Add dynamics, rotate and scale joint origins, also build openable_joint_ids

    openable_joint_ids = []
    joint_idx = 0
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type", "fixed")
        
        # Build openable_joint_ids for revolute/prismatic joints
        if joint_type in ["revolute", "prismatic"]:
            joint_name = joint.attrib.get("name", "")
            
            # Determine opening direction from joint limits
            limit_elem = joint.find("limit")
            if limit_elem is not None:
                lower = float(limit_elem.attrib.get("lower", "0"))
                upper = float(limit_elem.attrib.get("upper", "0"))
                
                # Heuristic: If lower is closer to 0, positive direction is opening
                # If upper is closer to 0, negative direction is opening
                if abs(lower) < abs(upper):
                    direction = 1  # Positive = opening
                else:
                    direction = -1  # Negative = opening
            else:
                direction = 1  # Default 
                
            # Use list format expected by OmniGibson's transformation
            openable_joint_ids.append([joint_idx, joint_name, direction])
            joint_idx += 1
    
    print(f"Auto-detected openable joints: {openable_joint_ids}")
    
    # Process all joints for origins, axes, and dynamics
    # When we remove the base joint, its rotation must be applied to all joint origins and axes
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type", "fixed")
        
        # Only apply scale to joint origin xyz (no rotation - it's a relative transform)  
        joint_origin = joint.find("origin")
        if joint_origin is not None:
            origin_xyz = joint_origin.attrib.get("xyz", "0 0 0")
            xyz_values = [float(v) for v in origin_xyz.split()]
            
            # Apply rotation from removed base joint
            if base_to_root_rotation is not None:
                xyz_arr = np.array(xyz_values)
                xyz_arr = base_to_root_rotation[:3, :3] @ xyz_arr
                xyz_values = xyz_arr.tolist()
            
            # Apply scale
            if scale != 1.0:
                xyz_values = [v * scale for v in xyz_values]
            
            joint_origin.attrib["xyz"] = " ".join(str(v) for v in xyz_values)
        
        # Transform joint axis
        joint_axis = joint.find("axis")
        if joint_axis is not None and base_to_root_rotation is not None:
            axis_xyz = joint_axis.attrib.get("xyz", "0 0 1")
            axis_values = [float(v) for v in axis_xyz.split()]
            # Apply the same rotation to the axis
            axis_arr = np.array(axis_values)
            axis_arr = base_to_root_rotation[:3, :3] @ axis_arr
            axis_values = axis_arr.tolist()
            joint_axis.attrib["xyz"] = " ".join(str(v) for v in axis_values)
        
        if joint_type == "fixed":
            continue
        
        # Get child link to find properties
        child_elem = joint.find("child")
        if child_elem is None:
            continue
        
        child_link = child_elem.attrib.get("link", "")
        props = link_to_props.get(child_link, {})
        
        # Get damping and friction values
        damping = props.get("joint_damping", 0.5)  # Default damping
        friction = REVOLUTE_JOINT_FRIC if joint_type == "revolute" else PRISMATIC_JOINT_FRIC
        
     
        for old_dyn in joint.findall("dynamics"):
            joint.remove(old_dyn)
        
        # Add dynamics element
        dynamics_elem = ET.SubElement(joint, "dynamics")
        dynamics_elem.attrib = {
            "damping": str(damping),
            # "friction": str(friction), # TODO: Remove this once we have the correct friction values
            "friction": str(friction),
        }
    
    # Save processed URDF
    output_urdf_path = _save_xmltree_as_urdf(
        root_element=root,
        name=f"{model}",
        dirpath=str(urdf_dir),
        unique_urdf=False,
    )
    
    # Generate metadata
    record_obj_metadata_from_urdf(
        urdf_path=output_urdf_path,
        obj_dir=str(obj_dir),
        joint_setting="zero",
        openable_joint_ids=openable_joint_ids,
        overwrite=overwrite,
    )
    
    return output_urdf_path
