# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cameras that belong to the robot asset rather than to a rig config.

OmniGibson exposes ``Camera`` prims inside the robot USD as sensors, and the
evaluation stage reads the wrist camera straight off the robot's own
observations as ``wrist_image_left``. Everything here is read-only: a camera's
pose is the arm's joint configuration, and there is no ``external_sensors``
file to write it to. Poses come out in the robot's own frame, the same frame
the rig configs use (``pose_frame: parent``).
"""

import math
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom

import robot_pose
import scene_io
import usd_cache

#: What the evaluation stage calls this image when it hands it to the policy.
WRIST_OBSERVATION_KEY = "wrist_image_left"

#: Camera prims named this take their name from the link they hang off instead.
GENERIC_PRIM_NAMES = frozenset({"Camera", "camera", "Cam", "cam"})

#: Aspect ratio OmniGibson renders a robot's own camera at. Pixel counts vary
#: between stages, so the resolution is deliberately not reported; the aspect
#: is what decides framing.
ROBOT_CAMERA_ASPECT = 1.0


def _fov_degrees(camera, aspect):
    """Field of view a USD camera covers, in degrees.

    Args:
        camera (UsdGeom.Camera): The camera prim's schema.
        aspect (float): Width over height of the image it is rendered at.

    Returns:
        tuple[float or None, float or None]: Horizontal and vertical degrees, or
            ``(None, None)`` when the prim carries no usable optics.
    """
    focal = float(camera.GetFocalLengthAttr().Get() or 0.0)
    aperture = float(camera.GetHorizontalApertureAttr().Get() or 0.0)
    if focal <= 0.0 or aperture <= 0.0 or aspect <= 0.0:
        return None, None
    horizontal = 2.0 * math.degrees(math.atan(aperture / (2.0 * focal)))
    # The authored vertical aperture is ignored: Omniverse derives it from the
    # render resolution, so the vertical angle follows the requested aspect.
    vertical = 2.0 * math.degrees(math.atan(aperture / aspect / (2.0 * focal)))
    return horizontal, vertical


def _decompose(matrix, meters_per_unit):
    """Split a row-vector 4x4 into a position in metres and an XYZW quaternion.

    Args:
        matrix (np.ndarray): (4, 4) row-vector transform in stage units.
        meters_per_unit (float): The stage's own unit scale.

    Returns:
        tuple[list[float], list[float]]: ``(position, orientation)``.
    """
    gf = Gf.Matrix4d(*[float(v) for v in np.asarray(matrix, dtype=np.float64).ravel()])
    # Remove scale and shear first so ExtractRotationQuat reads a pure rotation.
    quaternion = gf.RemoveScaleShear().ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    position = [float(v) * meters_per_unit for v in gf.ExtractTranslation()]
    orientation = [
        float(imaginary[0]), float(imaginary[1]), float(imaginary[2]),
        float(quaternion.GetReal()),
    ]
    return position, orientation


def _unique(name, taken):
    """Make *name* unique against *taken*, which it joins.

    The browser keys every record by name, so a robot camera called the same
    thing as a prop would replace it in the object map rather than sit beside it.

    Args:
        name (str): Preferred name.
        taken (set): Names already spoken for. Mutated.

    Returns:
        str: A name not in *taken* before the call.
    """
    candidate = name
    suffix = 2
    while candidate in taken:
        candidate = f"{name}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def read_robot_cameras(usd_path, joint_pose=None, taken=None, on_warning=print):
    """Read every camera prim out of a robot USD, at its saved joint pose.

    Args:
        usd_path (str or Path): The robot's visual USD, as resolved by
            ``scene_io.resolve_robot_usd``.
        joint_pose (robot_pose.RobotJointPose or None): Solved articulation. When
            given, a camera on a moving link is reported where that link now is
            rather than where the USD authors it. When None the rest pose is
            used, which is what the viewport draws in that case too.
        taken (set or None): Names already in use in the scene.
        on_warning (callable): Receives one string per camera that had to be
            skipped.

    Returns:
        list[dict]: Records shaped like ``camera_io.load_cameras``', with
            ``read_only`` set. Empty when the asset carries no camera.
    """
    if usd_path is None:
        return []
    stage = usd_cache.open_stage(usd_path)
    if stage is None:
        on_warning(f"{Path(usd_path).name}: could not be opened; no robot cameras.")
        return []

    taken = set() if taken is None else taken
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage)) or 1.0
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    cameras = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Camera):
            continue
        path = str(prim.GetPath())
        horizontal, vertical = _fov_degrees(UsdGeom.Camera(prim), ROBOT_CAMERA_ASPECT)
        if horizontal is None:
            on_warning(f"{path}: no usable focal length or aperture; skipped.")
            continue

        matrix = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        # Row-vector, so the correction composes on the right — the same order
        # the robot's meshes are baked with.
        correction = None if joint_pose is None else joint_pose.correction_for(path)
        if correction is not None:
            matrix = matrix @ correction
        position, orientation = _decompose(matrix, meters_per_unit)

        parent = prim.GetParent()
        preferred = (parent.GetName() if prim.GetName() in GENERIC_PRIM_NAMES
                     and parent and parent.GetName() else prim.GetName())
        cameras.append({
            "index": len(cameras),
            "name": _unique(preferred, taken),
            "position": position,
            "orientation": orientation,
            "relative_prim_path": path,
            "pose_frame": "parent",
            # Not reported: see ROBOT_CAMERA_ASPECT. The browser falls back to
            # `aspect` for the frustum.
            "image_width": None,
            "image_height": None,
            "aspect": ROBOT_CAMERA_ASPECT,
            "h_fov_deg": horizontal,
            "v_fov_deg": vertical,
            "modalities": ["rgb"],
            "clipping_range": None,
            # This pose is the arm's; the editor has nowhere to write it.
            "read_only": True,
            "why": (f"part of {Path(usd_path).name} — it rides the arm, and its pose "
                    "is the saved joint configuration rather than anything this "
                    "editor writes"),
            "posed": correction is not None,
        })
    return cameras


def observation_entry(camera, count):
    """Say which policy input a robot camera fills.

    The evaluation stage takes the *first* robot sensor whose key mentions a
    camera and calls it ``wrist_image_left``. With one camera that is
    unambiguous; with several, the order is invisible to this editor, so the
    claim is marked uncertain.

    Args:
        camera (dict): One record from :func:`read_robot_cameras`.
        count (int): How many cameras the asset carries.

    Returns:
        dict: An entry for the ``observation.cameras`` map.
    """
    certain = count == 1
    return {
        "key": WRIST_OBSERVATION_KEY,
        "short": "wrist",
        "certain": certain,
        "detail": (
            f"the robot's own sensor, read as {WRIST_OBSERVATION_KEY}"
            if certain else
            f"{count} cameras on this robot; the evaluation stage takes the first "
            f"one its observation dict names, which may not be this one"
        ),
    }


def scene_robot_cameras(scene, scene_json_path, robot_asset_dir, on_warning=print):
    """Find the scene's robot and read the cameras built into its asset.

    Args:
        scene (dict): Parsed scene JSON.
        scene_json_path (str or Path): Where it was read from.
        robot_asset_dir (str or Path): Root of ``omnigibson-robot-assets``.
        on_warning (callable): Receives one string per skipped camera or robot.

    Returns:
        tuple[list[dict], dict]: The camera records, and their entries for the
            ``observation.cameras`` map keyed by camera name. Both empty when the
            scene has no robot, the asset is unmapped, or it carries no camera.
    """
    records = list(scene_io.iter_objects(
        scene, scene_json_path, robot_asset_dir=robot_asset_dir, usd_facts=False))
    taken = {record["name"] for record in records}
    cameras, observation = [], {}
    for record in records:
        if record["kind"] != "robot" or record["usd"] is None:
            continue
        info = scene.get("objects_info", {}).get("init_info", {}).get(record["name"], {})
        joint_pos = robot_pose.saved_joint_pos(scene, record["name"])
        pose = None
        if joint_pos is not None:
            # Warnings are suppressed: the extraction has already reported the
            # same thing about the same robot.
            pose = robot_pose.solve_arm_pose(
                record["usd"], info.get("class_name"),
                info.get("args", {}).get("end_effector"), joint_pos,
                on_warning=lambda _message: None,
            )
        found = read_robot_cameras(
            record["usd"], joint_pose=pose, taken=taken, on_warning=on_warning)
        for camera in found:
            observation[camera["name"]] = observation_entry(camera, len(found))
        cameras.extend(found)
    return cameras, observation
