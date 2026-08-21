#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualize an articulated URDF by sweeping movable joints through their limits."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

from simfoundry.utils.python_utils import sanitize_path_component


def default_urdf_path(root_dir: Path, scene_name: str, object_name: str) -> Path:
    """Resolve a stage-9 URDF, accepting the pre-sanitizer layout for older runs."""
    s9_dir = root_dir / scene_name / "s9_articulate_objects"
    sanitized = s9_dir / sanitize_path_component(scene_name) / sanitize_path_component(object_name) / "results" / "mobility.urdf"
    if sanitized.is_file():
        return sanitized
    legacy = s9_dir / scene_name / object_name.replace(" ", "_").replace("/", "_") / "results" / "mobility.urdf"
    return legacy if legacy.is_file() else sanitized


def make_pybullet_compatible_urdf(urdf_path: Path) -> tuple[Path, list[Path]]:
    """Rewrite unsupported mesh extensions to existing OBJ companions for PyBullet."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    changed = False
    cleanup_paths: list[Path] = []

    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue

        mesh_path = (urdf_path.parent / filename).resolve()
        if mesh_path.suffix.lower() not in {".glb", ".gltf"}:
            continue

        obj_path = mesh_path.with_suffix(".obj")
        if not obj_path.exists():
            try:
                import trimesh
            except ImportError as exc:
                raise RuntimeError(
                    f"PyBullet cannot load {mesh_path.suffix} meshes and no OBJ companion exists for {mesh_path}."
                ) from exc
            loaded = trimesh.load(mesh_path, force="mesh")
            obj_path = mesh_path.with_suffix(".pybullet.obj")
            loaded.export(obj_path)
            cleanup_paths.append(obj_path)

        mesh.attrib["filename"] = os.path.relpath(obj_path, urdf_path.parent)
        changed = True

    if not changed:
        return urdf_path, cleanup_paths

    with tempfile.NamedTemporaryFile(
        dir=urdf_path.parent,
        prefix=f".{urdf_path.stem}_pybullet_",
        suffix=".urdf",
        delete=False,
    ) as tmp:
        temp_urdf = Path(tmp.name)
        tree.write(tmp, encoding="utf-8", xml_declaration=True)
    cleanup_paths.append(temp_urdf)
    return temp_urdf, cleanup_paths


def finite_joint_limits(joint_info) -> tuple[float, float]:
    joint_type = joint_info[2]
    lower = float(joint_info[8])
    upper = float(joint_info[9])
    if math.isfinite(lower) and math.isfinite(upper) and upper > lower:
        return lower, upper
    if joint_type == p.JOINT_REVOLUTE:
        return -math.pi / 2, math.pi / 2
    if joint_type == p.JOINT_PRISMATIC:
        return 0.0, 0.25
    return 0.0, 0.0


def movable_joints(body_id: int) -> list[dict]:
    joints = []
    for joint_idx in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, joint_idx)
        if info[2] not in {p.JOINT_REVOLUTE, p.JOINT_PRISMATIC}:
            continue
        lower, upper = finite_joint_limits(info)
        joints.append(
            {
                "index": joint_idx,
                "name": info[1].decode("utf-8"),
                "type": "revolute" if info[2] == p.JOINT_REVOLUTE else "prismatic",
                "lower": lower,
                "upper": upper,
            }
        )
    return joints


def body_aabb(body_id: int) -> tuple[np.ndarray, np.ndarray]:
    lows = []
    highs = []
    for link_idx in [-1, *range(p.getNumJoints(body_id))]:
        try:
            low, high = p.getAABB(body_id, link_idx)
        except Exception:
            continue
        lows.append(low)
        highs.append(high)
    if not lows:
        return np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5])
    return np.min(np.array(lows), axis=0), np.max(np.array(highs), axis=0)


def camera_frame(body_id: int, width: int, height: int, yaw: float, pitch: float, distance: float) -> np.ndarray:
    low, high = body_aabb(body_id)
    target = (low + high) / 2.0
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target.tolist(),
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=width / height, nearVal=0.01, farVal=10.0)
    _, _, rgba, _, _ = p.getCameraImage(width, height, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]


def sweep_value(lower: float, upper: float, frame_idx: int, total_frames: int, cycles: float) -> float:
    if total_frames <= 1:
        return lower
    phase = frame_idx / (total_frames - 1)
    alpha = 0.5 - 0.5 * math.cos(2.0 * math.pi * cycles * phase)
    return lower + alpha * (upper - lower)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=None, help="URDF to visualize. Defaults to the scene/object stage 9 output.")
    parser.add_argument("--root-dir", type=Path, default=Path("Data"), help="Pipeline root directory.")
    parser.add_argument("--scene-name", default="pull_scene_0", help="Scene name used when --urdf is omitted.")
    parser.add_argument("--object-name", default="wooden desk organizer", help="Articulated object name used when --urdf is omitted.")
    parser.add_argument("--gui", action="store_true", help="Use PyBullet GUI instead of headless rendering.")
    parser.add_argument("--output-video", type=Path, default=None, help="Optional MP4 path for headless or GUI runs.")
    parser.add_argument("--duration", type=float, default=4.0, help="Animation duration in seconds.")
    parser.add_argument("--fps", type=int, default=24, help="Output/video frame rate.")
    parser.add_argument("--cycles", type=float, default=1.0, help="Number of full open-close cycles.")
    parser.add_argument("--width", type=int, default=960, help="Rendered video width.")
    parser.add_argument("--height", type=int, default=720, help="Rendered video height.")
    parser.add_argument("--camera-distance", type=float, default=1.1)
    parser.add_argument("--camera-yaw", type=float, default=45.0)
    parser.add_argument("--camera-pitch", type=float, default=-25.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.resolve()
    urdf_path = args.urdf.resolve() if args.urdf else default_urdf_path(root_dir, args.scene_name, args.object_name).resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    output_video = args.output_video
    if output_video is None and not args.gui:
        output_video = urdf_path.parent / "joint_range_preview.mp4"

    client = p.connect(p.GUI if args.gui else p.DIRECT)
    load_urdf_path = urdf_path
    cleanup_paths: list[Path] = []
    try:
        load_urdf_path, cleanup_paths = make_pybullet_compatible_urdf(urdf_path)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.loadURDF("plane.urdf")

        cwd = os.getcwd()
        os.chdir(load_urdf_path.parent)
        try:
            body_id = p.loadURDF(str(load_urdf_path), useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
        finally:
            os.chdir(cwd)

        joints = movable_joints(body_id)
        print(f"Loaded {urdf_path}")
        if not joints:
            print("No movable revolute/prismatic joints found.")
            return
        for joint in joints:
            print(
                f"Joint {joint['index']}: {joint['name']} "
                f"({joint['type']}) limits=[{joint['lower']:.4f}, {joint['upper']:.4f}]"
            )

        writer = None
        if output_video is not None:
            import imageio

            output_video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(output_video, fps=args.fps)

        total_frames = max(1, int(args.duration * args.fps))
        for frame_idx in range(total_frames):
            for joint in joints:
                value = sweep_value(joint["lower"], joint["upper"], frame_idx, total_frames, args.cycles)
                p.resetJointState(body_id, joint["index"], value)
            p.stepSimulation()

            if writer is not None:
                frame = camera_frame(
                    body_id,
                    width=args.width,
                    height=args.height,
                    yaw=args.camera_yaw,
                    pitch=args.camera_pitch,
                    distance=args.camera_distance,
                )
                writer.append_data(frame)
            if args.gui:
                time.sleep(1.0 / max(args.fps, 1))

        if writer is not None:
            writer.close()
            print(f"Wrote {output_video}")
    finally:
        p.disconnect(client)
        for path in cleanup_paths:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
