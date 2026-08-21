# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Headless random-action smoke test for a generated OmniGibson scene."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

import imageio
import numpy as np
import torch as th
from PIL import Image

from simfoundry import CFG_DIR, REPO_DIR, configure_omnigibson_data_path, import_og_dependencies
from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths

configure_omnigibson_data_path(force=True)
import_og_dependencies()

import omnigibson as og
from omnigibson.utils.config_utils import parse_config


def _as_uint8_rgb(frame):
    frame = frame.cpu().numpy() if isinstance(frame, th.Tensor) else np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    if frame.shape[-1] == 4:
        frame = frame[:, :, :3]
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8) if frame.max(initial=0) <= 1.0 else frame.astype(np.uint8)
    return frame


def _resize_to_height(frame, target_height):
    if frame.shape[0] == target_height:
        return frame
    target_width = max(1, round(frame.shape[1] * target_height / frame.shape[0]))
    return np.asarray(Image.fromarray(frame).resize((target_width, target_height), Image.Resampling.BILINEAR))


def collect_frame(obs, robot):
    frames = []
    robot_obs = obs.get(robot.name, {})
    for cam_obs in robot_obs.values():
        if isinstance(cam_obs, dict) and "rgb" in cam_obs:
            frames.append(cam_obs["rgb"][:, :, :3])
    for cam_obs in obs.get("external", {}).values():
        if isinstance(cam_obs, dict) and "rgb" in cam_obs:
            frames.append(cam_obs["rgb"][:, :, :3])
    if not frames:
        return None
    np_frames = [_as_uint8_rgb(f) for f in frames]
    target_height = max(frame.shape[0] for frame in np_frames)
    np_frames = [_resize_to_height(frame, target_height) for frame in np_frames]
    return np.concatenate(np_frames, axis=1).astype(np.uint8)


def _parse_override_value(overrides: list[str], key: str, default: str) -> str:
    prefix = f"{key}="
    for item in overrides:
        if item.startswith(prefix):
            return item.split("=", 1)[1]
    return default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-json", default=None)
    parser.add_argument("--task-config", default="none")
    parser.add_argument("--external-sensors-config", default="sim_franka_teleop")
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--video-fps", type=int, default=10)
    args, overrides = parser.parse_known_args()

    scene_name = _parse_override_value(overrides, "scene_name", "home_coffee_4")
    root_dir = _parse_override_value(overrides, "root_dir", str(Path(REPO_DIR) / "Data"))
    n_steps = int(_parse_override_value(overrides, "application_smoke.n_steps", str(args.n_steps)))
    video_fps = int(_parse_override_value(overrides, "application_smoke.video_fps", str(args.video_fps)))
    task_config = _parse_override_value(overrides, "application_smoke.task_config", args.task_config)
    scene_json_path = Path(args.scene_json or Path(root_dir) / scene_name / "s14_og" / "reconstructed_og_scene.json")
    if not scene_json_path.exists():
        raise FileNotFoundError(f"Generated scene JSON not found: {scene_json_path}")

    out_dir = Path(root_dir) / scene_name / "application_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(
        _parse_override_value(
            overrides,
            "application_smoke.video_path",
            args.video_path or str(out_dir / "random_action_smoke.mp4"),
        )
    )

    os.chdir(CFG_DIR)
    task_cfg = None
    if task_config.lower() not in {"none", "null", ""}:
        task_cfg_path = Path(task_config)
        if not task_cfg_path.exists():
            task_cfg_path = Path(CFG_DIR) / "task" / f"{task_config}.yaml"
        task_cfg = parse_config(str(task_cfg_path)).get("og_task_config")
    external_sensors_cfg = parse_config(
        str(Path(CFG_DIR) / "external_sensors" / f"{args.external_sensors_config}.yaml")
    )["external_sensors"]
    scene_json = load_json_with_absolute_usd_paths(str(scene_json_path))

    og_cfg = {
        "env": {
            "external_sensors": external_sensors_cfg,
            "action_frequency": 15,
            "rendering_frequency": 15,
            "physics_frequency": 120,
        },
        "scene": {
            "type": "Scene",
            "scene_file": scene_json,
            "use_floor_plane": True,
            "floor_plane_visible": False,
            "use_skybox": True,
            "include_robots": True,
        },
    }
    if task_cfg is not None:
        og_cfg["task"] = task_cfg

    env = og.Environment(configs=og_cfg)
    obs, _ = env.reset()
    robot = env.robots[0]
    print(f"Robot: {robot.name}, action dim: {robot.action_space.shape}")

    video_frames = []
    frame = collect_frame(obs, robot)
    if frame is not None:
        video_frames.append(frame)

    try:
        for step in range(n_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {step + 1:2d} | reward={reward:.4f} | done={terminated or truncated}")
            frame = collect_frame(obs, robot)
            if frame is not None:
                video_frames.append(frame)
            if terminated or truncated:
                obs, _ = env.reset()

        if video_frames:
            imageio.mimwrite(video_path, video_frames, fps=video_fps)
            print(f"Saved rollout video: {video_path}")
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
