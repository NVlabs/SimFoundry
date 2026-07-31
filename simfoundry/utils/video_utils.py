# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight video read/write/slice helpers (cv2 + ffmpeg).

Kept dependency-light on purpose (cv2, numpy, subprocess, pathlib only) so the
auto_bg void-driver stages can import these without pulling the heavier
``processing_utils`` import chain. All re-encoding goes through an intermediate
PNG dump so chunk boundaries don't depend on source keyframe placement (cv2
frame seek + libx264 re-encode preserves exact frame indexing).
"""
import subprocess
from pathlib import Path

import cv2
import numpy as np


def read_video_frames(mp4: Path) -> np.ndarray:
    """Read every frame of a video into an (T, H, W, 3) uint8 RGB array."""
    cap = cv2.VideoCapture(str(mp4))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, axis=0)  # (T, H, W, 3) uint8


def write_video(frames: np.ndarray, out_path: Path, fps: int, crf: int = 16) -> None:
    """Encode an (T, H, W, 3) uint8 RGB array to an mp4 (libx264, yuv420p)."""
    tmp_dir = out_path.parent / f"_tmp_{out_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(tmp_dir / f"{i:04d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(tmp_dir / "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    for p in tmp_dir.glob("*.png"):
        p.unlink()
    tmp_dir.rmdir()


def slice_video(in_path: Path, out_path: Path, start: int, count: int, fps: int, crf: int = 16) -> None:
    """Re-encode frames [start, start+count) from @in_path to a new mp4 (libx264, yuv420p)."""
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open {in_path}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start + count > n_total:
        raise SystemExit(f"chunk start={start} count={count} exceeds {in_path} ({n_total} frames)")

    tmp_dir = out_path.parent / f"_tmp_{out_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    written = 0
    while written < count:
        ok, frm = cap.read()
        if not ok:
            break
        cv2.imwrite(str(tmp_dir / f"{written:04d}.png"), frm)
        written += 1
    cap.release()
    if written != count:
        raise SystemExit(f"only got {written}/{count} frames from {in_path} starting at {start}")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(tmp_dir / "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    for p in tmp_dir.glob("*.png"):
        p.unlink()
    tmp_dir.rmdir()


def slice_video_lossless(in_path: Path, out_path: Path, start: int, count: int, fps: int) -> None:
    """Same as @slice_video but yuv444p qp=0 (lossless; preserves quadmask {0,63,127,255})."""
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open {in_path}")
    tmp_dir = out_path.parent / f"_tmp_{out_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    written = 0
    while written < count:
        ok, frm = cap.read()
        if not ok:
            break
        cv2.imwrite(str(tmp_dir / f"{written:04d}.png"), frm)
        written += 1
    cap.release()
    if written != count:
        raise SystemExit(f"only got {written}/{count} mask frames")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(tmp_dir / "%04d.png"),
        "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast",
        "-pix_fmt", "yuv444p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    for p in tmp_dir.glob("*.png"):
        p.unlink()
    tmp_dir.rmdir()
