"""Section 3.2 camera poses, using the COLMAP binary when it is installed.

The PDF names COLMAP and does not publish extractor settings. The commands
below are COLMAP 3.7's sequential pipeline for one video camera. A banner
that says "without CUDA" forces SIFT onto the CPU. A missing binary is not
a reconstruction, and this stage does not train a splat.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def use_gpu_from_banner(banner):
    """True only when the COLMAP banner does not say the build lacks CUDA."""
    return "without CUDA" not in banner


def pose_commands(database, image_dir, sparse_dir, use_gpu):
    """Three COLMAP 3.7 steps. ``use_gpu`` is the SIFT device flag, 0 or 1."""
    gpu = "1" if use_gpu else "0"
    database = str(database)
    image_dir = str(image_dir)
    sparse_dir = str(sparse_dir)
    return [
        [
            "colmap", "feature_extractor", "--log_to_stderr", "1",
            "--database_path", database,
            "--image_path", image_dir,
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", gpu,
            "--SiftExtraction.max_image_size", "1280",
        ],
        [
            "colmap", "sequential_matcher", "--log_to_stderr", "1",
            "--database_path", database,
            "--SiftMatching.use_gpu", gpu,
            "--SequentialMatching.overlap", "5",
        ],
        [
            "colmap", "mapper", "--log_to_stderr", "1",
            "--database_path", database,
            "--image_path", image_dir,
            "--output_path", sparse_dir,
        ],
    ]


def run_poses(video_path, work_dir, fps=1, max_frames=16):
    """Sample frames and run COLMAP. The frame cap is this runner's, not the paper's."""
    colmap = shutil.which("colmap")
    ffmpeg = shutil.which("ffmpeg")
    if not colmap or not ffmpeg:
        return {
            "ran": False,
            "ok": False,
            "reason": "colmap or ffmpeg is not on PATH",
            "colmap": colmap,
            "ffmpeg": ffmpeg,
        }
    work = Path(work_dir)
    images = work / "images"
    sparse = work / "sparse"
    if images.exists():
        shutil.rmtree(images)
    images.mkdir(parents=True, exist_ok=True)
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True, exist_ok=True)
    database = work / "database.db"
    if database.exists():
        database.unlink()
    extract = _run([
        ffmpeg, "-y", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-frames:v", str(max_frames),
        str(images / "%04d.jpg"),
    ], timeout_s=180)
    frames = sorted(path.name for path in images.glob("*.jpg"))
    if extract["returncode"] != 0 or not frames:
        return {
            "ran": True,
            "ok": False,
            "failed_step": "ffmpeg",
            "frames": frames,
            "steps": {"ffmpeg": extract},
        }
    banner = subprocess.run([colmap, "-h"], check=False, capture_output=True, text=True)
    use_gpu = use_gpu_from_banner((banner.stdout or "") + (banner.stderr or ""))
    steps = {"ffmpeg": extract}
    for argv in pose_commands(database, images, sparse, use_gpu):
        step_name = argv[1]
        steps[step_name] = _run(argv, timeout_s=900)
        if steps[step_name]["returncode"] != 0:
            return {
                "ran": True,
                "ok": False,
                "use_gpu": use_gpu,
                "failed_step": step_name,
                "frames": frames,
                "steps": steps,
            }
    model_dir = sparse / "0"
    if not (model_dir / "cameras.bin").is_file():
        return {
            "ran": True,
            "ok": False,
            "use_gpu": use_gpu,
            "failed_step": "mapper",
            "frames": frames,
            "reason": "mapper wrote no sparse/0/cameras.bin",
            "steps": steps,
        }
    steps["model_analyzer"] = _run(
        [colmap, "model_analyzer", "--path", str(model_dir)],
        timeout_s=120,
    )
    analyzer_ok = steps["model_analyzer"]["returncode"] == 0
    return {
        "ran": True,
        "ok": analyzer_ok,
        "use_gpu": use_gpu,
        "frames": frames,
        "model_dir": str(model_dir),
        "analyzer": steps["model_analyzer"],
        "steps": steps,
        "reason": "" if analyzer_ok else "model_analyzer failed",
    }


def _run(command, timeout_s=600):
    try:
        env = os.environ.copy()
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "command": command,
            "returncode": None,
            "timed_out": True,
            "stdout_tail": (stdout or "")[-2000:],
            "stderr_tail": (stderr or "")[-2000:],
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "timed_out": False,
        "stdout_tail": (completed.stdout or "")[-2000:],
        "stderr_tail": (completed.stderr or "")[-2000:],
    }
