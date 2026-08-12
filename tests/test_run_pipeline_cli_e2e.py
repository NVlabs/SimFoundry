# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_pipeline_cli_dry_run_smoke():
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/pipeline/A_reconstruction/run_reconstruction.py"),
        "--dry-run",
        "--exec-mode",
        "direct",
        "--include",
        "2,3",
        "scene_name=droid_desk_1",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(REPO_ROOT))
    out = proc.stdout
    assert "[Stage 2]" in out
    assert "[Stage 3]" in out
    assert "WALL:" in out


def test_reconstruction_cli_routes_stage_2c_to_nerfstudio_env():
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/pipeline/A_reconstruction/run_reconstruction.py"),
        "--dry-run",
        "--exec-mode",
        "mamba",
        # Stage 2c only enters the plan under --bg-splat; --include alone cannot select it.
        "--bg-splat",
        "--include",
        "2c",
        "--env-nerfstudio",
        "custom-nerfstudio",
        "scene_name=droid_desk_1",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(REPO_ROOT))

    assert "mamba run -n custom-nerfstudio python" in proc.stdout
    assert "A_reconstruction/stages/2c_train_bg_splat.py" in proc.stdout


def test_augmentation_cli_dry_run_smoke():
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/pipeline/B_augmentation/run_augmentation.py"),
        "--dry-run",
        "--exec-mode",
        "direct",
        "--phases",
        "scene-variations,task-generation",
        "scene_name=home_coffee_4",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert "[Stage 6]" in proc.stdout
    assert "[Stage 7]" in proc.stdout


def test_application_cli_dry_run_random_smoke():
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/pipeline/C_application/run_application.py"),
        "--dry-run",
        "--exec-mode",
        "direct",
        "--mode",
        "smoke-random",
        "scene_name=home_coffee_4",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert "[Stage smoke]" in proc.stdout
