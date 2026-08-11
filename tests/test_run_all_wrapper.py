# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path


def test_run_all_defaults_to_home_coffee_fixture_dry_run():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            str(repo_root / "scripts" / "pipeline" / "A_reconstruction" / "run.sh"),
            "--dry-run",
            "--exec-mode",
            "direct",
            "--no-stream",
            "--include",
            "1b",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    expected_video = repo_root / "Data" / "home_coffee_4" / "s1_video" / "video" / "scene.mp4"
    assert f"scene_name=home_coffee_4" in result.stdout
    assert f"s1_video.video_fpath={expected_video}" in result.stdout


def test_run_all_resolves_relative_model_cache_dir_from_repo_root(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    probe = tmp_path / "probe_python"
    probe.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"SIMFOUNDRY_MODEL_CACHE_DIR=${SIMFOUNDRY_MODEL_CACHE_DIR}\"\n",
        encoding="utf-8",
    )
    probe.chmod(0o755)

    result = subprocess.run(
        [
            str(repo_root / "scripts" / "pipeline" / "A_reconstruction" / "run.sh"),
            "--python-bin",
            str(probe),
            "--exec-mode",
            "direct",
            "--include",
            "1b",
            "--cache-mode",
            "--model-cache-dir",
            ".cache/simfoundry/model_calls",
        ],
        cwd=repo_root / "scripts" / "cfg",
        check=True,
        capture_output=True,
        text=True,
    )

    expected_cache_dir = repo_root / ".cache" / "simfoundry" / "model_calls"
    assert f"SIMFOUNDRY_MODEL_CACHE_DIR={expected_cache_dir}" in result.stdout


def test_run_all_forwards_nerfstudio_environment(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    probe = tmp_path / "probe_python"
    probe.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    probe.chmod(0o755)

    result = subprocess.run(
        [
            str(repo_root / "scripts" / "pipeline" / "A_reconstruction" / "run.sh"),
            "--python-bin",
            str(probe),
            "--exec-mode",
            "direct",
            "--include",
            "2c",
            "--env-nerfstudio",
            "custom-nerfstudio",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    forwarded = result.stdout.splitlines()
    env_flag_index = forwarded.index("--env-nerfstudio")
    assert forwarded[env_flag_index + 1] == "custom-nerfstudio"
