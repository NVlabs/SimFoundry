# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "scripts" / "pipeline"


def test_subpipeline_entrypoints_exist():
    assert (PIPELINE_ROOT / "A_reconstruction" / "run.sh").is_file()
    assert (PIPELINE_ROOT / "B_augmentation" / "run.sh").is_file()
    assert (PIPELINE_ROOT / "C_application" / "run.sh").is_file()
    assert (PIPELINE_ROOT / "run.sh").is_file()


def test_automated_pipeline_scripts_have_no_live_breakpoints():
    automated_roots = [
        PIPELINE_ROOT / "A_reconstruction" / "stages",
        PIPELINE_ROOT / "B_augmentation" / "stages",
    ]
    offenders = []
    for root in automated_roots:
        for path in root.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "breakpoint()" in text:
                offenders.append(str(path))
    assert offenders == []


def test_superseded_upcoming_pipeline_removed():
    assert not (REPO_ROOT / "scripts" / "upcoming_pipeline").exists()
