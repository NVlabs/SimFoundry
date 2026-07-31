# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SIMFOUNDRY_TEST_DATA_ROOT", REPO_ROOT / "Data")).expanduser()
REFERENCE_SCENES = tuple(
    scene.strip()
    for scene in os.environ.get("SIMFOUNDRY_REFERENCE_SCENES", "droid_desk_1,droid_desk_2").split(",")
    if scene.strip()
)
CORE_STAGE_DIRS = (
    "s1_video",
    "s2_da",
    "s3_ground",
    "s4_frame",
    "s5_scene",
    "s7_mesh",
    "s8_pose",
    "s9_compile",
    "s10_sim",
    "s11_physics",
)


def _available_reference_scenes():
    scenes = [scene for scene in REFERENCE_SCENES if (DATA_ROOT / scene).exists()]
    if not scenes:
        pytest.skip(
            "Optional reference datasets are not present. Set SIMFOUNDRY_TEST_DATA_ROOT and "
            "SIMFOUNDRY_REFERENCE_SCENES to enable these checks."
        )
    return scenes


def _load_stage_info(scene: str, stage_dir: str) -> dict:
    fpath = DATA_ROOT / scene / stage_dir / "stage_info.json"
    assert fpath.exists(), f"Missing stage info: {fpath}"
    return json.loads(fpath.read_text(encoding="utf-8"))


def test_reference_data_contains_core_stage_outputs():
    found = 0
    scenes = _available_reference_scenes()
    for scene in scenes:
        for stage in CORE_STAGE_DIRS:
            fpath = DATA_ROOT / scene / stage / "stage_info.json"
            if not fpath.exists():
                continue
            info = json.loads(fpath.read_text(encoding="utf-8"))
            assert isinstance(info, dict)
            assert "out_dirname" in info
            found += 1
    # Snapshot can be partial per scene; require broad overall coverage.
    min_stage_infos = int(os.environ.get(
        "SIMFOUNDRY_REFERENCE_MIN_STAGE_INFOS",
        str(min(12, len(scenes) * len(CORE_STAGE_DIRS))),
    ))
    assert found >= min_stage_infos


def test_reference_data_has_partial_multistep_artifacts():
    scene = os.environ.get("SIMFOUNDRY_REFERENCE_LINKAGE_SCENE", "droid_desk_1")
    scene_dir = DATA_ROOT / scene
    if not scene_dir.exists():
        pytest.skip(
            f"Optional reference dataset '{scene}' is not present under {DATA_ROOT}. "
            "Set SIMFOUNDRY_REFERENCE_LINKAGE_SCENE to a generated scene to enable this check."
        )

    # Step 5 -> 6 linkage
    masked = sorted((scene_dir / "s5_scene" / "masked_object").glob("iter_*.png"))
    upsampled = sorted((scene_dir / "s6_upsample" / "upsampled").glob("iter_*.png"))
    assert len(masked) > 0
    assert len(upsampled) > 0

    # Step 6 -> 7 linkage
    transparent = sorted((scene_dir / "s6_upsample" / "upsampled").glob("iter_*_transparent.png"))
    textured = sorted((scene_dir / "s7_mesh" / "textured_mesh").rglob("iter_*_mesh.glb"))
    assert len(transparent) > 0
    assert len(textured) > 0
