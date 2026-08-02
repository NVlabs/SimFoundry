# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

_STAGE8B_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/pipeline/A_reconstruction/stages/8b_articulate_objects.py"
)

# Stage 8b (articulation) is an optional component; these tests skip when it is absent.
pytestmark = pytest.mark.skipif(
    not _STAGE8B_SCRIPT.is_file(),
    reason="articulation stage 8b is not available in this release",
)


def _load_stage8b_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts/pipeline/A_reconstruction/stages/8b_articulate_objects.py"
    cwd = os.getcwd()
    try:
        spec = importlib.util.spec_from_file_location("stage8b_articulate_objects", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(cwd)


def _cfg(**overrides):
    values = {
        "force_articulated": [],
        "force_non_articulated": [],
        "interactive_review": False,
    }
    values.update(overrides)
    return OmegaConf.create({"s8b_articulate_objects": values})


def test_parse_articulated_object_selection_accepts_only_detected_names():
    stage8b = _load_stage8b_module()

    articulated, non_articulated, ignored, raw = stage8b.parse_articulated_object_selection(
        """
        ```json
        {"articulated_objects": ["wooden desk organizer", "cabinet", "iter_12"]}
        ```
        """,
        ["red marker", "wooden desk organizer", "white bottle"],
    )

    assert articulated == ["wooden desk organizer"]
    assert non_articulated == ["red marker", "white bottle"]
    assert ignored == ["cabinet", "iter_12"]
    assert raw == ["wooden desk organizer", "cabinet", "iter_12"]


def test_parse_articulated_object_selection_supports_json_list_and_normalized_names():
    stage8b = _load_stage8b_module()

    articulated, non_articulated, ignored, raw = stage8b.parse_articulated_object_selection(
        '["Wooden_Desk Organizer"]',
        ["wooden desk organizer", "white bottle"],
    )

    assert articulated == ["wooden desk organizer"]
    assert non_articulated == ["white bottle"]
    assert ignored == []
    assert raw == ["Wooden_Desk Organizer"]


def test_force_articulated_override_can_add_detected_object():
    stage8b = _load_stage8b_module()

    articulated, non_articulated = stage8b.review_classification(
        articulated=[],
        non_articulated=["red marker"],
        object_list={"red marker": "iter_1", "wooden desk organizer": "iter_12"},
        cfg=_cfg(force_articulated=["red marker"]),
    )

    assert articulated == ["red marker"]
    assert non_articulated == []
