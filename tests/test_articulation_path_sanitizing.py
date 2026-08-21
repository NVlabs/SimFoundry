# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 9 writes articulation results; stage 11 reads them back.

Regression cover for a writer/reader mismatch: stage 9 used the raw scene name and a
non-lowercased object name, while stage 11 lowercased both. A scene named ``Laptop``
therefore had its results written to ``.../Laptop/...`` and looked up at
``.../laptop/...``, so a successful articulation was silently discarded and the object
fell back to a rigid import.
"""

import os
from pathlib import Path

import pytest

from simfoundry.utils.python_utils import sanitize_path_component

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_9 = REPO_ROOT / "scripts/pipeline/A_reconstruction/stages/9_articulate_objects.py"
STAGE_11 = REPO_ROOT / "scripts/pipeline/A_reconstruction/stages/11_make_objects_sim_ready.py"


def _load_resolver():
    """Pull the resolver out of stage 11 without importing its heavy dependencies."""
    src = STAGE_11.read_text(encoding="utf-8")
    start = src.index("def resolve_articulation_results_dir")
    end = src.index("def invalid_articulated_links")
    ns = {"os": os, "sanitize_path_component": sanitize_path_component}
    exec(src[start:end], ns)
    return ns["resolve_articulation_results_dir"]


resolve_articulation_results_dir = _load_resolver()


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Laptop", "laptop"),
        ("laptop", "laptop"),
        ("Black Laptop", "black_laptop"),
        ("MarkerInCup", "markerincup"),
        ("a/b c", "a_b_c"),
    ],
)
def test_sanitizer_is_idempotent_and_case_folding(value, expected):
    assert sanitize_path_component(value) == expected
    assert sanitize_path_component(expected) == expected


def test_writer_and_reader_agree_on_a_capitalized_scene(tmp_path):
    """The exact bug: capitalized scene name must round-trip."""
    scene, obj = "Laptop", "black laptop"
    out_dir = tmp_path / "s9_articulate_objects"

    # Writer side, mirroring stage 9: both components sanitized.
    written = out_dir / sanitize_path_component(scene) / sanitize_path_component(obj) / "results"
    written.mkdir(parents=True)
    (written / "mobility.urdf").write_text("<robot/>")

    assert resolve_articulation_results_dir(str(out_dir), scene, obj) == str(written)


def test_legacy_unsanitized_layout_is_still_found(tmp_path):
    """Runs produced before the fix must not silently degrade to a rigid import."""
    scene, obj = "Laptop", "black laptop"
    out_dir = tmp_path / "s9_articulate_objects"
    legacy = out_dir / "Laptop" / "black_laptop" / "results"
    legacy.mkdir(parents=True)
    (legacy / "mobility.urdf").write_text("<robot/>")

    assert resolve_articulation_results_dir(str(out_dir), scene, obj) == str(legacy)


def test_sanitized_layout_wins_when_both_exist(tmp_path):
    scene, obj = "Laptop", "black laptop"
    out_dir = tmp_path / "s9_articulate_objects"
    for parts in (("laptop", "black_laptop"), ("Laptop", "black_laptop")):
        d = out_dir.joinpath(*parts, "results")
        d.mkdir(parents=True)
        (d / "mobility.urdf").write_text("<robot/>")

    resolved = resolve_articulation_results_dir(str(out_dir), scene, obj)
    assert resolved == str(out_dir / "laptop" / "black_laptop" / "results")


def test_missing_result_reports_the_canonical_path(tmp_path):
    out_dir = tmp_path / "s9_articulate_objects"
    resolved = resolve_articulation_results_dir(str(out_dir), "Laptop", "black laptop")
    assert resolved.endswith("/laptop/black_laptop/results")


def test_both_stages_use_the_shared_sanitizer():
    """Guard against either side re-growing its own copy and drifting again."""
    for path in (STAGE_9, STAGE_11):
        src = path.read_text(encoding="utf-8")
        assert "sanitize_path_component" in src, f"{path.name} must use the shared sanitizer"
    # The old hand-rolled form must not come back in the writer's path construction.
    assert 'replace(" ", "_").replace("/", "_")\n' not in STAGE_9.read_text(encoding="utf-8")
