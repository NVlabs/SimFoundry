# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
from pathlib import Path
import sys

import pytest

# These target a refactor of stage 11 into discover_sim_ready_jobs / ConversionTask /
# convert_objects_parallel that is not in this release: 11_make_objects_sim_ready.py defines
# none of them (it has resolve_requested_indices and import_rigid_scene_object instead).
# Skipped rather than deleted so the intended interface survives for whoever lands that
# refactor -- and so `pytest` stays usable as the documented install check.
pytestmark = pytest.mark.skip(
    reason="stage 11 refactor (discover_sim_ready_jobs / ConversionTask) is not in this release"
)


def load_stage11_module(repo_root: Path):
    module_path = repo_root / "scripts" / "pipeline" / "A_reconstruction" / "stages" / "11_make_objects_sim_ready.py"
    name = "stage11_make_objects_sim_ready_for_tests"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    cwd = os.getcwd()
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
    return module


def test_discover_sim_ready_jobs_filters_requested_indices(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    stage11 = load_stage11_module(repo_root)
    mesh_dir = tmp_path / "pose" / "canonical_mesh"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "iter_1.glb").write_bytes(b"mesh")
    (mesh_dir / "iter_2.glb").write_bytes(b"mesh")
    (mesh_dir / "ignore.txt").write_text("x", encoding="utf-8")

    jobs = stage11.discover_sim_ready_jobs(
        scene_dir=str(tmp_path / "scene"),
        img_dir=str(tmp_path / "img"),
        pose_dir=str(tmp_path / "pose"),
        mesh_dir=str(mesh_dir),
        use_interactive_pose=False,
        interactive_suffix="",
        allowed_indices={2},
    )

    assert [job.idx for job in jobs] == [2]
    assert jobs[0].img_fpath.endswith("iter_2.png")
    assert jobs[0].pose_info_path.endswith("info/iter_2.json")


def test_convert_objects_parallel_invokes_each_conversion_once(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    stage11 = load_stage11_module(repo_root)
    calls = []

    def fake_import_custom_object(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(stage11, "import_custom_object", fake_import_custom_object)
    tasks = [
        stage11.ConversionTask(
            idx=idx,
            category="cup",
            model=f"aaaaa{idx}",
            name=f"iter_{idx}",
            mesh_path=str(tmp_path / f"iter_{idx}.glb"),
            dataset_root=str(tmp_path / "out"),
            collision_method="coacd",
            hull_count=32,
            n_submesh=10,
            scale=1.0,
            mass=0.2,
            friction=0.5,
            is_articulated=False,
        )
        for idx in (1, 2)
    ]

    results = stage11.convert_objects(tasks, parallel_workers=2)

    assert sorted(results) == [1, 2]
    assert len(calls) == 2
    assert {call["asset_path"] for call in calls} == {str(tmp_path / "iter_1.glb"), str(tmp_path / "iter_2.glb")}
