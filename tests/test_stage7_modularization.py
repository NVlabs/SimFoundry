# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
from pathlib import Path
import sys


def _load_stage7_module():
    script = Path(__file__).resolve().parents[1] / "scripts/pipeline/A_reconstruction/stages/7_generate_object_meshes.py"
    spec = importlib.util.spec_from_file_location("stage7", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stage7"] = mod
    assert spec.loader is not None
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod


def test_discover_mesh_jobs_filters_indices(tmp_path):
    mod = _load_stage7_module()

    upsampled_dir = tmp_path / "upsampled"
    shape_dir = tmp_path / "shape"
    texture_dir = tmp_path / "texture"
    upsampled_dir.mkdir()
    shape_dir.mkdir()
    texture_dir.mkdir()

    (upsampled_dir / "iter_1_transparent.png").write_bytes(b"x")
    (upsampled_dir / "iter_2_transparent.png").write_bytes(b"x")
    (upsampled_dir / "ignore.png").write_bytes(b"x")

    jobs = mod.discover_mesh_jobs(str(upsampled_dir), str(shape_dir), str(texture_dir), allowed_indices={2})

    assert len(jobs) == 1
    assert jobs[0].idx == 2
    assert jobs[0].mesh_name == "iter_2"
    assert jobs[0].shape_fpath.endswith("iter_2_shape.obj")
    assert jobs[0].texture_fpath.endswith("iter_2_mesh.glb")
