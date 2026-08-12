# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifacts become visible to a downstream stage only once they are complete.

The streaming subsequence dispatches a consumer as soon as a producer's artifact appears in
the watched directory. Writing in place makes that a race, which is how stage 8 came to read
a half-copied .glb as a mesh with zero triangles.
"""

import os

import pytest

from simfoundry.utils.python_utils import (
    PARTIAL_MARKER,
    atomic_copyfile,
    atomic_output_path,
    partial_path,
)


def test_partial_path_keeps_the_extension():
    # trimesh and PIL choose their output format from the extension, so the marker has to go
    # before it -- `foo.glb.partial` would not export as a GLB.
    assert partial_path("/a/b/iter_3_mesh.glb") == "/a/b/iter_3_mesh.partial.glb"
    assert partial_path("/a/b/iter_0_transparent.png") == "/a/b/iter_0_transparent.partial.png"
    assert partial_path("/a/b/iter_1.json") == "/a/b/iter_1.partial.json"


def test_partial_name_fails_the_watched_suffix_match():
    # This is what keeps an in-progress mesh or image invisible to the watcher.
    assert not partial_path("iter_3_mesh.glb").endswith("_mesh.glb")
    assert not partial_path("iter_0_transparent.png").endswith("_transparent.png")


def test_final_path_does_not_exist_until_the_block_completes(tmp_path):
    out = tmp_path / "iter_0_mesh.glb"
    with atomic_output_path(out) as tmp:
        with open(tmp, "w") as f:
            f.write("partially written")
        assert not out.exists(), "consumer could observe the artifact mid-write"
        assert os.path.isfile(tmp)
    assert out.read_text() == "partially written"


def test_a_failed_write_leaves_no_artifact_and_no_temp(tmp_path):
    out = tmp_path / "iter_0_mesh.glb"
    with pytest.raises(RuntimeError, match="generation blew up"):
        with atomic_output_path(out) as tmp:
            with open(tmp, "w") as f:
                f.write("half a mesh")
            raise RuntimeError("generation blew up")
    assert not out.exists(), "a failed run must not publish a corrupt artifact"
    assert not os.path.exists(partial_path(str(out))), "temp file left behind"


def test_publish_replaces_an_existing_artifact(tmp_path):
    out = tmp_path / "iter_0.json"
    out.write_text("old")
    with atomic_output_path(out) as tmp:
        with open(tmp, "w") as f:
            f.write("new")
    assert out.read_text() == "new"


def test_atomic_copyfile_publishes_the_whole_file(tmp_path):
    src = tmp_path / "src.glb"
    src.write_bytes(b"x" * 4096)
    dst = tmp_path / "iter_2_mesh.glb"
    atomic_copyfile(src, dst)
    assert dst.read_bytes() == b"x" * 4096
    assert not os.path.exists(partial_path(str(dst)))


# --------------------------------------------------------------------------------------
# Watcher side
# --------------------------------------------------------------------------------------

class _Node:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _mesh_cfg(tmp_path):
    scene = tmp_path / "scene"
    (scene / "s7_mesh" / "textured_mesh" / "trellis2").mkdir(parents=True)
    return _Node(s7_mesh=_Node(out_dir=str(scene / "s7_mesh"), texture_model="trellis2"))


def test_watcher_ignores_a_partial_artifact(tmp_path):
    from simfoundry.pipeline.stream_subsequence import discover_ready_indices

    cfg = _mesh_cfg(tmp_path)
    d = tmp_path / "scene" / "s7_mesh" / "textured_mesh" / "trellis2"
    (d / "iter_0_mesh.partial.glb").write_bytes(b"half")
    assert discover_ready_indices(cfg, 7, settle_s=0) == []

    # ...and picks it up once it is renamed into place.
    os.replace(d / "iter_0_mesh.partial.glb", d / "iter_0_mesh.glb")
    assert discover_ready_indices(cfg, 7, settle_s=0) == [0]


def test_watcher_ignores_a_partial_json_whose_suffix_still_matches(tmp_path):
    # `.json` is the watched suffix for stages 5 and 8, so `iter_1.partial.json` still ends
    # with it -- the explicit marker check is what rejects it.
    from simfoundry.pipeline.stream_subsequence import discover_ready_indices

    scene = tmp_path / "scene"
    (scene / "s5_scene" / "obj_cat_list").mkdir(parents=True)
    cfg = _Node(s5_scene=_Node(out_dir=str(scene / "s5_scene")))
    (scene / "s5_scene" / "obj_cat_list" / "iter_1.partial.json").write_text("{}")
    assert PARTIAL_MARKER in "iter_1.partial.json"
    assert "iter_1.partial.json".endswith(".json")
    assert discover_ready_indices(cfg, 5, settle_s=0) == []
