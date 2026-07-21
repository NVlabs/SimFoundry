from pathlib import Path

import pytest

from digital_cousins.pipeline.stream_subsequence import (
    discover_ready_artifact_mtimes,
    discover_ready_indices,
    per_index_override,
    subsequence_complete,
    validate_subsequence,
)


class _Node:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_validate_subsequence():
    assert validate_subsequence(5, 8) == [5, 6, 7, 8]
    assert validate_subsequence(6, 7) == [6, 7]
    with pytest.raises(ValueError):
        validate_subsequence(8, 5)


def test_per_index_override_keys():
    assert per_index_override(6, 3) == ["s6_upsample.object_indices=[3]"]
    assert per_index_override(7, 4) == ["s7_mesh.object_indices=[4]"]
    assert per_index_override(8, 5) == ["s8_pose.object_indices=[5]"]
    assert per_index_override(5, 1) == []


def test_subsequence_complete_waits_for_expected_final_indices():
    assert not subsequence_complete([5, 6, 7, 8], set(), {8: set()})
    assert not subsequence_complete([5, 6, 7, 8], {0, 1}, {8: {0}})
    assert subsequence_complete([5, 6, 7, 8], {0, 1}, {8: {0, 1, 2}})


def test_discover_ready_indices(tmp_path):
    scene = tmp_path / "scene"
    (scene / "s5_scene_gmask" / "obj_cat_list").mkdir(parents=True)
    (scene / "s6_upsample" / "upsampled").mkdir(parents=True)
    (scene / "s7_mesh" / "textured_mesh" / "hunyuan").mkdir(parents=True)
    (scene / "s8_pose" / "info").mkdir(parents=True)

    for name in ["iter_0.json", "iter_2.json"]:
        (scene / "s5_scene_gmask" / "obj_cat_list" / name).write_text("{}", encoding="utf-8")
    for name in ["iter_1_transparent.png", "iter_3_transparent.png"]:
        (scene / "s6_upsample" / "upsampled" / name).write_bytes(b"x")
    for name in ["iter_2_mesh.glb", "iter_4_mesh.glb"]:
        (scene / "s7_mesh" / "textured_mesh" / "hunyuan" / name).write_bytes(b"x")
    for name in ["iter_2.json", "iter_4.json"]:
        (scene / "s8_pose" / "info" / name).write_text("{}", encoding="utf-8")

    cfg = _Node(
        s5_scene_gmask=_Node(out_dir=str(scene / "s5_scene_gmask")),
        s6_upsample=_Node(out_dir=str(scene / "s6_upsample")),
        s7_mesh=_Node(out_dir=str(scene / "s7_mesh"), texture_model="hunyuan"),
        s8_pose=_Node(out_dir=str(scene / "s8_pose")),
    )

    assert discover_ready_indices(cfg, 5) == [0, 2]
    assert discover_ready_indices(cfg, 6) == [1, 3]
    assert discover_ready_indices(cfg, 7) == [2, 4]
    assert discover_ready_indices(cfg, 8) == [2, 4]
    mtimes = discover_ready_artifact_mtimes(cfg, 6)
    assert sorted(mtimes) == [1, 3]
    assert all(value > 0 for value in mtimes.values())
