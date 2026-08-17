# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Front canonicalization: yaw math, VLM answer parsing, and orientation stamps."""

import json

import numpy as np
import pytest

from simfoundry.pipeline.front_canonicalization import (
    FRONT_TARGET_AZIMUTH_DEG,
    VIEW_AZIMUTHS_DEG,
    applied_yaw_deg,
    canonicalize_front,
    parse_front_choice,
    read_orientation_yaw,
    yaw_rotation,
    yaw_snap_to_axes,
    yaw_to_target_rotation,
)


def _azimuth_dir(deg):
    a = np.deg2rad(deg)
    return np.array([np.cos(a), 0.0, np.sin(a)])


@pytest.mark.parametrize("front_az", VIEW_AZIMUTHS_DEG)
def test_rotation_maps_front_to_target(front_az):
    rot = yaw_to_target_rotation(front_az)
    mapped = rot @ _azimuth_dir(front_az)
    assert np.allclose(mapped, _azimuth_dir(FRONT_TARGET_AZIMUTH_DEG), atol=1e-9)
    assert np.allclose(rot @ np.array([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0])  # up preserved
    assert np.isclose(np.linalg.det(rot), 1.0)


def test_applied_yaw_is_normalized():
    assert applied_yaw_deg(90) == 0.0
    assert applied_yaw_deg(180) == 90.0
    assert applied_yaw_deg(315) == -135.0
    assert applied_yaw_deg(270) == -180.0


def test_parse_front_choice():
    labels = list("ABCDEFGH")
    assert parse_front_choice("C", labels) == "C"
    assert parse_front_choice("view D shows the front.", labels) == "D"
    assert parse_front_choice("NONE", labels) is None
    assert parse_front_choice("none - no clear front", labels) is None
    assert parse_front_choice("", labels) is None
    assert parse_front_choice("ANSWER", labels) is None  # no bare label token


def test_read_orientation_yaw(tmp_path):
    fpath = tmp_path / "iter_0_orientation.json"
    assert read_orientation_yaw(str(fpath)) == 0.0  # legacy: no stamp
    fpath.write_text(json.dumps({"applied_yaw_deg": -90.0, "status": "rotated"}))
    assert read_orientation_yaw(str(fpath)) == -90.0
    fpath.write_text("not json")
    assert read_orientation_yaw(str(fpath)) == 0.0


class _StubVLM:
    def __init__(self, answer):
        self.answer = answer
        self.image_paths = None

    def __call__(self, prompt, image_paths, **kwargs):
        self.image_paths = image_paths
        return self.answer

    def get_result_text(self, result):
        return result


def test_canonicalize_front_rotates_red_bin(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    glb = "Data/CubeInMail/s7_mesh/textured_mesh/hunyuan/iter_0_mesh.glb"
    import os
    if not os.path.isfile(glb):
        pytest.skip("red_bin fixture mesh not available")
    mesh = trimesh.load(glb, force="mesh")
    # red_bin's label faces -X = azimuth 180 = view E.
    vlm = _StubVLM("E")
    rot, info = canonicalize_front(mesh, str(tmp_path), vlm=vlm, refine=False)
    assert info["status"] == "rotated"
    assert info["front_azimuth_deg"] == 180
    assert info["applied_yaw_deg"] == 90.0
    assert len(vlm.image_paths) == len(VIEW_AZIMUTHS_DEG)  # no photo passed
    # Label normal -X must land on +Z.
    assert np.allclose(rot @ np.array([-1.0, 0.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-9)


class _SeqVLM:
    """Stub answering a fixed sequence: one answer per VLM call."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, prompt, image_paths, **kwargs):
        self.calls.append((prompt, list(image_paths)))
        return self.answers.pop(0)

    def get_result_text(self, result):
        return result


def test_canonicalize_front_refines_yaw(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=(1, 0.5, 0.8))
    # Coarse pick E = azimuth 180; fine views span 157.5..202.5 in 7.5-deg steps
    # (labels A..G), so fine pick C = 180 - 7.5 = 172.5.
    vlm = _SeqVLM(["E", "C"])
    rot, info = canonicalize_front(mesh, str(tmp_path), vlm=vlm, refine=True)
    assert info["status"] == "rotated"
    assert info["front_azimuth_coarse_deg"] == 180
    assert info["front_azimuth_deg"] == 172.5
    assert info["refine_status"] == "refined"
    assert info["refine_delta_deg"] == -7.5
    assert info["applied_yaw_deg"] == 82.5
    assert len(vlm.calls) == 2
    assert len(vlm.calls[1][1]) == 7  # fine pass renders 7 views
    # The final rotation maps the refined front azimuth onto the +Z target.
    assert np.allclose(rot @ _azimuth_dir(172.5), _azimuth_dir(90), atol=1e-9)


def test_canonicalize_front_refine_falls_back_to_coarse(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=(1, 0.5, 0.8))
    vlm = _SeqVLM(["E", "NONE"])
    rot, info = canonicalize_front(mesh, str(tmp_path), vlm=vlm, refine=True)
    assert info["status"] == "rotated"
    assert info["front_azimuth_deg"] == 180
    assert info["refine_status"] == "ambiguous"
    assert info["applied_yaw_deg"] == 90.0


def test_canonicalize_front_ambiguous_and_error(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    rot, info = canonicalize_front(mesh, str(tmp_path / "a"), vlm=_StubVLM("NONE"))
    assert rot is None and info["status"] == "ambiguous"
    assert info["applied_yaw_deg"] == 0.0

    class _Boom:
        def __call__(self, *a, **k):
            raise RuntimeError("vlm down")

    rot, info = canonicalize_front(mesh, str(tmp_path / "b"), vlm=_Boom())
    assert rot is None and info["status"].startswith("error")


def test_yaw_snap_recovers_known_rotation():
    rng = np.random.default_rng(3)
    # Box-like footprint: dense points in a w x h rectangle at random heights.
    xz = rng.uniform([-0.5, -0.35], [0.5, 0.35], size=(500, 2))
    verts = np.column_stack([xz[:, 0], rng.uniform(0, 1, 500), xz[:, 1]])
    for true_yaw in (17.0, -12.5, 31.0):
        rotated = verts @ yaw_rotation(true_yaw).T
        snap_rot, info = yaw_snap_to_axes(rotated)
        assert info["status"] == "snapped"
        # Randomly sampled footprints put the hull edges up to ~1 deg off the true sides.
        assert np.isclose(info["snap_yaw_deg"], -true_yaw, atol=1.0)
        # Applying the snap restores axis alignment.
        assert np.allclose(snap_rot @ yaw_rotation(true_yaw), np.eye(3), atol=0.02)


def test_yaw_snap_gates_round_footprint():
    rng = np.random.default_rng(4)
    t = rng.uniform(0, 2 * np.pi, 800)
    r = np.sqrt(rng.uniform(0, 1, 800))
    verts = np.column_stack([r * np.cos(t), rng.uniform(0, 1, 800), r * np.sin(t)])
    snap_rot, info = yaw_snap_to_axes(verts)
    assert snap_rot is None
    assert info["status"] == "not_boxy"
    assert info["rectangularity"] < 0.80


def test_canonicalize_front_skips_in_test_mode(tmp_path, monkeypatch):
    trimesh = pytest.importorskip("trimesh")
    monkeypatch.setenv("TEST_MODE", "1")
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    rot, info = canonicalize_front(mesh, str(tmp_path), vlm=_StubVLM("A"))
    assert rot is None and info["status"] == "skipped_test_mode"
