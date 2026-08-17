# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for baking a mesh's upright orientation out of stage 8's fitted pose.

Pure-math invariants only (numpy): no GPU, meshes, or registration involved. The key
contract is that for any fitted rotation, the tilt returned by tilt_from_fit uprights
the mesh while the adjusted pose still places it exactly where the fit did, with the
residual rotation reduced to pure gravity yaw.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from simfoundry.pipeline.front_canonicalization import (
    orientation_stamp_changed,
    read_orientation_stamp,
    read_orientation_yaw,
)
from simfoundry.pipeline.upright_from_fit import (
    MESH_UP,
    decide_tilt,
    rotation_between,
    tilt_from_fit,
    up_axis_spread_deg,
    up_in_mesh_frame,
)


def _rand_unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _yaw_about(axis, angle_rad):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)


def _assert_proper_rotation(rot):
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-10)


class TestRotationBetween:
    def test_maps_from_onto_to(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            a, b = _rand_unit(rng), _rand_unit(rng)
            rot = rotation_between(a, b)
            _assert_proper_rotation(rot)
            assert np.allclose(rot @ a, b, atol=1e-10)

    def test_parallel_is_identity(self):
        v = np.array([0.3, -0.9, 0.1])
        assert np.allclose(rotation_between(v, v), np.eye(3), atol=1e-12)

    def test_antiparallel(self):
        for v in ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.6, -0.48, 0.64]):
            v = np.asarray(v) / np.linalg.norm(v)
            rot = rotation_between(v, -v)
            _assert_proper_rotation(rot)
            assert np.allclose(rot @ v, -v, atol=1e-10)

    def test_zero_vector_rejected(self):
        with pytest.raises(ValueError):
            rotation_between([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])


class TestTiltFromFit:
    def test_recovers_synthetic_tilt(self):
        """A fit that sends the object's semantic up v onto gravity g (plus arbitrary
        gravity yaw) must yield a tilt that uprights v, and an adjusted pose that sends
        MESH_UP exactly onto g — i.e. the residual pose rotation is pure gravity yaw."""
        rng = np.random.default_rng(1)
        for _ in range(50):
            v = _rand_unit(rng)          # object's true up, in mesh coordinates
            g = _rand_unit(rng)          # scene gravity, in the fitted cloud's frame
            yaw = _yaw_about(g, rng.uniform(-np.pi, np.pi))
            rot_fit = yaw @ rotation_between(v, g)

            assert np.allclose(up_in_mesh_frame(rot_fit, g), v, atol=1e-9)

            tilt_deg, rot_tilt = tilt_from_fit(rot_fit, g)
            expected = np.degrees(np.arccos(np.clip(v @ MESH_UP, -1.0, 1.0)))
            assert np.isclose(tilt_deg, expected, atol=1e-6)
            _assert_proper_rotation(rot_tilt)
            # The baked mesh is upright: its true up moves onto the +Y convention.
            assert np.allclose(rot_tilt @ v, MESH_UP, atol=1e-9)
            # The adjusted pose sends +Y exactly onto gravity.
            adjusted = rot_fit @ rot_tilt.T
            assert np.allclose(adjusted @ MESH_UP, g, atol=1e-9)

    def test_pose_composition_places_points_identically(self):
        """p_cam = s*R_fit@p + t must equal s*(R_fit@rot_tilt.T)@(rot_tilt@p) + t."""
        rng = np.random.default_rng(2)
        rot_fit = rotation_between(_rand_unit(rng), _rand_unit(rng))
        g = _rand_unit(rng)
        _, rot_tilt = tilt_from_fit(rot_fit, g)
        s, t = 0.37, rng.normal(size=3)
        pts = rng.normal(size=(100, 3))
        direct = s * pts @ rot_fit.T + t
        via_canonical = s * (pts @ rot_tilt.T) @ (rot_fit @ rot_tilt.T).T + t
        assert np.allclose(direct, via_canonical, atol=1e-9)


class TestDecideTilt:
    G = np.array([0.0, 0.0, 1.0])

    @staticmethod
    def _info(rot):
        return {"tf_z_up": SimpleNamespace(rot=rot)}

    def _fit_with_tilt(self, tilt_deg, yaw_deg=30.0):
        v = _yaw_about([1.0, 0.0, 0.0], np.deg2rad(tilt_deg)) @ MESH_UP
        return _yaw_about(self.G, np.deg2rad(yaw_deg)) @ rotation_between(v, self.G)

    def test_bakes_agreeing_tilted_fits(self):
        rot = self._fit_with_tilt(34.0)
        rot_tilt, info = decide_tilt(
            [self._info(rot), self._info(self._fit_with_tilt(34.0, yaw_deg=120.0))],
            self.G, min_tilt_deg=10.0, max_tilt_deg=80.0, consensus_deg=20.0,
        )
        assert info["status"] == "baked"
        assert np.isclose(info["applied_tilt_deg"], 34.0, atol=1e-6)
        assert rot_tilt is not None

    def test_upright_mesh_below_threshold(self):
        rot_tilt, info = decide_tilt(
            [self._info(self._fit_with_tilt(3.0))],
            self.G, min_tilt_deg=10.0, max_tilt_deg=80.0, consensus_deg=20.0,
        )
        assert rot_tilt is None
        assert info["status"] == "below_threshold"
        assert info["applied_tilt_deg"] == 0.0

    def test_implausible_tilt_rejected(self):
        rot_tilt, info = decide_tilt(
            [self._info(self._fit_with_tilt(150.0))],
            self.G, min_tilt_deg=10.0, max_tilt_deg=80.0, consensus_deg=20.0,
        )
        assert rot_tilt is None
        assert info["status"] == "implausible_tilt"

    def test_disagreeing_fits_rejected(self):
        rot_tilt, info = decide_tilt(
            [self._info(self._fit_with_tilt(34.0)),
             self._info(rotation_between([1.0, 0.0, 0.0], self.G))],
            self.G, min_tilt_deg=10.0, max_tilt_deg=80.0, consensus_deg=20.0,
        )
        assert rot_tilt is None
        assert info["status"] == "inconsistent_fits"


class TestOrientationStamp:
    def test_legacy_yaw_only_file(self, tmp_path):
        fpath = tmp_path / "iter_0_orientation.json"
        fpath.write_text(json.dumps({"applied_yaw_deg": -90.0}))
        assert read_orientation_stamp(str(fpath)) == (-90.0, 0.0)
        assert read_orientation_yaw(str(fpath)) == -90.0

    def test_full_stamp_and_missing_file(self, tmp_path):
        fpath = tmp_path / "iter_0_orientation.json"
        assert read_orientation_stamp(str(fpath)) == (0.0, 0.0)
        fpath.write_text(json.dumps({"applied_yaw_deg": 45.0, "applied_tilt_deg": 33.8}))
        assert read_orientation_stamp(str(fpath)) == (45.0, 33.8)

    def test_changed_detection(self):
        assert orientation_stamp_changed((0.0, 0.0), (45.0, 0.0))          # coarse yaw change
        assert orientation_stamp_changed((45.0, 0.0), (45.0, 33.8))       # tilt appears
        assert not orientation_stamp_changed((45.0, 33.8), (45.0, 34.5))  # CPD jitter
        assert orientation_stamp_changed((45.0, 33.8), (45.0, 12.0))      # real tilt change
        assert not orientation_stamp_changed((45.0, 0.0), (52.5, 0.0))    # refine-bin jitter
        assert not orientation_stamp_changed((-180.0, 0.0), (180.0, 0.0))  # same angle on circle
