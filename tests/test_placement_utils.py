# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for predicate-based spatial placement (simfoundry/utils/placement_utils.py).

Uses stub objects, no OmniGibson required. Includes regression guards that pin
the behavior of the six original predicates and the overlap separator.
"""

import math
import random
from unittest import mock

import pytest
import torch as th

from simfoundry.utils.placement_utils import (
    PREDICATES,
    _aabb_overlap,
    place_with_predicate,
    resolve_gap,
    separate_overlapping_objects,
)


class StubLink:
    def __init__(self, lo, hi):
        self.visual_aabb = (th.tensor(lo, dtype=th.float32), th.tensor(hi, dtype=th.float32))


class StubObject:
    """Minimal object exposing the members placement_utils touches."""

    def __init__(self, half_extents, position, name="stub", links=None):
        self.half = th.tensor(half_extents, dtype=th.float32)
        self.pos = th.tensor(position, dtype=th.float32)
        self.ori = th.tensor([0.0, 0.0, 0.0, 1.0])
        self.name = name
        self.links = links or {}

    @property
    def aabb(self):
        return self.pos - self.half, self.pos + self.half

    def get_position_orientation(self):
        return self.pos.clone(), self.ori.clone()

    def set_position_orientation(self, position, orientation):
        self.pos = th.as_tensor(position, dtype=th.float32)
        self.ori = th.as_tensor(orientation)

    def keep_still(self):
        pass


def make_pair(obj_half=(0.05, 0.05, 0.05), obj_pos=(1.0, 1.0, 0.05),
              ref_half=(0.2, 0.15, 0.1), ref_pos=(0.0, 0.0, 0.1)):
    return StubObject(obj_half, obj_pos, "obj"), StubObject(ref_half, ref_pos, "ref")


BOUNDS = (th.tensor([-1.0, -1.0, 0.0]), th.tensor([1.0, 1.0, 1.0]))


class TestResolveGap:
    def test_scalar(self):
        assert resolve_gap(0.07, "left_of") == 0.07

    def test_range_sampled(self):
        random.seed(0)
        for _ in range(20):
            assert 0.02 <= resolve_gap([0.02, 0.04], "near") <= 0.04

    def test_dict_per_predicate(self):
        cfg = {"near": 0.09, "between": [0.01, 0.01], "default": 0.03}
        assert resolve_gap(cfg, "near") == 0.09
        assert resolve_gap(cfg, "between") == 0.01

    def test_dict_default_fallback(self):
        assert resolve_gap({"default": 0.03}, "inside_link") == 0.03

    def test_dict_missing_default(self):
        assert resolve_gap({"left_of": 0.1}, "near") == 0.05


class TestLegacyPredicatesRegression:
    """Pin the original behavior of the pre-existing predicates."""

    def test_predicates_list(self):
        assert PREDICATES == ["on_top", "left_of", "right_of", "behind", "in_front_of",
                              "inside", "near", "between", "inside_link"]

    def test_on_top_exact(self):
        obj, ref = make_pair()
        pos, _, _ = place_with_predicate(obj, ref, "on_top", gap=0.05, z_offset=0.01)
        assert th.allclose(pos, th.tensor([0.0, 0.0, 0.2 + 0.05 + 0.01]))

    def test_left_of_aligned_exact(self):
        obj, ref = make_pair()
        pos, _, _ = place_with_predicate(obj, ref, "left_of", gap=0.02, aligned=True)
        # X centered on ref, Y = ref_hi_y + obj_half_y + gap, Z preserved
        assert th.allclose(pos, th.tensor([0.0, 0.15 + 0.05 + 0.02, 0.05]))

    def test_unknown_predicate_raises(self):
        obj, ref = make_pair()
        with pytest.raises(ValueError, match="Unknown predicate"):
            place_with_predicate(obj, ref, "levitating_over", gap=0.05)


class TestNear:
    def test_forced_direction_exact(self):
        obj, ref = make_pair()
        # random.uniform(0, 2*pi) -> 0 (theta = 0 -> +X direction)
        with mock.patch("simfoundry.utils.placement_utils.random.uniform", side_effect=lambda a, b: a):
            pos, _, _ = place_with_predicate(obj, ref, "near", gap=0.03)
        # dist = ref_half_x + obj_half_x + gap along +X; Y at ref center; Z preserved
        assert th.allclose(pos, th.tensor([0.2 + 0.05 + 0.03, 0.0, 0.05]))

    def test_forced_diagonal_direction(self):
        obj, ref = make_pair()
        theta = math.pi / 2  # +Y
        with mock.patch("simfoundry.utils.placement_utils.random.uniform", return_value=theta):
            pos, _, _ = place_with_predicate(obj, ref, "near", gap=0.03)
        assert th.allclose(pos, th.tensor([0.0, 0.15 + 0.05 + 0.03, 0.05]), atol=1e-6)

    def test_never_overlaps_reference(self):
        random.seed(1234)
        for _ in range(200):
            obj, ref = make_pair()
            place_with_predicate(obj, ref, "near", gap=0.01)
            assert not _aabb_overlap(obj, ref), f"near placed overlapping at {obj.pos}"

    def test_z_offset_applied(self):
        obj, ref = make_pair()
        random.seed(0)
        pos, _, _ = place_with_predicate(obj, ref, "near", gap=0.03, z_offset=0.02)
        assert abs(float(pos[2]) - 0.07) < 1e-6

    def test_clamps_to_bounds(self):
        random.seed(7)
        tight = (th.tensor([-0.3, -0.3, 0.0]), th.tensor([0.3, 0.3, 1.0]))
        for _ in range(50):
            obj, ref = make_pair()
            place_with_predicate(obj, ref, "near", gap=0.02, bounds=tight)
            lo, hi = obj.aabb
            assert float(lo[0]) >= -0.3 - 1e-6 and float(hi[0]) <= 0.3 + 1e-6
            assert float(lo[1]) >= -0.3 - 1e-6 and float(hi[1]) <= 0.3 + 1e-6


class TestBetween:
    def test_on_segment_and_disjoint(self):
        random.seed(42)
        for _ in range(50):
            obj = StubObject((0.03, 0.03, 0.03), (2.0, 2.0, 0.03), "obj")
            ref1 = StubObject((0.1, 0.1, 0.05), (0.0, 0.0, 0.05), "ref1")
            ref2 = StubObject((0.08, 0.08, 0.05), (0.8, 0.4, 0.05), "ref2")
            pos, _, _ = place_with_predicate(obj, ref1, "between", gap=0.02,
                                             reference_obj_2=ref2)
            # Collinear with the segment between centers (2D cross product ~ 0)
            v1 = (pos[:2] - ref1.pos[:2])
            v2 = (ref2.pos[:2] - ref1.pos[:2])
            cross = float(v1[0] * v2[1] - v1[1] * v2[0])
            assert abs(cross) < 1e-5
            assert not _aabb_overlap(obj, ref1)
            assert not _aabb_overlap(obj, ref2)

    def test_forced_t_exact(self):
        obj = StubObject((0.03, 0.03, 0.03), (2.0, 2.0, 0.03), "obj")
        ref1 = StubObject((0.1, 0.1, 0.05), (0.0, 0.0, 0.05), "ref1")
        ref2 = StubObject((0.1, 0.1, 0.05), (1.0, 0.0, 0.05), "ref2")
        # random.uniform(t_min, t_max) -> t_min = 0.1 + 0.03 + 0.02 (all along +X)
        with mock.patch("simfoundry.utils.placement_utils.random.uniform", side_effect=lambda a, b: a):
            pos, _, _ = place_with_predicate(obj, ref1, "between", gap=0.02,
                                             reference_obj_2=ref2)
        assert th.allclose(pos, th.tensor([0.15, 0.0, 0.03]), atol=1e-6)

    def test_degenerate_identical_centers(self, caplog):
        obj = StubObject((0.03, 0.03, 0.03), (2.0, 2.0, 0.03), "obj")
        ref1 = StubObject((0.1, 0.1, 0.05), (0.5, 0.5, 0.05), "ref1")
        ref2 = StubObject((0.2, 0.2, 0.05), (0.5, 0.5, 0.05), "ref2")
        with caplog.at_level("WARNING", logger="simfoundry.utils.placement_utils"):
            pos, _, _ = place_with_predicate(obj, ref1, "between", reference_obj_2=ref2)
        assert th.allclose(pos[:2], th.tensor([0.5, 0.5]))
        assert any("coincident centers" in r.message for r in caplog.records)

    def test_no_free_span_falls_back_to_midpoint(self, caplog):
        obj = StubObject((0.05, 0.05, 0.05), (2.0, 2.0, 0.05), "obj")
        ref1 = StubObject((0.1, 0.1, 0.05), (0.0, 0.0, 0.05), "ref1")
        ref2 = StubObject((0.1, 0.1, 0.05), (0.25, 0.0, 0.05), "ref2")  # too close
        with caplog.at_level("WARNING", logger="simfoundry.utils.placement_utils"):
            pos, _, _ = place_with_predicate(obj, ref1, "between", gap=0.02,
                                             reference_obj_2=ref2)
        assert th.allclose(pos[:2], th.tensor([0.125, 0.0]), atol=1e-6)
        assert any("no free span" in r.message for r in caplog.records)

    def test_missing_second_reference_raises(self):
        obj, ref = make_pair()
        with pytest.raises(ValueError, match="between"):
            place_with_predicate(obj, ref, "between", gap=0.02)


class TestInsideLink:
    def make_shelf(self):
        links = {"shelf_level_2": StubLink([-0.15, -0.1, 0.3], [0.15, 0.1, 0.5])}
        return StubObject((0.2, 0.15, 0.4), (0.0, 0.0, 0.4), "shelf", links=links)

    def test_xy_within_shrunk_link_aabb(self):
        random.seed(3)
        for _ in range(100):
            obj = StubObject((0.03, 0.03, 0.04), (1.0, 1.0, 0.04), "obj")
            shelf = self.make_shelf()
            pos, _, _ = place_with_predicate(obj, shelf, "inside_link",
                                             link_name="shelf_level_2")
            assert -0.15 + 0.03 - 1e-6 <= float(pos[0]) <= 0.15 - 0.03 + 1e-6
            assert -0.1 + 0.03 - 1e-6 <= float(pos[1]) <= 0.1 - 0.03 + 1e-6

    def test_z_exact(self):
        obj = StubObject((0.03, 0.03, 0.04), (1.0, 1.0, 0.04), "obj")
        shelf = self.make_shelf()
        pos, _, _ = place_with_predicate(obj, shelf, "inside_link",
                                         link_name="shelf_level_2", z_offset=0.005)
        assert abs(float(pos[2]) - (0.3 + 0.04 + 0.005)) < 1e-6

    def test_missing_link_falls_back_to_object_aabb(self, caplog):
        obj = StubObject((0.03, 0.03, 0.04), (1.0, 1.0, 0.04), "obj")
        shelf = self.make_shelf()
        with caplog.at_level("WARNING", logger="simfoundry.utils.placement_utils"):
            pos, _, _ = place_with_predicate(obj, shelf, "inside_link",
                                             link_name="no_such_link")
        assert any("not found" in r.message for r in caplog.records)
        # Falls back to the shelf's own AABB: z = (0.4 - 0.4) + 0.04
        assert abs(float(pos[2]) - 0.04) < 1e-6

    def test_missing_link_name_raises(self):
        obj, ref = make_pair()
        with pytest.raises(ValueError, match="inside_link"):
            place_with_predicate(obj, ref, "inside_link")

    def test_object_wider_than_link_centers_axis(self, caplog):
        obj = StubObject((0.5, 0.03, 0.04), (1.0, 1.0, 0.04), "obj")  # wider than link X
        shelf = self.make_shelf()
        with caplog.at_level("WARNING", logger="simfoundry.utils.placement_utils"):
            pos, _, _ = place_with_predicate(obj, shelf, "inside_link",
                                             link_name="shelf_level_2")
        assert abs(float(pos[0]) - 0.0) < 1e-6  # centered on link X


class TestSeparation:
    def test_horizontal_pair_exact_legacy_shift(self):
        # Two overlapping objects, later one placed left_of -> shifted +X by
        # max extent * multiplier. Pins the original algorithm exactly.
        a = StubObject((0.05, 0.05, 0.05), (0.0, 0.0, 0.05), "a")
        b = StubObject((0.04, 0.04, 0.04), (0.02, 0.0, 0.04), "b")
        separate_overlapping_objects([(a, "in_front_of"), (b, "left_of")], multiplier=1.5)
        assert th.allclose(b.pos, th.tensor([0.02 + 0.1 * 1.5, 0.0, 0.04]))
        assert th.allclose(a.pos, th.tensor([0.0, 0.0, 0.05]))  # anchor unmoved

    def test_skips_on_top_pairs(self):
        a = StubObject((0.05, 0.05, 0.05), (0.0, 0.0, 0.05), "a")
        b = StubObject((0.04, 0.04, 0.04), (0.02, 0.0, 0.04), "b")
        separate_overlapping_objects([(a, "on_top"), (b, "left_of")])
        assert th.allclose(b.pos, th.tensor([0.02, 0.0, 0.04]))  # untouched

    def test_skips_between_and_inside_link(self):
        a = StubObject((0.05, 0.05, 0.05), (0.0, 0.0, 0.05), "a")
        b = StubObject((0.04, 0.04, 0.04), (0.02, 0.0, 0.04), "b")
        separate_overlapping_objects([(a, "between"), (b, "near")])
        assert th.allclose(b.pos, th.tensor([0.02, 0.0, 0.04]))
        separate_overlapping_objects([(a, "inside_link"), (b, "left_of")])
        assert th.allclose(b.pos, th.tensor([0.02, 0.0, 0.04]))

    def test_near_later_radial_push(self):
        a = StubObject((0.05, 0.05, 0.05), (0.0, 0.0, 0.05), "a")
        b = StubObject((0.04, 0.04, 0.04), (0.03, 0.04, 0.04), "b")
        separate_overlapping_objects([(a, "left_of"), (b, "near")], multiplier=1.5)
        # Pushed radially along normalize((0.03, 0.04)) by 0.1 * 1.5
        direction = th.tensor([0.6, 0.8])
        expected_xy = th.tensor([0.03, 0.04]) + direction * 0.15
        assert th.allclose(b.pos[:2], expected_xy, atol=1e-6)
        assert not _aabb_overlap(a, b)

    def test_near_anchor_allows_axis_shift(self):
        a = StubObject((0.05, 0.05, 0.05), (0.0, 0.0, 0.05), "a")
        b = StubObject((0.04, 0.04, 0.04), (0.02, 0.0, 0.04), "b")
        separate_overlapping_objects([(a, "near"), (b, "behind")], multiplier=1.5)
        # behind -> perpendicular axis Y, shift +Y by 0.1 * 1.5
        assert th.allclose(b.pos, th.tensor([0.02, 0.15, 0.04]))
