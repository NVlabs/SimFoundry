# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure AABB geometry helpers behind the task predicates."""

from simfoundry.tasks.geometry import (
    aabb_above,
    aabb_on_top,
    aabb_xy_overlap_fraction,
    is_lifted,
)


def box(cx, cy, cz, hx, hy, hz):
    """AABB corners (lo, hi) for a box centered at (cx, cy, cz) with half-extents (hx, hy, hz)."""
    return [cx - hx, cy - hy, cz - hz], [cx + hx, cy + hy, cz + hz]


class TestXYOverlapFraction:
    def test_full_overlap(self):
        inner = box(0, 0, 0, 0.1, 0.1, 0.1)
        outer = box(0, 0, 0, 0.5, 0.5, 0.1)
        assert aabb_xy_overlap_fraction(*inner, *outer) == 1.0

    def test_partial_overlap(self):
        # Inner is 0.2 x 0.2; shifted so exactly half its footprint overlaps
        inner = box(0.5, 0, 0, 0.1, 0.1, 0.1)
        outer = box(0, 0, 0, 0.5, 0.5, 0.1)
        assert abs(aabb_xy_overlap_fraction(*inner, *outer) - 0.5) < 1e-9

    def test_disjoint(self):
        inner = box(2.0, 0, 0, 0.1, 0.1, 0.1)
        outer = box(0, 0, 0, 0.5, 0.5, 0.1)
        assert aabb_xy_overlap_fraction(*inner, *outer) == 0.0

    def test_degenerate_inner_returns_zero(self):
        # Zero-area inner footprint must not divide by zero
        inner = box(0, 0, 0, 0.0, 0.0, 0.1)
        outer = box(0, 0, 0, 0.5, 0.5, 0.1)
        assert aabb_xy_overlap_fraction(*inner, *outer) == 0.0

    def test_works_with_torch_tensors(self):
        import torch as th

        inner_lo, inner_hi = (th.tensor(c) for c in box(0, 0, 0, 0.1, 0.1, 0.1))
        outer = box(0, 0, 0, 0.5, 0.5, 0.1)
        assert aabb_xy_overlap_fraction(inner_lo, inner_hi, *outer) == 1.0


class TestAabbOnTop:
    def test_resting_exactly_on_top(self):
        outer = box(0, 0, 0.05, 0.2, 0.2, 0.05)  # top at z=0.10
        inner = box(0, 0, 0.15, 0.05, 0.05, 0.05)  # bottom at z=0.10
        assert aabb_on_top(*inner, *outer)

    def test_z_tolerance_band(self):
        outer = box(0, 0, 0.05, 0.2, 0.2, 0.05)  # top at z=0.10
        # Bottom hovering 0.05 above the top: outside default 0.03, inside 0.06
        inner = box(0, 0, 0.20, 0.05, 0.05, 0.05)  # bottom at z=0.15
        assert not aabb_on_top(*inner, *outer, z_tolerance=0.03)
        assert aabb_on_top(*inner, *outer, z_tolerance=0.06)

    def test_interpenetration_within_tolerance(self):
        outer = box(0, 0, 0.05, 0.2, 0.2, 0.05)  # top at z=0.10
        inner = box(0, 0, 0.14, 0.05, 0.05, 0.05)  # bottom at z=0.09 (1cm sunk in)
        assert aabb_on_top(*inner, *outer)

    def test_xy_overlap_threshold(self):
        outer = box(0, 0, 0.05, 0.1, 0.1, 0.05)  # top at z=0.10, footprint 0.2 x 0.2
        # Inner 0.1 x 0.1, shifted so 30% of its footprint overlaps (x overlap 0.03 of 0.1)
        inner = box(0.12, 0, 0.15, 0.05, 0.05, 0.05)
        assert not aabb_on_top(*inner, *outer, xy_overlap_threshold=0.5)
        assert aabb_on_top(*inner, *outer, xy_overlap_threshold=0.25)

    def test_inner_larger_than_outer(self):
        # Overlap fraction is measured against the INNER footprint: a large
        # plate on a small stand only reaches (stand area / plate area).
        outer = box(0, 0, 0.05, 0.05, 0.05, 0.05)  # small stand, top z=0.10
        inner = box(0, 0, 0.12, 0.2, 0.2, 0.02)  # large plate, bottom z=0.10
        fraction = (0.1 * 0.1) / (0.4 * 0.4)
        assert not aabb_on_top(*inner, *outer, xy_overlap_threshold=0.5)
        assert aabb_on_top(*inner, *outer, xy_overlap_threshold=fraction - 1e-9)


class TestAabbAbove:
    def test_resting_counts_with_zero_clearance(self):
        outer = box(0, 0, 0.05, 0.2, 0.2, 0.05)  # top at z=0.10
        inner = box(0, 0, 0.15, 0.05, 0.05, 0.05)  # bottom at z=0.10
        assert aabb_above(*inner, *outer, min_clearance=0.0)

    def test_min_clearance(self):
        outer = box(0, 0, 0.05, 0.2, 0.2, 0.05)  # top at z=0.10
        inner = box(0, 0, 0.17, 0.05, 0.05, 0.05)  # bottom at z=0.12
        assert aabb_above(*inner, *outer, min_clearance=0.02)
        assert not aabb_above(*inner, *outer, min_clearance=0.03)

    def test_below_top_fails(self):
        outer = box(0, 0, 0.05, 0.2, 0.2, 0.05)  # top at z=0.10
        inner = box(0, 0, 0.10, 0.05, 0.05, 0.05)  # bottom at z=0.05
        assert not aabb_above(*inner, *outer)

    def test_optional_xy_alignment(self):
        outer = box(0, 0, 0.05, 0.1, 0.1, 0.05)  # top at z=0.10
        # High enough, but laterally offset with zero footprint overlap
        inner = box(1.0, 0, 0.30, 0.05, 0.05, 0.05)
        assert aabb_above(*inner, *outer)  # no alignment required
        assert not aabb_above(*inner, *outer, xy_overlap_threshold=0.5)


class TestIsLifted:
    def test_threshold(self):
        assert not is_lifted(0.14, 0.10, min_height=0.05)
        assert is_lifted(0.16, 0.10, min_height=0.05)

    def test_exact_threshold_counts(self):
        assert is_lifted(0.15, 0.10, min_height=0.05)

    def test_lowered_object(self):
        assert not is_lifted(0.05, 0.10, min_height=0.05)
