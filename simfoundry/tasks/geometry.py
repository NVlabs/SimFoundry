# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pure AABB geometry helpers for SimFoundry task predicates.

These functions operate on (lo, hi) corner tensors (3-element, world frame) so
they can be unit-tested without OmniGibson. The predicate classes in
``simfoundry.tasks.predicates`` wrap them with object-state access.

Threshold comparisons are epsilon-tolerant so exact boundary values (e.g. an
object resting precisely at a surface) are not rejected by float error.
"""

_EPS = 1e-9


def aabb_xy_overlap_fraction(inner_lo, inner_hi, outer_lo, outer_hi):
    """
    Fraction of the inner AABB's XY footprint that overlaps the outer AABB's.

    Args:
        inner_lo, inner_hi: (3,) corners of the inner AABB
        outer_lo, outer_hi: (3,) corners of the outer AABB

    Returns:
        float: intersection area / inner XY area, in [0, 1]. Returns 0.0 for
            disjoint footprints or a degenerate (zero-area) inner footprint.
    """
    ix = min(float(inner_hi[0]), float(outer_hi[0])) - max(float(inner_lo[0]), float(outer_lo[0]))
    iy = min(float(inner_hi[1]), float(outer_hi[1])) - max(float(inner_lo[1]), float(outer_lo[1]))
    if ix <= 0.0 or iy <= 0.0:
        return 0.0

    inner_area = (float(inner_hi[0]) - float(inner_lo[0])) * (float(inner_hi[1]) - float(inner_lo[1]))
    if inner_area <= 0.0:
        return 0.0

    return (ix * iy) / inner_area


def aabb_on_top(inner_lo, inner_hi, outer_lo, outer_hi, z_tolerance=0.03, xy_overlap_threshold=0.5):
    """
    Check whether the inner AABB is resting on top of the outer AABB.

    True iff the inner AABB's bottom is within ``z_tolerance`` of the outer
    AABB's top (above or below, to tolerate settling interpenetration) AND at
    least ``xy_overlap_threshold`` of the inner XY footprint overlaps the outer.

    Args:
        inner_lo, inner_hi: (3,) corners of the object expected on top
        outer_lo, outer_hi: (3,) corners of the supporting object
        z_tolerance (float): allowed |inner bottom - outer top| in meters
        xy_overlap_threshold (float): required overlap fraction of the inner footprint

    Returns:
        bool
    """
    inner_bottom = float(inner_lo[2])
    outer_top = float(outer_hi[2])
    if not (outer_top - z_tolerance - _EPS <= inner_bottom <= outer_top + z_tolerance + _EPS):
        return False
    return aabb_xy_overlap_fraction(inner_lo, inner_hi, outer_lo, outer_hi) >= xy_overlap_threshold


def aabb_above(inner_lo, inner_hi, outer_lo, outer_hi, min_clearance=0.0, xy_overlap_threshold=None):
    """
    Check whether the inner AABB is entirely above the outer AABB's top surface.

    Args:
        inner_lo, inner_hi: (3,) corners of the object expected above
        outer_lo, outer_hi: (3,) corners of the reference object
        min_clearance (float): required gap (meters) between the outer top and
            the inner bottom; 0.0 accepts a resting object
        xy_overlap_threshold (None or float): if set, additionally require this
            fraction of the inner XY footprint to overlap the outer's (i.e. the
            object must also be vertically aligned with the reference)

    Returns:
        bool
    """
    if float(inner_lo[2]) < float(outer_hi[2]) + min_clearance - _EPS:
        return False
    if xy_overlap_threshold is None:
        return True
    return aabb_xy_overlap_fraction(inner_lo, inner_hi, outer_lo, outer_hi) >= xy_overlap_threshold


def is_lifted(current_z, start_z, min_height=0.05):
    """
    Check whether a height has risen at least ``min_height`` above its start value.

    Args:
        current_z (float or 0-d tensor): current height (e.g. AABB bottom Z)
        start_z (float or 0-d tensor): height at episode start
        min_height (float): required gain in meters

    Returns:
        bool
    """
    return float(current_z) - float(start_z) >= min_height - _EPS
