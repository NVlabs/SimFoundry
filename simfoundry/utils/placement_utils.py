# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for predicate-based spatial placement of objects.

Supports placing an object relative to a reference object using spatial
predicates: on_top, left_of, right_of, behind, in_front_of, inside, near,
between, inside_link.

Axis conventions:
    - behind      = +X side of reference
    - in_front_of = -X side of reference
    - right_of    = -Y side of reference (robot's right)
    - left_of     = +Y side of reference (robot's left)

Predicate semantics:
    - The four horizontal predicates displace the primary axis by ``gap``
      (edge-to-edge) and sample the secondary axis; they preserve the object's
      current Z and ignore ``z_offset``.
    - ``near`` places at edge-to-edge clearance ``gap`` in a uniformly random
      XY direction around the reference; Z = current Z + ``z_offset``.
    - ``between`` places on the XY segment between TWO references (pass
      ``reference_obj_2``), uniformly over the span that keeps ``gap``
      clearance from both; Z = current Z + ``z_offset``.
    - ``inside_link`` places within a named link's AABB of the reference (pass
      ``link_name``), e.g. a drawer or shelf level; ``gap`` is ignored and
      Z = link bottom + object half-height + ``z_offset``.
    - ``near`` and ``between`` clamp their final XY into ``bounds``; the legacy
      predicates only use ``bounds`` to sample the secondary axis (their
      primary axis is NOT clamped); ``inside_link`` ignores ``bounds`` (the
      link AABB is the authoritative container).
"""

import logging
import math
import random

import torch as th

log = logging.getLogger(__name__)

# All supported spatial predicates
PREDICATES = ["on_top", "left_of", "right_of", "behind", "in_front_of", "inside",
              "near", "between", "inside_link"]


def resolve_gap(gap_cfg, predicate):
    """
    Resolve a gap configuration into a concrete float value.

    Supports three formats:
        - ``float``:  fixed gap for all predicates.
        - ``[min, max]``:  uniformly sampled range for all predicates.
        - ``dict``:  per-predicate values.  Each value can be a float or
          ``[min, max]``.  A ``"default"`` key is used as fallback for
          predicates not explicitly listed.

    Args:
        gap_cfg: One of the three formats described above.
        predicate (str): The predicate being placed (used for dict lookup).

    Returns:
        float: The resolved gap in meters.
    """
    if isinstance(gap_cfg, dict):
        entry = gap_cfg.get(predicate, gap_cfg.get("default", 0.05))
    else:
        entry = gap_cfg

    if isinstance(entry, (list, tuple)):
        return random.uniform(entry[0], entry[1])
    return float(entry)


def _support_radius_xy(half_extents, cos_t, sin_t):
    """
    Distance from an AABB's center to its boundary along XY direction (cos_t, sin_t).

    Placing two AABBs with center distance ``r_a + r_b + gap`` along that
    direction guarantees they are separated on at least one axis (the
    effective edge-to-edge gap is exactly ``gap`` on axis-aligned directions
    and at worst ``gap / sqrt(2)`` at 45 degrees).
    """
    return float(half_extents[0]) * abs(cos_t) + float(half_extents[1]) * abs(sin_t)


def _clamp_xy_to_bounds(pos, obj_half, bounds):
    """
    Clamp ``pos`` in XY so the object's AABB stays within ``bounds``.

    The bounds are shrunk by the object's XY half-extents; if the shrunk
    interval is inverted (object wider than the bounds), the bounds midpoint
    is used on that axis. Returns ``pos`` unchanged when ``bounds`` is None.
    """
    if bounds is None:
        return pos
    clamped = pos.clone()
    for axis in (0, 1):
        lo = bounds[0][axis].item() + float(obj_half[axis])
        hi = bounds[1][axis].item() - float(obj_half[axis])
        if lo > hi:
            clamped[axis] = 0.5 * (bounds[0][axis].item() + bounds[1][axis].item())
        else:
            clamped[axis] = min(max(float(clamped[axis]), lo), hi)
    return clamped


def place_with_predicate(obj, reference_obj, predicate, gap=0.05, z_offset=0.0,
                         bounds=None, aligned=False, reference_obj_2=None,
                         link_name=None):
    """
    Place *obj* relative to *reference_obj* using a spatial predicate.

    For horizontal predicates (``left_of``, ``right_of``, ``behind``,
    ``in_front_of``) the placement is **semantic** by default: only the
    *primary* axis (the one implied by the predicate) is displaced by ``gap``;
    the *secondary* axis is uniformly sampled within ``bounds`` (or within the
    reference object's AABB extent if no bounds are given).  Set
    ``aligned=True`` to revert to the old behaviour where the secondary axis
    is centred on the reference object.

    ``on_top`` and ``inside`` are always centred on the reference object
    regardless of ``aligned``.

    Args:
        obj: OmniGibson object to place.
        reference_obj: OmniGibson object used as the spatial reference.
        predicate (str): One of ``PREDICATES``.
        gap (float): Clearance between the two objects' AABB surfaces (meters).
            Ignored for ``inside``.
        z_offset (float): Additional vertical offset (meters).  Used as the
            primary offset for ``on_top`` and ``inside``; ignored for
            horizontal predicates.
        bounds (tuple or None): World-frame placement bounds as
            ``(lower_tensor, upper_tensor)`` each of shape ``(3,)``.  Used to
            constrain the secondary axis for horizontal predicates when
            ``aligned=False``.  If ``None``, the reference object's AABB
            extent on the secondary axis is used as the sampling range.
        aligned (bool): If ``True``, centre the secondary axis on the
            reference object (legacy behaviour).
        reference_obj_2: Second reference object; required by ``between``
            (ignored by every other predicate).
        link_name (str or None): Name of a link of *reference_obj* whose AABB
            to place within; required by ``inside_link`` (ignored by every
            other predicate). Falls back to the whole-object AABB with a
            warning if the link is missing.

    Raises:
        ValueError: If *predicate* is not in ``PREDICATES``, if ``between`` is
            used without *reference_obj_2*, or if ``inside_link`` is used
            without *link_name*.
    """
    if predicate not in PREDICATES:
        raise ValueError(
            f"Unknown predicate '{predicate}'. Must be one of {PREDICATES}"
        )

    # Reference object AABB
    ref_lo, ref_hi = reference_obj.aabb
    ref_center = (ref_lo + ref_hi) / 2.0

    # Object AABB half-extents (used to offset from surfaces)
    obj_lo, obj_hi = obj.aabb
    obj_half = (obj_hi - obj_lo) / 2.0

    # Preserve current orientation and Z (support surface height)
    current_pos, current_ori = obj.get_position_orientation()
    current_z = current_pos[2]

    # ---- helper: sample secondary axis value ----
    def _sample_secondary(axis_idx):
        """Return a random value for the secondary (free) axis.

        If *bounds* is provided, sample uniformly within those bounds on the
        given axis.  Otherwise fall back to the reference object's AABB extent
        on that axis.
        """
        if bounds is not None:
            lo = bounds[0][axis_idx].item()
            hi = bounds[1][axis_idx].item()
        else:
            lo = ref_lo[axis_idx].item()
            hi = ref_hi[axis_idx].item()
        return random.uniform(lo, hi)

    if predicate == "on_top":
        pos = th.tensor([
            ref_center[0],
            ref_center[1],
            ref_hi[2] + obj_half[2] + z_offset,
        ], dtype=th.float32)

    elif predicate == "left_of":  # +Y (robot's left)
        secondary_x = ref_center[0] if aligned else _sample_secondary(0)
        pos = th.tensor([
            secondary_x,
            ref_hi[1] + obj_half[1] + gap,
            current_z,
        ], dtype=th.float32)

    elif predicate == "right_of":  # -Y (robot's right)
        secondary_x = ref_center[0] if aligned else _sample_secondary(0)
        pos = th.tensor([
            secondary_x,
            ref_lo[1] - obj_half[1] - gap,
            current_z,
        ], dtype=th.float32)

    elif predicate == "behind":  # +X
        secondary_y = ref_center[1] if aligned else _sample_secondary(1)
        pos = th.tensor([
            ref_hi[0] + obj_half[0] + gap,
            secondary_y,
            current_z,
        ], dtype=th.float32)

    elif predicate == "in_front_of":  # -X
        secondary_y = ref_center[1] if aligned else _sample_secondary(1)
        pos = th.tensor([
            ref_lo[0] - obj_half[0] - gap,
            secondary_y,
            current_z,
        ], dtype=th.float32)

    elif predicate == "inside":
        pos = th.tensor([
            ref_center[0],
            ref_center[1],
            ref_center[2] + z_offset,
        ], dtype=th.float32)

    elif predicate == "near":
        ref_half = (ref_hi - ref_lo) / 2.0
        # Rejection-sample a direction whose placement stays inside bounds;
        # after max attempts, keep the last candidate clamped (the effective
        # gap may then shrink).
        for attempt in range(8):
            theta = random.uniform(0.0, 2.0 * math.pi)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            dist = _support_radius_xy(ref_half, cos_t, sin_t) \
                + _support_radius_xy(obj_half, cos_t, sin_t) + gap
            candidate = th.tensor([
                float(ref_center[0]) + dist * cos_t,
                float(ref_center[1]) + dist * sin_t,
                current_z + z_offset,
            ], dtype=th.float32)
            pos = _clamp_xy_to_bounds(candidate, obj_half, bounds)
            if th.equal(pos, candidate):
                break
        else:
            log.debug("'near' placement clamped to bounds after 8 attempts; effective gap may be reduced")

    elif predicate == "between":
        if reference_obj_2 is None:
            raise ValueError("Predicate 'between' requires a second reference object (reference_obj_2).")
        ref2_lo, ref2_hi = reference_obj_2.aabb
        ref2_center = (ref2_lo + ref2_hi) / 2.0
        ref_half = (ref_hi - ref_lo) / 2.0
        ref2_half = (ref2_hi - ref2_lo) / 2.0

        segment = ref2_center[:2] - ref_center[:2]
        length = float(th.norm(segment))
        if length < 1e-4:
            log.warning("'between' references have coincident centers; placing at their midpoint.")
            xy = (ref_center[:2] + ref2_center[:2]) / 2.0
        else:
            direction = segment / length
            cos_t, sin_t = float(direction[0]), float(direction[1])
            # Span along the segment that keeps `gap` edge clearance from both refs
            t_min = _support_radius_xy(ref_half, cos_t, sin_t) \
                + _support_radius_xy(obj_half, cos_t, sin_t) + gap
            t_max = length - (_support_radius_xy(ref2_half, cos_t, sin_t)
                              + _support_radius_xy(obj_half, cos_t, sin_t) + gap)
            if t_min >= t_max:
                log.warning("'between' references leave no free span; placing at the segment midpoint.")
                dist = length / 2.0
            else:
                dist = random.uniform(t_min, t_max)
            xy = ref_center[:2] + direction * dist
        pos = th.tensor([
            float(xy[0]),
            float(xy[1]),
            current_z + z_offset,
        ], dtype=th.float32)
        pos = _clamp_xy_to_bounds(pos, obj_half, bounds)

    elif predicate == "inside_link":
        if link_name is None:
            raise ValueError("Predicate 'inside_link' requires link_name.")
        if hasattr(reference_obj, "links") and link_name in reference_obj.links:
            link_lo, link_hi = reference_obj.links[link_name].visual_aabb
        else:
            log.warning(
                f"Link '{link_name}' not found in object "
                f"'{getattr(reference_obj, 'name', reference_obj)}'. Using whole object AABB."
            )
            link_lo, link_hi = ref_lo, ref_hi
        coords = []
        for axis in (0, 1):
            lo = float(link_lo[axis]) + float(obj_half[axis])
            hi = float(link_hi[axis]) - float(obj_half[axis])
            if lo > hi:
                log.warning(
                    f"Object wider than link '{link_name}' on axis {axis}; centering on that axis."
                )
                coords.append(0.5 * (float(link_lo[axis]) + float(link_hi[axis])))
            else:
                coords.append(random.uniform(lo, hi))
        pos = th.tensor([
            coords[0],
            coords[1],
            float(link_lo[2]) + float(obj_half[2]) + z_offset,
        ], dtype=th.float32)

    obj.set_position_orientation(position=pos, orientation=current_ori)
    obj.keep_still()

    return pos, current_ori, predicate


# Maps each predicate to the horizontal axis index (0=X, 1=Y) perpendicular
# to its primary placement direction.  Used by ``separate_overlapping_objects``
# to decide which axis to shift along when two objects overlap.
_PERPENDICULAR_AXIS = {
    "left_of": 0,       # primary Y -> shift along X
    "right_of": 0,      # primary Y -> shift along X
    "behind": 1,        # primary X -> shift along Y
    "in_front_of": 1,   # primary X -> shift along Y
}

# Predicates that participate in overlap separation. `near` has no fixed axis
# and is pushed radially instead; `on_top`/`inside`/`between`/`inside_link`
# are skipped because moving them would destroy their semantics.
_SEPARABLE_PREDICATES = set(_PERPENDICULAR_AXIS) | {"near"}


def _aabb_overlap(obj_a, obj_b):
    """Return True if the AABBs of *obj_a* and *obj_b* intersect."""
    a_lo, a_hi = obj_a.aabb
    b_lo, b_hi = obj_b.aabb
    return bool(th.all(a_lo < b_hi) and th.all(b_lo < a_hi))


def separate_overlapping_objects(placed_objects, multiplier=1.5):
    """
    Resolve AABB overlaps among predicate-placed objects by shifting them
    apart along the perpendicular horizontal axis (or radially away from the
    other object, for ``near``-placed objects).

    Args:
        placed_objects: list of ``(obj, predicate)`` tuples -- every object
            that was positioned via ``place_with_predicate`` in the current
            reset.  Order matters: earlier items are treated as anchored and
            later items are the ones that get shifted.
        multiplier (float): The shift distance is
            ``max(extent_a, extent_b) * multiplier`` where the extents are
            measured along the perpendicular axis.  A value of 1.5 gives
            comfortable separation.
    """
    for i in range(len(placed_objects)):
        obj_a, pred_a = placed_objects[i]
        for j in range(i + 1, len(placed_objects)):
            obj_b, pred_b = placed_objects[j]

            # Only separate predicates with movable placements
            if pred_a not in _SEPARABLE_PREDICATES or pred_b not in _SEPARABLE_PREDICATES:
                continue

            if not _aabb_overlap(obj_a, obj_b):
                continue

            a_lo, a_hi = obj_a.aabb
            b_lo, b_hi = obj_b.aabb
            pos_b, ori_b = obj_b.get_position_orientation()

            if pred_b == "near":
                # `near` has no fixed axis: push obj_b radially away from obj_a
                pos_a, _ = obj_a.get_position_orientation()
                direction = (pos_b[:2] - pos_a[:2]).to(th.float32)
                norm = float(th.norm(direction))
                if norm < 1e-6:
                    angle = random.uniform(0.0, 2.0 * math.pi)
                    direction = th.tensor([math.cos(angle), math.sin(angle)], dtype=th.float32)
                else:
                    direction = direction / norm
                extent_a = max((a_hi[0] - a_lo[0]).item(), (a_hi[1] - a_lo[1]).item())
                extent_b = max((b_hi[0] - b_lo[0]).item(), (b_hi[1] - b_lo[1]).item())
                shift = max(extent_a, extent_b) * multiplier
                new_pos = pos_b.clone()
                new_pos[0] += direction[0] * shift
                new_pos[1] += direction[1] * shift
            else:
                # Choose perpendicular axis of the *later* object's predicate
                axis = _PERPENDICULAR_AXIS[pred_b]

                # Compute shift distance from the larger AABB extent on that axis
                extent_a = (a_hi[axis] - a_lo[axis]).item()
                extent_b = (b_hi[axis] - b_lo[axis]).item()
                shift = max(extent_a, extent_b) * multiplier

                # Shift obj_b in the positive direction of the perpendicular axis
                new_pos = pos_b.clone()
                new_pos[axis] += shift

            obj_b.set_position_orientation(position=new_pos, orientation=ori_b)
            obj_b.keep_still()
