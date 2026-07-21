"""
Utility functions for predicate-based spatial placement of objects.

Supports placing an object relative to a reference object using spatial
predicates: on_top, left_of, right_of, behind, in_front_of, inside.

Axis conventions:
    - behind      = +X side of reference
    - in_front_of = -X side of reference
    - right_of    = -Y side of reference (robot's right)
    - left_of     = +Y side of reference (robot's left)
"""

import random

import torch as th

# All supported spatial predicates
PREDICATES = ["on_top", "left_of", "right_of", "behind", "in_front_of", "inside"]


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


def place_with_predicate(obj, reference_obj, predicate, gap=0.05, z_offset=0.0,
                         bounds=None, aligned=False):
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

    Raises:
        ValueError: If *predicate* is not in ``PREDICATES``.
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


def _aabb_overlap(obj_a, obj_b):
    """Return True if the AABBs of *obj_a* and *obj_b* intersect."""
    a_lo, a_hi = obj_a.aabb
    b_lo, b_hi = obj_b.aabb
    return bool(th.all(a_lo < b_hi) and th.all(b_lo < a_hi))


def separate_overlapping_objects(placed_objects, multiplier=1.5):
    """
    Resolve AABB overlaps among predicate-placed objects by shifting them
    apart along the perpendicular horizontal axis.

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

            # Only separate horizontal predicates
            perp_a = _PERPENDICULAR_AXIS.get(pred_a)
            perp_b = _PERPENDICULAR_AXIS.get(pred_b)
            if perp_a is None or perp_b is None:
                continue

            if not _aabb_overlap(obj_a, obj_b):
                continue

            # Choose perpendicular axis of the *later* object's predicate
            axis = perp_b

            # Compute shift distance from the larger AABB extent on that axis
            a_lo, a_hi = obj_a.aabb
            b_lo, b_hi = obj_b.aabb
            extent_a = (a_hi[axis] - a_lo[axis]).item()
            extent_b = (b_hi[axis] - b_lo[axis]).item()
            shift = max(extent_a, extent_b) * multiplier

            # Shift obj_b in the positive direction of the perpendicular axis
            pos_b, ori_b = obj_b.get_position_orientation()
            new_pos = pos_b.clone()
            new_pos[axis] += shift
            obj_b.set_position_orientation(position=new_pos, orientation=ori_b)
            obj_b.keep_still()
