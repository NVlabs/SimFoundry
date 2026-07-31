# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for adding distractor / additional objects to OmniGibson scenes.

Provides functions to:
- Build a candidate pool of assets from a dataset using filters.
- Randomly sample N objects from the pool.
- Place objects near existing scene objects without AABB collisions.
"""

import math
import random
from typing import Any, Dict, List, Optional

import torch as th

from simfoundry.utils.asset_query_utils import (
    detect_assets_dir,
    query_assets,
)


def build_candidate_pool(
    dataset_name: str = "behavior-1k-assets",
    filters: Optional[List[str]] = None,
    category: Optional[str] = None,
    abilities: Optional[List[str]] = None,
    specific_assets: Optional[List[Dict[str, str]]] = None,
    assets_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build a pool of candidate assets that can be sampled as distractors.

    If *specific_assets* is provided, only those exact ``{category, model}``
    pairs are returned (no filtering).  Otherwise the dataset is scanned and
    filtered by *filters*, *category*, and *abilities*.

    Args:
        dataset_name: Dataset to scan (e.g. ``behavior-1k-assets``).
        filters: Filter expressions (e.g. ``["volume < 0.001"]``).
        category: Category substring filter.
        abilities: Required ability tags.
        specific_assets: Explicit list of ``{"category": ..., "model": ...}`` dicts.
        assets_dir: Override for the dataset root path.  If ``None``,
            auto-detected from *dataset_name*.

    Returns:
        List of candidate dicts with at least ``category``, ``model``, and
        ``dataset_name`` keys.
    """
    if assets_dir is None:
        assets_dir = detect_assets_dir(dataset_name)

    if specific_assets:
        # Return explicit list — no scanning needed
        pool = []
        for entry in specific_assets:
            pool.append({
                "category": entry["category"],
                "model": entry["model"],
                "dataset_name": dataset_name,
            })
        return pool

    records = query_assets(
        assets_dir=assets_dir,
        filters=filters,
        category=category,
        abilities=abilities,
    )

    # Attach dataset_name to each record
    for r in records:
        r["dataset_name"] = dataset_name

    return records


def sample_distractors(
    pool: List[Dict[str, Any]],
    n: int,
) -> List[Dict[str, Any]]:
    """
    Randomly sample up to *n* entries from *pool* (without replacement if
    the pool is large enough).
    """
    if not pool:
        return []
    if len(pool) <= n:
        return list(pool)
    return random.sample(pool, n)


def compute_scene_centroid(scene_objects) -> th.Tensor:
    """
    Compute the XY centroid of all non-robot, non-background scene objects.

    Args:
        scene_objects: Iterable of OmniGibson objects.

    Returns:
        Tensor of shape (3,) — centroid position (Z is the average).
    """
    from omnigibson.robots import BaseRobot

    positions = []
    for obj in scene_objects:
        if isinstance(obj, BaseRobot):
            continue
        if obj.name == "gs_background":
            continue
        pos, _ = obj.get_position_orientation()
        positions.append(pos)

    if not positions:
        return th.zeros(3)
    return th.stack(positions).mean(dim=0)


def place_distractor(
    obj,
    existing_objects,
    centroid: th.Tensor,
    placement_radius: float = 0.3,
    z_offset: float = 0.02,
    max_attempts: int = 20,
    support_z: Optional[float] = None,
    placement_bounds: Optional[tuple] = None,
) -> bool:
    """
    Place *obj* at a random collision-free position near *centroid*.

    The object is placed on the support surface (estimated from existing
    objects or supplied via *support_z*) at a random XY position.  When
    *placement_bounds* is provided, XY is sampled uniformly within those
    rectangular bounds; otherwise XY is sampled within a circle of
    *placement_radius* around *centroid*.

    AABB overlap against *existing_objects* is checked; if all attempts
    fail the object is **not** placed and ``False`` is returned.

    Args:
        obj: OmniGibson object (already added to scene / loaded).
        existing_objects: List of OmniGibson objects to avoid overlapping.
        centroid: XYZ centre around which to place (used when
            *placement_bounds* is ``None``).
        placement_radius: Max XY distance from centroid (used when
            *placement_bounds* is ``None``).
        z_offset: Small gap above the support surface.
        max_attempts: Number of random placement tries.
        support_z: If given, use this Z as the support surface.  Otherwise
            estimated from the lowest AABB bottom of *existing_objects*.
        placement_bounds: Optional world-frame XY bounds as
            ``(xy_lower, xy_upper)`` where each is a tensor of shape ``(2,)``.
            When provided, XY is sampled uniformly within this rectangle
            instead of using *centroid* + *placement_radius*.

    Returns:
        ``True`` if placement succeeded, ``False`` otherwise.
    """
    from omnigibson.robots import BaseRobot

    # Estimate support surface Z if not given
    if support_z is None:
        z_bottoms = []
        for eobj in existing_objects:
            if isinstance(eobj, BaseRobot) or eobj.name == "gs_background":
                continue
            try:
                z_bottoms.append(float(eobj.aabb[0][2].item()))
            except Exception:
                pass
        support_z = min(z_bottoms) if z_bottoms else 0.0

    # Object half-height from AABB
    obj_half_z = 0.02
    try:
        obj_lo, obj_hi = obj.aabb
        obj_half_z = float((obj_hi[2] - obj_lo[2]).item()) / 2.0
    except Exception:
        pass

    _, ori = obj.get_position_orientation()

    for _ in range(max_attempts):
        if placement_bounds is not None:
            # Sample XY uniformly within rectangular bounds
            xy_lo, xy_hi = placement_bounds
            x = random.uniform(float(xy_lo[0]), float(xy_hi[0]))
            y = random.uniform(float(xy_lo[1]), float(xy_hi[1]))
        else:
            # Random XY within circle of placement_radius
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, placement_radius)
            x = float(centroid[0].item()) + radius * math.cos(angle)
            y = float(centroid[1].item()) + radius * math.sin(angle)

        z = support_z + obj_half_z + z_offset

        candidate_pos = th.tensor([x, y, z], dtype=th.float32)
        obj.set_position_orientation(position=candidate_pos, orientation=ori)
        obj.keep_still()

        # Check AABB overlap against all existing objects
        overlap = False
        try:
            new_lo, new_hi = obj.aabb
        except Exception:
            print(f"WARNING: Could not get AABB for object {obj.name} — just accepting placement")
            return True  # Can't check — just accept placement

        for eobj in existing_objects:
            if eobj is obj:
                continue
            if isinstance(eobj, BaseRobot) or eobj.name == "gs_background":
                continue
            try:
                e_lo, e_hi = eobj.aabb
            except Exception:
                continue
            if bool(th.all(new_lo < e_hi) and th.all(e_lo < new_hi)):
                overlap = True
                break

        if not overlap:
            return True

    return False
