# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Success conditions (predicates) and AABB helpers used by SimFoundry tasks."""

from enum import IntEnum

import torch as th
from omnigibson.object_states.aabb import AABB
from omnigibson.termination_conditions.termination_condition_base import SuccessCondition
from omnigibson.utils.ui_utils import create_module_logger

from simfoundry.tasks.geometry import aabb_above, aabb_on_top, is_lifted

# Create module logger
log = create_module_logger(module_name=__name__)

# States handled by SimFoundry directly rather than OmniGibson's state registry.
# IsGrasping is milestone-only; PlaceOnTop is init-only; the rest are goal + milestone.
SPECIAL_STATE_NAMES = ("PlaceOnTop", "InsideAABB", "OnTopAABB", "AboveAABB", "Lifted", "IsGrasping")


def check_inside_aabb(inner_obj, outer_obj, outer_link_name=None, shrink_factor=0.0, shrink_z_factor=None,
                      volume_threshold=None, debug=False):
    """
    Check if inner_obj is inside outer_obj's AABB.
    This is a simpler alternative to the Inside state which requires container meta links.

    Args:
        inner_obj: The object that should be inside
        outer_obj: The container object
        outer_link_name: Optional name of a specific link of outer_obj to use for AABB check.
                        If None, uses the entire object's AABB.
        shrink_factor: Fraction (0.0 to 1.0) to shrink the outer AABB uniformly for stricter checking.
                      0.0 = no shrinking (check full AABB)
                      0.1 = shrink by 10% on each side (check inner 80% of volume)
                      0.25 = shrink by 25% on each side (check inner 50% of volume)
        shrink_z_factor: Optional separate shrink factor for Z-axis only. If provided, overrides
                        shrink_factor for the Z dimension. Useful for drawers where you want
                        stricter vertical checking but less strict horizontal checking.
        volume_threshold: If provided (0.0 to 1.0), check that at least this fraction of the
                         inner object's AABB volume is inside the outer AABB.
                         E.g., 0.75 means 75% of volume must be inside.
                         If None, uses center-point check (legacy behavior).
        debug: If True, print debug information about the AABB check

    Returns:
        bool: True if inner_obj is sufficiently inside outer_obj's (or link's) AABB
    """
    inner_aabb_lo, inner_aabb_hi = inner_obj.states[AABB].get_value()
    inner_center = (inner_aabb_lo + inner_aabb_hi) / 2.0

    # If a specific link is specified, use that link's AABB instead of the whole object
    if outer_link_name is not None:
        if hasattr(outer_obj, 'links') and outer_link_name in outer_obj.links:
            link = outer_obj.links[outer_link_name]
            outer_aabb_lo, outer_aabb_hi = link.visual_aabb
        else:
            log.warning(f"Link '{outer_link_name}' not found in object '{outer_obj.name}'. Using whole object AABB.")
            outer_aabb_lo, outer_aabb_hi = outer_obj.states[AABB].get_value()
    else:
        outer_aabb_lo, outer_aabb_hi = outer_obj.states[AABB].get_value()

    # Clone tensors to avoid modifying the original AABBs
    outer_aabb_lo = outer_aabb_lo.clone()
    outer_aabb_hi = outer_aabb_hi.clone()

    # Apply shrink factor to make check stricter (require object to be well inside)
    if shrink_factor > 0.0:
        aabb_size = outer_aabb_hi - outer_aabb_lo
        shrink_amount = aabb_size * shrink_factor
        outer_aabb_lo = outer_aabb_lo + shrink_amount
        outer_aabb_hi = outer_aabb_hi - shrink_amount

    # Apply separate Z-axis shrink factor if provided
    if shrink_z_factor is not None and shrink_z_factor > 0.0:
        aabb_size = outer_aabb_hi - outer_aabb_lo
        z_shrink_amount = aabb_size[2] * shrink_z_factor
        outer_aabb_lo[2] = outer_aabb_lo[2] + z_shrink_amount
        outer_aabb_hi[2] = outer_aabb_hi[2] - z_shrink_amount

    # If volume_threshold is specified, compute AABB intersection volume
    if volume_threshold is not None:
        # Compute intersection of the two AABBs
        intersect_lo = th.maximum(inner_aabb_lo, outer_aabb_lo)
        intersect_hi = th.minimum(inner_aabb_hi, outer_aabb_hi)

        # Check if there's any intersection
        if (intersect_lo >= intersect_hi).any():
            # No intersection
            overlap_ratio = 0.0
        else:
            # Compute volumes
            intersect_size = intersect_hi - intersect_lo
            inner_size = inner_aabb_hi - inner_aabb_lo

            intersect_volume = th.prod(intersect_size).item()
            inner_volume = th.prod(inner_size).item()

            if inner_volume > 0:
                overlap_ratio = intersect_volume / inner_volume
            else:
                overlap_ratio = 0.0

        result = overlap_ratio >= volume_threshold
    else:
        # Legacy center-point check
        result = (th.le(outer_aabb_lo, inner_center).all() and th.le(inner_center, outer_aabb_hi).all()).item()

    return result


class PredicateType(IntEnum):
    ALL = 0
    ANY = 1
    SPECIFIC = 2


class MultiPredicate(SuccessCondition):
    """
    MultiPredicate (success condition) used for PickPlaceTask
    Episode terminates if all the predicates are satisfied
    """
    def __init__(self, predicates):
        self.predicates = predicates

    def _reset(self, task, env):
        for predicate in self.predicates:
            predicate.reset(task, env)

    def _step(self, task, env, action):
        return all(predicate.step(task, env, action)[1] for predicate in self.predicates)


class Predicate(SuccessCondition):
    """
    PredicateGoal (success condition) used for PickPlaceTask
    Episode terminates if all the predicates are satisfied

    Args:
        goal_fcn (method): function for calculating goal(s). Function signature should be:

            goals = goal_fcn()

            where @goals is a list of bddl.condition_evaluation.HEAD -- compiled BDDL goal conditions
    """

    def __init__(
            self,
            objs,
            predicate_type,
            obj_state,
            obj_state_val,
            obj_state_kwargs,
    ):
        # Store values
        self.objs = objs
        self.predicate_type = predicate_type
        self.obj_state = obj_state
        self.obj_state_val = obj_state_val
        self.obj_state_kwargs = dict() if obj_state_kwargs is None else obj_state_kwargs

        # Run super
        super().__init__()

    def _resolve_op(self):
        """Map self.predicate_type to the aggregation op used over self.objs."""
        if self.predicate_type == PredicateType.ALL:
            return all
        elif self.predicate_type == PredicateType.ANY:
            return any
        elif self.predicate_type == PredicateType.SPECIFIC:
            assert len(self.objs) == 1
            return all
        else:
            raise ValueError(f"Predicate type {self.predicate_type} not supported")

    def _step(self, task, env, action):
        op = self._resolve_op()

        # Terminate if predicate condition is met
        done = op(obj.states[self.obj_state].get_value(**self.obj_state_kwargs) == self.obj_state_val for obj in self.objs)

        return done

    def set(self):
        op = self._resolve_op()

        # Set the values
        success = op(obj.states[self.obj_state].set_value(**self.obj_state_kwargs, new_value=True) for obj in self.objs)

        return success


class InsideAABBPredicate(SuccessCondition):
    """
    Predicate for checking if an object is inside another object using AABB containment.
    This doesn't require container meta links, just checks if the inner object is
    within the outer object's axis-aligned bounding box.

    Optionally, can check against a specific link of the outer object (e.g., a specific drawer).

    New: supports volume_threshold to check that a percentage of the object's volume
    is inside (e.g., 0.75 = 75% of volume must be inside).
    """

    def __init__(self, inner_objs, outer_obj, outer_link_name=None, shrink_factor=0.0, shrink_z_factor=None,
                 volume_threshold=None, debug=False, expected_value=True):
        self.inner_objs = inner_objs
        self.outer_obj = outer_obj
        self.outer_link_name = outer_link_name
        self.shrink_factor = shrink_factor
        self.shrink_z_factor = shrink_z_factor
        self.volume_threshold = volume_threshold
        self.debug = debug
        self.expected_value = expected_value
        super().__init__()

    def _step(self, task, env, action):
        # Check if all inner objects are inside the outer object (or specific link)
        results = [check_inside_aabb(
            inner_obj, self.outer_obj, self.outer_link_name,
            self.shrink_factor, self.shrink_z_factor,
            self.volume_threshold, self.debug
        ) for inner_obj in self.inner_objs]
        is_inside = all(results)
        return is_inside == self.expected_value


class OnTopAABBPredicate(SuccessCondition):
    """
    AABB-based on-top check: each top object's AABB bottom must sit within
    z_tolerance of the bottom object's AABB top, with sufficient XY footprint
    overlap. More reliable than OmniGibson's raycast OnTop for custom meshes.

    Check-only: has no set(). Use PlaceOnTop for init placement; listing this
    state under init_predicates_* logs a warning and does nothing.
    """

    def __init__(self, top_objs, bottom_obj, z_tolerance=0.03, xy_overlap_threshold=0.5,
                 expected_value=True):
        self.top_objs = top_objs
        self.bottom_obj = bottom_obj
        self.z_tolerance = z_tolerance
        self.xy_overlap_threshold = xy_overlap_threshold
        self.expected_value = expected_value
        super().__init__()

    def _step(self, task, env, action):
        bottom_lo, bottom_hi = self.bottom_obj.states[AABB].get_value()
        results = []
        for obj in self.top_objs:
            lo, hi = obj.states[AABB].get_value()
            results.append(aabb_on_top(lo, hi, bottom_lo, bottom_hi,
                                       z_tolerance=self.z_tolerance,
                                       xy_overlap_threshold=self.xy_overlap_threshold))
        return all(results) == self.expected_value


class AboveAABBPredicate(SuccessCondition):
    """
    AABB-based height check: each upper object's AABB bottom must be at least
    min_clearance above the lower object's AABB top. Set xy_overlap_threshold
    to additionally require vertical alignment (fraction of the upper object's
    XY footprint overlapping the lower's); None skips the alignment check.

    Check-only: has no set().
    """

    def __init__(self, upper_objs, lower_obj, min_clearance=0.0, xy_overlap_threshold=None,
                 expected_value=True):
        self.upper_objs = upper_objs
        self.lower_obj = lower_obj
        self.min_clearance = min_clearance
        self.xy_overlap_threshold = xy_overlap_threshold
        self.expected_value = expected_value
        super().__init__()

    def _step(self, task, env, action):
        lower_lo, lower_hi = self.lower_obj.states[AABB].get_value()
        results = []
        for obj in self.upper_objs:
            lo, hi = obj.states[AABB].get_value()
            results.append(aabb_above(lo, hi, lower_lo, lower_hi,
                                      min_clearance=self.min_clearance,
                                      xy_overlap_threshold=self.xy_overlap_threshold))
        return all(results) == self.expected_value


class LiftedPredicate(SuccessCondition):
    """
    Unary height-gain check: each object's AABB bottom must have risen at
    least min_height above its episode-start value.

    The episode-start heights are recorded by capture_start_state(), which
    PickPlaceTask calls explicitly at the end of reset() — OmniGibson's
    BaseTerminationCondition.reset() does not dispatch to _reset(), so the
    framework hook alone is not sufficient. Objects missing a recorded start
    height are captured lazily on first _step (and read as not-lifted for
    that step).

    Check-only: has no set().
    """

    def __init__(self, objs, min_height=0.05, expected_value=True):
        self.objs = objs
        self.min_height = min_height
        self.expected_value = expected_value
        self._start_z = {}
        super().__init__()

    def capture_start_state(self):
        """Record each object's current AABB-bottom Z as its episode-start height."""
        for obj in self.objs:
            lo, _ = obj.states[AABB].get_value()
            self._start_z[obj.name] = float(lo[2])

    def _reset(self, task, env):
        # Forward-compat only: not invoked by the pinned OmniGibson (see class docstring)
        self.capture_start_state()

    def _step(self, task, env, action):
        results = []
        for obj in self.objs:
            lo, _ = obj.states[AABB].get_value()
            if obj.name not in self._start_z:
                self._start_z[obj.name] = float(lo[2])
            results.append(is_lifted(lo[2], self._start_z[obj.name], self.min_height))
        return all(results) == self.expected_value


def make_aabb_predicate(state_name, objs, other_obj, expected_value, state_kwargs):
    """
    Build an OnTopAABB / AboveAABB / Lifted predicate from task-config fields.

    Args:
        state_name (str): "OnTopAABB", "AboveAABB", or "Lifted"
        objs (list): objects from the predicate's `group`
        other_obj: the single object from `other_group` (None for Lifted)
        expected_value (bool): the `value` field
        state_kwargs (None or dict): the `state_kwargs` field

    Raises:
        ValueError: if state_name is not one of the three AABB states.
    """
    kwargs = dict() if state_kwargs is None else state_kwargs
    if state_name == "OnTopAABB":
        return OnTopAABBPredicate(
            top_objs=objs,
            bottom_obj=other_obj,
            z_tolerance=kwargs.get("z_tolerance", 0.03),
            xy_overlap_threshold=kwargs.get("xy_overlap_threshold", 0.5),
            expected_value=expected_value,
        )
    if state_name == "AboveAABB":
        return AboveAABBPredicate(
            upper_objs=objs,
            lower_obj=other_obj,
            min_clearance=kwargs.get("min_clearance", 0.0),
            xy_overlap_threshold=kwargs.get("xy_overlap_threshold", None),
            expected_value=expected_value,
        )
    if state_name == "Lifted":
        return LiftedPredicate(
            objs=objs,
            min_height=kwargs.get("min_height", 0.05),
            expected_value=expected_value,
        )
    raise ValueError(f"Unknown AABB predicate state '{state_name}'")


class DirectPlacementPredicate(SuccessCondition):
    """
    Directly places an object on top of another using AABB calculations.
    Much more reliable than OmniGibson's OnTop.set_value() for custom objects
    whose collision meshes may not work well with kinematics sampling.
    """

    def __init__(self, obj, target_obj, z_offset=0.02):
        self.obj = obj
        self.target_obj = target_obj
        self.z_offset = z_offset
        super().__init__()

    def _step(self, task, env, action):
        # Not used for termination — only for init placement via set()
        return False

    def set(self):
        """Place self.obj centered on top of self.target_obj using AABB."""
        # Get target (e.g. plate) AABB
        target_lo, target_hi = self.target_obj.states[AABB].get_value()
        target_center_xy = (target_lo[:2] + target_hi[:2]) / 2.0
        target_top_z = target_hi[2].item()

        # Get object (e.g. bowl) AABB to compute half-height
        obj_lo, obj_hi = self.obj.states[AABB].get_value()
        obj_half_height = ((obj_hi[2] - obj_lo[2]) / 2.0).item()

        # Position: centered on target XY, sitting on top of target Z
        new_pos = th.tensor([
            target_center_xy[0].item(),
            target_center_xy[1].item(),
            target_top_z + obj_half_height + self.z_offset,
        ])

        # Keep current orientation
        _, current_ori = self.obj.get_position_orientation()
        self.obj.set_position_orientation(position=new_pos, orientation=current_ori)
        self.obj.keep_still()

        log.info(f"Directly placed '{self.obj.name}' on top of '{self.target_obj.name}' at z={new_pos[2]:.4f}")
        return True
