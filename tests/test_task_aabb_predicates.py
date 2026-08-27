from collections import OrderedDict

import torch as th

from omnigibson.object_states.aabb import AABB

# The predicate classes live in simfoundry/tasks/predicates.py, not in pick_place_task.py.
# This test was written against a tree that kept them inline; the split is what the editor's
# task panel and the light editor's anti-drift check both read, so the import follows the
# split rather than the split being flattened back for the import.
from simfoundry.tasks.pick_place_task import PickPlaceTask
from simfoundry.tasks.predicates import (
    AboveAABBPredicate,
    LiftedPredicate,
    OnTopAABBPredicate,
    make_aabb_predicate,
)


class FakeAABBState:
    def __init__(self, lo, hi):
        self.set(lo, hi)

    def set(self, lo, hi):
        self.lo = th.tensor(lo, dtype=th.float32)
        self.hi = th.tensor(hi, dtype=th.float32)

    def get_value(self):
        return self.lo, self.hi


class FakeObject:
    def __init__(self, name, lo, hi):
        self.name = name
        self.aabb_state = FakeAABBState(lo, hi)
        self.states = {AABB: self.aabb_state}


def test_on_top_and_above_predicates_use_aabb_thresholds():
    support = FakeObject("support", [-0.2, -0.2, 0.0], [0.2, 0.2, 0.1])
    item = FakeObject("item", [-0.05, -0.05, 0.1], [0.05, 0.05, 0.2])

    assert OnTopAABBPredicate([item], support)._step(None, None, None)
    assert AboveAABBPredicate([item], support, min_clearance=0.0)._step(None, None, None)
    assert not AboveAABBPredicate([item], support, min_clearance=0.02)._step(None, None, None)


def test_lifted_predicate_compares_against_captured_episode_start():
    item = FakeObject("item", [0.0, 0.0, 0.1], [0.1, 0.1, 0.2])
    predicate = LiftedPredicate([item], min_height=0.05)
    predicate.capture_start_state()

    item.aabb_state.set([0.0, 0.0, 0.14], [0.1, 0.1, 0.24])
    assert not predicate._step(None, None, None)
    item.aabb_state.set([0.0, 0.0, 0.16], [0.1, 0.1, 0.26])
    assert predicate._step(None, None, None)


def test_factory_preserves_existing_config_schema_and_defaults():
    support = FakeObject("support", [-0.2, -0.2, 0.0], [0.2, 0.2, 0.1])
    item = FakeObject("item", [-0.05, -0.05, 0.1], [0.05, 0.05, 0.2])

    on_top = make_aabb_predicate("OnTopAABB", [item], support, True, None)
    above = make_aabb_predicate(
        "AboveAABB", [item], support, True, {"min_clearance": 0.02, "xy_overlap_threshold": 0.75}
    )
    lifted = make_aabb_predicate("Lifted", [item], None, False, {"min_height": 0.08})

    assert on_top.z_tolerance == 0.03
    assert on_top.xy_overlap_threshold == 0.5
    assert above.min_clearance == 0.02
    assert above.xy_overlap_threshold == 0.75
    assert lifted.min_height == 0.08
    assert lifted.expected_value is False


def test_goal_config_builds_custom_predicates_without_registered_og_states():
    support = FakeObject("support", [-0.2, -0.2, 0.0], [0.2, 0.2, 0.1])
    item = FakeObject("item", [-0.05, -0.05, 0.1], [0.05, 0.05, 0.2])
    task = PickPlaceTask.__new__(PickPlaceTask)
    task.group_objs = {"items": [item], "support": [support]}

    predicates = task._create_predicates(
        [{"group": "items", "other_group": "support", "state": "OnTopAABB", "value": True}],
        None,
        None,
    )

    assert isinstance(predicates["predicate_all_0"], OnTopAABBPredicate)


def test_milestone_wiring_preserves_sequential_dependencies():
    item = FakeObject("item", [0.0, 0.0, 0.1], [0.1, 0.1, 0.2])
    task = PickPlaceTask.__new__(PickPlaceTask)
    task.group_objs = {"items": [item]}
    task.milestone_predicates_config = [
        {
            "name": "lift_item",
            "group": "items",
            "state": "Lifted",
            "value": True,
            "state_kwargs": {"min_height": 0.07},
            "requires": "grasp_item",
        }
    ]

    task._create_milestone_predicates()

    assert isinstance(task.milestone_predicates["lift_item"], LiftedPredicate)
    assert task.milestone_requires["lift_item"] == ["grasp_item"]
    assert task.milestones_achieved == OrderedDict([("lift_item", 0)])


def test_task_capture_hook_records_goal_and_milestone_baselines():
    goal_item = FakeObject("goal_item", [0.0, 0.0, 0.1], [0.1, 0.1, 0.2])
    milestone_item = FakeObject("milestone_item", [0.0, 0.0, 0.3], [0.1, 0.1, 0.4])
    goal = LiftedPredicate([goal_item])
    milestone = LiftedPredicate([milestone_item])
    task = PickPlaceTask.__new__(PickPlaceTask)
    task._termination_conditions = {"predicates": type("Goals", (), {"predicates": [goal]})()}
    task.milestone_predicates = {"lift": milestone}

    task._capture_predicate_start_states()

    assert goal._start_z == {"goal_item": float(goal_item.aabb_state.lo[2])}
    assert milestone._start_z == {"milestone_item": float(milestone_item.aabb_state.lo[2])}
