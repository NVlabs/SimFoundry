# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a saved scene's floor plane actually ends up.

``ground_plane_info`` is the only thing a prop can rest on under a Gaussian-splat
room, and three places have to agree about it: policy evaluation (through
``PickPlaceTask._load``) and the light editor's two physics gates, ``settle.py``
and ``parity_check.py``. The gates restore a scene through ``og.sim.restore()``,
which does **not** read the block — so they simulated against a floor at z=0
whatever the scene said, and reported drift they had invented.

The shared applier is what makes the three agree, and it is deliberately free of
OmniGibson: it takes the floor prim as an argument, so the arithmetic and the
precedence can be checked without a ninety-second Kit boot.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

th = pytest.importorskip("torch")

from simfoundry.utils.ground_plane_utils import (  # noqa: E402
    apply_ground_plane_info,
    describe,
    read_ground_plane_info,
)


class FakePlane:
    """Just enough of ``og.sim.floor_plane`` to record what it was told."""

    def __init__(self):
        self.position = None
        self.orientation = None
        self.visible = "untouched"

    def set_position_orientation(self, position=None, orientation=None):
        self.position = [float(v) for v in position]
        self.orientation = [float(v) for v in orientation]


def scene(**block):
    return {"ground_plane_info": block} if block else {}


def test_a_scene_that_states_a_height_gets_it():
    plane = FakePlane()
    applied = apply_ground_plane_info(
        scene(position=[0.0, 0.0, 0.03], orientation=[0.0, 0.0, 0.0, 1.0]), plane)
    # float32, because that is what the tensor handed to the prim is.
    assert plane.position == pytest.approx([0.0, 0.0, 0.03], abs=1e-6)
    assert applied["authored"] is True


def test_a_scene_that_states_nothing_gets_the_default_and_says_so():
    """Which is not the same as a scene that states z=0: the second is a
    decision and the first is an absence, and a report that cannot tell them
    apart cannot say whether the gate modelled anything."""
    plane = FakePlane()
    applied = apply_ground_plane_info({}, plane)
    assert plane.position == [0.0, 0.0, 0.0]
    assert applied["authored"] is False
    assert "the scene states none" in describe(applied)


def test_visibility_is_only_touched_when_the_scene_has_an_opinion():
    plane = FakePlane()
    apply_ground_plane_info(
        scene(position=[0, 0, 0], orientation=[0, 0, 0, 1]), plane)
    assert plane.visible == "untouched"
    apply_ground_plane_info(
        scene(position=[0, 0, 0], orientation=[0, 0, 0, 1], visible=False), plane)
    assert plane.visible is False


def test_a_run_config_offset_wins_over_the_scene_s_height():
    """`PickPlaceTask.ground_plane_z_offset` is an override, so it is applied
    after the scene's own value rather than instead of being read."""
    plane = FakePlane()
    apply_ground_plane_info(
        scene(position=[0.0, 0.0, 0.03], orientation=[0, 0, 0, 1]), plane,
        z_offset=0.1)
    assert plane.position == pytest.approx([0.0, 0.0, 0.1], abs=1e-6)


def test_the_whole_orientation_is_applied_not_only_the_height():
    """The plane is infinite, so only z matters *unless* the scene tilts it —
    and a scene that tilted it meant to."""
    plane = FakePlane()
    apply_ground_plane_info(
        scene(position=[0, 0, 0.02], orientation=[0.0, 0.05, 0.0, 0.9987]), plane)
    assert plane.orientation == pytest.approx([0.0, 0.05, 0.0, 0.9987], abs=1e-6)


def test_a_malformed_block_falls_back_rather_than_raising():
    """A gate that crashed on a hand-edited scene would be worse than one that
    reports what it did with it."""
    for bad in ({"position": "nope"}, {"position": [1, 2]}, {"position": None}):
        plane = FakePlane()
        applied = apply_ground_plane_info(scene(**bad), plane)
        assert plane.position == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
        # Still `authored`: the block was there, it just said nothing usable.
        assert applied["authored"] is True


def test_no_floor_plane_at_all_is_reported_rather_than_ignored():
    """`use_floor_plane` is a run-config decision this cannot override. Saying
    so is the point: under a splat room it means nothing is holding the props
    up, and silence reads as "handled"."""
    assert apply_ground_plane_info(scene(position=[0, 0, 0.03]), None) is None
    assert "nothing is holding the props up" in describe(None)


def test_a_boolean_is_not_a_coordinate():
    assert read_ground_plane_info(
        {"ground_plane_info": {"position": [0, 0, True]}})["position"] == [0.0, 0.0, 0.0]
