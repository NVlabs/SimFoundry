# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the stage-7 task proposer's predicate vocabulary and parsing."""

import importlib.util
from pathlib import Path
import sys

import pytest


def _load_stage_module():
    # The stage imports simfoundry.models.vlm at module scope; skip in envs without it.
    pytest.importorskip("simfoundry.models.vlm")
    script = Path(__file__).resolve().parents[1] / "scripts/pipeline/B_augmentation/stages/7_propose_scene_tasks.py"
    spec = importlib.util.spec_from_file_location("propose_scene_tasks", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["propose_scene_tasks"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_og_object_states_lists_new_predicates():
    mod = _load_stage_module()
    states = [s.strip() for s in mod.OG_OBJECT_STATES.split(",")]
    assert states == ["OnTop", "OnTopAABB", "InsideAABB", "AboveAABB", "Lifted"]


def test_prompt_template_formats_cleanly():
    # Guards against un-escaped curly braces in newly added prompt text
    mod = _load_stage_module()
    prompt = mod.PROMPT_TEMPLATE.format(
        object_list="1 -> cup, blue_cup_1",
        states=mod.OG_OBJECT_STATES,
        num_tasks=2,
        object_constraints="",
        robot_type="franka",
        robot_constraint="constraint",
    )
    for state in ("OnTopAABB", "InsideAABB", "AboveAABB", "Lifted"):
        assert state in prompt
    assert "volume_threshold" in prompt
    assert "min_height" in prompt


def test_parse_vlm_tasks_new_states():
    mod = _load_stage_module()
    response = """--- Task 1
task_name: put_pear_in_bowl
semantic_group_mapping:
  pear: [pear_abc_3]
  bowl: [bowl_def_7]
goal_predicates_all:
  - state: InsideAABB
    state_kwargs:
      volume_threshold: 0.5
    value: true
    group: pear
    other_group: bowl
goal_predicates_any: null
--- Task 2
task_name: lift_cup_onto_plate
semantic_group_mapping:
  cup: [blue_cup_1]
  plate: [teal_plate_2]
goal_predicates_all:
  - state: Lifted
    state_kwargs:
      min_height: 0.08
    value: true
    group: cup
    other_group: null
  - state: OnTopAABB
    state_kwargs:
      z_tolerance: 0.03
      xy_overlap_threshold: 0.5
    value: true
    group: cup
    other_group: plate
goal_predicates_any: null
"""
    tasks = mod.parse_vlm_tasks(response)
    assert len(tasks) == 2

    inside = tasks[0]["goal_predicates_all"][0]
    assert inside["state"] == "InsideAABB"
    assert inside["state_kwargs"]["volume_threshold"] == 0.5

    lifted, on_top = tasks[1]["goal_predicates_all"]
    assert lifted["state"] == "Lifted"
    assert lifted["other_group"] is None
    assert lifted["state_kwargs"]["min_height"] == 0.08
    assert on_top["state"] == "OnTopAABB"
    assert on_top["state_kwargs"]["xy_overlap_threshold"] == 0.5
