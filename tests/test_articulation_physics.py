# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 11 sources articulated-object physics from the articulation pipeline.

The articulation pipeline writes results/physics_properties.json (estimates)
and the refinement UI writes results/physics_overrides.json (user edits);
simfoundry.pipeline.articulation_physics resolves them with user edits on top
and falls back to stage 11's legacy VLM estimation only when the pipeline file
is absent. These tests lock that precedence and the <dynamics> preservation
contract used by import_articulated_object.
"""

import json

import pytest

from simfoundry.pipeline.articulation_physics import (
    load_physics_overrides,
    load_pipeline_physics,
    merge_parts_properties,
    resolve_articulation_physics,
    resolve_joint_dynamics,
)

PIPELINE_PHYSICS = {
    "version": 1,
    "source": "vlm",
    "parts": [
        {"name": "body_link", "mass_kg": 4.5, "friction": 0.6},
        {"name": "door_link", "mass_kg": 1.2, "friction": 0.4, "joint_damping": 0.15},
    ],
    "joints": {"body_link_to_door_link": {"damping": 0.15, "friction": 0.04}},
}


def write(results_dir, name, payload):
    with open(results_dir / name, "w") as f:
        json.dump(payload, f)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_load_pipeline_physics(tmp_path):
    assert load_pipeline_physics(str(tmp_path)) is None
    write(tmp_path, "physics_properties.json", PIPELINE_PHYSICS)
    physics = load_pipeline_physics(str(tmp_path))
    assert [p["name"] for p in physics["parts"]] == ["body_link", "door_link"]
    assert physics["joints"] == {"body_link_to_door_link": {"damping": 0.15, "friction": 0.04}}


def test_load_pipeline_physics_rejects_unusable(tmp_path):
    (tmp_path / "physics_properties.json").write_text("{broken")
    assert load_pipeline_physics(str(tmp_path)) is None
    write(tmp_path, "physics_properties.json", {"parts": [], "joints": {}})
    assert load_pipeline_physics(str(tmp_path)) is None
    write(tmp_path, "physics_properties.json",
          {"parts": [{"name": "a", "mass_kg": "heavy"}], "joints": {}})
    assert load_pipeline_physics(str(tmp_path)) is None


def test_load_physics_overrides_tolerant(tmp_path):
    assert load_physics_overrides(str(tmp_path)) == {"parts": {}, "joints": {}}
    write(tmp_path, "physics_overrides.json",
          {"parts": {"door_link": {"mass_kg": 2.0, "bogus": 1}},
           "joints": {"j": {"damping": "soft"}, "k": {"friction": 0.2}}})
    overrides = load_physics_overrides(str(tmp_path))
    assert overrides["parts"] == {"door_link": {"mass_kg": 2.0}}
    assert overrides["joints"] == {"k": {"friction": 0.2}}


# ---------------------------------------------------------------------------
# Resolution precedence
# ---------------------------------------------------------------------------

def test_resolve_prefers_pipeline_and_skips_fallback(tmp_path):
    write(tmp_path, "physics_properties.json", PIPELINE_PHYSICS)

    def fallback():
        raise AssertionError("legacy VLM fallback must not run when pipeline physics exist")

    parts, joint_overrides, joint_defaults, source = resolve_articulation_physics(
        str(tmp_path), fallback_parts_fn=fallback)
    assert source == "articulation_pipeline"
    assert parts[1]["mass_kg"] == 1.2
    # Pipeline estimates are the DEFAULTS tier; nothing overrides the URDF.
    assert joint_defaults == {"body_link_to_door_link": {"damping": 0.15, "friction": 0.04}}
    assert joint_overrides == {}


def test_resolve_layers_user_overrides_on_pipeline(tmp_path):
    write(tmp_path, "physics_properties.json", PIPELINE_PHYSICS)
    write(tmp_path, "physics_overrides.json",
          {"parts": {"door_link": {"mass_kg": 2.5}},
           "joints": {"body_link_to_door_link": {"damping": 0.8}}})
    parts, joint_overrides, joint_defaults, source = resolve_articulation_physics(str(tmp_path))
    assert source == "articulation_pipeline"
    by_name = {p["name"]: p for p in parts}
    assert by_name["door_link"]["mass_kg"] == 2.5   # user wins
    assert by_name["door_link"]["friction"] == 0.4  # pipeline value kept
    # Only the user's edit is an override; the pipeline estimate stays a default.
    assert joint_overrides == {"body_link_to_door_link": {"damping": 0.8}}
    assert joint_defaults["body_link_to_door_link"] == {"damping": 0.15, "friction": 0.04}


def test_resolve_lifts_parts_table_joint_damping(tmp_path):
    """The UI parts table writes joint_damping per child link; it must reach
    the child's joint as an override (the silent-loss regression)."""
    write(tmp_path, "physics_properties.json", PIPELINE_PHYSICS)
    write(tmp_path, "physics_overrides.json",
          {"parts": {"door_link": {"joint_damping": 0.9}}})
    urdf = tmp_path / "mobility.urdf"
    urdf.write_text("""<?xml version='1.0'?>
<robot name="cabinet">
 <link name="base" /><link name="body_link" /><link name="door_link" />
 <joint type="fixed" name="base_to_body_link">
  <parent link="base" /><child link="body_link" /></joint>
 <joint type="revolute" name="body_link_to_door_link">
  <parent link="body_link" /><child link="door_link" />
  <limit lower="0" upper="1.5" effort="5" velocity="5" /></joint>
</robot>""")
    parts, joint_overrides, joint_defaults, _ = resolve_articulation_physics(
        str(tmp_path), urdf_path=str(urdf))
    assert joint_overrides == {"body_link_to_door_link": {"damping": 0.9}}
    # An explicit joints-section damping still beats the lifted parts value.
    write(tmp_path, "physics_overrides.json",
          {"parts": {"door_link": {"joint_damping": 0.9}},
           "joints": {"body_link_to_door_link": {"damping": 1.5}}})
    _, joint_overrides, _, _ = resolve_articulation_physics(str(tmp_path), urdf_path=str(urdf))
    assert joint_overrides["body_link_to_door_link"]["damping"] == 1.5


def test_resolve_falls_back_to_legacy(tmp_path):
    fallback_parts = [{"name": "door_link", "mass_kg": 1.0, "friction": 0.5, "joint_damping": 0.5}]
    parts, joint_overrides, joint_defaults, source = resolve_articulation_physics(
        str(tmp_path), fallback_parts_fn=lambda: fallback_parts)
    assert source == "legacy_fallback"
    assert parts == fallback_parts
    assert parts is not fallback_parts  # copied, caller list untouched
    assert joint_overrides == {} and joint_defaults == {}


def test_resolve_fallback_with_overrides_only(tmp_path):
    write(tmp_path, "physics_overrides.json",
          {"parts": {"new_link": {"friction": 0.9}}, "joints": {"j": {"friction": 0.02}}})
    parts, joint_overrides, joint_defaults, source = resolve_articulation_physics(
        str(tmp_path), fallback_parts_fn=lambda: [])
    assert source == "legacy_fallback"
    assert parts == [{"name": "new_link", "friction": 0.9}]
    assert joint_overrides == {"j": {"friction": 0.02}}


def test_merge_parts_properties_appends_unknown_links():
    merged = merge_parts_properties(
        [{"name": "a", "mass_kg": 1.0}], {"a": {"mass_kg": 2.0}, "b": {"friction": 0.7}})
    assert merged == [{"name": "a", "mass_kg": 2.0}, {"name": "b", "friction": 0.7}]


# ---------------------------------------------------------------------------
# Joint dynamics preservation (import_articulated_object contract)
# ---------------------------------------------------------------------------

def test_joint_dynamics_preserves_urdf_values():
    damping, friction = resolve_joint_dynamics(
        "revolute", existing_attrib={"damping": "0.15", "friction": "0.04"},
        child_props={"joint_damping": 0.5}, override=None,
        revolute_friction=0.01, prismatic_friction=0.4)
    assert (damping, friction) == (0.15, 0.04)


def test_joint_dynamics_defaults_fill_gaps():
    damping, friction = resolve_joint_dynamics(
        "prismatic", existing_attrib=None,
        child_props={"joint_damping": 0.3}, override=None,
        revolute_friction=0.01, prismatic_friction=0.4)
    assert (damping, friction) == (0.3, 0.4)
    damping, friction = resolve_joint_dynamics(
        "revolute", existing_attrib={"damping": "not-a-number"},
        child_props={}, override=None,
        revolute_friction=0.01, prismatic_friction=0.4)
    assert (damping, friction) == (0.5, 0.01)


def test_joint_dynamics_override_wins():
    damping, friction = resolve_joint_dynamics(
        "revolute", existing_attrib={"damping": "0.15", "friction": "0.04"},
        child_props={"joint_damping": 0.5}, override={"damping": 0.9},
        revolute_friction=0.01, prismatic_friction=0.4)
    assert (damping, friction) == (0.9, 0.04)


def test_joint_dynamics_hand_edit_beats_pipeline_estimate():
    """A hand edit in the URDF must survive even when pipeline estimates exist
    (they are the defaults tier, not an override)."""
    damping, friction = resolve_joint_dynamics(
        "revolute", existing_attrib={"damping": "7.0", "friction": "0.9"},
        child_props={"joint_damping": 0.15}, override=None,
        revolute_friction=0.01, prismatic_friction=0.4,
        defaults_entry={"damping": 0.15, "friction": 0.04})
    assert (damping, friction) == (7.0, 0.9)


def test_joint_dynamics_defaults_tier_fills_missing_element():
    """URDF lost its <dynamics> (e.g. step-5 republish): the pipeline estimate
    fills in, ahead of the legacy child-props/constant defaults."""
    damping, friction = resolve_joint_dynamics(
        "revolute", existing_attrib=None,
        child_props={"joint_damping": 0.5}, override=None,
        revolute_friction=0.01, prismatic_friction=0.4,
        defaults_entry={"damping": 0.15, "friction": 0.04})
    assert (damping, friction) == (0.15, 0.04)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
