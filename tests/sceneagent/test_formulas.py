"""Unit tests for the published SceneAgent formulas used in phase 1."""

import math

import numpy as np
import pytest

from sceneagent.alignment import accept_alignment
from sceneagent.association import (
    centroid_threshold,
    gaussian_belongs,
    mask_cell_side_px,
    masks_same_object,
)
from sceneagent.evaluation import score_trials
from sceneagent.metrics import f1_at_threshold, score_reconstruction, symmetric_chamfer, wilson_interval
from sceneagent.physics import build_physics_record, resolve_body_type, vote_material
from sceneagent.sisters import box_mesh, geometry_sisters, mesh_volume


def test_unknown_class_weight_is_scaled_by_0_4():
    winner, scores = vote_material([
        {"class": "unknown", "weight": 1.0},
        {"class": "plastic", "weight": 0.3},
    ])
    assert scores["unknown"] == pytest.approx(0.4)
    assert winner == "unknown"
    winner, _ = vote_material([
        {"class": "unknown", "weight": 1.0},
        {"class": "plastic", "weight": 0.5},
    ])
    assert winner == "plastic"


def test_soft_body_threshold_is_inclusive():
    assert resolve_body_type(0.65, explicit="dynamic rigid") == ("soft deformable", "deformability_at_least_0.65")
    assert resolve_body_type(0.649, explicit="dynamic rigid") == ("dynamic rigid", "explicit_proposal")
    assert resolve_body_type(None, explicit=None) == (None, "unresolved")


def test_mass_is_stored_as_a_logarithm():
    record = build_physics_record(
        "marker",
        [{"source": "keyword", "class": "plastic", "weight": 1.0}],
        numerical={"mass_kg": 0.12, "density_kg_m3": 700.0, "rigidity": 0.4},
    )
    assert record["material_class"] == "plastic"
    assert record["log_mass"] == pytest.approx(math.log(0.12))
    assert record["log_density"] == pytest.approx(math.log(700.0))
    assert record["unpublished_class_lookups"] == [
        "youngs_modulus",
        "poisson_ratio",
        "rendering_material",
    ]


def test_mask_grouping_uses_published_cutoffs():
    extent = 0.2
    assert centroid_threshold(extent) == pytest.approx(0.10)
    assert centroid_threshold(0.1) == pytest.approx(0.06)
    assert masks_same_object(0.06, 0.1, 0.18, 0.0)
    assert not masks_same_object(0.0601, 0.1, 0.18, 0.0)
    assert masks_same_object(0.05, 0.2, 0.0, 0.08)
    assert not masks_same_object(0.05, 0.2, 0.179, 0.079)


def test_gaussian_front_surface_band():
    # eps_a = 0.08 dominates when 0.03 * depth is smaller.
    assert gaussian_belongs(True, 1.08, 1.0)
    assert not gaussian_belongs(True, 1.0801, 1.0)
    assert not gaussian_belongs(False, 1.0, 1.0)
    # At depth 10, eps_r * depth = 0.3 dominates.
    assert gaussian_belongs(True, 10.3, 10.0)
    assert not gaussian_belongs(True, 10.31, 10.0)


def test_mask_cell_side_is_capped_at_16():
    assert mask_cell_side_px(1920) == 10
    assert mask_cell_side_px(4000) == 16


def test_f1_and_chamfer_on_known_clouds():
    cloud = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.02, 0.0]])
    assert symmetric_chamfer(cloud, cloud) == pytest.approx(0.0)
    assert f1_at_threshold(cloud, cloud, tau=0.01) == pytest.approx(1.0)
    shifted = cloud + np.array([0.05, 0.0, 0.0])
    assert f1_at_threshold(cloud, shifted, tau=0.01) == pytest.approx(0.0)
    scored = score_reconstruction(
        {"a": cloud, "b": shifted},
        {"a": cloud, "missing": shifted + np.array([1.0, 0.0, 0.0])},
    )
    assert scored["f1_by_reference"]["missing"] == 0.0
    assert 0.0 <= scored["f1_mean"] <= 1.0


def test_wilson_matches_the_website_interval_for_one_of_ten():
    low, high = wilson_interval(1, 10)
    assert low == pytest.approx(0.018, abs=0.002)
    assert high == pytest.approx(0.404, abs=0.002)
    low0, high0 = wilson_interval(0, 10)
    assert low0 == pytest.approx(0.0, abs=1e-9)
    assert high0 == pytest.approx(0.278, abs=0.005)


def test_alignment_requires_caller_supplied_thresholds():
    residuals = {"normalized_median": 0.01, "p90": 0.02, "coverage": 0.9, "silhouette": 0.8}
    undecided = accept_alignment(residuals, {})
    assert undecided["decision"] == "undecided"
    assert undecided["accepted"] is False
    thresholds = {
        "normalized_median": {"max": 0.05},
        "p90": {"max": 0.08},
        "coverage": {"min": 0.5},
        "silhouette": {"min": 0.5},
    }
    assert accept_alignment(residuals, thresholds)["accepted"] is True
    residuals["p90"] = 0.2
    rejected = accept_alignment(residuals, thresholds)
    assert rejected["decision"] == "reject"
    assert "p90" in rejected["failed"]


def test_box_volume_scales_with_one_axis():
    verts, faces = box_mesh((1.0, 1.0, 1.0))
    assert mesh_volume(verts, faces) == pytest.approx(1.0)
    sisters = geometry_sisters((0.096, 0.04, 0.04), target_inner=(0.10, 0.10, 0.08), length_scales=(1.4,))
    by_name = {row["name"]: row for row in sisters}
    assert by_name["original"]["accepted"] is True
    assert by_name["length +40%"]["accepted"] is False
    assert "does_not_fit_target" in by_name["length +40%"]["reasons"]
    assert by_name["length +40%"]["volume_ratio"] == pytest.approx(1.4)


def test_scripted_trials_fail_outside_the_reach_disc():
    starts = [(0.0, 0.0), (0.04, 0.0), (0.02, 0.02)]
    successes, rows = score_trials(starts, reach_radius_m=0.03, sister_accepted=True)
    assert successes == 2
    assert rows[1]["success"] is False
    successes, _ = score_trials(starts, reach_radius_m=0.03, sister_accepted=False)
    assert successes == 0
