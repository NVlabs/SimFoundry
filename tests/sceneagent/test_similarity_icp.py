"""Section 3.3 trimmed similarity ICP, and the section 3.7 mesh USDA."""

import json

import numpy as np

from sceneagent.alignment import accept_alignment
from sceneagent.similarity_icp import apply_similarity, measure_residuals, trimmed_similarity_icp
from sceneagent.usd_contract import write_aligned_mesh_usda


def test_icp_recovers_a_known_similarity_and_does_not_invent_a_gate():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(48, 3))
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = apply_similarity(source, 2.5, rotation, np.array([4.0, -1.0, 0.25]))
    extras = rng.normal(size=(12, 3)) * 30.0
    result = trimmed_similarity_icp(source, np.vstack([target, extras]), iterations=25)
    aligned = apply_similarity(source, result["scale"], result["rotation"], result["translation"])
    gap = np.median(np.linalg.norm(aligned - target, axis=1))
    assert gap < 1e-3
    residuals = measure_residuals(result, np.vstack([target, extras]))
    assert residuals["silhouette"] is None
    assert residuals["thresholds_published"] is False
    decision = accept_alignment(
        {
            "normalized_median": residuals["normalized_median"],
            "p90": residuals["p90"],
            "coverage": residuals["coverage"],
            "silhouette": 0.0,
        },
        {},
    )
    assert decision["decision"] == "undecided"
    assert decision["accepted"] is False


def test_aligned_mesh_usda_keeps_the_bowl_and_replaces_the_box(tmp_path):
    contract = {
        "objects": [
            {
                "name": "box",
                "rigid_body": True,
                "mass_kg": 0.12,
                "density_kg_m3": 700.0,
                "friction_static": 0.45,
                "friction_dynamic": 0.35,
                "restitution": 0.05,
                "body_type": "dynamic rigid",
                "collision_approximation": "box",
                "material_class": "paper and cardboard",
            },
            {
                "name": "bowl",
                "rigid_body": True,
                "mass_kg": None,
                "density_kg_m3": None,
                "friction_static": None,
                "friction_dynamic": None,
                "restitution": None,
                "body_type": "static rigid",
                "collision_approximation": "convex hull",
                "material_class": "glass and ceramic",
            },
        ]
    }
    written = write_aligned_mesh_usda(
        contract,
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        [(0, 1, 2)],
        tmp_path / "aligned.usda",
    )
    text = written.read_text(encoding="utf-8")
    assert 'def Xform "aligned_mesh"' in text
    assert 'def Mesh "geometry"' in text
    assert 'def Xform "bowl"' in text
    assert 'def Xform "box"' not in text
    assert "physics:mass = 0.12" in text
