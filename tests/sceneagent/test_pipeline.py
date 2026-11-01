"""End-to-end phase-1 run on the checked-in capture and a SimFoundry name list."""

import json
from pathlib import Path

from sceneagent.adapter import from_simfoundry_scene, load_document
from sceneagent.pipeline import run_capture, run_path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "phase1_place_scene.json"


def test_phase1_capture_writes_the_scene_directory(tmp_path):
    summary = run_path(EXAMPLE, tmp_path / "run")
    assert summary["scene_name"] == "phase1_place"
    assert summary["grid_written"] is True
    run = tmp_path / "run"
    grid = json.loads((run / "evaluation" / "sister_grid.json").read_text(encoding="utf-8"))
    assert grid["not_a_learned_policy"] is True
    assert grid["lighting_simulated"] is False
    assert grid["lighting"][0] == "Neutral"
    assert grid["lighting"][-1] == "Overhead spot"
    by_name = {row["sister"]: row for row in grid["rows"]}
    assert by_name["length +40%"]["cells"]["Neutral"]["successes"] == 0
    original = by_name["original"]["cells"]["Hard right"]
    assert 0 < original["successes"] <= original["trials"] == 10
    low, high = original["wilson95"]
    assert low <= original["rate"] <= high
    html = (run / "evaluation" / "report.html").read_text(encoding="utf-8")
    assert "phase1_place" in html
    assert "Wilson 95%" in html
    assert "not a pi0.5" in html
    package = json.loads((run / "package" / "usd_physics_contract.json").read_text(encoding="utf-8"))
    assert package["usd_written"] is False
    box = next(obj for obj in package["objects"] if obj["name"] == "box")
    assert box["material_class"] == "paper and cardboard"
    assert box["mass_kg"] == 0.12
    ledger = json.loads((run / "ledger.json").read_text(encoding="utf-8"))
    decisions = {entry["item"]: entry["decision"] for entry in ledger["entries"]}
    assert decisions["alignment:box"] == "accept"
    assert decisions["alignment:bowl"] == "undecided"
    assert decisions["SCENEAGENT-VIDEO-TO-4D"] == "deferred"
    assert decisions["SCENEAGENT-ADAPT-VLA-POLICIES"] == "deferred"
    physics = json.loads((run / "objects" / "bowl" / "physics_prior.json").read_text(encoding="utf-8"))
    assert physics["material_class"] == "glass and ceramic"


def test_simfoundry_scene_json_is_read_without_meshes(tmp_path):
    scene = {
        "objects_info": {
            "init_info": {
                "marker_0": {"args": {}},
                "cup_0": {"args": {}},
            }
        }
    }
    capture = from_simfoundry_scene(scene, scene_name="from_simfoundry")
    assert capture["source"] == "simfoundry_objects_info"
    assert [obj["name"] for obj in capture["objects"]] == ["marker_0", "cup_0"]
    summary = run_capture(capture, tmp_path / "sf")
    assert summary["grid_written"] is False
    assert summary["materials"]["marker_0"] == "plastic"
    assert summary["materials"]["cup_0"] == "glass and ceramic"
    ledger = json.loads((tmp_path / "sf" / "ledger.json").read_text(encoding="utf-8"))
    sister = next(entry for entry in ledger["entries"] if entry["item"] == "sisters:marker_0")
    assert sister["decision"] == "deferred"
    loaded = load_document(EXAMPLE)
    assert loaded["scene_name"] == "phase1_place"
