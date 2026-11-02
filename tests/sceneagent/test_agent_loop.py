"""Section 3.1 loop: a rejected tool is repaired and the ledger records both."""

import json
from pathlib import Path

from sceneagent.agent_loop import append_step, parse_material_class, parse_plan, run_agent_loop
from sceneagent.gemini_client import choose_flash_model
from sceneagent.mjcf_contract import write_physics_mjcf
from sceneagent.pipeline import run_path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "phase1_place_scene.json"


def test_rejection_reruns_the_repair_tool(tmp_path):
    calls = []

    def fail_first(_context):
        calls.append("mujoco_load")
        return {"ok": False, "artifact": "scene.usda", "reason": "XML parse error line 1", "detail": {}}

    def repair(_context):
        calls.append("mujoco_load_repair")
        return {"ok": True, "artifact": "scene.xml", "reason": "", "detail": {"nbody": 3}}

    summary = run_agent_loop(
        tmp_path / "loop",
        {"mujoco_load": fail_first, "mujoco_load_repair": repair},
        ["mujoco_load"],
    )
    assert calls == ["mujoco_load", "mujoco_load_repair"]
    decisions = [(entry["decision"], entry["repair"]) for entry in summary["entries"]]
    assert decisions == [("reject", False), ("accept", True)]
    ledger = json.loads((tmp_path / "loop" / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["entries"][1]["attempt"] == 2
    assert (tmp_path / "loop" / "manifests" / "mujoco_load-2.json").is_file()
    plan = json.loads((tmp_path / "loop" / "plan.json").read_text(encoding="utf-8"))
    assert plan["steps"][0]["section"] == "3.7"


def test_append_step_keeps_prior_entries_and_accepts_only_when_ok(tmp_path):
    run_agent_loop(
        tmp_path / "loop",
        {"isaac_inspector": lambda _context: {"ok": False, "artifact": None, "reason": "stage closed", "detail": {}}},
        ["isaac_inspector"],
    )
    accepted = append_step(
        tmp_path / "loop",
        "isaac_inspector",
        {"ok": True, "artifact": "stage.txt", "reason": "", "detail": {}},
    )
    ledger = json.loads((tmp_path / "loop" / "ledger.json").read_text(encoding="utf-8"))
    decisions = [entry["decision"] for entry in ledger["entries"]]
    assert decisions == ["reject", "accept"]
    assert accepted["attempt"] == 2
    assert accepted["repair"] is False
    plan = json.loads((tmp_path / "loop" / "plan.json").read_text(encoding="utf-8"))
    assert plan["steps"][0]["id"] == "isaac_inspector"


def test_plan_parser_keeps_only_allowed_steps():
    steps = parse_plan('Here is the plan: {"steps": ["material_vote", "not-a-block", "sam3d"]}')
    assert steps == ["material_vote", "sam3d"]
    assert parse_material_class("The surface is food and organic matter.") == "food and organic matter"
    assert parse_material_class("no class here") is None


def test_flash_fallback_uses_an_id_the_error_names():
    chosen = choose_flash_model(
        "Please update your code to use models/gemini-3.8-flash for the latest features.",
        ["models/gemini-2.5-flash", "models/gemini-3.8-flash", "models/gemini-3.8-flash-tts"],
    )
    assert chosen == "models/gemini-3.8-flash"
    # The 404 names the dead id first. That id must not be selected again.
    skipped = choose_flash_model(
        "This model models/gemini-2.5-flash is no longer available to new users. "
        "Please update your code to use models/gemini-3.8-flash.",
        ["models/gemini-2.5-flash", "models/gemini-3.8-flash"],
    )
    assert skipped == "models/gemini-3.8-flash"


def test_mjcf_from_the_contract_carries_the_box_mass(tmp_path):
    run_path(EXAMPLE, tmp_path / "run")
    contract = json.loads((tmp_path / "run" / "package" / "usd_physics_contract.json").read_text(encoding="utf-8"))
    xml_path = write_physics_mjcf(contract, tmp_path / "scene.xml")
    text = xml_path.read_text(encoding="utf-8")
    assert text.startswith("<mujoco")
    assert 'mass="0.12"' in text
    assert "paper and cardboard" in text
