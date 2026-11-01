"""Gap-matrix integrity and a real SimFoundry example video."""

import json
from pathlib import Path

from sceneagent.catalog import counts, rows, to_markdown
from sceneagent.video import inspect_video, run_video

ROOT = Path(__file__).resolve().parents[2]
FRUITS = ROOT / "docs" / "assets" / "example_videos" / "Fruits.mp4"


def test_matrix_does_not_name_another_application_repo():
    blob = json.dumps(rows()).lower()
    assert "sceneagentx" not in blob
    assert "sandy-t-shelvan/sceneagentx" not in blob
    tally = counts()
    assert tally["total"] == len(rows())
    assert tally["done"] + tally["partial"] + tally["missing"] == tally["total"]
    assert tally["done"] >= 4
    text = to_markdown()
    assert "Gap matrix" in text
    assert str(tally["total"]) in text
    for row in rows():
        assert row["status"] in {"done", "partial", "missing"}
        assert row["paper"]
        assert row["simfoundry"]
        assert row["next"]


def test_fruits_example_video_is_an_mp4_with_a_duration():
    info = inspect_video(FRUITS)
    assert info["name"] == "Fruits.mp4"
    assert info["bytes"] == FRUITS.stat().st_size
    assert info["bytes"] > 1_000_000
    assert info["container"] == "mp4"
    assert info["duration_s"] > 1.0
    assert info["reconstruction"] == "not_run"


def test_video_run_defers_colmap(tmp_path):
    summary = run_video(FRUITS, tmp_path / "fruits")
    assert summary["colmap_available"] is False
    ledger = json.loads((tmp_path / "fruits" / "ledger.json").read_text(encoding="utf-8"))
    decisions = {entry["item"]: entry["decision"] for entry in ledger["entries"]}
    assert decisions["video"] == "accept"
    assert decisions["COLMAP camera poses"] == "deferred"
    assert ledger["source"] == "simfoundry_example_video"
