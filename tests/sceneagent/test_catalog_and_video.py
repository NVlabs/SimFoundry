"""Gap-matrix integrity and a real SimFoundry example video."""

import json
from pathlib import Path

from sceneagent.catalog import counts, rows, to_markdown
from sceneagent.colmap_stage import colmap_major, pose_commands, run_poses, use_gpu_from_banner
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


def test_pose_commands_match_colmap_37_and_skip_gpu_without_cuda():
    assert use_gpu_from_banner("COLMAP 3.7 -- Structure-from-Motion\n(Commit Unknown without CUDA)") is False
    assert use_gpu_from_banner("COLMAP 3.11 with CUDA") is True
    commands = pose_commands("database.db", "images", "sparse", use_gpu=False)
    assert [item[1] for item in commands] == ["feature_extractor", "sequential_matcher", "mapper"]
    extractor = commands[0]
    gpu_at = extractor.index("--SiftExtraction.use_gpu")
    assert extractor[gpu_at + 1] == "0"
    assert "--SiftMatching.use_gpu" in commands[1]
    banner = "COLMAP 4.2.1 (Commit Unknown on Unknown with CUDA, std hash maps)"
    assert colmap_major(banner) == 4
    assert use_gpu_from_banner(banner) is True
    cuda = pose_commands("database.db", "images", "sparse", use_gpu=True, major=4)
    assert "--FeatureExtraction.use_gpu" in cuda[0]
    assert cuda[0][cuda[0].index("--FeatureExtraction.use_gpu") + 1] == "1"
    assert "--log_to_stderr" not in cuda[0]
    assert "--FeatureMatching.use_gpu" in cuda[1]
    blob = json.dumps(commands).lower()
    assert "sceneagentx" not in blob


def test_video_run_defers_colmap(tmp_path, monkeypatch):
    monkeypatch.setattr("sceneagent.external_tools.shutil.which", lambda _name: None)
    summary = run_video(FRUITS, tmp_path / "fruits")
    assert summary["colmap_available"] is False
    assert summary["colmap_decision"] == "deferred"
    assert summary["reconstruction"] == "not_run"
    ledger = json.loads((tmp_path / "fruits" / "ledger.json").read_text(encoding="utf-8"))
    decisions = {entry["item"]: entry["decision"] for entry in ledger["entries"]}
    assert decisions["video"] == "accept"
    assert decisions["COLMAP camera poses"] == "deferred"
    assert ledger["source"] == "simfoundry_example_video"


def test_colmap_stage_does_not_invent_a_run_without_the_binary(tmp_path, monkeypatch):
    monkeypatch.setattr("sceneagent.colmap_stage.shutil.which", lambda _name: None)
    result = run_poses(FRUITS, tmp_path / "colmap")
    assert result["ran"] is False
    assert result["ok"] is False
    assert "PATH" in result["reason"]
