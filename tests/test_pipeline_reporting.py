# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from simfoundry.pipeline.reporting import build_scene_manifest, write_pipeline_report


def test_scene_manifest_records_stable_stage_invariants(tmp_path):
    scene_dir = tmp_path / "Data" / "home_coffee_4"
    (scene_dir / "s1_video" / "frames_subsampled_15").mkdir(parents=True)
    (scene_dir / "s5_scene" / "obj_cat_list").mkdir(parents=True)
    (scene_dir / "s6_upsample" / "upsampled").mkdir(parents=True)
    (scene_dir / "s7_mesh" / "textured_mesh" / "hunyuan").mkdir(parents=True)
    (scene_dir / "s8_pose" / "info").mkdir(parents=True)
    (scene_dir / "s11_sim").mkdir(parents=True)

    (scene_dir / "s1_video" / "stage_info.json").write_text(json.dumps({"success": True}), encoding="utf-8")
    (scene_dir / "s1_video" / "frames_subsampled_15" / "frame_0001.png").write_bytes(b"png")
    (scene_dir / "s5_scene" / "obj_cat_list" / "iter_1.json").write_text("{}", encoding="utf-8")
    (scene_dir / "s6_upsample" / "upsampled" / "iter_1_transparent.png").write_bytes(b"png")
    (scene_dir / "s7_mesh" / "textured_mesh" / "hunyuan" / "iter_1_mesh.glb").write_bytes(b"glb")
    (scene_dir / "s8_pose" / "info" / "iter_1.json").write_text("{}", encoding="utf-8")
    (scene_dir / "s11_sim" / "scene_objects_info.json").write_text(json.dumps({"1": {}}), encoding="utf-8")

    manifest = build_scene_manifest(scene_dir)

    assert manifest["invariants"]["s1_subsampled_frames"] == 1
    assert manifest["invariants"]["s11_scene_objects"] == 1
    assert manifest["stages"]["1_video"]["stage_info"]["success"] is True
    assert len(manifest["manifest_signature"]) == 64


def test_write_pipeline_report_writes_manifest_and_report(tmp_path):
    scene_dir = tmp_path / "Data" / "home_coffee_4"
    scene_dir.mkdir(parents=True)

    written = write_pipeline_report(
        scene_dir=scene_dir,
        stage_durations_s={"1b": 1.5},
        wall_time_s=2.0,
        memory_samples={"1b": {"before_used_gb": None, "after_used_gb": None}},
    )

    assert Path(written["manifest_path"]).exists()
    report = json.loads(Path(written["report_path"]).read_text(encoding="utf-8"))
    assert report["stage_durations_s"] == {"1b": 1.5}
    assert report["wall_time_s"] == 2.0
