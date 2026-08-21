# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline manifest and run-report helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any


STAGE_DIRS = {
    "1_video": "s1_video",
    "2_da": "s2_da",
    "2_depth": "s2_depth",
    "3_ground": "s3_ground",
    "4_frame": "s4_frame",
    "5_scene": "s5_scene",
    "6_upsample": "s6_upsample",
    "7_mesh": "s7_mesh",
    "8_pose": "s8_pose",
    "9_articulate": "s9_articulate_objects",
    "10_compile": "s10_compile",
    "11_sim": "s11_sim",
    "12_physics": "s12_physics",
    "13_usd": "s13_usd",
    "14_og": "s14_og",
}


def _count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.glob(pattern) if item.is_file())


def _dir_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "file_count": 0, "total_bytes": 0}
    file_count = 0
    total_bytes = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            total_bytes += item.stat().st_size
    return {"exists": True, "file_count": file_count, "total_bytes": total_bytes}


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"json_error": True}


def _stage_info(stage_dir: Path) -> dict[str, Any]:
    payload = _load_json(stage_dir / "stage_info.json")
    if payload is None:
        return {"exists": False, "success": None}
    if isinstance(payload, dict) and payload.get("json_error"):
        return {"exists": True, "success": None, "json_error": True}
    return {"exists": True, "success": payload.get("success") if isinstance(payload, dict) else None}


def _textured_mesh_count(scene_dir: Path) -> int:
    """Count stage 7 textured meshes, whichever mesh backend produced them.

    Stage 7 writes to `textured_mesh/<texture_model>/`, so the directory name follows the
    configured backend (hunyuan, trellis2, direct3d, ...). Prefer the model recorded in the
    stage's own `stage_info.json`; if the stage has not written one yet, fall back to summing
    every backend directory so a partial or hand-run stage still reports a count.
    """
    textured_dir = scene_dir / "s7_mesh" / "textured_mesh"
    stage_info = _load_json(scene_dir / "s7_mesh" / "stage_info.json")
    if isinstance(stage_info, dict):
        texture_model = stage_info.get("texture_model")
        if isinstance(texture_model, str) and texture_model:
            return _count_files(textured_dir / texture_model, "*_mesh.glb")
    if not textured_dir.is_dir():
        return 0
    return sum(_count_files(backend_dir, "*_mesh.glb") for backend_dir in textured_dir.iterdir() if backend_dir.is_dir())


def _scene_object_count(scene_dir: Path) -> int:
    payload = _load_json(scene_dir / "s11_sim" / "scene_objects_info.json")
    return len(payload) if isinstance(payload, dict) else 0


def build_scene_manifest(scene_dir: str | Path) -> dict[str, Any]:
    scene_dir = Path(scene_dir)
    stages: dict[str, Any] = {}
    for stage_id, dirname in STAGE_DIRS.items():
        stage_dir = scene_dir / dirname
        stages[stage_id] = {
            **_dir_stats(stage_dir),
            "stage_info": _stage_info(stage_dir),
        }

    invariants = {
        "s1_subsampled_frames": _count_files(scene_dir / "s1_video" / "frames_subsampled_15", "*.png"),
        "s5_object_categories": _count_files(scene_dir / "s5_scene" / "obj_cat_list", "*.json"),
        "s6_upsampled_objects": _count_files(scene_dir / "s6_upsample" / "upsampled", "*_transparent.png"),
        "s7_textured_meshes": _textured_mesh_count(scene_dir),
        "s8_pose_infos": _count_files(scene_dir / "s8_pose" / "info", "*.json"),
        "s11_scene_objects": _scene_object_count(scene_dir),
    }
    manifest_core = {
        "scene_dir": str(scene_dir),
        "stages": stages,
        "invariants": invariants,
    }
    signature = hashlib.sha256(json.dumps(manifest_core, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {**manifest_core, "manifest_signature": signature}


def write_pipeline_report(
    *,
    scene_dir: str | Path,
    stage_durations_s: dict[str, float],
    wall_time_s: float,
    memory_samples: dict[str, dict[str, float | None]],
) -> dict[str, Path]:
    scene_dir = Path(scene_dir)
    manifest = build_scene_manifest(scene_dir)
    manifest_path = scene_dir / "pipeline_manifest.json"
    report_path = scene_dir / "pipeline_run_report.json"
    streaming_report_path = scene_dir / "streaming_resource_report.json"
    streaming_report = _load_json(streaming_report_path)

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scene_dir": str(scene_dir),
        "cache_mode": os.environ.get("CACHE_MODE", ""),
        "test_mode": os.environ.get("TEST_MODE", ""),
        "model_cache_dir": os.environ.get("SIMFOUNDRY_MODEL_CACHE_DIR", ""),
        "stage_durations_s": stage_durations_s,
        "wall_time_s": wall_time_s,
        "memory_samples": memory_samples,
        "manifest_signature": manifest["manifest_signature"],
        "streaming_resource_report": streaming_report,
    }

    for path, payload in ((manifest_path, manifest), (report_path, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")

    return {"manifest_path": manifest_path, "report_path": report_path}
