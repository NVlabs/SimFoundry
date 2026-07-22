# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for streaming contiguous subsequences of stages 5-8."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from digital_cousins.pipeline.stage_utils import list_object_iteration_indices


SUPPORTED_STAGES = (5, 6, 7, 8)


@dataclass(frozen=True)
class StageIO:
    stage: int
    watch_dir_fn: Callable[[object], str]
    artifact_suffix: str
    override_key: str | None


STAGE_IO = {
    5: StageIO(stage=5, watch_dir_fn=lambda cfg: f"{cfg.s5_scene_gmask.out_dir}/obj_cat_list", artifact_suffix=".json", override_key=None),
    6: StageIO(stage=6, watch_dir_fn=lambda cfg: f"{cfg.s6_upsample.out_dir}/upsampled", artifact_suffix="_transparent.png", override_key="s6_upsample.object_indices"),
    7: StageIO(stage=7, watch_dir_fn=lambda cfg: f"{cfg.s7_mesh.out_dir}/textured_mesh/{cfg.s7_mesh.texture_model}", artifact_suffix="_mesh.glb", override_key="s7_mesh.object_indices"),
    8: StageIO(stage=8, watch_dir_fn=lambda cfg: f"{cfg.s8_pose.out_dir}/info", artifact_suffix=".json", override_key="s8_pose.object_indices"),
}


def validate_subsequence(start_stage: int, end_stage: int) -> list[int]:
    if start_stage not in SUPPORTED_STAGES or end_stage not in SUPPORTED_STAGES:
        raise ValueError(f"start/end must be in {SUPPORTED_STAGES}")
    if start_stage > end_stage:
        raise ValueError("start_stage must be <= end_stage")
    return list(range(start_stage, end_stage + 1))


def subsequence_complete(stages: list[int], expected_indices: set[int], completed_artifacts: dict[int, set[int]]) -> bool:
    if not expected_indices:
        return False
    return expected_indices.issubset(completed_artifacts[stages[-1]])


def discover_ready_indices(cfg, stage: int) -> list[int]:
    return sorted(discover_ready_artifact_mtimes(cfg, stage))


def discover_ready_artifact_mtimes(cfg, stage: int) -> dict[int, float]:
    info = STAGE_IO[stage]
    watch_dir = info.watch_dir_fn(cfg)
    if not Path(watch_dir).is_dir():
        return {}
    mtimes: dict[int, float] = {}
    for path in Path(watch_dir).iterdir():
        if not path.is_file() or not path.name.endswith(info.artifact_suffix):
            continue
        indices = list_object_iteration_indices([path.name], suffix=info.artifact_suffix)
        if indices:
            mtimes[indices[0]] = path.stat().st_mtime
    return mtimes


def per_index_override(stage: int, idx: int) -> list[str]:
    key = STAGE_IO[stage].override_key
    if key is None:
        return []
    return [f"{key}=[{idx}]"]
