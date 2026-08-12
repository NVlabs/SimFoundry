# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for streaming contiguous subsequences of stages 5-8."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

from simfoundry.pipeline.stage_utils import list_object_iteration_indices
from simfoundry.utils.python_utils import PARTIAL_MARKER


SUPPORTED_STAGES = (5, 6, 7, 8)

#: Seconds an artifact must go without being written to before a downstream stage may open it.
#: Producers now publish atomically (see atomic_output_path), so this is a backstop for any
#: writer that still updates a watched path in place rather than the primary guarantee. It was
#: the primary guarantee once, and could not hold: a producer may stall mid-write for an
#: unbounded time, so no window is large enough. Kept small because it costs a poll.
DEFAULT_ARTIFACT_SETTLE_S = 1.0


@dataclass(frozen=True)
class StageIO:
    stage: int
    watch_dir_fn: Callable[[object], str]
    artifact_suffix: str
    override_key: str | None


STAGE_IO = {
    5: StageIO(stage=5, watch_dir_fn=lambda cfg: f"{cfg.s5_scene.out_dir}/obj_cat_list", artifact_suffix=".json", override_key=None),
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


def artifact_settle_s(cfg) -> float:
    stream_cfg = getattr(cfg, "stream_subseq", None)
    if stream_cfg is None:
        return DEFAULT_ARTIFACT_SETTLE_S
    if hasattr(stream_cfg, "get"):
        value = stream_cfg.get("artifact_settle_s", None)
    else:
        value = getattr(stream_cfg, "artifact_settle_s", None)
    return DEFAULT_ARTIFACT_SETTLE_S if value is None else float(value)


def discover_ready_indices(cfg, stage: int, *, settle_s: float | None = None, now: float | None = None) -> list[int]:
    return sorted(discover_ready_artifact_mtimes(cfg, stage, settle_s=settle_s, now=now))


def discover_ready_artifact_mtimes(
    cfg, stage: int, *, settle_s: float | None = None, now: float | None = None,
) -> dict[int, float]:
    """Indices whose artifact exists and has stopped being written to.

    An artifact is only reported once `settle_s` has passed since its last write, so a
    downstream stage never opens a file the producer is still appending to.
    """
    info = STAGE_IO[stage]
    watch_dir = info.watch_dir_fn(cfg)
    if not Path(watch_dir).is_dir():
        return {}
    settle_s = artifact_settle_s(cfg) if settle_s is None else float(settle_s)
    now = time.time() if now is None else now
    mtimes: dict[int, float] = {}
    for path in Path(watch_dir).iterdir():
        if not path.is_file() or not path.name.endswith(info.artifact_suffix):
            continue
        if PARTIAL_MARKER in path.name:
            # Producers build artifacts under `<name>.partial.<ext>` and rename them into
            # place, so a partial is by definition not ready. Stated explicitly rather than
            # relying on the index parse below happening to reject the name.
            continue
        indices = list_object_iteration_indices([path.name], suffix=info.artifact_suffix)
        if not indices:
            continue
        mtime = path.stat().st_mtime
        if now - mtime < settle_s:
            continue
        mtimes[indices[0]] = mtime
    return mtimes


def per_index_override(stage: int, idx: int) -> list[str]:
    key = STAGE_IO[stage].override_key
    if key is None:
        return []
    return [f"{key}=[{idx}]"]
