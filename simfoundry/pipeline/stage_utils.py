# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared runtime helpers for script-based pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from simfoundry import CFG_DIR
from simfoundry.utils.processing_utils import dump_json


@dataclass(frozen=True)
class StageResult:
    """Serialized result info for a pipeline stage run."""

    success: bool
    additional_info: dict[str, Any] | None = None


def bootstrap_hydra_workdir(script_file: str) -> str:
    """Switch cwd to scripts/cfg to keep Hydra path behavior stable across scripts."""
    os.chdir(CFG_DIR)
    return CFG_DIR


def resolve_base_iteration(scene_dir: str, idx: int) -> int | None:
    """Which prior stage-5 iteration's outputs object `idx` was detected against.

    Stage 5 records this as `base_iter` in obj_cat_list/iter_{idx}.json. Iteration
    numbers can have gaps (a detection pass that finds no masks consumes an index
    without writing artifacts), so `idx - 1` is not necessarily an iteration that
    exists. Returns None when the object was detected on the original source frame.

    Data recorded before `base_iter` existed derives it from the artifacts instead:
    the last iteration below `idx` that wrote a post-removal image — the same rule
    stage 5 uses to resume — which is correct for gapped and contiguous scenes alike.
    """
    manifest_fpath = os.path.join(scene_dir, "obj_cat_list", f"iter_{idx}.json")
    if os.path.isfile(manifest_fpath):
        with open(manifest_fpath) as f:
            manifest = json.load(f)
        if "base_iter" in manifest:
            return manifest["base_iter"]
    prior_iters = [
        int(p.stem.split("_")[-1])
        for p in Path(scene_dir, "post_object_removal").glob("iter_*.png")
    ]
    return max((i for i in prior_iters if i < idx), default=None)


def finalize_stage(stage_cfg: Any, out_dir: str | Path, result: StageResult) -> dict[str, Any]:
    """Write a `stage_info.json` file and return the serialized payload."""
    payload = OmegaConf.to_object(stage_cfg)
    payload["success"] = result.success
    if result.additional_info:
        payload.update(result.additional_info)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dump_json(payload, str(out_path / "stage_info.json"))
    return payload


def resolve_generation_kwargs(cfg: Any, stage_cfg: Any) -> dict[str, Any]:
    """Resolve a mesh stage's backend generation options into a plain dict.

    Reads `stage_cfg.generation_kwargs` and defaults `seed` to the global `cfg.seed`, so a run is
    reproducible without repeating the seed per stage (backends otherwise fall back to their own
    hardcoded default, e.g. Pixal3D's 42).

    Returns a plain dict rather than a DictConfig so it can be splatted into generator calls,
    which take ordinary Python values.
    """
    raw = stage_cfg.get("generation_kwargs", None)
    if raw is None:
        raw = {}
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    kwargs = dict(raw)
    if "seed" not in kwargs:
        seed = cfg.get("seed", None)
        if seed is not None:
            kwargs["seed"] = int(seed)
    return kwargs


def parse_iter_index(name: str, *, prefix: str = "iter_") -> int | None:
    """Extract integer index from strings that contain `<prefix><int>`."""
    if prefix not in name:
        return None
    try:
        return int(name.split(prefix)[-1])
    except ValueError:
        return None


def list_object_iteration_indices(filenames: list[str], *, suffix: str) -> list[int]:
    """Return sorted unique iteration ids from artifact file names."""
    ids = set()
    for name in filenames:
        if not name.endswith(suffix):
            continue
        idx = parse_iter_index(name.removesuffix(suffix))
        if idx is not None:
            ids.add(idx)
    return sorted(ids)
