# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared runtime helpers for script-based pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
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
