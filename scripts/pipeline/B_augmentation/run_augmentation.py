# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified runner for reconstructed scene augmentation."""

from __future__ import annotations

import argparse
import time

from simfoundry import REPO_DIR
from simfoundry.pipeline.orchestrator import format_duration, run_pipeline


PHASE_STAGE_IDS = {
    "object-cousins": ["1", "2", "3", "4", "5"],
    "scene-variations": ["6"],
    "task-generation": ["7"],
    "p2p": ["8"],
}


def _stage_ids_for_phases(phases_csv: str | None) -> str | None:
    if not phases_csv:
        return None
    stage_ids: list[str] = []
    for raw_phase in phases_csv.split(","):
        phase = raw_phase.strip()
        if not phase:
            continue
        if phase not in PHASE_STAGE_IDS:
            valid = ", ".join(sorted(PHASE_STAGE_IDS))
            raise ValueError(f"Unsupported phase '{phase}'. Expected one of: {valid}")
        stage_ids.extend(PHASE_STAGE_IDS[phase])
    return ",".join(dict.fromkeys(stage_ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include", default=None, help="Comma-separated stage ids to include")
    parser.add_argument("--exclude", default=None, help="Comma-separated stage ids to exclude")
    parser.add_argument("--phases", default=None, help="Comma-separated phases: object-cousins,scene-variations,task-generation,p2p")
    parser.add_argument("--include-p2p", action="store_true", help="Enable optional point-to-point cousin matching stage")
    parser.add_argument("--exec-mode", choices=["mamba", "direct"], default="mamba")
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--env-simfoundry", default="simfoundry")
    parser.add_argument("--env-mesh", default="hunyuan")
    parser.add_argument("--env-b1k", default="simfoundry")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides forwarded to each stage")
    args = parser.parse_args()

    include_ids = args.include or _stage_ids_for_phases(args.phases)

    run_pipeline(
        cwd=REPO_DIR,
        pipeline_name="augmentation",
        input_mode="video",
        include_ids_csv=include_ids,
        exclude_ids_csv=args.exclude,
        exec_mode=args.exec_mode,
        python_bin=args.python_bin,
        env_map={"simfoundry": args.env_simfoundry, "mesh": args.env_mesh, "b1k": args.env_b1k, "da3": "da3"},
        dry_run=args.dry_run,
        stream_subseq_enabled=False,
        stream_start_stage=5,
        stream_end_stage=8,
        extra_overrides=args.overrides,
        include_p2p=args.include_p2p or (include_ids is not None and "8" in include_ids.split(",")),
    )


if __name__ == "__main__":
    start = time.perf_counter()
    try:
        main()
    finally:
        print(f"[B_augmentation] total wall time: {format_duration(time.perf_counter() - start)}")
