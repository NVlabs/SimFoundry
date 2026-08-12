# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified runner for steps 1-13 with optional streaming for contiguous stages 5-8."""

from __future__ import annotations

import argparse
import os
import time

from simfoundry import REPO_DIR
from simfoundry.pipeline.orchestrator import format_duration, run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mode", choices=["video", "stereo"], default="video")
    parser.add_argument("--include", default=None, help="Comma-separated stage ids to include (e.g. 2,3,4)")
    parser.add_argument("--exclude", default=None, help="Comma-separated stage ids to exclude")
    parser.add_argument("--exec-mode", choices=["mamba", "direct"], default="mamba")
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--env-simfoundry", default="simfoundry")
    parser.add_argument("--env-nerfstudio", default="nerfstudio_simfoundry")
    parser.add_argument("--env-da3", default="da3")
    parser.add_argument("--env-mesh", default="hunyuan")
    parser.add_argument("--env-b1k", default="simfoundry")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stream-5-8", action="store_true", help="Enable streaming for a contiguous subsequence in stages 5-8")
    parser.add_argument("--stream-start-stage", type=int, default=5, help="Streaming subsequence start stage (5-8)")
    parser.add_argument("--stream-end-stage", type=int, default=8, help="Streaming subsequence end stage (5-8)")
    parser.add_argument("--detect-articulation", action="store_true", help="Run stage 8b after pose matching to decompose articulated objects (requires the optional articulate envs; ignored with a warning if absent)")
    parser.add_argument("--bg-splat", action="store_true", help="Include stage 2c (nerfstudio background Gaussian splat). Opt-in only — not enabled by default.")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides forwarded to each stage")
    args = parser.parse_args()

    repo_root = REPO_DIR
    stream_enabled = args.stream_5_8
    stream_start = args.stream_start_stage
    stream_end = args.stream_end_stage

    run_pipeline(
        cwd=repo_root,
        pipeline_name="reconstruction",
        input_mode=args.input_mode,
        include_ids_csv=args.include,
        exclude_ids_csv=args.exclude,
        exec_mode=args.exec_mode,
        python_bin=args.python_bin,
        env_map={
            "simfoundry": args.env_simfoundry,
            "nerfstudio": args.env_nerfstudio,
            "da3": args.env_da3,
            "mesh": args.env_mesh,
            "b1k": args.env_b1k,
        },
        dry_run=args.dry_run,
        stream_subseq_enabled=stream_enabled,
        stream_start_stage=stream_start,
        stream_end_stage=stream_end,
        extra_overrides=args.overrides,
        detect_articulation=args.detect_articulation,
        bg_splat=args.bg_splat,
    )


if __name__ == "__main__":
    start = time.perf_counter()
    try:
        main()
    finally:
        print(f"[Pipeline] total wall time: {format_duration(time.perf_counter() - start)}")
