# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified runner for application, evaluation, and training stages."""

from __future__ import annotations

import argparse
import time

from simfoundry import REPO_DIR
from simfoundry.pipeline.orchestrator import format_duration, run_pipeline


MODE_INCLUDES = {
    "smoke-random": "smoke",
    "eval": "1",
    "demo": "2,3,3b,4,5,6",
    "full": "1,2,3,3b,4,5,6",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODE_INCLUDES), default="smoke-random")
    parser.add_argument("--include", default=None, help="Comma-separated stage ids to include")
    parser.add_argument("--exclude", default=None, help="Comma-separated stage ids to exclude")
    parser.add_argument("--exec-mode", choices=["mamba", "direct"], default="mamba")
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--env-simfoundry", default="simfoundry")
    parser.add_argument("--env-da3", default="da3")
    parser.add_argument("--env-mesh", default="hunyuan")
    parser.add_argument("--env-b1k", default="simfoundry")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Additional overrides forwarded to each stage")
    args = parser.parse_args()

    include_ids = args.include or MODE_INCLUDES[args.mode]
    run_pipeline(
        cwd=REPO_DIR,
        pipeline_name="application",
        input_mode="video",
        include_ids_csv=include_ids,
        exclude_ids_csv=args.exclude,
        exec_mode=args.exec_mode,
        python_bin=args.python_bin,
        env_map={"simfoundry": args.env_simfoundry, "b1k": args.env_b1k, "mesh": args.env_mesh, "da3": args.env_da3},
        dry_run=args.dry_run,
        stream_subseq_enabled=False,
        stream_start_stage=5,
        stream_end_stage=8,
        extra_overrides=args.overrides,
    )


if __name__ == "__main__":
    start = time.perf_counter()
    try:
        main()
    finally:
        print(f"[C_application] total wall time: {format_duration(time.perf_counter() - start)}")
