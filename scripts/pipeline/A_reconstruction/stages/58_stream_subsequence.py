# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream contiguous subsequences across stages 5-8.

The first stage in the subsequence is run as a normal producer process.
Each downstream stage listens for per-iteration outputs from the previous stage
and runs with a single-index override as soon as artifacts appear.

Examples:
  python scripts/pipeline/A_reconstruction/stages/58_stream_subsequence.py \
      stream_subseq.start_stage=5 stream_subseq.end_stage=8

  python scripts/pipeline/A_reconstruction/stages/58_stream_subsequence.py \
      stream_subseq.start_stage=6 stream_subseq.end_stage=8

  python scripts/pipeline/A_reconstruction/stages/58_stream_subsequence.py \
      stream_subseq.start_stage=7 stream_subseq.end_stage=8
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import json
import shlex
import subprocess
from collections import defaultdict
from threading import Event, Lock, Thread
import time

import hydra

from simfoundry.pipeline.orchestrator import TIMING_LOG_ENV_VAR
from simfoundry.pipeline.resource_scheduler import SingleGpuMemoryScheduler
from simfoundry.pipeline.stage_utils import bootstrap_hydra_workdir
from simfoundry.pipeline.stream_subsequence import (
    discover_ready_artifact_mtimes,
    per_index_override,
    subsequence_complete,
    validate_subsequence,
)


from simfoundry import CFG_DIR, REPO_DIR
logger = logging.getLogger(__name__)
bootstrap_hydra_workdir(__file__)
REPO_ROOT = REPO_DIR


SCRIPT_BY_STAGE = {
    5: "scripts/pipeline/A_reconstruction/stages/5_decompose_scene.py",
    6: "scripts/pipeline/A_reconstruction/stages/6_upsample_object_images.py",
    7: "scripts/pipeline/A_reconstruction/stages/7_generate_object_meshes.py",
    8: "scripts/pipeline/A_reconstruction/stages/8_match_object_poses.py",
}


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.2f}s"

    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m {rem:.2f}s"


def _append_timing_log(log_path: Path | None, *, stage_id: str, description: str, elapsed_s: float, status: str = "completed") -> None:
    if log_path is None:
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{timestamp} | stage={stage_id} | status={status} | elapsed_s={elapsed_s:.6f} | elapsed={_format_duration(elapsed_s)} | {description}\n"
        )


def _run_cmd(
    cmd: str,
    extra_overrides: list[str],
    *,
    stage_id: int | None = None,
    stage_stats: dict[int, dict[str, float]] | None = None,
    stage_stats_lock: Lock | None = None,
    scheduler: SingleGpuMemoryScheduler | None = None,
):
    full_cmd = f"{cmd} {' '.join(extra_overrides)}".strip()
    logger.info("Running: %s", full_cmd)
    wait_s = 0.0
    start = time.perf_counter()

    if scheduler is not None and stage_id is not None:
        with scheduler.reserve(stage_id) as reservation:
            wait_s = reservation.wait_s
            if wait_s > 0.01:
                logger.info("Stage %s waited %.2fs for GPU memory budget", stage_id, wait_s)
            subprocess.run(shlex.split(full_cmd), check=True, cwd=REPO_ROOT)
    else:
        subprocess.run(shlex.split(full_cmd), check=True, cwd=REPO_ROOT)

    elapsed = time.perf_counter() - start

    if stage_id is not None and stage_stats is not None and stage_stats_lock is not None:
        with stage_stats_lock:
            stats = stage_stats[stage_id]
            stats["elapsed_s"] += elapsed
            stats["resource_wait_s"] += wait_s
            stats["calls"] += 1

    if stage_id is not None:
        logger.info("Stage %s invocation completed in %s", stage_id, _format_duration(elapsed))


def _build_default_cmd(stage: int) -> str:
    return f"python {SCRIPT_BY_STAGE[stage]}"


def _missing_downstream_stages(stages: list[int], stage_stats: dict[int, dict[str, float]]) -> list[int]:
    return [stage for stage in stages[1:] if int(stage_stats[stage]["calls"]) == 0]


def _stage_vram_config(raw_config) -> dict[int, float]:
    if raw_config is None:
        return {}
    return {int(k): float(v) for k, v in dict(raw_config).items()}


def _stream_analytics_path(cfg) -> Path:
    configured = cfg.stream_subseq.get("analytics_path", None)
    if configured:
        return Path(str(configured))
    return Path(str(cfg.root_dir)) / str(cfg.scene_name) / "streaming_resource_report.json"


def _listener_worker(
    cfg,
    from_stage: int,
    to_stage: int,
    base_cmd: str,
    stop_event: Event,
    last_activity_ref: dict,
    last_activity_lock: Lock,
    stage_stats: dict[int, dict[str, float]],
    stage_stats_lock: Lock,
    scheduler: SingleGpuMemoryScheduler,
    controller_start_time: float,
    worker_errors: list[BaseException],
    worker_error_lock: Lock,
    active_ref: dict[str, int],
    active_lock: Lock,
    completed_artifacts: dict[int, set[int]],
    completed_artifacts_lock: Lock,
):
    baseline_mtimes = discover_ready_artifact_mtimes(cfg, from_stage)
    seen: set[int] = set()
    launched: set[int] = set()
    poll_interval = float(cfg.stream_subseq.poll_interval_s)

    while not stop_event.is_set():
        ready = discover_ready_artifact_mtimes(cfg, from_stage)
        for idx, mtime in ready.items():
            with completed_artifacts_lock:
                upstream_completed_idx = idx in completed_artifacts[from_stage]
            artifact_updated = mtime > baseline_mtimes.get(idx, controller_start_time)
            if not artifact_updated and not upstream_completed_idx:
                continue
            if idx not in seen:
                seen.add(idx)
                with last_activity_lock:
                    last_activity_ref["t"] = time.time()

        pending = sorted(idx for idx in seen if idx not in launched)
        for idx in pending:
            overrides = per_index_override(to_stage, idx)
            try:
                with active_lock:
                    active_ref["count"] += 1
                _run_cmd(
                    base_cmd,
                    overrides,
                    stage_id=to_stage,
                    stage_stats=stage_stats,
                    stage_stats_lock=stage_stats_lock,
                    scheduler=scheduler,
                )
                with completed_artifacts_lock:
                    completed_artifacts[to_stage].add(idx)
            except BaseException as exc:
                logger.exception("Streaming worker for stage %s failed on index %s", to_stage, idx)
                with worker_error_lock:
                    worker_errors.append(exc)
                stop_event.set()
                return
            finally:
                with active_lock:
                    active_ref["count"] = max(0, active_ref["count"] - 1)
            launched.add(idx)
            with last_activity_lock:
                last_activity_ref["t"] = time.time()

        time.sleep(poll_interval)


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    stages = validate_subsequence(int(cfg.stream_subseq.start_stage), int(cfg.stream_subseq.end_stage))
    logger.info("Streaming subsequence: %s", stages)
    wall_start = time.perf_counter()
    controller_start_time = time.time()

    # stage command map: allow per-stage command overrides from config
    cmd_by_stage = {
        stage: cfg.stream_subseq.get(f"s{stage}_cmd", _build_default_cmd(stage))
        for stage in stages
    }

    stop_event = Event()
    last_activity_ref = {"t": time.time()}
    last_activity_lock = Lock()
    stage_stats: dict[int, dict[str, float]] = defaultdict(lambda: {"elapsed_s": 0.0, "resource_wait_s": 0.0, "calls": 0})
    stage_stats_lock = Lock()
    worker_errors: list[BaseException] = []
    worker_error_lock = Lock()
    active_ref = {"count": 0}
    active_lock = Lock()
    completed_artifacts: dict[int, set[int]] = defaultdict(set)
    completed_artifacts_lock = Lock()
    timing_log_path = os.getenv(TIMING_LOG_ENV_VAR)
    timing_log = Path(timing_log_path) if timing_log_path else None
    # The budget is a fraction of total GPU memory by default, so one config works across
    # card sizes. An explicit max_vram_gb (e.g. from --max-vram-gb) still wins if set.
    explicit_max_vram_gb = cfg.stream_subseq.get("max_vram_gb", None)
    scheduler = SingleGpuMemoryScheduler(
        max_vram_gb=None if explicit_max_vram_gb is None else float(explicit_max_vram_gb),
        max_vram_frac=float(cfg.stream_subseq.get("max_vram_frac", 0.9)),
        stage_vram_gb=_stage_vram_config(cfg.stream_subseq.get("stage_vram_gb", {})),
        gpu_index=int(cfg.stream_subseq.get("gpu_index", 0)),
        hard_cap=bool(cfg.stream_subseq.get("hard_vram_cap", True)),
        poll_interval_s=float(cfg.stream_subseq.get("memory_poll_interval_s", cfg.stream_subseq.poll_interval_s)),
        wait_log_interval_s=float(cfg.stream_subseq.get("vram_wait_log_interval_s", 60.0)),
        wait_timeout_s=cfg.stream_subseq.get("vram_wait_timeout_s", None),
    )

    workers: list[Thread] = []
    for i in range(1, len(stages)):
        from_stage = stages[i - 1]
        to_stage = stages[i]
        t = Thread(
            target=_listener_worker,
            kwargs={
                "cfg": cfg,
                "from_stage": from_stage,
                "to_stage": to_stage,
                "base_cmd": cmd_by_stage[to_stage],
                "stop_event": stop_event,
                "last_activity_ref": last_activity_ref,
                "last_activity_lock": last_activity_lock,
                "stage_stats": stage_stats,
                "stage_stats_lock": stage_stats_lock,
                "scheduler": scheduler,
                "controller_start_time": controller_start_time,
                "worker_errors": worker_errors,
                "worker_error_lock": worker_error_lock,
                "active_ref": active_ref,
                "active_lock": active_lock,
                "completed_artifacts": completed_artifacts,
                "completed_artifacts_lock": completed_artifacts_lock,
            },
            daemon=True,
        )
        t.start()
        workers.append(t)

    # Run first stage as producer
    producer_error: BaseException | None = None
    expected_indices: set[int] = set()
    try:
        _run_cmd(
            cmd_by_stage[stages[0]],
            [],
            stage_id=stages[0],
            stage_stats=stage_stats,
            stage_stats_lock=stage_stats_lock,
            scheduler=scheduler,
        )
        with completed_artifacts_lock:
            # The producer has exited, so everything it wrote is complete -- skip the settle
            # window here, or an artifact finished in its last seconds would be left out of
            # expected_indices and the subsequence would call itself done without it.
            expected_indices = set(discover_ready_artifact_mtimes(cfg, stages[0], settle_s=0))
            completed_artifacts[stages[0]].update(expected_indices)
        with last_activity_lock:
            last_activity_ref["t"] = time.time()
    except BaseException as exc:
        producer_error = exc
        stop_event.set()

    # Wait until no new activity before stopping listeners.
    idle_timeout = float(cfg.stream_subseq.idle_timeout_s)
    while True:
        if stop_event.is_set():
            break
        with active_lock:
            active_count = active_ref["count"]
        if active_count > 0:
            time.sleep(float(cfg.stream_subseq.poll_interval_s))
            continue
        with completed_artifacts_lock:
            if subsequence_complete(stages, expected_indices, completed_artifacts):
                break
        with last_activity_lock:
            idle_for = time.time() - last_activity_ref["t"]
        if idle_for >= idle_timeout:
            break
        time.sleep(float(cfg.stream_subseq.poll_interval_s))

    stop_event.set()
    for t in workers:
        t.join(timeout=2.0)

    with worker_error_lock:
        first_worker_error = worker_errors[0] if worker_errors else None

    missing_stages: list[int] = []
    logger.info("Streaming timing summary:")
    with stage_stats_lock:
        missing_stages = _missing_downstream_stages(stages, stage_stats)
        for stage in stages:
            stats = stage_stats[stage]
            logger.info(
                "  Stage %s: %s across %d call(s), %.2fs waiting on resource budget",
                stage,
                _format_duration(stats["elapsed_s"]),
                int(stats["calls"]),
                stats["resource_wait_s"],
            )
            _append_timing_log(
                timing_log,
                stage_id=str(stage),
                description=f"Streaming stage {stage} cumulative runtime across {int(stats['calls'])} call(s)",
                elapsed_s=stats["elapsed_s"],
            )
    wall_elapsed = time.perf_counter() - wall_start
    logger.info("  Wall time: %s", _format_duration(wall_elapsed))
    _append_timing_log(
        timing_log,
        stage_id=f"{stages[0]}{stages[-1]}_WALL",
        description=f"Streaming stages {stages[0]}->{stages[-1]} wall time",
        elapsed_s=wall_elapsed,
    )
    analytics_path = _stream_analytics_path(cfg)
    analytics_path.parent.mkdir(parents=True, exist_ok=True)
    with analytics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "stages": {str(stage): dict(stage_stats[stage]) for stage in stages},
                "wall_time_s": wall_elapsed,
                "scheduler": scheduler.snapshot(),
                "events": scheduler.event_log(),
                "failure": {
                    "producer_error": repr(producer_error) if producer_error is not None else None,
                    "worker_error": repr(first_worker_error) if first_worker_error is not None else None,
                },
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    logger.info("Streaming resource report: %s", analytics_path)
    if producer_error is not None:
        raise producer_error
    if first_worker_error is not None:
        raise RuntimeError("Streaming subsequence failed in a downstream worker") from first_worker_error
    if missing_stages:
        raise RuntimeError(
            "Streaming subsequence ended before downstream stages ran: "
            f"{missing_stages}. Check watcher input paths and generated artifacts."
        )
    logger.info("Streaming subsequence complete: %s", stages)


if __name__ == "__main__":
    main()
