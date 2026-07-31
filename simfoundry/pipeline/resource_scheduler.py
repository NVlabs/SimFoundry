# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resource gating helpers for local pipeline parallelism."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import subprocess
import threading
import time
from typing import Callable, Iterator


logger = logging.getLogger(__name__)

GpuMemorySampleFn = Callable[[int], float | None]

# Keep the in-memory event log bounded; a stage blocked for hours would otherwise
# append a "wait" record every poll interval forever.
MAX_EVENT_LOG_ENTRIES = 2000


def _query_gpu_memory_gb(gpu_index: int, field: str) -> float | None:
    cmd = [
        "nvidia-smi",
        f"--id={gpu_index}",
        f"--query-gpu={field}",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, TypeError):
        return None

    try:
        value = proc.stdout.strip().splitlines()[0].strip()
        return float(value) / 1024.0
    except (TypeError, ValueError, IndexError):
        return None


def query_gpu_memory_used_gb(gpu_index: int = 0) -> float | None:
    """Return currently used GPU memory in GiB, or None when unavailable."""
    return _query_gpu_memory_gb(gpu_index, "memory.used")


def query_gpu_memory_total_gb(gpu_index: int = 0) -> float | None:
    """Return total GPU memory in GiB, or None when unavailable."""
    return _query_gpu_memory_gb(gpu_index, "memory.total")


@dataclass(frozen=True)
class ResourceReservation:
    stage_id: int
    estimated_vram_gb: float
    wait_s: float
    measured_used_gb: float | None


DEFAULT_MAX_VRAM_FRAC = 0.9
# Used only when the total cannot be read (no nvidia-smi); keeps the old behaviour.
FALLBACK_TOTAL_VRAM_GB = 32.0


class SingleGpuMemoryScheduler:
    """Hard-cap scheduler for subprocesses sharing one GPU.

    The budget is expressed as a *fraction of total GPU memory* (`max_vram_frac`),
    resolved once at construction from `nvidia-smi`. Sizing it relative to the card
    means the same config works on a 24 GiB and a 96 GiB GPU, and an unrelated GPU
    consumer no longer deadlocks the pipeline on a large card.

    `max_vram_gb` remains supported as an explicit absolute override.

    The scheduler combines a conservative per-stage reservation ledger with the
    live GPU memory reading when `nvidia-smi` is available. If live sampling is
    unavailable, the reservation ledger still prevents known stage combinations
    from exceeding the configured budget.
    """

    def __init__(
        self,
        *,
        max_vram_gb: float | None = None,
        max_vram_frac: float | None = None,
        stage_vram_gb: dict[int, float],
        gpu_index: int = 0,
        hard_cap: bool = True,
        poll_interval_s: float = 3.0,
        wait_log_interval_s: float = 60.0,
        wait_timeout_s: float | None = None,
        sample_fn: GpuMemorySampleFn = query_gpu_memory_used_gb,
        total_fn: GpuMemorySampleFn = query_gpu_memory_total_gb,
    ) -> None:
        self.gpu_index = int(gpu_index)
        self.total_vram_gb = total_fn(self.gpu_index)

        if max_vram_gb is not None:
            if float(max_vram_gb) <= 0:
                raise ValueError("max_vram_gb must be positive")
            self.max_vram_gb = float(max_vram_gb)
            self.max_vram_frac = None
            budget_source = f"explicit max_vram_gb={self.max_vram_gb:.2f}"
        else:
            frac = DEFAULT_MAX_VRAM_FRAC if max_vram_frac is None else float(max_vram_frac)
            if not 0.0 < frac <= 1.0:
                raise ValueError("max_vram_frac must be in (0, 1]")
            self.max_vram_frac = frac
            total = self.total_vram_gb
            if total is None or total <= 0:
                total = FALLBACK_TOTAL_VRAM_GB
                logger.warning(
                    "Could not read total GPU memory (is nvidia-smi available?); "
                    "assuming %.1f GiB for the %.0f%% budget.",
                    total,
                    frac * 100,
                )
            self.max_vram_gb = total * frac
            budget_source = f"{frac * 100:.0f}% of {total:.1f} GiB total"

        self.stage_vram_gb = {int(k): float(v) for k, v in stage_vram_gb.items()}
        self.hard_cap = bool(hard_cap)
        self.poll_interval_s = max(float(poll_interval_s), 0.1)
        self.wait_log_interval_s = max(float(wait_log_interval_s), 1.0)
        self.wait_timeout_s = None if wait_timeout_s in (None, 0) else float(wait_timeout_s)
        self._sample_fn = sample_fn
        self._condition = threading.Condition()
        self._reserved_gb = 0.0
        self._active: dict[int, float] = {}
        self._events: list[dict[str, object]] = []

        logger.info(
            "GPU memory budget: %.2f GiB (%s), hard_cap=%s, gpu_index=%d",
            self.max_vram_gb,
            budget_source,
            self.hard_cap,
            self.gpu_index,
        )

    def estimate_for(self, stage_id: int) -> float:
        return float(self.stage_vram_gb.get(int(stage_id), 0.0))

    def memory_used_gb(self) -> float | None:
        return self._sample_fn(self.gpu_index)

    def _blocking_reason(self, estimate: float, measured_used_gb: float | None) -> str | None:
        """Return a human-readable reason the reservation cannot proceed, else None."""
        if self._reserved_gb + estimate > self.max_vram_gb:
            held = ", ".join(f"stage {s}: {gb:.1f} GiB" for s, gb in sorted(self._active.items()))
            return (
                f"other pipeline stages hold {self._reserved_gb:.1f} GiB of the "
                f"{self.max_vram_gb:.1f} GiB budget and this stage needs {estimate:.1f} GiB"
                + (f" (held by {held})" if held else "")
            )
        if self.hard_cap and measured_used_gb is not None and measured_used_gb + estimate > self.max_vram_gb:
            return (
                f"the GPU already reports {measured_used_gb:.1f} GiB in use and this stage "
                f"needs {estimate:.1f} GiB, which exceeds the {self.max_vram_gb:.1f} GiB budget. "
                "If another process (or a leftover run) is holding memory, free it, raise "
                "stream_subseq.max_vram_frac, or set stream_subseq.hard_vram_cap=false"
            )
        return None

    def _can_reserve(self, estimate: float, measured_used_gb: float | None) -> bool:
        return self._blocking_reason(estimate, measured_used_gb) is None

    def _record_event(
        self,
        event: str,
        *,
        stage_id: int,
        estimate: float,
        measured_used_gb: float | None,
        wait_s: float = 0.0,
    ) -> None:
        self._events.append(
            {
                "event": event,
                "stage_id": stage_id,
                "timestamp_s": time.time(),
                "estimated_vram_gb": estimate,
                "reserved_vram_gb": self._reserved_gb,
                "measured_used_gb": measured_used_gb,
                "wait_s": wait_s,
            }
        )
        if len(self._events) > MAX_EVENT_LOG_ENTRIES:
            # Drop the oldest waits; acquire/release records at the tail are what matter.
            del self._events[: len(self._events) - MAX_EVENT_LOG_ENTRIES]

    def acquire(self, stage_id: int) -> ResourceReservation:
        stage_id = int(stage_id)
        estimate = self.estimate_for(stage_id)
        if estimate > self.max_vram_gb:
            raise RuntimeError(
                f"Stage {stage_id} estimated VRAM {estimate:.2f} GiB exceeds "
                f"the configured cap {self.max_vram_gb:.2f} GiB"
            )

        start = time.perf_counter()
        last_log = start
        announced = False
        while True:
            with self._condition:
                measured = self.memory_used_gb()
                reason = self._blocking_reason(estimate, measured)
                if reason is None:
                    if announced:
                        logger.info(
                            "Stage %s acquired its GPU budget after %.0fs.",
                            stage_id,
                            time.perf_counter() - start,
                        )
                    self._reserved_gb += estimate
                    self._active[stage_id] = self._active.get(stage_id, 0.0) + estimate
                    wait_s = time.perf_counter() - start
                    self._record_event(
                        "acquire",
                        stage_id=stage_id,
                        estimate=estimate,
                        measured_used_gb=measured,
                        wait_s=wait_s,
                    )
                    return ResourceReservation(
                        stage_id=stage_id,
                        estimated_vram_gb=estimate,
                        wait_s=wait_s,
                        measured_used_gb=measured,
                    )
                waited = time.perf_counter() - start
                self._record_event(
                    "wait",
                    stage_id=stage_id,
                    estimate=estimate,
                    measured_used_gb=measured,
                    wait_s=waited,
                )
                # Say something the first time we block, then at a steady interval, so an
                # indefinite wait is never silent.
                now = time.perf_counter()
                if not announced or (now - last_log) >= self.wait_log_interval_s:
                    logger.info(
                        "Stage %s is PAUSED waiting for GPU memory (%.0fs so far): %s.",
                        stage_id,
                        waited,
                        reason,
                    )
                    last_log = now
                    announced = True

                if self.wait_timeout_s is not None and waited >= self.wait_timeout_s:
                    raise TimeoutError(
                        f"Stage {stage_id} waited {waited:.1f}s for GPU memory and gave up "
                        f"(stream_subseq.vram_wait_timeout_s={self.wait_timeout_s:g}). "
                        f"Reason: {reason}."
                    )

                self._condition.wait(timeout=self.poll_interval_s)

    def release(self, reservation: ResourceReservation) -> None:
        with self._condition:
            active_value = self._active.get(reservation.stage_id, 0.0) - reservation.estimated_vram_gb
            if active_value <= 1e-9:
                self._active.pop(reservation.stage_id, None)
            else:
                self._active[reservation.stage_id] = active_value
            self._reserved_gb = max(0.0, self._reserved_gb - reservation.estimated_vram_gb)
            self._record_event(
                "release",
                stage_id=reservation.stage_id,
                estimate=reservation.estimated_vram_gb,
                measured_used_gb=self.memory_used_gb(),
            )
            self._condition.notify_all()

    @contextmanager
    def reserve(self, stage_id: int) -> Iterator[ResourceReservation]:
        reservation = self.acquire(stage_id)
        try:
            yield reservation
        finally:
            self.release(reservation)

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "max_vram_gb": self.max_vram_gb,
                "max_vram_frac": self.max_vram_frac,
                "total_vram_gb": self.total_vram_gb,
                "gpu_index": self.gpu_index,
                "hard_cap": self.hard_cap,
                "reserved_vram_gb": self._reserved_gb,
                "active_reservations": dict(self._active),
                "measured_used_gb": self.memory_used_gb(),
            }

    def event_log(self) -> list[dict[str, object]]:
        with self._condition:
            return list(self._events)
