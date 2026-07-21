"""Resource gating helpers for local pipeline parallelism."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import subprocess
import threading
import time
from typing import Callable, Iterator


GpuMemorySampleFn = Callable[[int], float | None]


def query_gpu_memory_used_gb(gpu_index: int = 0) -> float | None:
    """Return currently used GPU memory in GiB, or None when unavailable."""
    cmd = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=memory.used",
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


@dataclass(frozen=True)
class ResourceReservation:
    stage_id: int
    estimated_vram_gb: float
    wait_s: float
    measured_used_gb: float | None


class SingleGpuMemoryScheduler:
    """Hard-cap scheduler for subprocesses sharing one GPU.

    The scheduler combines a conservative per-stage reservation ledger with the
    live GPU memory reading when `nvidia-smi` is available. If live sampling is
    unavailable, the reservation ledger still prevents known stage combinations
    from exceeding the configured budget.
    """

    def __init__(
        self,
        *,
        max_vram_gb: float,
        stage_vram_gb: dict[int, float],
        gpu_index: int = 0,
        hard_cap: bool = True,
        poll_interval_s: float = 3.0,
        sample_fn: GpuMemorySampleFn = query_gpu_memory_used_gb,
    ) -> None:
        if max_vram_gb <= 0:
            raise ValueError("max_vram_gb must be positive")
        self.max_vram_gb = float(max_vram_gb)
        self.stage_vram_gb = {int(k): float(v) for k, v in stage_vram_gb.items()}
        self.gpu_index = int(gpu_index)
        self.hard_cap = bool(hard_cap)
        self.poll_interval_s = max(float(poll_interval_s), 0.1)
        self._sample_fn = sample_fn
        self._condition = threading.Condition()
        self._reserved_gb = 0.0
        self._active: dict[int, float] = {}
        self._events: list[dict[str, object]] = []

    def estimate_for(self, stage_id: int) -> float:
        return float(self.stage_vram_gb.get(int(stage_id), 0.0))

    def memory_used_gb(self) -> float | None:
        return self._sample_fn(self.gpu_index)

    def _can_reserve(self, estimate: float, measured_used_gb: float | None) -> bool:
        if self._reserved_gb + estimate > self.max_vram_gb:
            return False
        if self.hard_cap and measured_used_gb is not None and measured_used_gb + estimate > self.max_vram_gb:
            return False
        return True

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

    def acquire(self, stage_id: int) -> ResourceReservation:
        stage_id = int(stage_id)
        estimate = self.estimate_for(stage_id)
        if estimate > self.max_vram_gb:
            raise RuntimeError(
                f"Stage {stage_id} estimated VRAM {estimate:.2f} GiB exceeds "
                f"the configured cap {self.max_vram_gb:.2f} GiB"
            )

        start = time.perf_counter()
        while True:
            with self._condition:
                measured = self.memory_used_gb()
                if self._can_reserve(estimate, measured):
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
                self._record_event(
                    "wait",
                    stage_id=stage_id,
                    estimate=estimate,
                    measured_used_gb=measured,
                    wait_s=time.perf_counter() - start,
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
                "gpu_index": self.gpu_index,
                "hard_cap": self.hard_cap,
                "reserved_vram_gb": self._reserved_gb,
                "active_reservations": dict(self._active),
                "measured_used_gb": self.memory_used_gb(),
            }

    def event_log(self) -> list[dict[str, object]]:
        with self._condition:
            return list(self._events)
