# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from threading import Thread
import time

import pytest

from simfoundry.pipeline.resource_scheduler import SingleGpuMemoryScheduler


def test_scheduler_blocks_until_reserved_memory_is_released():
    scheduler = SingleGpuMemoryScheduler(
        max_vram_gb=10,
        stage_vram_gb={7: 8, 8: 4},
        poll_interval_s=0.01,
        sample_fn=lambda _gpu: 0.0,
    )
    first = scheduler.acquire(7)
    acquired = []

    def acquire_second():
        acquired.append(scheduler.acquire(8))

    thread = Thread(target=acquire_second)
    thread.start()
    time.sleep(0.05)
    assert acquired == []

    scheduler.release(first)
    thread.join(timeout=1)
    assert len(acquired) == 1
    assert acquired[0].stage_id == 8
    scheduler.release(acquired[0])


def test_scheduler_fails_fast_when_stage_estimate_exceeds_cap():
    scheduler = SingleGpuMemoryScheduler(
        max_vram_gb=10,
        stage_vram_gb={7: 12},
        poll_interval_s=0.01,
        sample_fn=lambda _gpu: 0.0,
    )
    try:
        scheduler.acquire(7)
    except RuntimeError as exc:
        assert "exceeds the configured cap" in str(exc)
    else:
        raise AssertionError("Expected stage estimate over cap to fail fast")


def test_budget_is_sized_as_a_fraction_of_total_gpu_memory():
    scheduler = SingleGpuMemoryScheduler(
        max_vram_frac=0.9,
        stage_vram_gb={5: 18},
        sample_fn=lambda _gpu: 0.0,
        total_fn=lambda _gpu: 95.0,
    )
    assert scheduler.max_vram_gb == pytest.approx(85.5)
    assert scheduler.total_vram_gb == pytest.approx(95.0)


def test_unrelated_gpu_usage_does_not_deadlock_a_large_card():
    """Regression: a 30 GiB absolute cap made 14 GiB of foreign usage block stage 5 forever
    on a 95 GiB card. A fractional budget must admit it immediately."""
    scheduler = SingleGpuMemoryScheduler(
        max_vram_frac=0.9,
        stage_vram_gb={5: 18},
        sample_fn=lambda _gpu: 14.0,
        total_fn=lambda _gpu: 95.0,
        poll_interval_s=0.01,
        wait_timeout_s=0.5,
    )
    reservation = scheduler.acquire(5)
    assert reservation.wait_s < 0.5
    scheduler.release(reservation)


def test_wait_times_out_and_explains_external_gpu_usage():
    scheduler = SingleGpuMemoryScheduler(
        max_vram_frac=0.9,
        stage_vram_gb={5: 18},
        sample_fn=lambda _gpu: 80.0,
        total_fn=lambda _gpu: 95.0,
        poll_interval_s=0.01,
        wait_timeout_s=0.2,
    )
    with pytest.raises(TimeoutError) as exc:
        scheduler.acquire(5)
    assert "already reports 80.0 GiB in use" in str(exc.value)
    assert "hard_vram_cap" in str(exc.value)


def test_wait_reason_names_the_blocking_pipeline_stage():
    scheduler = SingleGpuMemoryScheduler(
        max_vram_frac=0.9,
        stage_vram_gb={5: 60, 7: 40},
        sample_fn=lambda _gpu: 0.0,
        total_fn=lambda _gpu: 95.0,
        poll_interval_s=0.01,
        wait_timeout_s=0.2,
    )
    held = scheduler.acquire(5)
    with pytest.raises(TimeoutError) as exc:
        scheduler.acquire(7)
    assert "other pipeline stages hold" in str(exc.value)
    assert "stage 5" in str(exc.value)
    scheduler.release(held)


def test_falls_back_to_a_default_total_when_nvidia_smi_is_unavailable():
    scheduler = SingleGpuMemoryScheduler(
        max_vram_frac=0.5,
        stage_vram_gb={5: 1},
        sample_fn=lambda _gpu: None,
        total_fn=lambda _gpu: None,
    )
    assert scheduler.total_vram_gb is None
    assert scheduler.max_vram_gb > 0  # usable budget, not a crash


def test_event_log_stays_bounded_while_blocked():
    from simfoundry.pipeline.resource_scheduler import MAX_EVENT_LOG_ENTRIES

    scheduler = SingleGpuMemoryScheduler(
        max_vram_frac=0.9,
        stage_vram_gb={5: 90},
        sample_fn=lambda _gpu: 0.0,
        total_fn=lambda _gpu: 95.0,
        poll_interval_s=0.0001,
        wait_timeout_s=0.3,
    )
    scheduler.stage_vram_gb[5] = 85.0
    scheduler._reserved_gb = 85.0  # force the ledger branch to block
    with pytest.raises(TimeoutError):
        scheduler.acquire(5)
    assert len(scheduler.event_log()) <= MAX_EVENT_LOG_ENTRIES
