from threading import Thread
import time

from digital_cousins.pipeline.resource_scheduler import SingleGpuMemoryScheduler


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
