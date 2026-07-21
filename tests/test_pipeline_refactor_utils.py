import time

from digital_cousins.pipeline.depth_backends import DepthAnythingV3Backend, FoundationStereoBackend, create_backend
from digital_cousins.pipeline.stage_utils import list_object_iteration_indices, parse_iter_index
from digital_cousins.pipeline.streaming import run_streaming_two_stage


def test_depth_backend_registry():
    assert isinstance(create_backend("fs"), FoundationStereoBackend)
    assert isinstance(create_backend("da3"), DepthAnythingV3Backend)


def test_parse_iter_index_and_listing():
    assert parse_iter_index("iter_12") == 12
    assert parse_iter_index("foo") is None

    filenames = [
        "iter_3_transparent.png",
        "iter_1_transparent.png",
        "iter_3_transparent.png",
        "not_an_iter.png",
    ]
    assert list_object_iteration_indices(filenames, suffix="_transparent.png") == [1, 3]


def test_streaming_two_stage_beats_sequential():
    items = list(range(6))

    def stage1(_):
        time.sleep(0.04)

    def stage2(_):
        time.sleep(0.04)

    # Sequential baseline
    t0 = time.perf_counter()
    for i in items:
        stage1(i)
        stage2(i)
    sequential = time.perf_counter() - t0

    stats = run_streaming_two_stage(items, stage1, stage2, stage2_workers=3)

    # Pipelined runtime should be materially lower than strict sequential.
    assert stats.total_runtime_s < sequential * 0.9, (stats.total_runtime_s, sequential)
