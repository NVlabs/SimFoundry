# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from simfoundry.pipeline.depth_backends import DepthAnythingV3Backend, FoundationStereoBackend, create_backend
from simfoundry.pipeline.stage_utils import list_object_iteration_indices, parse_iter_index


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
