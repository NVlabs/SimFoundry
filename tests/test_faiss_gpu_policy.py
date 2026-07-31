# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from simfoundry.utils import faiss_utils


class _FakeIndex:
    def __init__(self, owner, dim, kind):
        self._owner = owner
        self._dim = dim
        self._kind = kind
        self._vectors = None

    def add(self, vectors):
        self._vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, queries, k):
        if self._vectors is None:
            raise RuntimeError("index has no vectors")
        if self._kind == "gpu":
            self._owner.gpu_searches += 1
        else:
            self._owner.cpu_searches += 1
        queries = np.asarray(queries, dtype=np.float32)
        distances = ((queries[:, None, :] - self._vectors[None, :, :]) ** 2).sum(axis=2)
        indices = np.argsort(distances, axis=1)[:, :k]
        return np.take_along_axis(distances, indices, axis=1), indices


class _FakeGpuFaiss:
    __file__ = "/fake/faiss-gpu"

    def __init__(self):
        self.gpu_searches = 0
        self.cpu_searches = 0

    def get_num_gpus(self):
        return 1

    class StandardGpuResources:
        pass

    def GpuIndexFlatL2(self, resources, dim):
        return _FakeIndex(self, dim, "gpu")

    def IndexFlatL2(self, dim):
        return _FakeIndex(self, dim, "cpu")


class _FakeCpuFaiss:
    __file__ = "/fake/faiss-cpu"

    def __init__(self):
        self.cpu_searches = 0

    def get_num_gpus(self):
        return 0

    def IndexFlatL2(self, dim):
        return _FakeIndex(self, dim, "cpu")


@pytest.fixture(autouse=True)
def reset_faiss_policy(monkeypatch):
    monkeypatch.delenv("ENFORCE_FAISS_GPU", raising=False)
    faiss_utils._CPU_FALLBACK_WARNED = False


def test_l2_search_enforces_gpu_by_default(monkeypatch):
    fake_faiss = _FakeGpuFaiss()
    monkeypatch.setattr(faiss_utils, "faiss", fake_faiss)

    distances, indices = faiss_utils.l2_search(
        np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
        2,
    )

    assert fake_faiss.gpu_searches == 1
    assert fake_faiss.cpu_searches == 0
    assert indices.tolist() == [[0, 1]]
    assert distances.shape == (1, 2)


def test_l2_search_raises_when_enforced_and_gpu_missing(monkeypatch):
    monkeypatch.setattr(faiss_utils, "faiss", _FakeCpuFaiss())

    with pytest.raises(faiss_utils.FaissGpuUnavailableError, match="ENFORCE_FAISS_GPU"):
        faiss_utils.l2_search(
            np.array([[0.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 1.0]], dtype=np.float32),
            1,
        )


def test_l2_search_allows_cpu_fallback_when_explicitly_disabled(monkeypatch):
    fake_faiss = _FakeCpuFaiss()
    monkeypatch.setattr(faiss_utils, "faiss", fake_faiss)
    monkeypatch.setenv("ENFORCE_FAISS_GPU", "0")

    distances, indices = faiss_utils.l2_search(
        np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        np.array([[2.0, 0.0]], dtype=np.float32),
        1,
    )

    assert fake_faiss.cpu_searches == 1
    assert indices.tolist() == [[1]]
    assert distances.tolist() == [[1.0]]
