# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import faiss
import numpy as np


logger = logging.getLogger(__name__)
_CPU_FALLBACK_WARNED = False
_FALSE_VALUES = {"0", "false", "no", "off"}


class FaissGpuUnavailableError(RuntimeError):
    """Raised when ENFORCE_FAISS_GPU is enabled but FAISS GPU cannot be used."""


def enforce_faiss_gpu():
    return os.environ.get("ENFORCE_FAISS_GPU", "1").strip().lower() not in _FALSE_VALUES


def _as_faiss_array(values):
    return np.ascontiguousarray(values, dtype=np.float32)


def _require_gpu_faiss():
    required = ("StandardGpuResources", "GpuIndexFlatL2", "get_num_gpus")
    missing = [name for name in required if not hasattr(faiss, name)]
    if missing:
        raise FaissGpuUnavailableError(
            "ENFORCE_FAISS_GPU is enabled, but this FAISS build is missing GPU symbols: "
            f"{missing}. Loaded FAISS from {getattr(faiss, '__file__', '<unknown>')}."
        )

    num_gpus = faiss.get_num_gpus()
    if num_gpus < 1:
        raise FaissGpuUnavailableError(
            "ENFORCE_FAISS_GPU is enabled, but FAISS reports no visible GPUs. "
            "Check NVIDIA driver visibility, CUDA runtime compatibility, and CUDA_VISIBLE_DEVICES."
        )


def _gpu_l2_search(xb, xq, k):
    _require_gpu_faiss()
    try:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, xb.shape[1])
        index.add(xb)
        return index.search(xq, k)
    except FaissGpuUnavailableError:
        raise
    except Exception as exc:
        raise FaissGpuUnavailableError(
            "ENFORCE_FAISS_GPU is enabled and FAISS GPU search failed. "
            "CPU fallback is disabled."
        ) from exc


def _cpu_l2_search(xb, xq, k):
    index = faiss.IndexFlatL2(xb.shape[1])
    index.add(xb)
    return index.search(xq, k)


def l2_search(index_vectors, query_vectors, k):
    """Run L2 nearest-neighbor search.

    FAISS GPU is enforced by default. Set ENFORCE_FAISS_GPU=0 to allow CPU fallback.
    """
    global _CPU_FALLBACK_WARNED

    xb = _as_faiss_array(index_vectors)
    xq = _as_faiss_array(query_vectors)
    if xb.ndim != 2 or xq.ndim != 2:
        raise ValueError(f"FAISS inputs must be 2-D arrays, got {xb.shape=} and {xq.shape=}")
    if xb.shape[1] != xq.shape[1]:
        raise ValueError(f"FAISS input dimensions differ, got {xb.shape[1]} and {xq.shape[1]}")
    if xb.shape[0] == 0:
        raise ValueError("Cannot search an empty FAISS index")

    if enforce_faiss_gpu():
        return _gpu_l2_search(xb, xq, k)

    if (
        hasattr(faiss, "StandardGpuResources")
        and hasattr(faiss, "GpuIndexFlatL2")
        and getattr(faiss, "get_num_gpus", lambda: 0)() > 0
    ):
        try:
            return _gpu_l2_search(xb, xq, k)
        except Exception as exc:
            if not _CPU_FALLBACK_WARNED:
                logger.warning("Falling back to CPU FAISS search after GPU FAISS failed: %s", exc)
                _CPU_FALLBACK_WARNED = True

    if not _CPU_FALLBACK_WARNED:
        logger.warning("Using CPU FAISS search because GPU FAISS resources are unavailable")
        _CPU_FALLBACK_WARNED = True
    return _cpu_l2_search(xb, xq, k)
