#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


FAISS_GPU_VERSION="${FAISS_GPU_VERSION:-1.12}"
FAISS_GPU_TEMP_MEMORY="${FAISS_GPU_TEMP_MEMORY:-67108864}"

install_faiss_gpu() {
  local env_name="${1:?env name is required}"
  echo "Installing FAISS GPU ${FAISS_GPU_VERSION} in ${env_name}"
  mamba install -n "${env_name}" -c pytorch "faiss-gpu=${FAISS_GPU_VERSION}" -y
  validate_faiss_gpu "${env_name}"
}

validate_faiss_gpu() {
  local env_name="${1:?env name is required}"
  mamba run -n "${env_name}" python - <<PY
import sys

import numpy as np

try:
    import faiss
except Exception as exc:
    raise SystemExit(f"FAISS import failed in ${env_name}: {exc}") from exc

required = ("StandardGpuResources", "GpuIndexFlatL2", "get_num_gpus")
missing = [name for name in required if not hasattr(faiss, name)]
if missing:
    raise SystemExit(
        f"FAISS in ${env_name} is not GPU-capable; missing Python symbols: {missing}. "
        f"Loaded {getattr(faiss, '__file__', '<unknown>')}"
    )

num_gpus = faiss.get_num_gpus()
if num_gpus < 1:
    raise SystemExit(
        f"FAISS GPU is installed in ${env_name}, but FAISS sees {num_gpus} GPUs. "
        "Check NVIDIA driver visibility and CUDA runtime compatibility."
    )

resources = faiss.StandardGpuResources()
if hasattr(resources, "setTempMemory"):
    resources.setTempMemory(int("${FAISS_GPU_TEMP_MEMORY}"))
index = faiss.GpuIndexFlatL2(resources, 8)
xb = np.random.default_rng(0).random((32, 8), dtype=np.float32)
xq = np.random.default_rng(1).random((4, 8), dtype=np.float32)
index.add(xb)
distances, indices = index.search(xq, 3)
if distances.shape != (4, 3) or indices.shape != (4, 3):
    raise SystemExit(f"FAISS GPU smoke search returned unexpected shapes in ${env_name}.")

print(
    f"Validated FAISS GPU in ${env_name}: "
    f"version={getattr(faiss, '__version__', '<unknown>')}, "
    f"gpus={num_gpus}, file={getattr(faiss, '__file__', '<unknown>')}"
)
PY
}
