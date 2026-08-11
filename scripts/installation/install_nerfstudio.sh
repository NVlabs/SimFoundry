#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Install the `nerfstudio_simfoundry` env used by the auto-BG pipeline's BG splat
# training/export (step 5: ns-train splatfacto-big + ns-export).
#
# The only previous recipe (auto_bg_reconstruction/README.md) pinned torch 2.1.2+cu121
# and gsplat 1.4.0 — which have NO sm_120 kernels and fail on the RTX 5090 with
# "CUDA error: no kernel image is available".
#
# nerfstudio 1.1.5 requires only torch>=1.13.1 / torchvision>=0.14.1 (floors), but
# HARD-pins gsplat==1.4.0. So we install the right torch ONCE (2.7.1+cu128) up front,
# let nerfstudio pull its gsplat 1.4.0, then override gsplat -> 1.5.3 (the only package
# that actually needs swapping). Final stack:
#   - torch 2.7.1+cu128, gsplat 1.5.3 (JIT-builds its CUDA extension)
#   - conda cuda-toolkit 12.8 (provides nvcc + headers for the gsplat JIT build)
#
# NOTE: the gsplat JIT build needs CUDA headers, which conda places under
# targets/x86_64-linux/include/. 5_train_bg_splat.py's _ns_env() sets CPLUS_INCLUDE_PATH
# to that dir automatically at runtime, so no extra step is required here.

set -euo pipefail

if ! command -v mamba >/dev/null 2>&1; then
  echo "Error: mamba was not found on PATH. Install Miniforge first." >&2
  exit 127
fi

eval "$(mamba shell hook --shell bash)"

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
project_root="$(cd "$SCRIPT_DIR/../.." && pwd)"
env_name="nerfstudio_simfoundry"
DEFAULT=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --project-root) project_root="$2"; shift 2 ;;
    --env-name)     env_name="$2"; shift 2 ;;
    --default)      DEFAULT=true; shift ;;
    -h|--help)      echo "Usage: $0 [--project-root DIR] [--env-name NAME] [--default]"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ ! ${DEFAULT} == true ]]; then
  read -p "Enter environment name (default: ${env_name}): " ENV_NAME
  env_name="${ENV_NAME:-$env_name}"
fi

echo "=== nerfstudio_simfoundry Environment Setup ==="
echo "  env_name: ${env_name}"

# ------------------------------------------------------------------------------
# Step 1: env + the sm_120 torch stack UP FRONT.
# nerfstudio 1.1.5 requires only torch>=1.13.1 / torchvision>=0.14.1 (floors, not
# ceilings — verified on PyPI), so installing torch 2.7.1+cu128 first means the
# `pip install nerfstudio` below leaves it untouched. No need to install an old
# cu121 torch and force-reinstall it later.
# ------------------------------------------------------------------------------
mamba create -y -n "${env_name}" python=3.10
mamba run -n "${env_name}" pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128

# ------------------------------------------------------------------------------
# Step 2: nerfstudio 1.1.5. It keeps the torch above (>=1.13.1) and pulls its
# HARD-pinned gsplat==1.4.0 (pure-python). gsplat is the one package we must
# install-then-override, since 1.4.0 has no sm_120 kernels — done in Step 3.
# ------------------------------------------------------------------------------
mamba run -n "${env_name}" pip install nerfstudio==1.1.5 "hydra-core>=1.3,<1.4"

# ------------------------------------------------------------------------------
# Step 1b: apply the env-var-gated depth-loss patch to the installed nerfstudio.
# auto_bg.yaml defaults to use_depth_loss=true; without this patch splatfacto.py
# ignores NERFSTUDIO_DEPTH_LOSS and training runs unsupervised. Documented in
# auto_bg_reconstruction/README.md §4.2. Non-fatal: warns (doesn't abort) on drift.
# ------------------------------------------------------------------------------
PATCH="$(cd "$SCRIPT_DIR/../.." && pwd)/patches/splatfacto_depth_loss.patch"
if [ -f "${PATCH}" ]; then
  SP_DIR="$(mamba run -n "${env_name}" python -c 'import nerfstudio, os; print(os.path.dirname(os.path.dirname(nerfstudio.__file__)))')"
  if patch -p1 -d "${SP_DIR}" --reverse --dry-run -f < "${PATCH}" >/dev/null 2>&1; then
    echo "splatfacto_depth_loss.patch already applied"
  elif patch -p1 -d "${SP_DIR}" --forward -f < "${PATCH}" >/dev/null 2>&1; then
    echo "Applied splatfacto_depth_loss.patch -> ${SP_DIR}/nerfstudio/models/splatfacto.py"
  else
    echo "WARNING: could not apply splatfacto_depth_loss.patch (nerfstudio version drift?);"
    echo "         depth-supervised training (use_depth_loss=true) will be disabled."
  fi
else
  echo "WARNING: ${PATCH} not found; depth-supervised training will be unavailable."
fi

# ------------------------------------------------------------------------------
# Step 3: override gsplat's hard ==1.4.0 pin with the sm_120-capable 1.5.3 (it
# JIT-builds its CUDA extension), and add the conda CUDA toolkit (nvcc + headers
# for that JIT build). pip prints a cosmetic "nerfstudio requires gsplat==1.4.0"
# warning — splatfacto runs fine on 1.5.3 (verified end-to-end).
# ------------------------------------------------------------------------------
mamba run -n "${env_name}" pip install gsplat==1.5.3
mamba install -y -n "${env_name}" -c nvidia cuda-toolkit=12.8

echo "Verifying nerfstudio_simfoundry env..."
mamba run -n "${env_name}" python -c "import hydra, torch, gsplat; print('ns env OK: torch', torch.__version__, '| gsplat', gsplat.__version__, '| cuda', torch.cuda.is_available())"

echo "Completed installation of nerfstudio_simfoundry environment: ${env_name}"
