#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Install the `3dgrut` env used by the auto-BG pipeline's step 7 (PLY -> NuRec USDZ).
#
# This wraps the cloned 3dgrut repo's OWN two-step installer:
#   1. scripts/create_conda.sh  — creates the conda env + CUDA toolkit + persisted build vars
#   2. install_env_uv.sh        — installs the project, Kaolin, ppisp, fused-ssim, slangc
#
# It supersedes the previous version, which called an upstream `install_env.sh` that has
# since been renamed `install_env_uv.sh`. Two patches are applied below:
# `patches/3dgrut.patch` (ply_to_usd.py export_cameras=False) and
# `patches/3dgrut_nounset.patch` (set +u in create_conda.sh, so conda's compiler
# deactivate hooks can't abort the build on unset CONDA_BACKUP_* variables).
# For CUDA 12.8 the upstream installer pins torch 2.8.0+cu128 with
# TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0+PTX" (sm_120 / RTX 5090 Blackwell).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
project_root="$(cd "$SCRIPT_DIR/../.." && pwd)"
env_name="3dgrut"
cuda_version="${CUDA_VERSION:-12.8}"
DEFAULT=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [--project-root DIR] [--env-name NAME] [--cuda-version VER] [--default]

Options:
  --project-root DIR   Repo root (default: auto-detected: ${project_root})
  --env-name NAME      Conda env name (default: ${env_name})
  --cuda-version VER   CUDA toolkit version for the env (default: ${cuda_version})
  --default            Non-interactive; accept all defaults
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --project-root) project_root="$2"; shift 2 ;;
    --env-name)     env_name="$2"; shift 2 ;;
    --cuda-version) cuda_version="$2"; shift 2 ;;
    --default)      DEFAULT=true; shift ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ ! ${DEFAULT} == true ]]; then
  read -p "Enter project root (default: $project_root): " IN_ROOT
  project_root="${IN_ROOT:-$project_root}"
  read -p "Enter environment name (default: ${env_name}): " IN_ENV
  env_name="${IN_ENV:-$env_name}"
fi

THREEDGRUT_DIR="${project_root}/deps/3dgrut"

echo "=== 3DGRUT Environment Setup ==="
echo "  project_root: ${project_root}"
echo "  env_name:     ${env_name}"
echo "  cuda_version: ${cuda_version}"
echo ""

# ------------------------------------------------------------------------------
# Step 0: prerequisites — a conda channel configured
#
# NOTE: `uv` is NOT required on PATH here. The 3dgrut env doesn't exist yet (it's
# created in Step 2), and the upstream installer's `uv pip install` targets the env's
# interpreter (UV_PYTHON / UV_PROJECT_ENVIRONMENT persisted by create_conda.sh). So we
# install uv INTO the env after it's created (Step 2.5) instead of depending on a
# global uv; `mamba run -n "${env_name}"` in Step 3 then picks up the env-local uv.
# ------------------------------------------------------------------------------
# miniforge installs here have sometimes had no channels configured, which makes the
# plain `conda create ... python=3.11` inside create_conda.sh fail with
# NoChannelsConfiguredError. Ensure conda-forge is present.
if ! conda config --show channels 2>/dev/null | grep -q "conda-forge"; then
  echo "Adding conda-forge channel (none was configured)..."
  conda config --add channels conda-forge
fi

# ------------------------------------------------------------------------------
# Step 1: clone (if needed) and init submodules (tiny-cuda-nn, optix-dev)
# ------------------------------------------------------------------------------
mkdir -p "${project_root}/deps"
if [ ! -d "${THREEDGRUT_DIR}" ]; then
  echo "Cloning 3dgrut..."
  git clone --recursive https://github.com/nv-tlabs/3dgrut.git "${THREEDGRUT_DIR}"
  # TODO(SimFoundry): confirm this SHA matches a tested build before release (pinned 2026-07-27).
  git -C "${THREEDGRUT_DIR}" checkout --detach "${THREEDGRUT_COMMIT:-a37ef721012dea0f29c0fcfff2d525023b4e854a}"
fi
( cd "${THREEDGRUT_DIR}" && git submodule update --init --recursive )

# Re-apply our ply_to_usd.py fix (export_cameras=False for dataset-less PLY export).
# A fresh clone would otherwise drop it and step 7's PLY->USDZ would fail with
# "ValueError: export_cameras=True requires a dataset".
PATCH="${project_root}/patches/3dgrut.patch"
if [ -f "${PATCH}" ]; then
  if git -C "${THREEDGRUT_DIR}" apply --check --reverse "${PATCH}" 2>/dev/null; then
    echo "patches/3dgrut.patch already applied"
  else
    git -C "${THREEDGRUT_DIR}" apply "${PATCH}"
    echo "Applied patches/3dgrut.patch (ply_to_usd export_cameras=False)"
  fi
else
  echo "WARNING: ${PATCH} not found; the ply_to_usd export_cameras fix will be missing."
fi

# Make create_conda.sh nounset-safe (set +u before its conda operations). The conda()
# wrapper reactivates the env after each `conda install`, sourcing deactivate.d hooks
# that can reference unset CONDA_BACKUP_* variables; under the script's `set -u` that
# aborts the build mid-way and leaves a half-built env.
NOUNSET_PATCH="${project_root}/patches/3dgrut_nounset.patch"
if [ -f "${NOUNSET_PATCH}" ]; then
  if git -C "${THREEDGRUT_DIR}" apply --check --reverse "${NOUNSET_PATCH}" 2>/dev/null; then
    echo "patches/3dgrut_nounset.patch already applied"
  else
    git -C "${THREEDGRUT_DIR}" apply "${NOUNSET_PATCH}"
    echo "Applied patches/3dgrut_nounset.patch (create_conda.sh set +u for conda hooks)"
  fi
else
  echo "WARNING: ${NOUNSET_PATCH} not found; create_conda.sh may abort on conda's compiler hooks under set -u."
fi

# ------------------------------------------------------------------------------
# Step 2: create the conda env (+ CUDA toolkit + persisted build vars)
#
# NOTE: on machines where conda is rooted inside another env (so `conda info --base`
# is NOT <miniforge>, e.g. it returns <miniforge>/envs/simfoundry here), a name-based
# `conda create` lands the env under that root's `envs/`. Deactivate to the real base
# first, then verify the env is resolvable by `mamba run -n`.
# ------------------------------------------------------------------------------
eval "$(conda shell.bash hook)"
# conda/cuda-nvcc (de)activate hooks aren't `set -u` safe (they reference
# NVCC_PREPEND_FLAGS with no default); relax nounset while toggling envs.
set +u
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
set -u

echo "Creating conda env '${env_name}' (CUDA ${cuda_version})..."
( cd "${THREEDGRUT_DIR}" && CUDA_VERSION="${cuda_version}" bash scripts/create_conda.sh "${env_name}" )

if ! mamba run -n "${env_name}" true 2>/dev/null; then
  echo "ERROR: env '${env_name}' was created but 'mamba run -n ${env_name}' can't find it."
  echo "  It may have been nested under an active env. Check 'mamba env list', remove the"
  echo "  nested copy with 'conda env remove -p <path>', and re-run from a base shell."
  exit 1
fi

# ------------------------------------------------------------------------------
# Step 2.5: ensure uv is available inside the env
#
# install_env_uv.sh (Step 3) requires `uv` on PATH and uses it to `uv pip install`
# into the env. Rather than depend on a global uv, install it into the env itself so
# `mamba run -n "${env_name}"` in Step 3 resolves an env-local uv. Idempotent: skip if
# the env already provides one.
# ------------------------------------------------------------------------------
if ! mamba run -n "${env_name}" command -v uv &>/dev/null; then
  echo "Installing uv into '${env_name}' (pip)..."
  mamba run -n "${env_name}" python -m pip install -U uv
fi
echo "  uv: $(mamba run -n "${env_name}" uv --version)"

# ------------------------------------------------------------------------------
# Step 3: install the project (+ Kaolin, ppisp, fused-ssim, slangc)
#
# Use `mamba run -n` rather than `conda activate`: on the conda-rooted-in-another-env
# machines above, `conda activate <name>` resolves to the wrong prefix. `mamba run`
# re-applies the env's persisted nvcc/uv/CC/CXX/TORCH_* vars correctly.
# ------------------------------------------------------------------------------
echo "Installing 3dgrut project into '${env_name}'..."
( cd "${THREEDGRUT_DIR}" && mamba run -n "${env_name}" bash install_env_uv.sh "${env_name}" )

# ------------------------------------------------------------------------------
# Step 4: (optional) torch-cache .so reference fix
# See https://github.com/nv-tlabs/3dgrut/issues/167#issuecomment-3558219094
# Harmless no-op if the cached .so files don't exist (current versions JIT-compile
# into the torch cache and load from there without needing this copy).
# ------------------------------------------------------------------------------
TORCH_CACHE_DIR="$HOME/.cache/torch_extensions/py311_cu128"
for pair in "lib3dgrt_cc:threedgrt_tracer" "lib3dgut:threedgut_tracer"; do
  lib="${pair%%:*}"; dest="${pair##*:}"
  src="${TORCH_CACHE_DIR}/${lib}/${lib}.so"
  if [ -f "${src}" ]; then
    cp "${src}" "${THREEDGRUT_DIR}/${dest}/${lib}.so"
    echo "Copied ${lib}.so into ${dest}/ (torch-cache reference fix)"
  fi
done

# ------------------------------------------------------------------------------
# Step 5: verify
# (No FAISS here — this env only does PLY->USDZ; faiss-gpu=1.12 would risk a
#  numpy/torch downgrade against the env's torch 2.8.0+cu128.)
# ------------------------------------------------------------------------------
echo "Verifying 3dgrut install..."
mamba run -n "${env_name}" python -c "import threedgrut, kaolin, ppisp; from fused_ssim import fused_ssim; print('3dgrut import OK')"

echo ""
echo "Completed installation of 3DGRUT environment: ${env_name}"
echo "  (The 3dgut CUDA plugin lib3dgut_cc.so JIT-compiles on the first PLY->USDZ run.)"
