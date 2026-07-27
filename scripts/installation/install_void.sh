#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Install the `void` env used by the auto-BG pipeline's VOID inpainting passes (steps 2/3).
#
# There was previously no installer for this env (it was created by hand), so a clean
# rebuild regressed to a broken state. This script bakes in the RTX 5090 / sm_120 fixes:
#   - torch 2.7.1+cu128 (default PyPI wheel is cu126 and lacks sm_120 kernels)
#   - av==12.3.0 (av 17.x dropped av.logging, which cascades into a bogus
#     "cannot import name 'T5EncoderModel'" via torchvision)
#   - numpy==1.26.4, rp (required by make_warped_noise.py)
# It also clones deps/void-model (code) and re-applies patches/void-model.patch
# (the rp.save_video_mp4 libx264-overflow fix). Model WEIGHTS come from
# download_checkpoints.sh, not here.

set -euo pipefail

if ! command -v mamba >/dev/null 2>&1; then
  echo "Error: mamba was not found on PATH. Install Miniforge first." >&2
  exit 127
fi

eval "$(mamba shell hook --shell bash)"

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
project_root="$(cd "$SCRIPT_DIR/../.." && pwd)"
env_name="void"
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
  read -p "Enter project root (default: $project_root): " PROJECT_ROOT
  project_root="${PROJECT_ROOT:-$project_root}"
  read -p "Enter environment name (default: ${env_name}): " ENV_NAME
  env_name="${ENV_NAME:-$env_name}"
fi
PROJECT_ROOT="$(cd "$project_root" && pwd)"

echo "=== VOID Environment Setup ==="
echo "  project_root: ${PROJECT_ROOT}"
echo "  env_name:     ${env_name}"

# ------------------------------------------------------------------------------
# Step 1: clone deps/void-model (inference code) and re-apply our patch
# ------------------------------------------------------------------------------
VOID_DIR="${PROJECT_ROOT}/deps/void-model"
mkdir -p "${PROJECT_ROOT}/deps"
if [[ ! -d "${VOID_DIR}/.git" ]]; then
  echo "VOID source checkout is missing; preserving existing weights and fetching source..."
  mkdir -p "${VOID_DIR}"
  git -C "${VOID_DIR}" init
  git -C "${VOID_DIR}" remote add origin https://github.com/netflix/void-model
  # TODO(SimFoundry): confirm this SHA matches a tested build before release (pinned 2026-07-27).
  git -C "${VOID_DIR}" fetch --depth 1 origin "${VOID_COMMIT:-e3914f8f551dd4b880661991fd6b28cd1699a97a}"
  git -C "${VOID_DIR}" checkout --detach FETCH_HEAD
fi
if [[ ! -f "${VOID_DIR}/requirements.txt" ]]; then
  echo "ERROR: VOID source hydration did not produce requirements.txt." >&2
  exit 1
fi

PATCH="${PROJECT_ROOT}/patches/void-model.patch"
if [ -f "${PATCH}" ]; then
  if git -C "${VOID_DIR}" apply --check --reverse "${PATCH}" 2>/dev/null; then
    echo "patches/void-model.patch already applied"
  elif git -C "${VOID_DIR}" apply --check "${PATCH}"; then
    git -C "${VOID_DIR}" apply "${PATCH}"
    echo "Applied patches/void-model.patch (rp.save_video_mp4 libx264-overflow fix)"
  else
    echo "ERROR: patches/void-model.patch cannot be applied cleanly." >&2
    exit 1
  fi
else
  echo "WARNING: ${PATCH} not found; the VOID Pass 2 libx264-overflow fix will be missing."
fi

# ------------------------------------------------------------------------------
# Step 2: create the env + install deps with the sm_120 / av fixes
# ------------------------------------------------------------------------------
mamba create -y -n "${env_name}" python=3.11
mamba run -n "${env_name}" pip install -r "${VOID_DIR}/requirements.txt"
# torch cu128 for sm_120 (RTX 5090 / Blackwell)
mamba run -n "${env_name}" pip install --force-reinstall torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
# av==12.3.0 (exports av.logging, which torchvision 0.22.1 needs); numpy pin; rp
mamba run -n "${env_name}" pip install "av==12.3.0" "numpy==1.26.4" rp

echo "Verifying void env..."
mamba run -n "${env_name}" python -c "import av; av.logging.set_level(av.logging.ERROR); import torch, rp; print('void OK: torch', torch.__version__, '| av', av.__version__, '| cuda', torch.cuda.is_available())"

echo "Completed installation of VOID environment: ${env_name}"
echo "  (Model weights: run scripts/installation/download_checkpoints.sh)"
