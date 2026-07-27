#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Installation script for OpenPI Pi0.5 environment
# Sets up environment for evaluating OpenPI pretrained checkpoints on OmniGibson

eval "$(mamba shell hook --shell bash)"

# Get script dir
# repo dir is grandparent directory, by default
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
source "${SCRIPT_DIR}/faiss_gpu.sh"
project_root="$(cd "$SCRIPT_DIR/../.." && pwd)"
env_name="openpi"
DEFAULT=false

# Parse command-line options
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-root) project_root="$2"; shift 2 ;;
        --env-name) env_name="$2"; shift 2 ;;
        --default) DEFAULT=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! ${DEFAULT} == true ]]; then
  read -p "Enter project root (default: $project_root): " PROJECT_ROOT
fi
PROJECT_ROOT=${PROJECT_ROOT:-$project_root}

# ==============================================================================
# OPENPI ENVIRONMENT SETUP
# ==============================================================================

# Get environment name from user
echo "=== OpenPI Pi0.5 Environment Setup ==="

if [[ ! ${DEFAULT} == true ]]; then
  read -p "Enter environment name (default: ${env_name}): " ENV_NAME
fi
ENV_NAME=${ENV_NAME:-${env_name}}  # Use default name if empty

echo "Creating environment: $ENV_NAME"

# Step 1: Create mamba environment with Python 3.11
# Can't use pip 25.3+ because it uses a separate version of setuptools which doesn't play nice
#    with newest version of torch during source package installs
mamba create -y -n "$ENV_NAME" python=3.11 "pip<25.3" || { echo "Error: Failed to create mamba environment ${ENV_NAME}"; exit 1; }
mamba activate "$ENV_NAME" || { echo "Error: Failed to activate mamba environment ${ENV_NAME}"; exit 1; }

# Make sure compatible setuptools is installed
pip install "setuptools<80"

# Step 2: Install conda-build and base requirements
cd $PROJECT_ROOT  # Change to your project root

mamba install conda-build -y

# Step 3: Create dependencies directory if not exists
if [ ! -d "deps" ]; then
  mkdir deps
fi
cd deps

# Step 4: Install BEHAVIOR-1K
echo "=== Installing BEHAVIOR-1K ==="
# TODO(SimFoundry): confirm this SHA matches a tested build before release (pinned 2026-07-27).
# Was tracking branch feat/isaac-5.0.
BEHAVIOR1K_COMMIT="${BEHAVIOR1K_COMMIT:-d89aae4e0e9a1de3cf8285cb9669c11d8c8bb864}"
if [ ! -d "BEHAVIOR-1K" ]; then
  git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
fi
cd BEHAVIOR-1K
git checkout --detach "${BEHAVIOR1K_COMMIT}"
echo "Starting BEHAVIOR-1K setup, this may take a while..."
./setup.sh --bddl --omnigibson --cuda-version 12.8 --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
echo "Installed BEHAVIOR-1K"
cd ..

# Step 5: Install PyTorch with CUDA 12.8 support
echo "=== Installing PyTorch ==="
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
# Make sure compatible setuptools is installed (may have been overwritten)
pip install "setuptools<80"
echo "Installed PyTorch 2.7.1 with CUDA 12.8"

# Step 6: Clone OpenPI repository
echo "=== Installing OpenPI ==="
# TODO(SimFoundry): confirm this SHA matches a tested build before release (pinned 2026-07-27).
OPENPI_COMMIT="${OPENPI_COMMIT:-15a9616a00943ada6c20a0f158e3adb39df2ccac}"
if [ ! -d "openpi" ]; then
  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
  git -C openpi checkout --detach "${OPENPI_COMMIT}"
  git -C openpi submodule update --init --recursive
fi
cd openpi

# # Step 7: Install uv package manager (used by OpenPI)
# echo "Installing uv package manager..."
# curl -LsSf https://astral.sh/uv/install.sh | sh
# source $HOME/.local/bin/env 2>/dev/null || true  # Source uv if just installed
# export PATH="$HOME/.local/bin:$PATH"

# # Step 8: Install OpenPI dependencies using uv
# # GIT_LFS_SKIP_SMUDGE=1 skips downloading large files during install
# echo "Installing OpenPI dependencies..."
# GIT_LFS_SKIP_SMUDGE=1 uv sync

# Step 9: Patch OpenPI to fix lerobot import paths
# OpenPI uses "lerobot.common" but newer versions of lerobot use just "lerobot"
echo "Patching OpenPI lerobot imports..."
find . -type f -name "*.py" -exec sed -i 's/lerobot\.common/lerobot/g' {} +
echo "Patched lerobot imports"

# Step 10: Install OpenPI in editable mode
echo "Installing OpenPI package..."
GIT_LFS_SKIP_SMUDGE=1 pip install -e .

# Alternative: Install with regular pip if uv fails
# pip install -e .

echo "Installed OpenPI"
cd ..

# Step 11: Install additional dependencies
echo "=== Installing additional dependencies ==="
pip install einops hydra-core omegaconf typing_extensions

# Step 12: Install digital cousins
cd $PROJECT_ROOT
pip install -r requirements.txt > /dev/null
pip install -e . > /dev/null
echo "Installed CDC"

# Step 13: Install additional packages for simulation + LeRobot dep
pip install packaging==25
pip install lerobot
pip uninstall -y numpy
mamba install "numpy<2" -y
install_faiss_gpu "$ENV_NAME"
echo "Fixed numpy version"

echo ""
echo "============================================================"
echo "Completed installation of OpenPI environment: $ENV_NAME"
echo "============================================================"
echo ""
echo "To activate the environment, run:"
echo "  mamba activate $ENV_NAME"
echo ""
echo "To test OpenPI loading, run:"
echo "  python -c \"from openpi.policies import policy_config; print('OpenPI loaded successfully!')\""
echo ""

mamba deactivate
