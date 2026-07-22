#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Error handling: Exit and print the offending line and error message on failure
error_handler() {
    local exit_code=$?
    local line_no=$1
    echo "Error occurred at line $line_no: $(sed "${line_no}q;d" "$0")"
    echo "Exit code: $exit_code"
    exit $exit_code
}
trap 'error_handler $LINENO' ERR
set -o errexit
set -o pipefail

if ! command -v mamba >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Error: mamba was not found on PATH.

Install Miniforge, or install mamba into your base conda
environment, then rerun this script.
EOF
  exit 127
fi

eval "$(mamba shell hook --shell bash)"

# Get script dir
# repo dir is grandparent directory, by default
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
source "${SCRIPT_DIR}/faiss_gpu.sh"
project_root="$(cd "$SCRIPT_DIR/../.." && pwd)"
env_name="da3"
DEFAULT=false
CUDA_ARCH_LIST=""

# Parse command-line options
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-root) project_root="$2"; shift 2 ;;
        --env-name) env_name="$2"; shift 2 ;;
        --default) DEFAULT=true; shift ;;
        --cuda-arch-list) CUDA_ARCH_LIST="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! ${DEFAULT} == true ]]; then
  read -p "Enter project root (default: $project_root): " PROJECT_ROOT
fi
PROJECT_ROOT=${PROJECT_ROOT:-$project_root}

# ==============================================================================
# DEPTH ANYTHING 3 ENVIRONMENT SETUP
# ==============================================================================

# Get environment name from user
echo "=== Depth Anything 3 Environment Setup ==="

if [[ ! ${DEFAULT} == true ]]; then
  read -p "Enter environment name (default: ${env_name}): " ENV_NAME
fi
ENV_NAME=${ENV_NAME:-${env_name}}  # Use default name if empty

# Set TORCH_CUDA_ARCH_LIST: use provided value or auto-detect from GPUs
if [[ -n "$CUDA_ARCH_LIST" ]]; then
  export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST"
  echo "Using provided TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
else
  echo "Auto-detecting GPU compute capabilities..."
  if command -v nvidia-smi &> /dev/null; then
    # Query compute capabilities of all GPUs, deduplicate, and sort
    DETECTED_ARCHS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u -V | tr '\n' ';' | sed 's/;$//')
    if [[ -n "$DETECTED_ARCHS" ]]; then
      export TORCH_CUDA_ARCH_LIST="$DETECTED_ARCHS"
      echo "Auto-detected TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
    else
      echo "WARNING: nvidia-smi found but no GPU compute capabilities detected. TORCH_CUDA_ARCH_LIST will not be set."
    fi
  else
    echo "WARNING: nvidia-smi not found. TORCH_CUDA_ARCH_LIST will not be set."
  fi
fi

echo "Creating environment: $ENV_NAME"

# Create environment
# Can't use pip 25.3+ because it uses a separate version of setuptools which doesn't play nice
#    with newest version of torch during source package installs
mamba create -y -n "$ENV_NAME" python=3.11 "pip<25.3"
mamba activate "$ENV_NAME"

# Install PyTorch
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128 >> /dev/null
pip install "setuptools<80"
mamba install -y -c nvidia cuda-toolkit=12.8
pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@v0.0.30#egg=xformers
echo "Installed PyTorch, xformers, and CUDA toolkit 12.8 for DA3"

# Install gsplat (required for 3DGS rendering)
pip install git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70 >> /dev/null
echo "Installed gsplat for DA3"

# Install pycolmap from pip
pip install pycolmap >> /dev/null
echo "Installed pycolmap for DA3"

# Install pybullet from pip
pip install pybullet >> /dev/null
echo "Installed pybullet for DA3"

# Install Depth Anything 3 from source
cd $PROJECT_ROOT
pip install -r requirements.txt >> /dev/null
pip install -e . >> /dev/null
if [ ! -d "deps" ]; then
  mkdir deps
fi
cd deps

if [ ! -d "Depth-Anything-3" ]; then
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
fi
cd Depth-Anything-3
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt >> /dev/null
fi
pip install -e . >> /dev/null
echo "Installed Depth Anything 3"

# Install other deps
pip install ipython addict

install_faiss_gpu "$ENV_NAME"

echo "Completed installation of Depth Anything 3 environment: $ENV_NAME"

mamba deactivate
