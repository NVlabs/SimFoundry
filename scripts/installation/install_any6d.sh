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
env_name="any6d"
DEFAULT=false
CUDA_VERSION="12.8"
CUDA_ARCH_LIST=""

# Parse command-line options
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-root) project_root="$2"; shift 2 ;;
        --env-name) env_name="$2"; shift 2 ;;
        --default) DEFAULT=true; shift ;;
        --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
        --cuda-arch-list) CUDA_ARCH_LIST="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! ${DEFAULT} == true ]]; then
  read -p "Enter project root (default: $project_root): " PROJECT_ROOT
fi
PROJECT_ROOT=${PROJECT_ROOT:-$project_root}

# ==============================================================================
# ANY6D POSE ESTIMATION ENVIRONMENT SETUP
# ==============================================================================

# Get environment name from user
echo "=== Any6D Environment Setup ==="

if [[ ! ${DEFAULT} == true ]]; then
  read -p "Enter environment name (default: ${env_name}): " ENV_NAME
fi
ENV_NAME=${ENV_NAME:-${env_name}}  # Use default name if empty

echo "Creating environment: $ENV_NAME"

# CUDA toolkit is required to build CUDA extensions (nvdiffrast, foundationpose).
# Point CUDA_HOME at the system toolkit, falling back to nvcc on PATH or a
# toolkit installed into the conda env.
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:$LIBRARY_PATH

ensure_cuda_toolkit() {
  if [[ -x "${CUDA_HOME}/bin/nvcc" ]]; then
    return
  fi

  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    export CUDA_HOME
    export LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH:-}"
    return
  fi

  echo "nvcc was not found at ${CUDA_HOME}/bin/nvcc; installing CUDA toolkit ${CUDA_VERSION} into ${ENV_NAME}"
  mamba install -c nvidia "cuda-toolkit=${CUDA_VERSION}" -y > /dev/null
  export CUDA_HOME="${CONDA_PREFIX}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH:-}"

  if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "Error: nvcc was not found after installing cuda-toolkit into ${ENV_NAME}." >&2
    exit 1
  fi
}

# Step 4.1: Install Any6D
cd $PROJECT_ROOT

#Step 4.2: Create dependencies directory if not exists
if [ ! -d "deps" ]; then
  mkdir deps
fi
cd deps

# Step 4.3: clone Any6D
# TODO(SimFoundry): confirm this SHA matches a tested build before release (pinned 2026-07-27).
ANY6D_COMMIT="${ANY6D_COMMIT:-80eb4866a1c96ecb18be18836aba4f4bd6e80e9e}"
if [ ! -d "Any6D" ]; then
  git clone https://github.com/taeyeopl/Any6D.git
  git -C Any6D checkout --detach "${ANY6D_COMMIT}"
  cd Any6D
  git apply ../../patches/Any6D.patch
  cd ..
fi

cd Any6D
# Step 4.4: Create environment
# Can't use pip 25.3+ because it uses a separate version of setuptools which doesn't play nice
#    with newest version of torch during source package installs
mamba create -y -n "$ENV_NAME" python=3.10 pip=25.2
mamba activate "$ENV_NAME"

# Make sure CUDA_HOME points at a usable toolkit before building CUDA extensions
ensure_cuda_toolkit
echo "Using CUDA_HOME=${CUDA_HOME}"

# Set TORCH_CUDA_ARCH_LIST: use provided value or auto-detect from GPUs.
# Without this, torch builds CUDA extensions for every arch baked into the
# cu128 wheel (sm_50..sm_120); the old archs lack atomicAdd(double*) and break
# foundationpose's mycuda build.
if [[ -n "$CUDA_ARCH_LIST" ]]; then
  export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST"
  echo "Using provided TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
else
  echo "Auto-detecting GPU compute capabilities..."
  if command -v nvidia-smi &> /dev/null; then
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

# Step 4.5: Install PyTorch
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 xformers --index-url https://download.pytorch.org/whl/cu128 >> /dev/null
echo "Installed PyTorch for Any6D"

# Make sure compatible setuptools is installed
pip install "setuptools<80"

# Step 4.6: Install Eigen3
mamba install conda-forge::eigen=3.4.0 -y >> /dev/null
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$CONDA_PREFIX/include/eigen3"
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include/eigen3:$CPLUS_INCLUDE_PATH

# Step 4.7: Install requirements
pip install -r requirements.txt

# Step 4.8: Install NVDiffRast 
# TODO(SimFoundry): confirm this SHA matches a tested build before release (pinned 2026-07-27).
pip install --quiet --no-cache-dir --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae

# Step 4.9: Install Kaolin
pip install --no-cache-dir --quiet kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu128.html 

# Step 4.10: Install PyTorch3D
pip install --quiet --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.7.0cu128

# Step 4.11: Install foundationpose
# mycpp's CMake build needs Boost (system, program_options); install it and put
# $CONDA_PREFIX on CMAKE_PREFIX_PATH so cmake finds BoostConfig.cmake/Eigen3.
mamba install boost -y >> /dev/null
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/pybind11/share/cmake/pybind11:$CONDA_PREFIX/include/eigen3:$CONDA_PREFIX bash foundationpose/build_all_conda.sh >> /dev/null
echo "Installed foundationpose"

# Step 4.12: Install sam2
cd sam2 && pip install -e . && cd .. >> /dev/null
echo "Installed sam2"

# Step 4.13: Install bop_toolkit
cd bop_toolkit && python setup.py install && cd .. >> /dev/null
echo "Installed bop_toolkit"

# Step 4.14: Install CDC requirements
cd ../.. # back in root
pip install -r requirements.txt >> /dev/null
pip install -e . >> /dev/null
echo "Installed ACDC in Any6D environment"

# Step 4.15: Install and validate faiss-gpu
install_faiss_gpu "$ENV_NAME"

echo "Completed installation of Any6D environment: $ENV_NAME"

mamba deactivate

