#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# errexit + pipefail so a failed build/install ABORTS loudly instead of leaving a
# silently-incomplete env. (No `-u`: conda/cuda-nvcc activate hooks reference unbound
# vars like NVCC_PREPEND_FLAGS and aren't nounset-safe.)
set -eo pipefail

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
env_name="hunyuan"
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
# HUNYUAN 3D ENVIRONMENT SETUP
# ==============================================================================

# Get environment name from user
echo "=== Hunyuan3D Environment Setup ==="

if [[ ! ${DEFAULT} == true ]]; then
  read -p "Enter environment name (default: ${env_name}): " ENV_NAME
fi
ENV_NAME=${ENV_NAME:-${env_name}}  # Use default name if empty

# Get CUDA version from user
DEFAULT_CUDA_VERSION="${CUDA_VERSION}"
if [[ ! ${DEFAULT} == true ]]; then
  read -p "Enter CUDA Toolkit version to install into the env (default: ${CUDA_VERSION}): " CUDA_VERSION
fi
CUDA_VERSION=${CUDA_VERSION:-${DEFAULT_CUDA_VERSION}}  # Use default version if empty

# NOTE: CUDA_HOME is NOT set to /usr/local/cuda-* here. On conda-managed machines that
# path doesn't exist, which made deepspeed + custom_rasterizer + DifferentiableRenderer
# all fail to build. Instead we install cuda-toolkit INTO the env below and point
# CUDA_HOME at $CONDA_PREFIX after activation.

# Set TORCH_CUDA_ARCH_LIST: use provided value or auto-detect from GPUs
if [[ -n "$CUDA_ARCH_LIST" ]]; then
  export TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST"
  echo "Using provided TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
else
  echo "Auto-detecting GPU compute capabilities..."
  if command -v nvidia-smi &> /dev/null; then
    # Query compute capabilities of all GPUs, deduplicate, and sort (e.g. RTX 5090 -> 12.0)
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

# Convert CUDA version to PyTorch URL format (e.g., 12.8 -> cu128)
CUDA_VERSION_SHORT=$(echo $CUDA_VERSION | sed 's/\.//')
CUDA_URL_SUFFIX="cu${CUDA_VERSION_SHORT}"

echo "Creating environment: $ENV_NAME"

# Step 3.1: Install Hunyuan3D
cd $PROJECT_ROOT  # Change to your project root

#Step 3.2: Create dependencies directory if not exists
if [ ! -d "deps" ]; then
  mkdir deps
fi
cd deps

# Step 3.3: hydrate the source checkout without discarding pre-downloaded checkpoints.
HUNYUAN_DIR="${PROJECT_ROOT}/deps/Hunyuan3D-2.1"
HUNYUAN_COMMIT="82920d643c0dc2f7bfd7255f45f62d386edfe60c"
if [[ ! -d "${HUNYUAN_DIR}/.git" ]]; then
  echo "Hunyuan3D source checkout is missing; preserving existing files and fetching source..."
  mkdir -p "${HUNYUAN_DIR}"
  git -C "${HUNYUAN_DIR}" init
  git -C "${HUNYUAN_DIR}" remote add origin https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
fi
if ! git -C "${HUNYUAN_DIR}" cat-file -e "${HUNYUAN_COMMIT}^{commit}" 2>/dev/null; then
  git -C "${HUNYUAN_DIR}" fetch --depth 1 origin "${HUNYUAN_COMMIT}"
fi
git -C "${HUNYUAN_DIR}" checkout --detach "${HUNYUAN_COMMIT}"

HUNYUAN_PATCH="${PROJECT_ROOT}/patches/Hunyuan3D-2.1.patch"
if git -C "${HUNYUAN_DIR}" apply --check --reverse "${HUNYUAN_PATCH}" 2>/dev/null; then
  echo "patches/Hunyuan3D-2.1.patch already applied"
elif git -C "${HUNYUAN_DIR}" apply --check "${HUNYUAN_PATCH}"; then
  git -C "${HUNYUAN_DIR}" apply "${HUNYUAN_PATCH}"
  echo "Applied patches/Hunyuan3D-2.1.patch"
else
  echo "ERROR: patches/Hunyuan3D-2.1.patch cannot be applied cleanly." >&2
  exit 1
fi
if [[ ! -f "${HUNYUAN_DIR}/requirements.txt" ]]; then
  echo "ERROR: Hunyuan3D source hydration did not produce requirements.txt." >&2
  exit 1
fi
cd "${HUNYUAN_DIR}"

# Step 3.4: Create environment
# Can't use pip 25.3+ because it uses a separate version of setuptools which doesn't play nice
#    with newest version of torch during source package installs
mamba create -y -n "$ENV_NAME" python=3.10 pip=25.2
mamba activate "$ENV_NAME"

# Step 3.4b: Install CUDA toolkit INTO the env (provides nvcc + headers) and point the
# build toolchain at it. Required to compile custom_rasterizer + DifferentiableRenderer.
mamba install -y -c nvidia "cuda-toolkit=${CUDA_VERSION}"
export CUDA_HOME="$CONDA_PREFIX"
# conda's cuda-toolkit puts headers under targets/x86_64-linux/include (not include/),
# and the libcuda stub under lib/stubs — expose both to the extension builds.
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib/stubs:${LIBRARY_PATH:-}"
echo "CUDA_HOME=$CUDA_HOME (nvcc: $(command -v nvcc))"

# Step 3.5: Install PyTorch and Blender Python API
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 xformers --index-url https://download.pytorch.org/whl/${CUDA_URL_SUFFIX} >> /dev/null
pip install bpy==4.0 --extra-index-url https://download.blender.org/pypi/

# Make sure compatible setuptools is installed
pip install "setuptools<80"

# Step 3.6: Install requirements, with two lines filtered out:
#   - The Tencent/Aliyun `--extra-index-url` mirrors (requirements.txt lines 1-2): they're
#     China mirrors that are very slow outside China — strip them so pip uses pypi.org.
#   - `deepspeed`: NOT imported anywhere in Hunyuan3D, and its setup.py needs a valid
#     CUDA_HOME at metadata-generation time — when it failed it aborted the ENTIRE
#     requirements install (leaving trimesh/diffusers/transformers/etc. uninstalled).
pip install -r <(grep -viE '^[[:space:]]*(deepspeed|--extra-index-url.*(aliyun|tencent))' requirements.txt)
echo "Installed Hunyuan3D requirements (pypi.org index; deepspeed excluded)"
# NOTE: the numpy-2 alignment (numpy/open3d/onnxruntime/opencv) is intentionally done LAST
# (Step 3.9) so that no later step — faiss, requirements_hunyuan, or `pip install -e .` —
# can revert numpy back to 1.x. Do NOT move it up here.

# Step 3.7: Install gcc (host compiler for the CUDA/C++ extension builds)
mamba install -c conda-forge gcc=13 gxx=13 -y
echo "Installed gcc for hunyuan"

# Build the custom CUDA rasterizer (uses CUDA_HOME/CPLUS_INCLUDE_PATH set above)
cd hy3dpaint/custom_rasterizer
pip install -e .
cd ../..

# Build the differentiable mesh painter (pybind11 C++ module)
cd hy3dpaint/DifferentiableRenderer
bash compile_mesh_painter.sh
cd ../..

# Step 3.8: Install digital cousins requirements
cd ../.. # back in root
pip install -r requirements_hunyuan.txt
pip install -e .

# Step 3.9: numpy-2 alignment — done LAST so nothing downstream reverts it. torch 2.7.0+cu128
# (required for RTX 5090 / sm_120) is built against numpy 2.x; torch.from_numpy raises
# "expected np.ndarray (got numpy.ndarray)" under numpy 1.x. The requirements pin numpy 1.24.4
# plus several numpy-1.x C-extensions that crash under numpy 2.x, so force the numpy-2 set:
#   - numpy 1.24.4 -> 2.2.6
#   - open3d 0.18.0 -> 0.19.0      (0.18 SEGFAULTS when trimesh.as_open3d gets numpy-2 arrays)
#   - onnxruntime 1.16.3 -> 1.19.2 (1.16 is numpy-1 ABI: "_ARRAY_API not found")
#   - opencv: a single numpy-2 build (pinned 4.10.0.84 is numpy-1; rembg also pulls headless)
#   - scipy 1.14.1 (from requirements) is already numpy-2 — pin it so it can't drift.
# NO FAISS: the hunyuan env only runs stage 7 (mesh-gen) and never imports faiss; conda
# faiss-gpu=1.12 requires numpy<2 and would downgrade numpy, breaking this whole stack.
pip uninstall -y opencv-python opencv-python-headless || true
pip install --no-cache-dir \
  "numpy==2.2.6" "scipy==1.14.1" "open3d==0.19.0" "onnxruntime==1.19.2" "opencv-python-headless==4.11.0.86"
echo "Completed installation of Hunyuan3D environment: $ENV_NAME"

mamba deactivate
