#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Error handling: exit and print the offending line on failure.
# Resolve the script path up front: $0 is relative and the script cd's around.
SELF_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
error_handler() {
    local exit_code=$?
    local line_no=$1
    echo "Error occurred at line $line_no: $(sed "${line_no}q;d" "$SELF_PATH")"
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
source "${SCRIPT_DIR}/git_safe.sh"
project_root="$(cd "$SCRIPT_DIR/../.." && pwd)"
env_name="simfoundry"
DEFAULT=false
# Exported ENV_NAME/PROJECT_ROOT must not silently override the flags; note and clear.
INHERITED_ENV_NAME="${ENV_NAME:-}"
INHERITED_PROJECT_ROOT="${PROJECT_ROOT:-}"
unset ENV_NAME PROJECT_ROOT
CUDA_VERSION="12.8"
INSTALL_TRELLIS=false
INSTALL_ZED=false
CUDA_ARCH_LIST=""
ROBOT_ASSET_FALLBACK_ROOT=""

# Parse command-line options
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-root) project_root="$2"; shift 2 ;;
        --env-name) env_name="$2"; shift 2 ;;
        --default) DEFAULT=true; shift ;;
        --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
        --trellis) INSTALL_TRELLIS=true; shift ;;
        --zed) INSTALL_ZED=true; shift ;;
        --cuda-arch-list) CUDA_ARCH_LIST="$2"; shift 2 ;;
        --robot-asset-fallback-root) ROBOT_ASSET_FALLBACK_ROOT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! ${DEFAULT} == true ]]; then
  read -p "Enter project root (default: $project_root): " PROJECT_ROOT
fi
PROJECT_ROOT=${PROJECT_ROOT:-$project_root}
if [[ -n "${INHERITED_PROJECT_ROOT}" && "${INHERITED_PROJECT_ROOT}" != "${PROJECT_ROOT}" ]]; then
  echo "NOTE: ignoring PROJECT_ROOT='${INHERITED_PROJECT_ROOT}' exported in the environment; using '${PROJECT_ROOT}'." \
       "Pass --project-root to choose a different root."
fi

# ==============================================================================
# MAIN SimFoundry ENVIRONMENT SETUP
# ==============================================================================

# Get environment name from user
echo "=== SimFoundry Environment Setup ==="

if [[ ! ${DEFAULT} == true ]]; then
  read -p "Enter environment name (default: ${env_name}): " ENV_NAME
fi
ENV_NAME=${ENV_NAME:-${env_name}}  # Use default name if empty
if [[ -n "${INHERITED_ENV_NAME}" && "${INHERITED_ENV_NAME}" != "${ENV_NAME}" ]]; then
  echo "NOTE: ignoring ENV_NAME='${INHERITED_ENV_NAME}' exported in the environment; using '${ENV_NAME}'." \
       "Pass --env-name to choose a different env."
fi

# Get CUDA version from user
DEFAULT_CUDA_VERSION="${CUDA_VERSION}"
if [[ ! ${DEFAULT} == true ]]; then
  read -p "Enter CUDA Toolkit version installed on the system (default: ${CUDA_VERSION}): " CUDA_VERSION
fi
CUDA_VERSION=${CUDA_VERSION:-${DEFAULT_CUDA_VERSION}}  # Use default version if empty

export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:$LIBRARY_PATH

_check_nvcc_version() {
  local nvcc_bin="$1"
  local found
  found=$("$nvcc_bin" --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | sed 's/release //')
  if [[ -z "$found" ]]; then
    echo "WARNING: could not parse CUDA version from $nvcc_bin --version" >&2
    return
  fi
  if [[ "$found" != "$CUDA_VERSION" ]]; then
    echo "Error: nvcc at $nvcc_bin reports CUDA ${found}, but CUDA_VERSION=${CUDA_VERSION} was requested." >&2
    echo "Re-run with --cuda-version ${found} to match your installed toolkit, or install CUDA ${CUDA_VERSION}." >&2
    exit 1
  fi
}

ensure_cuda_toolkit() {
  if [[ -x "${CUDA_HOME}/bin/nvcc" ]]; then
    _check_nvcc_version "${CUDA_HOME}/bin/nvcc"
    return
  fi

  local nvcc_path
  if nvcc_path="$(command -v nvcc 2>/dev/null)"; then
    _check_nvcc_version "$nvcc_path"
    CUDA_HOME="$(dirname "$(dirname "$nvcc_path")")"
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

# Convert CUDA version to PyTorch URL format (e.g., 12.6 -> cu126)
CUDA_VERSION_SHORT=$(echo $CUDA_VERSION | sed 's/\.//')
CUDA_URL_SUFFIX="cu${CUDA_VERSION_SHORT}"

# Step 2.1: Create new environment
# Can't use pip 25.3+ because it uses a separate version of setuptools which doesn't play nice
#    with newest version of torch during source package installs
if mamba env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  # Reuse is not a no-op: `pip install -e .` rebinds the env to this checkout.
  echo "WARNING: environment '$ENV_NAME' already exists and will be REUSED." >&2
  echo "         Continuing will install into it and rebind its 'simfoundry' package to: ${PROJECT_ROOT}" >&2
  echo "         Pass --env-name to install into a differently named environment instead." >&2
  if [[ ! ${DEFAULT} == true ]]; then
    read -p "Reuse environment '$ENV_NAME'? [y/N] " REUSE_REPLY || REUSE_REPLY=""
    if [[ ! "${REUSE_REPLY}" =~ ^[Yy] ]]; then
      echo "Aborted: environment '$ENV_NAME' was not modified."
      exit 1
    fi
  fi
  echo "Using existing environment: $ENV_NAME"
else
  echo "Creating environment: $ENV_NAME"
  mamba create -y -n "$ENV_NAME" python=3.11 "pip<25.3"
fi
mamba activate "$ENV_NAME"

pip install opencv-python numpy cython pyopengl requests argparse >> /dev/null
# Step 2.1.5: Install ZED (optional)
if [[ ${INSTALL_ZED} == true ]]; then
  echo "Installing ZED Python API..."
 
  cd "/usr/local/zed/"
  python get_python_api.py >> /dev/null
  echo "Installed ZED Python API"
  cd $PROJECT_ROOT
fi

# Step 2.2: Install conda-build and base requirements
cd $PROJECT_ROOT  # Change to your project root

mamba install conda-build -y
echo "Installing PyTorch and torch-cluster"
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 xformers --index-url https://download.pytorch.org/whl/${CUDA_URL_SUFFIX} >> /dev/null
# Make sure compatible setuptools is installed
pip install "setuptools<80"
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+${CUDA_URL_SUFFIX}.html >> /dev/null
echo "Installed PyTorch and torch-cluster"
pip install -r requirements.txt > /dev/null
pip install -e . > /dev/null
echo "Installed SimFoundry"

#Step 2.3: Create dependencies directory if not exists
if [ ! -d "deps" ]; then
  mkdir deps
fi
cd deps

# Step 2.4: Install DINOv2
DINOV2_COMMIT="${DINOV2_COMMIT:-7764ea0f912e53c92e82eb78a2a1631e92725fc8}"
if [ ! -d "dinov2" ]; then
  git clone https://github.com/facebookresearch/dinov2.git
  git -C dinov2 checkout --detach "${DINOV2_COMMIT}"
fi
cd dinov2
conda-develop .  # Do NOT run 'pip install -r requirements.txt'!!
echo "Installed DINOv2"
cd ..

echo "Now in: $(pwd)"

# Step 2.6: Install CLIP
echo "Installing CLIP"
pip install git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6 > /dev/null
echo "Installed CLIP"

# Step 2.7: Install Foundation Stereo
FOUNDATIONSTEREO_COMMIT="${FOUNDATIONSTEREO_COMMIT:-6e8806816b533e4d13ddbb95ffa907b797060a62}"
if [ ! -d "FoundationStereo" ]; then
  git clone https://github.com/NVlabs/FoundationStereo.git
  git -C FoundationStereo checkout --detach "${FOUNDATIONSTEREO_COMMIT}"
fi
cd FoundationStereo
echo "Installing Foundation Stereo"
# No need to actually install FS deps, since it's subsumed within SimFoundry's / other deps
echo "Installed Foundation Stereo"
echo "Installing Flash Attention"
pip install packaging ninja
ensure_cuda_toolkit
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl > /dev/null
# pip install flash-attn==2.7.3 --no-build-isolation
pip install flash-attn==2.8.3 --no-build-isolation
echo "Installed Flash Attention for Foundation Stereo"
cd ..

# Step 2.8: Install FoundationPose
if [ ! -d "FoundationPose" ]; then
  git clone https://github.com/NVlabs/FoundationPose.git
  git -C FoundationPose checkout e3d597b8c6b851d053094ebd6fa240191c5238f8
  git -C FoundationPose apply "${PROJECT_ROOT}/patches/FoundationPose.patch"
fi
cd FoundationPose

pip install -r requirements.txt > /dev/null
# Install compiler toolchain needed for nvdiffrast.
# kernel-headers >= 6.12 is required because pip-built C extensions (evdev) generate code by
# scanning the HOST's /usr/include (kernel >= 6.7 constants like KEY_LINK_PHONE) but compile
# against the conda sysroot; the older 5.14 sysroot headers make that compile fail.
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
  mamba install -c conda-forge gcc=13 gxx=13 gcc_linux-aarch64=13 gxx_linux-aarch64=13 "kernel-headers_linux-aarch64>=6.12.0" -y > /dev/null
else
  mamba install -c conda-forge gcc=13 gxx=13 gcc_linux-64=13 gxx_linux-64=13 "kernel-headers_linux-64>=6.12.0" -y > /dev/null
fi
# pip install --quiet --no-cache-dir --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
if [ ! -d "extensions" ]; then
    mkdir extensions
fi
cd extensions

if [ ! -d "nvdiffrast" ]; then
  git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git nvdiffrast
  git -C nvdiffrast checkout --detach "${NVDIFFRAST_COMMIT:-253ac4fcea7de5f396371124af597e6cc957bfae}"
fi
pip install nvdiffrast/ --no-build-isolation
cd ..
mamba install boost cmake -y > /dev/null
mamba install -c conda-forge eigen=3.4.0 -y > /dev/null

EIGEN_INCLUDE="${CONDA_PREFIX}/include/eigen3"
if [[ ! -f "${EIGEN_INCLUDE}/Eigen/Dense" ]]; then
  echo "ERROR: Conda Eigen headers not found at ${EIGEN_INCLUDE}" >&2
  exit 1
fi
PYBIND11_CMAKE_DIR="$(python -m pybind11 --cmakedir)"
export CPLUS_INCLUDE_PATH="${EIGEN_INCLUDE}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export CMAKE_PREFIX_PATH="${PYBIND11_CMAKE_DIR}:${CONDA_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

# FoundationPose's helper does not enable errexit and ends with a successful cd,
# which otherwise masks failed CMake, NVCC, or pip extension builds.
bash -e build_all_conda.sh
python -c "import torch, common, gridencoder"
FOUNDATIONPOSE_MYCPP_MODULES=(mycpp/build/mycpp*.so)
if [[ ! -s "${FOUNDATIONPOSE_MYCPP_MODULES[0]}" ]]; then
  echo "ERROR: FoundationPose mycpp extension was not produced" >&2
  exit 1
fi
echo "Installed FoundationPose"
cd ..

# Step 2.9: Install ml-depth-pro
if [ ! -d "ml-depth-pro" ]; then
  git clone https://github.com/apple/ml-depth-pro.git
  git -C ml-depth-pro checkout 9efe5c1def37a26c5367a71df664b18e1306c708
  git -C ml-depth-pro apply "${PROJECT_ROOT}/patches/ml-depth-pro.patch"
fi
cd ml-depth-pro

pip install -e . > /dev/null
echo "Installed ml-depth-pro"
cd ..

# Step 2.10: Install Prior Depth Anything
if [ ! -d "Prior-Depth-Anything" ]; then
  git clone https://github.com/SpatialVision/Prior-Depth-Anything.git
  git -C Prior-Depth-Anything checkout 8c029cbca669443fe0bbf8dcefb5f91ad531084d
  git -C Prior-Depth-Anything apply "${PROJECT_ROOT}/patches/PriorDepthAnything.patch"
fi
cd Prior-Depth-Anything

pip install -e . > /dev/null
echo "Installed Prior Depth Anything"
cd ..

# Step 2.11: Install Depth Anything 3 runtime for PDA geometric depth refinement
DA3_COMMIT="${DA3_COMMIT:-3d835ec1a5802d64a8b8b15f817a1ab54809bfe4}"
if [ ! -d "Depth-Anything-3" ]; then
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
  git -C Depth-Anything-3 checkout --detach "${DA3_COMMIT}"
fi
cd Depth-Anything-3
pip install "numpy==1.26.4" moviepy==1.0.3 pycolmap plyfile e3nn evo pillow_heif fastapi uvicorn typer safetensors > /dev/null
pip install --no-deps -e . > /dev/null
echo "Installed Depth Anything 3 runtime"
cd ..

# Step 2.12: Install BEHAVIOR-1K
# Was tracking branch feat/isaac-5.0.
BEHAVIOR1K_COMMIT="${BEHAVIOR1K_COMMIT:-d89aae4e0e9a1de3cf8285cb9669c11d8c8bb864}"
if [ ! -d "BEHAVIOR-1K" ]; then
  git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
fi
cd BEHAVIOR-1K
git_safe_checkout_detached "." "${BEHAVIOR1K_COMMIT}" "deps/BEHAVIOR-1K"
# Detect OS architecture to choose correct gcc/g++ packages.
# kernel-headers pin: see the toolchain install above (evdev scan/compile header mismatch).
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
  echo "Detected aarch64 architecture. Installing gcc_linux-aarch64/gxx_linux-aarch64."
  mamba install -c conda-forge gcc=13 gxx=13 gcc_linux-aarch64=13 gxx_linux-aarch64=13 "kernel-headers_linux-aarch64>=6.12.0" -y > /dev/null
else
  echo "Detected x86_64 or other architecture. Installing gcc_linux-64/gxx_linux-64."
  mamba install -c conda-forge gcc=13 gxx=13 gcc_linux-64=13 gxx_linux-64=13 "kernel-headers_linux-64>=6.12.0" -y > /dev/null
fi
echo "Starting BEHAVIOR-1K setup, this may take a while..."
./setup.sh --bddl --omnigibson --cuda-version ${CUDA_VERSION} --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos > /dev/null
echo "Installed BEHAVIOR-1K"
cd .. # back to deps directory

# Step 2.12: Install CUDA toolkit and pytorch3d
ensure_cuda_toolkit
# Was tracking the mutable 'stable' branch.
pip install "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
pip install packaging==25 > /dev/null
echo "Installed CUDA toolkit, pytorch3d, and packaging for SimFoundry"

# Step 2.13: Install coacd
pip install coacd > /dev/null
# OmniGibson already requires the PyPI pymeshlab~=2022.2 package. Asking conda-forge
# for pymeshlab again forces a redundant full-prefix solve that can take indefinitely
# in this mixed pip/conda env. Install only missing Python packages, with a hard deadline.
PYTHON_PACKAGE_TIMEOUT=${PYTHON_PACKAGE_TIMEOUT:-300}
if ! [[ "${PYTHON_PACKAGE_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PYTHON_PACKAGE_TIMEOUT must be a positive integer (seconds)." >&2
  exit 2
fi

missing_python_packages=()
python -c 'import pymeshlab' >/dev/null 2>&1 || missing_python_packages+=("pymeshlab~=2022.2")
python -c 'import evdev' >/dev/null 2>&1 || missing_python_packages+=("evdev==1.9.3")

if (( ${#missing_python_packages[@]} )); then
  echo "Installing missing mesh/input packages: ${missing_python_packages[*]}"
  echo "Package install deadline: ${PYTHON_PACKAGE_TIMEOUT}s"
  if timeout --signal=TERM --kill-after=30s "${PYTHON_PACKAGE_TIMEOUT}s" \
      python -m pip install --timeout 30 --retries 3 "${missing_python_packages[@]}"; then
    echo "Installed missing mesh/input packages"
  else
    install_status=$?
    if [[ ${install_status} -eq 124 || ${install_status} -eq 137 ]]; then
      echo "ERROR: Mesh/input package installation exceeded ${PYTHON_PACKAGE_TIMEOUT}s." >&2
    else
      echo "ERROR: Mesh/input package installation failed with exit code ${install_status}." >&2
    fi
    exit "${install_status}"
  fi
else
  echo "pymeshlab and evdev are already importable; skipping package installation"
fi

python -c 'import coacd, evdev, pymeshlab; print("Verified coacd, pymeshlab, and evdev imports")'
echo "Installed coacd, pymeshlab, and evdev"

# Step 2.14: Install SAM3
SAM3_COMMIT="${SAM3_COMMIT:-46957e47805eaa273f4aa7bbbd25a88bca9108ce}"
if [ ! -d "sam3" ]; then
  git clone https://github.com/facebookresearch/sam3.git
  git -C sam3 checkout --detach "${SAM3_COMMIT}"
fi
cd sam3

pip install -e . > /dev/null
pip install decord pycocotools 
echo "Installed SAM3"
cd .. # back to deps directory

# Install rembg. Use the [cpu] extra (plain onnxruntime), NOT [gpu]: the [gpu] extra
# pulls onnxruntime-gpu, whose recent builds (>=1.27) link CUDA 13 (libcudart.so.13) and
# fail to import against this env's CUDA 12.8. The codebase only ever runs rembg with
# CPUExecutionProvider (stage 6 + processing_utils.py), so the CPU build is sufficient.
pip install "rembg[cpu]" > /dev/null
echo "Installed rembg (CPU onnxruntime)"

# Step 2.15: Install TRELLIS2 (optional)
if [[ ${INSTALL_TRELLIS} == true ]]; then
  if [ ! -d "TRELLIS.2" ]; then
    git clone https://github.com/microsoft/TRELLIS.2.git --recursive
    git -C TRELLIS.2 checkout --detach "${TRELLIS2_COMMIT:-75fbf0183001ed9876c8dbb35de6b68552ee08bd}"
    git -C TRELLIS.2 submodule update --init --recursive
  fi
  cd TRELLIS.2
  pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers==4.57.6 gradio==6.0.1 tensorboard pandas lpips zstandard utils3d
  pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
  # >=3.3.1 for sm_120 (Blackwell) tl.dot support; <3.4 for torch 2.7.0 inductor compatibility.
  pip install "triton>=3.3.1,<3.4"
  # Check if libjpeg-dev is installed (required for pillow-simd)
  if ! dpkg -s libjpeg-dev &> /dev/null; then
    echo "ERROR: libjpeg-dev is not installed but is required for TRELLIS2."
    echo "Please install it manually before running this script:"
    echo "  sudo apt install -y libjpeg-dev"
    exit 1
  fi
  pip install pillow-simd
  pip install kornia timm psutil
  conda-develop .

  # Now install all the extensions for TRELLIS2
  if [ ! -d "extensions" ]; then
    mkdir extensions
  fi
  cd extensions

  if [ ! -d "nvdiffrast" ]; then
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git nvdiffrast
    git -C nvdiffrast checkout --detach "${NVDIFFRAST_COMMIT:-253ac4fcea7de5f396371124af597e6cc957bfae}"
  fi
  pip install nvdiffrast/ --no-build-isolation

  if [ ! -d "nvdiffrec" ]; then
    git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git nvdiffrec
    git -C nvdiffrec checkout --detach "${NVDIFFREC_COMMIT:-b296927cc7fd01c2ac1087c8065c4d7248f72da4}"
  fi
  pip install nvdiffrec/ --no-build-isolation

  if [ ! -d "CuMesh" ]; then
    git clone https://github.com/JeffreyXiang/CuMesh.git CuMesh --recursive
    git -C CuMesh checkout --detach "${CUMESH_COMMIT:-12289e1062f0603f2f0d0771b02e1395d247f26f}"
    git -C CuMesh submodule update --init --recursive
  fi
  pip install CuMesh/ --no-build-isolation

  if [ ! -d "FlexGEMM" ]; then
    git clone https://github.com/JeffreyXiang/FlexGEMM.git FlexGEMM --recursive
    git -C FlexGEMM checkout --detach "${FLEXGEMM_COMMIT:-6dd94a859c26ee8246888502eada3dd8ad85532e}"
    git -C FlexGEMM submodule update --init --recursive
  fi
  pip install FlexGEMM/ --no-build-isolation

  if [ ! -d "o-voxel" ]; then
    cp -r ../o-voxel o-voxel
  fi
  pip install o-voxel/ --no-build-isolation
  
  echo "Installed TRELLIS.2 and all extensions"

  cd ../../ # back to deps directory
fi

# step 2.16: install openpi
# TODO(SimFoundry): ship later with training code.
# OPENPI_COMMIT="${OPENPI_COMMIT:-15a9616a00943ada6c20a0f158e3adb39df2ccac}"
# if [ ! -d "openpi" ]; then
# git clone https://github.com/Physical-Intelligence/openpi.git
# git -C openpi checkout --detach "${OPENPI_COMMIT}"
# fi
# cd openpi
# pip install -e packages/openpi-client/ > /dev/null
# echo "Installed OpenPI client"
# cd .. # back to deps directory

# step 2.17: install LeRobot
mamba install --freeze-installed ffmpeg=7.1.1 -c conda-forge -y
pip install --no-deps lerobot@git+https://github.com/huggingface/lerobot.git@577cd10974b84bea1f06b6472eb9e5e74e07f77a
pip install \
  "datasets>=2.19.0,<3" \
  jsonlines \
  "av>=14.2.0" \
  cmake \
  "deepdiff>=7.0.1,<9" \
  draccus==0.10.0 \
  pynput \
  pyserial \
  "rerun-sdk>=0.21.0,<0.23.0"

# Finally, misc dependencies
pip install zmq
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
rm -rf "${SITE_PACKAGES}"/numpy "${SITE_PACKAGES}"/numpy.libs "${SITE_PACKAGES}"/numpy-*.dist-info
pip install --no-cache-dir "numpy==1.26.4" "coverage==7.6.1" "typing_extensions>=4.15.0" "psutil==5.9.8"

copy_robot_asset_from_fallback() {
  local rel_path="$1"
  local src="${ROBOT_ASSET_FALLBACK_ROOT}/deps/BEHAVIOR-1K/datasets/omnigibson-robot-assets/${rel_path}"
  local dst="${ROBOT_ASSETS_DIR}/${rel_path}"
  if [[ -n "${ROBOT_ASSET_FALLBACK_ROOT}" && -e "${src}" && ! -e "${dst}" ]]; then
    echo "Copying optional robot asset from fallback: ${rel_path}"
    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
    return 0
  fi
  return 1
}

validate_robot_asset_file() {
  local rel_path="$1"
  local required="$2"
  local fallback_rel_path="${3:-${rel_path}}"
  if [[ -f "${ROBOT_ASSETS_DIR}/${rel_path}" ]]; then
    return 0
  fi
  if copy_robot_asset_from_fallback "${fallback_rel_path}"; then
    [[ -f "${ROBOT_ASSETS_DIR}/${rel_path}" ]] && return 0
  fi
  if [[ "${required}" == "required" ]]; then
    echo "ERROR: Required OmniGibson robot asset is missing: ${ROBOT_ASSETS_DIR}/${rel_path}" >&2
    echo "All required assets come from the public behavior-1k datasets on Hugging Face." >&2
    echo "Check network access to huggingface.co, then re-run install.sh." >&2
    exit 1
  fi
  echo "WARNING: Optional OmniGibson robot asset is unavailable: ${rel_path}" >&2
  echo "         Pass --robot-asset-fallback-root <repo> if you have a local copy." >&2
}

export OMNI_KIT_ACCEPT_EULA=YES
echo "Ensuring OmniGibson robot assets are installed..."
python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"
ROBOT_ASSETS_DIR="${PROJECT_ROOT}/deps/BEHAVIOR-1K/datasets/omnigibson-robot-assets"

# The franka_robotiq end effector, which most task configs use. It lives in the public
# behavior-1k/omnigibson-robot-assets dataset repo, but NOT in the pre-built
# omnigibson-robot-assets.zip that download_omnigibson_robot_assets() unpacks -- that zip is an
# older snapshot, taken before franka_robotiq was added to the dataset. So fetch that one
# subtree directly.
#
# Must run AFTER download_omnigibson_robot_assets(), for two reasons: unpacking the zip would
# overwrite these files, and download_omnigibson_robot_assets() skips entirely when its target
# directory already exists, so creating that directory first would silently drop the rest of
# the bundle.
OG_ROBOT_ASSETS_HF_REPO="${OG_ROBOT_ASSETS_HF_REPO:-behavior-1k/omnigibson-robot-assets}"
FRANKA_ROBOTIQ_REL="models/franka/franka_robotiq"

fetch_franka_robotiq_assets() {
  if [[ -f "${ROBOT_ASSETS_DIR}/${FRANKA_ROBOTIQ_REL}/usd/franka_robotiq.usda" ]]; then
    echo "franka_robotiq robot assets already present; skipping download."
    return 0
  fi
  echo "Fetching franka_robotiq robot assets from ${OG_ROBOT_ASSETS_HF_REPO}..."
  mkdir -p "${ROBOT_ASSETS_DIR}"
  python - "${ROBOT_ASSETS_DIR}" "${OG_ROBOT_ASSETS_HF_REPO}" "${FRANKA_ROBOTIQ_REL}" <<'PY'
import sys

from huggingface_hub import snapshot_download

local_dir, repo_id, rel_path = sys.argv[1], sys.argv[2], sys.argv[3]
# Public dataset, so no token is needed. allow_patterns keeps this to ~225 MB rather than
# pulling the multi-GB repo, and lands the files at <local_dir>/models/franka/franka_robotiq/
# because the dataset uses the same layout as the robot-assets tree.
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=[f"{rel_path}/**"],
    local_dir=local_dir,
)
PY
}

if ! fetch_franka_robotiq_assets; then
  echo "WARNING: could not fetch franka_robotiq robot assets; falling back to --robot-asset-fallback-root." >&2
fi

YAMLAB_REPO_COMMIT="ec0455d2b4ce35f21fc126418ea5e74ac567133d"
YAMLAB_YAM_USD_URL="https://raw.githubusercontent.com/ARISE-Initiative/yamlab/${YAMLAB_REPO_COMMIT}/yamlab/robot/yam/arm/yam.usd"
YAM_REL="models/yam/usd/yam.usda"

fetch_yam_assets() {
  if [[ -f "${ROBOT_ASSETS_DIR}/${YAM_REL}" ]]; then
    echo "yam robot asset already present; skipping download."
    return 0
  fi
  echo "Fetching yam robot asset from ${YAMLAB_YAM_USD_URL}..."
  mkdir -p "${ROBOT_ASSETS_DIR}/models/yam/usd"
  local tmp="${ROBOT_ASSETS_DIR}/models/yam/usd/.yam_crate_download.usd"
  curl -fsSL -o "${tmp}" "${YAMLAB_YAM_USD_URL}" || { rm -f "${tmp}"; return 1; }
  python - "${tmp}" "${ROBOT_ASSETS_DIR}/${YAM_REL}" <<'PY'
import sys
from pxr import Sdf

src, dst = sys.argv[1], sys.argv[2]
layer = Sdf.Layer.FindOrOpen(src)
if layer is None:
    sys.exit(f"could not open downloaded yam.usd: {src}")
layer.documentation = ""  # strip the "Generated from ... /home/<user>/..." breadcrumb
if not layer.Export(dst):
    sys.exit(f"could not write {dst}")
PY
  local status=$?
  rm -f "${tmp}"
  return $status
}

if ! fetch_yam_assets; then
  echo "WARNING: could not fetch yam robot asset; falling back to --robot-asset-fallback-root." >&2
fi

validate_robot_asset_file "models/franka/franka_panda/usd/franka_panda.usda" required "models/franka/franka_panda"
validate_robot_asset_file "models/background/sky.jpg" required "models/background/sky.jpg"
validate_robot_asset_file "models/franka/franka_robotiq/usd/franka_robotiq.usda" required "models/franka/franka_robotiq"
validate_robot_asset_file "${YAM_REL}" optional "models/yam"

install_faiss_gpu "$ENV_NAME"

echo "Completed installation of SimFoundry environment: $ENV_NAME"

mamba deactivate
