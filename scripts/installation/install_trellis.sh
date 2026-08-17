#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# install_trellis.sh — install TRELLIS.2 (microsoft/TRELLIS.2) and all of its CUDA
# extensions into the SimFoundry env, so stage 7 can run with
# `s7_mesh.shape_model=trellis2 s7_mesh.texture_model=trellis2`.
#
# TRELLIS.2 is NOT installed by install_everything.sh — it is opt-in. install_simfoundry.sh
# installs it only when passed --trellis; this script installs it into an env that already
# exists, without rebuilding the whole SimFoundry env.
#
# Behavior:
#   * If the target env exists  -> install TRELLIS.2 into it.
#   * If it does NOT exist      -> say so, ask for confirmation, then build the full
#                                  SimFoundry env via install_simfoundry.sh --trellis
#                                  (which installs TRELLIS.2 as part of that run).
#
# Installed into the env:
#   TRELLIS.2 (conda-develop, i.e. added to the env's path) + extensions:
#     nvdiffrast, nvdiffrec, CuMesh, FlexGEMM, o-voxel
#
# Usage:
#   bash scripts/installation/install_trellis.sh [--project-root DIR] [--env-name NAME]
#                                                [--yes] [--default]
#                                                [--cuda-version X.Y] [--cuda-arch-list LIST]
#
#   --project-root DIR     Repo root; TRELLIS.2 is cloned to <root>/deps. Default: repo root.
#   --env-name NAME        Target mamba env. Default: simfoundry
#   --yes, -y              Assume "yes" for the create-the-env prompt.
#   --default              Fully non-interactive (implies --yes).
#   --cuda-version X.Y     CUDA toolkit version, used only when creating the env. Default: 12.8
#   --cuda-arch-list LIST  TORCH_CUDA_ARCH_LIST override, e.g. "8.9;12.0". Default: auto-detect.
#
# Prereqs: mamba (Miniforge) on PATH, and libjpeg-dev (`sudo apt install -y libjpeg-dev`)
# which pillow-simd needs to build. Model weights (microsoft/TRELLIS.2-4B) are NOT downloaded
# here — they are fetched at runtime by from_pretrained on the first stage 7 run.
#

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

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
project_root="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_NAME="simfoundry"
DEFAULT=false
ASSUME_YES=false
CUDA_VERSION="12.8"
CUDA_ARCH_LIST=""

usage() { sed -n '4,40p' "$0"; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --project-root)   project_root="$2"; shift 2 ;;
        --env-name)       ENV_NAME="$2"; shift 2 ;;
        --yes|-y)         ASSUME_YES=true; shift ;;
        --default)        DEFAULT=true; ASSUME_YES=true; shift ;;
        --cuda-version)   CUDA_VERSION="$2"; shift 2 ;;
        --cuda-arch-list) CUDA_ARCH_LIST="$2"; shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done
PROJECT_ROOT="$(cd "$project_root" && pwd)"

# Pinned upstream SHAs. Keep these in sync with the --trellis block in install_simfoundry.sh.
TRELLIS2_COMMIT="${TRELLIS2_COMMIT:-75fbf0183001ed9876c8dbb35de6b68552ee08bd}"
NVDIFFRAST_COMMIT="${NVDIFFRAST_COMMIT:-253ac4fcea7de5f396371124af597e6cc957bfae}"
NVDIFFREC_COMMIT="${NVDIFFREC_COMMIT:-b296927cc7fd01c2ac1087c8065c4d7248f72da4}"
CUMESH_COMMIT="${CUMESH_COMMIT:-12289e1062f0603f2f0d0771b02e1395d247f26f}"
FLEXGEMM_COMMIT="${FLEXGEMM_COMMIT:-6dd94a859c26ee8246888502eada3dd8ad85532e}"
UTILS3D_COMMIT="${UTILS3D_COMMIT:-9a4eb15e4021b67b12c460c7057d642626897ec8}"

env_exists() {
  mamba env list | awk '{print $1}' | grep -Fxq "$1"
}

# Prompt unless running non-interactively / with --yes. Defaults to "no" on empty input.
confirm() {
  local prompt="$1"
  if [[ ${ASSUME_YES} == true ]]; then
    echo "${prompt} [auto-yes]"
    return 0
  fi
  if [[ ! -t 0 ]]; then
    echo "Not running interactively and --yes/--default was not passed; aborting." >&2
    return 1
  fi
  local reply
  read -r -p "${prompt} [y/N]: " reply
  [[ "${reply}" =~ ^[Yy]([Ee][Ss])?$ ]]
}

echo "============================================================"
echo "install_trellis.sh"
echo "  project_root:  ${PROJECT_ROOT}"
echo "  target env:    ${ENV_NAME}"
echo "============================================================"

# pillow-simd builds against libjpeg headers. Check before any cloning/building so the
# failure is immediate and actionable rather than 20 minutes into the extension builds.
if ! dpkg -s libjpeg-dev &> /dev/null; then
  echo "ERROR: libjpeg-dev is not installed but is required for TRELLIS.2 (pillow-simd)." >&2
  echo "Please install it manually before running this script:" >&2
  echo "  sudo apt install -y libjpeg-dev" >&2
  exit 1
fi

# ==============================================================================
# ENV EXISTENCE CHECK
# ==============================================================================
if ! env_exists "${ENV_NAME}"; then
  echo ""
  echo "NOTE: mamba env '${ENV_NAME}' does not exist."
  echo ""
  echo "TRELLIS.2 installs *into* the SimFoundry env (it is not a standalone env), so the"
  echo "env has to be built first. Proceeding will run:"
  echo ""
  echo "  install_simfoundry.sh --project-root ${PROJECT_ROOT} --env-name ${ENV_NAME} \\"
  echo "                        --cuda-version ${CUDA_VERSION} --trellis"
  echo ""
  echo "That builds the full SimFoundry env (PyTorch, BEHAVIOR-1K, SAM3, pytorch3d, ...)"
  echo "with TRELLIS.2 included. It takes a long time and downloads several GB."
  echo ""
  if ! confirm "Create env '${ENV_NAME}' now?"; then
    echo "Aborted. Create the env first, e.g.:"
    echo "  bash ${SCRIPT_DIR}/install_simfoundry.sh --trellis"
    exit 1
  fi

  SIMFOUNDRY_ARGS=(
    --project-root "${PROJECT_ROOT}"
    --env-name "${ENV_NAME}"
    --cuda-version "${CUDA_VERSION}"
    --trellis
    --default
  )
  if [[ -n "${CUDA_ARCH_LIST}" ]]; then
    SIMFOUNDRY_ARGS+=(--cuda-arch-list "${CUDA_ARCH_LIST}")
  fi
  # --default is passed so the values collected here are used verbatim instead of
  # re-prompting for project root / env name / CUDA version.
  bash "${SCRIPT_DIR}/install_simfoundry.sh" "${SIMFOUNDRY_ARGS[@]}"

  echo ""
  echo "============================================================"
  echo "install_trellis.sh: DONE (env '${ENV_NAME}' created with TRELLIS.2)"
  echo "============================================================"
  exit 0
fi

echo "Found existing env '${ENV_NAME}'."

# TRELLIS.2 pins transformers/gradio inside the shared SimFoundry env, so flag it before
# touching an env the rest of the pipeline depends on.
echo ""
echo "WARNING: this installs into the shared '${ENV_NAME}' env and pins:"
echo "           transformers==4.57.6, gradio==6.0.1, pillow-simd (replaces pillow)"
echo "         Other SimFoundry stages run in this same env."
if ! confirm "Continue installing TRELLIS.2 into '${ENV_NAME}'?"; then
  echo "Aborted."
  exit 1
fi

mamba activate "${ENV_NAME}"

# ==============================================================================
# BUILD TOOLCHAIN (needed by the CUDA extensions)
# ==============================================================================
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-${CUDA_VERSION}}"
export LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH:-}"

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
ensure_cuda_toolkit

# nvdiffrast / CuMesh / FlexGEMM / o-voxel compile device code, so the arch list must cover
# the GPUs this env will actually run on.
if [[ -n "${CUDA_ARCH_LIST}" ]]; then
  export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}"
  echo "Using provided TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
else
  echo "Auto-detecting GPU compute capabilities..."
  if command -v nvidia-smi &> /dev/null; then
    DETECTED_ARCHS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u -V | tr '\n' ';' | sed 's/;$//')
    if [[ -n "${DETECTED_ARCHS}" ]]; then
      export TORCH_CUDA_ARCH_LIST="${DETECTED_ARCHS}"
      echo "Auto-detected TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
    else
      echo "WARNING: nvidia-smi found but no GPU compute capabilities detected. TORCH_CUDA_ARCH_LIST will not be set."
    fi
  else
    echo "WARNING: nvidia-smi not found. TORCH_CUDA_ARCH_LIST will not be set."
  fi
fi

# conda-develop is what puts the TRELLIS.2 checkout on the env's import path. The full
# install_simfoundry.sh run installs conda-build, but an env built another way may not have it.
if ! command -v conda-develop >/dev/null 2>&1; then
  echo "conda-develop not found in '${ENV_NAME}'; installing conda-build..."
  mamba install conda-build -y > /dev/null
fi

# torch must already be present: every extension below builds against it.
if ! python -c 'import torch' >/dev/null 2>&1; then
  echo "ERROR: torch is not importable in env '${ENV_NAME}'." >&2
  echo "       The TRELLIS.2 extensions build against torch. Build the env with" >&2
  echo "       install_simfoundry.sh first, then rerun this script." >&2
  exit 1
fi

# ==============================================================================
# TRELLIS.2
# ==============================================================================
if [ ! -d "${PROJECT_ROOT}/deps" ]; then
  mkdir "${PROJECT_ROOT}/deps"
fi
cd "${PROJECT_ROOT}/deps"

if [ ! -d "TRELLIS.2" ]; then
  git clone https://github.com/microsoft/TRELLIS.2.git --recursive
  git -C TRELLIS.2 checkout --detach "${TRELLIS2_COMMIT}"
  git -C TRELLIS.2 submodule update --init --recursive
fi
cd TRELLIS.2

pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers==4.57.6 gradio==6.0.1 tensorboard pandas lpips zstandard utils3d
pip install "git+https://github.com/EasternJournalist/utils3d.git@${UTILS3D_COMMIT}"

# >=3.3.1 for sm_120 (Blackwell) tl.dot support; <3.4 for torch 2.7.0 inductor compatibility.
pip install "triton>=3.3.1,<3.4"
pip install pillow-simd
pip install kornia timm psutil
conda-develop .

# Extensions
if [ ! -d "extensions" ]; then
  mkdir extensions
fi
cd extensions

if [ ! -d "nvdiffrast" ]; then
  git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git nvdiffrast
  git -C nvdiffrast checkout --detach "${NVDIFFRAST_COMMIT}"
fi
pip install nvdiffrast/ --no-build-isolation

if [ ! -d "nvdiffrec" ]; then
  git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git nvdiffrec
  git -C nvdiffrec checkout --detach "${NVDIFFREC_COMMIT}"
fi
pip install nvdiffrec/ --no-build-isolation

if [ ! -d "CuMesh" ]; then
  git clone https://github.com/JeffreyXiang/CuMesh.git CuMesh --recursive
  git -C CuMesh checkout --detach "${CUMESH_COMMIT}"
  git -C CuMesh submodule update --init --recursive
fi
pip install CuMesh/ --no-build-isolation

if [ ! -d "FlexGEMM" ]; then
  git clone https://github.com/JeffreyXiang/FlexGEMM.git FlexGEMM --recursive
  git -C FlexGEMM checkout --detach "${FLEXGEMM_COMMIT}"
  git -C FlexGEMM submodule update --init --recursive
fi
pip install FlexGEMM/ --no-build-isolation

# o-voxel ships inside the TRELLIS.2 checkout rather than as a separate remote.
if [ ! -d "o-voxel" ]; then
  cp -r ../o-voxel o-voxel
fi
pip install o-voxel/ --no-build-isolation

echo "Installed TRELLIS.2 and all extensions"

# ==============================================================================
# VERIFY
# ==============================================================================
# Import from a neutral cwd so a stray ./trellis2 dir can't shadow the installed package.
cd "${PROJECT_ROOT}"
VERIFY_OK=true
if python -c 'import o_voxel; import trellis2; print("Verified o_voxel and trellis2 imports")'; then
  :
else
  VERIFY_OK=false
fi

echo ""
echo "============================================================"
if [[ ${VERIFY_OK} == true ]]; then
  echo "install_trellis.sh: DONE"
else
  echo "install_trellis.sh: FINISHED WITH WARNINGS"
  echo "  The import check failed. Some TRELLIS.2 imports pull in CUDA-dependent"
  echo "  modules, so this can fail on a machine with no visible GPU even when the"
  echo "  install is fine. Verify on a GPU host with:"
  echo "    mamba run -n ${ENV_NAME} python -c 'import trellis2, o_voxel'"
fi
echo ""
echo "  TRELLIS.2 checkout: ${PROJECT_ROOT}/deps/TRELLIS.2"
echo "  Weights (microsoft/TRELLIS.2-4B) download on first use."
echo ""
echo "  Run stage 7 with TRELLIS.2 — note --env-mesh points the mesh-generation env slot at"
echo "  '${ENV_NAME}', since that is where TRELLIS.2 now lives:"
echo ""
echo "    scripts/pipeline/A_reconstruction/run.sh --env-mesh ${ENV_NAME} \\"
echo "      -- s7_mesh.shape_model=trellis2 s7_mesh.texture_model=trellis2"
echo "============================================================"
