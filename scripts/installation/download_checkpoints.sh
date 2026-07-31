#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

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
repo_dir="$(cd "$SCRIPT_DIR/../.." && pwd)"
env_name="simfoundry"
DEFAULT=false
CHECKPOINT_FALLBACK_ROOT="${SIMFOUNDRY_CHECKPOINT_FALLBACK_ROOT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) repo_dir="$2"; shift 2 ;;
    --env-name) env_name="$2"; shift 2 ;;
    --checkpoint-fallback-root) CHECKPOINT_FALLBACK_ROOT="$2"; shift 2 ;;
    --default) DEFAULT=true; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/installation/download_checkpoints.sh [options]

Options:
  --project-root DIR
  --env-name NAME
  --checkpoint-fallback-root DIR   Optional repo root to copy checkpoints from
                                  if a network download fails.
  --default                        Do not prompt.
EOF
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ "${DEFAULT}" != true ]]; then
  read -p "Enter project root (default: ${repo_dir}): " REPO_DIR
  repo_dir="${REPO_DIR:-${repo_dir}}"
  read -p "Enter environment name to use for downloads (default: ${env_name}): " ENV_NAME
  env_name="${ENV_NAME:-${env_name}}"
fi

REPO_DIR="$(cd "${repo_dir}" && pwd)"
cd "${REPO_DIR}"
# conda/cuda-nvcc activate hooks are not `set -u` safe (the cuda-nvcc hook does
# NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS} ..." with no default -> "unbound variable"
# under nounset). Relax -u just for activation, then restore it.
set +u
mamba activate "$env_name"
set -u

if ! command -v gdown >/dev/null 2>&1; then
  python -m pip install gdown
fi

mkdir -p checkpoints
CKPT_DIR="${REPO_DIR}/checkpoints"
cd "${CKPT_DIR}"

copy_dir_from_fallback() {
  local rel_dir="$1"
  local dest_dir="${REPO_DIR}/${rel_dir}"
  if [[ -z "${CHECKPOINT_FALLBACK_ROOT}" ]]; then
    return 1
  fi
  local src_dir="${CHECKPOINT_FALLBACK_ROOT}/${rel_dir}"
  if [[ ! -d "${src_dir}" ]]; then
    return 1
  fi
  echo "Copying checkpoint directory from fallback: ${src_dir} -> ${dest_dir}"
  mkdir -p "${dest_dir}"
  cp -a "${src_dir}/." "${dest_dir}/"
}

copy_file_from_fallback() {
  local rel_file="$1"
  local dest_file="${REPO_DIR}/${rel_file}"
  if [[ -z "${CHECKPOINT_FALLBACK_ROOT}" ]]; then
    return 1
  fi
  local src_file="${CHECKPOINT_FALLBACK_ROOT}/${rel_file}"
  if [[ ! -f "${src_file}" ]]; then
    return 1
  fi
  echo "Copying checkpoint file from fallback: ${src_file} -> ${dest_file}"
  mkdir -p "$(dirname "${dest_file}")"
  cp -a "${src_file}" "${dest_file}"
}

run_download_or_fallback_dir() {
  local desc="$1"
  local marker="$2"
  local rel_dir="$3"
  shift 3
  if [[ -f "${marker}" ]]; then
    return
  fi
  echo "Downloading ${desc}..."
  mkdir -p "${REPO_DIR}/${rel_dir}"
  set +e
  "$@"
  local status=$?
  set -e
  if [[ ${status} -ne 0 || ! -f "${marker}" ]]; then
    echo "WARNING: Download failed or marker missing for ${desc}."
    if ! copy_dir_from_fallback "${rel_dir}"; then
      echo "ERROR: Could not download ${desc} and no fallback copy was available." >&2
      return 1
    fi
  fi
}

run_download_or_fallback_file() {
  local desc="$1"
  local rel_file="$2"
  local url="$3"
  local dest_file="${REPO_DIR}/${rel_file}"
  if [[ -f "${dest_file}" ]]; then
    return
  fi
  echo "Downloading ${desc}..."
  mkdir -p "$(dirname "${dest_file}")"
  set +e
  wget -O "${dest_file}" "${url}"
  local status=$?
  set -e
  if [[ ${status} -ne 0 || ! -f "${dest_file}" ]]; then
    rm -f "${dest_file}"
    echo "WARNING: Download failed for ${desc}."
    if ! copy_file_from_fallback "${rel_file}"; then
      echo "ERROR: Could not download ${desc} and no fallback copy was available." >&2
      return 1
    fi
  fi
}

# FoundationStereo checkpoint
FS_PATH="${REPO_DIR}/deps/FoundationStereo/pretrained_models/23-51-11"
if [[ ! -f "${FS_PATH}/cfg.yaml" ]]; then
  run_download_or_fallback_dir \
    "FoundationStereo checkpoint" \
    "${FS_PATH}/cfg.yaml" \
    "deps/FoundationStereo/pretrained_models/23-51-11" \
    gdown --folder 'https://drive.google.com/drive/folders/1BbhoPliFqPJlrtD65TgNX49sJYuYcwA-?usp=drive_link' -O "${FS_PATH}"
fi

# FoundationPose checkpoint
FP_REFINER_PATH="${REPO_DIR}/deps/FoundationPose/weights/2023-10-28-18-33-37"
if [[ ! -f "${FP_REFINER_PATH}/model_best.pth" ]]; then
  run_download_or_fallback_dir \
    "FoundationPose Refiner checkpoint" \
    "${FP_REFINER_PATH}/model_best.pth" \
    "deps/FoundationPose/weights/2023-10-28-18-33-37" \
    gdown --folder 'https://drive.google.com/drive/folders/1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC' -O "${FP_REFINER_PATH}"
fi
FP_SCORER_PATH="${REPO_DIR}/deps/FoundationPose/weights/2024-01-11-20-02-45"
if [[ ! -f "${FP_SCORER_PATH}/model_best.pth" ]]; then
  run_download_or_fallback_dir \
    "FoundationPose Scorer checkpoint" \
    "${FP_SCORER_PATH}/model_best.pth" \
    "deps/FoundationPose/weights/2024-01-11-20-02-45" \
    gdown --folder 'https://drive.google.com/drive/folders/12Te_3TELLes5cim1d7F7EBTwUSe7iRBj' -O "${FP_SCORER_PATH}"
fi

# SAM2.1 checkpoint
if [[ ! -f "${REPO_DIR}/checkpoints/sam2.1_hiera_large.pt" ]]; then
  run_download_or_fallback_file \
    "SAM2.1 checkpoint" \
    "checkpoints/sam2.1_hiera_large.pt" \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
fi

# DepthPro checkpoint
DP_PATH="${REPO_DIR}/deps/ml-depth-pro/checkpoints"
if [[ ! -f "${DP_PATH}/depth_pro.pt" ]]; then
  run_download_or_fallback_file \
    "DepthPro checkpoint" \
    "deps/ml-depth-pro/checkpoints/depth_pro.pt" \
    "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
fi

# Hunyuan checkpoint
HY_PATH="${REPO_DIR}/deps/Hunyuan3D-2.1/ckpt"
if [[ ! -f "${HY_PATH}/RealESRGAN_x4plus.pth" ]]; then
  run_download_or_fallback_file \
    "RealESRGAN_x4plus checkpoint" \
    "deps/Hunyuan3D-2.1/ckpt/RealESRGAN_x4plus.pth" \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
fi

# VOID inpainting weights (auto_bg VOID passes 1+2; ~15 GB total). Downloaded via the
# Hugging Face CLI: the CogVideoX-Fun base model + the two VOID pass transformers.
# netflix/void-model is gated — run `huggingface-cli login` first.
# Must land at the deps/void-model ROOT (not a checkpoints/ subdir): the runner resolves
# model_name/model_checkpoint against VOID_ROOT=deps/void-model (2_run_void_pass1.py),
# and auto_bg.yaml uses root-relative names (e.g. void_pass1.safetensors).
VOID_CKPT_DIR="${REPO_DIR}/deps/void-model"
HF_BIN="$(command -v hf || command -v huggingface-cli || true)"
download_void_hf() {
  local desc="$1" rel_marker="$2"
  shift 2
  if [[ -f "${REPO_DIR}/${rel_marker}" ]]; then
    return
  fi
  if [[ -z "${HF_BIN}" ]]; then
    echo "WARNING: neither 'hf' nor 'huggingface-cli' on PATH for ${desc}." >&2
  else
    echo "Downloading ${desc}..."
    set +e
    "${HF_BIN}" "$@"
    set -e
  fi
  if [[ ! -f "${REPO_DIR}/${rel_marker}" ]]; then
    echo "WARNING: Download failed or marker missing for ${desc}."
    if ! copy_file_from_fallback "${rel_marker}"; then
      echo "ERROR: Could not download ${desc} and no fallback copy was available." >&2
      return 1
    fi
  fi
}
download_void_hf \
  "CogVideoX-Fun-V1.5-5b-InP (VOID base model)" \
  "deps/void-model/CogVideoX-Fun-V1.5-5b-InP/transformer/config.json" \
  download alibaba-pai/CogVideoX-Fun-V1.5-5b-InP --local-dir "${VOID_CKPT_DIR}/CogVideoX-Fun-V1.5-5b-InP"
download_void_hf \
  "VOID Pass 1 transformer" \
  "deps/void-model/void_pass1.safetensors" \
  download netflix/void-model void_pass1.safetensors --local-dir "${VOID_CKPT_DIR}"
download_void_hf \
  "VOID Pass 2 transformer" \
  "deps/void-model/void_pass2.safetensors" \
  download netflix/void-model void_pass2.safetensors --local-dir "${VOID_CKPT_DIR}"

# # Any6D checkpoint
# FP_REFINER_PATH="${REPO_DIR}/deps/Any6D/foundationpose/weights/2023-10-28-18-33-37"
# if [[ ! -f "${FP_REFINER_PATH}/model_best.pth" ]]; then
#   echo "Downloading FoundationPose Refiner checkpoint..."
#   mkdir -p ${FP_REFINER_PATH}
#   gdown --folder 'https://drive.google.com/drive/folders/1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC' -O ${FP_REFINER_PATH}
# fi
# FP_SCORER_PATH="${REPO_DIR}/deps/Any6D/foundationpose/weights/2024-01-11-20-02-45"
# if [[ ! -f "${FP_SCORER_PATH}/model_best.pth" ]]; then
#   echo "Downloading FoundationPose Scorer checkpoint..."
#   mkdir -p ${FP_SCORER_PATH}
#   gdown --folder 'https://drive.google.com/drive/folders/12Te_3TELLes5cim1d7F7EBTwUSe7iRBj' -O ${FP_SCORER_PATH}
# fi
