#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT_ROOT="${CDC_PROJECT_ROOT:-${SCRIPT_DIR}}"
ENV_SUFFIX="${CDC_ENV_SUFFIX:-}"
CUDA_VERSION="${CDC_CUDA_VERSION:-12.8}"
CUDA_ARCH_LIST="${CDC_CUDA_ARCH_LIST:-}"

ENV_CDC="${CDC_ENV_CDC:-cdc${ENV_SUFFIX}}"
ENV_HUNYUAN="${CDC_ENV_HUNYUAN:-hunyuan${ENV_SUFFIX}}"
ENV_DA3="${CDC_ENV_DA3:-da3${ENV_SUFFIX}}"
RECREATE_ENVS="${CDC_RECREATE_ENVS:-0}"
CHECKPOINT_FALLBACK_ROOT="${CDC_CHECKPOINT_FALLBACK_ROOT:-}"
ROBOT_ASSET_FALLBACK_ROOT="${CDC_ROBOT_ASSET_FALLBACK_ROOT:-}"

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

Installs the environments and checkpoints needed to run CDC A/B/C pipelines.

Environment variables:
  CDC_ENV_SUFFIX=-test              Suffix default env names, e.g. cdc-test.
  CDC_ENV_CDC=NAME                  Override CDC/OmniGibson env name.
  CDC_ENV_HUNYUAN=NAME              Override Hunyuan env name.
  CDC_ENV_DA3=NAME                  Override Depth Anything 3 env name.
  CDC_CUDA_VERSION=12.8             CUDA toolkit version to install/use.
  CDC_CUDA_ARCH_LIST=12.0           Optional TORCH_CUDA_ARCH_LIST override.
  CDC_RECREATE_ENVS=1               Remove target envs before installation.
  CDC_CHECKPOINT_FALLBACK_ROOT=DIR   Optional local repo root to copy checkpoints from
                                    if Google Drive downloads fail.
  CDC_ROBOT_ASSET_FALLBACK_ROOT=DIR  Optional local repo root to copy CDC-specific
                                    OmniGibson robot assets from.

Options mirror the environment variables above:
  --env-suffix SUFFIX
  --env-cdc NAME
  --env-hunyuan NAME
  --env-da3 NAME
  --cuda-version VERSION
  --cuda-arch-list LIST
  --recreate-envs
  --checkpoint-fallback-root DIR
  --robot-asset-fallback-root DIR
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-suffix) ENV_SUFFIX="$2"; shift 2 ;;
    --env-cdc) ENV_CDC="$2"; shift 2 ;;
    --env-hunyuan) ENV_HUNYUAN="$2"; shift 2 ;;
    --env-da3) ENV_DA3="$2"; shift 2 ;;
    --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
    --cuda-arch-list) CUDA_ARCH_LIST="$2"; shift 2 ;;
    --recreate-envs) RECREATE_ENVS=1; shift ;;
    --checkpoint-fallback-root) CHECKPOINT_FALLBACK_ROOT="$2"; shift 2 ;;
    --robot-asset-fallback-root) ROBOT_ASSET_FALLBACK_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Recompute defaults if --env-suffix was provided without explicit env names.
if [[ -z "${CDC_ENV_CDC:-}" && "${ENV_CDC}" == "cdc${CDC_ENV_SUFFIX:-}" ]]; then
  ENV_CDC="cdc${ENV_SUFFIX}"
fi
if [[ -z "${CDC_ENV_HUNYUAN:-}" && "${ENV_HUNYUAN}" == "hunyuan${CDC_ENV_SUFFIX:-}" ]]; then
  ENV_HUNYUAN="hunyuan${ENV_SUFFIX}"
fi
if [[ -z "${CDC_ENV_DA3:-}" && "${ENV_DA3}" == "da3${CDC_ENV_SUFFIX:-}" ]]; then
  ENV_DA3="da3${ENV_SUFFIX}"
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: Required command not found: $1" >&2
    exit 1
  fi
}

env_exists() {
  mamba env list | awk '{print $1}' | grep -qx "$1"
}

remove_env_if_needed() {
  local env_name="$1"
  if env_exists "$env_name"; then
    if [[ "${RECREATE_ENVS}" == "1" ]]; then
      echo "[install] Removing existing mamba env: ${env_name}"
      mamba env remove -y -n "$env_name"
    else
      echo "ERROR: Mamba env already exists: ${env_name}" >&2
      echo "Set CDC_RECREATE_ENVS=1 or pass --recreate-envs to remove it first." >&2
      exit 1
    fi
  fi
}

echo "[install] Project root: ${PROJECT_ROOT}"
echo "[install] Environments: CDC=${ENV_CDC}, Hunyuan=${ENV_HUNYUAN}, DA3=${ENV_DA3}"
echo "[install] CUDA version: ${CUDA_VERSION}"
if [[ -n "${CUDA_ARCH_LIST}" ]]; then
  echo "[install] TORCH_CUDA_ARCH_LIST override: ${CUDA_ARCH_LIST}"
fi

require_cmd git
require_cmd mamba
require_cmd python
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found. GPU-dependent install or E2E stages may fail." >&2
else
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi

available_gb="$(df -Pk "${PROJECT_ROOT}" | awk 'NR==2 {printf "%.1f", $4 / 1024 / 1024}')"
echo "[install] Available disk at project root: ${available_gb} GiB"

remove_env_if_needed "${ENV_CDC}"
remove_env_if_needed "${ENV_HUNYUAN}"
remove_env_if_needed "${ENV_DA3}"

INSTALL_ARGS=(--project-root "${PROJECT_ROOT}" --default)
if [[ -n "${CUDA_ARCH_LIST}" ]]; then
  INSTALL_ARGS+=(--cuda-arch-list "${CUDA_ARCH_LIST}")
fi

echo "[install] Installing CDC / OmniGibson env: ${ENV_CDC}"
CDC_INSTALL_ARGS=("${INSTALL_ARGS[@]}" --env-name "${ENV_CDC}" --cuda-version "${CUDA_VERSION}")
if [[ -n "${ROBOT_ASSET_FALLBACK_ROOT}" ]]; then
  CDC_INSTALL_ARGS+=(--robot-asset-fallback-root "${ROBOT_ASSET_FALLBACK_ROOT}")
fi
bash "${PROJECT_ROOT}/scripts/installation/install_cdc.sh" \
  "${CDC_INSTALL_ARGS[@]}"

echo "[install] Installing Hunyuan env: ${ENV_HUNYUAN}"
bash "${PROJECT_ROOT}/scripts/installation/install_hunyuan.sh" \
  "${INSTALL_ARGS[@]}" \
  --env-name "${ENV_HUNYUAN}" \
  --cuda-version "${CUDA_VERSION}"

echo "[install] Installing Depth Anything 3 env: ${ENV_DA3}"
bash "${PROJECT_ROOT}/scripts/installation/install_da3.sh" \
  "${INSTALL_ARGS[@]}" \
  --env-name "${ENV_DA3}"

CHECKPOINT_ARGS=(--project-root "${PROJECT_ROOT}" --env-name "${ENV_CDC}" --default)
if [[ -n "${CHECKPOINT_FALLBACK_ROOT}" ]]; then
  CHECKPOINT_ARGS+=(--checkpoint-fallback-root "${CHECKPOINT_FALLBACK_ROOT}")
fi

echo "[install] Downloading checkpoints"
bash "${PROJECT_ROOT}/scripts/installation/download_checkpoints.sh" "${CHECKPOINT_ARGS[@]}"

echo "[install] Complete"
