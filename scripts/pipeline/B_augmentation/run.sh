#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  cat <<'EOF'
Usage: scripts/pipeline/B_augmentation/run.sh [options] [-- hydra_override=...]

Runs reconstructed scene augmentation: object cousins, scene variations, and task generation.

Options:
  --scene-name NAME               Hydra scene_name override. Default: home_coffee_4
  --root-dir DIR                  Hydra root_dir override. Default: <repo>/Data
  --phases CSV                    object-cousins,scene-variations,task-generation,p2p
  --include IDS                   Comma-separated stage ids to include.
  --exclude IDS                   Comma-separated stage ids to exclude.
  --include-p2p                   Run optional cousin p2p matching.
  --exec-mode mamba|direct        Process execution mode. Default: mamba
  --no-env-switch                 Alias for --exec-mode direct.
  --python-bin PATH               Python executable inside each target env. Default: python
  --env-simfoundry NAME           Mamba env for SimFoundry stages. Default: simfoundry
  --env-mesh NAME                 Mamba env for mesh generation. Default: hunyuan
  --env-b1k NAME                  Mamba env for OmniGibson stages. Default: simfoundry
  --max-vram-gb N                 Opt-in absolute VRAM budget, in GiB. Default: unset
  --cache-mode                    Cache raw remote model responses.
  --test-mode                     Replay remote model responses from cache.
  --model-cache-dir DIR           Cache root. Default: .cache/simfoundry/model_calls
  --dry-run                       Print commands without running stages.
  -h, --help                      Show this help.

Additional Hydra overrides may be passed after --.
EOF
}

SCENE_NAME="${SCENE_NAME:-home_coffee_4}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/Data}"
EXEC_MODE="${EXEC_MODE:-mamba}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_SIMFOUNDRY="${ENV_SIMFOUNDRY:-simfoundry}"
ENV_MESH="${ENV_MESH:-hunyuan}"
ENV_B1K="${ENV_B1K:-simfoundry}"
MAX_VRAM_GB="${MAX_VRAM_GB:-}"
CACHE_MODE_ENABLED=0
TEST_MODE_ENABLED=0
MODEL_CACHE_DIR="${SIMFOUNDRY_MODEL_CACHE_DIR:-}"
DRY_RUN=0
INCLUDE_IDS=""
EXCLUDE_IDS=""
PHASES=""
INCLUDE_P2P=0
HYDRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene-name) SCENE_NAME="$2"; shift 2 ;;
    --root-dir) ROOT_DIR="$2"; shift 2 ;;
    --phases|--only) PHASES="$2"; shift 2 ;;
    --include) INCLUDE_IDS="$2"; shift 2 ;;
    --exclude) EXCLUDE_IDS="$2"; shift 2 ;;
    --include-p2p) INCLUDE_P2P=1; shift ;;
    --exec-mode) EXEC_MODE="$2"; shift 2 ;;
    --no-env-switch) EXEC_MODE="direct"; shift ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --env-simfoundry) ENV_SIMFOUNDRY="$2"; shift 2 ;;
    --env-mesh) ENV_MESH="$2"; shift 2 ;;
    --env-b1k) ENV_B1K="$2"; shift 2 ;;
    --max-vram-gb) MAX_VRAM_GB="$2"; shift 2 ;;
    --cache-mode) CACHE_MODE_ENABLED=1; shift ;;
    --test-mode) TEST_MODE_ENABLED=1; shift ;;
    --model-cache-dir) MODEL_CACHE_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; HYDRA_OVERRIDES+=("$@"); break ;;
    *) HYDRA_OVERRIDES+=("$1"); shift ;;
  esac
done

if [[ "${CACHE_MODE_ENABLED}" == "1" && "${TEST_MODE_ENABLED}" == "1" ]]; then
  echo "--cache-mode and --test-mode are mutually exclusive." >&2
  exit 2
fi

if [[ "${CACHE_MODE_ENABLED}" == "1" ]]; then
  export CACHE_MODE=1
  unset TEST_MODE || true
elif [[ "${TEST_MODE_ENABLED}" == "1" ]]; then
  export TEST_MODE=1
  unset CACHE_MODE || true
fi

if [[ "${CACHE_MODE_ENABLED}" == "1" || "${TEST_MODE_ENABLED}" == "1" ]]; then
  MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${REPO_DIR}/.cache/simfoundry/model_calls}"
fi
if [[ -n "${MODEL_CACHE_DIR}" ]]; then
  if [[ "${MODEL_CACHE_DIR}" != /* ]]; then
    MODEL_CACHE_DIR="${REPO_DIR}/${MODEL_CACHE_DIR}"
  fi
  export SIMFOUNDRY_MODEL_CACHE_DIR="${MODEL_CACHE_DIR}"
fi

RUNNER_CMD=("${PYTHON_BIN}")
if [[ "${EXEC_MODE}" == "mamba" ]]; then
  RUNNER_CMD=(mamba run -n "${ENV_SIMFOUNDRY}" "${PYTHON_BIN}")
fi

CMD=(
  "${RUNNER_CMD[@]}"
  "scripts/pipeline/B_augmentation/run_augmentation.py"
  "--exec-mode" "${EXEC_MODE}"
  "--python-bin" "${PYTHON_BIN}"
  "--env-simfoundry" "${ENV_SIMFOUNDRY}"
  "--env-mesh" "${ENV_MESH}"
  "--env-b1k" "${ENV_B1K}"
)

if [[ -n "${INCLUDE_IDS}" ]]; then
  CMD+=("--include" "${INCLUDE_IDS}")
fi
if [[ -n "${EXCLUDE_IDS}" ]]; then
  CMD+=("--exclude" "${EXCLUDE_IDS}")
fi
if [[ -n "${PHASES}" ]]; then
  CMD+=("--phases" "${PHASES}")
fi
if [[ "${INCLUDE_P2P}" == "1" ]]; then
  CMD+=("--include-p2p")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=("--dry-run")
fi

CMD+=(
  "root_dir=${ROOT_DIR}"
  "scene_name=${SCENE_NAME}"
  "prompt_cousin_structured.num_geometry_variation=2"
  "prompt_cousin_structured.num_topology_variation=2"
  "prompt_cousin_structured.num_visual_variation=2"
  "prompt_cousin_structured.min_keep_per_dim=1"
  "prompt_cousin_structured.max_objects=2"
  "prompt_cousin_structured.max_components=2"
  "prompt_cousin_structured.max_generated_images_per_object=2"
  "prompt_cousin_structured.text_model=gemini-2.5-flash"
  "generate_cousins_combination.num_variations_used_per_object=2"
  "generate_cousins_combination.num_obj_to_swap=2"
  "generate_cousins_combination.max_combinations=2"
  "s13_og.auto_generation=true"
  "s13_og.auto_iter_num=2"
  "propose_scene_task.num_tasks=2"
  "${HYDRA_OVERRIDES[@]}"
)

# Absolute cap is opt-in; without it the config default (null) applies.
if [[ -n "${MAX_VRAM_GB}" ]]; then
  CMD+=("augmentation.max_vram_gb=${MAX_VRAM_GB}")
fi

cd "${REPO_DIR}"
exec "${CMD[@]}"
