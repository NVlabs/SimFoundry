#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  cat <<'EOF'
Usage: scripts/pipeline/C_application/run.sh [options] [-- overrides...]

Runs application, evaluation, demo collection, demo generation, or a random-action smoke test.

Options:
  --mode smoke-random|eval|demo|full  Default: smoke-random
  --scene-name NAME                   Hydra scene_name override. Default: home_coffee_4
  --root-dir DIR                      Hydra root_dir override. Default: <repo>/Data
  --include IDS                       Comma-separated stage ids to include.
  --exclude IDS                       Comma-separated stage ids to exclude.
  --exec-mode mamba|direct            Process execution mode. Default: mamba
  --no-env-switch                     Alias for --exec-mode direct.
  --python-bin PATH                   Python executable inside each target env. Default: python
  --env-simfoundry NAME               Mamba env for SimFoundry stages. Default: simfoundry
  --env-b1k NAME                      Mamba env for OmniGibson stages. Default: simfoundry
  --dry-run                           Print commands without running stages.
  -h, --help                          Show this help.
EOF
}

MODE="${MODE:-smoke-random}"
SCENE_NAME="${SCENE_NAME:-home_coffee_4}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/Data}"
EXEC_MODE="${EXEC_MODE:-mamba}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_SIMFOUNDRY="${ENV_SIMFOUNDRY:-simfoundry}"
ENV_B1K="${ENV_B1K:-simfoundry}"
DRY_RUN=0
INCLUDE_IDS=""
EXCLUDE_IDS=""
OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --scene-name) SCENE_NAME="$2"; shift 2 ;;
    --root-dir) ROOT_DIR="$2"; shift 2 ;;
    --include) INCLUDE_IDS="$2"; shift 2 ;;
    --exclude) EXCLUDE_IDS="$2"; shift 2 ;;
    --exec-mode) EXEC_MODE="$2"; shift 2 ;;
    --no-env-switch) EXEC_MODE="direct"; shift ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --env-simfoundry) ENV_SIMFOUNDRY="$2"; shift 2 ;;
    --env-b1k) ENV_B1K="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; OVERRIDES+=("$@"); break ;;
    *) OVERRIDES+=("$1"); shift ;;
  esac
done

RUNNER_CMD=("${PYTHON_BIN}")
if [[ "${EXEC_MODE}" == "mamba" ]]; then
  RUNNER_CMD=(mamba run -n "${ENV_SIMFOUNDRY}" "${PYTHON_BIN}")
fi

CMD=(
  "${RUNNER_CMD[@]}"
  "scripts/pipeline/C_application/run_application.py"
  "--mode" "${MODE}"
  "--exec-mode" "${EXEC_MODE}"
  "--python-bin" "${PYTHON_BIN}"
  "--env-simfoundry" "${ENV_SIMFOUNDRY}"
  "--env-b1k" "${ENV_B1K}"
)

if [[ -n "${INCLUDE_IDS}" ]]; then
  CMD+=("--include" "${INCLUDE_IDS}")
fi
if [[ -n "${EXCLUDE_IDS}" ]]; then
  CMD+=("--exclude" "${EXCLUDE_IDS}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=("--dry-run")
fi

CMD+=(
  "root_dir=${ROOT_DIR}"
  "scene_name=${SCENE_NAME}"
  "${OVERRIDES[@]}"
)

cd "${REPO_DIR}"
exec "${CMD[@]}"
