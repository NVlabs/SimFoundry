#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# install_everything.sh — build EVERY conda env the CDC real2sim + auto-BG pipeline needs,
# end to end, with the RTX 5090 / sm_120 fixes baked in. Builds all 7 envs (cdc, hunyuan,
# any6d, da3, void, nerfstudio_simfoundry, 3dgrut); use --only for a subset.
#
# Envs built (in order):
#   1. cdc                    (install_cdc.sh)         — main pipeline env + digital_cousins pkg
#   2. hunyuan                (install_hunyuan.sh)     — stage 7 mesh generation
#   3. any6d                  (install_any6d.sh)       — stage 8 pose
#   4. da3                    (install_da3.sh)         — DepthAnything-3
#   5. void                   (install_void.sh)        — VOID inpainting (auto_bg steps 2/3)
#   6. nerfstudio_simfoundry  (install_nerfstudio.sh)  — BG splat train/export (auto_bg step 5)
#   7. 3dgrut                 (install_3dgrut.sh)      — PLY -> USDZ (auto_bg step 7)
# Model checkpoints (download_checkpoints.sh) are OPT-IN via --checkpoints (off by default).
#
# Usage:
#   bash scripts/installation/install_everything.sh [--project-root DIR] [--fresh]
#                                                   [--checkpoints] [--only "a b c"]
#
#   --fresh             Remove each target env before (re)installing it (clean rebuild).
#   --checkpoints       Also run download_checkpoints.sh at the end. OPT-IN: checkpoints
#                       are NOT downloaded by default. The gated netflix/void-model weights
#                       require `huggingface-cli login` first, so the recommended flow is:
#                         1) install_everything.sh          (build envs, no checkpoints)
#                         2) login_services.sh              (huggingface-cli login, etc.)
#                         3) download_checkpoints.sh        (or rerun this with --checkpoints)
#   --only "names..."   Space-separated subset of: cdc hunyuan any6d da3 void nerfstudio 3dgrut
#
# Prereqs: mamba (Miniforge) and `uv` (for 3dgrut) on PATH. The void + nerfstudio installers
# need internet for the cu128 torch wheels; download_checkpoints needs `huggingface-cli login`
# for the gated netflix/void-model weights.

set -euo pipefail

if ! command -v mamba >/dev/null 2>&1; then
  echo "Error: mamba was not found on PATH. Install Miniforge first." >&2
  exit 127
fi
eval "$(mamba shell hook --shell bash)"

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
project_root="$(cd "$SCRIPT_DIR/../.." && pwd)"
FRESH=false
DOWNLOAD_CHECKPOINTS=false
ONLY=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --project-root)     project_root="$2"; shift 2 ;;
    --fresh)            FRESH=true; shift ;;
    --checkpoints)      DOWNLOAD_CHECKPOINTS=true; shift ;;
    --only)             ONLY="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,35p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done
PROJECT_ROOT="$(cd "$project_root" && pwd)"

# Map short name -> "installer_script env_name"
declare -A INSTALLER=(
  [cdc]="install_cdc.sh cdc"
  [hunyuan]="install_hunyuan.sh hunyuan"
  [any6d]="install_any6d.sh any6d"
  [da3]="install_da3.sh da3"
  [void]="install_void.sh void"
  [nerfstudio]="install_nerfstudio.sh nerfstudio_simfoundry"
  [3dgrut]="install_3dgrut.sh 3dgrut"
)
ORDER=(cdc hunyuan any6d da3 void nerfstudio 3dgrut)
if [[ -n "${ONLY}" ]]; then
  read -r -a ORDER <<< "${ONLY}"
fi

echo "============================================================"
echo "install_everything.sh"
echo "  project_root:     ${PROJECT_ROOT}"
echo "  envs to install:  ${ORDER[*]}"
echo "  fresh rebuild:    ${FRESH}"
echo "  download ckpts:   ${DOWNLOAD_CHECKPOINTS}"
echo "============================================================"

# conda-forge is required by 3dgrut's create_conda.sh (and harmless otherwise).
if ! conda config --show channels 2>/dev/null | grep -q "conda-forge"; then
  echo "Adding conda-forge channel (none was configured)..."
  conda config --add channels conda-forge
fi

for key in "${ORDER[@]}"; do
  entry="${INSTALLER[$key]:-}"
  if [[ -z "${entry}" ]]; then
    echo "WARNING: unknown env '${key}', skipping."; continue
  fi
  script="${entry%% *}"; envn="${entry##* }"
  echo ""
  echo ">>> [${key}] ${script} (env: ${envn})"
  if [[ "${FRESH}" == true ]]; then
    echo "    --fresh: removing existing env '${envn}'..."
    mamba env remove -n "${envn}" -y 2>/dev/null || true
  elif mamba run -n "${envn}" true 2>/dev/null; then
    # Resumable: the per-env installers `mamba create` and would error on an
    # existing env, so skip envs already present (use --fresh to force a rebuild).
    echo "    env '${envn}' already exists — skipping (use --fresh to rebuild)."
    continue
  fi
  bash "${SCRIPT_DIR}/${script}" --project-root "${PROJECT_ROOT}" --env-name "${envn}" --default
done

if [[ "${DOWNLOAD_CHECKPOINTS}" == true ]]; then
  echo ""
  echo ">>> download_checkpoints.sh"
  bash "${SCRIPT_DIR}/download_checkpoints.sh" --project-root "${PROJECT_ROOT}" --default
fi

echo ""
echo "============================================================"
echo "install_everything.sh: DONE"
echo "  Verify:  mamba env list"
if [[ "${DOWNLOAD_CHECKPOINTS}" != true ]]; then
  echo ""
  echo "  Checkpoints were NOT downloaded (opt-in via --checkpoints). Next steps:"
  echo "    1) bash ${SCRIPT_DIR#${PROJECT_ROOT}/}/login_services.sh       # huggingface-cli login (gated VOID weights)"
  echo "    2) bash ${SCRIPT_DIR#${PROJECT_ROOT}/}/download_checkpoints.sh # download model checkpoints"
fi
echo "  Then run the pipeline: see README.md and INSTALL.md"
echo "============================================================"
