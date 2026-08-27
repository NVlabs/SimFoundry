#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Installs the browser light scene editor's environment.
#
# The editor needs its OWN environment, never the simfoundry env: it uses
# standalone usd-core, and Isaac Sim ships its own pxr that is only importable
# once Kit is running -- installing usd-core alongside it risks shadowing that
# copy. Nothing in this env needs a GPU, Isaac Sim, or OmniGibson.
#
# Idempotent: re-run after a pull to pick up new dependencies. The editor's
# optional halves degrade quietly rather than refusing to start (no ruamel.yaml
# means task-config saves lose comments; no google-genai means the Generate
# task panel reports itself missing), so a stale env looks like a working one.
#
# Usage:
#   bash scripts/installation/install_light_editor.sh
#
#   LIGHT_EDITOR_ENV=my-env     override the env name (default: simfoundry-editor)
#   LIGHT_EDITOR_PYTHON=3.12    override the python version (default: 3.11)

set -euo pipefail

ENV_NAME="${LIGHT_EDITOR_ENV:-simfoundry-editor}"
PYTHON_VERSION="${LIGHT_EDITOR_PYTHON:-3.11}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQUIREMENTS="${REPO_ROOT}/scripts/interactive/light_editor/requirements.txt"

MAMBA="$(command -v mamba || command -v conda || true)"
if [[ -z "${MAMBA}" ]]; then
  echo "ERROR: neither mamba nor conda is on PATH." >&2
  exit 1
fi

if ! "${MAMBA}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Creating conda env '${ENV_NAME}' (python ${PYTHON_VERSION})"
  "${MAMBA}" create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
else
  echo "Env '${ENV_NAME}' already exists; updating its dependencies"
fi

"${MAMBA}" run -n "${ENV_NAME}" python -m pip install --upgrade -r "${REQUIREMENTS}"

"${MAMBA}" run -n "${ENV_NAME}" python - <<'PY'
import importlib
missing = []
for module in ("pxr", "trimesh", "ruamel.yaml", "scipy", "numpy", "PIL", "yaml"):
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append(f"{module}: {exc}")
if missing:
    raise SystemExit("Light editor env validation failed:\n  " + "\n  ".join(missing))
from pxr import Usd
print(f"Light editor env OK (usd-core {Usd.GetVersion()})")
PY

echo
echo "Done. Launch the editor with:"
echo "  ${MAMBA##*/} run -n ${ENV_NAME} python ${REPO_ROOT}/scripts/interactive/light_editor/server.py \\"
echo "    --scene /absolute/path/to/<scene>_scene_state_latest.json"
