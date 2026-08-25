#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Usage:
#   ./run_editor.sh                       # resume the saved scene state (default)
#   ./run_editor.sh fresh                 # build the scene from pipeline outputs in Data/
#
# Requires the SimFoundry env (override with SIMFOUNDRY_ENV) and a display.

set -euo pipefail

eval "$(mamba shell hook --shell bash)"

mamba activate "${SIMFOUNDRY_ENV:-simfoundry}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_PATH="${REPO_ROOT}/Data"
SCENE_NAME="${SCENE_NAME:-droid_desk_put_away_trash}"
MODE="${1:-resume}"

# An env can have this repo's simfoundry package but a *different* checkout's
# OmniGibson, which silently runs the editor against the wrong deps/ tree — wrong
# OmniGibson version stamped into saved scenes, and deps/ edits that never apply.
OG_PATH="$(python -c 'import omnigibson; print(omnigibson.__file__)' 2>/dev/null || true)"
if [[ -n "${OG_PATH}" && "${OG_PATH}" != "${REPO_ROOT}/"* ]]; then
  echo "WARNING: omnigibson resolves outside this repo." >&2
  echo "  this repo:  ${REPO_ROOT}" >&2
  echo "  omnigibson: ${OG_PATH}" >&2
  echo "  Saved scenes will record that tree's OmniGibson version." >&2
  echo "  Pick a matching env with: SIMFOUNDRY_ENV=<name> $0 $*" >&2
fi

# The editor opens an Isaac Sim window, so it needs an X display. Non-login
# shells often have DISPLAY unset even though a desktop session is running.
export DISPLAY="${DISPLAY:-:1}"

# -u keeps Python's prints interleaved with Isaac Sim's C++ logging when this
# script is piped to a file; without it every print() is stuck in a block buffer.
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "${MODE}" == "fresh" ]]; then
  # Builds the scene from scratch. Needs pipeline stages s4/s10/s11 in Data/<scene>.
  python -u interactive_scene_editor.py \
    --scene_name "${SCENE_NAME}" \
    --mesh_background "${REPO_ROOT}/assets/mesh_backgrounds/droid_desk_mesh.usd" \
    --cam2world "${DATA_PATH}/${SCENE_NAME}/s4_frame/image_0_cam2world.npy" \
    --scene_objects_info "${DATA_PATH}/${SCENE_NAME}/s10_sim/scene_objects_info.json" \
    --pb_scene_poses "${DATA_PATH}/${SCENE_NAME}/s11_physics/pb_scene_poses.json" \
    --scene_objects_categories blue_cup black_trash_can \
    --robot FrankaPanda:robotiq
else
  # Resumes the last saved state. Works with only assets/ present.
  SCENE_STATE="${REPO_ROOT}/assets/scenes/${SCENE_NAME}/${SCENE_NAME}_scene_state_latest.json"
  if [[ ! -f "${SCENE_STATE}" ]]; then
    echo "ERROR: no saved scene state at ${SCENE_STATE}" >&2
    echo "Available scenes:" >&2
    ls "${REPO_ROOT}/assets/scenes" >&2
    exit 1
  fi
  python -u interactive_scene_editor.py \
    --scene_name "${SCENE_NAME}" \
    --load_scene "${SCENE_STATE}"
fi
