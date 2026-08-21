#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  cat <<'EOF'
Usage: scripts/pipeline/A_reconstruction/run.sh [options] [-- hydra_override=...]

Runs the canonical A_reconstruction 1-14 real2sim pipeline.

Defaults target the local fixture:
  root_dir    = <repo>/Data
  scene_name  = home_coffee_4
  video_fpath = <repo>/Data/home_coffee_4/s1_video/video/scene.mp4

Options:
  --pipeline video|stereo|zed     Input pipeline mode. zed is an alias for stereo. Default: video
  --scene-name NAME               Hydra scene_name override. Default: home_coffee_4
  --root-dir DIR                  Hydra root_dir override. Default: <repo>/Data
  --video-fpath PATH              Source video path for video mode.
  --include IDS                   Comma-separated stage ids to include, e.g. 1b,2,3
  --exclude IDS                   Comma-separated stage ids to exclude.
  --exec-mode mamba|direct        Process execution mode. Default: mamba
  --no-env-switch                 Alias for --exec-mode direct.
  --python-bin PATH               Python executable inside each target env. Default: python
  --env-simfoundry NAME           Mamba env for SimFoundry stages. Default: simfoundry
  --env-nerfstudio NAME           Mamba env for stage 2c. Default: nerfstudio_simfoundry
  --env-da3 NAME                  Mamba env for depth stage. Default: da3
  --env-mesh NAME                 Mamba env for mesh generation. Default: hunyuan (use simfoundry for trellis.2 if installed)
  --env-b1k NAME                  Mamba env for OmniGibson stages. Default: simfoundry
  --stream / --no-stream          Enable/disable stages 5-8 streaming. Default: enabled
  --stream-start-stage N          Streaming start stage, 5-8. Default: 5
  --stream-end-stage N            Streaming end stage, 5-8. Default: 8
  --max-vram-frac F               Streaming VRAM budget as a fraction of total GPU memory. Default: stream_subseq.max_vram_frac (0.9)
  --max-vram-gb N                 Opt-in absolute hard VRAM budget for streaming, in GiB. Default: unset (uses the fraction)
  --detect-articulation           Run automated articulation decomposition stage 9 after stage 8.
  --bg-splat                      Include stage 2c (nerfstudio background Gaussian splat). Opt-in only — not run by default.
  --skip-successful               Skip stages already recorded as successful for this scene
                                  (stage_info.json). Means "completed once", not "up to date":
                                  re-running an earlier stage does not invalidate later markers.
  --cache-mode                    Cache raw remote model responses.
  --test-mode                     Replay remote model responses from cache.
  --model-cache-dir DIR           Cache root. Default: .cache/simfoundry/model_calls
  --dry-run                       Print commands without running stages.
  -h, --help                      Show this help.

Additional Hydra overrides may be passed after --.
EOF
}

PIPELINE="${PIPELINE:-video}"
SCENE_NAME="${SCENE_NAME:-home_coffee_4}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/Data}"
VIDEO_FPATH="${VIDEO_FPATH:-${ROOT_DIR}/${SCENE_NAME}/s1_video/video/scene.mp4}"
EXEC_MODE="${EXEC_MODE:-mamba}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_SIMFOUNDRY="${ENV_SIMFOUNDRY:-simfoundry}"
ENV_NERFSTUDIO="${ENV_NERFSTUDIO:-nerfstudio_simfoundry}"
ENV_DA3="${ENV_DA3:-da3}"
ENV_MESH="${ENV_MESH:-hunyuan}"
ENV_B1K="${ENV_B1K:-simfoundry}"
STREAM_ENABLED="${STREAM_ENABLED:-1}"
STREAM_START_STAGE="${STREAM_START_STAGE:-5}"
STREAM_END_STAGE="${STREAM_END_STAGE:-8}"
# Budget the streamed stages as a fraction of the card's total memory, so one setting works
# on a 24 GiB and a 96 GiB GPU. An absolute GiB cap is opt-in: it is only forwarded when the
# user asks for it, since with hard_vram_cap the budget counts TOTAL GPU usage, and a value
# that is too small for the card stalls stages.
MAX_VRAM_FRAC="${MAX_VRAM_FRAC:-}"
MAX_VRAM_GB="${MAX_VRAM_GB:-}"
DETECT_ARTICULATION="${DETECT_ARTICULATION:-0}"
BG_SPLAT="${BG_SPLAT:-0}"
SKIP_SUCCESSFUL="${SKIP_SUCCESSFUL:-0}"
CACHE_MODE_ENABLED=0
TEST_MODE_ENABLED=0
MODEL_CACHE_DIR="${SIMFOUNDRY_MODEL_CACHE_DIR:-}"
DRY_RUN=0
INCLUDE_IDS=""
EXCLUDE_IDS=""
HYDRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipeline)
      PIPELINE="$2"
      shift 2
      ;;
    --scene-name)
      SCENE_NAME="$2"
      shift 2
      ;;
    --root-dir)
      ROOT_DIR="$2"
      shift 2
      ;;
    --video-fpath)
      VIDEO_FPATH="$2"
      shift 2
      ;;
    --include)
      INCLUDE_IDS="$2"
      shift 2
      ;;
    --exclude)
      EXCLUDE_IDS="$2"
      shift 2
      ;;
    --exec-mode)
      EXEC_MODE="$2"
      shift 2
      ;;
    --no-env-switch)
      EXEC_MODE="direct"
      shift
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --env-simfoundry)
      ENV_SIMFOUNDRY="$2"
      shift 2
      ;;
    --env-nerfstudio)
      ENV_NERFSTUDIO="$2"
      shift 2
      ;;
    --env-da3|--env-da)
      ENV_DA3="$2"
      shift 2
      ;;
    --env-mesh)
      ENV_MESH="$2"
      shift 2
      ;;
    --env-b1k)
      ENV_B1K="$2"
      shift 2
      ;;
    --stream)
      STREAM_ENABLED=1
      shift
      ;;
    --no-stream)
      STREAM_ENABLED=0
      shift
      ;;
    --stream-start-stage)
      STREAM_START_STAGE="$2"
      shift 2
      ;;
    --stream-end-stage)
      STREAM_END_STAGE="$2"
      shift 2
      ;;
    --max-vram-frac)
      MAX_VRAM_FRAC="$2"
      shift 2
      ;;
    --max-vram-gb)
      MAX_VRAM_GB="$2"
      shift 2
      ;;
    --detect-articulation)
      DETECT_ARTICULATION=1
      shift
      ;;
    --bg-splat)
      BG_SPLAT=1
      shift
      ;;
    --skip-successful)
      SKIP_SUCCESSFUL=1
      shift
      ;;
    --cache-mode)
      CACHE_MODE_ENABLED=1
      shift
      ;;
    --test-mode)
      TEST_MODE_ENABLED=1
      shift
      ;;
    --model-cache-dir)
      MODEL_CACHE_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      HYDRA_OVERRIDES+=("$@")
      break
      ;;
    *)
      HYDRA_OVERRIDES+=("$1")
      shift
      ;;
  esac
done

case "${PIPELINE}" in
  video)
    INPUT_MODE="video"
    ;;
  stereo|zed)
    INPUT_MODE="stereo"
    ;;
  *)
    echo "Unsupported --pipeline '${PIPELINE}'. Expected video, stereo, or zed." >&2
    exit 2
    ;;
esac

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
  "scripts/pipeline/A_reconstruction/run_reconstruction.py"
  "--input-mode" "${INPUT_MODE}"
  "--exec-mode" "${EXEC_MODE}"
  "--python-bin" "${PYTHON_BIN}"
  "--env-simfoundry" "${ENV_SIMFOUNDRY}"
  "--env-nerfstudio" "${ENV_NERFSTUDIO}"
  "--env-da3" "${ENV_DA3}"
  "--env-mesh" "${ENV_MESH}"
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
if [[ "${STREAM_ENABLED}" == "1" ]]; then
  CMD+=(
    "--stream-5-8"
    "--stream-start-stage" "${STREAM_START_STAGE}"
    "--stream-end-stage" "${STREAM_END_STAGE}"
  )
fi
if [[ "${DETECT_ARTICULATION}" == "1" ]]; then
  CMD+=("--detect-articulation")
fi
if [[ "${BG_SPLAT}" == "1" ]]; then
  CMD+=("--bg-splat")
fi
if [[ "${SKIP_SUCCESSFUL}" == "1" ]]; then
  CMD+=("--skip-successful")
fi

CMD+=(
  "root_dir=${ROOT_DIR}"
  "scene_name=${SCENE_NAME}"
)

if [[ "${INPUT_MODE}" == "video" ]]; then
  CMD+=("s1_video.video_fpath=${VIDEO_FPATH}")
fi

if [[ -n "${MAX_VRAM_FRAC}" ]]; then
  CMD+=("stream_subseq.max_vram_frac=${MAX_VRAM_FRAC}")
fi
if [[ -n "${MAX_VRAM_GB}" ]]; then
  CMD+=("stream_subseq.max_vram_gb=${MAX_VRAM_GB}")
fi
if [[ "${DETECT_ARTICULATION}" == "1" ]]; then
  CMD+=("s9_articulate_objects.interactive_review=false")
fi
CMD+=("${HYDRA_OVERRIDES[@]}")

cd "${REPO_DIR}"
exec "${CMD[@]}"
