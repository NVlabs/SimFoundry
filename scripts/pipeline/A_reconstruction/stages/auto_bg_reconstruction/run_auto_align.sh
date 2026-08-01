#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Background-only add-on: adds a BG 3DGS splat to an EXISTING canonical reconstruction.
#   -> assets/scenes/<scene>/<scene>_scene_state_auto_bg.json
#
# PRECONDITION — the canonical reconstruction must already have been run in splat-prep
#   mode (one DA3 world for FG + BG), producing under Data/<scene>/:
#       s1_video/frames_subsampled_<N>/  (672x384) + s1_video/input_video.mp4
#       s2_da/da/exports/npz/results.npz (orig-DA3, <N> frames, DA3 backend)
#       s4_frame/image_<N>_cam2world.npy
#       s13_og/reconstructed_og_scene.json
#   Canonical splat-prep command (run BEFORE this script):
#       OMNIGIBSON_HEADLESS=1 scripts/pipeline/A_reconstruction/run.sh \
#         --scene-name <scene> --video-fpath <video> --no-stream -- \
#         s1_video.splat_prep=true s1_video.n_subsampled_frames=400 \
#         s1_video.target_w=672 s1_video.target_h=384 \
#         s5_scene.pda_geometric_backend=depth_pro s10_sim.vlm_model=gemini-2.5-pro
#   (Stage 2 uses the DA3 backend by default; stage 13 auto-exits via s13_og.interactive=false.)
#
# This script REUSES those outputs — it does NOT re-subsample, re-run orig-DA3, or
# re-run stages 3-13. It runs:
#   Steps 1-4 — background ingest: stage VOID input (symlink canonical frames + video) ->
#             quadmask -> VOID Pass 1/2 chunked -> void-DA3 @ res448 -> seed PLY.
#   Steps 5-7 — splat (splatfacto-big + SO3xR3) -> bridge to OG world -> build assets.
#
# Usage:
#   scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/run_auto_align.sh <scene_name> <video_path> \
#       [--clean] [--num-frames 400] [--floor-category 'desk, table, or counter']
#   (<video_path> is used only for the precondition guidance message.)
# Example:
#   scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/run_auto_align.sh quillen_table_2 \
#       <repo>/Data/Scene_Video/quillen_table_2.MOV

set -euo pipefail

# ---------- args ----------
if [[ $# -lt 2 ]]; then
    echo "usage: $0 <scene_name> <video_path> [--clean] [--num-frames N] [--floor-category 'desk, table, or counter']"
    exit 1
fi
SCENE="$1"; shift
VIDEO="$1"; shift
CLEAN=0
NUM_FRAMES=400
FLOOR_CATEGORY="desk, table, or counter"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean) CLEAN=1; shift ;;
        --num-frames) NUM_FRAMES="$2"; shift 2 ;;
        --floor-category) FLOOR_CATEGORY="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
DATA_DIR="${REPO_ROOT}/Data/${SCENE}"
ASSETS_DIR="${REPO_ROOT}/assets/scenes/${SCENE}"
LOG_DIR="${DATA_DIR}/_logs"

# Mamba env names — flip if your install uses different names. (Foreground-stage envs
# hunyuan/any6d are not needed: stages 3-13 belong to the canonical reconstruction.)
SIMFOUNDRY_ENV="simfoundry"
DA3_ENV="da3"

# VIDEO is used only for the precondition guidance message; it need not still exist.
[[ -f "$VIDEO" ]] || echo "[orchestrator] note: video '$VIDEO' not found (only used for the guidance message)"

cd "$REPO_ROOT"

if [[ "$CLEAN" -eq 1 ]]; then
    # Background-only: clean ONLY the auto_bg-owned outputs. The canonical
    # reconstruction (s1_video / s2_da / s3-s13) is a precondition and is preserved.
    echo "[orchestrator] --clean: removing $DATA_DIR/auto_bg and $ASSETS_DIR"
    rm -rf "$DATA_DIR/auto_bg" "$ASSETS_DIR"
fi
mkdir -p "$LOG_DIR"

# run "<label>" "<env>" -- <cmd...>
# Streams stdout/stderr to a per-stage log under $LOG_DIR.
run() {
    local label="$1"; shift
    local env="$1"; shift
    [[ "${1:-}" == "--" ]] && shift
    local logf
    logf="${LOG_DIR}/$(printf '%s' "$label" | tr ' /' '__').log"
    echo "[orchestrator] $(date +%H:%M:%S) START  $label  (env=$env)"
    echo "[orchestrator]   log: $logf"
    local t0=$SECONDS
    if mamba run -n "$env" "$@" > "$logf" 2>&1; then
        echo "[orchestrator] $(date +%H:%M:%S) OK     $label  ($((SECONDS - t0))s)"
    else
        local rc=$?
        echo "[orchestrator] $(date +%H:%M:%S) FAIL   $label  (exit=$rc, $((SECONDS - t0))s)"
        echo "[orchestrator]   tail of $logf:"
        tail -40 "$logf" | sed 's/^/[orchestrator]     /'
        exit $rc
    fi
}

# run_gpu_locked "<label>" "<env>" -- <cmd...>
# Same as run(), but when SIMFOUNDRY_GPU_LOCK_FILE is set, wraps the command with
# `flock` against that file so only one parallel orchestrator process can
# enter the stage at a time. Used for the GPU-heavy splat training (step 5) to
# prevent OOMs when running multiple scenes concurrently on one GPU.
# No-op (identical to run()) when SIMFOUNDRY_GPU_LOCK_FILE is unset.
run_gpu_locked() {
    local label="$1"; shift
    local env="$1"; shift
    [[ "${1:-}" == "--" ]] && shift
    if [[ -n "${SIMFOUNDRY_GPU_LOCK_FILE:-}" ]]; then
        run "$label" "$env" -- flock "${SIMFOUNDRY_GPU_LOCK_FILE}" "$@"
    else
        run "$label" "$env" -- "$@"
    fi
}

# run_or_skip "<label>" "<env>" "<output_marker>" -- <cmd...>
# Same as run(), but checks <output_marker> first. If that file exists, the
# stage is logged as SKIP and not executed. Used for the ingest steps so that
# retrying a partial run reuses the expensive VOID Pass 1+2 outputs and the
# DA3 NPZs from the previous attempt.
run_or_skip() {
    local label="$1"; shift
    local env="$1"; shift
    local marker="$1"; shift
    [[ "${1:-}" == "--" ]] && shift
    if [[ -e "$marker" ]]; then
        echo "[orchestrator] $(date +%H:%M:%S) SKIP   $label  (have $(basename "$marker"))"
        return 0
    fi
    run "$label" "$env" -- "$@"
}

# ===================== Precondition: canonical splat-prep reconstruction =====================

AUTO_BG_DIR="${DATA_DIR}/auto_bg"
VOID_DIR="${AUTO_BG_DIR}/void"
VOID_INPUT_DIR="${VOID_DIR}/input"
PASS1_STITCHED="${VOID_DIR}/pass1/pass1.mp4"
PASS2_STITCHED="${VOID_DIR}/pass2/pass2.mp4"

# Canonical outputs this add-on REUSES (must already exist; we never re-create them).
S1_FRAMES_DIR="${DATA_DIR}/s1_video/frames_subsampled_${NUM_FRAMES}"
S1_VIDEO_MP4="${DATA_DIR}/s1_video/input_video.mp4"
S2_DA_NPZ="${DATA_DIR}/s2_da/da/exports/npz/results.npz"
S4_FRAME_DIR="${DATA_DIR}/s4_frame"
S13_OG_JSON="${DATA_DIR}/s13_og/reconstructed_og_scene.json"

precondition_fail() {
    echo "[orchestrator] PRECONDITION FAILED: $1" >&2
    cat >&2 <<EOF
[orchestrator]
[orchestrator] auto_bg is a background-only add-on. Run the canonical reconstruction in
[orchestrator] splat-prep mode FIRST (so foreground + BG splat share one DA3 world):
[orchestrator]
[orchestrator]   OMNIGIBSON_HEADLESS=1 scripts/pipeline/A_reconstruction/run.sh \\
[orchestrator]     --scene-name ${SCENE} --video-fpath ${VIDEO} --no-stream -- \\
[orchestrator]     s1_video.splat_prep=true s1_video.n_subsampled_frames=${NUM_FRAMES} \\
[orchestrator]     s1_video.target_w=672 s1_video.target_h=384 \\
[orchestrator]     s5_scene.pda_geometric_backend=depth_pro
[orchestrator]
[orchestrator] (Stages 5/10 use Vertex AI Gemini; stage 2 uses DA3; stage 13 auto-exits via s13_og.interactive=false.)
EOF
    exit 1
}

[[ -f "$S2_DA_NPZ" ]]    || precondition_fail "missing orig-DA3 npz: $S2_DA_NPZ"
[[ -f "$S1_VIDEO_MP4" ]] || precondition_fail "missing $S1_VIDEO_MP4 (canonical run must use s1_video.splat_prep=true)"
[[ -d "$S1_FRAMES_DIR" ]] || precondition_fail "missing frames dir: $S1_FRAMES_DIR"
[[ -f "$S13_OG_JSON" ]]  || precondition_fail "missing OG scene state: $S13_OG_JSON (canonical stage 13)"
ls "${S4_FRAME_DIR}"/image_*_cam2world.npy >/dev/null 2>&1 \
    || precondition_fail "missing s4_frame/image_<N>_cam2world.npy (canonical stage 4)"

# Validate the orig-DA3 npz (DA3 schema + exactly NUM_FRAMES frames) and that the
# canonical frames are NUM_FRAMES PNGs @ 672x384 — the invariants the splat trainer
# and seed-PLY builder hard-assert.
mamba run -n "$SIMFOUNDRY_ENV" python - "$S2_DA_NPZ" "$S1_FRAMES_DIR" "$NUM_FRAMES" <<'PY' \
    || precondition_fail "canonical orig-DA3 / frames check failed (see message above)"
import glob, sys
import numpy as np
from PIL import Image
npz_path, frames_dir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = np.load(npz_path)
for k in ("extrinsics", "intrinsics", "depth", "image"):
    if k not in d:
        sys.exit(f"  npz missing DA3 key '{k}' (wrong depth backend? need s2_depth.backend=da3)")
ne = int(d["extrinsics"].shape[0])
if ne != n:
    sys.exit(f"  orig-DA3 has {ne} frames, expected {n} (canonical frame count != --num-frames {n})")
pngs = sorted(glob.glob(frames_dir + "/*.png"))
if len(pngs) != n:
    sys.exit(f"  {frames_dir} has {len(pngs)} PNGs, expected {n}")
w, h = Image.open(pngs[0]).size
if (w, h) != (672, 384):
    sys.exit(f"  frame size {(w, h)} != (672, 384) (canonical run must set splat_prep target_w/h)")
print(f"  OK: orig-DA3 {ne} frames; {len(pngs)} frames @ {w}x{h}")
PY
echo "[orchestrator] precondition OK — reusing canonical reconstruction in ${DATA_DIR}"

# ===================== Background ingest (steps 1-4) =====================

# Stage VOID input from the canonical s1_video (NO re-subsampling, NO orig-DA3 here).
# void/input/ must be a REAL dir — step 1 writes quadmask/prompt/meta into it; only the two
# inputs are symlinked back to the canonical frame set, so VOID cleans exactly the
# frames orig-DA3 saw.
mkdir -p "${VOID_INPUT_DIR}"
ln -sfn "${S1_FRAMES_DIR}" "${VOID_INPUT_DIR}/subsampled"
ln -sfn "${S1_VIDEO_MP4}" "${VOID_INPUT_DIR}/input_video.mp4"

# Fragmentation-friendly allocator for the chunked void-DA3 (per-chunk subprocess).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The auto_bg stage scripts are Hydra-driven: they read scripts/cfg/auto_bg.yaml
# and take overrides as key=val (e.g. scene_name=..., floor_category=...).
# floor_category contains commas ("desk, table, or counter"); Hydra treats an unquoted
# comma value as a list, so wrap it in LITERAL single quotes that Hydra strips to a string.
run_or_skip "1 quadmask" "$SIMFOUNDRY_ENV" "${VOID_INPUT_DIR}/quadmask_0.mp4" -- \
    python scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/1_generate_quadmask_for_void.py \
        scene_name="${SCENE}" \
        floor_category="'${FLOOR_CATEGORY}'"

run_or_skip "2 pass1" "$SIMFOUNDRY_ENV" "${PASS1_STITCHED}" -- \
    python scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/2_run_void_pass1.py \
        scene_name="${SCENE}"

run_or_skip "3 pass2" "$SIMFOUNDRY_ENV" "${PASS2_STITCHED}" -- \
    python scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/3_run_void_pass2.py \
        scene_name="${SCENE}"

# da3_void = VOID DA3 via the canonical stage 2b, pointed at the void cleaned_frames + a separate
# out_dir (depth GT for the splat; different input → own world). orig-DA3 is NOT run here —
# it is owned by the canonical reconstruction (the reused s2_da NPZ checked above).
run_or_skip "da3_void (canonical 2b)" "$DA3_ENV" \
    "${DATA_DIR}/auto_bg/da3/void/da/exports/npz/results.npz" -- \
    python scripts/pipeline/A_reconstruction/stages/2b_run_da.py \
        scene_name="${SCENE}" \
        s2_da.frames_dir="${DATA_DIR}/auto_bg/void/pass2/cleaned_frames" \
        s2_da.out_dir="${DATA_DIR}/auto_bg/da3/void"

run_or_skip "4 build_seed_ply" "$SIMFOUNDRY_ENV" \
    "${DATA_DIR}/auto_bg/seed.ply" -- \
    python scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/4_build_seed_ply_from_void_da3.py \
        scene_name="${SCENE}"

# Top-level convenience symlink: auto_bg/clean_frames -> void/pass2/cleaned_frames.
ln -sfn "void/pass2/cleaned_frames" "${AUTO_BG_DIR}/clean_frames"

# (Foreground stages 3-13 are NOT run here — they belong to the canonical reconstruction,
# verified by the precondition above. s4_frame + s13_og are reused by steps 6-7.)

# ===================== Splat + bridge + assets (steps 5-7) =====================

# Step 5 trains splatfacto-big WITH the env-var-gated depth-loss path enabled
# (use_depth_loss=true, depth_loss_mult=0.5 in auto_bg.yaml). DA3 depth from
# auto_bg/da3/void (matches the cleaned RGB the splat trains against) supervises
# rendered depth via L1, suppressing floaters that vanilla splatfacto parks
# above flat surfaces. See README §3 for the full rationale. Requires the
# env-var-gated patch in nerfstudio's splatfacto.py.
run_gpu_locked "5 train_bg_splat" "$SIMFOUNDRY_ENV" -- \
    python scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/5_train_bg_splat.py \
        scene_name="${SCENE}"

# bridge_bg_splat_to_og.py reads the canonical orig-DA3 (s2_da/.../results.npz) + the
# canonical s4_frame cam2world by default — both produced by the canonical reconstruction
# and verified by the precondition. YAML defaults point --in-ply == --out-ply at
# <scene>_bg.ply so the bridge only writes the <scene>_bg.ply.pose.json sidecar.
BG_PLY="${DATA_DIR}/auto_bg/splat/export/${SCENE}_bg.ply"
run "6 bridge_to_og" "$SIMFOUNDRY_ENV" -- \
    python scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/6_bridge_bg_splat_to_og.py \
        scene_name="${SCENE}"

run "7 build_assets" "$SIMFOUNDRY_ENV" -- \
    python scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/7_build_og_scene_assets.py \
        scene_name="${SCENE}"

# ---------- final output ----------
FINAL_JSON="${ASSETS_DIR}/${SCENE}_scene_state_auto_bg.json"
if [[ -f "$FINAL_JSON" ]]; then
    echo
    echo "[orchestrator] ============================================"
    echo "[orchestrator] SUCCESS"
    echo "[orchestrator]   scene state: $FINAL_JSON"
    echo "[orchestrator]   bg splat:    ${BG_PLY}"
    echo "[orchestrator]   void frames: ${AUTO_BG_DIR}/clean_frames/"
    echo "[orchestrator]   logs:        $LOG_DIR"
    echo "[orchestrator] ============================================"
else
    echo "[orchestrator] FAILURE: $FINAL_JSON not produced"
    exit 1
fi
