#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# install_pixal3d.sh — install Pixal3D (TencentARC/Pixal3D) on top of an existing TRELLIS.2
# env, so stage 7 can run with `s7_mesh.shape_model=pixal3d s7_mesh.texture_model=pixal3d`.
#
# Pixal3D is NOT installed by install_everything.sh — it is opt-in.
#
# Pixal3D is built on the TRELLIS.2 backbone and reuses its CUDA extension stack
# (o-voxel, nvdiffrast, nvdiffrec, CuMesh, FlexGEMM), so this script does NOT rebuild them.
# It requires an env that install_trellis.sh has already prepared, and adds on top:
#   Pixal3D checkout (conda-develop) + its requirements + natten + utils3d
#
# IMPORTANT — why the default env is 'pixal3d' and not 'simfoundry':
#   Pixal3D's requirements.txt pins transformers==4.57.3 and pillow==12.0.0, whereas
#   install_trellis.sh pins transformers==4.57.6 and installs pillow-simd. Installing Pixal3D
#   into the shared 'simfoundry' env would downgrade transformers and replace pillow-simd for
#   every other stage. Build it in its own env instead:
#
#     bash scripts/installation/install_trellis.sh  --env-name pixal3d   # base + extensions
#     bash scripts/installation/install_pixal3d.sh  --env-name pixal3d   # this script
#
#   then run the pipeline with `--env-mesh pixal3d`.
#
# Usage:
#   bash scripts/installation/install_pixal3d.sh [--project-root DIR] [--env-name NAME]
#                                                [--yes] [--default]
#                                                [--cuda-arch-list LIST] [--natten-workers N]
#                                                [--allow-shared-env]
#
#   --project-root DIR     Repo root; Pixal3D is cloned to <root>/deps. Default: repo root.
#   --env-name NAME        Target mamba env. Default: pixal3d
#   --yes, -y              Assume "yes" for prompts.
#   --default              Fully non-interactive (implies --yes).
#   --cuda-arch-list LIST  NATTEN_CUDA_ARCH override, e.g. "8.9". Default: auto-detect.
#   --cuda-version X.Y     CUDA toolkit to build natten against (needs >= 12.0). Default: 12.8
#   --natten-workers N     Build parallelism for natten. Default: half the CPUs.
#   --allow-shared-env     Permit installing into 'simfoundry' despite the pin conflicts above.
#   --skip-weights         Do not pre-download pinned model snapshots / NAF. Not recommended:
#                          the backend then resolves mutable Hugging Face repos at runtime.
#
# Prereqs: mamba (Miniforge) on PATH, and an env already built by install_trellis.sh.
#
# Model weights: revision-pinned snapshots of TencentARC/Pixal3D, MoGe-2 and DINOv3 ARE
# downloaded here, into deps/pixal3d-weights/, along with a checksummed NAF checkout and
# checkpoint placed in torch's hub cache (--skip-weights opts out, at the cost of
# reproducibility). DINOv3 is gated: accept its terms and run `hf auth login` first.
# Pixal3D's non-commercial background remover (briaai/RMBG-2.0) is deliberately never loaded;
# see the note at the end.
#

# Error handling: Exit and print the offending line and error message on failure
# Resolve the script path up front: this script cd's into deps/Pixal3D, after which a relative
# $0 no longer resolves and the handler printed an empty line instead of the failing command.
SELF_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
error_handler() {
    local exit_code=$?
    local line_no=$1
    echo "Error occurred at line $line_no: $(sed "${line_no}q;d" "${SELF_PATH}" 2>/dev/null)"
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
DEFAULT_ENV_NAME="pixal3d"
ENV_NAME="${DEFAULT_ENV_NAME}"
DEFAULT=false
ASSUME_YES=false
CUDA_ARCH_LIST=""
NATTEN_WORKERS=""
# natten builds from source and its cmake requires CUDAToolkit >= 12.0; this must match the
# toolkit torch was built against (install_trellis.sh uses 12.8).
CUDA_VERSION="12.8"
ALLOW_SHARED_ENV=false
SKIP_WEIGHTS=false

# Pinned upstream SHA. Pixal3D publishes no tags or releases, so a commit is the only
# thing that can be pinned. NOTE: the default branch is `master`; there is no `main`,
# despite what the upstream README's branch table says.
# TODO(SimFoundry): confirm this SHA matches a tested build before release (pinned 2026-08-07).
PIXAL3D_COMMIT="${PIXAL3D_COMMIT:-cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af}"
NATTEN_VERSION="${NATTEN_VERSION:-0.21.0}"
UTILS3D_WHEEL="${UTILS3D_WHEEL:-https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl}"
# Third-party wheel from a personal GitHub release rather than PyPI, so verify its bytes.
# Set UTILS3D_WHEEL_SHA256="" to skip (e.g. when overriding UTILS3D_WHEEL).
UTILS3D_WHEEL_SHA256="${UTILS3D_WHEEL_SHA256-ff63440827d6933807dd06c8a5a2db7e51fd5f33c7f3dddcc766a80e0f419252}"
# Upstream's requirements.txt installs MoGe from an unpinned branch
# (`git+https://github.com/microsoft/MoGe.git`), so a rebuild can silently pick up new code.
# Pin it here instead; see the MoGe substitution below.
MOGE_COMMIT="${MOGE_COMMIT:-925b8ed835a7a9cdb7578ba15c658a0afc969030}"

# Revision-pinned model snapshots, downloaded into deps/pixal3d-weights/ unless --skip-weights.
# The backend (simfoundry/models/mesh_generator.py, Pixal3D.resolve_model_source) prefers these
# local directories over the bare repo ids. This is a correctness control, not just
# reproducibility: neither Pixal3D's from_pretrained nor the DINOv3/MoGe loaders accept a
# `revision`, so a bare repo id resolves to whatever main holds at first run — including the
# pipeline.json that selects the background remover.
PIXAL3D_MODEL_REVISION="${PIXAL3D_MODEL_REVISION:-0b31f9160aa400719af409098bff7936a932f726}"
MOGE_MODEL_REVISION="${MOGE_MODEL_REVISION:-39c4d5e957afe587e04eec59dc2bcc3be5ecd968}"
DINOV3_MODEL_REVISION="${DINOV3_MODEL_REVISION:-ea8dc2863c51be0a264bab82070e3e8836b02d51}"

# NAF is loaded by Pixal3D via `torch.hub.load("valeoai/NAF", ..., trust_repo=True)` with no
# revision (deps/Pixal3D/pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py), and
# its hubconf then pulls a .pth through torch.hub.load_state_dict_from_url with no integrity
# check. That is arbitrary remote code plus an unverified pickle, resolved at runtime.
# Mitigation: pre-populate torch's hub cache with a pinned checkout and a checksummed
# checkpoint, so the unpinned runtime call finds the cache and fetches nothing.
NAF_COMMIT="${NAF_COMMIT:-37f2dfc180f2de53d98bd601109c0da0dd6b0f43}"
NAF_CKPT_URL="${NAF_CKPT_URL:-https://github.com/valeoai/NAF/releases/download/model/naf_release.pth}"
NAF_CKPT_SHA256="${NAF_CKPT_SHA256:-c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f}"

# Pinning status of everything this backend pulls, so it is clear what a rebuild can change:
#   PINNED   Pixal3D source           - PIXAL3D_COMMIT (git SHA)
#   PINNED   MoGe source              - MOGE_COMMIT (git SHA)
#   PINNED   natten                   - NATTEN_VERSION (built from source for this arch)
#   PINNED   utils3d                  - UTILS3D_WHEEL + UTILS3D_WHEEL_SHA256
#   PINNED   TRELLIS.2 + extensions   - install_trellis.sh
#   PINNED   NAF source + checkpoint  - NAF_COMMIT, NAF_CKPT_SHA256 (prefetched into torch hub)
#   PINNED   HF model snapshots       - *_MODEL_REVISION above, unless --skip-weights

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-root)     project_root="$2"; shift 2 ;;
        --env-name)         ENV_NAME="$2"; shift 2 ;;
        --yes|-y)           ASSUME_YES=true; shift ;;
        --default)          DEFAULT=true; ASSUME_YES=true; shift ;;
        --cuda-arch-list)   CUDA_ARCH_LIST="$2"; shift 2 ;;
        --cuda-version)     CUDA_VERSION="$2"; shift 2 ;;
        --natten-workers)   NATTEN_WORKERS="$2"; shift 2 ;;
        --allow-shared-env) ALLOW_SHARED_ENV=true; shift ;;
        --skip-weights)     SKIP_WEIGHTS=true; shift ;;
        -h|--help)          sed -n '6,51p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

PROJECT_ROOT="$(cd "$project_root" && pwd)"

confirm() {
  if [[ "${ASSUME_YES}" == true ]]; then
    return 0
  fi
  read -r -p "$1 [y/N] " reply
  [[ "${reply}" =~ ^[Yy]$ ]]
}

env_exists() {
  mamba env list | awk '{print $1}' | grep -Fxq "$1"
}

echo "============================================================"
echo "install_pixal3d.sh"
echo "  project_root:  ${PROJECT_ROOT}"
echo "  target env:    ${ENV_NAME}"
echo "  Pixal3D SHA:   ${PIXAL3D_COMMIT}"
echo "============================================================"

# ==============================================================================
# PRECONDITIONS
# ==============================================================================
if [[ "${ENV_NAME}" == "simfoundry" && "${ALLOW_SHARED_ENV}" != true ]]; then
  cat >&2 <<EOF

ERROR: refusing to install Pixal3D into the shared 'simfoundry' env.

Pixal3D pins transformers==4.57.3 and pillow==12.0.0. install_trellis.sh pins
transformers==4.57.6 and pillow-simd, and every other SimFoundry stage runs in that env.
Installing here would silently downgrade them pipeline-wide.

Build a dedicated env instead:

  bash scripts/installation/install_trellis.sh  --env-name ${DEFAULT_ENV_NAME}
  bash scripts/installation/install_pixal3d.sh  --env-name ${DEFAULT_ENV_NAME}

then run the pipeline with --env-mesh ${DEFAULT_ENV_NAME}.

Pass --allow-shared-env to override.
EOF
  exit 1
fi

if ! env_exists "${ENV_NAME}"; then
  cat >&2 <<EOF

ERROR: mamba env '${ENV_NAME}' does not exist.

Pixal3D builds on the TRELLIS.2 backbone and reuses its CUDA extensions rather than
rebuilding them. Create the base env first:

  bash scripts/installation/install_trellis.sh --env-name ${ENV_NAME}

then rerun this script.
EOF
  exit 1
fi

mamba activate "${ENV_NAME}"

# torch and the TRELLIS.2 extension stack must already be present.
if ! python -c 'import torch' >/dev/null 2>&1; then
  echo "ERROR: torch is not importable in env '${ENV_NAME}'." >&2
  echo "       Build the env with install_trellis.sh --env-name ${ENV_NAME} first." >&2
  exit 1
fi

# o_voxel is the one extension Pixal3D calls directly (postprocess.to_glb). Its presence is a
# good proxy for "install_trellis.sh has run here".
if ! python -c 'import o_voxel' >/dev/null 2>&1; then
  cat >&2 <<EOF

ERROR: o_voxel is not importable in env '${ENV_NAME}'.

Pixal3D exports GLBs through o_voxel.postprocess.to_glb, which ships inside the TRELLIS.2
checkout. Install the backbone first:

  bash scripts/installation/install_trellis.sh --env-name ${ENV_NAME}

then rerun this script.
EOF
  exit 1
fi

echo ""
echo "WARNING: Pixal3D's requirements pin packages that differ from TRELLIS.2's:"
echo "           transformers==4.57.3   (install_trellis.sh pins 4.57.6)"
echo "           pillow==12.0.0         (install_trellis.sh installs pillow-simd)"
echo "         These will be changed inside '${ENV_NAME}'."
if ! confirm "Continue installing Pixal3D into '${ENV_NAME}'?"; then
  echo "Aborted."
  exit 1
fi

# ==============================================================================
# BUILD TOOLCHAIN (natten compiles device code)
# ==============================================================================
# Resolution order matters. Preferring whatever `nvcc` happens to be on PATH picks up a distro
# toolkit (e.g. /usr/bin/nvcc from CUDA 11.5) that natten's cmake rejects with
#   "Could NOT find CUDAToolkit: Found unsuitable version 11.5.119, but required is at least 12.0"
# even though torch here is cu128. So try the versioned toolkit first, exactly as
# install_trellis.sh does, and only then fall back.
for candidate in "${CUDA_HOME:-}" "/usr/local/cuda-${CUDA_VERSION}" "${CONDA_PREFIX}" "/usr/local/cuda"; do
  if [[ -n "${candidate}" && -x "${candidate}/bin/nvcc" ]]; then
    CUDA_HOME="${candidate}"
    break
  fi
done
if [[ ! -x "${CUDA_HOME:-}/bin/nvcc" ]] && command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
if [[ ! -x "${CUDA_HOME:-}/bin/nvcc" ]]; then
  echo "ERROR: nvcc was not found (looked in /usr/local/cuda-${CUDA_VERSION}, \$CONDA_PREFIX, \$PATH)." >&2
  echo "       natten is built from source and needs a CUDA toolkit >= 12.0." >&2
  echo "       Pass --cuda-version X.Y or set CUDA_HOME explicitly." >&2
  exit 1
fi
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"

# natten's cmake requires CUDAToolkit >= 12.0; fail here with an actionable message rather than
# 200 lines into a cmake traceback.
NVCC_VERSION="$("${CUDA_HOME}/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
if [[ "${NVCC_VERSION%%.*}" -lt 12 ]]; then
  echo "ERROR: nvcc at ${CUDA_HOME}/bin/nvcc is version ${NVCC_VERSION}, but natten requires >= 12.0." >&2
  echo "       Install a 12.x toolkit (e.g. /usr/local/cuda-12.8) or pass --cuda-version X.Y." >&2
  exit 1
fi
echo "Using CUDA toolkit ${NVCC_VERSION} at ${CUDA_HOME}"
export LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH:-}"

if [[ -n "${CUDA_ARCH_LIST}" ]]; then
  NATTEN_ARCH="${CUDA_ARCH_LIST}"
  echo "Using provided NATTEN_CUDA_ARCH: ${NATTEN_ARCH}"
else
  echo "Auto-detecting GPU compute capabilities..."
  NATTEN_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u -V | tr '\n' ';' | sed 's/;$//' || true)"
  if [[ -z "${NATTEN_ARCH}" ]]; then
    echo "ERROR: could not detect GPU compute capability; pass --cuda-arch-list explicitly." >&2
    exit 1
  fi
  echo "Auto-detected NATTEN_CUDA_ARCH: ${NATTEN_ARCH}"
fi

if [[ -z "${NATTEN_WORKERS}" ]]; then
  NATTEN_WORKERS="$(( $(nproc) / 2 ))"
  [[ "${NATTEN_WORKERS}" -lt 1 ]] && NATTEN_WORKERS=1
fi

if ! command -v conda-develop >/dev/null 2>&1; then
  echo "conda-develop not found in '${ENV_NAME}'; installing conda-build..."
  mamba install conda-build -y > /dev/null
fi

# ==============================================================================
# PIXAL3D
# ==============================================================================
if [ ! -d "${PROJECT_ROOT}/deps" ]; then
  mkdir "${PROJECT_ROOT}/deps"
fi
cd "${PROJECT_ROOT}/deps"

# NOTE: default branch is `master`. Cloning -b main fails.
if [ ! -d "Pixal3D" ]; then
  git clone https://github.com/TencentARC/Pixal3D.git
fi
# Checkout is outside the clone guard on purpose: install_trellis.sh keeps it inside, which
# means a pre-existing checkout silently keeps whatever SHA it had. Pin every run instead.
git -C Pixal3D fetch --depth 50 origin "${PIXAL3D_COMMIT}" 2>/dev/null || git -C Pixal3D fetch origin
git -C Pixal3D checkout --detach "${PIXAL3D_COMMIT}"
cd Pixal3D

# Upstream Step 2: additional dependencies. Installed before natten so natten builds against
# the final torch. `gradio` in this file is only needed by app.py, which SimFoundry does not use.
# MoGe is stripped out and reinstalled at a pinned SHA immediately after, because upstream lists
# it as an unpinned git URL.
PINNED_REQS="$(mktemp)"
trap 'rm -f "${PINNED_REQS}"' EXIT
# gradio is only needed by upstream app.py, which SimFoundry does not vendor or run. It is
# unpinned in requirements.txt and drags in a large dependency tree, so drop it too.
grep -v -e 'github.com/microsoft/MoGe' -e '^gradio' requirements.txt > "${PINNED_REQS}"
pip install -r "${PINNED_REQS}"
pip install "git+https://github.com/microsoft/MoGe.git@${MOGE_COMMIT}"

# Upstream Step 3: natten, built from source for this machine's arch.
#
# natten compiles arch-specific cubins and emits NO PTX, so a binary built for one compute
# capability cannot JIT onto another. pip, however, caches the built wheel keyed on
# sdist+interpreter tags only — NATTEN_CUDA_ARCH is NOT part of that key. So a plain
# `pip install natten==X` on a second machine (or after --cuda-arch-list was set wrong the
# first time) silently reuses a wheel built for the wrong GPU: the install "succeeds",
# `import natten.libnatten` succeeds because CUDA module loading is lazy, and stage 7 then dies
# minutes into generation with "no kernel image is available for execution on the device".
#
# Check the installed library for cubins matching every requested arch, and force a genuine
# rebuild (bypassing the wheel cache) when any is missing.
natten_is_current() {
  # Version FIRST: cubins prove which GPU the binary targets, not which natten it is. An older
  # natten built for this same arch passes every cubin check, so without this the installer would
  # keep it and still report NATTEN_VERSION as installed.
  local installed
  installed="$(python -c 'import natten; print(getattr(natten, "__version__", ""))' 2>/dev/null)"
  if [[ "${installed}" != "${NATTEN_VERSION}" ]]; then
    echo "natten ${installed:-<none>} installed but ${NATTEN_VERSION} requested; rebuilding."
    return 1
  fi
  local so
  so="$(python -c 'import glob, os, natten; print((glob.glob(os.path.join(os.path.dirname(natten.__file__), "libnatten*.so")) or [""])[0])' 2>/dev/null)"
  [[ -n "${so}" && -f "${so}" ]] || return 1
  command -v cuobjdump >/dev/null 2>&1 || return 1   # cannot verify -> force rebuild
  local listing arch sm
  listing="$(cuobjdump -lelf "${so}" 2>/dev/null)" || return 1
  for arch in ${NATTEN_ARCH//;/ }; do
    sm="sm_${arch//./}"
    grep -q "${sm}" <<<"${listing}" || return 1
  done
  return 0
}

if python -c 'import natten' >/dev/null 2>&1 && natten_is_current; then
  echo "natten==${NATTEN_VERSION} already installed with cubins for arch '${NATTEN_ARCH}'; skipping rebuild."
else
  echo "Building natten==${NATTEN_VERSION} for arch '${NATTEN_ARCH}' with ${NATTEN_WORKERS} workers (this is slow)..."
  # --no-cache-dir is load-bearing: without it pip serves the previously built wheel and
  # NATTEN_CUDA_ARCH is never consulted. --no-deps keeps --force-reinstall away from pinned torch.
  NATTEN_CUDA_ARCH="${NATTEN_ARCH}" NATTEN_N_WORKERS="${NATTEN_WORKERS}" \
    pip install "natten==${NATTEN_VERSION}" \
      --no-build-isolation --force-reinstall --no-deps --no-cache-dir
fi

# Upstream Step 4: utils3d wheel. Downloaded first so its digest can be checked before install.
# Download into a temp DIRECTORY keeping the original basename: pip parses the wheel filename for
# name/version/tags and rejects an mktemp-style name with
#   "Invalid wheel filename (wrong number of parts)"
# regardless of a .whl suffix.
UTILS3D_TMPDIR="$(mktemp -d)"
UTILS3D_LOCAL="${UTILS3D_TMPDIR}/$(basename "${UTILS3D_WHEEL}")"
trap 'rm -f "${PINNED_REQS}"; rm -rf "${UTILS3D_TMPDIR}"' EXIT
curl -fsSL -o "${UTILS3D_LOCAL}" "${UTILS3D_WHEEL}"
if [[ -n "${UTILS3D_WHEEL_SHA256}" ]]; then
  echo "${UTILS3D_WHEEL_SHA256}  ${UTILS3D_LOCAL}" | sha256sum --check --status \
    || { echo "ERROR: utils3d wheel digest mismatch (expected ${UTILS3D_WHEEL_SHA256})." >&2
         echo "       Actual: $(sha256sum "${UTILS3D_LOCAL}" | cut -d' ' -f1)" >&2; exit 1; }
  echo "utils3d wheel digest verified."
else
  echo "NOTE: UTILS3D_WHEEL_SHA256 is unset; installing ${UTILS3D_WHEEL} without integrity check."
  echo "      Record its digest with: sha256sum <wheel>"
fi
pip install "${UTILS3D_LOCAL}"

# Re-assert after the torch-dependent installs above: pip resolves torch's exact
# triton==3.3.0 pin and silently downgrades, which breaks sm_120 (Blackwell).
pip install "triton>=3.3.1,<3.4"

# Put the checkout on the env's import path. The backend loads inference.py by absolute path,
# but `import pixal3d` must resolve for it.
conda-develop .

# ==============================================================================
# PINNED MODEL SNAPSHOTS + NAF PREFETCH
# ==============================================================================
WEIGHTS_DIR="${PROJECT_ROOT}/deps/pixal3d-weights"

if [[ "${SKIP_WEIGHTS}" == true ]]; then
  echo ""
  echo "WARNING: --skip-weights given. The backend will resolve mutable Hugging Face repos at"
  echo "         runtime, and Pixal3D will fetch NAF's code and checkpoint unpinned and"
  echo "         unverified on first use. Not suitable for a reproducible or audited install."
else
  echo ""
  echo "Downloading revision-pinned model snapshots into ${WEIGHTS_DIR} ..."
  mkdir -p "${WEIGHTS_DIR}"

  # huggingface_hub >= 1.0 retired the `huggingface-cli` entry point. It still resolves on PATH,
  # so `command -v` finds it, but every invocation prints "deprecated and no longer works" and
  # exits 1 — which under `set -o errexit` aborted this script with a misleading
  # "failed to download <repo> at revision <sha>" plus the gating hint, sending users to
  # re-check HF credentials that were fine. Resolve `hf` first and keep the old name only as a
  # fallback for pre-1.0 hubs. Same order as download_checkpoints.sh.
  HF_CLI="$(command -v hf || command -v huggingface-cli || true)"
  if [[ -z "${HF_CLI}" ]]; then
    echo "" >&2
    echo "ERROR: neither 'hf' nor 'huggingface-cli' is on PATH in env '${ENV_NAME}'." >&2
    echo "       Install it with:  pip install -U 'huggingface_hub[cli]'" >&2
    echo "       Or rerun with --skip-weights to skip pinned snapshots (not recommended)." >&2
    exit 1
  fi

  download_snapshot() {
    local repo="$1" revision="$2" subdir="$3" gated_hint="$4"
    local dest="${WEIGHTS_DIR}/${subdir}"
    if [[ -f "${dest}/.simfoundry-revision" ]] \
       && [[ "$(cat "${dest}/.simfoundry-revision")" == "${revision}" ]]; then
      echo "  ${repo}@${revision:0:8} already present."
      return 0
    fi
    echo "  ${repo}@${revision:0:8} -> ${dest}"
    if ! "${HF_CLI}" download "${repo}" --revision "${revision}" --local-dir "${dest}" >/dev/null; then
      echo "" >&2
      echo "ERROR: failed to download ${repo} at revision ${revision}." >&2
      [[ -n "${gated_hint}" ]] && echo "       ${gated_hint}" >&2
      return 1
    fi
    echo "${revision}" > "${dest}/.simfoundry-revision"
  }

  download_snapshot "TencentARC/Pixal3D" "${PIXAL3D_MODEL_REVISION}" "Pixal3D" ""
  download_snapshot "Ruicheng/moge-2-vitl" "${MOGE_MODEL_REVISION}" "moge-2-vitl" ""
  download_snapshot "facebook/dinov3-vitl16-pretrain-lvd1689m" "${DINOV3_MODEL_REVISION}" \
    "dinov3-vitl16-pretrain-lvd1689m" \
    "DINOv3 is GATED: accept its terms on the model page, then run 'hf auth login'."

  # NAF: pre-populate torch's hub cache so the unpinned runtime torch.hub.load finds it and
  # fetches nothing. torch names the directory <owner>_<repo>_<ref>; NAF's default ref is main.
  echo ""
  echo "Prefetching NAF (pinned ${NAF_COMMIT:0:8}) into torch's hub cache ..."
  TORCH_HUB_DIR="$(python -c 'import torch.hub; print(torch.hub.get_dir())')"
  NAF_DIR="${TORCH_HUB_DIR}/valeoai_NAF_main"
  mkdir -p "${TORCH_HUB_DIR}/checkpoints"
  if [[ ! -d "${NAF_DIR}/.git" ]]; then
    rm -rf "${NAF_DIR}"
    git clone https://github.com/valeoai/NAF.git "${NAF_DIR}"
  fi
  git -C "${NAF_DIR}" fetch origin "${NAF_COMMIT}" 2>/dev/null || git -C "${NAF_DIR}" fetch origin
  git -C "${NAF_DIR}" checkout --detach "${NAF_COMMIT}"

  # torch.hub.load_state_dict_from_url caches by basename and skips the download when present.
  NAF_CKPT="${TORCH_HUB_DIR}/checkpoints/$(basename "${NAF_CKPT_URL}")"
  if [[ ! -f "${NAF_CKPT}" ]]; then
    curl -fsSL -o "${NAF_CKPT}" "${NAF_CKPT_URL}"
  fi
  if ! echo "${NAF_CKPT_SHA256}  ${NAF_CKPT}" | sha256sum --check --status; then
    echo "ERROR: NAF checkpoint digest mismatch." >&2
    echo "       expected ${NAF_CKPT_SHA256}" >&2
    echo "       actual   $(sha256sum "${NAF_CKPT}" | cut -d' ' -f1)" >&2
    echo "       Refusing to install an unverified checkpoint; it is deserialized by torch.load." >&2
    rm -f "${NAF_CKPT}"
    exit 1
  fi
  echo "  NAF checkpoint digest verified."
fi

# ==============================================================================
# VERIFY
# ==============================================================================
echo ""
echo "Checking for dependency conflicts (pip check)..."
# This env deliberately mixes TRELLIS.2's and Pixal3D's pins, so some complaints are expected
# and pip check is a warning by default. Set STRICT_PIP_CHECK=1 to make it fail the install,
# which is what a release build should do once the expected conflicts are pinned away.
if ! pip check; then
  if [[ "${STRICT_PIP_CHECK:-0}" == "1" ]]; then
    echo "ERROR: pip reported dependency conflicts and STRICT_PIP_CHECK=1." >&2
    exit 1
  fi
  echo "WARNING: pip reported dependency conflicts (see above). Re-run with STRICT_PIP_CHECK=1 to treat as fatal."
fi

echo ""
echo "Verifying imports and native extensions..."
# A failure here fails the install. Previously this only printed a warning and still exited 0,
# so a broken env looked installed until the first stage 7 run. Run it as an `if` condition so
# the ERR trap does not fire and bury the diagnostics under a stack-trace-style message.
if ! python - <<'PY'
import importlib
import sys

failures = []

for mod in ("torch", "o_voxel", "natten", "utils3d", "pixal3d", "moge", "trimesh"):
    try:
        importlib.import_module(mod)
        print(f"  OK   import {mod}")
    except Exception as exc:
        print(f"  FAIL import {mod}: {type(exc).__name__}: {exc}")
        failures.append(mod)

# natten's Python package imports fine even when its compiled kernels are missing, which then
# surfaces as a runtime error mid-generation. Probe the native side explicitly.
if "natten" not in failures:
    try:
        import natten
        has_cuda = getattr(natten, "has_cuda", None)
        if callable(has_cuda):
            if has_cuda():
                print("  OK   natten CUDA kernels available (natten.has_cuda())")
            else:
                print("  FAIL natten built without CUDA kernels (natten.has_cuda() is False)")
                failures.append("natten:cuda")
        else:
            # Older/newer layouts without has_cuda(): importing the compiled extension directly
            # is the equivalent check.
            importlib.import_module("natten.libnatten")
            print("  OK   natten native extension importable (natten.libnatten)")
    except Exception as exc:
        print(f"  FAIL natten native extension: {type(exc).__name__}: {exc}")
        failures.append("natten:native")

# The backend calls o_voxel.postprocess.to_glb with grid_size=/use_tqdm=; verify this build
# actually accepts them rather than discovering it after a multi-minute generation.
if "o_voxel" not in failures:
    try:
        import inspect
        import o_voxel
        params = inspect.signature(o_voxel.postprocess.to_glb).parameters
        missing = [p for p in ("grid_size", "attr_layout", "use_tqdm") if p not in params]
        if missing:
            print(f"  FAIL o_voxel.postprocess.to_glb is missing parameters: {missing}")
            failures.append("o_voxel:to_glb")
        else:
            print("  OK   o_voxel.postprocess.to_glb accepts grid_size/attr_layout/use_tqdm")
    except Exception as exc:
        print(f"  FAIL o_voxel.postprocess.to_glb check: {type(exc).__name__}: {exc}")
        failures.append("o_voxel:to_glb")

if failures:
    print(f"\nVerification FAILED: {failures}")
    sys.exit(1)
print("\nVerification passed.")
PY
then
  echo ""
  echo "============================================================"
  echo "install_pixal3d.sh: FAILED — the environment is not usable."
  echo "  Fix the failures listed above and rerun. Common causes:"
  echo "    natten built without CUDA / wrong arch ->"
  echo "      pip uninstall -y natten && pip cache remove natten"
  echo "      then rerun with --cuda-arch-list matching your GPU."
  echo "      (pip caches the built wheel and ignores NATTEN_CUDA_ARCH, so a plain rerun is a no-op)"
  echo "    o_voxel signature mismatch -> rerun install_trellis.sh --env-name ${ENV_NAME}"
  echo "============================================================"
  exit 1
fi

echo ""
echo "============================================================"
echo "install_pixal3d.sh: DONE"
echo "============================================================"
cat <<EOF

Run ONLY stage 7 with Pixal3D:

    scripts/pipeline/A_reconstruction/run.sh --env-mesh ${ENV_NAME} --include 7 \\
        s7_mesh.shape_model=pixal3d s7_mesh.texture_model=pixal3d

Or the full pipeline, with streaming disabled:

    scripts/pipeline/A_reconstruction/run.sh --env-mesh ${ENV_NAME} --no-stream \\
        s7_mesh.shape_model=pixal3d s7_mesh.texture_model=pixal3d

IMPORTANT: stage 5-8 streaming is ENABLED BY DEFAULT, and it relaunches stage 7 as a fresh
process per object. That reloads Pixal3D, four DINOv3 encoders, MoGe-2 and NAF for every
single object -- minutes of load time each. Use --include 7 or --no-stream until stage 7
grows a persistent worker.

Pixal3D generates shape and texture in one pipeline, so both must be set to 'pixal3d'.

Tune generation via s7_mesh.generation_kwargs. That dict starts empty, so adding a key on
the command line needs Hydra's '+' prefix:

    +s7_mesh.generation_kwargs.resolution=1024 +s7_mesh.generation_kwargs.fov=0.2

(Keys already present in the YAML are overridden with plain key=value.)

Before the first run:

  * One GATED model must be accepted on Hugging Face, then authenticate once with
    \`hf auth login\`:

      - facebook/dinov3-vitl16-pretrain-lvd1689m  (manual approval)
        SimFoundry loads this official Meta repo rather than the ungated third-party
        mirror hardcoded upstream.

    Without it, the first stage 7 run fails with a gated-repo error at model load.

  * Pixal3D's config also selects briaai/RMBG-2.0 for background removal, which is gated and
    CC-BY-NC-4.0 (NON-COMMERCIAL). SimFoundry does NOT load it: a stub is substituted before
    the pipeline is built, so this integration does not download, execute, or use BRIA code or
    weights. Evaluate BRIA's terms yourself if you re-enable it. The trade-off is that inputs
    MUST be RGBA with alpha already isolating the object -- which is exactly what stage 6 writes
    as *_transparent.png. Other inputs are rejected with an explanatory error.

  * Peak VRAM is ~18 GB at the default 1536 resolution. On a 24 GB card set
    s7_mesh.low_vram=true (~10-12 GB, resolution 1024) and check stream_subseq.stage_vram_gb.

  * Pixal3D is pixel-aligned: it generates in the input view's frame rather than a canonical
    one. Validate stage 8 pose matching before relying on it for a full scene.
EOF
