# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

eval "$(mamba shell hook --shell bash)"

# Get script dir
# repo dir is grandparent directory, by default
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
project_root="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT=false

# Parse command-line options
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-root) project_root="$2"; shift 2 ;;
        --default) DEFAULT=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! ${DEFAULT} == true ]]; then
  read -p "Enter project root (default: $project_root): " PROJECT_ROOT
fi
PROJECT_ROOT=${PROJECT_ROOT:-$project_root}

# ==============================================================================
# ARTICULATE ENVIRONMENT SETUP
# ==============================================================================

echo "=== Articulate Environment Setup ==="

cd $PROJECT_ROOT  # Change to your project root

# Public articulate-anything fork (SimFoundry release). Override ARTICULATE_ANYTHING_REPO /
# ARTICULATE_ANYTHING_BRANCH to point at a different mirror. Large assets (embeddings) are
# stored in Git LFS, so git-lfs must be installed (handled above).
export ARTICULATE_ANYTHING_REPO="${ARTICULATE_ANYTHING_REPO:-https://github.com/nadunRanawaka1/articulate-anything-sf.git}"
export ARTICULATE_ANYTHING_BRANCH="${ARTICULATE_ANYTHING_BRANCH:-oss_release}"
export ARTICULATE_ANYTHING_COMMIT="${ARTICULATE_ANYTHING_COMMIT:-a6759b62992bcea9fecb6c1259e6a4311bda4ee4}"

# ==============================================================================
# SYSTEM DEPENDENCIES (run first)
# ==============================================================================
# Everything that needs sudo lives here at the top, so the password prompt
# appears immediately instead of hours later, after the long unattended conda
# env builds. Do the interactive/sudo work up front, then walk away.

echo "=== Installing system dependencies (this step may prompt for your sudo password) ==="
# Prime sudo now so the credential is cached for the rest of this block.
sudo -v

# Git LFS is required to fetch large assets (embeddings, meshes) on clone.
if ! command -v git-lfs >/dev/null 2>&1; then
  sudo apt-get install -y git-lfs
fi
git lfs install

# Do NOT use the Ubuntu 'blender' apt package: it is a stripped build that runs
# the system python3 (missing numpy) and is compiled WITHOUT OpenImageDenoise,
# which makes articulate-anything's Cycles renders fail with
# "Error: Build without OpenImageDenoiser". We install the official blender.org
# build below instead, which bundles its own Python (with numpy) and OIDN.
echo "=== Installing ffmpeg + Blender runtime libraries ==="
sudo apt-get install -y ffmpeg \
  libxrender1 libxi6 libxxf86vm1 libxfixes3 libxkbcommon0 libsm6 libgl1

echo "=== Installing Blender (official blender.org build) ==="
# Override BLENDER_VERSION to pin a different release.
BLENDER_VERSION="${BLENDER_VERSION:-4.2.3}"
BLENDER_SERIES="${BLENDER_VERSION%.*}"                 # e.g. 4.2.3 -> 4.2
BLENDER_INSTALL_DIR="${BLENDER_INSTALL_DIR:-/opt/blender-${BLENDER_VERSION}}"

if blender --version 2>/dev/null | grep -q "Blender ${BLENDER_VERSION}"; then
  echo "Blender ${BLENDER_VERSION} already installed, skipping download."
else
  BLENDER_TARBALL="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
  BLENDER_URL="https://download.blender.org/release/Blender${BLENDER_SERIES}/${BLENDER_TARBALL}"
  TMP_BLENDER="$(mktemp -d)"
  wget -O "${TMP_BLENDER}/${BLENDER_TARBALL}" "${BLENDER_URL}"
  sudo rm -rf "${BLENDER_INSTALL_DIR}"
  sudo mkdir -p "${BLENDER_INSTALL_DIR}"
  sudo tar -xf "${TMP_BLENDER}/${BLENDER_TARBALL}" -C "${BLENDER_INSTALL_DIR}" --strip-components=1
  # Symlink into /usr/local/bin (takes precedence over any /usr/bin/blender).
  sudo ln -sf "${BLENDER_INSTALL_DIR}/blender" /usr/local/bin/blender
  rm -rf "${TMP_BLENDER}"
  hash -r
  echo "Blender installed: $(blender --version 2>/dev/null | head -1)"
fi

# ==============================================================================
# REPOSITORIES & CONDA ENVIRONMENTS (long-running, unattended)
# ==============================================================================

if [ ! -d "deps" ]; then
  mkdir deps
fi
cd deps

if [ ! -d "articulate-anything" ]; then
  git clone -b "${ARTICULATE_ANYTHING_BRANCH}" "${ARTICULATE_ANYTHING_REPO}" articulate-anything
fi

cd articulate-anything
git fetch origin "${ARTICULATE_ANYTHING_BRANCH}"
git checkout --detach "${ARTICULATE_ANYTHING_COMMIT}"

bash installation_hunyuan.sh   # create hunyuan environment
bash installation_partfield.sh # create partfield environment

# ==============================================================================
# LIBIGL (watertight mesh conversion, required in hunyuan env)
# ==============================================================================

echo "=== Installing libigl in articulate-anything-hunyuan ==="
mamba activate articulate-anything-hunyuan
conda install -c conda-forge igl -y
mamba deactivate

# ==============================================================================
# DOWNLOAD MODEL CHECKPOINTS
# ==============================================================================

echo "=== Downloading P3-SAM weights ==="
P3SAM_WEIGHTS_DIR="deps/Hunyuan3D-Part/P3-SAM/weights"
mkdir -p $P3SAM_WEIGHTS_DIR
if [ ! -f "$P3SAM_WEIGHTS_DIR/p3sam.safetensors" ]; then
    wget -O $P3SAM_WEIGHTS_DIR/p3sam.safetensors \
        https://huggingface.co/tencent/Hunyuan3D-Part/resolve/main/p3sam/p3sam.safetensors
else
    echo "P3-SAM weights already present, skipping download."
fi

