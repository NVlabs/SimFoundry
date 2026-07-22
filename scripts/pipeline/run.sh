#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/pipeline/run.sh A_reconstruction|B_augmentation|C_application [options]

Aliases:
  reconstruction, A
  augmentation, B
  application, C
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

PIPELINE="$1"
shift

case "${PIPELINE}" in
  A|A_reconstruction|reconstruction)
    exec "${SCRIPT_DIR}/A_reconstruction/run.sh" "$@"
    ;;
  B|B_augmentation|augmentation)
    exec "${SCRIPT_DIR}/B_augmentation/run.sh" "$@"
    ;;
  C|C_application|application)
    exec "${SCRIPT_DIR}/C_application/run.sh" "$@"
    ;;
  -h|--help)
    usage
    ;;
  *)
    echo "Unknown pipeline '${PIPELINE}'." >&2
    usage >&2
    exit 2
    ;;
esac
