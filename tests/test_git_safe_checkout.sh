#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Behaviour tests for git_safe_checkout_detached against real throwaway repos.
#
# The bug these cover: the guard used to ask "does HEAD have commits the pin lacks?", which is
# true of every freshly cloned dependency (a pin is older than the remote's default branch by
# construction). Every clean clone was therefore skipped, leaving the install on the wrong
# revision and failing ~40 minutes later with an unrelated ImportError.
#
# Usage: bash tests/test_git_safe_checkout.sh [path/to/git_safe.sh]
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GIT_SAFE="${1:-${REPO_ROOT}/scripts/installation/git_safe.sh}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

export GIT_AUTHOR_NAME=test GIT_AUTHOR_EMAIL=test@example.com
export GIT_COMMITTER_NAME=test GIT_COMMITTER_EMAIL=test@example.com

# shellcheck disable=SC1090
source "${GIT_SAFE}"

PASS=0
FAIL=0

check() {  # check <description> <expected> <actual>
  if [[ "$2" == "$3" ]]; then
    echo "  PASS  $1"
    PASS=$((PASS + 1))
  else
    echo "  FAIL  $1 (expected '$2', got '$3')"
    FAIL=$((FAIL + 1))
  fi
}

# Build an "upstream" remote and a clone of it. The remote's main is ahead of PIN, exactly like
# a real dependency repo whose default branch has moved on past the pinned commit.
make_clone() {  # make_clone <name>; echoes "<clone_dir> <pin_sha>"
  local name="$1"
  local remote="${WORK}/${name}_remote" clone="${WORK}/${name}"
  rm -rf "${remote}" "${clone}"
  mkdir -p "${remote}"
  git -C "${remote}" init -q -b main
  echo pinned > "${remote}/pinned.txt"
  echo shared > "${remote}/shared.txt"
  git -C "${remote}" add -A
  git -C "${remote}" commit -qm "pin target"
  local pin
  pin="$(git -C "${remote}" rev-parse HEAD)"
  echo newer > "${remote}/pinned.txt"
  git -C "${remote}" add -A
  git -C "${remote}" commit -qm "upstream moved on"
  git -q clone -q "${remote}" "${clone}" 2>/dev/null || git clone -q "${remote}" "${clone}"
  echo "${clone} ${pin}"
}

head_is_pin() {  # head_is_pin <dir> <pin> -> "yes"/"no"
  [[ "$(git -C "$1" rev-parse HEAD)" == "$2" ]] && echo yes || echo no
}

echo "=== the regression: a clean fresh clone must be moved onto the pin ==="
read -r D PIN <<< "$(make_clone fresh)"
git_safe_checkout_detached "${D}" "${PIN}" "deps/fresh" >/dev/null 2>&1
check "clean clone ahead of the pin is checked out" yes "$(head_is_pin "${D}" "${PIN}")"

echo
echo "=== genuine local work is still protected ==="
read -r D PIN <<< "$(make_clone localwork)"
echo "my change" >> "${D}/shared.txt"
git -C "${D}" add -A
git -C "${D}" commit -qm "user's unpushed commit"
git_safe_checkout_detached "${D}" "${PIN}" "deps/localwork" >/dev/null 2>&1
check "unpushed commit blocks the checkout" no "$(head_is_pin "${D}" "${PIN}")"
check "the unpushed commit is still there" 1 \
  "$(git -C "${D}" log --oneline | grep -c "user's unpushed commit")"

echo
echo "=== a dirty worktree is still protected ==="
read -r D PIN <<< "$(make_clone dirty)"
echo "uncommitted" >> "${D}/shared.txt"
git_safe_checkout_detached "${D}" "${PIN}" "deps/dirty" >/dev/null 2>&1
check "uncommitted change blocks the checkout" no "$(head_is_pin "${D}" "${PIN}")"
check "the uncommitted change survives" 1 "$(grep -c uncommitted "${D}/shared.txt")"

echo
echo "=== a branch with no remote to compare against is left alone ==="
read -r D PIN <<< "$(make_clone noupstream)"
git -C "${D}" checkout -q -b private-branch
git_safe_checkout_detached "${D}" "${PIN}" "deps/noupstream" >/dev/null 2>&1
check "branch without an upstream is skipped" no "$(head_is_pin "${D}" "${PIN}")"

echo
echo "=== already on the pin is a no-op ==="
read -r D PIN <<< "$(make_clone already)"
git -C "${D}" checkout -q --detach "${PIN}"
git_safe_checkout_detached "${D}" "${PIN}" "deps/already" >/dev/null 2>&1
check "no-op when HEAD already equals the pin" yes "$(head_is_pin "${D}" "${PIN}")"

echo
echo "=== the force flag: overrides guards, but never discards work ==="
read -r D PIN <<< "$(make_clone forcework)"
echo "my change" >> "${D}/shared.txt"
git -C "${D}" add -A
git -C "${D}" commit -qm "user's unpushed commit"
USER_SHA="$(git -C "${D}" rev-parse main)"
SIMFOUNDRY_FORCE_DEP_CHECKOUT=1 git_safe_checkout_detached "${D}" "${PIN}" "deps/forcework" >/dev/null 2>&1
check "force checks out the pin" yes "$(head_is_pin "${D}" "${PIN}")"
check "force leaves the user's commit on its branch" "${USER_SHA}" "$(git -C "${D}" rev-parse main)"

read -r D PIN <<< "$(make_clone forcedirty)"
echo "uncommitted" >> "${D}/pinned.txt"   # pinned.txt differs between HEAD and the pin
SIMFOUNDRY_FORCE_DEP_CHECKOUT=1 git_safe_checkout_detached "${D}" "${PIN}" "deps/forcedirty" >/dev/null 2>&1
rc=$?
check "force cannot clobber a conflicting local edit (git refuses)" 1 "${rc}"
check "the conflicting edit survives" 1 "$(grep -c uncommitted "${D}/pinned.txt")"

echo
echo "=== git_safe_sync_branch: the force flag IS destructive here (documented, not a bug) ==="
# Unlike git_safe_checkout_detached, this path runs `checkout -B` + `reset --hard`. Pinned so
# the asymmetry between the two functions stays deliberate and documented.
read -r D PIN <<< "$(make_clone syncforce)"
echo "uncommitted" >> "${D}/shared.txt"
git -C "${D}" add -A
git -C "${D}" commit -qm "user's unpushed commit"
echo "uncommitted again" >> "${D}/shared.txt"
SIMFOUNDRY_FORCE_DEP_CHECKOUT=1 git_safe_sync_branch "${D}" origin main "deps/syncforce" >/dev/null 2>&1
check "force discards the unpushed commit from the branch" 0 \
  "$(git -C "${D}" log --oneline main | grep -c "user's unpushed commit")"
check "force discards the uncommitted change" 0 "$(grep -c "uncommitted again" "${D}/shared.txt" || true)"

# Without the flag, the same repo must be left alone.
read -r D PIN <<< "$(make_clone syncsafe)"
echo "uncommitted" >> "${D}/shared.txt"
git_safe_sync_branch "${D}" origin main "deps/syncsafe" >/dev/null 2>&1
check "without force, the uncommitted change survives" 1 "$(grep -c uncommitted "${D}/shared.txt")"

echo
echo "=== ${PASS} passed, ${FAIL} failed ==="
[[ "${FAIL}" -eq 0 ]]
