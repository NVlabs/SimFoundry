#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Non-destructive checkout helpers for dependency repos under deps/.
#
# SIMFOUNDRY_FORCE_DEP_CHECKOUT=1 skips the guards below. Safe in
# git_safe_checkout_detached (plain `checkout --detach`, never `-f`); destructive in
# git_safe_sync_branch (`checkout -B` + `reset --hard`).

# Print a warning explaining why a checkout was left alone.
_git_safe_skip_notice() {
  local repo_dir="$1" reason="$2" target="$3"
  echo "" >&2
  echo "NOTE: leaving ${repo_dir} as-is (${reason})." >&2
  echo "      Not checking out ${target}, so your local work is preserved." >&2
  echo "      Commit or stash your changes, or set SIMFOUNDRY_FORCE_DEP_CHECKOUT=1 to check out anyway." >&2
  echo "" >&2
}

# True when the working tree or index has changes worth protecting.
git_safe_is_dirty() {
  local repo_dir="${1:?repo dir is required}"
  [[ -n "$(git -C "${repo_dir}" status --porcelain --untracked-files=no 2>/dev/null)" ]]
}

# Check out a pinned commit without ever discarding local work.
#   git_safe_checkout_detached <repo_dir> <commit-ish> [label]
git_safe_checkout_detached() {
  local repo_dir="${1:?repo dir is required}"
  local target="${2:?commit-ish is required}"
  local label="${3:-${repo_dir}}"

  if [[ "${SIMFOUNDRY_FORCE_DEP_CHECKOUT:-0}" == "1" ]]; then
    git -C "${repo_dir}" checkout --detach "${target}"
    return
  fi

  # Already there: nothing to do, and no chance of disturbing anything.
  local head_sha target_sha
  head_sha="$(git -C "${repo_dir}" rev-parse HEAD 2>/dev/null || true)"
  target_sha="$(git -C "${repo_dir}" rev-parse "${target}^{commit}" 2>/dev/null || true)"
  if [[ -n "${head_sha}" && "${head_sha}" == "${target_sha}" ]]; then
    return
  fi

  if git_safe_is_dirty "${repo_dir}"; then
    _git_safe_skip_notice "${label}" "it has uncommitted local changes" "${target}"
    return
  fi

  # Don't move off a branch carrying local development, i.e. commits not on its own remote.
  # Comparing against the pin instead would skip every clean clone, since a pin is older than
  # the remote's default branch by construction.
  local branch
  branch="$(git -C "${repo_dir}" symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
  if [[ -n "${branch}" ]]; then
    local upstream unpushed
    upstream="$(git -C "${repo_dir}" rev-parse --verify --quiet "${branch}@{upstream}" 2>/dev/null || true)"
    if [[ -z "${upstream}" ]]; then
      # No tracking config; fall back to the conventional remote ref.
      upstream="$(git -C "${repo_dir}" rev-parse --verify --quiet "origin/${branch}" 2>/dev/null || true)"
    fi

    if [[ -z "${upstream}" ]]; then
      # Nothing to compare against, so assume the branch is the user's.
      _git_safe_skip_notice "${label}" "branch '${branch}' has no remote to compare against" "${target}"
      return
    fi

    unpushed="$(git -C "${repo_dir}" rev-list --count "${upstream}..HEAD" 2>/dev/null || true)"
    if [[ -n "${unpushed}" && "${unpushed}" != "0" ]]; then
      _git_safe_skip_notice "${label}" "branch '${branch}' has ${unpushed} commit(s) not on its remote" "${target}"
      return
    fi
  fi

  git -C "${repo_dir}" checkout --detach "${target}"
}

# Fast-forward a repo to a remote branch, never rewriting local history.
#   git_safe_sync_branch <repo_dir> <remote> <branch> [label]
git_safe_sync_branch() {
  local repo_dir="${1:?repo dir is required}"
  local remote="${2:?remote is required}"
  local branch="${3:?branch is required}"
  local label="${4:-${repo_dir}}"

  git -C "${repo_dir}" fetch "${remote}" "${branch}"

  if [[ "${SIMFOUNDRY_FORCE_DEP_CHECKOUT:-0}" == "1" ]]; then
    git -C "${repo_dir}" checkout -B "${branch}" "${remote}/${branch}"
    git -C "${repo_dir}" reset --hard "${remote}/${branch}"
    return
  fi

  if git_safe_is_dirty "${repo_dir}"; then
    _git_safe_skip_notice "${label}" "it has uncommitted local changes" "${remote}/${branch}"
    return
  fi

  local current
  current="$(git -C "${repo_dir}" symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
  if [[ -n "${current}" && "${current}" != "${branch}" ]]; then
    _git_safe_skip_notice "${label}" "it is on branch '${current}', not '${branch}'" "${remote}/${branch}"
    return
  fi

  # --ff-only fails rather than rewriting local commits.
  if ! git -C "${repo_dir}" merge --ff-only "${remote}/${branch}" 2>/dev/null; then
    _git_safe_skip_notice "${label}" "it has diverged from ${remote}/${branch}" "${remote}/${branch}"
  fi
}
