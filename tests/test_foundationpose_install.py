# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script_path",
    [
        "scripts/installation/install_simfoundry.sh",
        "scripts/installation/install_any6d.sh",
    ],
)
def test_foundationpose_install_uses_conda_eigen_and_fails_fast(script_path):
    script = (REPO_ROOT / script_path).read_text(encoding="utf-8")

    assert 'EIGEN_INCLUDE="${CONDA_PREFIX}/include/eigen3"' in script
    assert '"${EIGEN_INCLUDE}/Eigen/Dense"' in script
    assert 'CPLUS_INCLUDE_PATH="${EIGEN_INCLUDE}' in script
    assert "bash -e " in script and "build_all_conda.sh" in script
    assert 'python -c "import torch, common, gridencoder"' in script
    assert "FOUNDATIONPOSE_MYCPP_MODULES=" in script


@pytest.mark.parametrize(
    "patch_path",
    ["patches/FoundationPose.patch", "patches/Any6D.patch"],
)
def test_foundationpose_patch_adds_conda_eigen_include(patch_path):
    patch = (REPO_ROOT / patch_path).read_text(encoding="utf-8")

    assert 'os.environ["CONDA_PREFIX"], "include", "eigen3"' in patch
    assert "include_dirs=eigen_include_dirs" in patch
