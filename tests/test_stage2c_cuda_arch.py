# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stage 2c builds gsplat's CUDA extension for one arch. Getting that arch wrong is silent at
build time and fatal at run time ("no kernel image is available for execution on the device"),
after COLMAP has already run — so the selection logic is worth pinning down.
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE = REPO_ROOT / "scripts/pipeline/A_reconstruction/stages/2c_train_bg_splat.py"


@pytest.fixture(scope="module")
def stage():
    # The stage filename starts with a digit, so it cannot be imported by module name.
    spec = importlib.util.spec_from_file_location("stage_2c", STAGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_env_override_wins(stage, monkeypatch):
    monkeypatch.setenv("SIMFOUNDRY_CUDA_ARCH", "9.0")

    assert stage.resolve_cuda_arch() == "9.0"


def test_detects_the_installed_gpu(stage, monkeypatch):
    monkeypatch.delenv("SIMFOUNDRY_CUDA_ARCH", raising=False)
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device visible")

    major, minor = torch.cuda.get_device_capability(0)

    # The whole point: the arch tracks the actual device, not a constant.
    assert stage.resolve_cuda_arch() == f"{major}.{minor}"


def test_falls_back_when_no_device_is_visible(stage, monkeypatch):
    monkeypatch.delenv("SIMFOUNDRY_CUDA_ARCH", raising=False)
    torch = pytest.importorskip("torch")
    # Force the no-GPU branch. Without this the test passes on a GPU host without ever
    # reaching the fallback, which is exactly the kind of test that proves nothing.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert stage.resolve_cuda_arch(fallback="7.5") == "7.5"


def test_falls_back_when_detection_raises(stage, monkeypatch):
    monkeypatch.delenv("SIMFOUNDRY_CUDA_ARCH", raising=False)
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def boom(_index):
        raise RuntimeError("driver unavailable")

    monkeypatch.setattr(torch.cuda, "get_device_capability", boom)

    # A driver error must not propagate out of env setup; it degrades to the fallback.
    assert stage.resolve_cuda_arch(fallback="7.5") == "7.5"


def test_empty_override_does_not_win(stage, monkeypatch):
    # An unset-but-exported variable (SIMFOUNDRY_CUDA_ARCH="") must not select "" as the arch,
    # which would produce an empty TORCH_CUDA_ARCH_LIST and a confusing build failure.
    monkeypatch.setenv("SIMFOUNDRY_CUDA_ARCH", "")

    assert stage.resolve_cuda_arch(fallback="7.5") != ""
