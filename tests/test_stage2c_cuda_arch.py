# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stage 2c builds gsplat's CUDA extension for a specific architecture. A wrong value is silent
at build time and fatal at the first kernel launch ("no kernel image is available for
execution on the device"), so the selection must reflect the machine or refuse to answer.
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


def test_reports_the_installed_gpu(stage, monkeypatch):
    monkeypatch.delenv("SIMFOUNDRY_CUDA_ARCH", raising=False)
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device visible")

    major, minor = torch.cuda.get_device_capability(0)

    assert f"{major}.{minor}" in stage.resolve_cuda_arch().split(";")


def test_covers_every_visible_gpu_not_just_device_zero(stage, monkeypatch):
    monkeypatch.delenv("SIMFOUNDRY_CUDA_ARCH", raising=False)
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    # A mixed host: building only for device 0 yields a binary that dies on the others.
    caps = {0: (8, 6), 1: (9, 0), 2: (8, 6)}
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: caps[i])

    assert stage.resolve_cuda_arch() == "8.6;9.0"


def test_raises_when_no_device_is_visible(stage, monkeypatch):
    monkeypatch.delenv("SIMFOUNDRY_CUDA_ARCH", raising=False)
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Must NOT guess an architecture: a guess only relocates the failure to first kernel launch.
    with pytest.raises(RuntimeError, match="no CUDA device available"):
        stage.resolve_cuda_arch()


def test_raises_when_the_driver_query_fails(stage, monkeypatch):
    monkeypatch.delenv("SIMFOUNDRY_CUDA_ARCH", raising=False)
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def boom(_index):
        raise RuntimeError("CUDA driver version is insufficient")

    monkeypatch.setattr(torch.cuda, "get_device_capability", boom)

    with pytest.raises(RuntimeError) as excinfo:
        stage.resolve_cuda_arch()

    # The driver's own message must survive — it is the actionable part.
    assert "CUDA driver version is insufficient" in str(excinfo.value)


def test_failure_message_names_the_override(stage, monkeypatch):
    monkeypatch.delenv("SIMFOUNDRY_CUDA_ARCH", raising=False)
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="SIMFOUNDRY_CUDA_ARCH"):
        stage.resolve_cuda_arch()


def test_empty_override_does_not_win(stage, monkeypatch):
    # An exported-but-empty variable must not select "" as the arch, which would produce an
    # empty TORCH_CUDA_ARCH_LIST and a confusing build failure.
    monkeypatch.setenv("SIMFOUNDRY_CUDA_ARCH", "")
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (8, 6))

    assert stage.resolve_cuda_arch() == "8.6"
