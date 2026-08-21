# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
from pathlib import Path
import sys
import types

import pytest

# These target a refactor of stage 13 into an import_usd_asset() helper that is not in this
# release: 13_import_usd.py defines resolve_pipeline_script, resolve_reparent_script,
# imported_usd_path and main, and does the import inline. Skipped rather than deleted so the
# intended interface survives for whoever lands that refactor -- and so `pytest` stays usable
# as the documented install check.
pytestmark = pytest.mark.skip(
    reason="stage 13 refactor (import_usd_asset) is not in this release"
)


def load_stage13_module(monkeypatch, repo_root: Path, dataset_root: Path):
    asset_utils = types.ModuleType("omnigibson.utils.asset_utils")
    asset_utils.get_dataset_path = lambda dataset_name: str(dataset_root / dataset_name)

    og_utils = types.ModuleType("omnigibson.utils")
    omnigibson = types.ModuleType("omnigibson")
    omnigibson.utils = og_utils

    monkeypatch.setitem(sys.modules, "omnigibson", omnigibson)
    monkeypatch.setitem(sys.modules, "omnigibson.utils", og_utils)
    monkeypatch.setitem(sys.modules, "omnigibson.utils.asset_utils", asset_utils)

    module_path = repo_root / "scripts" / "pipeline" / "A_reconstruction" / "stages" / "13_import_usd.py"
    name = "stage13_import_usd_for_tests"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    cwd = os.getcwd()
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
    return module


def test_import_usd_asset_uses_configured_dataset_and_fails_fast(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    stage13 = load_stage13_module(monkeypatch, repo_root, tmp_path / "datasets")
    urdf_path = tmp_path / "cup.urdf"
    urdf_path.write_text("<robot />", encoding="utf-8")
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        assert check is True
        dataset_name = cmd[cmd.index("--dataset-name") + 1]
        category = cmd[cmd.index("--category") + 1]
        model = cmd[cmd.index("--model") + 1]
        usd_path = Path(stage13.expected_usd_path(dataset_name, category, model))
        usd_path.parent.mkdir(parents=True)
        usd_path.write_text("#usda", encoding="utf-8")

    monkeypatch.setattr(stage13.subprocess, "run", fake_run)

    usd_path = stage13.import_usd_asset(
        dataset_name="real2sim-assets",
        urdf_path=str(urdf_path),
        category="white_cup",
        model="abcdef",
        object_name="iter_1",
    )

    assert Path(usd_path).exists()
    assert "--dataset-name" in calls[0]
    assert calls[0][calls[0].index("--dataset-name") + 1] == "real2sim-assets"


def test_import_usd_asset_raises_when_importer_creates_no_usd(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    stage13 = load_stage13_module(monkeypatch, repo_root, tmp_path / "datasets")
    urdf_path = tmp_path / "cup.urdf"
    urdf_path.write_text("<robot />", encoding="utf-8")
    monkeypatch.setattr(stage13.subprocess, "run", lambda cmd, check: None)

    with pytest.raises(FileNotFoundError, match="USD importer did not create expected asset"):
        stage13.import_usd_asset(
            dataset_name="real2sim-assets",
            urdf_path=str(urdf_path),
            category="white_cup",
            model="abcdef",
            object_name="iter_1",
        )
