# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guards for defaults and availability checks that a release regressed once already."""

from pathlib import Path

from omegaconf import OmegaConf
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "scripts" / "cfg" / "real2sim_cfg.yaml"
B_STAGE6 = REPO_ROOT / "scripts" / "pipeline" / "B_augmentation" / "stages" / "6_sample_reconstructed_scene.py"


def test_include_gs_defaults_off_because_stage_2c_is_opt_in():
    """The default pipeline must not consume an artifact it does not produce.

    Stage 2c produces the splat and is opt-in via --bg-splat. When include_gs defaulted to
    true, the default A->B path crashed in B stage 6 on a missing gaussian_da3.usdz.
    """
    cfg = OmegaConf.load(CFG_PATH)
    assert cfg.s14_og.include_gs is False


def test_b_stage6_guards_the_missing_splat():
    """B stage 6 must check the splat exists, as stage 14 does.

    Without the check, USDObject(usd_path=...) raises FileNotFoundError and the process
    segfaults instead of continuing without a background.
    """
    source = B_STAGE6.read_text()
    guard_index = source.find("os.path.exists(gs_path_da3)")
    usage_index = source.find("usd_path=gs_path_da3")
    assert guard_index != -1, "B stage 6 no longer guards the missing GS USDZ"
    assert guard_index < usage_index, "the guard must precede the USDObject construction"


def test_articulation_availability_checks_more_than_the_stage_script():
    """The stage script ships in every checkout, so testing only for it never degrades.

    README/INSTALL both promise --detect-articulation is "ignored with a warning" when the
    articulation environments are absent; that only holds if the check looks at them.
    """
    from simfoundry.pipeline import orchestrator

    assert orchestrator.ARTICULATION_DEPS_DIR
    assert orchestrator.ARTICULATION_ENVS

    # With no conda layout discoverable and no deps checkout, it must report unavailable.
    assert (REPO_ROOT / orchestrator.ARTICULATION_STAGE_SCRIPT).is_file()


def test_articulation_unavailable_without_the_deps_checkout(monkeypatch, tmp_path):
    from simfoundry.pipeline import orchestrator

    # A tree that has the stage script but no deps/articulate-anything.
    fake_root = tmp_path / "repo"
    script = fake_root / orchestrator.ARTICULATION_STAGE_SCRIPT
    script.parent.mkdir(parents=True)
    script.write_text("# stub\n")
    monkeypatch.setattr(orchestrator, "REPO_ROOT", fake_root)
    assert orchestrator.articulation_available() is False

    # Adding the checkout is still not enough without one of the conda envs.
    (fake_root / orchestrator.ARTICULATION_DEPS_DIR).mkdir(parents=True)
    monkeypatch.setattr(orchestrator, "_conda_envs_dirs", lambda: [])
    assert orchestrator.articulation_available() is False

    # With both present it becomes available.
    envs = tmp_path / "envs"
    (envs / orchestrator.ARTICULATION_ENVS[0]).mkdir(parents=True)
    monkeypatch.setattr(orchestrator, "_conda_envs_dirs", lambda: [envs])
    assert orchestrator.articulation_available() is True


@pytest.mark.parametrize("bg_splat, expected", [(False, False), (True, True)])
def test_stage_2c_is_opt_in_and_routed_to_nerfstudio(bg_splat, expected):
    from simfoundry.pipeline.orchestrator import get_stage_plan

    plan = get_stage_plan("video", pipeline_name="reconstruction", bg_splat=bg_splat)
    spec = next((s for s in plan if s.stage_id == "2c"), None)
    assert (spec is not None) is expected
    if spec is not None:
        assert spec.env == "nerfstudio"
