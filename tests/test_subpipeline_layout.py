# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "scripts" / "pipeline"


def test_subpipeline_entrypoints_exist():
    assert (PIPELINE_ROOT / "A_reconstruction" / "run.sh").is_file()
    assert (PIPELINE_ROOT / "B_augmentation" / "run.sh").is_file()
    assert (PIPELINE_ROOT / "C_application" / "run.sh").is_file()
    assert (PIPELINE_ROOT / "run.sh").is_file()


def test_automated_pipeline_scripts_have_no_live_breakpoints():
    """No debugger residue anywhere in shipped code.

    Covers the whole tree (package + all three sub-pipelines + interactive tools), and
    rejects commented-out debugger calls too — they are dead code in a release.
    """
    scanned_roots = [
        REPO_ROOT / "simfoundry",
        PIPELINE_ROOT,
        REPO_ROOT / "scripts" / "interactive",
        REPO_ROOT / "scripts" / "installation",
    ]
    needles = ("breakpoint()", "pdb.set_trace", "import pdb")
    offenders = []
    for root in scanned_roots:
        for path in sorted(root.rglob("*.py")) + sorted(root.rglob("*.sh")):
            # The light editor keeps its own virtualenv at
            # scripts/interactive/light_editor/.venv-light, which is inside a scanned root
            # and full of third-party code that legitimately ships debugger calls (pip's
            # vendored distlib, for one). Only our own sources are in scope, so skip any
            # dot-directory rather than reporting a dependency's residue as ours.
            if any(part.startswith(".") for part in path.relative_to(REPO_ROOT).parts):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in needles:
                if needle in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {needle}")
    assert offenders == [], offenders


def test_superseded_upcoming_pipeline_removed():
    assert not (REPO_ROOT / "scripts" / "upcoming_pipeline").exists()
