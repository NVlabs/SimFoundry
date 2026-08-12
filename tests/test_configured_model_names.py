# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every Gemini model name the repo references must be one the client can construct.

Gemini.__init__ runs assert_valid_key against VERSIONS, so an unregistered name is a hard
failure at the first VLM call -- minutes into a stage, not at config load. This drifted once
already: `gemini-3-pro-image-preview` outlived its GA rename and survived in eight places,
including a hardcoded fallback default that the documented workaround did not cover.
"""

from pathlib import Path
import re

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VLM_PATH = REPO_ROOT / "simfoundry" / "models" / "vlm.py"
SEARCH_DIRS = (REPO_ROOT / "scripts", REPO_ROOT / "simfoundry")

# Matches a gemini model id wherever it appears: yaml value, python default, or comment.
MODEL_RE = re.compile(r"gemini-[a-z0-9.\-]*[a-z0-9]")


def registered_models() -> set[str]:
    """Parse VERSIONS without importing vlm.py, which pulls in heavy optional deps."""
    src = VLM_PATH.read_text()
    block = src.split("VERSIONS = {", 1)[1]
    return set(re.findall(r'"(gemini-[^"]+)":\s*\{', block))


def referenced_models() -> dict[str, list[str]]:
    """Every gemini-* token in the tracked config and source trees, by name."""
    found: dict[str, list[str]] = {}
    for root in SEARCH_DIRS:
        for path in list(root.rglob("*.yaml")) + list(root.rglob("*.py")):
            if path == VLM_PATH:
                continue  # VERSIONS itself is the registry, not a reference
            for line_no, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
                for name in MODEL_RE.findall(line):
                    found.setdefault(name, []).append(f"{path.relative_to(REPO_ROOT)}:{line_no}")
    return found


def test_registry_is_parseable():
    models = registered_models()
    assert "gemini-3-pro-image" in models
    assert len(models) > 5


def test_no_configured_model_is_unregistered():
    registered = registered_models()
    referenced = referenced_models()
    unknown = {name: sites for name, sites in referenced.items() if name not in registered}
    assert not unknown, "unregistered Gemini model name(s) referenced:\n" + "\n".join(
        f"  {name}: {', '.join(sites)}" for name, sites in sorted(unknown.items())
    )


def test_the_retired_preview_image_name_is_gone():
    """`gemini-3-pro-image-preview` was retired from Gemini/Vertex; the GA name replaced it."""
    assert "gemini-3-pro-image-preview" not in referenced_models()
