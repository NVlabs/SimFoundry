# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Domain Randomization Module for OmniGibson.

Provides visual/appearance and lighting randomization for sim-to-real transfer.
Materials are sourced from local NVIDIA material libraries (Base, vMaterials_2,
or custom directories), curated presets ("default_train", "default_eval"), or
Nucleus server URLs. Lighting is controlled via OmniGibson's skybox API.

Usage:
    - Teleoperation: Randomization disabled (clean human demonstrations)
    - Training: Full randomization enabled (material + lighting)
    - Evaluation: Moderate randomization for robust policy testing

Quick start:
    from simfoundry.domain_randomization import DomainRandomizationCfg, DomainRandomizationManager

    cfg = DomainRandomizationCfg.create_for_training(object_names=["obj_0", "obj_1"])
    manager = DomainRandomizationManager(cfg)
    manager.initialize()   # After scene is loaded
    manager.randomize()    # On each reset
"""

from .configs import (
    DomainRandomizationCfg,
    MaterialRandomizationCfg,
    MaterialLibrarySourceCfg,
    LightingRandomizationCfg,
    PerObjectMaterialCfg,
)
from .materials import (
    VMATERIAL_CATEGORIES,
    MATERIAL_PRESETS,
    MaterialLibraryDiscovery,
    VMaterialsDiscovery,
    MaterialVariant,
    MaterialLibrary,
    get_all_visual_mesh_prim_paths,
)
from .lighting import (
    color_temperature_to_rgb,
    list_hdri_files,
    LightingRandomizer,
)
from .manager import DomainRandomizationManager

__all__ = [
    # Configs
    "DomainRandomizationCfg",
    "MaterialRandomizationCfg",
    "MaterialLibrarySourceCfg",
    "LightingRandomizationCfg",
    "PerObjectMaterialCfg",
    # Materials
    "VMATERIAL_CATEGORIES",
    "MATERIAL_PRESETS",
    "MaterialLibraryDiscovery",
    "VMaterialsDiscovery",
    "MaterialVariant",
    "MaterialLibrary",
    "get_all_visual_mesh_prim_paths",
    # Lighting
    "color_temperature_to_rgb",
    "list_hdri_files",
    "LightingRandomizer",
    # Manager
    "DomainRandomizationManager",
]
