# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration classes for domain randomization.

Pure Python dataclass configs (no IsaacLab dependency). These define what aspects
of the environment are randomized and the parameters for each type of randomization.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict


# Default paths for material and HDRI resources
_DEFAULT_VMATERIALS_ROOT = os.environ.get("VMATERIALS_ROOT_DIR", "/opt/nvidia/mdl/vMaterials_2")
_DEFAULT_BASE_MATERIALS_ROOT = os.environ.get(
    "BASE_MATERIALS_ROOT_DIR",
    os.path.expanduser("~/Downloads/NVIDIA_Materials/Materials/Base"),
)
_DEFAULT_HDR_BACKGROUNDS_ROOT = os.environ.get(
    "HDR_BACKGROUNDS_ROOT_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "hdr_backgrounds"),
)
_DEFAULT_OBJECT_NAMES = ["obj_0", "obj_1"]


def _active_scene_object_names() -> List[str]:
    """Return active OmniGibson scene objects, or simulator-free defaults."""
    try:
        import omnigibson as og
        scene = og.sim.scenes[0]
    except (AttributeError, IndexError, ImportError, TypeError):
        return list(_DEFAULT_OBJECT_NAMES)
    return [obj.name for obj in scene.objects if ("robot" not in obj.name and "background" not in obj.name)]


@dataclass
class MaterialLibrarySourceCfg:
    """Configuration for a single material library source directory.

    Each library points to a local directory tree containing .mdl files
    organized in category subdirectories (e.g., Wood/, Metals/, Plastics/).

    Attributes:
        name: Human-readable name for this library (e.g., "vMaterials2", "Base").
        root_dir: Absolute path to the library root directory. If the directory
                  does not exist at runtime, the library is silently skipped.
        enabled: Whether to include this library when discovering materials.
    """
    name: str = ""
    root_dir: str = ""
    enabled: bool = True


@dataclass
class PerObjectMaterialCfg:
    """Per-object material category configuration.

    Controls which materials an individual object can be assigned. When using
    the curated material set (default), the ``include_*`` flags select which
    groups to draw from. When ``use_all_vmaterials`` is enabled on the parent
    config, ``categories`` filters by vMaterials_2 folder names instead.

    Attributes:
        include_wood: Include curated wood materials.
        include_metal: Include curated metal materials.
        include_plastic: Include curated plastic materials.
        include_paint: Include curated paint/carpet/textile materials.
        categories: vMaterials_2 category folder names to sample from when
                   ``use_all_vmaterials=True`` (e.g., ["Wood", "Metal"]).
                   None = all categories.
        use_local_materials: Whether to use additional local material files.
        local_material_paths: List of paths to local .mdl files.
    """
    include_wood: bool = True
    include_metal: bool = True
    include_plastic: bool = True
    include_paint: bool = True
    categories: Optional[List[str]] = None
    use_local_materials: bool = False
    local_material_paths: List[str] = field(default_factory=list)


@dataclass
class MaterialRandomizationCfg:
    """Configuration for material/texture randomization.

    By default, uses a curated set of NVIDIA Omniverse materials (wood, metal,
    plastic, paint) loaded via Nucleus URLs — the same set used in the reference
    IsaacLab domain-randomization implementation. Set ``use_all_vmaterials=True``
    to instead discover and use *all* materials from a local vMaterials_2
    installation on disk.

    Attributes:
        enabled: Whether material randomization is enabled.
        object_names: List of object names to randomize (e.g., ["obj_0", "obj_1"]).
        randomize_robot: Whether to randomize robot arm materials.
        use_all_vmaterials: When False (default), use the curated material lists.
                           When True, discover all .mdl files from vMaterials_2.
        per_object_materials: Maps object name to per-object material config.
        include_wood: Include curated wood materials (default set only).
        include_metal: Include curated metal materials (default set only).
        include_plastic: Include curated plastic materials (default set only).
        include_paint: Include curated paint/carpet/textile materials (default set only).
        categories: vMaterials_2 categories when ``use_all_vmaterials=True``.
        randomize_texture_rotation: Randomly rotate textures.
        texture_rotation_range: (min, max) rotation in degrees.
        randomize_texture_translation: Randomly translate textures in UV space.
        texture_translation_range: (min, max) UV translation.
        randomize_color_tint: Apply random RGB color tints.
        color_tint_range: (min, max) for each RGB channel as a multiplier.
        num_variants_per_material: Number of randomized variant copies to preload
            per base material subidentifier. When set to an integer, that many
            copies with independently randomized shader params (rotation, tint,
            etc.) are created for each subidentifier.  When ``None``, exactly
            one copy per subidentifier is created — i.e., all natural material
            variants are used without artificial duplication.
        vmaterials_root_dir: Root directory of vMaterials_2 installation.
    """
    enabled: bool = False
    object_names: List[str] = field(default_factory=list)
    randomize_robot: bool = False

    # Material source mode
    use_all_vmaterials: bool = False

    # Material libraries to discover .mdl files from (scanned at preload time).
    # Libraries whose root_dir does not exist are silently skipped.
    material_libraries: List[MaterialLibrarySourceCfg] = field(default_factory=lambda: [
        MaterialLibrarySourceCfg(name="vMaterials2", root_dir=_DEFAULT_VMATERIALS_ROOT, enabled=False),
        MaterialLibrarySourceCfg(name="Base", root_dir=_DEFAULT_BASE_MATERIALS_ROOT, enabled=True),
    ])

    # Pre-configured material preset name. When set, overrides library discovery
    # and uses an explicit curated list of material URLs instead.
    # Available presets: "default_train", "default_eval"
    material_preset: Optional[str] = None

    # Per-object material configuration
    per_object_materials: Dict[str, PerObjectMaterialCfg] = field(default_factory=dict)

    # Default curated material group flags (used when use_all_vmaterials=False)
    include_wood: bool = True
    include_metal: bool = True
    include_plastic: bool = True
    include_paint: bool = True

    # vMaterials_2 categories (used when use_all_vmaterials=True)
    categories: Optional[List[str]] = None  # None = all 19 categories

    # Texture transform randomization
    randomize_texture_rotation: bool = True
    texture_rotation_range: Tuple[float, float] = (0.0, 360.0)
    randomize_texture_translation: bool = True
    texture_translation_range: Tuple[float, float] = (0.0, 100.0)

    # Color tint randomization
    randomize_color_tint: bool = True
    color_tint_range: Tuple[float, float] = (0.5, 1.5)

    # Number of material variants to preload per base material subidentifier.
    # None = one copy per subidentifier (all natural variants, no duplication).
    num_variants_per_material: Optional[int] = 3

    # vMaterials_2 root directory (legacy; prefer material_libraries instead)
    vmaterials_root_dir: str = _DEFAULT_VMATERIALS_ROOT


@dataclass
class LightingRandomizationCfg:
    """Configuration for lighting randomization.

    Controls scene lighting variations via OmniGibson's skybox (dome light).

    Attributes:
        enabled: Whether lighting randomization is enabled.
        dome_intensity_range: (min, max) dome light intensity.
        dome_color_temperature_range: (min, max) color temperature in Kelvin.
        use_hdri_textures: Whether to use HDRI textures for environment lighting.
        hdr_backgrounds_root_dir: Path to folder containing .hdr/.exr files.
        randomize_hdri_rotation: Whether to randomly rotate the HDRI environment.
        hdri_rotation_range: (min, max) rotation in degrees around Z-axis.
    """
    enabled: bool = False

    # Dome light intensity and color
    dome_intensity_range: Tuple[float, float] = (300.0, 1200.0)
    dome_color_temperature_range: Tuple[float, float] = (4000.0, 8000.0)

    # HDRI-based lighting
    use_hdri_textures: bool = False
    hdr_backgrounds_root_dir: str = _DEFAULT_HDR_BACKGROUNDS_ROOT
    randomize_hdri_rotation: bool = True
    hdri_rotation_range: Tuple[float, float] = (0.0, 360.0)


@dataclass
class DomainRandomizationCfg:
    """Master configuration for all domain randomization.

    Top-level configuration controlling both material and lighting randomization.
    Provides factory classmethods for common use cases.

    Attributes:
        enabled: Master switch for all domain randomization.
        materials: Material randomization configuration.
        lighting: Lighting randomization configuration.
        randomize_on_reset: Apply randomization on environment reset.
        randomize_interval_steps: Apply randomization every N steps (0 = only on reset).
    """
    enabled: bool = False
    materials: MaterialRandomizationCfg = field(default_factory=MaterialRandomizationCfg)
    lighting: LightingRandomizationCfg = field(default_factory=LightingRandomizationCfg)

    # When to apply randomization
    randomize_on_reset: bool = True
    randomize_interval_steps: int = 0

    def __post_init__(self):
        """Propagate enabled state to sub-configs."""
        if not self.enabled:
            self.materials.enabled = False
            self.lighting.enabled = False

    @classmethod
    def create_for_training(cls, object_names: Optional[List[str]] = None) -> "DomainRandomizationCfg":
        """Create config for training with full randomization.

        Args:
            object_names: Objects to randomize. Defaults to all objects in the active scene, minus the robots and background.
        """
        if object_names is None:
            object_names = _active_scene_object_names()

        return cls(
            enabled=True,
            materials=MaterialRandomizationCfg(
                enabled=True,
                object_names=object_names,
                randomize_robot=True,
            ),
            lighting=LightingRandomizationCfg(
                enabled=True,
                use_hdri_textures=True,
            ),
            randomize_on_reset=True,
            randomize_interval_steps=30,
        )

    @classmethod
    def create_for_teleoperation(cls) -> "DomainRandomizationCfg":
        """Create config for teleoperation (no randomization)."""
        return cls(enabled=False)

    @classmethod
    def create_for_evaluation(cls, object_names: Optional[List[str]] = None) -> "DomainRandomizationCfg":
        """
        Create config for policy evaluation with moderate randomization.
        
        Args:
            object_names: Objects to randomize. Defaults to all objects in the active scene, minus the robots and background.
        """
        if object_names is None:
            object_names = _active_scene_object_names()

        return cls(
            enabled=True,
            materials=MaterialRandomizationCfg(
                enabled=True,
                object_names=object_names,
                randomize_robot=True,
                num_variants_per_material=3,
            ),
            lighting=LightingRandomizationCfg(
                enabled=True,
                dome_intensity_range=(300.0, 1200.0),
            ),
            randomize_on_reset=True,
        )
