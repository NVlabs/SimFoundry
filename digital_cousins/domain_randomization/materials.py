# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Material randomization utilities for OmniGibson environments.

This module provides:
- Curated NVIDIA material lists (wood, metal, plastic, paint) matching the
  reference IsaacLab domain-randomization implementation
- Pre-configured material presets for train/eval splits ("default_train",
  "default_eval")
- MaterialLibraryDiscovery: Scans local material library directories for
  .mdl files (supports vMaterials_2, Base, and arbitrary libraries)
- MaterialLibrary: Preloads materials into USD stage and provides random
  selection/binding
- Helper utilities for traversing OmniGibson object visual meshes
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from .configs import MaterialRandomizationCfg, PerObjectMaterialCfg, _DEFAULT_BASE_MATERIALS_ROOT


def _run_async_safely(coro):
    """Run an async coroutine safely, handling both cases:
    - No running event loop: use asyncio.run()
    - Running event loop: run in a separate thread with new event loop

    Args:
        coro: Async coroutine to run

    Returns:
        Result of the coroutine
    """
    import asyncio
    import concurrent.futures

    try:
        asyncio.get_running_loop()
        # Running loop exists — run in a separate thread
        def run_in_thread():
            return asyncio.run(coro)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result(timeout=30)
    except RuntimeError:
        # No running event loop
        return asyncio.run(coro)


# ============================================================================
# Curated NVIDIA material lists (local Base library paths)
# These match the reference IsaacLab domain-randomization implementation and
# are used by the "default_train" / "default_eval" presets.
# Paths are resolved relative to _DEFAULT_BASE_MATERIALS_ROOT.
# ============================================================================

def _base_mat(*rel_parts: str) -> str:
    """Build a path under the Base materials library root."""
    return os.path.join(_DEFAULT_BASE_MATERIALS_ROOT, *rel_parts)


CURATED_WOOD_MATERIALS = [
    _base_mat("Wood", "Ash.mdl"),
    _base_mat("Wood", "Ash_Planks.mdl"),
    _base_mat("Wood", "Bamboo.mdl"),
    _base_mat("Wood", "Bamboo_Planks.mdl"),
    _base_mat("Wood", "Beadboard.mdl"),
    _base_mat("Wood", "Birch.mdl"),
    _base_mat("Wood", "Cherry.mdl"),
    _base_mat("Wood", "Cherry_Planks.mdl"),
    _base_mat("Wood", "Cork.mdl"),
    _base_mat("Wood", "Mahogany.mdl"),
    _base_mat("Wood", "Mahogany_Planks.mdl"),
    _base_mat("Wood", "Oak.mdl"),
    _base_mat("Wood", "Oak_Planks.mdl"),
    _base_mat("Wood", "Parquet_Floor.mdl"),
    _base_mat("Wood", "Plywood.mdl"),
    _base_mat("Wood", "Timber.mdl"),
    _base_mat("Wood", "Timber_Cladding.mdl"),
    _base_mat("Wood", "Walnut.mdl"),
    _base_mat("Wood", "Walnut_Planks.mdl"),
]

CURATED_METAL_MATERIALS = [
    _base_mat("Metals", "Aluminum_Anodized_Black.mdl"),
    _base_mat("Metals", "Aluminum_Anodized_Blue.mdl"),
    _base_mat("Metals", "Aluminum_Anodized_Red.mdl"),
    _base_mat("Metals", "Brass.mdl"),
    _base_mat("Metals", "Bronze.mdl"),
    _base_mat("Metals", "Brushed_Antique_Copper.mdl"),
    _base_mat("Metals", "Cast_Metal_Silver_Vein.mdl"),
    _base_mat("Metals", "Copper.mdl"),
    _base_mat("Metals", "CorrugatedMetal.mdl"),
    _base_mat("Metals", "Iron.mdl"),
    _base_mat("Metals", "Metal_Door.mdl"),
    _base_mat("Metals", "Metal_Seamed_Roof.mdl"),
    _base_mat("Metals", "RustedMetal.mdl"),
    _base_mat("Metals", "Silver.mdl"),
    _base_mat("Metals", "Steel_Blued.mdl"),
    _base_mat("Metals", "Steel_Carbon.mdl"),
    _base_mat("Metals", "Steel_Stainless.mdl"),
]

CURATED_PLASTIC_MATERIALS = [
    _base_mat("Plastics", "Rubber_Smooth.mdl"),
    _base_mat("Plastics", "Rubber_Textured.mdl"),
    _base_mat("Plastics", "Veneer_Z5_Maple.mdl"),
]

CURATED_PAINT_MATERIALS = [
    _base_mat("Carpet", "Carpet_Pattern_Squares_Multi.mdl"),
    _base_mat("Carpet", "Carpet_Berber_Gray.mdl"),
    _base_mat("Carpet", "Carpet_Berber_Multi.mdl"),
    _base_mat("Carpet", "Carpet_Charcoal.mdl"),
    _base_mat("Carpet", "Carpet_Diamond_Yellow.mdl"),
    _base_mat("Carpet", "Carpet_Gray.mdl"),
    _base_mat("Carpet", "Carpet_Pattern_Leaf_Squares_Tan.mdl"),
    _base_mat("Textiles", "Leather_Black.mdl"),
]

CURATED_MATERIAL_GROUPS = {
    "Wood": CURATED_WOOD_MATERIALS,
    "Metal": CURATED_METAL_MATERIALS,
    "Plastic": CURATED_PLASTIC_MATERIALS,
    "Paint": CURATED_PAINT_MATERIALS,
}

# ============================================================================
# Holdout / evaluation materials (disjoint from training set)
# ============================================================================

DEFAULT_EVAL_MATERIALS = [
    _base_mat("Wood", "Birch_Planks.mdl"),
    _base_mat("Wood", "Parquet_Floor.mdl"),
    _base_mat("Metals", "Brushed_Antique_Copper.mdl"),
    _base_mat("Metals", "Gold.mdl"),
    _base_mat("Metals", "Chrome.mdl"),
    _base_mat("Plastics", "Plastic.mdl"),
    _base_mat("Plastics", "Veneer_UX_Walnut_Cherry.mdl"),
    _base_mat("Carpet", "Carpet_Beige.mdl"),
    _base_mat("Carpet", "Carpet_Forest.mdl"),
    _base_mat("Carpet", "Carpet_Pattern_Loop.mdl"),
    _base_mat("Carpet", "Carpet_Cream.mdl"),
    _base_mat("Wall_Board", "Cardboard.mdl"),
    _base_mat("Wall_Board", "MDF.mdl"),
    _base_mat("Stone", "Retaining_Block.mdl"),
    _base_mat("Textiles", "Cloth_Black.mdl"),
    _base_mat("Textiles", "Cloth_Gray.mdl"),
    _base_mat("Textiles", "Leather_Brown.mdl"),
    _base_mat("Textiles", "Linen_Blue.mdl"),
]

DEFAULT_TRAIN_MATERIALS = (
    CURATED_WOOD_MATERIALS
    + CURATED_METAL_MATERIALS
    + CURATED_PLASTIC_MATERIALS
    + CURATED_PAINT_MATERIALS
)

MATERIAL_PRESETS: Dict[str, List[str]] = {
    "default_train": DEFAULT_TRAIN_MATERIALS,
    "default_eval": DEFAULT_EVAL_MATERIALS,
}

# ============================================================================
# vMaterials_2 category folder names (kept for backward compatibility)
# ============================================================================

VMATERIAL_CATEGORIES = [
    "Carpet",
    "Ceramic",
    "Composite",
    "Concrete",
    "Fabric",
    "Gems",
    "Glass",
    "Ground",
    "Leather",
    "Liquids",
    "Masonry",
    "Metal",
    "Other",
    "Paint",
    "Paper",
    "Plaster",
    "Plastic",
    "Stone",
    "Wood",
]


# ============================================================================
# Material library discovery
# ============================================================================

class MaterialLibraryDiscovery:
    """Discovers .mdl material files from a local material library directory.

    Works with any NVIDIA material library that organizes .mdl files under
    category subdirectories (e.g., vMaterials_2, Base, or custom libraries).
    The category names are discovered dynamically from the directory structure.

    Args:
        name: Human-readable library name (for logging).
        root_dir: Root directory of the material library.
    """

    def __init__(self, name: str, root_dir: str):
        self._name = name
        self._root_dir = root_dir
        self._materials_by_category: Dict[str, List[str]] = {}
        self._discovered = False

    @property
    def is_discovered(self) -> bool:
        return self._discovered

    @property
    def name(self) -> str:
        return self._name

    @property
    def root_dir(self) -> str:
        return self._root_dir

    @property
    def categories(self) -> List[str]:
        """Category names discovered from subdirectories (empty until discover() is called)."""
        return list(self._materials_by_category.keys())

    def discover(self) -> Dict[str, List[str]]:
        """Scan the library directory for .mdl files organized by category subdirectory.

        Returns:
            Dict mapping category name to list of absolute .mdl file paths.
        """
        if self._discovered:
            return self._materials_by_category

        if not os.path.isdir(self._root_dir):
            print(f"[WARNING] Material library '{self._name}' not found at: {self._root_dir}, skipping")
            self._discovered = True
            return {}

        for entry in sorted(os.listdir(self._root_dir)):
            category_dir = os.path.join(self._root_dir, entry)
            if not os.path.isdir(category_dir):
                continue
            mdl_files = []
            for dirpath, _dirnames, filenames in os.walk(category_dir):
                for fname in sorted(filenames):
                    if fname.endswith(".mdl"):
                        mdl_files.append(os.path.join(dirpath, fname))
            if mdl_files:
                self._materials_by_category[entry] = mdl_files

        self._discovered = True
        total = sum(len(v) for v in self._materials_by_category.values())
        print(
            f"[INFO] Library '{self._name}': discovered {total} .mdl files "
            f"across {len(self._materials_by_category)} categories"
        )
        return self._materials_by_category

    def get_materials(self, categories: Optional[List[str]] = None) -> List[str]:
        """Get material paths filtered by category.

        Args:
            categories: List of category names to include. None = all categories.

        Returns:
            List of .mdl file paths.
        """
        if not self._discovered:
            self.discover()

        if categories is None:
            return [path for paths in self._materials_by_category.values() for path in paths]

        result = []
        for cat in categories:
            if cat in self._materials_by_category:
                result.extend(self._materials_by_category[cat])
        return result


class VMaterialsDiscovery(MaterialLibraryDiscovery):
    """Backward-compatible alias for MaterialLibraryDiscovery.

    Defaults the library name to "vMaterials_2".

    Args:
        vmaterials_root_dir: Root directory of the vMaterials_2 installation
                            (e.g., /opt/nvidia/mdl/vMaterials_2).
    """

    def __init__(self, vmaterials_root_dir: str):
        super().__init__(name="vMaterials_2", root_dir=vmaterials_root_dir)


# ============================================================================
# Material variant and library
# ============================================================================

@dataclass
class MaterialVariant:
    """A preloaded material variant with randomized shader parameters.

    Attributes:
        prim_path: USD prim path of the material in the stage.
        base_mdl_path: Original .mdl file path on disk.
        category: Material category (e.g., "Wood", "Metal"), or "Preset" / "Local".
        library: Name of the source library (e.g., "Base", "vMaterials2", "Preset").
        texture_rotation: Rotation angle in degrees.
        texture_translation: UV translation (u, v).
        color_tint: RGB color tint (r, g, b) as multiplier, or None.
    """
    prim_path: str
    base_mdl_path: str
    category: str = ""
    library: str = ""
    texture_rotation: float = 0.0
    texture_translation: Tuple[float, float] = (0.0, 0.0)
    color_tint: Optional[Tuple[float, float, float]] = None


class MaterialLibrary:
    """Manages a library of preloaded materials for domain randomization.

    Supports three material sourcing modes (checked in priority order):

    1. **Preset mode** (``cfg.material_preset`` is set): Uses a curated list
       of Nucleus URLs from ``MATERIAL_PRESETS`` (e.g., "default_train",
       "default_eval"). Library discovery is skipped entirely.

    2. **Multi-library discovery** (``cfg.material_libraries`` has enabled
       entries): Scans each enabled library's root directory for .mdl files.
       Libraries whose directories don't exist are silently skipped.

    3. **Legacy vMaterials_2 discovery** (``cfg.vmaterials_root_dir``):
       Falls back to scanning the single vMaterials_2 directory if no
       libraries are configured.

    Usage:
        library = MaterialLibrary(cfg)
        library.preload()  # Call once at startup
        mat_path = library.get_random_material(object_name="obj_0")
        library.bind_to_prim(target_prim_path, mat_path)
    """

    def __init__(self, cfg: MaterialRandomizationCfg):
        self.cfg = cfg

        # Build discoveries from configured material libraries
        self._discoveries: Dict[str, MaterialLibraryDiscovery] = {}
        for lib_cfg in cfg.material_libraries:
            if lib_cfg.enabled:
                self._discoveries[lib_cfg.name] = MaterialLibraryDiscovery(
                    name=lib_cfg.name,
                    root_dir=lib_cfg.root_dir,
                )

        # Legacy fallback: if no libraries configured, use vmaterials_root_dir
        if not self._discoveries and cfg.vmaterials_root_dir:
            self._discoveries["vMaterials_2"] = VMaterialsDiscovery(cfg.vmaterials_root_dir)

        self._materials: List[MaterialVariant] = []
        self._preloaded = False
        self._material_root_path = "/World/Looks/DomainRand"

    @property
    def is_preloaded(self) -> bool:
        return self._preloaded

    @property
    def num_materials(self) -> int:
        return len(self._materials)

    @property
    def discoveries(self) -> Dict[str, MaterialLibraryDiscovery]:
        """All configured discovery instances, keyed by library name."""
        return self._discoveries

    @property
    def discovery(self) -> Optional[MaterialLibraryDiscovery]:
        """First available discovery instance (backward compatibility)."""
        if self._discoveries:
            return next(iter(self._discoveries.values()))
        return None

    def _get_categories_for_object(self, object_name: Optional[str] = None) -> Optional[List[str]]:
        """Resolve which material categories to use for a given object."""
        if object_name and object_name in self.cfg.per_object_materials:
            return self.cfg.per_object_materials[object_name].categories
        return self.cfg.categories

    def _collect_mdl_sources(self) -> List[Tuple[str, str, str]]:
        """Collect (mdl_path, category, library_name) tuples from all sources.

        Returns:
            List of (mdl_path, category, library_name) to create prims from.
        """
        # Priority 1: Preset mode
        if self.cfg.material_preset:
            preset_name = self.cfg.material_preset
            if preset_name not in MATERIAL_PRESETS:
                print(
                    f"[WARNING] Unknown material preset '{preset_name}'. "
                    f"Available: {list(MATERIAL_PRESETS.keys())}"
                )
                return []
            print(f"[INFO] Using material preset '{preset_name}' ({len(MATERIAL_PRESETS[preset_name])} materials)")
            return [(url, "Preset", "Preset") for url in MATERIAL_PRESETS[preset_name]]

        # Priority 2: Multi-library discovery
        mdl_sources: List[Tuple[str, str, str]] = []

        all_needed_categories = set()
        if self.cfg.categories is not None:
            all_needed_categories.update(self.cfg.categories)
        for obj_cfg in self.cfg.per_object_materials.values():
            if obj_cfg.categories is not None:
                all_needed_categories.update(obj_cfg.categories)
        use_all_categories = not all_needed_categories

        for lib_name, discovery in self._discoveries.items():
            materials_by_cat = discovery.discover()
            if not materials_by_cat:
                continue

            if use_all_categories:
                cats_to_use = materials_by_cat.keys()
            else:
                cats_to_use = all_needed_categories

            for cat in cats_to_use:
                for mdl_path in materials_by_cat.get(cat, []):
                    mdl_sources.append((mdl_path, cat, lib_name))

        # Also add any per-object local material paths
        if self.cfg.per_object_materials:
            for obj_cfg in self.cfg.per_object_materials.values():
                if obj_cfg.use_local_materials:
                    for lp in obj_cfg.local_material_paths:
                        mdl_sources.append((lp, "Local", "Local"))

        return mdl_sources

    def preload(self) -> bool:
        """Preload materials into the USD stage.

        Discovers MDL files, creates material prims with randomized shader
        parameters. Should be called once at startup.

        Returns:
            True if preloading succeeded.
        """
        if self._preloaded:
            print("[INFO] Materials already preloaded, skipping")
            return True

        if not self.cfg.enabled:
            print("[INFO] Material randomization disabled, skipping preload")
            return True

        try:
            from pxr import Sdf, Gf
            import omni.kit.commands
            import omni.kit.app
        except ImportError as e:
            print(f"[WARNING] Failed to import Omniverse modules: {e}")
            print("[WARNING] Material randomization will be disabled")
            self.cfg.enabled = False
            return False

        # Enable material library extension
        try:
            manager = omni.kit.app.get_app().get_extension_manager()
            manager.set_extension_enabled_immediate("omni.kit.material.library", True)
            import omni.kit.material.library
        except Exception as e:
            print(f"[WARNING] Failed to enable omni.kit.material.library: {e}")
            print("[WARNING] Proceeding without subidentifier fetching")
            omni_mat_lib = None
        else:
            omni_mat_lib = omni.kit.material.library

        import omnigibson as og
        stage = og.sim.stage

        # Create root scope prim for domain randomization materials
        if not stage.GetPrimAtPath(self._material_root_path).IsValid():
            stage.DefinePrim(self._material_root_path, "Scope")

        # Collect all material sources
        mdl_sources = self._collect_mdl_sources()
        if not mdl_sources:
            print("[WARNING] No materials found from any source. Material randomization disabled.")
            self.cfg.enabled = False
            return False

        material_count = 0
        for mdl_path, category, library_name in mdl_sources:
            # Get subidentifiers (variants within the MDL file)
            subidentifiers = [None]
            if omni_mat_lib is not None:
                try:
                    subidentifiers = _run_async_safely(
                        omni_mat_lib.get_subidentifier_from_mdl(mdl_path)
                    )
                    if not subidentifiers:
                        subidentifiers = [None]
                except Exception as e:
                    print(f"[WARNING] Failed to get subidentifiers for {mdl_path}: {e}")
                    subidentifiers = [None]

            n_variants = self.cfg.num_variants_per_material if self.cfg.num_variants_per_material is not None else 1
            for subidentifier in subidentifiers:
                for _variant_idx in range(n_variants):
                    material_prim_path = f"{self._material_root_path}/Material_{material_count}"

                    try:
                        success, _result = omni.kit.commands.execute(
                            "CreateMdlMaterialPrimCommand",
                            mtl_url=str(mdl_path),
                            mtl_name=str(subidentifier) if subidentifier else "",
                            mtl_path=material_prim_path,
                        )
                        if not success:
                            print(f"[WARNING] Failed to create material: {material_prim_path}")
                            continue
                    except Exception as e:
                        print(f"[WARNING] Error creating material from {mdl_path}: {e}")
                        continue

                    # Apply randomized shader properties
                    texture_rotation = 0.0
                    texture_translation = (0.0, 0.0)
                    color_tint = None

                    shader_prim_path = f"{material_prim_path}/Shader"
                    shader_prim = stage.GetPrimAtPath(shader_prim_path)

                    if shader_prim.IsValid():
                        # Enable UV projection
                        shader_prim.CreateAttribute("inputs:project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
                        shader_prim.CreateAttribute("inputs:world_or_object", Sdf.ValueTypeNames.Bool).Set(True)

                        if self.cfg.randomize_texture_rotation:
                            texture_rotation = np.random.uniform(*self.cfg.texture_rotation_range)
                            shader_prim.CreateAttribute(
                                "inputs:texture_rotate", Sdf.ValueTypeNames.Float
                            ).Set(texture_rotation)

                        if self.cfg.randomize_texture_translation:
                            texture_translation = (
                                np.random.uniform(*self.cfg.texture_translation_range),
                                np.random.uniform(*self.cfg.texture_translation_range),
                            )
                            shader_prim.CreateAttribute(
                                "inputs:texture_translate", Sdf.ValueTypeNames.Float2
                            ).Set(texture_translation)

                        if self.cfg.randomize_color_tint:
                            color_tint = (
                                np.random.uniform(*self.cfg.color_tint_range),
                                np.random.uniform(*self.cfg.color_tint_range),
                                np.random.uniform(*self.cfg.color_tint_range),
                            )
                            shader_prim.CreateAttribute(
                                "inputs:diffuse_tint", Sdf.ValueTypeNames.Color3f
                            ).Set(Gf.Vec3f(*color_tint))

                    variant = MaterialVariant(
                        prim_path=material_prim_path,
                        base_mdl_path=mdl_path,
                        category=category,
                        library=library_name,
                        texture_rotation=texture_rotation,
                        texture_translation=texture_translation,
                        color_tint=color_tint,
                    )
                    self._materials.append(variant)
                    material_count += 1

        self._preloaded = True
        print(f"[INFO] Preloaded {material_count} material variants from {len(mdl_sources)} base materials")
        return True

    def get_random_material(self, object_name: Optional[str] = None) -> Optional[str]:
        """Get a random material prim path, optionally filtered for a specific object.

        Args:
            object_name: If provided and has a per_object_materials entry, filters
                        to only materials from that object's allowed categories.

        Returns:
            Prim path of a random material, or None if no materials available.
        """
        if not self._materials:
            return None

        categories = self._get_categories_for_object(object_name)

        if categories is not None:
            filtered = [m for m in self._materials if m.category in categories]
            if not filtered:
                print(f"[WARNING] No materials for categories {categories}, using full pool")
                filtered = self._materials
        else:
            filtered = self._materials

        variant = filtered[np.random.randint(len(filtered))]
        return variant.prim_path

    @staticmethod
    def bind_to_prim(target_prim_path: str, material_prim_path: str) -> bool:
        """Bind a material to a prim using OmniGibson's bind_material utility.

        Args:
            target_prim_path: Path of the prim to bind material to.
            material_prim_path: Path of the material to bind.

        Returns:
            True if binding succeeded.
        """
        try:
            from omnigibson.utils.physx_utils import bind_material
            bind_material(target_prim_path, material_prim_path)
            return True
        except Exception as e:
            print(f"[WARNING] Failed to bind material {material_prim_path} to {target_prim_path}: {e}")
            return False


def get_all_visual_mesh_prim_paths(obj) -> List[str]:
    """Get all visual mesh prim paths for an OmniGibson object.

    Traverses obj.links -> link.visual_meshes -> mesh.prim_path to collect
    all visual geometry that can have materials applied.

    Args:
        obj: An OmniGibson object (EntityPrim) with a .links property.

    Returns:
        List of prim path strings for all visual meshes.
    """
    paths = []
    for link in obj.links.values():
        if link.visual_meshes is not None:
            for mesh in link.visual_meshes.values():
                paths.append(mesh.prim_path)
    return paths
