# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the domain randomization module.

These tests run WITHOUT a simulator. They validate:
- Configuration dataclass defaults and factory methods
- vMaterials_2 discovery from local disk
- Color temperature -> RGB conversion
- HDRI file discovery
- Material category definitions
"""

import os
import sys
import unittest
from dataclasses import fields

# Ensure the project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from simfoundry.domain_randomization.configs import (
    DomainRandomizationCfg,
    MaterialRandomizationCfg,
    LightingRandomizationCfg,
    PerObjectMaterialCfg,
)
from simfoundry.domain_randomization.materials import (
    VMATERIAL_CATEGORIES,
    MATERIAL_PRESETS,
    MaterialLibraryDiscovery,
    VMaterialsDiscovery,
)
from simfoundry.domain_randomization.lighting import (
    color_temperature_to_rgb,
    list_hdri_files,
)


# ============================================================================
# Config Tests
# ============================================================================

class TestConfigs(unittest.TestCase):
    """Test configuration dataclasses."""

    def test_default_domain_randomization_cfg(self):
        cfg = DomainRandomizationCfg()
        self.assertFalse(cfg.enabled)
        self.assertFalse(cfg.materials.enabled)
        self.assertFalse(cfg.lighting.enabled)
        self.assertTrue(cfg.randomize_on_reset)
        self.assertEqual(cfg.randomize_interval_steps, 0)

    def test_post_init_propagation_disabled(self):
        """When master is disabled, sub-configs should be disabled."""
        cfg = DomainRandomizationCfg(
            enabled=False,
            materials=MaterialRandomizationCfg(enabled=True),
            lighting=LightingRandomizationCfg(enabled=True),
        )
        self.assertFalse(cfg.materials.enabled)
        self.assertFalse(cfg.lighting.enabled)

    def test_post_init_propagation_enabled(self):
        """When master is enabled, sub-configs keep their own enabled state."""
        cfg = DomainRandomizationCfg(
            enabled=True,
            materials=MaterialRandomizationCfg(enabled=True),
            lighting=LightingRandomizationCfg(enabled=False),
        )
        self.assertTrue(cfg.materials.enabled)
        self.assertFalse(cfg.lighting.enabled)

    def test_factory_create_for_training(self):
        cfg = DomainRandomizationCfg.create_for_training(["obj_0", "obj_1", "obj_2"])
        self.assertTrue(cfg.enabled)
        self.assertTrue(cfg.materials.enabled)
        self.assertTrue(cfg.lighting.enabled)
        self.assertEqual(cfg.materials.object_names, ["obj_0", "obj_1", "obj_2"])
        self.assertTrue(cfg.materials.randomize_robot)
        self.assertTrue(cfg.randomize_on_reset)
        self.assertGreater(cfg.randomize_interval_steps, 0)

    def test_factory_create_for_training_defaults(self):
        cfg = DomainRandomizationCfg.create_for_training()
        self.assertEqual(cfg.materials.object_names, ["obj_0", "obj_1"])

    def test_factory_create_for_teleoperation(self):
        cfg = DomainRandomizationCfg.create_for_teleoperation()
        self.assertFalse(cfg.enabled)
        self.assertFalse(cfg.materials.enabled)
        self.assertFalse(cfg.lighting.enabled)

    def test_factory_create_for_evaluation(self):
        cfg = DomainRandomizationCfg.create_for_evaluation(["obj_0"])
        self.assertTrue(cfg.enabled)
        self.assertTrue(cfg.materials.enabled)
        self.assertTrue(cfg.lighting.enabled)
        self.assertEqual(cfg.materials.object_names, ["obj_0"])

    def test_per_object_material_cfg_defaults(self):
        obj_cfg = PerObjectMaterialCfg()
        self.assertIsNone(obj_cfg.categories)
        self.assertFalse(obj_cfg.use_local_materials)
        self.assertEqual(obj_cfg.local_material_paths, [])

    def test_per_object_material_cfg_with_categories(self):
        obj_cfg = PerObjectMaterialCfg(categories=["Wood", "Metal"])
        self.assertEqual(obj_cfg.categories, ["Wood", "Metal"])

    def test_material_cfg_defaults(self):
        cfg = MaterialRandomizationCfg()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.object_names, [])
        self.assertFalse(cfg.randomize_robot)
        self.assertIsNone(cfg.categories)
        self.assertTrue(cfg.randomize_texture_rotation)
        self.assertTrue(cfg.randomize_texture_translation)
        self.assertTrue(cfg.randomize_color_tint)
        self.assertEqual(cfg.num_variants_per_material, 3)

    def test_material_cfg_num_variants_none(self):
        """num_variants_per_material=None should be accepted (all natural variants)."""
        cfg = MaterialRandomizationCfg(num_variants_per_material=None)
        self.assertIsNone(cfg.num_variants_per_material)
        self.assertEqual(cfg.texture_rotation_range, (0.0, 360.0))
        self.assertEqual(cfg.texture_translation_range, (0.0, 100.0))
        self.assertEqual(cfg.color_tint_range, (0.5, 1.5))

    def test_lighting_cfg_defaults(self):
        cfg = LightingRandomizationCfg()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.dome_intensity_range, (300.0, 1200.0))
        self.assertEqual(cfg.dome_color_temperature_range, (4000.0, 8000.0))
        self.assertFalse(cfg.use_hdri_textures)
        self.assertTrue(cfg.randomize_hdri_rotation)
        self.assertEqual(cfg.hdri_rotation_range, (0.0, 360.0))

    def test_per_object_materials_in_material_cfg(self):
        cfg = MaterialRandomizationCfg(
            per_object_materials={
                "obj_0": PerObjectMaterialCfg(categories=["Wood"]),
                "obj_1": PerObjectMaterialCfg(categories=["Metal", "Plastic"]),
            }
        )
        self.assertIn("obj_0", cfg.per_object_materials)
        self.assertEqual(cfg.per_object_materials["obj_0"].categories, ["Wood"])
        self.assertEqual(cfg.per_object_materials["obj_1"].categories, ["Metal", "Plastic"])


# ============================================================================
# Material Category Tests
# ============================================================================

class TestMaterialCategories(unittest.TestCase):
    """Test VMATERIAL_CATEGORIES list."""

    def test_all_19_categories(self):
        self.assertEqual(len(VMATERIAL_CATEGORIES), 19)

    def test_expected_categories_present(self):
        expected = {"Wood", "Metal", "Plastic", "Carpet", "Ceramic", "Composite",
                    "Concrete", "Fabric", "Gems", "Glass", "Ground", "Leather",
                    "Liquids", "Masonry", "Other", "Paint", "Paper", "Plaster", "Stone"}
        self.assertEqual(set(VMATERIAL_CATEGORIES), expected)

    def test_categories_sorted(self):
        self.assertEqual(VMATERIAL_CATEGORIES, sorted(VMATERIAL_CATEGORIES))


# ============================================================================
# VMaterials Discovery Tests
# ============================================================================

VMATERIALS_ROOT = "/opt/nvidia/mdl/vMaterials_2"
VMATERIALS_AVAILABLE = os.path.isdir(VMATERIALS_ROOT)


@unittest.skipUnless(VMATERIALS_AVAILABLE, "vMaterials_2 not installed")
class TestVMaterialsDiscovery(unittest.TestCase):
    """Test VMaterialsDiscovery against real vMaterials_2 installation."""

    def setUp(self):
        self.discovery = VMaterialsDiscovery(VMATERIALS_ROOT)

    def test_discover_returns_dict(self):
        result = self.discovery.discover()
        self.assertIsInstance(result, dict)
        self.assertTrue(len(result) > 0)

    def test_discover_finds_categories(self):
        result = self.discovery.discover()
        # Should find at least Wood and Metal
        self.assertIn("Wood", result)
        self.assertIn("Metal", result)

    def test_discover_finds_mdl_files(self):
        result = self.discovery.discover()
        for category, paths in result.items():
            self.assertTrue(len(paths) > 0, f"No .mdl files in {category}")
            for path in paths:
                self.assertTrue(path.endswith(".mdl"), f"Non-MDL file: {path}")
                self.assertTrue(os.path.isfile(path), f"File not found: {path}")

    def test_discover_total_count(self):
        result = self.discovery.discover()
        total = sum(len(v) for v in result.values())
        # The plan says ~315 total, but we just check a reasonable minimum
        self.assertGreater(total, 100, f"Only found {total} .mdl files, expected >100")

    def test_get_materials_all(self):
        self.discovery.discover()
        all_materials = self.discovery.get_materials()
        self.assertGreater(len(all_materials), 100)

    def test_get_materials_filtered(self):
        self.discovery.discover()
        wood_only = self.discovery.get_materials(categories=["Wood"])
        metal_only = self.discovery.get_materials(categories=["Metal"])
        both = self.discovery.get_materials(categories=["Wood", "Metal"])
        self.assertGreater(len(wood_only), 0)
        self.assertGreater(len(metal_only), 0)
        self.assertEqual(len(both), len(wood_only) + len(metal_only))

    def test_get_materials_nonexistent_category(self):
        self.discovery.discover()
        result = self.discovery.get_materials(categories=["NonExistent"])
        self.assertEqual(len(result), 0)

    def test_is_discovered_flag(self):
        self.assertFalse(self.discovery.is_discovered)
        self.discovery.discover()
        self.assertTrue(self.discovery.is_discovered)

    def test_discover_idempotent(self):
        result1 = self.discovery.discover()
        result2 = self.discovery.discover()
        self.assertEqual(result1, result2)


class TestVMaterialsDiscoveryMissing(unittest.TestCase):
    """Test VMaterialsDiscovery with non-existent directory."""

    def test_discover_missing_dir(self):
        discovery = VMaterialsDiscovery("/nonexistent/path")
        result = discovery.discover()
        self.assertEqual(result, {})


# ============================================================================
# MaterialLibraryDiscovery Tests
# ============================================================================

BASE_MATERIALS_ROOT = os.path.expanduser("~/Downloads/NVIDIA_Materials/Materials/Base")
BASE_MATERIALS_AVAILABLE = os.path.isdir(BASE_MATERIALS_ROOT)


@unittest.skipUnless(BASE_MATERIALS_AVAILABLE, "Base materials library not installed")
class TestBaseLibraryDiscovery(unittest.TestCase):
    """Test MaterialLibraryDiscovery against the Base materials library."""

    def setUp(self):
        self.discovery = MaterialLibraryDiscovery("Base", BASE_MATERIALS_ROOT)

    def test_discover_returns_dict(self):
        result = self.discovery.discover()
        self.assertIsInstance(result, dict)
        self.assertTrue(len(result) > 0)

    def test_discover_finds_expected_categories(self):
        result = self.discovery.discover()
        self.assertIn("Wood", result)
        self.assertIn("Metals", result)

    def test_discover_finds_mdl_files(self):
        result = self.discovery.discover()
        for category, paths in result.items():
            self.assertTrue(len(paths) > 0, f"No .mdl files in {category}")
            for path in paths:
                self.assertTrue(path.endswith(".mdl"), f"Non-MDL file: {path}")
                self.assertTrue(os.path.isfile(path), f"File not found: {path}")

    def test_name_and_root(self):
        self.assertEqual(self.discovery.name, "Base")
        self.assertEqual(self.discovery.root_dir, BASE_MATERIALS_ROOT)

    def test_categories_property(self):
        self.discovery.discover()
        cats = self.discovery.categories
        self.assertIsInstance(cats, list)
        self.assertGreater(len(cats), 0)


class TestMaterialLibraryDiscoveryMissing(unittest.TestCase):
    """Test MaterialLibraryDiscovery with non-existent directory."""

    def test_discover_missing_dir(self):
        discovery = MaterialLibraryDiscovery("Missing", "/nonexistent/path")
        result = discovery.discover()
        self.assertEqual(result, {})
        self.assertTrue(discovery.is_discovered)


# ============================================================================
# Material Preset Tests
# ============================================================================

class TestMaterialPresets(unittest.TestCase):
    """Test MATERIAL_PRESETS definitions."""

    def test_presets_exist(self):
        self.assertIn("default_train", MATERIAL_PRESETS)
        self.assertIn("default_eval", MATERIAL_PRESETS)

    def test_train_preset_nonempty(self):
        self.assertGreater(len(MATERIAL_PRESETS["default_train"]), 0)

    def test_eval_preset_nonempty(self):
        self.assertGreater(len(MATERIAL_PRESETS["default_eval"]), 0)

    def test_train_eval_mostly_disjoint(self):
        """Train and eval presets should be mostly disjoint (matches reference)."""
        train_set = set(MATERIAL_PRESETS["default_train"])
        eval_set = set(MATERIAL_PRESETS["default_eval"])
        overlap = train_set & eval_set
        # Reference lists have 2 shared materials; ensure no further growth
        self.assertLessEqual(len(overlap), 2, f"Unexpected overlap growth: {overlap}")


# ============================================================================
# Color Temperature Tests
# ============================================================================

class TestColorTemperature(unittest.TestCase):
    """Test color temperature to RGB conversion."""

    def test_warm_temperature(self):
        """Warm light (2700K) should be reddish/yellowish."""
        r, g, b = color_temperature_to_rgb(2700)
        self.assertAlmostEqual(r, 1.0, places=1)  # Red near max
        self.assertGreater(r, g)  # More red than green
        self.assertGreater(g, b)  # More green than blue

    def test_cool_temperature(self):
        """Cool light (10000K) should be bluish."""
        r, g, b = color_temperature_to_rgb(10000)
        self.assertGreater(b, 0.5)  # Blue should be significant
        # At very high temps, blue approaches 1.0
        self.assertAlmostEqual(b, 1.0, places=1)

    def test_neutral_temperature(self):
        """Neutral daylight (6500K) should be close to white."""
        r, g, b = color_temperature_to_rgb(6500)
        self.assertGreater(r, 0.8)
        self.assertGreater(g, 0.8)
        self.assertGreater(b, 0.8)

    def test_rgb_in_range(self):
        """All RGB values should be in [0, 1]."""
        for temp in [1000, 2000, 3000, 4000, 5000, 6500, 8000, 10000, 20000, 40000]:
            r, g, b = color_temperature_to_rgb(temp)
            self.assertGreaterEqual(r, 0.0, f"Red < 0 at {temp}K")
            self.assertLessEqual(r, 1.0, f"Red > 1 at {temp}K")
            self.assertGreaterEqual(g, 0.0, f"Green < 0 at {temp}K")
            self.assertLessEqual(g, 1.0, f"Green > 1 at {temp}K")
            self.assertGreaterEqual(b, 0.0, f"Blue < 0 at {temp}K")
            self.assertLessEqual(b, 1.0, f"Blue > 1 at {temp}K")

    def test_clamping_low(self):
        """Temperatures below 1000K should be clamped."""
        r1, g1, b1 = color_temperature_to_rgb(500)
        r2, g2, b2 = color_temperature_to_rgb(1000)
        self.assertAlmostEqual(r1, r2)
        self.assertAlmostEqual(g1, g2)
        self.assertAlmostEqual(b1, b2)

    def test_clamping_high(self):
        """Temperatures above 40000K should be clamped."""
        r1, g1, b1 = color_temperature_to_rgb(50000)
        r2, g2, b2 = color_temperature_to_rgb(40000)
        self.assertAlmostEqual(r1, r2)
        self.assertAlmostEqual(g1, g2)
        self.assertAlmostEqual(b1, b2)


# ============================================================================
# HDRI Discovery Tests
# ============================================================================

HDR_BACKGROUNDS_DIR = os.path.join(PROJECT_ROOT, "assets", "hdr_backgrounds")
HDR_BACKGROUNDS_AVAILABLE = os.path.isdir(HDR_BACKGROUNDS_DIR)


@unittest.skipUnless(HDR_BACKGROUNDS_AVAILABLE, "HDR backgrounds directory not found")
class TestHDRIDiscovery(unittest.TestCase):
    """Test HDRI file discovery from assets/hdr_backgrounds/."""

    def test_find_files(self):
        files = list_hdri_files(HDR_BACKGROUNDS_DIR)
        self.assertGreater(len(files), 0, "No HDRI files found")

    def test_file_extensions(self):
        files = list_hdri_files(HDR_BACKGROUNDS_DIR)
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            self.assertIn(ext, [".hdr", ".exr"], f"Unexpected extension: {f}")

    def test_files_exist_on_disk(self):
        files = list_hdri_files(HDR_BACKGROUNDS_DIR)
        for f in files:
            self.assertTrue(os.path.isfile(f), f"File not found: {f}")

    def test_finds_subdirectories(self):
        """Should find files in subdirectories (e.g., indoor/)."""
        files = list_hdri_files(HDR_BACKGROUNDS_DIR)
        has_subdir = any("indoor" in f for f in files)
        if os.path.isdir(os.path.join(HDR_BACKGROUNDS_DIR, "indoor")):
            self.assertTrue(has_subdir, "Should find files in indoor/ subdirectory")

    def test_hdr_before_exr(self):
        """HDR files should appear before EXR files in the list."""
        files = list_hdri_files(HDR_BACKGROUNDS_DIR)
        hdr_indices = [i for i, f in enumerate(files) if f.lower().endswith(".hdr")]
        exr_indices = [i for i, f in enumerate(files) if f.lower().endswith(".exr")]
        if hdr_indices and exr_indices:
            self.assertLess(max(hdr_indices), min(exr_indices),
                          "HDR files should be listed before EXR files")


class TestHDRIDiscoveryMissing(unittest.TestCase):
    """Test HDRI discovery with non-existent directory."""

    def test_missing_dir_returns_empty(self):
        files = list_hdri_files("/nonexistent/path")
        self.assertEqual(files, [])


if __name__ == "__main__":
    unittest.main()
