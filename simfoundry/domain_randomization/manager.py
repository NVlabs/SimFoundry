# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Domain Randomization Manager for OmniGibson environments.

Top-level orchestrator that replaces IsaacLab's event system with explicit
randomization calls. Manages material and lighting randomization lifecycle.
"""

from typing import Optional, List

from .configs import DomainRandomizationCfg
from .materials import MaterialLibrary, get_all_visual_mesh_prim_paths
from .lighting import LightingRandomizer


class DomainRandomizationManager:
    """Manages all domain randomization for an OmniGibson scene.

    Encapsulates material preloading, binding, and lighting randomization
    in a single class. Call ``initialize()`` once after the scene is loaded,
    then ``randomize()`` on each reset or at desired intervals.

    Usage:
        manager = DomainRandomizationManager(cfg)
        manager.initialize()                       # After scene is loaded
        manager.randomize()                        # On each reset
        manager.maybe_randomize_on_step()          # Every sim step (interval-based)
    """

    def __init__(self, cfg: DomainRandomizationCfg):
        self.cfg = cfg
        self._material_library: Optional[MaterialLibrary] = None
        self._lighting_randomizer: Optional[LightingRandomizer] = None
        self._initialized = False
        self._step_count = 0

    @property
    def is_enabled(self) -> bool:
        return self.cfg.enabled

    @property
    def material_library(self) -> Optional[MaterialLibrary]:
        return self._material_library

    @property
    def lighting_randomizer(self) -> Optional[LightingRandomizer]:
        return self._lighting_randomizer

    def initialize(self) -> bool:
        """Initialize material library and lighting randomizer.

        Must be called after the OmniGibson scene is fully loaded so that
        ``og.sim.stage`` and ``og.sim.skybox`` are available.

        Returns:
            True if initialization succeeded.
        """
        if self._initialized:
            return True

        if not self.cfg.enabled:
            print("[INFO] Domain randomization is disabled")
            self._initialized = True
            return True

        # Initialize material library
        if self.cfg.materials.enabled:
            self._material_library = MaterialLibrary(self.cfg.materials)
            success = self._material_library.preload()
            if not success:
                print("[WARNING] Failed to preload materials")
                self.cfg.materials.enabled = False

        # Initialize lighting randomizer
        if self.cfg.lighting.enabled:
            self._lighting_randomizer = LightingRandomizer(self.cfg.lighting)
            success = self._lighting_randomizer.initialize()
            if not success:
                print("[WARNING] Failed to initialize lighting randomizer")
                self.cfg.lighting.enabled = False

        self._initialized = True

        num_materials = self._material_library.num_materials if self._material_library else 0
        print(f"[INFO] Domain randomization initialized:")
        print(f"  - Materials: {self.cfg.materials.enabled} ({num_materials} variants)")
        print(f"  - Lighting: {self.cfg.lighting.enabled}")

        return True

    def randomize(self, object_names: Optional[List[str]] = None) -> None:
        """Apply material and lighting randomization.

        For each named object found in the OmniGibson scene's object registry,
        iterates through all links and visual meshes, binding a random material
        to each mesh prim independently.

        Args:
            object_names: Override which objects to randomize. If None, uses
                         ``cfg.materials.object_names``.
        """
        if not self.cfg.enabled or not self._initialized:
            return

        # Material randomization
        if self.cfg.materials.enabled and self._material_library is not None:
            self._randomize_materials(object_names)

        # Lighting randomization
        if self.cfg.lighting.enabled and self._lighting_randomizer is not None:
            self._lighting_randomizer.randomize()

    def _randomize_materials(self, object_names: Optional[List[str]] = None) -> None:
        """Apply material randomization to scene objects."""
        try:
            import omnigibson as og
        except ImportError:
            return

        names_to_randomize = object_names or self.cfg.materials.object_names
        if not names_to_randomize:
            return

        scene = og.sim.scenes[0]
        if scene is None:
            return

        for obj_name in names_to_randomize:
            obj = scene.object_registry("name", obj_name)
            if obj is None:
                continue

            mesh_paths = get_all_visual_mesh_prim_paths(obj)
            for mesh_path in mesh_paths:
                material_path = self._material_library.get_random_material(object_name=obj_name)
                if material_path:
                    self._material_library.bind_to_prim(mesh_path, material_path)

        # Optionally randomize robot materials
        if self.cfg.materials.randomize_robot and scene.robots:
            for robot in scene.robots:
                mesh_paths = get_all_visual_mesh_prim_paths(robot)
                for mesh_path in mesh_paths:
                    material_path = self._material_library.get_random_material()
                    if material_path:
                        self._material_library.bind_to_prim(mesh_path, material_path)

    def maybe_randomize_on_step(self) -> None:
        """Apply randomization if the step interval has been reached.

        Call this every simulation step. Randomization only occurs when
        ``randomize_interval_steps > 0`` and the step count is a multiple
        of that interval.
        """
        if not self.cfg.enabled or self.cfg.randomize_interval_steps <= 0:
            return

        self._step_count += 1
        if self._step_count % self.cfg.randomize_interval_steps == 0:
            self.randomize()

    def reset_step_count(self) -> None:
        """Reset the step counter (call on environment reset)."""
        self._step_count = 0

    def cleanup(self) -> None:
        """Release resources."""
        self._material_library = None
        self._lighting_randomizer = None
        self._initialized = False
        self._step_count = 0
