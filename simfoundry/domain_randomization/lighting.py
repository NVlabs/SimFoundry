# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Lighting randomization utilities for OmniGibson environments.

Uses OmniGibson's skybox API (dome light) for:
- Intensity and color temperature randomization
- HDRI texture-based environment lighting
- HDRI rotation randomization
"""

import os
import glob
import numpy as np
from typing import Optional, List, Tuple

from .configs import LightingRandomizationCfg


def color_temperature_to_rgb(temperature: float) -> Tuple[float, float, float]:
    """Convert color temperature in Kelvin to RGB values.

    Uses the Tanner Helland algorithm for approximation.

    Args:
        temperature: Color temperature in Kelvin (1000-40000K).

    Returns:
        RGB tuple with values in [0, 1].
    """
    temperature = max(1000, min(40000, temperature))
    temperature = temperature / 100.0

    # Red
    if temperature <= 66:
        red = 255
    else:
        red = temperature - 60
        red = 329.698727446 * (red ** -0.1332047592)
        red = max(0, min(255, red))

    # Green
    if temperature <= 66:
        green = temperature
        green = 99.4708025861 * np.log(green) - 161.1195681661
        green = max(0, min(255, green))
    else:
        green = temperature - 60
        green = 288.1221695283 * (green ** -0.0755148492)
        green = max(0, min(255, green))

    # Blue
    if temperature >= 66:
        blue = 255
    elif temperature <= 19:
        blue = 0
    else:
        blue = temperature - 10
        blue = 138.5177312231 * np.log(blue) - 305.0447927307
        blue = max(0, min(255, blue))

    return (red / 255.0, green / 255.0, blue / 255.0)


def list_hdri_files(folder_path: str) -> List[str]:
    """List all HDR/EXR files in a folder recursively.

    Args:
        folder_path: Path to folder containing HDRI files.

    Returns:
        List of file paths (sorted, .hdr files first then .exr).
    """
    if not os.path.exists(folder_path):
        return []

    hdr_files = glob.glob(os.path.join(folder_path, "**", "*.hdr"), recursive=True)
    hdr_files.extend(glob.glob(os.path.join(folder_path, "**", "*.HDR"), recursive=True))
    exr_files = glob.glob(os.path.join(folder_path, "**", "*.exr"), recursive=True)
    exr_files.extend(glob.glob(os.path.join(folder_path, "**", "*.EXR"), recursive=True))

    # Deduplicate (case-insensitive filesystems may produce duplicates)
    all_files = list(dict.fromkeys(sorted(hdr_files) + sorted(exr_files)))
    return all_files


class LightingRandomizer:
    """Manages lighting randomization via OmniGibson's skybox.

    The skybox in OmniGibson is a dome light accessible via ``og.sim.skybox``.
    This class randomizes its intensity, color, HDRI texture, and orientation.

    Usage:
        randomizer = LightingRandomizer(cfg)
        randomizer.initialize()  # Call once at startup
        randomizer.randomize()   # Call on each reset
    """

    def __init__(self, cfg: LightingRandomizationCfg):
        self.cfg = cfg
        self._hdri_files: List[str] = []
        self._hdri_index = 0
        self._initialized = False

    @property
    def hdri_files(self) -> List[str]:
        return self._hdri_files

    def initialize(self) -> bool:
        """Initialize the randomizer: discover HDRI files.

        Returns:
            True if initialization succeeded.
        """
        if self._initialized:
            return True

        if not self.cfg.enabled:
            print("[INFO] Lighting randomization disabled")
            return True

        # Discover HDRI files
        if self.cfg.use_hdri_textures and self.cfg.hdr_backgrounds_root_dir:
            self._hdri_files = list_hdri_files(self.cfg.hdr_backgrounds_root_dir)
            if self._hdri_files:
                print(f"[INFO] Found {len(self._hdri_files)} HDRI files")
            else:
                print(f"[WARNING] No HDRI files found in {self.cfg.hdr_backgrounds_root_dir}")
                print("[INFO] Falling back to intensity/color randomization only")
                self.cfg.use_hdri_textures = False

        self._initialized = True
        return True

    def randomize(self) -> bool:
        """Apply lighting randomization via og.sim.skybox.

        Randomizes dome light intensity, color temperature, HDRI texture,
        and HDRI orientation.

        Returns:
            True if randomization succeeded.
        """
        if not self.cfg.enabled:
            return True

        try:
            import omnigibson as og
            from omnigibson.utils.transform_utils import euler2quat
            import torch as th
        except ImportError as e:
            print(f"[WARNING] Failed to import OmniGibson modules: {e}")
            return False

        skybox = og.sim.skybox
        if skybox is None:
            print("[WARNING] No skybox available in the scene")
            return False

        # Randomize intensity
        intensity = np.random.uniform(*self.cfg.dome_intensity_range)
        skybox.intensity = intensity

        # Randomize color via color temperature
        temperature = np.random.uniform(*self.cfg.dome_color_temperature_range)
        rgb = color_temperature_to_rgb(temperature)
        skybox.color = list(rgb)

        # Apply HDRI texture
        if self.cfg.use_hdri_textures and self._hdri_files:
            hdri_path = self._hdri_files[np.random.randint(len(self._hdri_files))]
            skybox.texture_file_path = hdri_path

        # Randomize HDRI rotation (Z-axis)
        if self.cfg.randomize_hdri_rotation:
            rotation_deg = np.random.uniform(*self.cfg.hdri_rotation_range)
            rotation_rad = np.radians(rotation_deg)
            orientation = euler2quat(th.tensor([0.0, 0.0, rotation_rad]))
            skybox.set_position_orientation(orientation=orientation)

        return True

    def cycle_next_hdri(self) -> bool:
        """Cycle to the next HDRI background (for interactive use).

        Returns:
            True if HDRI was changed.
        """
        if not self._hdri_files:
            print("[WARNING] No HDRI files available to cycle")
            return False

        try:
            import omnigibson as og
        except ImportError:
            return False

        self._hdri_index = (self._hdri_index + 1) % len(self._hdri_files)
        hdri_path = self._hdri_files[self._hdri_index]
        og.sim.skybox.texture_file_path = hdri_path
        print(f"[INFO] HDRI: {os.path.basename(hdri_path)} ({self._hdri_index + 1}/{len(self._hdri_files)})")
        return True
