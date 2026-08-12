#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Set the opacity threshold on USD materials that carry an alpha channel.

TRELLIS.2 bakes transparency into the texture it generates, so the URDF importer wires the
texture up as an `OmniPBR_Opacity` material with `enable_opacity_texture` on. That material's
`opacity_threshold` defaults to 0.0, which selects alpha *blending* -- every texel with alpha
below 1 is rendered semi-transparent and order-dependent, so the object comes out looking like
a cloud of particles instead of a solid surface.

A non-zero threshold switches the material to alpha *cutout*: a texel is either fully opaque or
fully discarded. 0.5 is the usual split point and matches what you would set by hand on the
shader prim in the Isaac Sim property panel.

Only materials that actually have an opacity texture are touched, so fully opaque materials
keep their existing behavior.

Usage:
    python set_usd_opacity_threshold.py <usd_path> [--threshold 0.5]
"""
import argparse
import os
import site
import sys
from pathlib import Path

# Set headless mode before any Isaac Sim imports
os.environ["OMNIGIBSON_HEADLESS"] = "1"
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

_PXR_BOOTSTRAP_ENV = "SIMFOUNDRY_PXR_BOOTSTRAPPED"

#: Shader inputs that indicate the material carries a real alpha channel.
_OPACITY_TEXTURE_INPUT = "opacity_texture"
_ENABLE_OPACITY_TEXTURE_INPUT = "enable_opacity_texture"
_THRESHOLD_INPUT = "opacity_threshold"


def _prepend_env_path(env: dict[str, str], key: str, value: str) -> None:
    current = env.get(key)
    paths = [] if not current else current.split(os.pathsep)
    if value not in paths:
        env[key] = value if not current else value + os.pathsep + current


def _find_usd_extension_root() -> Path:
    roots: list[Path] = []
    for site_root in site.getsitepackages():
        roots.extend(Path(site_root).glob("isaacsim/extscache/omni.usd.libs-*"))
    user_site = site.getusersitepackages()
    if user_site:
        roots.extend(Path(user_site).glob("isaacsim/extscache/omni.usd.libs-*"))

    valid_roots = sorted(root for root in roots if (root / "pxr").is_dir() and (root / "bin").is_dir())
    if not valid_roots:
        raise RuntimeError("Could not locate Isaac Sim omni.usd.libs extension containing pxr bindings.")
    return valid_roots[-1]


def _bootstrap_pxr():
    """Import pxr, re-execing once with the Isaac Sim USD libs on the path if needed.

    Editing material inputs is pure USD authoring, so this deliberately avoids starting a
    SimulationApp -- same approach as reparent_usd_joints.py.
    """
    try:
        from pxr import Sdf, Usd, UsdShade

        return Sdf, Usd, UsdShade
    except (ImportError, ModuleNotFoundError):
        if os.environ.get(_PXR_BOOTSTRAP_ENV) == "1":
            raise
        usd_root = _find_usd_extension_root()
        env = os.environ.copy()
        _prepend_env_path(env, "PYTHONPATH", str(usd_root))
        _prepend_env_path(env, "LD_LIBRARY_PATH", str(Path(sys.prefix) / "lib"))
        _prepend_env_path(env, "LD_LIBRARY_PATH", str(usd_root / "bin"))
        env[_PXR_BOOTSTRAP_ENV] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)
        raise RuntimeError("Failed to re-exec with USD bindings enabled.")


def _has_alpha_channel(shader, UsdShade) -> bool:
    """True when the material is textured with transparency, so cutout applies to it."""
    texture_input = shader.GetInput(_OPACITY_TEXTURE_INPUT)
    if texture_input is not None and texture_input.Get():
        return True
    enable_input = shader.GetInput(_ENABLE_OPACITY_TEXTURE_INPUT)
    return bool(enable_input is not None and enable_input.Get())


def set_opacity_threshold(usd_path: str, threshold: float = 0.5) -> int:
    """Set `inputs:opacity_threshold` on every alpha-textured material in `usd_path`.

    Returns:
        int: Number of shaders changed. 0 means nothing needed it, which is not an error --
            an object whose material has no alpha channel renders correctly already.
    """
    Sdf, Usd, UsdShade = _bootstrap_pxr()

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"Error: Could not open USD stage at {usd_path}")
        return 0

    changed = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader":
            continue
        shader = UsdShade.Shader(prim)
        if not _has_alpha_channel(shader, UsdShade):
            continue

        threshold_input = shader.GetInput(_THRESHOLD_INPUT)
        if threshold_input is None:
            threshold_input = shader.CreateInput(_THRESHOLD_INPUT, Sdf.ValueTypeNames.Float)

        current = threshold_input.Get()
        if current is not None and abs(float(current) - threshold) < 1e-9:
            continue
        threshold_input.Set(threshold)
        print(f"  {prim.GetPath()}: opacity_threshold {current} -> {threshold}")
        changed += 1

    if changed:
        stage.GetRootLayer().Save()
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd_path")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not os.path.isfile(args.usd_path):
        print(f"Error: USD file not found: {args.usd_path}")
        return 1

    changed = set_opacity_threshold(args.usd_path, args.threshold)
    print(f"Set opacity_threshold={args.threshold} on {changed} material(s) in {args.usd_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
