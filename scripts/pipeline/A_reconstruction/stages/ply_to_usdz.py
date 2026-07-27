# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are adapted from 3dgrut (https://github.com/nv-tlabs/3dgrut)
# at commit a37ef721012dea0f29c0fcfff2d525023b4e854a, Copyright (c) NVIDIA CORPORATION
# & AFFILIATES, licensed under the Apache License, Version 2.0.

"""
Standalone 3DGS PLY → USDZ converter for Isaac Sim / OmniGibson.

Converts a standard Gaussian Splatting PLY (nerfstudio splatfacto format) to
the NVIDIA NuRec USDZ format that Isaac Sim can render as a Gaussian splat.

Does NOT require the 3dgrut conda env — only needs numpy, pxr, msgpack, and plyfile,
all of which are present in the cdc env.

Usage:
    mamba run -n cdc python scripts/pipeline/A_reconstruction/stages/ply_to_usdz.py \\
        Data/output2/s2c_gs/export/splat.ply \\
        Data/output2/s2c_gs/export/splat.usdz
"""
import argparse
import gzip
import io
import struct
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import msgpack
import numpy as np
from plyfile import PlyData
from pxr import Gf, Sdf, Usd, UsdGeom, UsdVol


# ---------------------------------------------------------------------------
# USD stage helpers (adapted from threedgrut/export/usd/stage_utils.py)
# ---------------------------------------------------------------------------

@dataclass
class NamedSerialized:
    filename: str
    serialized: bytes

    def save_to_zip(self, zf: zipfile.ZipFile) -> None:
        zf.writestr(self.filename, self.serialized)


@dataclass
class NamedUSDStage:
    filename: str
    stage: Usd.Stage

    def save_to_zip(self, zf: zipfile.ZipFile) -> None:
        with tempfile.NamedTemporaryFile(suffix=".usda", delete=False) as f:
            tmp_path = f.name
        # Export the root layer (not the composed/flattened stage) to preserve
        # metadata (defaultPrim) and avoid flattening bugs with in-memory references.
        self.stage.GetRootLayer().Export(tmp_path)
        zf.write(tmp_path, self.filename)
        Path(tmp_path).unlink(missing_ok=True)


def _initialize_usd_stage(up_axis: str = "Z") -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, up_axis)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage


def _author_nurec_render_settings(stage: Usd.Stage) -> None:
    # These settings are required for NuRec GS compositing.
    # DO NOT include rtx:rendermode here — Isaac Sim fires renderSettingsChanged when
    # a loaded layer's customLayerData changes the rendermode, which restarts the NuRec
    # compositor (causing a ~20-frame blackout).  NuRec works in Real-Time mode;
    # the editor's pre-config block sets all other settings before the GS loads.
    render_settings = {
        "rtx:directLighting:sampledLighting:samplesPerPixel": 8,
        "rtx:post:histogram:enabled": False,
        # NuRec outputs linear-space colors. The compositor must invert the scene's
        # tone-map before blending and then re-apply it (3dgrut default = True).
        # False causes ACES exposure adaptation to crush the GS to near-black.
        "rtx:post:registeredCompositing:invertToneMap": True,
        "rtx:post:registeredCompositing:invertColorCorrection": True,
        "rtx:material:enableRefraction": False,
        "rtx:post:tonemap:op": 2,
        "rtx:raytracing:fractionalCutoutOpacity": False,
        "rtx:matteObject:visibility:secondaryRays": True,
    }
    stage.SetMetadataByDictKey("customLayerData", "renderSettings", render_settings)


# ---------------------------------------------------------------------------
# NuRec template (adapted from threedgrut/export/usd/nurec/templates.py)
# ---------------------------------------------------------------------------

def _build_nurec_template(
    positions: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    densities: np.ndarray,
    features_albedo: np.ndarray,
    features_specular: np.ndarray,
    n_active_features: int,
    dtype=np.float16,
) -> Dict[str, Any]:
    n = positions.shape[0]
    extra_signal = np.zeros((n, 0), dtype=dtype)

    template: Dict[str, Any] = {
        "nre_data": {
            "version": "0.2.576",
            "model": "nre",
            "config": {
                "layers": {
                    "gaussians": {
                        "name": "sh-gaussians",
                        "device": "cuda",
                        "density_activation": "sigmoid",
                        "scale_activation": "exp",
                        "rotation_activation": "normalize",
                        "precision": 16,
                        "particle": {
                            "density_kernel_planar": False,
                            "density_kernel_degree": 2,
                            "density_kernel_density_clamping": False,
                            "density_kernel_min_response": 0.0113,
                            "radiance_sph_degree": 3,
                        },
                        "transmittance_threshold": 0.001,
                    }
                },
                "renderer": {
                    "name": "3dgut-nrend",
                    "log_level": 3,
                    "force_update": False,
                    "update_step_train_batch_end": False,
                    "per_ray_features": False,
                    "global_z_order": False,
                    "projection": {
                        "n_rolling_shutter_iterations": 5,
                        "ut_dim": 3,
                        "ut_alpha": 1.0,
                        "ut_beta": 2.0,
                        "ut_kappa": 0.0,
                        "ut_require_all_sigma_points": False,
                        "image_margin_factor": 0.1,
                        "min_projected_ray_radius": 0.5477225575051661,
                    },
                    "culling": {
                        "rect_bounding": True,
                        "tight_opacity_bounding": True,
                        "tile_based": True,
                        "near_clip_distance": 1e-8,
                        "far_clip_distance": 3.402823466e38,
                    },
                    "render": {"mode": "kbuffer", "k_buffer_size": 0},
                },
                "name": "gaussians_primitive",
                "appearance_embedding": {"name": "skip-appearance", "embedding_dim": 0, "device": "cuda"},
                "background": {"name": "skip-background", "device": "cuda", "composite_in_linear_space": False},
            },
            "state_dict": {},
        }
    }

    sd = template["nre_data"]["state_dict"]
    sd[".gaussians_nodes.gaussians.positions"] = positions.astype(dtype).tobytes()
    sd[".gaussians_nodes.gaussians.rotations"] = rotations.astype(dtype).tobytes()
    sd[".gaussians_nodes.gaussians.scales"] = scales.astype(dtype).tobytes()
    sd[".gaussians_nodes.gaussians.densities"] = densities.astype(dtype).tobytes()
    sd[".gaussians_nodes.gaussians.features_albedo"] = features_albedo.astype(dtype).tobytes()
    sd[".gaussians_nodes.gaussians.features_specular"] = features_specular.astype(dtype).tobytes()
    sd[".gaussians_nodes.gaussians.extra_signal"] = extra_signal.tobytes()
    sd[".gaussians_nodes.gaussians.n_active_features"] = np.array([n_active_features], dtype=np.int64).tobytes()

    sd[".gaussians_nodes.gaussians.positions.shape"] = list(positions.shape)
    sd[".gaussians_nodes.gaussians.rotations.shape"] = list(rotations.shape)
    sd[".gaussians_nodes.gaussians.scales.shape"] = list(scales.shape)
    sd[".gaussians_nodes.gaussians.densities.shape"] = list(densities.shape)
    sd[".gaussians_nodes.gaussians.features_albedo.shape"] = list(features_albedo.shape)
    sd[".gaussians_nodes.gaussians.features_specular.shape"] = list(features_specular.shape)
    sd[".gaussians_nodes.gaussians.extra_signal.shape"] = list(extra_signal.shape)
    sd[".gaussians_nodes.gaussians.n_active_features.shape"] = []

    return template


# ---------------------------------------------------------------------------
# PLY reader (nerfstudio splatfacto format)
# ---------------------------------------------------------------------------

def _sh_degree_from_n_coeffs(n_dc: int, n_rest: int) -> int:
    """Infer SH degree from number of feature channels."""
    total = n_dc + n_rest  # per channel: (degree+1)^2
    per_ch = total // 3
    degree = int(per_ch ** 0.5) - 1
    return max(0, degree)


def read_gs_ply(ply_path: Path):
    """Read a nerfstudio-style 3DGS PLY and return arrays in NuRec format."""
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]

    positions = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    rotations = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    opacities = np.array(v["opacity"], dtype=np.float32).reshape(-1, 1)

    # DC SH coefficients → albedo (N, 3)
    albedo = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)

    # Higher-order SH — detect all f_rest_* fields
    rest_names = sorted([n for n in v.data.dtype.names if n.startswith("f_rest_")],
                        key=lambda s: int(s.split("_")[-1]))
    if rest_names:
        specular = np.stack([np.array(v[n], dtype=np.float32) for n in rest_names], axis=1)
        # nerfstudio stores SH rest as (N, n_rest_per_channel * 3) flattened by channel
        # Re-interpret as (N, n_rest_total) — already correct
    else:
        specular = np.zeros((positions.shape[0], 0), dtype=np.float32)

    n_rest_per_channel = len(rest_names) // 3 if rest_names else 0
    sh_degree = int((n_rest_per_channel + 1) ** 0.5)  # 0→0, 1→1, 4→2, 9→3

    print(f"  Gaussians: {positions.shape[0]:,}")
    print(f"  SH degree: {sh_degree} ({len(rest_names)} rest coeffs)")

    return positions, rotations, scales, opacities, albedo, specular, sh_degree


# ---------------------------------------------------------------------------
# USD stage builders (adapted from threedgrut serializer.py)
# ---------------------------------------------------------------------------

def _author_nurec_volume(stage: Usd.Stage, nurec_abs_path: str, positions: np.ndarray) -> None:
    # USD defaultPrim must be a ROOT prim (direct child of pseudo-root).
    # We use /World as the defaultPrim so that when OmniGibson references the USDZ,
    # /World's children appear as children of the USDObject prim.
    #
    # OmniGibson's entity_prim.update_links() scans those children for Xform prims.
    # The interactive editor expects the GS Volume at: <object_prim>/gauss/gauss
    #
    # Final hierarchy:
    #   /World               <- Xform (root prim, USD defaultPrim = "World")
    #   /World/gauss         <- Xform (becomes the "root link" when loaded by OmniGibson)
    #   /World/gauss/gauss   <- UsdVol.Volume (the actual NuRec Gaussian splat)
    #
    # nurec_abs_path must be an ABSOLUTE filesystem path so the NuRec hydra plugin can
    # find the .nurec file even when this USDA is loaded inside a USDZ reference.
    # (Relative paths inside a USDZ are cached for the first few frames then fail on reload.)

    UsdGeom.Xform.Define(stage, "/World")

    root_xform_path = "/World/gauss"
    volume_path = "/World/gauss/gauss"

    min_coord = positions.min(axis=0).tolist()
    max_coord = positions.max(axis=0).tolist()

    root_xform = UsdGeom.Xform.Define(stage, root_xform_path)
    root_xform.AddTransformOp().Set(Gf.Matrix4d(1.0))

    gauss_volume = UsdVol.Volume.Define(stage, volume_path)
    gauss_prim = gauss_volume.GetPrim()

    gauss_prim.CreateAttribute("omni:nurec:isNuRecVolume", Sdf.ValueTypeNames.Bool).Set(True)
    gauss_prim.CreateAttribute("omni:nurec:useProxyTransform", Sdf.ValueTypeNames.Bool).Set(False)

    density_path = volume_path + "/density_field"
    density_field = stage.DefinePrim(density_path, "OmniNuRecFieldAsset")
    gauss_volume.CreateFieldRelationship("density", density_path)

    emissive_path = volume_path + "/emissive_color_field"
    emissive_field = stage.DefinePrim(emissive_path, "OmniNuRecFieldAsset")
    gauss_volume.CreateFieldRelationship("emissiveColor", emissive_path)

    density_field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set(nurec_abs_path)
    density_field.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set("density")
    density_field.CreateAttribute("fieldDataType", Sdf.ValueTypeNames.Token).Set("float")
    density_field.CreateAttribute("fieldRole", Sdf.ValueTypeNames.Token).Set("density")

    emissive_field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set(nurec_abs_path)
    emissive_field.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set("emissiveColor")
    emissive_field.CreateAttribute("fieldDataType", Sdf.ValueTypeNames.Token).Set("float3")
    emissive_field.CreateAttribute("fieldRole", Sdf.ValueTypeNames.Token).Set("emissiveColor")
    emissive_field.CreateAttribute("omni:nurec:ccmR", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1.0, 0.0, 0.0, 0.0))
    emissive_field.CreateAttribute("omni:nurec:ccmG", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0.0, 1.0, 0.0, 0.0))
    emissive_field.CreateAttribute("omni:nurec:ccmB", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0.0, 0.0, 1.0, 0.0))

    gauss_prim.CreateAttribute(
        "extent", Sdf.ValueTypeNames.Float3Array
    ).Set([Gf.Vec3f(*min_coord), Gf.Vec3f(*max_coord)])
    gauss_prim.CreateAttribute("omni:nurec:offset", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3d(0.0, 0.0, 0.0))
    gauss_prim.CreateAttribute("omni:nurec:crop:minBounds", Sdf.ValueTypeNames.Float3).Set(
        Gf.Vec3d(*[float(v) for v in min_coord])
    )
    gauss_prim.CreateAttribute("omni:nurec:crop:maxBounds", Sdf.ValueTypeNames.Float3).Set(
        Gf.Vec3d(*[float(v) for v in max_coord])
    )
    gauss_prim.CreateRelationship("proxy")


def _build_gauss_stage(nurec_abs_path: str, positions: np.ndarray) -> NamedUSDStage:
    stage = _initialize_usd_stage()
    _author_nurec_volume(stage, nurec_abs_path, positions)
    _author_nurec_render_settings(stage)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    return NamedUSDStage(filename="default.usda", stage=stage)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(ply_path: Path, out_usdz: Path) -> None:
    print(f"Reading {ply_path} ...")
    positions, rotations, scales, opacities, albedo, specular, sh_degree = read_gs_ply(ply_path)

    # The .nurec file is written to disk ALONGSIDE the .usdz (not inside it).
    # The USDA inside the USDZ references it by absolute path so the NuRec Hydra plugin
    # can always reload it from the filesystem. Relative paths inside a USDZ are only
    # resolved from the zip namespace on first load; subsequent reloads fail silently
    # after a few frames, causing the GS to disappear.
    out_usdz.parent.mkdir(parents=True, exist_ok=True)
    nurec_filename = out_usdz.stem + ".nurec"
    nurec_path = out_usdz.parent / nurec_filename
    nurec_abs_path = str(nurec_path.resolve())

    print(f"Building NuRec template (sh_degree={sh_degree}) ...")
    template = _build_nurec_template(
        positions=positions,
        rotations=rotations,
        scales=scales,
        densities=opacities,
        features_albedo=albedo,
        features_specular=specular,
        n_active_features=sh_degree,
    )

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=0) as f:
        f.write(msgpack.packb(template))
    nurec_bytes = buf.getvalue()
    print(f"  NuRec payload: {len(nurec_bytes) / 1e6:.1f} MB")

    print(f"Writing {nurec_path} ...")
    nurec_path.write_bytes(nurec_bytes)

    print("Building USD stage ...")
    default_stage = _build_gauss_stage(nurec_abs_path, positions)

    print(f"Writing {out_usdz} ...")
    with zipfile.ZipFile(out_usdz, "w", compression=zipfile.ZIP_STORED) as zf:
        default_stage.save_to_zip(zf)   # default.usda first (USDZ spec)
        # nurec is NOT inside the zip — it lives on disk at nurec_abs_path

    print(f"Done. {out_usdz} ({out_usdz.stat().st_size / 1e6:.1f} MB)")
    print(f"      {nurec_path} ({nurec_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_ply", type=Path, help="Input PLY file (nerfstudio splatfacto format)")
    parser.add_argument("output_usdz", type=Path, help="Output USDZ file")
    args = parser.parse_args()
    convert(args.input_ply, args.output_usdz)
