# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate meshes from upsampled per-object RGBA images."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import json
import logging
import os
from pathlib import Path

import hydra

from digital_cousins.models.mesh_generator import (
    Direct3D,
    Hunyuan,
    MeshGenerator,
    ShapeGenerator,
    TextureGenerator,
    Trellis,
    Trellis2,
)
from digital_cousins.pipeline.stage_utils import (
    StageResult,
    bootstrap_hydra_workdir,
    finalize_stage,
    parse_iter_index,
)
from digital_cousins.utils.python_utils import assert_valid_key
import torch

# see https://github.com/facebookresearch/hydra/issues/2949#issue-2516892001
if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

from digital_cousins import CFG_DIR

logger = logging.getLogger(__name__)
bootstrap_hydra_workdir(__file__)

MESH_GENERATORS = {
    "direct3d": Direct3D,
    "hunyuan": Hunyuan,
    "trellis": Trellis,
    "trellis2": Trellis2,
}

hunyuan_repo_path = "../../deps/Hunyuan3D-2.1"

Hunyuan.set_repo_path(repo_path=hunyuan_repo_path)

trellis2_repo_path = "../../deps/TRELLIS.2"
Trellis2.set_repo_path(repo_path=trellis2_repo_path)


class GenerationMode(IntEnum):
    SHAPE_TEXTURE_SINGLE_MODEL = 0
    SHAPE_TEXTURE_SEPARATE_MODELS = 1
    SHAPE_ONLY = 2
    TEXTURE_ONLY = 3


@dataclass
class MeshJob:
    idx: int
    mesh_name: str
    input_img_path: str
    shape_fpath: str
    texture_fpath: str


def resolve_requested_indices(cfg) -> set[int] | None:
    """
    Resolve legacy + new index filters.
    - `s7_mesh.object_indices`: preferred.
    - `s7_mesh.only_indices`: backwards-compatible alias.
    """
    raw = cfg.s7_mesh.get("object_indices", None)
    if raw is None:
        raw = cfg.s7_mesh.get("only_indices", None)
    if raw is None:
        return None
    if isinstance(raw, int):
        return {raw}
    return {int(v) for v in raw}


def discover_mesh_jobs(upsampled_dir: str, shape_dir: str, texture_dir: str, allowed_indices: set[int] | None) -> list[MeshJob]:
    """Enumerate object jobs from `_transparent.png` artifacts."""
    jobs: list[MeshJob] = []
    for filename in sorted(os.listdir(upsampled_dir)):
        suffix = "_transparent.png"
        if not filename.endswith(suffix):
            continue
        mesh_name = filename.split(suffix)[0]
        idx = parse_iter_index(mesh_name)
        if idx is None:
            continue
        if allowed_indices is not None and idx not in allowed_indices:
            continue
        jobs.append(
            MeshJob(
                idx=idx,
                mesh_name=mesh_name,
                input_img_path=f"{upsampled_dir}/{filename}",
                shape_fpath=f"{shape_dir}/{mesh_name}_shape.obj",
                texture_fpath=f"{texture_dir}/{mesh_name}_mesh.glb",
            )
        )
    return jobs


def write_manifest(manifest_dir: str, job: MeshJob, status: str, details: dict | None = None) -> None:
    """Write per-object status for interpretability and post-hoc visualization."""
    Path(manifest_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "iter_idx": job.idx,
        "mesh_name": job.mesh_name,
        "input_img_path": job.input_img_path,
        "shape_fpath": job.shape_fpath,
        "texture_fpath": job.texture_fpath,
        "status": status,
    }
    if details:
        payload.update(details)
    with open(f"{manifest_dir}/{job.mesh_name}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    img_dir = cfg.s6_upsample.out_dir
    out_dir = cfg.s7_mesh.out_dir
    upsampled_dir = cfg.s6_upsample.out_dir + "/upsampled"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("Starting object meshes generation pipeline...")
    logger.info(f"Output directory: {out_dir}")
    logger.info("="*60)

    # Create model
    generate_shape, generate_texture = cfg.s7_mesh.generate_shape, cfg.s7_mesh.generate_texture
    low_vram = cfg.s7_mesh.get("low_vram", False)
    if low_vram:
        logger.info("Low VRAM mode enabled — will use model CPU offloading to reduce GPU memory")
    shape_generator_cls, texture_generator_cls, mesh_generator_cls = None, None, None
    shape_generator, texture_generator, mesh_generator = None, None, None
    shape_generator_name = cfg.s7_mesh.shape_model
    assert_valid_key(key=shape_generator_name, valid_keys=MESH_GENERATORS, name="shape_generator")
    shape_dir = f"{out_dir}/shape/{shape_generator_name}"
    Path(shape_dir).mkdir(parents=True, exist_ok=True)
    if generate_shape:
        shape_generator_cls = MESH_GENERATORS[shape_generator_name]
        assert issubclass(shape_generator_cls, ShapeGenerator)
    texture_generator_name = cfg.s7_mesh.texture_model
    assert_valid_key(key=texture_generator_name, valid_keys=MESH_GENERATORS, name="texture_generator")
    texture_dir = f"{out_dir}/textured_mesh/{texture_generator_name}"
    Path(texture_dir).mkdir(parents=True, exist_ok=True)
    if generate_texture:
        texture_generator_cls = MESH_GENERATORS[texture_generator_name]
        assert issubclass(texture_generator_cls, TextureGenerator)

    
    # Special torch functional fix for Hunyuan
    if shape_generator_name == "hunyuan":
        import sys
        sys.path.insert(0, f"{hunyuan_repo_path}")
        from torchvision_fix import apply_fix
        apply_fix()

    # Determine generation mode
    if generate_shape and generate_texture:
        if shape_generator_cls == texture_generator_cls:
            generation_mode = GenerationMode.SHAPE_TEXTURE_SINGLE_MODEL
        else:
            generation_mode = GenerationMode.SHAPE_TEXTURE_SEPARATE_MODELS
    elif not (generate_shape or generate_texture):
        # No class specified for either, raise error
        raise ValueError("At least shape generator or texture generator should be specified!")
    else:
        if generate_shape:
            generation_mode = GenerationMode.SHAPE_ONLY
        else:
            generation_mode = GenerationMode.TEXTURE_ONLY

    # Sanity check values
    assert generation_mode is not None

    # Create generators
    if generation_mode == GenerationMode.SHAPE_TEXTURE_SINGLE_MODEL:
        # Both shape + texture and shared class, so only create once
        assert issubclass(shape_generator_cls, MeshGenerator)
        mesh_generator = texture_generator_cls(
            create_shape_pipeline=True,
            create_texture_pipeline=True,
            low_vram=low_vram,
        )
    else:
        if shape_generator_cls is not None:
            shape_kwargs = dict()
            if issubclass(shape_generator_cls, MeshGenerator):
                shape_kwargs["create_shape_pipeline"] = True
                shape_kwargs["create_texture_pipeline"] = False
                shape_kwargs["low_vram"] = low_vram
            shape_generator = shape_generator_cls(**shape_kwargs)

        if texture_generator_cls is not None:
            texture_kwargs = dict()
            if issubclass(texture_generator_cls, MeshGenerator):
                texture_kwargs["create_shape_pipeline"] = False
                texture_kwargs["create_texture_pipeline"] = True
                texture_kwargs["low_vram"] = low_vram
            texture_generator = texture_generator_cls(**texture_kwargs)


    requested_indices = resolve_requested_indices(cfg)
    jobs = discover_mesh_jobs(
        upsampled_dir=upsampled_dir,
        shape_dir=shape_dir,
        texture_dir=texture_dir,
        allowed_indices=requested_indices,
    )
    manifest_dir = f"{out_dir}/manifest"
    logger.info("Discovered %s mesh jobs", len(jobs))

    # Iterate over all discovered jobs and pass them through generation process.
    for job in jobs:
        write_manifest(manifest_dir, job, status="started")
        # Process based on mode
        if generation_mode == GenerationMode.SHAPE_TEXTURE_SINGLE_MODEL:
            mesh_generator.generate_mesh(
                out_fpath=job.texture_fpath,
                shape_image_path=job.input_img_path,
                texture_image_path=job.input_img_path,
                visualize=cfg.visualize,
                save_intermediates=cfg.s7_mesh.save_intermediates,
            )
        else:
            if generation_mode in {GenerationMode.SHAPE_TEXTURE_SEPARATE_MODELS, GenerationMode.SHAPE_ONLY}:
                shape_generator.generate_shape(
                    image_path=job.input_img_path,
                    out_fpath=job.shape_fpath,
                    visualize=cfg.visualize,
                )
            if generation_mode in {GenerationMode.SHAPE_TEXTURE_SEPARATE_MODELS, GenerationMode.TEXTURE_ONLY}:
                texture_generator.generate_texture(
                    shape_fpath=job.shape_fpath,
                    image_path=job.input_img_path,
                    out_fpath=job.texture_fpath,
                    visualize=cfg.visualize,
                )
        write_manifest(manifest_dir, job, status="finished")

        if cfg.low_vram:
            torch.cuda.empty_cache()

    logger.info("="*60)
    logger.info("Object meshes generation complete!")
    logger.info("="*60)
    
    finalize_stage(
        stage_cfg=cfg.s7_mesh,
        out_dir=cfg.s7_mesh.out_dir,
        result=StageResult(
            success=True,
            additional_info={
                "n_jobs": len(jobs),
                "requested_indices": sorted(requested_indices) if requested_indices is not None else None,
            },
        ),
    )

if __name__ == "__main__":
    main()
