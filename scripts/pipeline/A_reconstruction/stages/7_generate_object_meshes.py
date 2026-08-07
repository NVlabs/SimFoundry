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

from simfoundry.models.mesh_generator import (
    Hunyuan,
    MeshGenerator,
    Pixal3D,
    ShapeGenerator,
    TextureGenerator,
    Trellis2,
    RESERVED_GENERATION_KWARGS,
    filter_generation_kwargs,
    get_mesh_generator_cls,
    make_generator,
)
from simfoundry.pipeline.stage_utils import (
    StageResult,
    bootstrap_hydra_workdir,
    finalize_stage,
    parse_iter_index,
    resolve_generation_kwargs,
)
import torch

# see https://github.com/facebookresearch/hydra/issues/2949#issue-2516892001
if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

from simfoundry import CFG_DIR

logger = logging.getLogger(__name__)
bootstrap_hydra_workdir(__file__)

hunyuan_repo_path = "../../deps/Hunyuan3D-2.1"

Hunyuan.set_repo_path(repo_path=hunyuan_repo_path)

trellis2_repo_path = "../../deps/TRELLIS.2"
Trellis2.set_repo_path(repo_path=trellis2_repo_path)

pixal3d_repo_path = "../../deps/Pixal3D"
Pixal3D.set_repo_path(repo_path=pixal3d_repo_path)


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
    generation_kwargs = resolve_generation_kwargs(cfg, cfg.s7_mesh)
    if generation_kwargs:
        logger.info("Forwarding generation kwargs to the backend: %s", generation_kwargs)
    shape_generator_cls, texture_generator_cls, mesh_generator_cls = None, None, None
    shape_generator, texture_generator, mesh_generator = None, None, None
    shape_generator_name = cfg.s7_mesh.shape_model
    # Resolve unconditionally so an invalid backend name fails fast even when that half of
    # generation is disabled (the output directory below is named after it either way).
    resolved_shape_cls = get_mesh_generator_cls(shape_generator_name)
    shape_dir = f"{out_dir}/shape/{shape_generator_name}"
    Path(shape_dir).mkdir(parents=True, exist_ok=True)
    if generate_shape:
        shape_generator_cls = resolved_shape_cls
        assert issubclass(shape_generator_cls, ShapeGenerator)
    texture_generator_name = cfg.s7_mesh.texture_model
    resolved_texture_cls = get_mesh_generator_cls(texture_generator_name)
    texture_dir = f"{out_dir}/textured_mesh/{texture_generator_name}"
    Path(texture_dir).mkdir(parents=True, exist_ok=True)
    if generate_texture:
        texture_generator_cls = resolved_texture_cls
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
        mesh_generator = make_generator(
            texture_generator_cls,
            low_vram=low_vram,
            create_shape_pipeline=True,
            create_texture_pipeline=True,
        )
    else:
        if shape_generator_cls is not None:
            shape_kwargs = dict()
            if issubclass(shape_generator_cls, MeshGenerator):
                shape_kwargs["create_shape_pipeline"] = True
                shape_kwargs["create_texture_pipeline"] = False
            shape_generator = make_generator(shape_generator_cls, low_vram=low_vram, **shape_kwargs)

        if texture_generator_cls is not None:
            texture_kwargs = dict()
            if issubclass(texture_generator_cls, MeshGenerator):
                texture_kwargs["create_shape_pipeline"] = False
                texture_kwargs["create_texture_pipeline"] = True
            texture_generator = make_generator(texture_generator_cls, low_vram=low_vram, **texture_kwargs)


    # Backends accept different option sets, so filter once here and say what was dropped rather
    # than silently ignoring a configured value (or raising TypeError on backends whose
    # generate_* signatures enumerate fixed parameters, e.g. hunyuan and direct3d).
    def prepare_kwargs(generate_fn, label):
        reserved = RESERVED_GENERATION_KWARGS[generate_fn.__name__]
        prepared = filter_generation_kwargs(generate_fn, generation_kwargs, reserved=reserved)
        dropped = sorted(set(generation_kwargs) - set(prepared))
        if dropped:
            logger.warning("%s does not accept %s — dropped from generation_kwargs", label, dropped)
        return prepared

    mesh_gen_kwargs = (
        prepare_kwargs(mesh_generator.generate_mesh, f"{shape_generator_name}.generate_mesh")
        if mesh_generator is not None else {}
    )
    shape_gen_kwargs = (
        prepare_kwargs(shape_generator.generate_shape, f"{shape_generator_name}.generate_shape")
        if shape_generator is not None else {}
    )
    texture_gen_kwargs = (
        prepare_kwargs(texture_generator.generate_texture, f"{texture_generator_name}.generate_texture")
        if texture_generator is not None else {}
    )

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
    failed_jobs = []
    for job in jobs:
        write_manifest(manifest_dir, job, status="started")
        # One bad object must not abort the rest, and must not leave a manifest stuck at
        # "started" — that state is indistinguishable from "killed mid-run" and hides which
        # object actually broke.
        try:
            # Process based on mode
            if generation_mode == GenerationMode.SHAPE_TEXTURE_SINGLE_MODEL:
                mesh_generator.generate_mesh(
                    out_fpath=job.texture_fpath,
                    shape_image_path=job.input_img_path,
                    texture_image_path=job.input_img_path,
                    visualize=cfg.visualize,
                    save_intermediates=cfg.s7_mesh.save_intermediates,
                    **mesh_gen_kwargs,
                )
                # generate_mesh saves the untextured shape as <texture_fpath>_untextured.glb but
                # never writes job.shape_fpath. Stage 8 needs the shape as an OBJ at shape_fpath,
                # so convert the untextured GLB that generate_mesh already produced.
                untextured_glb = job.texture_fpath.replace(".glb", "_untextured.glb")
                if os.path.exists(untextured_glb):
                    import trimesh as _trimesh
                    Path(job.shape_fpath).parent.mkdir(parents=True, exist_ok=True)
                    _trimesh.load(untextured_glb).export(job.shape_fpath)
            else:
                if generation_mode in {GenerationMode.SHAPE_TEXTURE_SEPARATE_MODELS, GenerationMode.SHAPE_ONLY}:
                    shape_generator.generate_shape(
                        image_path=job.input_img_path,
                        out_fpath=job.shape_fpath,
                        visualize=cfg.visualize,
                        **shape_gen_kwargs,
                    )
                if generation_mode in {GenerationMode.SHAPE_TEXTURE_SEPARATE_MODELS, GenerationMode.TEXTURE_ONLY}:
                    texture_generator.generate_texture(
                        shape_fpath=job.shape_fpath,
                        image_path=job.input_img_path,
                        out_fpath=job.texture_fpath,
                        visualize=cfg.visualize,
                        **texture_gen_kwargs,
                    )
        except KeyboardInterrupt:
            # Operator interrupt is not a per-object failure; leave the loop immediately.
            raise
        except Exception as exc:
            logger.exception("Mesh generation failed for %s", job.mesh_name)
            failed_jobs.append(job.mesh_name)
            write_manifest(
                manifest_dir, job, status="failed",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )
        else:
            write_manifest(manifest_dir, job, status="finished")

        # Use the already-resolved local, not cfg.low_vram: the top-level key exists only in
        # real2sim_cfg.yaml, so task configs that do not inherit it (e.g. PutCupInBowl.yaml)
        # raised ConfigAttributeError here after the first object had already been generated.
        if low_vram:
            torch.cuda.empty_cache()

    logger.info("="*60)
    if failed_jobs:
        logger.error("Object meshes generation finished with %s failure(s): %s",
                     len(failed_jobs), ", ".join(failed_jobs))
    else:
        logger.info("Object meshes generation complete!")
    logger.info("="*60)

    finalize_stage(
        stage_cfg=cfg.s7_mesh,
        out_dir=cfg.s7_mesh.out_dir,
        result=StageResult(
            success=not failed_jobs,
            additional_info={
                "n_jobs": len(jobs),
                "failed_jobs": failed_jobs,
                "requested_indices": sorted(requested_indices) if requested_indices is not None else None,
            },
        ),
    )

    # Raise AFTER finalize_stage so the result payload and per-object manifests are still
    # written. The orchestrator launches stages with subprocess.run(..., check=True)
    # (simfoundry/pipeline/orchestrator.py), which only inspects the exit code — it never reads
    # StageResult.success. Returning normally here therefore reported a clean stage even when
    # every mesh failed, and stages 8+ went on to pose-match and sim-ready objects whose .glb
    # was never produced. SystemExit matches how the auto_bg stages signal fatal conditions.
    if failed_jobs:
        raise SystemExit(
            f"Mesh generation failed for {len(failed_jobs)} of {len(jobs)} object(s): "
            f"{', '.join(failed_jobs)}. See the per-object manifests in "
            f"{manifest_dir} for the recorded error."
        )


if __name__ == "__main__":
    main()
