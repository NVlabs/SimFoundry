# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from env depending on which mesh generator is used!

Requires installing (depending on generator used):

- Hunyuan2.1, see https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
- TRELLIS, see https://github.com/microsoft/TRELLIS
- TRELLIS.2, see https://github.com/microsoft/TRELLIS.2
"""
from simfoundry.models.mesh_generator import (
    Hunyuan,
    MeshGenerator,
    Pixal3D,
    ShapeGenerator,
    TextureGenerator,
    Trellis2,
    RESERVED_GENERATION_KWARGS,
    filter_generation_kwargs,
    publish_mesh_atomically,
    get_mesh_generator_cls,
    make_generator,
)
from pathlib import Path
import math
import os
import tempfile
from enum import IntEnum
import hydra
from omegaconf import OmegaConf

from simfoundry import CFG_DIR, REPO_DIR
from simfoundry.pipeline.stage_utils import bootstrap_hydra_workdir, resolve_generation_kwargs

import torch
import trimesh
import json

# see https://github.com/facebookresearch/hydra/issues/2949#issue-2516892001
if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

bootstrap_hydra_workdir(__file__)
REPO_ROOT = Path(REPO_DIR)

hunyuan_repo_path = str(REPO_ROOT / "deps" / "Hunyuan3D-2.1")

Hunyuan.set_repo_path(repo_path=hunyuan_repo_path)

trellis2_repo_path = str(REPO_ROOT / "deps" / "TRELLIS.2")
Trellis2.set_repo_path(repo_path=trellis2_repo_path)

pixal3d_repo_path = str(REPO_ROOT / "deps" / "Pixal3D")
Pixal3D.set_repo_path(repo_path=pixal3d_repo_path)


class GenerationMode(IntEnum):
    SHAPE_TEXTURE_SINGLE_MODEL = 0
    SHAPE_TEXTURE_SEPARATE_MODELS = 1
    SHAPE_ONLY = 2
    TEXTURE_ONLY = 3


def get_aabb(mesh):
    """
    Returns:
        bounds: (2, 3) array -> [[minx, miny, minz], [maxx, maxy, maxz]]
        extents: (3,) array -> (dx, dy, dz)
    """
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    return bounds, extents


def rescale_glb(glb_path, scale):
    """
    Rescale a GLB mesh by a uniform factor.
    scale: float or (3,) array
    """
    mesh = trimesh.load(glb_path, force="scene")

    if isinstance(scale, (int, float)):
        scale = [scale, scale, scale]

    mesh.apply_scale(scale)
    mesh.export(glb_path)


def generate_rescaled_mesh(out_fpath, produce_fn, canonical_scene):
    """
    Generates into a staging path, rescales to the canonical mesh, then publishes once, atomically.

    Generation and rescaling have to land at the final path together. Publishing the generated
    mesh first and rescaling it in place leaves a window in which `out_fpath` exists but holds an
    unscaled mesh — and the loop below skips any target that already exists, so an interruption
    (or a crash mid-export) in that window makes the wrong mesh permanent: every later run treats
    it as finished. Staging also means a failed rescale never destroys a previously good file.

    Args:
        out_fpath (str): Final path to publish
        produce_fn (callable): Called with the staging path; must write the mesh there
        canonical_scene (trimesh.Scene): Reference whose largest extent sets the target size

    Returns:
        float: The scale factor applied
    """
    out_dir = os.path.dirname(os.path.abspath(out_fpath))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    extension = os.path.splitext(out_fpath)[1] or ".glb"
    fd, staging_fpath = tempfile.mkstemp(dir=out_dir, prefix=".tmp-staging-", suffix=extension)
    os.close(fd)
    try:
        produce_fn(staging_fpath)
        gen_scene = trimesh.load(staging_fpath, force="scene")
        _, gen_extents = get_aabb(gen_scene)
        _, can_extents = get_aabb(canonical_scene)
        gen_size = float(gen_extents.max())
        can_size = float(can_extents.max())
        if not (gen_size > 0.0) or not math.isfinite(gen_size):
            raise ValueError(
                f"Generated mesh has a degenerate bounding box (largest extent {gen_size}); "
                f"refusing to rescale and publish {out_fpath}"
            )
        scale = can_size / gen_size
        gen_scene.apply_scale(scale)
        publish_mesh_atomically(out_fpath, gen_scene.export)
        return scale
    finally:
        # Sweep by stem, not just the exact path. Backends derive sibling intermediates from the
        # path they are handed — MeshGenerator.generate_mesh writes
        # `<staging>_untextured.glb` — and because the staging stem is random, each failed or
        # retried cousin would otherwise strand a fresh hidden file next to the published mesh.
        stem = os.path.splitext(os.path.basename(staging_fpath))[0]
        for leftover in Path(out_dir).glob(f"{stem}*"):
            try:
                leftover.unlink()
            except OSError:
                pass


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    img_dir = cfg.prompt_cousin_structured.out_dir
    out_dir = cfg.cousin_generation.out_dir
    comb_dir = cfg.generate_cousins_combination.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # read cousins combinations
    comb_file = comb_dir + "/combinations.json"
    with open(comb_file, "r") as f:
        combinations = json.load(f)

    img_file_name = set()

    for combo in combinations:
        for path in combo.values():
            img_file_name.add(path)

    # print(img_dir)

    # Create model
    generate_shape, generate_texture = cfg.cousin_generation.generate_shape, cfg.cousin_generation.generate_texture
    shape_generator_cls, texture_generator_cls, mesh_generator_cls = None, None, None
    shape_generator, texture_generator, mesh_generator = None, None, None
    shape_generator_name = cfg.cousin_generation.shape_model
    # Resolve unconditionally so an invalid backend name fails fast even when that half of
    # generation is disabled (the output directory below is named after it either way).
    resolved_shape_cls = get_mesh_generator_cls(shape_generator_name)
    shape_dir = f"{out_dir}/shape/{shape_generator_name}"
    Path(shape_dir).mkdir(parents=True, exist_ok=True)
    if generate_shape:
        shape_generator_cls = resolved_shape_cls
        assert issubclass(shape_generator_cls, ShapeGenerator)
    texture_generator_name = cfg.cousin_generation.texture_model
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
    low_vram = bool(cfg.get("low_vram", False) or cfg.cousin_generation.get("low_vram", False))
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
    generation_kwargs = resolve_generation_kwargs(cfg, cfg.cousin_generation)
    if generation_kwargs:
        print(f"Forwarding generation kwargs to the backend: {generation_kwargs}")

    def prepare_kwargs(generate_fn, label):
        reserved = RESERVED_GENERATION_KWARGS[generate_fn.__name__]
        prepared = filter_generation_kwargs(generate_fn, generation_kwargs, reserved=reserved)
        dropped = sorted(set(generation_kwargs) - set(prepared))
        if dropped:
            print(f"WARNING: {label} does not accept {dropped} - dropped from generation_kwargs")
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

    


    # Iterate over all files in the img dir, and pass them through generation process
    for filename in sorted(img_file_name):

        # if any(f"iter_{i}" in filename for i in range(0, 6)):
        #     continue

        # Process any PNG file
        if not filename.endswith('.png'):
            continue
        
        # Get mesh name (remove .png extension)
        mesh_name = filename.rsplit('.png', 1)[0]
        # print(f"[DEBUG] mesh_name: {mesh_name}")

        input_img_path = f"{img_dir}/{filename}"
        shape_fpath = f"{shape_dir}/{mesh_name}_shape.obj"
        Path(shape_fpath).parent.mkdir(parents=True, exist_ok=True)
        texture_fpath = f"{texture_dir}/{mesh_name}_mesh.glb"
        Path(texture_fpath).parent.mkdir(parents=True, exist_ok=True)

        # if the mesh already exists, continue
        if Path(texture_fpath).exists():
            print(f"\nSkipping: {filename}")
            print(f"  Existing textured mesh found: {texture_fpath}")
            continue
        
        print(f"\nProcessing: {filename}")
        print(f"  Input: {input_img_path}")
        print(f"  Shape output: {shape_fpath}")
        print(f"  Textured output: {texture_fpath}")

        # get the canonical mesh for rescaling
        obj_name = mesh_name.split("/")[0]
        canonical_glb_path = f"{cfg.s8_pose.out_dir}/canonical_mesh/{obj_name}.glb"
        can_scene = trimesh.load(canonical_glb_path, force="scene")

        # Process based on mode
        if generation_mode == GenerationMode.SHAPE_TEXTURE_SINGLE_MODEL:
            scale = generate_rescaled_mesh(
                texture_fpath,
                lambda staging_fpath: mesh_generator.generate_mesh(
                    out_fpath=staging_fpath,
                    shape_image_path=input_img_path,
                    texture_image_path=input_img_path,
                    visualize=cfg.visualize,
                    **mesh_gen_kwargs,
                ),
                can_scene,
            )
            print(f"[DEBUG] In {generation_mode}, the rescaling factor is: {scale}.")
        else:
            if generation_mode in {GenerationMode.SHAPE_TEXTURE_SEPARATE_MODELS, GenerationMode.SHAPE_ONLY}:
                scale = generate_rescaled_mesh(
                    shape_fpath,
                    lambda staging_fpath: shape_generator.generate_shape(
                        image_path=input_img_path,
                        out_fpath=staging_fpath,
                        visualize=cfg.visualize,
                        **shape_gen_kwargs,
                    ),
                    can_scene,
                )
                print(f"[DEBUG] In {generation_mode}, the rescaling factor is: {scale}.")
            if generation_mode in {GenerationMode.SHAPE_TEXTURE_SEPARATE_MODELS, GenerationMode.TEXTURE_ONLY}:
                scale = generate_rescaled_mesh(
                    texture_fpath,
                    lambda staging_fpath: texture_generator.generate_texture(
                        shape_fpath=shape_fpath,
                        image_path=input_img_path,
                        out_fpath=staging_fpath,
                        visualize=cfg.visualize,
                        **texture_gen_kwargs,
                    ),
                    can_scene,
                )
                print(f"[DEBUG] In {generation_mode}, the rescaling factor is: {scale}.")

        # Use the already-resolved local, not cfg.low_vram: the top-level key exists only in
        # real2sim_cfg.yaml, so task configs that do not inherit it raised ConfigAttributeError
        # here after the first cousin had already been generated.
        if low_vram:
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
