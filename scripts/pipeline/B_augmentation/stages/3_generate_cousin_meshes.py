# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from env depending on which mesh generator is used!

Requires installing (depending on generator used):

- Hunyuan2.1, see https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
- TRELLIS, see https://github.com/microsoft/TRELLIS
"""
from simfoundry.models.mesh_generator import ShapeGenerator, MeshGenerator, TextureGenerator, Hunyuan, Direct3D, Trellis
from pathlib import Path
import os
import inspect
from enum import IntEnum
import hydra
from omegaconf import OmegaConf

from simfoundry.utils.python_utils import assert_valid_key
from simfoundry import CFG_DIR, REPO_DIR
from simfoundry.pipeline.stage_utils import bootstrap_hydra_workdir

import torch
import trimesh
import json

# see https://github.com/facebookresearch/hydra/issues/2949#issue-2516892001
if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

bootstrap_hydra_workdir(__file__)
REPO_ROOT = Path(REPO_DIR)

MESH_GENERATORS = {
    "direct3d": Direct3D,
    "hunyuan": Hunyuan,
    "trellis": Trellis,
}

hunyuan_repo_path = str(REPO_ROOT / "deps" / "Hunyuan3D-2.1")

Hunyuan.set_repo_path(repo_path=hunyuan_repo_path)


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


def make_generator(generator_cls, *, low_vram, **kwargs):
    """Instantiate a mesh generator while passing low_vram only when supported."""
    if "low_vram" in inspect.signature(generator_cls).parameters:
        kwargs["low_vram"] = low_vram
    return generator_cls(**kwargs)


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
    assert_valid_key(key=shape_generator_name, valid_keys=MESH_GENERATORS, name="shape_generator")
    shape_dir = f"{out_dir}/shape/{shape_generator_name}"
    Path(shape_dir).mkdir(parents=True, exist_ok=True)
    if generate_shape:
        shape_generator_cls = MESH_GENERATORS[shape_generator_name]
        assert issubclass(shape_generator_cls, ShapeGenerator)
    texture_generator_name = cfg.cousin_generation.texture_model
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
            mesh_generator.generate_mesh(
                out_fpath=texture_fpath,
                shape_image_path=input_img_path,
                texture_image_path=input_img_path,
                visualize=cfg.visualize,
            )
            gen_scene = trimesh.load(texture_fpath, force="scene")
            gen_bounds, gen_extents = get_aabb(gen_scene)
            can_bounds, can_extents = get_aabb(can_scene)
            gen_size = gen_extents.max()
            can_size = can_extents.max()
            scale = can_size / gen_size
            gen_scene.apply_scale(scale)
            gen_scene.export(texture_fpath)
            print(f"[DEBUG] In {generation_mode}, the rescaling factor is: {scale}.")
        else:
            if generation_mode in {GenerationMode.SHAPE_TEXTURE_SEPARATE_MODELS, GenerationMode.SHAPE_ONLY}:
                shape_generator.generate_shape(
                    image_path=input_img_path,
                    out_fpath=shape_fpath,
                    visualize=cfg.visualize,
                )
                gen_scene = trimesh.load(shape_fpath, force="scene")
                gen_bounds, gen_extents = get_aabb(gen_scene)
                can_bounds, can_extents = get_aabb(can_scene)
                gen_size = gen_extents.max()
                can_size = can_extents.max()
                scale = can_size / gen_size
                gen_scene.apply_scale(scale)
                gen_scene.export(shape_fpath)
                print(f"[DEBUG] In {generation_mode}, the rescaling factor is: {scale}.")
            if generation_mode in {GenerationMode.SHAPE_TEXTURE_SEPARATE_MODELS, GenerationMode.TEXTURE_ONLY}:
                texture_generator.generate_texture(
                    shape_fpath=shape_fpath,
                    image_path=input_img_path,
                    out_fpath=texture_fpath,
                    visualize=cfg.visualize,
                )
                gen_scene = trimesh.load(texture_fpath, force="scene")
                gen_bounds, gen_extents = get_aabb(gen_scene)
                can_bounds, can_extents = get_aabb(can_scene)
                gen_size = gen_extents.max()
                can_size = can_extents.max()
                scale = can_size / gen_size
                gen_scene.apply_scale(scale)
                gen_scene.export(texture_fpath)
                print(f"[DEBUG] In {generation_mode}, the rescaling factor is: {scale}.")

        if cfg.low_vram:
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
