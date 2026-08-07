# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from simfoundry

Requires installing:

- simfoundry, see the main README
"""
import numpy as np
import os
from pathlib import Path
import json
from simfoundry.utils.asset_conversion_utils import import_custom_object
from simfoundry.models.vlm import Gemini
from simfoundry.pipeline.front_canonicalization import canonicalize_front
import hydra
from simfoundry import CFG_DIR
from omegaconf import OmegaConf
from simfoundry.utils.processing_utils import dump_json
import trimesh
import random
import string
import logging
import os
import re

from simfoundry.utils.processing_utils import extract_numbers_from_str
from simfoundry.utils.prompt_utils import parse_json_response, prompt_object_mass_friction

# see https://github.com/facebookresearch/hydra/issues/2949#issue-2516892001
if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

logger = logging.getLogger(__name__)
DEFAULT_MASS_KG = 1.0
DEFAULT_FRICTION = 0.5
### At the start of every script, we cd into the scripts/config directory
# scripts_dir = os.path.dirname(os.path.abspath(__file__))
# cfg_dir = os.path.join(scripts_dir, "..", "..", "cfg")
# os.chdir(cfg_dir)

def cfg_path(path_value, *parts):
    path = Path(str(path_value))
    if not path.is_absolute():
        path = (Path(CFG_DIR) / path).resolve()
    return path.joinpath(*parts)


def generate_seeded_random_letters(seed_value, length=6):
    """
    Generates a string of random lowercase letters using a specified seed.

    Args:
        seed_value: The seed to initialize the random number generator.
        length: The desired length of the random string (default is 6).

    Returns:
        A string of random lowercase letters.
    """
    random.seed(seed_value)  # Set the seed for reproducibility
    letters = string.ascii_lowercase  # All lowercase letters
    random_string = ''.join(random.choices(letters, k=length))
    return random_string


# TODO: Include VLM-based annotations for mass and friction, based on (a) image, (b) bounding box (extents), and (c) category name / description

@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    scene_dir = cfg_path(cfg.s5_scene.out_dir)
    img_dir = cfg_path(cfg.prompt_cousin_structured.out_dir)
    pose_dir = cfg_path(cfg.s8_pose.out_dir)
    # Must track the backend that actually wrote the meshes; 3_generate_cousin_meshes.py names
    # this directory after `cousin_generation.texture_model`, not a fixed backend.
    mesh_dir = cfg_path(cfg.cousin_generation.out_dir, "textured_mesh", cfg.cousin_generation.texture_model)
    out_dir = cfg_path(cfg.sim.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("Starting objects simulation ready pipeline...")
    logger.info(f"Output directory: {out_dir}")
    logger.info("="*60)

    # Create VLM for physical property annotation
    vlm = Gemini(
        project=cfg.gcloud_project,
        location="global",
        # model="gemini-2.5-flash",
        # model="gemini-3-pro-preview",
        model="gemini-3.1-pro-preview",
    )

    # Iterate over all meshes, and transform them accordingly
    scene_objects_info = dict()
    transparent_meshes = sorted(Path(mesh_dir).rglob("*transparent_mesh.glb"))
    for mesh_path in transparent_meshes:
        obj_name = mesh_path.parents[1].name
        idx = int(obj_name.split("iter_")[-1])
        print(f"[DEBUG] obj_name: {obj_name}")
        print(f"[DEBUG] mesh_path: {mesh_path}")
        # if not filename.endswith("transparent_mesh.glb"):
        #     continue
        # if "iter" not in filename:
        #     continue
        # img_name = filename.split(".")[0]
        img_name = mesh_path.stem.replace("_transparent_mesh", "") + ".png"
        # idx = int(img_name.split("iter")[-1])

        var_type = mesh_path.parent.name
        print(f"[DEBUG] var_type: {var_type}")
        img_fpath = str(img_dir / obj_name / var_type / img_name)
        print(f"[DEBUG] img_fpath: {img_fpath}")
        # mesh_path = f"{mesh_dir}/{mesh_path}"
        obj_tf_info_fpath = pose_dir / "info" / f"{obj_name}.json"
        with open(obj_tf_info_fpath, "r") as f:
            obj_info = json.load(f)
        # Grab category info as well
        obj_cat_info_fpath = scene_dir / "obj_cat_list" / f"iter_{idx}.json"
        with open(obj_cat_info_fpath, "r") as f:
            cat_info = json.load(f)
        obj_phrase = cat_info["removed_obj_phrase"]
        obj_category = (
            obj_phrase.replace(" ", "_")
                    .replace("-", "_")
                    .replace("/", "_")
                    .replace("'", "_")
                    .replace("&", "_")
                    .lower()
            + "_"
            + mesh_path.stem.replace("_transparent_mesh", "")
        )
        seed_string = f"{cfg.scene_name}_{obj_category}_{idx}"
        obj_model = generate_seeded_random_letters(seed_value=seed_string, length=6)

        # Skip this object if destination already contains generated files.
        dst_urdf_dir = Path(out_dir) / "objects" / obj_category / obj_model / "urdf"
        dst_urdf_fpath = dst_urdf_dir / f"{obj_model}.urdf"
        if dst_urdf_fpath.exists() or (dst_urdf_dir.exists() and any(dst_urdf_dir.iterdir())):
            logger.info(f"Skipping existing object output: {dst_urdf_dir}")
            continue

        # Query VLM to annotate mass / friction
        scale = obj_info["z_up"]["scale"]
        tm = trimesh.load(mesh_path, process=True)
        tm.apply_scale(scale)
        # Yaw-canonicalize the cousin front so it matches the original's convention.
        if cfg.sim.get("canonicalize_front", True):
            front_render_dir = str(out_dir / "front_views" / obj_name / var_type / mesh_path.stem)
            front_rot, front_info = canonicalize_front(
                tm,
                render_dir=front_render_dir,
                vlm=vlm,
                photo_path=img_fpath,
                category=obj_phrase,
            )
            if front_rot is not None:
                _front_tf = np.eye(4)
                _front_tf[:3, :3] = front_rot
                tm.apply_transform(_front_tf)
            with open(Path(front_render_dir) / "orientation.json", "w") as f:
                json.dump(front_info, f, indent=4)
            logger.info(f"Front canonicalization for {obj_category}: {front_info['status']} "
                        f"(yaw={front_info['applied_yaw_deg']:.0f} deg)")
        obb_tf, obb_extent = trimesh.bounds.oriented_bounds(tm)

        result = vlm(
            prompt=prompt_object_mass_friction(obj_phrase=obj_phrase, bounding_box_cm=obb_extent * 100, volume_cm=tm.volume * (100 ** 3)),
            image_paths=img_fpath,
            temperature=0,
            top_p=0,
            seed=0,
            print_results=True,
        )
        result_text = vlm.get_result_text(result=result)
        try:
            result_json = parse_json_response(result_text)
            mass = float(result_json["mass"])
            friction = float(result_json["friction"])
        except Exception:
            result_str = ""
            include_line = False
            for line in result_text.split("\n"):
                if "answer" in line.lower():
                    include_line = True
                if include_line:
                    result_str += line
            if not result_str:
                result_str = result_text
            numbers = extract_numbers_from_str(result_str)
            if len(numbers) >= 2:
                mass, friction = float(numbers[0]), float(numbers[1])
                if len(numbers) > 2:
                    logger.warning(
                        f"Parsed {len(numbers)} numbers for {obj_name}. "
                        f"Using first two as mass/friction: {numbers[:2]}. "
                        f"Raw answer text: {result_str}"
                    )
            elif len(numbers) == 1:
                mass = float(numbers[0])
                friction = DEFAULT_FRICTION
                logger.warning(
                    f"Parsed only one number for {obj_name}. "
                    f"Using mass={mass}, default friction={friction}. "
                    f"Raw answer text: {result_str}"
                )
            else:
                mass = DEFAULT_MASS_KG
                friction = DEFAULT_FRICTION
                logger.warning(
                    f"Failed to parse mass/friction for {obj_name}. "
                    f"Using defaults mass={mass}, friction={friction}. "
                    f"Raw answer text: {result_str}"
                )

        print(f"[DEBUG] Before import_custom_object...")
        # Convert to URDF representation
        import_custom_object(
            asset_path=str(mesh_path),
            category=obj_category,
            model=obj_model,
            dataset_root=out_dir,
            collision_method="coacd",
            # collision_method="convex",
            hull_count=cfg.sim.hull_count,
            up_axis="y",
            scale=scale, #np.mean(obj_info["z_up"]["scale"]), # TODO: Cannot handle non-uniform scale currently obj_info["z_up"]["scale"],
            check_scale=False,
            rescale=False,
            overwrite=True,
            n_submesh=cfg.sim.n_submesh,
            mass=mass,
        )
        print(f"[DEBUG] After import_custom_object...")

        # Add to object list
        scene_objects_info[obj_category] = {
            "category": obj_category,
            "model": obj_model,
            "name": img_name,
            "friction": friction,
        }

    # Store scene object info
    scene_objects_info_fpath = out_dir / "scene_objects_info.json"
    merged_scene_objects_info = dict()
    if Path(scene_objects_info_fpath).exists():
        with open(scene_objects_info_fpath, "r") as f:
            merged_scene_objects_info = json.load(f)
    merged_scene_objects_info.update(scene_objects_info)
    with open(scene_objects_info_fpath, "w+") as f:
        json.dump(merged_scene_objects_info, f, indent=4)

    logger.info("="*60)
    logger.info("Objects simulation ready complete!")
    logger.info("="*60)
    
    end_stage(cfg, success=True)


def end_stage(cfg, success=False, additional_info: dict = None):
    """
    Function that is run at the end of every stage to save the stage config and additional info.

    # TODO: should this be a general function that can be used for all stages? or should we have a separate function for each stage?
    """
    save_dir = cfg_path(cfg.sim.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stage_cfg = OmegaConf.to_object(cfg.sim)
    stage_cfg['success'] = success
    if additional_info is not None:
        stage_cfg.update(additional_info)
    dump_json(stage_cfg, f"{save_dir}/stage_info.json")


if __name__ == "__main__":
    main()
