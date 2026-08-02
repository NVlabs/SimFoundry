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
from simfoundry.utils.asset_conversion_utils import import_custom_object, import_articulated_object
from simfoundry.models.vlm import Gemini
import hydra
import trimesh
import random
import string
import logging
import re
from simfoundry import CFG_DIR
from simfoundry.utils.processing_utils import extract_numbers_from_str
from simfoundry.utils.python_utils import sanitize_path_component
from simfoundry.utils.prompt_utils import prompt_object_mass_friction, prompt_articulated_object_parts_properties, parse_json_response
from simfoundry.pipeline.stage_utils import StageResult, bootstrap_hydra_workdir, finalize_stage

# see https://github.com/facebookresearch/hydra/issues/2949#issue-2516892001
if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

logger = logging.getLogger(__name__)
bootstrap_hydra_workdir(__file__)

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

def load_mesh_as_trimesh(mesh_path: str) -> trimesh.Trimesh:
    """Load a mesh file and ensure it's a Trimesh object (not a Scene)."""
    tm = trimesh.load(mesh_path, process=True)
    if isinstance(tm, trimesh.Scene):
        # Convert Scene to single Trimesh by combining all geometry
        tm = tm.to_geometry()
    return tm


def process_articulated_object(
    obj_name: str,
    urdf_path: str,
    mesh_parts_dir: str,
    image_path: str,
    vlm: Gemini,
    scale: float,
) -> dict:
    """Process an articulated object and return per-part physical properties."""
    
    # Parse URDF to get link names and mesh references
    import xml.etree.ElementTree as ET
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    parts_info = []
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        if link_name == "base":  # Skip virtual base link
            continue
            
        # Find corresponding mesh
        visual = link.find(".//visual/geometry/mesh")
        if visual is not None:
            mesh_file = visual.attrib.get("filename", "")
            mesh_path = os.path.join(mesh_parts_dir, os.path.basename(mesh_file))
            
            if os.path.exists(mesh_path):
                # Load as Trimesh (handle Scene objects from GLB files)
                tm = load_mesh_as_trimesh(mesh_path)
                
                # Apply scale before computing properties
                tm.apply_scale(scale)
                
                # Compute oriented bounding box
                obb_tf, obb_extent = trimesh.bounds.oriented_bounds(tm)
                
                # Volume is now correctly scaled (scales with cube of linear scale)
                volume_m3 = tm.volume
                volume_cm3 = volume_m3 * (100 ** 3)
                
                parts_info.append({
                    "name": link_name,
                    "mesh_path": mesh_path,
                    "bounding_box_cm": obb_extent * 100,
                    "volume_cm": volume_cm3,
                })
    
    # Query VLM for per-part properties
    prompt = prompt_articulated_object_parts_properties(obj_name, parts_info)
    result = vlm(prompt=prompt, image_paths=image_path, temperature=0, top_p=0, seed=0, print_results=True)
    result_json = parse_json_response(vlm.get_result_text(result))
    
    return result_json["parts"]


def link_has_existing_mesh(link, *, urdf_dir: Path, mesh_parts_dir: Path) -> bool:
    for mesh in link.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        if not filename:
            continue
        candidates = [
            urdf_dir / filename,
            mesh_parts_dir / filename,
            mesh_parts_dir / os.path.basename(filename),
        ]
        if any(path.exists() for path in candidates):
            return True
    return False


def resolve_articulation_results_dir(s8b_out_dir: str, scene_name: str, obj_phrase: str) -> str:
    """Locate a stage-8b articulation result directory.

    Stage 8b writes to ``<out_dir>/<sanitized scene>/<sanitized object>/results``. Runs
    produced before the sanitizer was shared used the raw, un-lowercased scene and object
    names, so those layouts are still accepted rather than silently falling back to a rigid
    import. The sanitized path always wins when both exist.
    """
    candidates = [
        (sanitize_path_component(scene_name), sanitize_path_component(obj_phrase)),
        # Legacy layout: scene name verbatim, object name not lowercased.
        (str(scene_name), str(obj_phrase).replace(" ", "_").replace("/", "_")),
    ]
    seen = set()
    for scene_part, obj_part in candidates:
        results_dir = f"{s8b_out_dir}/{scene_part}/{obj_part}/results"
        if results_dir in seen:
            continue
        seen.add(results_dir)
        if os.path.exists(f"{results_dir}/mobility.urdf"):
            return results_dir
    # Nothing on disk: return the canonical path so the caller's warning names it.
    return f"{s8b_out_dir}/{candidates[0][0]}/{candidates[0][1]}/results"


def invalid_articulated_links(urdf_path: str, mesh_parts_dir: str) -> list[str]:
    """Return non-virtual links that have no usable visual/collision mesh."""
    import xml.etree.ElementTree as ET

    urdf = Path(urdf_path)
    mesh_parts = Path(mesh_parts_dir)
    root = ET.parse(urdf_path).getroot()

    invalid = []
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        if link_name == "base":
            continue
        if not link_has_existing_mesh(link, urdf_dir=urdf.parent, mesh_parts_dir=mesh_parts):
            invalid.append(link_name)
    return invalid


def import_rigid_scene_object(
    *,
    cfg,
    vlm: Gemini,
    obj_phrase: str,
    img_fpath: str,
    mesh_path: str,
    obj_category: str,
    obj_model: str,
    img_name: str,
    out_dir: str,
    obb_extent,
    volume_m3: float,
    rigid_scale: float,
) -> dict:
    result = vlm(
        prompt=prompt_object_mass_friction(
            obj_phrase=obj_phrase,
            bounding_box_cm=obb_extent * 100,
            volume_cm=volume_m3 * (100 ** 3),
        ),
        image_paths=img_fpath,
        temperature=0,
        top_p=0,
        seed=0,
        print_results=True,
    )
    result_json = parse_json_response(vlm.get_result_text(result))
    mass, friction = result_json["mass"], result_json["friction"]

    rigid_collision_method = cfg.s10_sim.get("collision_method", "coacd")
    import_custom_object(
        asset_path=mesh_path,
        category=obj_category,
        model=obj_model,
        dataset_root=out_dir,
        collision_method=rigid_collision_method,
        hull_count=cfg.s10_sim.hull_count,
        up_axis="y",
        scale=rigid_scale,
        check_scale=False,
        rescale=False,
        overwrite=True,
        n_submesh=cfg.s10_sim.n_submesh,
        mass=mass,
    )

    return {
        "category": obj_category,
        "model": obj_model,
        "name": img_name,
        "friction": friction,
    }


# TODO: Include VLM-based annotations for mass and friction, based on (a) image, (b) bounding box (extents), and (c) category name / description

def resolve_requested_indices(cfg):
    """Resolve optional per-object filter for partial reruns (e.g. a single object)."""
    raw = cfg.s10_sim.get("object_indices", None)
    if raw is None:
        return None
    if isinstance(raw, int):
        return {raw}
    return {int(v) for v in raw}


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    scene_dir = cfg.s5_scene.out_dir
    img_dir = cfg.s6_upsample.out_dir
    pose_dir = cfg.s8_pose.out_dir
    mesh_dir = pose_dir + "/canonical_mesh"
    vlm_model = cfg.s10_sim.vlm_model
    out_dir = cfg.s10_sim.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("Starting objects simulation ready pipeline...")
    logger.info(f"Output directory: {out_dir}")
    logger.info("="*60)

    # Create VLM for physical property annotation
    vlm = Gemini(
        project=cfg.gcloud_project,
        location="global",
        model=vlm_model,
    )

    # Load articulation classification from step 7b
    articulation_info_path = f"{cfg.s8b_articulate_objects.out_dir}/object_classification.json"
    if os.path.exists(articulation_info_path):
        with open(articulation_info_path) as f:
            articulation_info = json.load(f)
        articulated_objects = set(articulation_info.get("articulated_objects", []))
    else:
        articulated_objects = set()

    scene_objects_info = dict()

    requested_indices = resolve_requested_indices(cfg)
    if requested_indices is not None:
        logger.info(f"Filtering to requested object indices: {sorted(requested_indices)}")

    # Iterate over all meshes, and transform them accordingly
    for filename in sorted(os.listdir(mesh_dir)):
        if not filename.endswith(".glb"):
            continue
        if "iter" not in filename:
            continue
        img_name = filename.split(".")[0]
        idx = int(img_name.split("iter_")[-1])

        if requested_indices is not None and idx not in requested_indices:
            continue

        img_fpath = f"{img_dir}/upsampled/{img_name}.png"
        mesh_path = f"{mesh_dir}/{filename}"
        if cfg.s9_compile.use_interactive_pose:
            obj_tf_info_fpath = f"{pose_dir}/info_interactive{cfg.s9_compile.interactive_suffix}/{img_name}.json"
        else:
            obj_tf_info_fpath = f"{pose_dir}/info/{img_name}.json"
        with open(obj_tf_info_fpath, "r") as f:
            obj_info = json.load(f)
        # Grab category info as well
        obj_cat_info_fpath = f"{scene_dir}/obj_cat_list/iter_{idx}.json"
        with open(obj_cat_info_fpath, "r") as f:
            cat_info = json.load(f)
        obj_phrase = cat_info["removed_obj_phrase"]
        # replace all non alphanumeric characters with an underscore
        obj_category = re.sub(r'[^a-zA-Z0-9]', '_', obj_phrase).lower()
        seed_string = f"{cfg.scene_name}_{obj_category}_{idx}"
        obj_model = generate_seeded_random_letters(seed_value=seed_string, length=6)

        is_articulated = obj_phrase in articulated_objects
        obj_results_dir = resolve_articulation_results_dir(
            cfg.s8b_articulate_objects.out_dir, cfg.scene_name, obj_phrase
        )
        mesh_parts_dir = f"{obj_results_dir}/meshes"
        urdf_path = f"{obj_results_dir}/mobility.urdf"

        if is_articulated:
            if not os.path.exists(urdf_path):
                logger.warning(
                    "Articulated result missing for %s (%s); falling back to rigid import.",
                    obj_phrase,
                    urdf_path,
                )
                is_articulated = False
            else:
                invalid_links = invalid_articulated_links(urdf_path, mesh_parts_dir)
                if invalid_links:
                    logger.warning(
                        "Articulated result for %s has meshless links %s; falling back to rigid import.",
                        obj_phrase,
                        invalid_links,
                    )
                    is_articulated = False

        # Get the full scale: pre_scale_factor × tf_scale
        pre_scale_factor = obj_info.get("pre_scale_factor", 1.0)
        tf_scale = obj_info["z_up"]["scale"]

        # Handle both scalar and array tf_scale
        if isinstance(tf_scale, list):
            tf_scale = np.mean(tf_scale)  # or use np.array(tf_scale) for non-uniform

        # Load mesh - canonical mesh from step 8 already has pre_scale_factor applied
        tm = load_mesh_as_trimesh(mesh_path)
        tm_original_extent = tm.bounding_box.extents
        logger.info(f"[DEBUG] Loaded mesh from {mesh_path}")
        logger.info(f"[DEBUG] Mesh extent before scaling: {tm_original_extent} m ({tm_original_extent*100} cm)")
        
        # Apply appropriate scale based on object type
        if not is_articulated: # processing for solid objects only
            rigid_scale = tf_scale
            logger.info(f"[DEBUG] Rigid object: applying only tf_scale={rigid_scale:.6f} (pre_scale_factor={pre_scale_factor:.6f} already baked in)")
            tm.apply_scale(rigid_scale)
        else:
            # For articulated objects, apply total scale (pre_scale_factor may not be baked in)
            total_scale = pre_scale_factor * tf_scale
            logger.info(f"[DEBUG] Articulated object: applying total_scale={total_scale:.6f} (pre_scale_factor={pre_scale_factor:.6f} * tf_scale={tf_scale:.6f})")
            tm.apply_scale(total_scale)
        
        obb_tf, obb_extent = trimesh.bounds.oriented_bounds(tm)
        logger.info(f"[DEBUG] Mesh extent after scaling: {obb_extent} m ({obb_extent*100} cm)")
        
        # Compute volume - use mesh volume if watertight, otherwise estimate from bounding box
        bbox_volume = np.prod(obb_extent)  # Axis-aligned bounding box volume
        min_reasonable_volume = bbox_volume * 0.1  # At least 10% of bbox volume
        
        if tm.is_volume and tm.volume > min_reasonable_volume:
            volume_m3 = tm.volume
            logger.info(f"[DEBUG] Mesh is watertight, using computed volume: {volume_m3} m³ ({volume_m3 * (100**3):.2f} cm³)")
        else:
            # Fallback: estimate volume from bounding box (assume ellipsoid-like shape)
            volume_m3 = bbox_volume * 0.65  # Approximate as 65% of bbox
            if not tm.is_volume:
                logger.info(f"[DEBUG] Mesh is not watertight (is_volume={tm.is_volume}), estimating from bbox")
            else:
                logger.info(f"[DEBUG] Mesh volume ({tm.volume * (100**3):.2f} cm³) is too small compared to bbox ({bbox_volume * (100**3):.2f} cm³), estimating from bbox")
            logger.info(f"[DEBUG] Bbox volume: {bbox_volume} m³ ({bbox_volume * (100**3):.2f} cm³)")
            logger.info(f"[DEBUG] Estimated volume (65% of bbox): {volume_m3} m³ ({volume_m3 * (100**3):.2f} cm³)")

        # Process rigid vs articulated objects separately
        if not is_articulated: # processing for solid objects only
            scene_objects_info[idx] = import_rigid_scene_object(
                cfg=cfg,
                vlm=vlm,
                obj_phrase=obj_phrase,
                img_fpath=img_fpath,
                mesh_path=mesh_path,
                obj_category=obj_category,
                obj_model=obj_model,
                img_name=img_name,
                out_dir=out_dir,
                obb_extent=obb_extent,
                volume_m3=volume_m3,
                rigid_scale=rigid_scale,
            )

        else: # process articulated objects
            # For articulated objects, mesh parts from step 8b are from the canonical mesh
            # (already pre-scaled by pre_scale_factor), so we only apply tf_scale
            # Get per-part physical properties from VLM
            parts_properties = process_articulated_object(
                obj_phrase, urdf_path, mesh_parts_dir, img_fpath, vlm, tf_scale
            )
            
            # Import articulated object with physical properties
            import_articulated_object(
                urdf_path=urdf_path,
                mesh_parts_dir=mesh_parts_dir,
                parts_properties=parts_properties,
                category=obj_category,
                model=obj_model,
                dataset_root=out_dir,
                scale=tf_scale,
                collision_method=cfg.s10_sim.get("collision_method", "coacd"),
                hull_count=cfg.s10_sim.hull_count,
                # up_axis="z" is default - no rotation needed since mobility.urdf already works correctly
                overwrite=True,
                apply_base_rotation=True,
            )
            
            # Add to object list (use first part's friction as representative)
            representative_friction = parts_properties[0].get("friction", 0.5) if parts_properties else 0.5
            scene_objects_info[idx] = {
                "category": obj_category,
                "model": obj_model,
                "name": img_name,
                "friction": representative_friction,
                "is_articulated": True,
                "parts_properties": parts_properties,
            }


    # Store scene object info. When only a subset of objects was (re)processed,
    # merge into the existing file so we don't drop entries for untouched objects.
    scene_objects_info_fpath = f"{out_dir}/scene_objects_info.json"
    merged_info = {}
    if requested_indices is not None and os.path.exists(scene_objects_info_fpath):
        with open(scene_objects_info_fpath, "r") as f:
            merged_info = json.load(f)
    for k, v in scene_objects_info.items():
        merged_info[str(k)] = v
    with open(scene_objects_info_fpath, "w+") as f:
        json.dump(merged_info, f, indent=4)

    logger.info("="*60)
    logger.info("Objects simulation ready complete!")
    logger.info("="*60)
    
    finalize_stage(
        stage_cfg=cfg.s10_sim,
        out_dir=cfg.s10_sim.out_dir,
        result=StageResult(success=True),
    )


if __name__ == "__main__":
    main()
