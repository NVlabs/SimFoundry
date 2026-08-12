# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from `simfoundry` env

Structured approach to generating digital cousins:
1. Decompose object into functional components based on affordance
2. Generate variation options for each component across 3 dimensions:
   - Geometric dimension
   - Topological dimension  
   - Visual dimension (material/surface)
3. Apply variations through nano-banana

Requires installing:
- simfoundry, see the main README

Don't forget to register API key from Gemini CLI!
"""
from simfoundry.models.vlm import Gemini
from pathlib import Path
from PIL import Image
import json
import os 
import re
import hydra
import logging
from omegaconf import OmegaConf
from rembg import remove, new_session
import tempfile
from simfoundry import CFG_DIR

# Set up logger
logger = logging.getLogger(__name__)

# Initialize rembg session once for reuse
SESSION = new_session("bria-rmbg")

# see https://github.com/facebookresearch/hydra/issues/2949#issue-2516892001
if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()


def cfg_path(path_value, *parts):
    path = Path(str(path_value))
    if not path.is_absolute():
        path = (Path(CFG_DIR) / path).resolve()
    return path.joinpath(*parts)


def _optional_int(value):
    return None if value is None else int(value)


def _optional_iter_set(value):
    if value is None:
        return None
    if isinstance(value, str):
        return {int(item.strip()) for item in value.split(",") if item.strip()}
    return {int(item) for item in value}


def parse_component_list(response_text):
    """Parse numbered component list from Gemini response, starting from '1.'"""
    components = []
    started = False  # Flag to track if we've found the start of the list
    
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Check if this is the start of the numbered list (1. ...)
        if not started:
            if re.match(r'^1[\.\)\-\:]\s*.+$', line):
                started = True
            else:
                # Skip any text before the numbered list starts
                continue
        
        # Once started, parse numbered items
        if started:
            match = re.match(r'^\d+[\.\)\-\:]\s*(.+)$', line)
            if match:
                components.append(match.group(1))
    
    return components


def parse_yes_no_response(response_text):
    """Parse yes/no response from Gemini."""
    text = response_text.strip().lower()
    # Look for yes or no in the response
    if 'yes' in text and 'no' not in text:
        return True
    elif 'no' in text:
        return False
    # Default to False if unclear
    return False


def parse_variation_options(response_text):
    """
    Parse variation options from Gemini response.
    Expected format:
    Component: [component_name]
    Geometry:
    1. variation 1
    2. variation 2
    3. variation 3
    Topology:
    1. variation 1
    ...
    Visual:
    1. variation 1
    ...
    """
    variations = {}
    current_component = None
    current_dimension = None
    
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check for component header
        if line.lower().startswith('component:'):
            current_component = line.split(':', 1)[1].strip()
            variations[current_component] = {
                'geometry': [],
                'topology': [],
                'visual': []
            }
            continue
        
        # Check for dimension headers
        if 'geometry' in line.lower() and ':' in line:
            current_dimension = 'geometry'
            continue
        elif 'topology' in line.lower() and ':' in line or 'topological' in line.lower():
            current_dimension = 'topology'
            continue
        elif 'visual' in line.lower() and ':' in line:
            current_dimension = 'visual'
            continue
        
        # Parse numbered variation
        if current_component and current_dimension:
            match = re.match(r'^\d+[\.\)\-\:]\s*(.+)$', line)
            if match:
                variations[current_component][current_dimension].append(match.group(1))
    
    return variations

def remove_background_with_rembg(in_path, out_path):
    """Remove background with rembg and save as PNG (with transparency)."""
    pil_img = Image.open(in_path)
    output_img = remove(pil_img, session=SESSION)
    out_path = Path(out_path).with_suffix(".png")
    output_img.save(out_path)
    return True

def parse_reasonableness_response(response_text):
    """
    Parse real-world and scene reasonableness from Gemini response.
    Expected format:
    RealWorldReasonable: <yes/no>; RealWorldReason: <text>
    SceneReasonable: <yes/no>; SceneReason: <text>
    """
    text = response_text.strip()

    def _match_bool(label):
        m = re.search(rf"(?i){label}\s*[:\-]\s*(yes|no)\b", text)
        if not m:
            return None
        return m.group(1).strip().lower() == "yes"

    def _match_reason(label):
        m = re.search(rf"(?i){label}\s*[:\-]\s*(.+)", text)
        return m.group(1).strip() if m else None

    real_world_reasonable = _match_bool("RealWorldReasonable")
    scene_reasonable = _match_bool("SceneReasonable")
    real_world_reason = _match_reason("RealWorldReason")
    scene_reason = _match_reason("SceneReason")

    return {
        "real_world_reasonable": real_world_reasonable,
        "scene_reasonable": scene_reasonable,
        "real_world_reason": real_world_reason,
        "scene_reason": scene_reason,
        "raw": text,
    }

@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    img_dir = cfg_path(cfg.s5_scene.out_dir)
    scene_dir = cfg_path(cfg.s1_video.out_dir)
    unsampled_img_dir = cfg_path(cfg.s6_upsample.out_dir, "upsampled")
    out_dir = cfg_path(cfg.prompt_cousin_structured.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get optional reasonableness check flag from config
    check_reasonableness = cfg.prompt_cousin_structured.get('check_reasonableness', False)
    print(f"Reasonableness check: {'ENABLED' if check_reasonableness else 'DISABLED'}")

    # Scene context
    scene_img_path = scene_dir / "frames_all" / "frame_0001.png"
    if scene_img_path.exists():
        print(f"Scene image found: {scene_img_path}")
    else:
        print("Scene image not found. Proceeding without scene context.")
        scene_img_path = None

    # Max retries for reasonableness check
    max_reasonableness_attempts = cfg.prompt_cousin_structured.get(
        "max_reasonableness_attempts",
        cfg.prompt_cousin_structured.get("max_scene_score_attempts", 3),
    )
    print(f"Reasonableness max attempts: {max_reasonableness_attempts}")
    include_iters = _optional_iter_set(cfg.prompt_cousin_structured.get("include_iters"))
    max_objects = _optional_int(cfg.prompt_cousin_structured.get("max_objects"))
    max_components = _optional_int(cfg.prompt_cousin_structured.get("max_components"))
    max_generated_images_per_object = _optional_int(
        cfg.prompt_cousin_structured.get("max_generated_images_per_object")
    )
    print(
        "Prompt limits: "
        f"include_iters={sorted(include_iters) if include_iters is not None else None}, "
        f"max_objects={max_objects}, max_components={max_components}, "
        f"max_generated_images_per_object={max_generated_images_per_object}"
    )

    # Create Gemini clients
    print("Initializing Gemini models...")
    gemini_pro = Gemini(
        project=cfg.gcloud_project,
        location="global",
        model=cfg.prompt_cousin_structured.get("text_model", "gemini-2.5-pro"),
    )
    nano_banana = Gemini(
        project=cfg.gcloud_project,
        location="global",
        model=cfg.prompt_cousin_structured.get("image_model", "gemini-3-pro-image"),
    )

    # Iterate over all upsampled transparent images from step 6
    processed_objects = 0
    for filename in sorted(os.listdir(unsampled_img_dir)):

        if not filename.endswith('_transparent.png'):
            continue
        
        # Get mesh name (remove .png extension)
        img_name = filename.rsplit('_transparent.png', 1)[0]  # masked_object_iterX
        idx = int(img_name.split("iter_")[-1]) 

        if include_iters is not None and idx not in include_iters:
            continue

        # read json file
        obj_cat_info_fpath = img_dir / "obj_cat_list" / f"iter_{idx}.json"
        with open(obj_cat_info_fpath) as f:
            obj_cat_info = json.load(f)
        # Skip any invalid ones
        if not obj_cat_info["is_valid_removed_obj"]:
            continue
        if max_objects is not None and processed_objects >= max_objects:
            print(f"Reached max_objects={max_objects}; stopping Stage 1 object loop.")
            break
        processed_objects += 1
        obj_phrase = obj_cat_info["removed_obj_phrase"]

        # TODO: prompt the object phrase to the Gemini model to remove color description
        color_removal_prompt = f"""
        Given an object description, extract the canonical object name.

        Rules:
        - Remove all color descriptions.
        - Remove material, texture, size, and state descriptors.
        - Keep the full object name if it is a multi-word noun.
        - Do NOT invent or generalize the object category.
        - Return ONLY the object name, with no explanation and no punctuation.

        Object description: "{obj_phrase}"
        """

        result = gemini_pro(
            prompt=color_removal_prompt,
            temperature=0,
            top_p=0,
            seed=cfg.seed,
        )

        canonical_obj_name = gemini_pro.get_result_text(result=result)
        canonical_obj_name = canonical_obj_name.strip().lower()

        # Sanity check
        if not canonical_obj_name or len(canonical_obj_name.split()) > 5:
            logger.warning(
                f"Invalid canonical object name '{canonical_obj_name}', "
                f"fallback to original phrase '{obj_phrase}'"
            )
            canonical_obj_name = obj_phrase

        logger.info(f"Processing object (canonical): {canonical_obj_name}")

        print(f"[DEBUG] canonical_obj_name: {canonical_obj_name}")

        input_img_path = str(unsampled_img_dir / filename)
        print(f"\n{'='*80}")
        print(f"Processing {input_img_path}")
        print(f"{'='*80}\n")

        # ========================================================================
        # STEP 1: Decompose object into functional components
        # ========================================================================
        print("STEP 1: Decomposing object into functional components...")
        component_prompt = (
            "With the given object image, decompose the object into functional components based on grasp affordance, list the component name only in number list, "
            "if there is a main body, treat its edge or bottom as a same component inside main body. Make sure to only generate the list of component names, don't generate any other additional text."
        )
        
        component_result = gemini_pro(
            prompt=component_prompt,
            image_paths=input_img_path,
            temperature=0,
            top_p=0,
            seed=cfg.seed,
            print_results=cfg.visualize,
        )
        print("RAW RESULT:", component_result)

        if component_result is None:
            print(f"Failed to decompose components for {input_img_path}")
            continue

        component_text = gemini_pro.get_result_text(result=component_result)
        if not component_text.strip():
            print("Empty Gemini response, skipping.")
            continue
        components = parse_component_list(component_text)
        if max_components is not None:
            components = components[:max_components]
        
        print(f"Found {len(components)} components:")
        for i, comp in enumerate(components, 1):
            print(f"  {i}. {comp}")
        
        # Save component list
        component_output_path = out_dir / img_name / "components.txt"
        os.makedirs(os.path.dirname(component_output_path), exist_ok=True)
        with open(component_output_path, 'w') as f:
            f.write(f"Image: {input_img_path}\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"Functional Components:\n")
            f.write(f"{'='*80}\n\n")
            f.write(component_text)
        print(f"Saved components to {component_output_path}\n")

        # ========================================================================
        # STEP 2: Generate variation options for each component
        # ========================================================================
        print("STEP 2: Generating variation options for each component...")
        
        all_variations = {}
        
        # Get variation counts from config
        num_geometry = cfg.prompt_cousin_structured.num_geometry_variation
        num_topology = cfg.prompt_cousin_structured.num_topology_variation
        num_visual = cfg.prompt_cousin_structured.num_visual_variation
        
        print(f"  Variation counts: Geometry={num_geometry}, Topology={num_topology}, Visual={num_visual}")
        
        for i, component in enumerate(components, 1):
            print(f"\n  Processing component {i}/{len(components)}: {component}")
            
            # Check if topology variation is reasonable for this component
            print(f"    Checking topology feasibility...")
            topology_check_prompt = (
                f"For the {component} of the object shown, is it reasonable to make variation on topology "
                f"such that the modified object still looks realistic and is reasonable enough that we would "
                f"see this object in our daily task? Answer with simply yes or no. If it is likely not to be "
                f"normal, make it a no."
            )
            
            topology_check_result = gemini_pro(
                prompt=topology_check_prompt,
                image_paths=input_img_path,
                temperature=0,
                top_p=0,
                seed=cfg.seed,
                print_results=cfg.visualize,
            )
            
            include_topology = False
            if topology_check_result is not None:
                topology_check_text = gemini_pro.get_result_text(result=topology_check_result)
                include_topology = parse_yes_no_response(topology_check_text)
                print(f"    Topology variations: {'YES' if include_topology else 'NO'}")
            else:
                print(f"    Failed to check topology feasibility, defaulting to NO")
            
            # Build dynamic prompt with configurable variation counts
            geometry_list = "\n".join([f"{j+1}. [variation description]" for j in range(num_geometry)])
            topology_list = "\n".join([f"{j+1}. [variation description]" for j in range(num_topology)])
            visual_list = "\n".join([f"{j+1}. [variation description]" for j in range(num_visual)])
            
            # Build variation prompt based on topology feasibility
            if include_topology:
                variation_prompt = (
                    f"For the component '{component}', list {num_geometry} ways to generate variation "
                    f"through Geometry dimension, {num_topology} ways through Topological dimension, "
                    f"and {num_visual} ways through Visual dimension:\n"
                    f"1. Geometry dimension\n"
                    f"2. Topological dimension\n"
                    f"3. Visual dimension\n\n"
                    f"Make sure to be determined in the description, prevent using words like or, such as, like, or similar to in the description\n"
                    f"Make sure to still maintain the object to have a realistic and reasonable look like something you would see in the daily life\n"
                    f"For example, a red banana and a banana with a hole in it is a bad variation, because we rarely see those in daily life.\n"
                    f"But a banana with green skin is a good varaition since the unripe banana is usually green.\n"
                    f"Format your response as:\n"
                    f"Component: {component}\n"
                    f"Geometry:\n"
                    f"{geometry_list}\n"
                    f"Topology:\n"
                    f"{topology_list}\n"
                    f"Visual:\n"
                    f"{visual_list}"
                )
            else:
                variation_prompt = (
                    f"For the component '{component}', list {num_geometry} ways to generate variation "
                    f"through Geometry dimension and {num_visual} ways through Visual dimension:\n"
                    f"1. Geometry dimension\n"
                    f"2. Visual dimension\n\n"
                    f"Make sure to be determined in the description, prevent using words like or, such as, like, or similar to in the description\n"
                    f"Make sure to still maintain the object to have a realistic and reasonable look like something you would see in the daily life\n"
                    f"For example, a red banana and a banana with a hole in it is a bad variation, because we rarely see those in daily life.\n"
                    f"But a banana with green skin is a good varaition since the unripe banana is usually green, and a banana with dark brown speckles is also a good variation since it looks like a ripe banana\n"
                    f"Format your response as:\n"
                    f"Component: {component}\n"
                    f"Geometry:\n"
                    f"{geometry_list}\n"
                    f"Visual:\n"
                    f"{visual_list}"
                )
            
            variation_result = gemini_pro(
                prompt=variation_prompt,
                image_paths=input_img_path,
                temperature=0.7,
                top_p=0.95,
                seed=cfg.seed,
                print_results=cfg.visualize,
            )

            if variation_result is None:
                print(f"    Failed to generate variations for component: {component}")
                continue

            variation_text = gemini_pro.get_result_text(result=variation_result)
            component_variations = parse_variation_options(variation_text)
            
            # Store variations
            if component_variations:
                all_variations.update(component_variations)
                for comp_name, dims in component_variations.items():
                    if len(dims['topology']) > 0:
                        print(f"    Generated {len(dims['geometry'])} geometry, "
                              f"{len(dims['topology'])} topology, "
                              f"{len(dims['visual'])} visual variations")
                    else:
                        print(f"    Generated {len(dims['geometry'])} geometry, "
                              f"{len(dims['visual'])} visual variations (topology skipped)")
        
        # Save all variations to JSON and text
        variations_json_path = out_dir / img_name / "variations.json"
        with open(variations_json_path, 'w') as f:
            json.dump(all_variations, f, indent=2)
        print(f"\nSaved variations to {variations_json_path}")
        
        variations_txt_path = out_dir / img_name / "variations.txt"
        with open(variations_txt_path, 'w') as f:
            f.write(f"Image: {input_img_path}\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"Variation Options by Component:\n")
            f.write(f"{'='*80}\n\n")
            for comp_name, dims in all_variations.items():
                f.write(f"\nComponent: {comp_name}\n")
                f.write(f"{'-'*60}\n")
                f.write(f"Geometry Variations:\n")
                for j, var in enumerate(dims['geometry'], 1):
                    f.write(f"  {j}. {var}\n")
                if len(dims['topology']) > 0:
                    f.write(f"\nTopology Variations:\n")
                    for j, var in enumerate(dims['topology'], 1):
                        f.write(f"  {j}. {var}\n")
                else:
                    f.write(f"\nTopology Variations: (skipped - not feasible for realistic object)\n")
                f.write(f"\nVisual Variations:\n")
                for j, var in enumerate(dims['visual'], 1):
                    f.write(f"  {j}. {var}\n")
                f.write(f"\n")
        print(f"Saved variations to {variations_txt_path}\n")

        # ========================================================================
        # STEP 3: Generate modified images using nano-banana
        # ========================================================================
        print("STEP 3: Generating modified images with variations...")
        
        generated_images = []
        variation_metadata = []
        attempted_variation_count = 0
        rejected_pool = {
            "geometry": [],
            "topology": [],
            "visual": [],
        }
        
        for comp_name, dims in all_variations.items():
            for dimension_name in ['geometry', 'topology', 'visual']:
                variations_list = dims[dimension_name]
                if (
                    max_generated_images_per_object is not None
                    and attempted_variation_count >= max_generated_images_per_object
                ):
                    break
                
                # Skip if no variations for this dimension
                if len(variations_list) == 0:
                    if dimension_name == 'topology':
                        print(f"\n  Skipping topology variations for {comp_name} (not feasible)")
                    continue
                
                for var_idx, variation_desc in enumerate(variations_list, 1):
                    if (
                        max_generated_images_per_object is not None
                        and attempted_variation_count >= max_generated_images_per_object
                    ):
                        break
                    attempted_variation_count += 1
                    print(f"\n  {'='*70}")
                    print(f"  Component: {comp_name}")
                    print(f"  Dimension: {dimension_name}")
                    print(f"  Variation {var_idx}: {variation_desc}")
                    print(f"  {'='*70}")
                    comp_safe = re.sub(r'[^a-zA-Z0-9_-]', '_', comp_name)
                    
                    # Create image generation prompt
                    image_gen_prompt = (
                        f"Modify the following component of the object: '{comp_name}'. \n"
                        f"Make sure to generate a realistic modified version, make sure it looks normal and reasonable like something you would usually see in the daily life. \n"
                        f"Do not generating variations that are weird or unusual, such as a banana with a hole in it or a banana with a red skin. \n"
                        f"Apply this {dimension_name} variation: {variation_desc}. \n"
                        f"Keep all other components unchanged. \n"
                        f"Generate the image with transparent background. \n"
                        f"If a scene image is provided, make sure the modified object style fits the scene context. \n"
                    )

                    accepted_image = None
                    accepted_reason = None
                    last_rejected_reason = None
                    last_rejected_judgment_log = None
                    reasonableness_attempts_log = []

                    # attempting generate cousins
                    for attempt_idx in range(1, max_reasonableness_attempts + 1):
                        image_paths = [input_img_path]
                        if scene_img_path:
                            image_paths.append(str(scene_img_path))

                        image_result = nano_banana(
                            prompt=image_gen_prompt,
                            image_paths=image_paths,
                            temperature=0.7,
                            top_p=0.95,
                            seed=cfg.seed,
                            print_results=cfg.visualize,
                        )

                        if image_result is None:
                            print(f"    ✗ Failed to generate image (attempt {attempt_idx})")
                            continue

                        # Extract generated image(s)
                        try:
                            images = nano_banana.get_result_images(result=image_result)
                        except (TypeError, AttributeError) as e:
                            print(f"    ✗ Error extracting images (attempt {attempt_idx}): {e}")
                            print(f"    Possible safety filter or content policy block")
                            continue 
                        
                        if not images:
                            print(f"    ✗ No images returned (attempt {attempt_idx})")
                            continue

                        candidate_img = images[0]

                        # If reasonableness checking is disabled or no scene image exists, accept immediately.
                        if not check_reasonableness or not scene_img_path:
                            accepted_image = candidate_img
                            accepted_reason = {
                                "skipped": True,
                                "check_reasonableness": bool(check_reasonableness),
                                "scene_image": str(scene_img_path) if scene_img_path else None,
                            }
                            break

                        # Check reasonableness in real world and scene
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                            candidate_img.save(tmp.name)
                            reasonableness_prompt = (
                                "You will see a scene image and a candidate object image. "
                                "Judge whether the candidate object is reasonable in the real world and also reasonable in the given scene. "
                                "Be strict and practical: reject unusual, implausible, or scene-inconsistent objects. "
                                "Return format:\n"
                                "RealWorldReasonable: <yes/no>; RealWorldReason: <short reason>\n"
                                "SceneReasonable: <yes/no>; SceneReason: <short reason>\n"
                            )
                            reasonableness_result = gemini_pro(
                                prompt=reasonableness_prompt,
                                image_paths=[str(scene_img_path), tmp.name],  # pass in scene image and candidate image at the same time
                                temperature=0,
                                top_p=0,
                                seed=cfg.seed,
                                print_results=cfg.visualize,
                            )
                            if reasonableness_result is None:
                                print(f"    ✗ Failed to check reasonableness (attempt {attempt_idx})")
                                continue
                            reasonableness_text = gemini_pro.get_result_text(result=reasonableness_result)
                            reasonableness_data = parse_reasonableness_response(reasonableness_text)

                        if reasonableness_data["real_world_reasonable"] is None or reasonableness_data["scene_reasonable"] is None:
                            print(f"    ✗ Invalid reasonableness response: '{reasonableness_text}' (attempt {attempt_idx})")
                            continue

                        is_reasonable = reasonableness_data["real_world_reasonable"] and reasonableness_data["scene_reasonable"]
                        reasonableness_attempts_log.append({
                            "attempt": attempt_idx,
                            "real_world_reasonable": reasonableness_data["real_world_reasonable"],
                            "scene_reasonable": reasonableness_data["scene_reasonable"],
                            "is_reasonable": is_reasonable,
                            "real_world_reason": reasonableness_data["real_world_reason"],
                            "scene_reason": reasonableness_data["scene_reason"],
                            "main_dimension": dimension_name,
                            "raw": reasonableness_data["raw"],
                        })
                        print(f"    Reasonableness: {'YES' if is_reasonable else 'NO'} (attempt {attempt_idx})")
                        if is_reasonable:
                            accepted_image = candidate_img
                            accepted_reason = {
                                "real_world_reasonable": reasonableness_data["real_world_reasonable"],
                                "scene_reasonable": reasonableness_data["scene_reasonable"],
                                "real_world_reason": reasonableness_data["real_world_reason"],
                                "scene_reason": reasonableness_data["scene_reason"],
                            }
                            break
                        last_rejected_reason = {
                            "real_world_reasonable": reasonableness_data["real_world_reasonable"],
                            "scene_reasonable": reasonableness_data["scene_reasonable"],
                            "real_world_reason": reasonableness_data["real_world_reason"],
                            "scene_reason": reasonableness_data["scene_reason"],
                        }
                        last_rejected_judgment_log = reasonableness_attempts_log[-1]

                        # store every failed attempt for potential fallback
                        rejected_pool[dimension_name].append({
                            "image": candidate_img,
                            "metadata": {
                                "component": comp_name,
                                "dimension": dimension_name,
                                "variation_index": var_idx,
                                "attempt": attempt_idx,
                                "description": variation_desc,
                                "prompt": image_gen_prompt,
                                "reasonableness_judgment": {
                                    "is_reasonable": False,
                                    "reason": last_rejected_reason,
                                },
                                "scene_image": str(scene_img_path) if scene_img_path else None,
                                "reasonableness_attempts": reasonableness_attempts_log,
                                "low_style_fit": True,
                            },
                            "judgment_log": last_rejected_judgment_log,
                            "comp_safe": comp_safe,
                        })
                    
                    if accepted_image is not None:
                        generated_images.append(accepted_image)
                        variation_metadata.append({
                            'component': comp_name,
                            'dimension': dimension_name,
                            'variation_index': var_idx,
                            'description': variation_desc,
                            'prompt': image_gen_prompt,
                            'reasonableness_judgment': {
                                'is_reasonable': True,
                                'reason': accepted_reason,
                            },
                            'scene_image': str(scene_img_path) if scene_img_path else None,
                            'reasonableness_attempts': reasonableness_attempts_log,
                            'low_style_fit': False,
                        })
                        print(f"    ✓ Successfully generated variation")
                    else:
                        print(f"    ✗ Rejected after {max_reasonableness_attempts} attempts (not reasonable)")
            if (
                max_generated_images_per_object is not None
                and attempted_variation_count >= max_generated_images_per_object
            ):
                break

        # Enforce Top-K fallback per variation category
        min_keep_per_dim = cfg.prompt_cousin_structured.min_keep_per_dim
        for dim in ["geometry", "topology", "visual"]:
            if (
                max_generated_images_per_object is not None
                and len(generated_images) >= max_generated_images_per_object
            ):
                break
            kept_count = sum(1 for m in variation_metadata if m["dimension"] == dim)
            needed = max(0, min_keep_per_dim - kept_count)
            if max_generated_images_per_object is not None:
                needed = min(needed, max_generated_images_per_object - len(generated_images))
            pool = rejected_pool[dim]
            for pick in pool[:needed]:
                generated_images.append(pick["image"])
                variation_metadata.append(pick["metadata"])
                print(f"    [Top-K Fallback] Kept rejected {dim} variation to satisfy min_keep_per_dim")
            # Save all remaining rejected (not selected)
            for pick in pool[needed:]:
                rejected_dir = out_dir / img_name / "rejected" / dim
                os.makedirs(rejected_dir, exist_ok=True)
                rejected_filename = (
                    f"{rejected_dir}/"
                    f"cousin_rejected_{pick['comp_safe']}_attempt_{pick['metadata']['attempt']}.png"
                )
                pick["image"].save(rejected_filename)
                rejected_meta_path = rejected_filename.replace(".png", ".json")
                with open(rejected_meta_path, "w") as f:
                    json.dump(pick["metadata"], f, indent=2)
                print(f"    Saved rejected image: {rejected_filename}")
                print(f"    Saved rejected metadata: {rejected_meta_path}")

        print(f"\n{'='*80}")
        print(f"Total generated images: {len(generated_images)}")
        print(f"{'='*80}\n")
        
        # Save each generated image with descriptive naming
        for i, (img, metadata) in enumerate(zip(generated_images, variation_metadata)):
            comp_safe = re.sub(r'[^a-zA-Z0-9_-]', '_', metadata['component'])
            output_img_filename = (
                f"{img_name}/"
                f"{metadata['dimension']}/"
                f"cousin_{i+1:03d}_v{metadata['variation_index']}.png"
            )
            output_img_path = str(out_dir / output_img_filename)
            os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
            img.save(output_img_path)
            # Add filename to metadata
            metadata['filename'] = output_img_filename
            print(f"Saved: {output_img_path}")

            # prune background with rembg
            trans_output_img_filename = (
                f"{img_name}/"
                f"{metadata['dimension']}/"
                f"cousin_{i+1:03d}_v{metadata['variation_index']}_transparent.png"
            )
            trans_output_img_path = str(out_dir / trans_output_img_filename)
            os.makedirs(os.path.dirname(trans_output_img_path), exist_ok=True)
            success = remove_background_with_rembg(
                in_path=output_img_path,
                out_path=trans_output_img_path,
            )
            if success:
                print(f"    ✓ Successfully generated transparent image")
            else:
                print(f"    ✗ No images returned")
            
        
        # Save metadata for all generated images
        metadata_path = out_dir / img_name / "generation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(variation_metadata, f, indent=2)
        print(f"\nSaved generation metadata to {metadata_path}")


if __name__ == "__main__":
    main()
