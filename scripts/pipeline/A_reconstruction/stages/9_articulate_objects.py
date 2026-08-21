# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pipeline stage 9: Articulate detected objects using articulate-anything.

This script:
1. Loads detected objects from previous pipeline steps
2. Uses VLM to classify which objects are articulated
3. Runs the articulation workflow in the appropriate conda environment
"""
from simfoundry.utils.prompt_utils import prompt_list_articulated_objects, parse_json_response
from simfoundry.utils.processing_utils import dump_json
from simfoundry.utils.python_utils import sanitize_path_component
from simfoundry.models.vlm import Gemini
from omegaconf import OmegaConf
from pathlib import Path
import shutil
import subprocess
import logging
import json
import hydra
import os
from simfoundry import CFG_DIR
from simfoundry.pipeline.frame_selection import resolve_img_idx
from simfoundry.pipeline.stage_utils import StageResult, finalize_stage
from simfoundry.pipeline.front_canonicalization import orientation_stamp_changed, read_orientation_stamp

logger = logging.getLogger(__name__)

# Change to cfg directory for Hydra
scripts_dir = os.path.dirname(os.path.abspath(__file__))
cfg_dir = CFG_DIR
os.chdir(cfg_dir)

# Paths and constants
ARTICULATE_SIMFOUNDRY_PATH = os.path.abspath("../../deps/articulate-anything/simfoundry")

CONDA_ENVS = {
    "hunyuan": "articulate-anything-hunyuan",
    "partfield": "articulate-anything-partfield",
}

def _normalize_name(name: str) -> str:
    return " ".join(str(name).replace("_", " ").replace("/", " ").lower().split())


def parse_articulated_object_selection(
    response_text: str,
    valid_objects: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Parse VLM articulation selection and keep only exact detected object names."""
    parsed = parse_json_response(response_text)
    if isinstance(parsed, dict):
        raw_selection = parsed.get("articulated_objects", parsed.get("objects", []))
    elif isinstance(parsed, list):
        raw_selection = parsed
    else:
        raise ValueError(f"Expected articulation VLM response to be a JSON object or list, got {type(parsed).__name__}")

    if not isinstance(raw_selection, list):
        raise ValueError("Expected articulation VLM response field 'articulated_objects' to be a list")

    valid_by_normalized = {_normalize_name(obj): obj for obj in valid_objects}
    selected: list[str] = []
    ignored: list[str] = []
    raw_items = [str(item).strip() for item in raw_selection if str(item).strip()]

    for raw_item in raw_items:
        exact = raw_item if raw_item in valid_objects else None
        normalized = valid_by_normalized.get(_normalize_name(raw_item))
        selected_name = exact or normalized
        if selected_name is None:
            ignored.append(raw_item)
        elif selected_name not in selected:
            selected.append(selected_name)

    non_articulated = [obj for obj in valid_objects if obj not in selected]
    return selected, non_articulated, ignored, raw_items


def get_articulation_query_image_path(cfg) -> str | None:
    """Use the same source scene frame that stage 5 uses for VLM scene decomposition."""
    # img_idx indexes the subsampled frame set, which is what stage 5 reads -- indexing
    # frames_all instead only happened to agree with stage 5 when img_idx was 0.
    raw_img_dir = Path(cfg.s1_video.out_dir) / f"frames_subsampled_{cfg.s1_video.n_subsampled_frames}"
    if not raw_img_dir.exists():
        return None
    raw_imgs = sorted(path for path in raw_img_dir.iterdir() if path.suffix.lower() == ".png")
    if not raw_imgs:
        return None
    img_idx = resolve_img_idx(cfg, stage_key="s5_scene")
    if img_idx < 0 or img_idx >= len(raw_imgs):
        logger.warning("Stage 9 articulation image index %s is out of range for %s frames; using frame 0", img_idx, len(raw_imgs))
        img_idx = 0
    return str(raw_imgs[img_idx])


def get_object_scale(cfg, iter_num: str) -> float | None:
    """Real-world scale (stage 8's tf_scale) for one object, or None.

    The articulation workflow's physics-estimation step uses it so the mass
    VLM sees true dimensions — the same scale stage 11 applies to part meshes.
    Reads the same pose-info variant stage 11 consumes (interactive when
    s10_compile.use_interactive_pose is set). Optional by design: the workflow
    also runs standalone on meshes with no pose info at all.
    """
    if cfg.s10_compile.get("use_interactive_pose", False):
        info_dirname = f"info_interactive{cfg.s10_compile.get('interactive_suffix', '')}"
    else:
        info_dirname = "info"
    info_fpath = f"{cfg.s8_pose.out_dir}/{info_dirname}/{iter_num}.json"
    if not os.path.exists(info_fpath):
        logger.warning("No pose info at %s; articulation physics will estimate without real-world scale", info_fpath)
        return None
    try:
        with open(info_fpath) as f:
            info = json.load(f)
        tf_scale = info["z_up"]["scale"]
    except (json.JSONDecodeError, OSError, KeyError, TypeError) as exc:
        logger.warning("Could not read scale from %s (%s); estimating without it", info_fpath, exc)
        return None
    if isinstance(tf_scale, list):
        tf_scale = sum(float(v) for v in tf_scale) / len(tf_scale)
    return float(tf_scale)


def get_conda_env_path(env_name: str) -> str:
    """Get full path to a conda environment (works even when another env is active)."""
    # Get conda base from CONDA_EXE (e.g., /path/to/miniconda3/bin/conda)
    conda_exe = os.environ.get("CONDA_EXE", "")
    if conda_exe:
        conda_base = os.path.dirname(os.path.dirname(conda_exe))
        env_path = os.path.join(conda_base, "envs", env_name)
        if os.path.exists(env_path):
            return env_path
    
    # Fallback: try conda info
    try:
        result = subprocess.run(["conda", "info", "--base"], capture_output=True, text=True, check=True)
        conda_base = result.stdout.strip()
        env_path = os.path.join(conda_base, "envs", env_name)
        if os.path.exists(env_path):
            return env_path
    except subprocess.CalledProcessError:
        pass
    
    raise RuntimeError(f"Could not find conda environment: {env_name}")


def review_classification(articulated: list, non_articulated: list, object_list: dict, cfg) -> tuple:
    """
    Allow user to review and modify the VLM's articulation classification.
    
    Supports three modes:
    1. Config override: Use force_articulated/force_non_articulated from config
    2. Interactive: Prompt user in terminal to modify classification
    3. Skip: Just use VLM classification as-is
    
    Returns:
        tuple: (articulated_objects, non_articulated_objects)
    """
    articulated = list(articulated)  # Make mutable copy
    non_articulated = list(non_articulated)
    
    # 1. Apply config-based overrides first
    force_articulated = cfg.s9_articulate_objects.get("force_articulated", []) or []
    force_non_articulated = cfg.s9_articulate_objects.get("force_non_articulated", []) or []
    
    for obj in force_articulated:
        if obj in non_articulated:
            non_articulated.remove(obj)
            if obj not in articulated:
                articulated.append(obj)
            logger.info(f"Config override: '{obj}' moved to articulated")
    
    for obj in force_non_articulated:
        if obj in articulated:
            articulated.remove(obj)
            if obj not in non_articulated:
                non_articulated.append(obj)
            logger.info(f"Config override: '{obj}' moved to non-articulated")
    
    # 2. Interactive review (if enabled)
    interactive = cfg.s9_articulate_objects.get("interactive_review", False)
    if not interactive:
        return articulated, non_articulated
    
    print("\n" + "=" * 60)
    print("REVIEW ARTICULATION CLASSIFICATION")
    print("=" * 60)
    print(f"\nAll objects: {list(object_list.keys())}")
    print(f"\nCurrently ARTICULATED: {articulated}")
    print(f"Currently NON-ARTICULATED: {non_articulated}")
    print("\nOptions:")
    print("  [a] <name>  - Move object to ARTICULATED")
    print("  [n] <name>  - Move object to NON-ARTICULATED")
    print("  [l]         - List current classification")
    print("  [d]         - Done (accept current classification)")
    print("  [i]         - IPython (manual editing)")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nEnter command: ").strip()
        except EOFError:
            break
        
        if not user_input:
            continue
        
        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()
        
        if cmd == "d":
            print("Accepting current classification.")
            break
        elif cmd == "l":
            print(f"  ARTICULATED: {articulated}")
            print(f"  NON-ARTICULATED: {non_articulated}")
        elif cmd == "i":
            print("Entering IPython. Modify 'articulated' and 'non_articulated' lists, then exit.")
            from IPython import embed
            embed()
        elif cmd == "a" and len(parts) > 1:
            obj_name = parts[1]
            # Find matching object (partial match)
            matches = [o for o in object_list.keys() if obj_name.lower() in o.lower()]
            if len(matches) == 1:
                obj = matches[0]
                if obj in non_articulated:
                    non_articulated.remove(obj)
                if obj not in articulated:
                    articulated.append(obj)
                print(f"  Moved '{obj}' to ARTICULATED")
            elif len(matches) > 1:
                print(f"  Ambiguous match. Did you mean: {matches}?")
            else:
                print(f"  Object '{obj_name}' not found in object list.")
        elif cmd == "n" and len(parts) > 1:
            obj_name = parts[1]
            matches = [o for o in object_list.keys() if obj_name.lower() in o.lower()]
            if len(matches) == 1:
                obj = matches[0]
                if obj in articulated:
                    articulated.remove(obj)
                if obj not in non_articulated:
                    non_articulated.append(obj)
                print(f"  Moved '{obj}' to NON-ARTICULATED")
            elif len(matches) > 1:
                print(f"  Ambiguous match. Did you mean: {matches}?")
            else:
                print(f"  Object '{obj_name}' not found in object list.")
        else:
            print("  Unknown command. Use [a], [n], [l], [d], or [i].")
    
    return articulated, non_articulated


def run_articulation(config_name: str, conda_env: str, simfoundry_path: str, expected_urdfs: list[str]):
    """Run the articulation workflow"""
  
    env_path = get_conda_env_path(conda_env)
    
    cmd = [
        "conda", "run", "--prefix", env_path, "--no-capture-output", "--cwd", simfoundry_path,
        "python", f"{simfoundry_path}/complete_workflow.py", "--config-name", config_name
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        raise RuntimeError(f"Articulation failed with return code: {result.returncode}")

    missing_urdfs = [path for path in expected_urdfs if not os.path.exists(path)]
    if missing_urdfs:
        missing = "\n  ".join(missing_urdfs)
        raise RuntimeError(
            "Articulation subprocess exited successfully but did not create expected URDF(s):\n"
            f"  {missing}"
        )


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    out_dir = cfg.s9_articulate_objects.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Starting articulate objects pipeline...")
    logger.info("=" * 60)

    # Load detected objects from scene decomposition
    object_list = {}
    obj_cat_dir = f"{cfg.s5_scene.out_dir}/obj_cat_list"
    for fname in os.listdir(obj_cat_dir):
        with open(f"{obj_cat_dir}/{fname}") as f:
            info = json.load(f)
        if info["is_valid_removed_obj"]:
            iter_num = fname.replace(".json", "")
            object_list[info["removed_obj_phrase"]] = iter_num

    if not object_list:
        raise RuntimeError(
            f"No valid detected objects in {obj_cat_dir} (every entry has is_valid_removed_obj=false?). "
            "Articulation classification needs stage 5's detected objects; re-run stage 5 before stage 9."
        )

    # Get upsampled images for downstream articulate-anything object inputs.
    upsampled_dir = f"{cfg.s6_upsample.out_dir}/upsampled"
    for obj_name, iter_num in object_list.items():
        img_path = f"{upsampled_dir}/{iter_num}.png"
        if not os.path.exists(img_path):
            logger.warning(f"Missing image for {obj_name}")

    # Infer articulated candidates from the source scene frame and the exact list of
    # valid detected objects, via Vertex AI Gemini.
    vlm = Gemini(
        project=cfg.gcloud_project,
        location="global",
        model=cfg.s9_articulate_objects.vlm_model,
    )

    prompt = prompt_list_articulated_objects(list(object_list.keys()))
    query_image_path = get_articulation_query_image_path(cfg)
    result = vlm(
        prompt=prompt,
        image_paths=query_image_path,
        temperature=0,
        top_p=0,
        seed=0,
        print_results=cfg.visualize,
    )
    if result is None:
        raise RuntimeError("Articulation VLM query failed and returned no result.")
    result_text = vlm.get_result_text(result)
    articulated, non_articulated, ignored_vlm_objects, raw_vlm_selection = parse_articulated_object_selection(
        result_text,
        list(object_list.keys()),
    )
    
    logger.info(f"VLM Classification:")
    logger.info(f"  Articulated: {articulated}")
    logger.info(f"  Non-articulated: {non_articulated}")
    if ignored_vlm_objects:
        logger.warning(f"  Ignored non-detected VLM outputs: {ignored_vlm_objects}")

    raw_articulated = list(articulated)
    raw_non_articulated = list(non_articulated)

    # Allow user to override classification via config or interactively
    articulated, non_articulated = review_classification(
        articulated=articulated,
        non_articulated=non_articulated,
        object_list=object_list,
        cfg=cfg,
    )
    
    # Save final classification
    dump_json({
        "query_image_path": query_image_path,
        "raw_vlm_response": result_text,
        "raw_vlm_selection": raw_vlm_selection,
        "raw_articulated_objects": raw_articulated,
        "raw_non_articulated_objects": raw_non_articulated,
        "ignored_vlm_objects": ignored_vlm_objects,
        "articulated_objects": articulated,
        "non_articulated_objects": non_articulated,
        "object_list": object_list,
    }, f"{out_dir}/object_classification.json")
    
    logger.info(f"Final Classification:")
    logger.info(f"  Articulated: {articulated}")
    logger.info(f"  Non-articulated: {non_articulated}")

    if cfg.s9_articulate_objects.get("classification_only", False):
        logger.info("classification_only=true; stopping after articulation candidate inference.")
        return

    # Build objects list for articulate-simfoundry
    mesh_dir = f"{cfg.s8_pose.out_dir}/canonical_mesh"
    # The articulation workflow lays results out as <root_dir>/<scene_name>/<object>/results/.
    # Sanitize the scene name here (not just the object) so a capitalized scene like "Laptop"
    # lands where stage 11 looks for it.
    sanitized_scene_name = sanitize_path_component(cfg.scene_name)
    objects_list = []
    expected_urdfs = []
    unprocessable = []  # articulated objects whose inputs are missing
    completed_objects = []  # already articulated; may still need physics / refinement

    def record_stage_result(success, **additional_info):
        # cfg may carry an api_key override; never persist credentials into stage_info.json.
        stage_cfg = OmegaConf.create(OmegaConf.to_container(cfg.s9_articulate_objects, resolve=True))
        if "api_key" in stage_cfg:
            stage_cfg.api_key = None
        finalize_stage(
            stage_cfg=stage_cfg,
            out_dir=out_dir,
            result=StageResult(success=success, additional_info=additional_info),
        )

    for obj_name in articulated:
        if obj_name not in object_list:
            logger.warning(f"'{obj_name}' not in object list, skipping")
            unprocessable.append(obj_name)
            continue

        iter_num = object_list[obj_name]
        mesh_path = f"{mesh_dir}/{iter_num}.glb"
        image_path = f"{upsampled_dir}/{iter_num}.png"

        if not os.path.exists(mesh_path):
            logger.warning(f"Mesh not found: {mesh_path}")
            unprocessable.append(obj_name)
            continue
        
        # Check if already articulated (skip if output exists AND the mesh orientation
        # it was built from is unchanged).
        # Both components go through the shared sanitizer so stage 11 can find these again;
        # see simfoundry.utils.python_utils.sanitize_path_component.
        sanitized_name = sanitize_path_component(obj_name)
        obj_out_dir = f"{out_dir}/{sanitized_scene_name}/{sanitized_name}"
        output_urdf = f"{obj_out_dir}/results/mobility.urdf"
        mesh_stamp = read_orientation_stamp(f"{mesh_dir}/{iter_num}_orientation.json")
        stamp_fpath = f"{obj_out_dir}/front_orientation.json"

        object_entry = {
            "name": sanitized_name,
            "mesh_path": os.path.abspath(mesh_path),
            "image_path": os.path.abspath(image_path),
        }
        obj_scale = get_object_scale(cfg, iter_num)
        if obj_scale is not None:
            object_entry["scale"] = obj_scale

        if os.path.exists(output_urdf):
            if not orientation_stamp_changed(read_orientation_stamp(stamp_fpath), mesh_stamp):
                logger.info(f"Skipping '{obj_name}' - already articulated")
                completed_objects.append((object_entry, output_urdf))
                continue
            logger.info(f"Re-articulating '{obj_name}': mesh orientation changed")
            shutil.rmtree(obj_out_dir)

        os.makedirs(obj_out_dir, exist_ok=True)
        with open(stamp_fpath, "w") as f:
            json.dump({"applied_yaw_deg": mesh_stamp[0], "applied_tilt_deg": mesh_stamp[1]}, f)

        objects_list.append(object_entry)
        expected_urdfs.append(output_urdf)
    
    # Already-articulated objects are not re-articulated, but they still go
    # through the workflow when they lack physics estimates or the user asked
    # for the interactive joint-refinement UI — the workflow's per-step
    # artifact checks skip steps 1-5 and only run the physics/refinement steps.
    interactive_refinement = cfg.s9_articulate_objects.get("interactive_joint_refinement", False)
    postprocess_objects = [
        (entry, urdf) for entry, urdf in completed_objects
        if interactive_refinement
        or not os.path.exists(f"{os.path.dirname(urdf)}/physics_properties.json")
    ]
    if postprocess_objects:
        logger.info(
            "Including %d already-articulated object(s) for physics/refinement only: %s",
            len(postprocess_objects), [e["name"] for e, _ in postprocess_objects],
        )
        objects_list.extend(entry for entry, _ in postprocess_objects)
        expected_urdfs.extend(urdf for _, urdf in postprocess_objects)

    if not objects_list:
        # Nothing left to run: success only when every articulated object either has its
        # URDF already or none were detected — missing inputs are recorded as failure so
        # --skip-successful re-runs the stage once upstream provides them.
        logger.warning("No valid articulated objects to process!")
        record_stage_result(
            success=not unprocessable,
            articulated_objects=list(articulated),
            unprocessable_objects=unprocessable,
        )
        return
    
    logger.info(f"Processing {len(objects_list)} objects: {[o['name'] for o in objects_list]}")

    # Load and configure articulation config
    method = cfg.s9_articulate_objects.get("method", "hunyuan")
    template_path = f"{ARTICULATE_SIMFOUNDRY_PATH}/cfg/{method}_template.yaml"
    
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    articulate_cfg = OmegaConf.load(template_path)
    # Must match the sanitized component used in expected_urdfs above; run_articulation
    # validates those paths, so a mismatch here fails loudly rather than silently.
    articulate_cfg.scene_name = sanitized_scene_name
    articulate_cfg.root_dir = os.path.abspath(out_dir)
    articulate_cfg.gcloud_project = cfg.gcloud_project
    articulate_cfg.verbose = True
    articulate_cfg.objects = objects_list
    tree_model = cfg.s9_articulate_objects.get("tree_model", None)
    if tree_model and "s2_generate_articulation_tree" in articulate_cfg:
        articulate_cfg.s2_generate_articulation_tree.model_name = tree_model
    merge_model = cfg.s9_articulate_objects.get("merge_model", None)
    if merge_model and "s4_merge_mesh_parts" in articulate_cfg:
        articulate_cfg.s4_merge_mesh_parts.model_name = merge_model
        articulate_cfg.s4_merge_mesh_parts.interactive_correction = cfg.s9_articulate_objects.get(
            "merge_interactive_correction", False
        )
    if postprocess_objects and "s5_articulate" in articulate_cfg:
        # Never re-articulate: stage 9 already decided which objects are
        # complete (and rmtree'd the stale ones), so the template's
        # s5_articulate.rerun=true must not redo published objects that were
        # included only for physics/refinement.
        articulate_cfg.s5_articulate.rerun = False
    if "s6_refine_articulation" in articulate_cfg:
        articulate_cfg.s6_refine_articulation.enabled = interactive_refinement
    elif interactive_refinement:
        logger.warning(
            "interactive_joint_refinement requested but the articulation template "
            "has no s6_refine_articulation section; update deps/articulate-anything."
        )
    s5_model = cfg.s9_articulate_objects.get("s5_model", merge_model)
    if s5_model and "s5_articulate" in articulate_cfg:
        base_s5_cfg_path = articulate_cfg.s5_articulate.articulation_cfg_path
        if not os.path.isabs(base_s5_cfg_path):
            base_s5_cfg_path = os.path.abspath(os.path.join(ARTICULATE_SIMFOUNDRY_PATH, "..", base_s5_cfg_path))
        s5_cfg = OmegaConf.load(base_s5_cfg_path)
        s5_cfg.model_name = s5_model
        s5_cfg.gcloud_project = cfg.gcloud_project
        s5_cfg.gcloud_location = cfg.get("gcloud_location", "global")
        s5_cfg.vlm_backend = cfg.s9_articulate_objects.get("vlm_backend", "vertex")
        s5_cfg.api_key = cfg.s9_articulate_objects.get("api_key", None)
        s5_cfg.actor_critic.actor_only = cfg.s9_articulate_objects.get("s5_actor_only", True)
        s5_cfg.actor_critic.max_iter = cfg.s9_articulate_objects.get("s5_max_iter", 1)
        s5_cfg.actor_critic.num_seeds = cfg.s9_articulate_objects.get("s5_num_seeds", 1)
        s5_cfg_path = os.path.abspath(f"{out_dir}/s5_articulation_cfg.yaml")
        OmegaConf.save(s5_cfg, s5_cfg_path)
        articulate_cfg.s5_articulate.articulation_cfg_path = s5_cfg_path
    
    # Save config (in simfoundry/cfg for Hydra, and local copy for reference)
    config_name = f"generated_{cfg.scene_name}"
    OmegaConf.save(articulate_cfg, f"{ARTICULATE_SIMFOUNDRY_PATH}/cfg/{config_name}.yaml")
    OmegaConf.save(articulate_cfg, f"{out_dir}/articulate_cfg.yaml")
    logger.info(f"Config saved: {config_name}")

    # Get conda environment
    conda_env = cfg.s9_articulate_objects.get("conda_env") or CONDA_ENVS.get(method)
    if not conda_env:
        raise ValueError(f"Unknown method: {method}")
    
    logger.info(f"Using conda env: {conda_env}")
    
    # Run articulation
    run_articulation(config_name, conda_env, ARTICULATE_SIMFOUNDRY_PATH, expected_urdfs)

    # run_articulation raises when the subprocess fails or an expected URDF is absent;
    # verify the deliverables anyway so the recorded success is a direct artifact check.
    missing_urdfs = [p for p in expected_urdfs if not os.path.exists(p)]
    record_stage_result(
        success=not missing_urdfs and not unprocessable,
        articulated_objects=list(articulated),
        unprocessable_objects=unprocessable,
        expected_urdfs=expected_urdfs,
        missing_urdfs=missing_urdfs,
    )

    logger.info("=" * 60)
    logger.info("Articulation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
