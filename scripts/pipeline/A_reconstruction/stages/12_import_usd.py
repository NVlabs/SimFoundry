# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from b1k

Requires installing:

- BEHAVIOR-1K, see https://github.com/StanfordVL/BEHAVIOR-1K
"""
import os
import sys
# We need to run this in headless mode for omnigibson
os.environ["OMNIGIBSON_HEADLESS"] = "1"

import omnigibson as og
from omnigibson.utils.asset_utils import get_dataset_path
from pathlib import Path
import json
import subprocess
import hydra
from simfoundry import CFG_DIR
from simfoundry.pipeline.stage_utils import StageResult, bootstrap_hydra_workdir, finalize_stage
import logging

scripts_dir = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
bootstrap_hydra_workdir(__file__)


def resolve_pipeline_script(script_name: str) -> str:
    """Find a USD post-processing helper shared with augmentation."""
    pipeline_dir = os.path.dirname(os.path.dirname(scripts_dir))
    candidates = [
        os.path.join(scripts_dir, script_name),
        os.path.join(pipeline_dir, "B_augmentation", "stages", script_name),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find {script_name} in any expected location: " + ", ".join(candidates)
    )


def resolve_reparent_script() -> str:
    """Find the USD joint reparenting helper shared with augmentation."""
    return resolve_pipeline_script("reparent_usd_joints.py")


def imported_usd_path(dataset_name: str, obj_category: str, obj_model: str) -> str:
    dataset_path = get_dataset_path(dataset_name)
    return os.path.join(dataset_path, "objects", obj_category, obj_model, "usd", f"{obj_model}.usd")


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    sim_dir = cfg.s10_sim.out_dir
    out_dir = cfg.s12_usd.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # null disables the pass, leaving whatever the importer authored.
    opacity_threshold = cfg.s12_usd.get("opacity_threshold", 0.5)

    # Load scene objects info
    scene_objects_info_fpath = f"{sim_dir}/scene_objects_info.json"
    with open(scene_objects_info_fpath, "r") as f:
        scene_objects_info = json.load(f)

    logger.info("="*60)
    logger.info("Starting USD import pipeline...")
    logger.info(f"Output directory: {out_dir}")
    logger.info("="*60)

    for idx, obj_info in reversed(scene_objects_info.items()):
        obj_category = obj_info["category"]
        obj_model = obj_info["model"]
        obj_name = obj_info["name"]

        # Make sure there are no "-" or " " in the string -- this causes downstream importing issues
        assert "-" not in obj_category
        assert " " not in obj_category

        obj_urdf_fpath = f"{sim_dir}/objects/{obj_category}/{obj_model}/urdf/{obj_model}.urdf"

        # Check if this is an articulated object
        is_articulated = obj_info.get("is_articulated", False)
        
        # Import
        subprocess.run([
            "python",
            "-m", "omnigibson.examples.objects.import_custom_object",
            "--dataset-name", "real2sim-assets",
            "--asset-path", obj_urdf_fpath,
            "--category", obj_category,
            "--model", obj_model,
            "--collision-method", "none",
            "--no_keep_instanceable",
            # "--no_import_inertia",
            "--headless",
            "--overwrite",
        ], check=True)
        
        # For articulated objects, reparent joints in the USD file
        # OmniGibson expects joints to be children of their parent link prims,
        # but Isaac Sim's URDF importer places them in a separate /joints scope
        usd_path = imported_usd_path(cfg.s12_usd.dataset_name, obj_category, obj_model)

        if is_articulated:
            if os.path.exists(usd_path):
                logger.info(f"Reparenting joints for articulated object: {obj_name}")
                # Run reparenting as a separate subprocess (needs its own SimulationApp instance)
                reparent_script = resolve_reparent_script()
                subprocess.run(["python", reparent_script, usd_path], check=True)
            else:
                logger.warning(f"USD file not found for reparenting: {usd_path}")

        # TRELLIS.2 bakes transparency into its textures, so the importer wires them up as
        # OmniPBR_Opacity materials whose opacity_threshold defaults to 0.0 -- alpha blending,
        # which renders the object as a translucent cloud of particles. A non-zero threshold
        # switches the material to alpha cutout so it renders solid.
        if opacity_threshold is not None:
            if os.path.exists(usd_path):
                logger.info(f"Setting material opacity threshold for {obj_name}: {opacity_threshold}")
                subprocess.run([
                    "python", resolve_pipeline_script("set_usd_opacity_threshold.py"),
                    usd_path, "--threshold", str(opacity_threshold),
                ], check=True)
            else:
                logger.warning(f"USD file not found for opacity threshold: {usd_path}")

    logger.info("="*60)
    logger.info("USD import complete!")
    logger.info("="*60)

    finalize_stage(
        stage_cfg=cfg.s12_usd,
        out_dir=cfg.s12_usd.out_dir,
        result=StageResult(success=True),
    )

    # Bypass OmniGibson/Isaac Sim teardown — avoids SIGSEGV in libomni.syntheticdata.plugin.so
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
