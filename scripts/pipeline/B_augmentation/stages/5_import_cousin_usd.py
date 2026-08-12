# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from b1k

Requires installing:

- BEHAVIOR-1K, see https://github.com/StanfordVL/BEHAVIOR-1K
"""

from omnigibson.utils.asset_utils import get_dataset_path
from pathlib import Path
import json
import subprocess
import hydra
import time
import os
import sys
from simfoundry import CFG_DIR

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["OMNIGIBSON_HEADLESS"] = "1"
import omnigibson as og


def cfg_path(path_value, *parts):
    path = Path(str(path_value))
    if not path.is_absolute():
        path = (Path(CFG_DIR) / path).resolve()
    return path.joinpath(*parts)


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    sim_dir = cfg_path(cfg.sim.out_dir)
    out_dir = cfg_path(cfg.usd.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shared with A_reconstruction stage 12 so cousins and reconstructed objects render alike.
    opacity_threshold = cfg.s12_usd.get("opacity_threshold", 0.5)

    # Load scene objects info
    scene_objects_info_fpath = sim_dir / "scene_objects_info.json"
    with open(scene_objects_info_fpath, "r") as f:
        scene_objects_info = json.load(f)

    for idx, obj_info in reversed(scene_objects_info.items()):
        obj_category = obj_info["category"]
        obj_model = obj_info["model"]
        obj_name = obj_info["name"]

        # Make sure there are no "-" or " " in the string -- this causes downstream importing issues
        assert "-" not in obj_category
        assert " " not in obj_category

        obj_urdf_fpath = sim_dir / "objects" / obj_category / obj_model / "urdf" / f"{obj_model}.urdf"

        dst_usd_fpaths = [
            Path(out_dir) / "objects" / obj_category / obj_model / "usd" / f"{obj_model}.usd",
        ]
        try:
            dataset_path = get_dataset_path("custom-assets")
            dst_usd_fpaths.append(
                Path(dataset_path) / "objects" / obj_category / obj_model / "usd" / f"{obj_model}.usd"
            )
        except Exception:
            dataset_path = None

        if any(dst_usd_fpath.exists() for dst_usd_fpath in dst_usd_fpaths):
            print(
                f"Skipping existing USD for {obj_name}: "
                f"{next(str(path) for path in dst_usd_fpaths if path.exists())}"
            )
            continue

        # Import
        subprocess.run([
            "python",
            "-m", "omnigibson.examples.objects.import_custom_object",
            "--dataset-name", "custom-assets", #"flag_scene_cousins-assets",
            "--asset-path", str(obj_urdf_fpath),
            "--category", obj_category,
            "--model", obj_model,
            "--collision-method", "none",
            "--no_keep_instanceable",
            # "--no_import_inertia",
            "--headless",
            "--overwrite",
        ], check=True)

        # Cousin meshes come from the same generators as the reconstructed objects, so they
        # carry the same baked-in transparency; without a cutout threshold they render as a
        # translucent cloud of particles. See A_reconstruction stage 12.
        if opacity_threshold is not None:
            imported = next((path for path in dst_usd_fpaths if path.exists()), None)
            if imported is not None:
                subprocess.run([
                    "python", str(Path(__file__).parent / "set_usd_opacity_threshold.py"),
                    str(imported), "--threshold", str(opacity_threshold),
                ], check=True)
            else:
                print(f"Warning: imported USD not found for {obj_name}, skipping opacity threshold")

    og.shutdown()

if __name__ == "__main__":
    main()
