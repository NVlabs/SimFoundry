# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import a URDF asset into OmniGibson's USD format.

Replaces `python -m omnigibson.examples.objects.import_custom_object` for stage 13.

That example script is unusable on current OmniGibson `main`: it calls
`import_og_asset_from_urdf(..., keep_instanceable=not no_keep_instanceable)`, but the
function no longer declares `keep_instanceable`, so every invocation dies with

    TypeError: import_og_asset_from_urdf() got an unexpected keyword argument 'keep_instanceable'

Upstream dropped the parameter and now controls instancing with the module constant
`_ALLOW_INSTANCING = False` in `omnigibson/utils/asset_conversion_utils.py` — which is exactly
the behavior stage 13 was asking for by passing `--no_keep_instanceable`. The example simply was
not updated alongside the function.

Rather than patch a gitignored third-party checkout, stage 13 calls this module. Only the URDF
branch of the example is reproduced, because SimFoundry always passes a `.urdf` (stage 11 writes
it) and never the raw-mesh branch that needs `generate_urdf_for_mesh`.

Arguments are filtered against the installed signature, so this works both on the commit
`install_simfoundry.sh` pins (which accepts `keep_instanceable`) and on `main` (which does not),
instead of breaking whenever that checkout moves.
"""

from __future__ import annotations

import argparse
import inspect
import sys


def build_supported_kwargs(import_fn, requested):
    """
    Drops requested kwargs that @import_fn does not declare.

    OmniGibson's `import_og_asset_from_urdf` signature differs across the versions this repo can
    have checked out in `deps/BEHAVIOR-1K`. Passing an argument it does not accept is a hard
    TypeError, and silently dropping one that changes behavior would be worse, so anything
    dropped is reported.

    Args:
        import_fn (callable): `import_og_asset_from_urdf`
        requested (dict): Arguments stage 13 wants to pass

    Returns:
        tuple[dict, list[str]]: (accepted kwargs, sorted names that were dropped)
    """
    parameters = inspect.signature(import_fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(requested), []
    accepted = {
        name for name, p in parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    supported = {k: v for k, v in requested.items() if k in accepted}
    dropped = sorted(set(requested) - set(supported))
    return supported, dropped


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--asset-path", required=True, help="Absolute path to the .urdf to import")
    parser.add_argument("--category", required=True)
    parser.add_argument("--model", required=True, help="6 alphabetic characters, unique in the dataset")
    parser.add_argument("--collision-method", default="none",
                        help="'none', 'coacd' or 'convex'. 'none' is passed through as None.")
    parser.add_argument("--hull-count", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-instanceable", action="store_true",
                        help="Only honored by OmniGibson versions that still accept it.")
    parser.add_argument("--no-import-inertia", action="store_true")
    parser.add_argument("--asset-pipeline-materials", action="store_true",
                        help="Re-bind materials through OmniGibson's asset pipeline so the "
                             "map_Pm/map_Pr maps stage 11 writes reach the USD. Objects whose "
                             "base color carries real alpha are skipped (they keep the URDF "
                             "importer's OmniPBR_Opacity materials for the opacity pass).")
    return parser.parse_args(argv)


def base_color_has_alpha(model_root, min_translucent_fraction=0.01):
    """True when any map_Kd texture under @model_root has a meaningful alpha channel."""
    import numpy as np
    from PIL import Image
    from pathlib import Path

    for mtl_path in Path(model_root).rglob("*.mtl"):
        with open(mtl_path) as f:
            for line in f:
                if not line.startswith("map_Kd"):
                    continue
                texture_path = (mtl_path.parent / line.split(" ", 1)[1].strip()).resolve()
                if not texture_path.exists():
                    continue
                image = Image.open(texture_path)
                if "A" not in image.getbands():
                    continue
                alpha = np.asarray(image.convert("RGBA"))[..., 3]
                if (alpha < 250).mean() > min_translucent_fraction:
                    return True
    return False


def apply_asset_pipeline_materials(category, model, dataset_name, usd_path, prim, source_urdf_path):
    """Run the asset-pipeline material pass import_obj_metadata gates off by default.

    import_og_asset_from_urdf never passes force_asset_pipeline_materials=True, so the
    metalness/glossiness maps stage 11 writes into the MTLs are parsed by nothing. This
    invokes the same pass on the already-imported prim and saves the stage. Fail-soft:
    any error keeps the URDF importer's materials (today's behavior).

    The pass parses {model_root}/urdf/{model}_with_meta_links.urdf and the OBJ/MTL files
    it references, but the import only copies material/, misc/, and usd/ into the dataset —
    so the source URDF tree (stage 11's output) is staged into the dataset first, keeping
    the USD's texture references inside the model directory.
    """
    import json
    import os
    import shutil
    import traceback

    # The kit process swallows stdout, so record the outcome in a file beside the USD.
    report = {"status": "unknown"}
    report_path = None
    try:
        import omnigibson.lazy as lazy
        from omnigibson.utils.asset_conversion_utils import _force_asset_pipeline_materials
        from omnigibson.utils.asset_utils import get_dataset_path

        dataset_root = get_dataset_path(dataset_name)
        model_root = os.path.join(dataset_root, "objects", category, model)
        report_path = os.path.join(os.path.dirname(str(usd_path)), "material_pass_report.json")
        source_urdf_dir = os.path.dirname(os.path.abspath(source_urdf_path))
        dataset_urdf_dir = os.path.join(model_root, "urdf")
        shutil.copytree(source_urdf_dir, dataset_urdf_dir, dirs_exist_ok=True)
        meta_links_urdf = os.path.join(dataset_urdf_dir, f"{model}_with_meta_links.urdf")
        if not os.path.exists(meta_links_urdf):
            shutil.copy2(
                os.path.join(dataset_urdf_dir, os.path.basename(source_urdf_path)),
                meta_links_urdf,
            )
        if base_color_has_alpha(model_root):
            report["status"] = "skipped_alpha"
            print(f"[og_asset_import] {category}/{model}: base color has alpha; keeping "
                  "URDF-importer materials so the opacity-threshold pass still applies.")
            return
        stage = prim.GetStage()
        stage.SetEditTarget(stage.GetRootLayer())
        _force_asset_pipeline_materials(
            obj_prim=prim,
            obj_category=category,
            obj_model=model,
            usd_path=str(usd_path),
            dataset_root=dataset_root,
        )
        material_paths = [p.GetPath() for p in stage.Traverse() if p.GetTypeName() == "Material"]
        report["materials"] = [str(p) for p in material_paths]
        report["material_layers"] = {
            str(p): {
                "root": stage.GetRootLayer().GetPrimAtPath(p) is not None,
                "session": stage.GetSessionLayer().GetPrimAtPath(p) is not None,
            }
            for p in material_paths
        }
        # The material kit commands author on the session layer regardless of the edit
        # target; Stage.Save() never persists that. Export the composed stage instead so
        # the bindings land in the file (single-layer object USD, safe to flatten).
        in_session_only = any(
            layers["session"] and not layers["root"]
            for layers in report["material_layers"].values()
        )
        if in_session_only:
            stage.Export(str(usd_path))
            report["persisted_via"] = "export_flattened"
        else:
            stage.Save()
            report["persisted_via"] = "save"
        report["status"] = "bound"
        report["stage_root"] = stage.GetRootLayer().identifier
        print(f"[og_asset_import] {category}/{model}: bound asset-pipeline materials.")
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
        print(f"[og_asset_import] {category}/{model}: asset-pipeline material pass failed; "
              "keeping the URDF importer's materials.")
    finally:
        if report_path is not None:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)


def main(argv=None):
    args = parse_args(argv)

    if not args.asset_path.endswith(".urdf"):
        # The raw-mesh branch of the upstream example is deliberately not reproduced here.
        raise SystemExit(f"--asset-path must be a .urdf, got: {args.asset_path}")
    # Mirrors the upstream example's assertion; OmniGibson's dataset layout relies on it.
    if not (len(args.model) == 6 and args.model.isalpha()):
        raise SystemExit(f"--model must be 6 alphabetic characters, got: {args.model!r}")

    import omnigibson as og
    from omnigibson.utils.asset_conversion_utils import import_og_asset_from_urdf

    requested = dict(
        dataset_name=args.dataset_name,
        category=args.category,
        model=args.model,
        urdf_path=args.asset_path,
        collision_method=None if args.collision_method == "none" else args.collision_method,
        hull_count=args.hull_count,
        overwrite=args.overwrite,
        keep_instanceable=args.keep_instanceable,
        import_inertia_tensor=not args.no_import_inertia,
        use_usda=False,
    )
    supported, dropped = build_supported_kwargs(import_og_asset_from_urdf, requested)
    if dropped:
        print(
            f"[og_asset_import] This OmniGibson build does not accept {dropped}; "
            f"not passing them. Instancing is governed by asset_conversion_utils._ALLOW_INSTANCING "
            f"(False upstream), which matches SimFoundry's --no-keep-instanceable intent."
        )

    try:
        result = import_og_asset_from_urdf(**supported)
        if args.asset_pipeline_materials:
            try:
                _, usd_path, prim = result
            except (TypeError, ValueError):
                print("[og_asset_import] This OmniGibson build's import_og_asset_from_urdf "
                      "did not return (urdf, usd, prim); skipping the material pass.")
            else:
                apply_asset_pipeline_materials(
                    category=args.category,
                    model=args.model,
                    dataset_name=args.dataset_name,
                    usd_path=usd_path,
                    prim=prim,
                    source_urdf_path=args.asset_path,
                )
    except BaseException:
        import os
        import traceback
        traceback.print_exc()
        # og.shutdown() exits 0 and would mask the failure; die hard with a real code.
        os._exit(1)
    # Tear the simulator down, otherwise the process hangs holding the GPU.
    og.shutdown()


if __name__ == "__main__":
    sys.exit(main())
