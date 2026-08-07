# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import a URDF asset into OmniGibson's USD format.

Replaces `python -m omnigibson.examples.objects.import_custom_object` for stage 12.

That example script is unusable on current OmniGibson `main`: it calls
`import_og_asset_from_urdf(..., keep_instanceable=not no_keep_instanceable)`, but the
function no longer declares `keep_instanceable`, so every invocation dies with

    TypeError: import_og_asset_from_urdf() got an unexpected keyword argument 'keep_instanceable'

Upstream dropped the parameter and now controls instancing with the module constant
`_ALLOW_INSTANCING = False` in `omnigibson/utils/asset_conversion_utils.py` — which is exactly
the behavior stage 12 was asking for by passing `--no_keep_instanceable`. The example simply was
not updated alongside the function.

Rather than patch a gitignored third-party checkout, stage 12 calls this module. Only the URDF
branch of the example is reproduced, because SimFoundry always passes a `.urdf` (stage 10 writes
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
        requested (dict): Arguments stage 12 wants to pass

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
    return parser.parse_args(argv)


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
        import_og_asset_from_urdf(**supported)
    finally:
        # Always tear the simulator down, otherwise the process hangs holding the GPU.
        og.shutdown()


if __name__ == "__main__":
    sys.exit(main())
