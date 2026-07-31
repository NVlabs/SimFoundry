# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Interactive visualization of domain randomization for OmniGibson scenes.

Loads an OmniGibson scene from a saved JSON config and provides keyboard
controls for applying material and lighting randomization interactively.

Keyboard controls:
    R - Randomize all (materials + lighting)
    M - Randomize materials only
    L - Randomize lighting only
    H - Cycle to next HDRI background
    ESC - Exit

Usage:
    python scripts/interactive/visualize_domain_randomization.py \\
        --scene_json <path_to_scene_json> \\
        --object_names obj_0 obj_1 \\
        --vmaterials_root /opt/nvidia/mdl/vMaterials_2 \\
        --hdr_backgrounds_root assets/hdr_backgrounds
"""

import os
import sys
import argparse

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive domain randomization visualization for OmniGibson scenes."
    )
    parser.add_argument(
        "--scene_json", type=str, required=True,
        help="Path to a saved OmniGibson scene JSON file (from interactive_scene_editor.py).",
    )
    parser.add_argument(
        "--object_names", type=str, nargs="+", default=None,
        help="Object names to randomize (e.g., obj_0 obj_1). If not specified, randomizes all scene objects.",
    )
    parser.add_argument(
        "--material_preset", type=str, default=None,
        help="Pre-configured material preset name (e.g., 'default_train', 'default_eval'). "
             "Overrides library discovery when set.",
    )
    parser.add_argument(
        "--vmaterials_root", type=str, default=None,
        help="Root directory of vMaterials_2. Defaults to $VMATERIALS_ROOT_DIR or /opt/nvidia/mdl/vMaterials_2.",
    )
    parser.add_argument(
        "--hdr_backgrounds_root", type=str, default=None,
        help="Root directory for HDR backgrounds. Defaults to $HDR_BACKGROUNDS_ROOT_DIR or assets/hdr_backgrounds/.",
    )
    parser.add_argument(
        "--categories", type=str, nargs="+", default=None,
        help="Material categories to use (e.g., Wood Metals Plastics). Default: all categories.",
    )
    parser.add_argument(
        "--num_variants", type=int, default=3,
        help="Number of randomized variants per base material (default: 3). "
             "Use 0 or negative to get one copy per natural variant (no duplication).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate scene JSON exists
    if not os.path.exists(args.scene_json):
        print(f"Error: Scene JSON not found: {args.scene_json}")
        sys.exit(1)

    # Import OmniGibson (triggers Isaac Sim initialization)
    import omnigibson as og
    import omnigibson.lazy as lazy
    from omnigibson.utils.ui_utils import KeyboardEventHandler

    # Import domain randomization
    from simfoundry.domain_randomization import (
        DomainRandomizationCfg,
        MaterialRandomizationCfg,
        LightingRandomizationCfg,
        DomainRandomizationManager,
    )
    from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths

    # ----------------------------------------------------------------
    # Load scene
    # ----------------------------------------------------------------
    print(f"Loading scene from: {args.scene_json}")
    scene_json_dict = load_json_with_absolute_usd_paths(args.scene_json)
    scene_json_dict["init_info"]["args"]["use_skybox"] = True

    og.launch()
    og.sim.restore(scene_files=[scene_json_dict])
    scene = og.sim.scenes[0]

    # Step briefly for initialization
    og.sim.play()
    for _ in range(10):
        og.sim.step()

    # ----------------------------------------------------------------
    # Determine objects to randomize
    # ----------------------------------------------------------------
    if args.object_names is not None:
        object_names = args.object_names
    else:
        # Randomize all non-robot objects
        object_names = [obj.name for obj in scene.objects if ("robot" not in obj.name and "background" not in obj.name)]

    print(f"Objects to randomize: {object_names}")

    # ----------------------------------------------------------------
    # Build domain randomization config
    # ----------------------------------------------------------------
    mat_cfg_kwargs = dict(
        enabled=True,
        object_names=object_names,
        num_variants_per_material=args.num_variants if args.num_variants > 0 else None,
    )
    if args.material_preset is not None:
        mat_cfg_kwargs["material_preset"] = args.material_preset
    if args.vmaterials_root is not None:
        mat_cfg_kwargs["vmaterials_root_dir"] = args.vmaterials_root
    if args.categories is not None:
        mat_cfg_kwargs["categories"] = args.categories

    light_cfg_kwargs = dict(
        enabled=True,
        use_hdri_textures=True,
    )
    if args.hdr_backgrounds_root is not None:
        light_cfg_kwargs["hdr_backgrounds_root_dir"] = args.hdr_backgrounds_root

    cfg = DomainRandomizationCfg(
        enabled=True,
        materials=MaterialRandomizationCfg(**mat_cfg_kwargs),
        lighting=LightingRandomizationCfg(**light_cfg_kwargs),
    )

    # ----------------------------------------------------------------
    # Initialize domain randomization
    # ----------------------------------------------------------------
    manager = DomainRandomizationManager(cfg)
    print("Initializing domain randomization (preloading materials)...")
    manager.initialize()
    print("Domain randomization ready.")

    # ----------------------------------------------------------------
    # Keyboard controls
    # ----------------------------------------------------------------
    def randomize_all():
        print("[DR] Randomizing all (materials + lighting)...")
        manager.randomize()
        print("[DR] Done.")

    def randomize_materials():
        print("[DR] Randomizing materials only...")
        manager._randomize_materials()
        print("[DR] Done.")

    def randomize_lighting():
        print("[DR] Randomizing lighting only...")
        if manager.lighting_randomizer is not None:
            manager.lighting_randomizer.randomize()
        print("[DR] Done.")

    def cycle_hdri():
        if manager.lighting_randomizer is not None:
            manager.lighting_randomizer.cycle_next_hdri()

    def exit_viewer():
        print("Exiting domain randomization viewer...")
        manager.cleanup()
        og.shutdown()
        sys.exit(0)

    KeyboardEventHandler.initialize()

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.R,
        callback_fn=randomize_all,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.M,
        callback_fn=randomize_materials,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.L,
        callback_fn=randomize_lighting,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.H,
        callback_fn=cycle_hdri,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.ESCAPE,
        callback_fn=exit_viewer,
    )

    # ----------------------------------------------------------------
    # Print controls and run main loop
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Domain Randomization Viewer")
    print("=" * 60)
    print("  R     - Randomize all (materials + lighting)")
    print("  M     - Randomize materials only")
    print("  L     - Randomize lighting only")
    print("  H     - Cycle to next HDRI background")
    print("  ESC   - Exit")
    print("=" * 60 + "\n")

    try:
        while True:
            og.sim.step()
            og.sim.render()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        manager.cleanup()
        og.shutdown()


if __name__ == "__main__":
    main()
