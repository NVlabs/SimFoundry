# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Check that OmniGibson agrees with what the light editor wrote.

Loads an export in OmniGibson and checks:

1. the exact set of authored objects loads -- no object missing, none extra;
2. each one lands at the authored position, orientation *and* scale, before any
   physics runs, and again after two ``scene.reset()`` calls;
3. the scene steps without exploding, with post-step drift reported separately;
4. every requested external camera renders.

The comparison itself lives in :mod:`parity`, which has no OmniGibson import
and is unit-tested offline.

Scope: this restores a scene through ``og.sim.restore()`` and renders cameras
by moving ``og.sim.viewer_camera``. It does not construct the evaluation
``Environment`` with a task and its ``external_sensors``, so a pass proves the
export loads and survives resets -- not that the sensors initialize in the real
evaluation harness. Treat a pass as necessary, not sufficient.

Rendered camera frames are written out so the frustum convention can be
compared against the browser's camera view.

Needs OmniGibson: run it in the SimFoundry environment, not the light editor's
venv.

Usage:
    python parity_check.py --scene <scene_state.json> [--cameras <cfg name>] [--out DIR]

Exit status is non-zero if any check above fails.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _write_report(path, payload):
    if path is None:
        return
    try:
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"WARNING: could not write report to {path}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Verify OmniGibson parity of a light-editor export")
    parser.add_argument("--scene", required=True, help="Scene JSON produced by the light editor")
    parser.add_argument("--cameras", default=None, help="external_sensors config name or path")
    parser.add_argument("--out", default=None, help="Directory for rendered frames (default: alongside the scene)")
    parser.add_argument(
        "--tolerance", type=float, default=1e-4,
        help="Metres of pose difference tolerated before physics runs (default: 1e-4)",
    )
    parser.add_argument("--steps", type=int, default=60, help="Physics steps to prove stability")
    parser.add_argument("--report", default=None, help="Write a JSON report here")
    args = parser.parse_args()

    scene_json = Path(args.scene).resolve()
    out_dir = Path(args.out) if args.out else scene_json.parent / "parity"
    report = {
        "ok": False, "scene": str(scene_json), "tolerance": args.tolerance,
        "pose_mismatches": [], "checked": 0, "cameras": [], "stepped": False,
        # Where the floor plane was actually placed.
        "ground_plane": None, "error": None,
    }

    if not scene_json.exists():
        report["error"] = f"scene JSON not found: {scene_json}"
        _write_report(args.report, report)
        sys.exit(f"ERROR: {report['error']}")

    try:
        from omnigibson.macros import gm
        gm.HEADLESS = True

        import omnigibson as og
        import torch as th
        from simfoundry.utils.ground_plane_utils import (
            apply_ground_plane_info,
            describe as describe_ground_plane,
        )
        from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths

        from parity import authored_transforms, compare_scene, format_report
        from scene_io import load_scene

        authored_doc = load_scene(scene_json)
        authored = authored_transforms(authored_doc)

        print(f"Loading {scene_json.name} in OmniGibson ...")
        og.launch()
        # usd_path values are relative to the scene JSON, but restore() resolves
        # them against the process cwd, so hand it a pre-resolved document.
        og.sim.restore(scene_files=[load_json_with_absolute_usd_paths(str(scene_json))])

        # restore() does not read `ground_plane_info`; apply it before anything
        # steps so the floor sits where the scene says, not at z=0.
        applied = apply_ground_plane_info(authored_doc, og.sim.floor_plane)
        report["ground_plane"] = applied
        print(f"Ground: {describe_ground_plane(applied)}")
        if applied is None and authored_doc.get("ground_plane_info"):
            print("WARNING: the scene carries a ground_plane_info but this run has no "
                  "floor plane, so nothing here is holding the props up.",
                  file=sys.stderr)

        scene = og.sim.scenes[0]

        def observe():
            """Read back every loaded object's full transform."""
            observed = {}
            for obj in scene.objects:
                pos, ori = obj.get_position_orientation()
                entry = {
                    "position": [float(v) for v in pos],
                    "orientation": [float(v) for v in ori],
                }
                scale = getattr(obj, "scale", None)
                if scale is not None:
                    entry["scale"] = [float(v) for v in scale]
                observed[obj.name] = entry
            # Only compare what was authored: OmniGibson adds fixtures such as
            # the ground plane that never appear in objects_info.
            return {k: v for k, v in observed.items() if k in authored}

        # --- 1. as loaded, before physics can move anything -----------------
        stages = {}
        stages["load"] = compare_scene(
            authored, observe(), position_tol=args.tolerance,
        )
        print(f"\n--- as loaded ---\n{format_report(stages['load'])}")

        # --- 2. across resets ----------------------------------------------
        # Evaluation resets the scene before every episode, so a pose that only
        # survives until the first reset is not usable.
        for index in (1, 2):
            try:
                scene.reset()
                # The evaluation task re-places the floor on every reset, so
                # this gate does too.
                apply_ground_plane_info(authored_doc, og.sim.floor_plane)
                stages[f"reset{index}"] = compare_scene(
                    authored, observe(), position_tol=args.tolerance,
                )
                print(f"\n--- after reset {index} ---\n"
                      f"{format_report(stages[f'reset{index}'])}")
            except Exception as e:  # noqa: BLE001 - recorded, not fatal
                stages[f"reset{index}"] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                                           "missing": [], "unexpected": [], "objects": {}}
                print(f"\n--- after reset {index} --- FAILED: {e}")

        # --- 3. does it actually run ---------------------------------------
        print(f"\nStepping {args.steps} steps ...")
        og.sim.play()
        for _ in range(args.steps):
            og.sim.step()
        report["stepped"] = True
        # Drift under physics is a settling question, not an authoring-parity
        # one; reported but not gated.
        stages["stepped"] = compare_scene(authored, observe(), position_tol=args.tolerance)
        print(f"\n--- after {args.steps} steps (drift, informational) ---\n"
              f"{format_report(stages['stepped'])}")

        report["stages"] = stages
        report["checked"] = len(stages["load"]["objects"])
        report["pose_mismatches"] = [
            {"name": name, "failures": result["failures"]}
            for name, result in stages["load"]["objects"].items()
            if not result["ok"]
        ]
        report["missing"] = stages["load"]["missing"]
        report["unexpected"] = stages["load"]["unexpected"]

        # --- 4. cameras render ---------------------------------------------
        if args.cameras:
            from camera_io import load_cameras, resolve_camera_config
            repo_root = HERE.parents[2]
            cfg = resolve_camera_config(args.cameras, repo_root)
            cameras, _ = load_cameras(cfg)
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nRendering {len(cameras)} camera(s) from {cfg.name} ...")

            for cam in cameras:
                entry = {"name": cam["name"], "rendered": False, "image": None, "error": None}
                try:
                    sensor = og.sim.viewer_camera
                    # External sensors are recreated by the eval stage, not by
                    # restore(); placing the viewer at the camera's world pose
                    # is what tests the pose convention.
                    parent = next(
                        (o for o in scene.objects
                         if cam["relative_prim_path"].split("/")[1].endswith(o.name)),
                        None,
                    )
                    pos = th.tensor(cam["position"], dtype=th.float32)
                    ori = th.tensor(cam["orientation"], dtype=th.float32)
                    if parent is not None:
                        import omnigibson.utils.transform_utils as T
                        p_pos, p_ori = parent.get_position_orientation()
                        pos = p_pos + T.quat2mat(p_ori) @ pos
                        ori = T.quat_multiply(p_ori, ori)
                    sensor.set_position_orientation(position=pos, orientation=ori)
                    for _ in range(3):
                        og.sim.render()
                    obs, _ = sensor.get_obs()
                    rgb = obs["rgb"][:, :, :3].cpu().numpy() if hasattr(obs["rgb"], "cpu") else obs["rgb"][:, :, :3]
                    from PIL import Image
                    path = out_dir / f"{cam['name']}.png"
                    Image.fromarray(rgb.astype("uint8")).save(path)
                    entry.update(rendered=True, image=str(path))
                    print(f"  {cam['name']:22s} -> {path.name}")
                except Exception as e:  # noqa: BLE001 - one bad camera is not fatal
                    entry["error"] = f"{type(e).__name__}: {e}"
                    print(f"  {cam['name']:22s} FAILED: {entry['error']}")
                report["cameras"].append(entry)

        # Every requested camera must render for the gate to pass.
        camera_failures = [c["name"] for c in report["cameras"] if not c["rendered"]]
        report["camera_failures"] = camera_failures

        # Load and both resets must agree with what was authored; post-step
        # drift is settle.py's job.
        gated = [name for name in ("load", "reset1", "reset2") if name in stages]
        report["ok"] = (
            report["stepped"]
            and not camera_failures
            and all(stages[name].get("ok") for name in gated)
        )
    except Exception as e:  # noqa: BLE001 - the report must survive any failure
        report["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        _write_report(args.report, report)

    print("\n" + "=" * 42)
    if report["error"]:
        print(f"PARITY FAILED: {report['error']}")
    elif report["pose_mismatches"]:
        print(f"PARITY FAILED: {len(report['pose_mismatches'])} pose mismatch(es)")
    else:
        print(f"PARITY OK: {report['checked']} object(s) within {args.tolerance} m; scene stepped")
    if report["cameras"]:
        print(f"Camera frames in {out_dir} — compare against the browser camera view.")

    # og.shutdown() calls app.close(), which would replace this exit status.
    sys.stdout.flush()
    os._exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
