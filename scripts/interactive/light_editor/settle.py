# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Settle a hand-edited scene under physics, headless, and report what moved.

Drops the scene into contact in the simulator so floating or intersecting
placements show up as measured drift. Needs OmniGibson: run it in the
SimFoundry environment, not the light editor's venv. The browser server only
invokes it when launched with ``--settle-after-save``.

Usage:
    python settle.py --scene <scene_state.json> [--steps 240] [--promote]

Exit status is non-zero if any object moved more than the tolerances, so this
can gate a pipeline step.
"""

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _write_report(path, payload):
    """Write the machine-readable report, if one was requested."""
    if path is None:
        return
    try:
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"WARNING: could not write report to {path}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Settle an edited scene under physics")
    parser.add_argument("--scene", required=True, help="Scene JSON to settle")
    parser.add_argument("--steps", type=int, default=240, help="Physics steps (default: 240, ~4s)")
    # Three tolerances: root drift and sliding joints are metres, hinges are
    # degrees.
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.005,
        help="Root translation in metres above which an object is reported as "
             "unsettled (default: 0.005)",
    )
    parser.add_argument(
        "--prismatic-tolerance",
        type=float,
        default=0.005,
        help="Sliding-joint travel in metres above which a joint is reported as "
             "drifted (default: 0.005)",
    )
    parser.add_argument(
        "--revolute-tolerance-deg",
        type=float,
        default=2.0,
        help="Hinge rotation in degrees above which a joint is reported as "
             "drifted (default: 2)",
    )
    parser.add_argument("--promote", action="store_true", help="Also write _scene_state_latest.json")
    parser.add_argument("--report", default=None, help="Write a JSON report to this path")
    parser.add_argument("--headless", action="store_true", default=True, help="No GUI (default)")
    parser.add_argument("--gui", dest="headless", action="store_false", help="Show the viewport")
    args = parser.parse_args()

    scene_json = Path(args.scene).resolve()
    report = {
        "ok": False,
        "scene": str(scene_json),
        "settled_path": None,
        "steps": args.steps,
        "tolerance": args.tolerance,
        "prismatic_tolerance": args.prismatic_tolerance,
        "revolute_tolerance_deg": args.revolute_tolerance_deg,
        "moved": [],
        # Objects whose joints could not be compared (joint-count mismatch
        # between scene and simulator). Blocks promotion like a drift does.
        "joints_unchecked": [],
        # Joint drift, kept apart from root drift in `moved`.
        "joints_moved": [],
        "all": [],
        "promoted": False,
        # --promote was requested but validation refused it.
        "promotion_blocked": False,
        "preserved_keys": [],
        # Where the floor plane was actually placed.
        "ground_plane": None,
        "error": None,
    }

    if not scene_json.exists():
        report["error"] = f"scene JSON not found: {scene_json}"
        _write_report(args.report, report)
        sys.exit(f"ERROR: {report['error']}")

    try:
        # Imported after argument parsing so --help works without OmniGibson.
        from omnigibson.macros import gm
        gm.HEADLESS = args.headless

        import omnigibson as og
        import torch as th
        from omnigibson.utils.constants import JointType
        from simfoundry.utils.ground_plane_utils import (
            apply_ground_plane_info,
            describe as describe_ground_plane,
        )
        from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths

        from scene_io import (
            TargetChanged,
            atomic_write_text,
            file_digest,
            latest_path,
            load_scene,
            merge_settled_scene,
            promote_scene_text,
            save_scene,
        )

        before = load_scene(scene_json)
        registry_before = before["state"]["registry"]["object_registry"]
        # Digest read before physics runs, so a later promotion is compared
        # against the file this settle was based on.
        latest = latest_path(scene_json)
        latest_before = file_digest(latest)

        print(f"Loading {scene_json.name} ...")
        og.launch()
        # usd_path values are relative to the scene JSON, but restore() resolves
        # them against the process cwd, so hand it a pre-resolved document.
        og.sim.restore(scene_files=[load_json_with_absolute_usd_paths(str(scene_json))])

        # restore() does not read `ground_plane_info`. Without this the floor
        # sits at z=0 whatever the scene says, and the settle reports drift the
        # gate itself invented.
        applied = apply_ground_plane_info(before, og.sim.floor_plane)
        report["ground_plane"] = applied
        print(f"Ground:  {describe_ground_plane(applied)}")
        if applied is None and before.get("ground_plane_info"):
            print("WARNING: this scene carries a ground_plane_info but this run has no "
                  "floor plane (use_floor_plane is off), so the settle is not modelling "
                  "the surface the props were placed on.", file=sys.stderr)

        og.sim.play()

        print(f"Settling for {args.steps} steps ...")
        for _ in range(args.steps):
            og.sim.step()

        scene = og.sim.scenes[0]
        print(f"\n{'object':28s} {'root (m)':>9s}  {'joint drift':<34s}")
        print("-" * 76)

        for obj in scene.objects:
            prior = registry_before.get(obj.name, {}).get("root_link", {}).get("pos")
            if prior is None:
                continue
            now, _ = obj.get_position_orientation()
            delta = float(th.linalg.norm(now - th.tensor(prior, dtype=now.dtype)))

            # Joint drift is checked per joint, against the tolerance for its
            # own unit (slides in metres, hinges in radians): gravity can close
            # a drawer while the root link never moves.
            prior_joints = registry_before.get(obj.name, {}).get("joint_pos")
            drifted, unchecked = [], None
            if prior_joints:
                try:
                    now_joints = obj.get_joint_positions()
                    names = list(getattr(obj, "joints", {}) or {})
                    if len(names) != len(prior_joints) or len(now_joints) != len(prior_joints):
                        unchecked = (f"the scene records {len(prior_joints)} joint value(s) "
                                     f"and the simulator reports {len(now_joints)} across "
                                     f"{len(names)} named joint(s)")
                    else:
                        for index, joint_name in enumerate(names):
                            moved_by = abs(float(now_joints[index]) - float(prior_joints[index]))
                            revolute = getattr(obj.joints[joint_name], "joint_type",
                                               None) == JointType.JOINT_REVOLUTE
                            if revolute:
                                degrees = math.degrees(moved_by)
                                over = degrees > args.revolute_tolerance_deg
                                shown, unit = degrees, "deg"
                            else:
                                over = moved_by > args.prismatic_tolerance
                                shown, unit = moved_by, "m"
                            if over:
                                drifted.append({
                                    "joint": joint_name,
                                    "type": "revolute" if revolute else "prismatic",
                                    "delta": round(shown, 4),
                                    "unit": unit,
                                })
                except Exception as e:  # noqa: BLE001 - a report must not fail the run
                    unchecked = f"{type(e).__name__}: {e}"

            entry = {"name": obj.name, "delta": delta, "joints": drifted}
            report["all"].append(entry)
            if delta > args.tolerance:
                report["moved"].append(entry)
            if drifted:
                report["joints_moved"].append(entry)
            if unchecked:
                report["joints_unchecked"].append({"name": obj.name, "reason": unchecked})

            if unchecked:
                joints_column = "unverifiable"
            elif drifted:
                joints_column = ", ".join(
                    f"{d['joint']} {d['delta']}{'°' if d['unit'] == 'deg' else ' m'}"
                    for d in drifted)
            else:
                joints_column = "-" if prior_joints else ""
            flag = "  <-- unsettled" if (delta > args.tolerance or drifted or unchecked) else ""
            print(f"{obj.name:28s} {delta:9.4f}  {joints_column:<34s}{flag}")

        report["ok"] = not (report["moved"] or report["joints_moved"]
                            or report["joints_unchecked"])

        scene.update_initial_file()
        out = save_scene(load_scene(scene_json), scene_json, suffix="settled", promote_latest=False)
        # Use the simulator's own serialization so joint and velocity state
        # stay internally consistent.
        og.sim.save(json_paths=[str(out)])

        # og.sim.save() emits only the keys the simulator owns, so the authored
        # document is folded back around it; ground_plane_info in particular
        # cannot be reconstructed once dropped.
        envelope = load_scene(scene_json)
        settled_doc = load_scene(out)
        # `scene_json_path` restores relative usd_path values: the simulator was
        # handed absolute paths and would otherwise serialize a scene that loads
        # on this machine only.
        merged, preserved = merge_settled_scene(envelope, settled_doc, scene_json_path=out)
        report["preserved_keys"] = preserved
        atomic_write_text(out, json.dumps(merged, indent=4, allow_nan=False))
        report["settled_path"] = str(out)
        if preserved:
            print(f"Preserved {len(preserved)} authored field(s): {', '.join(preserved)}")

        print(f"\nSettled scene written to {out}")
        if report["joints_unchecked"]:
            print(f"\n{len(report['joints_unchecked'])} object(s) whose joints could not be "
                  "checked:")
            for entry in report["joints_unchecked"]:
                print(f"  {entry['name']:28s} {entry['reason']}")
        if report["joints_moved"]:
            print(f"\n{len(report['joints_moved'])} object(s) whose joints moved more than "
                  f"{args.prismatic_tolerance} m (slide) / "
                  f"{args.revolute_tolerance_deg}° (hinge):")
            for entry in report["joints_moved"]:
                for drift in entry["joints"]:
                    unit = "°" if drift["unit"] == "deg" else " m"
                    print(f"  {entry['name']:28s} {drift['joint']} {drift['delta']}{unit}")
            print("  A drawer or door left open in the editor is closed by gravity here. "
                  "Settle it open by giving it something to rest against, or accept it shut.")
        if report["moved"]:
            print(f"\n{len(report['moved'])} object(s) moved more than {args.tolerance} m:")
            for entry in sorted(report["moved"], key=lambda x: -x["delta"]):
                print(f"  {entry['name']}: {entry['delta']:.4f} m")
            print("\nLarge movement usually means an object was floating or intersecting.")

        if args.promote and report["ok"]:
            # Promotion is atomic and compared against the digest read before
            # physics: a settle takes minutes, long enough for another writer
            # to have replaced _latest.
            try:
                promote_scene_text(out.read_text(encoding="utf-8"), out,
                                   expect=latest_before)
                report["promoted"] = True
                print(f"\nPromoted to {latest.name}")
            except TargetChanged as e:
                report["promotion_blocked"] = True
                report["promotion_note"] = str(e)
                print(f"\nNOT promoted: {e} Inspect {out.name} and promote by hand "
                      "if it is still the layout you want.")
        elif args.promote:
            report["promotion_blocked"] = True
            print(
                f"\nNOT promoted: {len(report['moved'])} object(s) moved beyond "
                f"{args.tolerance} m. _latest is unchanged; inspect {out.name} first."
            )
    except Exception as e:  # noqa: BLE001 - the report must survive any failure
        report["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        _write_report(args.report, report)

    # og.shutdown() calls app.close(), which would replace this exit status
    # with Isaac Sim's own. The report is already on disk, so exit directly.
    print(f"\nExiting with status {0 if report['ok'] else 1}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
