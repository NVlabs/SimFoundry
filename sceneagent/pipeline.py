"""Phase-1 conversion run.

Writes the scene directory the paper's agents pass through files: a plan, a
per-step manifest, a ledger of accepted and deferred outputs, a physics record
per object, model-free sisters, a USD physics contract, and the sister grid.
"""

from __future__ import annotations

import json
from pathlib import Path

from sceneagent.adapter import load_document
from sceneagent.alignment import accept_alignment
from sceneagent.evaluation import sister_grid, trial_starts
from sceneagent.physics import build_physics_record
from sceneagent.report import render_report
from sceneagent.sisters import appearance_sister, geometry_sisters, scene_sisters, task_sisters
from sceneagent.skills import plan_steps
from sceneagent.taxonomy import keyword_material


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _proposals_for(obj):
    proposals = [keyword_material(obj["name"])]
    for extra in obj.get("proposals") or []:
        proposals.append(extra)
    return proposals


def _package_body(record, movable):
    body = record["body_type"]
    source = record["body_type_source"]
    if body is None and movable:
        return "dynamic rigid", "export_default_rigid_for_movable"
    if body is None and not movable:
        return "static rigid", "export_default_static_for_fixed"
    return body, source


def run_capture(capture, out_dir):
    out = Path(out_dir)
    ledger = []
    steps = plan_steps()
    _write(out / "plan.json", {"scene_name": capture["scene_name"], "source": capture["source"], "steps": steps})
    ledger.append({"item": "plan", "decision": "accept", "path": "plan.json"})

    _write(out / "steps" / "SCENEAGENT-INGEST-DATA" / "manifest.json", {
        "skill": "SCENEAGENT-INGEST-DATA",
        "input_format": capture["format"],
        "source": capture["source"],
        "n_objects": len(capture["objects"]),
        "limitation": "Phase 1 reads JSON. It does not run COLMAP, splatfacto, or SAM.",
    })
    ledger.append({"item": "ingest", "decision": "accept", "skill": "SCENEAGENT-INGEST-DATA"})

    task = capture["task"]
    manipulated_name = task["manipulated"]
    target_name = task["target"]
    target = next((obj for obj in capture["objects"] if obj["name"] == target_name), None)
    target_inner = None if target is None else target.get("inner_size")

    physics_records = []
    sister_rows = []
    package_objects = []
    for obj in capture["objects"]:
        record = build_physics_record(
            obj["name"],
            _proposals_for(obj),
            numerical=obj.get("numerical"),
            body_type=obj.get("body_type"),
            collision=obj.get("collision"),
        )
        physics_records.append(record)
        _write(out / "objects" / obj["name"] / "physics_prior.json", record)
        ledger.append({
            "item": f"physics:{obj['name']}",
            "decision": "accept",
            "skill": "SCENEAGENT-RUN-STEP",
            "material_class": record["material_class"],
        })

        alignment = obj.get("alignment")
        if alignment:
            decision = accept_alignment(alignment.get("residuals"), alignment.get("thresholds"))
            _write(out / "objects" / obj["name"] / "alignment.json", decision)
            ledger.append({
                "item": f"alignment:{obj['name']}",
                "decision": decision["decision"],
                "skill": "SCENEAGENT-ALIGN-SCENE",
                "accepted": decision["accepted"],
            })

        mesh = obj.get("mesh")
        if mesh and mesh.get("kind") == "box":
            eval_cfg = capture.get("evaluation") or {}
            scales = eval_cfg.get("length_scales")
            geom = geometry_sisters(
                mesh["size"],
                target_inner=target_inner if obj["name"] == manipulated_name else None,
                length_scales=scales if scales is not None else (0.95, 1.05),
            )
            appearance = obj.get("appearance")
            if appearance:
                geom.append(appearance_sister(
                    mesh["size"],
                    appearance["hue_degrees"],
                    appearance["mottle"],
                    appearance["speckle"],
                    appearance["noise"],
                    appearance.get("name") or f"{obj['name']} texture",
                ))
            _write(out / "objects" / obj["name"] / "sisters.json", geom)
            if obj["name"] == manipulated_name:
                sister_rows = geom
            ledger.append({
                "item": f"sisters:{obj['name']}",
                "decision": "accept",
                "skill": "SCENEAGENT-OBJECT-SISTERS",
                "n": len(geom),
                "n_rejected": sum(1 for row in geom if not row["accepted"]),
            })
        else:
            ledger.append({
                "item": f"sisters:{obj['name']}",
                "decision": "deferred",
                "skill": "SCENEAGENT-OBJECT-SISTERS",
                "reason": "No box mesh on this object. Names from a SimFoundry scene JSON do not include one.",
            })

        body, body_source = _package_body(record, bool(obj.get("movable", True)))
        package_objects.append({
            "name": obj["name"],
            "rigid_body": body in ("dynamic rigid", "static rigid", "kinematic"),
            "mass_kg": record["mass_kg"],
            "friction_static": record["friction_static"],
            "friction_dynamic": record["friction_dynamic"],
            "restitution": record["restitution"],
            "density_kg_m3": record["density_kg_m3"],
            "body_type": body,
            "body_type_source": body_source,
            "collision_approximation": record["collision_primitive"],
            "material_class": record["material_class"],
            "rigidity": record["rigidity"],
            "deformability": record["deformability"],
            "unpublished_class_lookups": record["unpublished_class_lookups"],
        })

    layouts = scene_sisters(
        task.get("anchor_xy_m") or (0.0, 0.0),
        float(task.get("gap_m") or 0.05),
    )
    instructions = task_sisters(task["instruction"], manipulated_name, target_name)
    _write(out / "scene_sisters.json", layouts)
    _write(out / "task_sisters.json", instructions)
    ledger.append({"item": "scene_sisters", "decision": "accept", "skill": "SCENEAGENT-SCENE-SISTERS", "n": len(layouts)})
    ledger.append({"item": "task_sisters", "decision": "accept", "skill": "SCENEAGENT-TASK-SISTERS", "n": len(instructions)})

    _write(out / "package" / "usd_physics_contract.json", {
        "format": "sceneagent_usd_physics_contract_v1",
        "usd_written": False,
        "targets": ["Isaac Lab", "MuJoCo", "Unreal Engine"],
        "note": (
            "OpenUSD physics attributes named in SceneAgent section 3.7. "
            "Phase 1 writes the contract. It does not emit a .usd or .usdz file, "
            "and it does not load Isaac Lab, MuJoCo, or Unreal."
        ),
        "objects": package_objects,
    })
    ledger.append({
        "item": "package",
        "decision": "accept",
        "skill": "SCENEAGENT-PACKAGE-DATA-PROCESSED",
        "usd_written": False,
    })

    grid = None
    if sister_rows:
        cfg = capture.get("evaluation") or {}
        n_trials = int(cfg.get("n_trials") or 10)
        starts = trial_starts(
            n_trials,
            float(cfg.get("jitter_half_extent_m") or 0.05),
            int(cfg.get("seed") or 0),
        )
        grid = sister_grid(sister_rows, starts, float(cfg.get("reach_radius_m") or 0.03))
        grid["scene_name"] = capture["scene_name"]
        grid["instruction"] = task["instruction"]
        _write(out / "evaluation" / "sister_grid.json", grid)
        report = render_report(capture["scene_name"], grid)
        _write(out / "evaluation" / "report.html", report)
        ledger.append({
            "item": "policy_eval",
            "decision": "accept",
            "skill": "SCENEAGENT-POLICY-EVAL",
            "policy": "scripted_geometric_place",
            "learned_policy_ran": False,
        })
    else:
        ledger.append({
            "item": "policy_eval",
            "decision": "deferred",
            "skill": "SCENEAGENT-POLICY-EVAL",
            "reason": "The manipulated object has no mesh sisters to score.",
        })

    for step in steps:
        if step["status"] == "not_implemented":
            ledger.append({
                "item": step["skill"],
                "decision": "deferred",
                "skill": step["skill"],
                "reason": step["reason"],
            })

    _write(out / "ledger.json", {"scene_name": capture["scene_name"], "entries": ledger})
    summary = {
        "scene_name": capture["scene_name"],
        "out_dir": str(out),
        "n_objects": len(physics_records),
        "n_sister_rows": len(sister_rows),
        "grid_written": grid is not None,
        "materials": {record["object"]: record["material_class"] for record in physics_records},
    }
    _write(out / "summary.json", summary)
    return summary


def run_path(input_path, out_dir):
    return run_capture(load_document(input_path), out_dir)
