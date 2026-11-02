"""Live section 3.1 loop over the Fruits artifacts already on disk.

Finished stages are inputs. This runner does not retrain COLMAP or the splat.
"""

from __future__ import annotations

import json
from pathlib import Path

from sceneagent.agent_loop import (
    ALLOWED_STEPS,
    default_plan,
    parse_material_class,
    parse_plan,
    run_agent_loop,
)
from sceneagent.gemini_client import PAPER_MODEL_ID
from sceneagent.mjcf_contract import write_physics_mjcf
from sceneagent.taxonomy import MATERIAL_CLASSES


# The model is asked to plan from the input state. The prompt does not
# contain a JSON object to copy and does not name a figure-5 harness.
PLAN_PROMPT = (
    "Section 3.1 of SceneAgent. You are planning the remaining conversion "
    "from the input state, not copying a script. The capture already has "
    "COLMAP poses, a splatfacto render, SAM masks, an OpenCLIP vector, a "
    "DINOv2 vector, association JSON, and a USDA file. Finished capture "
    "stages are inputs. Say which remaining processing steps you would run "
    "and why, in your own words. If you have no remaining steps, say that "
    "the input state needs nothing further."
)

VOTE_PROMPT = (
    "Section 3.4 material vote. Reply with exactly one class and no other words: "
    + ", ".join(MATERIAL_CLASSES)
)


def order_plan(model_steps):
    """Keep the model's order. Unknown ids are already dropped.

    An empty reply uses ``default_plan()``. That list is this tree's
    remaining local steps. It is not a call to a figure-5 coding harness.
    """
    chosen = [step for step in model_steps if step in ALLOWED_STEPS]
    if not chosen:
        return default_plan()
    return chosen


def load_mujoco_xml(path, steps=30):
    """Load an XML scene and advance it. USDA fails here; MJCF can succeed."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return {
        "ok": True,
        "artifact": str(path),
        "reason": "",
        "detail": {
            "nbody": int(model.nbody),
            "nq": int(model.nq),
            "nstep": steps,
            "qpos0": float(data.qpos[0]) if model.nq else None,
        },
    }


def _failed_load(path, exc):
    return {
        "ok": False,
        "artifact": str(path),
        "reason": f"{type(exc).__name__}: {exc}",
        "detail": {},
    }


def make_tools(client, paths, disk_free_gb, loader=load_mujoco_xml, start_sam3d=None):
    """Tools the loop may call.

    The loop's repair hook reruns whichever ``<step>_repair`` tool is
    registered. This runner registers a repair only for MuJoCo, because that
    repair writes an MJCF from the physics contract. It does not invent a
    pose for alignment or any other step.
    """

    def material_vote(_context):
        image = Path(paths["frame"]).read_bytes()
        result = client.generate_flash(VOTE_PROMPT, image_bytes=image)
        material = parse_material_class(result.get("text") or "")
        vote_path = Path(paths["vote"])
        vote_path.parent.mkdir(parents=True, exist_ok=True)
        vote_path.write_text(json.dumps({
            "paper_model_id": result.get("paper_model_id") or PAPER_MODEL_ID,
            "model_id": result.get("model_id"),
            "http_status": result.get("http_status"),
            "paper_http_status": result.get("paper_http_status"),
            "used_paper_model": result.get("used_paper_model"),
            "material_class": material,
            "text": result.get("text") or "",
            "error_body": result.get("error_body") or result.get("paper_error_body") or "",
        }, indent=2) + "\n", encoding="utf-8")
        ok = material is not None and result.get("http_status") == 200
        return {
            "ok": ok,
            "artifact": str(vote_path),
            "reason": "" if ok else (result.get("error_body") or "no published class in the reply"),
            "detail": {
                "model_id": result.get("model_id"),
                "paper_model_id": result.get("paper_model_id") or PAPER_MODEL_ID,
                "http_status": result.get("http_status"),
                "paper_http_status": result.get("paper_http_status"),
                "material_class": material,
                "used_paper_model": bool(result.get("used_paper_model")),
            },
        }

    def mujoco_load(_context):
        try:
            return loader(paths["usda"])
        except Exception as exc:
            return _failed_load(paths["usda"], exc)

    def mujoco_load_repair(_context):
        xml_path = write_physics_mjcf(
            json.loads(Path(paths["contract"]).read_text(encoding="utf-8")),
            paths["mjcf"],
        )
        try:
            loaded = loader(xml_path)
        except Exception as exc:
            return _failed_load(xml_path, exc)
        loaded["artifact"] = str(xml_path)
        report = Path(paths["mujoco_report"])
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(loaded["detail"], indent=2) + "\n", encoding="utf-8")
        return loaded

    def sam3d(_context):
        if disk_free_gb < 25:
            return {
                "ok": False,
                "artifact": None,
                "reason": f"live free space is {disk_free_gb:.1f} GB, under the 25 GB SAM 3D floor",
                "detail": {},
            }
        needed = paths.get("sam3d_bytes")
        free_bytes = disk_free_gb * (1024 ** 3)
        if needed and free_bytes < needed + 5 * (1024 ** 3):
            return {
                "ok": False,
                "artifact": None,
                "reason": (
                    f"SAM 3D weights are {needed} bytes and live free space is "
                    f"{disk_free_gb:.1f} GB, which cannot hold them with 5 GB spare"
                ),
                "detail": {"sam3d_bytes": needed, "disk_free_gb": disk_free_gb},
            }
        if start_sam3d is None:
            return {
                "ok": False,
                "artifact": None,
                "reason": "SAM 3D download was not started",
                "detail": {},
            }
        started = start_sam3d(paths["sam3d_dir"])
        mesh = started.get("mesh")
        if mesh and Path(mesh).is_file() and Path(mesh).stat().st_size > 0:
            return {"ok": True, "artifact": str(mesh), "reason": "", "detail": started}
        return {
            "ok": False,
            "artifact": started.get("log"),
            "reason": started.get("reason") or "SAM 3D download started; no mesh file yet",
            "detail": started,
        }

    def partsam(_context):
        mesh = paths.get("sam3d_mesh")
        if not mesh or not Path(mesh).is_file():
            return {
                "ok": False,
                "artifact": None,
                "reason": "PartSAM was not run: no SAM 3D mesh file",
                "detail": {},
            }
        return {
            "ok": False,
            "artifact": None,
            "reason": "PartSAM weights were not run in this loop",
            "detail": {},
        }

    def isaac_inspector(_context):
        return {
            "ok": False,
            "artifact": None,
            "reason": "The Isaac inspector did not open a Fruits scene. The empty-stage smoke is a different block.",
            "detail": {},
        }

    return {
        "material_vote": material_vote,
        "mujoco_load": mujoco_load,
        "mujoco_load_repair": mujoco_load_repair,
        "sam3d": sam3d,
        "partsam": partsam,
        "isaac_inspector": isaac_inspector,
    }


def plan_with_client(client):
    """Ask the model to plan from the input state.

    ``parse_plan`` drops ids that are not in ``ALLOWED_STEPS``. If that leaves
    nothing, ``order_plan`` falls back to ``default_plan()``. The fallback is
    local Python. It does not invent a figure-5 harness reply.
    """
    result = client.generate_flash(PLAN_PROMPT)
    text = result.get("text") or ""
    parsed = parse_plan(text)
    used_fallback = not parsed
    return {
        "steps": order_plan(parsed),
        "parsed_steps": parsed,
        "used_fallback": used_fallback,
        "fallback": "default_plan" if used_fallback else None,
        "model_id": result.get("model_id"),
        "http_status": result.get("http_status"),
        "used_paper_model": bool(result.get("used_paper_model")),
        "paper_model_id": result.get("paper_model_id") or PAPER_MODEL_ID,
        "paper_http_status": result.get("paper_http_status"),
        "error_body": result.get("error_body") or result.get("paper_error_body") or "",
        "text": text,
    }


def run_fruits(out_dir, client, paths, disk_free_gb, loader=load_mujoco_xml, start_sam3d=None):
    plan_info = plan_with_client(client)
    tools = make_tools(client, paths, disk_free_gb, loader=loader, start_sam3d=start_sam3d)
    summary = run_agent_loop(out_dir, tools, plan_info["steps"])
    summary["plan_model"] = {
        key: plan_info[key]
        for key in (
            "model_id",
            "http_status",
            "used_paper_model",
            "paper_model_id",
            "paper_http_status",
        )
    }
    return summary
