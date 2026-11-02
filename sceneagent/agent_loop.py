"""Section 3.1 agent loop.

An agent writes a plan, runs one tool, writes a manifest for that step, and
appends a ledger entry that accepts or rejects the artifact. On rejection it
repairs and reruns. The phase-1 pipeline ledger is only the file contract.
This loop is the part that inspects a result and chooses the next tool.
"""

from __future__ import annotations

import json
from pathlib import Path

from sceneagent.taxonomy import MATERIAL_CLASSES

# Remaining Harvard blocks this loop is allowed to schedule. Finished capture
# stages (COLMAP, splatfacto, SAM masks, CLIP, DINOv2, association, USDA) are
# inputs, not steps to rerun.
ALLOWED_STEPS = (
    "material_vote",
    "mujoco_load",
    "sam3d",
    "partsam",
    "isaac_inspector",
    "icp",
    "isaac_mesh",
    "mujoco_mesh",
    "cubepart",
    "articulate",
)

STEP_SECTION = {
    "material_vote": "3.4",
    "mujoco_load": "3.7",
    "sam3d": "3.3",
    "partsam": "3.4",
    "isaac_inspector": "3.1",
    "icp": "3.3",
    "isaac_mesh": "3.7",
    "mujoco_mesh": "3.7",
    "cubepart": "3.4",
    "articulate": "3.4",
}

# Figure 7 skill for that step. Physics, parts, and the step runner share
# SCENEAGENT-RUN-STEP because that is the skill figure 7 prints on those
# cards. Alignment, Isaac inspection, and articulation have their own ids.
# Figure 7 does not name a MuJoCo skill, so a MuJoCo load stays on the step
# runner rather than being labeled as the Isaac inspector.
STEP_SKILL = {
    "material_vote": "SCENEAGENT-RUN-STEP",
    "mujoco_load": "SCENEAGENT-RUN-STEP",
    "sam3d": "SCENEAGENT-RUN-STEP",
    "partsam": "SCENEAGENT-RUN-STEP",
    "isaac_inspector": "SCENEAGENT-INSPECT-ISAAC",
    "icp": "SCENEAGENT-ALIGN-SCENE",
    "isaac_mesh": "SCENEAGENT-INSPECT-ISAAC",
    "mujoco_mesh": "SCENEAGENT-RUN-STEP",
    "cubepart": "SCENEAGENT-RUN-STEP",
    "articulate": "SCENEAGENT-ARTICULATE-NATIVE-ASSETS",
}

# Figure 5 products. This loop does not call them.
EXTERNAL_HARNESSES = (
    "Codex",
    "Claude Code",
    "Antigravity",
    "Muse",
    "Stirrup",
    "Grok CLI",
)


def parse_material_class(text):
    """Return a published class named in the reply, or None."""
    lowered = (text or "").lower()
    for name in sorted(MATERIAL_CLASSES, key=len, reverse=True):
        if name in lowered:
            return name
    return None


def parse_plan(text):
    """Pull step ids out of a model reply. Unknown ids are dropped."""
    raw = text or ""
    start = raw.find("{")
    end = raw.rfind("}")
    steps = []
    if start >= 0 and end > start:
        try:
            payload = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            payload = {}
        for item in payload.get("steps") or []:
            if item in ALLOWED_STEPS and item not in steps:
                steps.append(item)
    if not steps:
        for item in ALLOWED_STEPS:
            if item in raw and item not in steps:
                steps.append(item)
    return steps


def default_plan():
    return list(ALLOWED_STEPS)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _manifest(out, step_id, attempt, result):
    path = out / "manifests" / f"{step_id}-{attempt}.json"
    body = {
        "step": step_id,
        "section": STEP_SECTION[step_id],
        "attempt": attempt,
        "ok": bool(result.get("ok")),
        "artifact": result.get("artifact"),
        "reason": result.get("reason") or "",
        "detail": result.get("detail") or {},
    }
    _write(path, body)
    return str(path)


def _entry(step_id, attempt, result, repair):
    detail = result.get("detail") or {}
    entry = {
        "item": step_id,
        "section": STEP_SECTION[step_id],
        "decision": "accept" if result.get("ok") else "reject",
        "attempt": attempt,
        "repair": repair,
        "skill": STEP_SKILL[step_id],
        "artifact": result.get("artifact"),
        "reason": result.get("reason") or "",
    }
    for key in (
        "model_id",
        "paper_model_id",
        "http_status",
        "paper_http_status",
        "material_class",
        "used_paper_model",
    ):
        if key in detail:
            entry[key] = detail[key]
    return entry


def append_step(out_dir, step_id, result, repair=False):
    """Record one later tool result without rewriting the plan or earlier entries.

    Section 3.1: a manifest for this attempt, then a ledger line that accepts
    only when the tool result is ok. A rejected stage stays a reject.
    """
    if step_id not in STEP_SECTION:
        raise KeyError(step_id)
    out = Path(out_dir)
    ledger_path = out / "ledger.json"
    if ledger_path.is_file():
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    else:
        payload = {"section": "3.1", "entries": []}
    entries = payload.setdefault("entries", [])
    prior = [entry["attempt"] for entry in entries if entry.get("item") == step_id]
    attempt = max(prior) + 1 if prior else 1
    _manifest(out, step_id, attempt, result)
    entry = _entry(step_id, attempt, result, repair=repair)
    entries.append(entry)
    _write(ledger_path, payload)
    return entry


def harness_status():
    """Figure 5 names six coding harnesses. None of them is called here."""
    return {
        "status": "still_open",
        "harnesses": list(EXTERNAL_HARNESSES),
        "called": [],
        "reason": (
            "Codex, Claude Code, Antigravity, Muse, Stirrup, and Grok CLI are "
            "external products. This tree does not call them and does not fake a reply."
        ),
    }


def _without_invented_pose(result):
    """Keep a pose only when the repair tool supplied one."""
    produced = dict(result or {})
    detail = dict(produced.get("detail") or {})
    if "pose" not in detail:
        detail["pose"] = None
        detail["pose_note"] = "No pose was supplied by the repair tool. None was invented."
    produced["detail"] = detail
    return produced


def consider_repair(step_id, result, tools, context, calls=None):
    """On rejection, rerun ``<step>_repair`` when that tool is registered.

    A missing repair tool leaves the rejection as the only record. This
    function does not create a pose.
    """
    produced = [(result, False)]
    if result.get("ok"):
        return produced
    name = f"{step_id}_repair"
    repair = (tools or {}).get(name)
    if repair is None:
        return produced
    if calls is not None:
        calls.append(name)
    repaired = repair(dict(context or {}))
    produced.append((_without_invented_pose(repaired), True))
    return produced


def record_and_repair(out_dir, step_id, result, tools=None, context=None):
    """Record a step and, on rejection, rerun a registered repair tool.

    Does not invent a pose when the repair tool is missing or returns none.
    """
    calls = []
    entries = []
    for item, is_repair in consider_repair(step_id, result, tools, context, calls):
        entries.append(append_step(out_dir, step_id, item, repair=is_repair))
    return {"entries": entries, "calls": calls}


def run_agent_loop(out_dir, tools, plan, context=None):
    """Run each planned step once. A rejection with a repair tool runs that tool next.

    ``tools`` maps a step id to ``callable(context) -> result`` and may also
    map ``\"<step>_repair\"`` to the repair callable. ``plan`` is an ordered
    list of step ids. Section 3.1: one tool, one manifest, one ledger entry.
    """
    out = Path(out_dir)
    context = dict(context or {})
    chosen = [step for step in plan if step in ALLOWED_STEPS]
    if not chosen:
        chosen = default_plan()
    _write(out / "plan.json", {
        "section": "3.1",
        "steps": [
            {"id": step, "section": STEP_SECTION[step]}
            for step in chosen
        ],
    })
    ledger = []
    calls = []
    for step_id in chosen:
        tool = tools.get(step_id)
        if tool is None:
            result = {"ok": False, "artifact": None, "reason": "no tool registered", "detail": {}}
        else:
            calls.append(step_id)
            result = tool(context)
        for index, (item, is_repair) in enumerate(
            consider_repair(step_id, result, tools, context, calls),
            start=1,
        ):
            _manifest(out, step_id, index, item)
            ledger.append(_entry(step_id, index, item, repair=is_repair))
    _write(out / "ledger.json", {"section": "3.1", "entries": ledger})
    return {"out_dir": str(out), "calls": calls, "entries": ledger}
