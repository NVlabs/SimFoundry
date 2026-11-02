"""Section 3.4 vote, audit prompt, part prompt, and two open tools.

Equation (2) is a weighted vote of the physics prior, the VLM, and keyword
priors. Unknown-class weights are scaled by 0.4. The paper says the other
weights are source confidence, or a fixed value for keyword priors, and it
does not publish those numbers. The keyword fixed weight is
``taxonomy.DEFAULT_KEYWORD_WEIGHT``, which this tree labels as its own.

CubePart's Quebec crash was a signature mismatch. This module only chooses
the call shape. Articulate Anything is not downloaded here.
"""

from __future__ import annotations

import json
from pathlib import Path

from sceneagent.agent_loop import parse_material_class
from sceneagent.gemini_client import PAPER_MODEL_ID
from sceneagent.physics import UNKNOWN_WEIGHT_SCALE, vote_material
from sceneagent.taxonomy import (
    DEFAULT_KEYWORD_WEIGHT,
    MATERIAL_CLASSES,
    keyword_material,
    require_material,
)

# diffusers 0.39: QwenEmbedRope.forward(video_fhw, device)
# The CubePart repo calls pos_embed(img_shapes, txt_seq_lens, device=...).
# Passing the repo arguments into the diffusers signature binds img_shapes to
# video_fhw and then passes device twice, which is the TypeError seen on Quebec.
DIFFUSERS_039_FORWARD = "QwenEmbedRope.forward(video_fhw, device)"
REPO_POS_EMBED_CALL = "pos_embed(img_shapes, txt_seq_lens, device=...)"
ROPE_CRASH = "TypeError: QwenEmbedRope.forward() got multiple values for argument 'device'"

# Listing from the Quebec zip. The download is not started by this module.
ARTICULATE_UNCOMPRESSED_BYTES = 15_616_309_082
ARTICULATE_SPARE_BYTES = 8 * (1024 ** 3)

HISTORICAL_VLM = (
    {"model_id": "gemini-2.5-flash", "http_status": 404},
    {"model_id": "gemini-3.8-flash", "material_class": "wood", "http_status": 200},
)


def equation2_vote(object_name, vlm_class=None, physics_prior=None, vlm_confidence=None):
    """Vote with equation (2). A missing prior is absent and is not the MLP.

    ``vlm_class`` is the existing class string. ``physics_prior`` is a mapping
    with ``class`` and ``confidence``, or None. ``claimed_mlp`` stays false
    unless the caller sets ``from_mlp`` on a prior this process did not train.
    """
    sources = []
    voters = []
    if physics_prior is None:
        sources.append({
            "source": "physics_prior",
            "status": "absent",
            "class": None,
            "weight": None,
            "claimed_mlp": False,
            "note": "No per-Gaussian MLP output. The class is not the MLP.",
        })
    else:
        material = require_material(physics_prior["class"])
        confidence = float(physics_prior["confidence"])
        from_mlp = bool(physics_prior.get("from_mlp"))
        sources.append({
            "source": "physics_prior",
            "status": "present",
            "class": material,
            "weight": confidence,
            "weight_basis": "caller_confidence",
            "claimed_mlp": from_mlp,
        })
        voters.append({"source": "physics_prior", "class": material, "weight": confidence})
    keyword = keyword_material(object_name, weight=DEFAULT_KEYWORD_WEIGHT)
    keyword["weight_basis"] = (
        "taxonomy.DEFAULT_KEYWORD_WEIGHT; the paper says keyword priors use a "
        "fixed weight and does not publish the number"
    )
    keyword["status"] = "present"
    sources.append(keyword)
    voters.append({"source": "keyword", "class": keyword["class"], "weight": keyword["weight"]})
    if vlm_class is None:
        sources.append({
            "source": "vlm",
            "status": "absent",
            "class": None,
            "weight": None,
            "note": "No VLM class string was supplied.",
        })
    else:
        material = require_material(vlm_class)
        if vlm_confidence is None:
            weight = 1.0
            basis = "fixed_1_because_the_paper_does_not_publish_a_VLM_confidence"
        else:
            weight = float(vlm_confidence)
            basis = "caller_confidence"
        sources.append({
            "source": "vlm",
            "status": "present",
            "class": material,
            "weight": weight,
            "weight_basis": basis,
        })
        voters.append({"source": "vlm", "class": material, "weight": weight})
    winner, scores = vote_material(voters)
    return {
        "equation": "2",
        "unknown_scale": UNKNOWN_WEIGHT_SCALE,
        "object": object_name,
        "material_class": winner,
        "scores": scores,
        "sources": sources,
        "physics_prior": "absent" if physics_prior is None else "present",
        "claimed_mlp": False if physics_prior is None else bool(physics_prior.get("from_mlp")),
        "class_is_mlp": bool(physics_prior and physics_prior.get("from_mlp") and winner is not None),
    }


def audit_prompt(object_name, provisional_class, sources):
    """Text for the second VLM pass. This string does not call the network."""
    lines = [
        "Section 3.4 VLM audit, second pass.",
        f"Object: {object_name}.",
        f"Provisional class from equation (2): {provisional_class}.",
        "Sources:",
    ]
    for source in sources:
        lines.append(
            f"- {source.get('source')}: status {source.get('status')}, "
            f"class {source.get('class')}, weight {source.get('weight')}"
        )
    lines.append(
        "Reply with exactly one published material class, or say the provisional class stands."
    )
    lines.append(
        "The earlier gemini-2.5-flash HTTP 404 and the gemini-3.8-flash class wood "
        "are historical and are not this pass."
    )
    return "\n".join(lines)


def request_audit(object_name, provisional_class, sources, api_key=None, live=False):
    """Return the audit prompt. Call Gemini only when ``live`` is set and a key is present.

    One ``generate_flash`` text call is the cheap call. The default is not to
    call. A live reply is stored separately from the historical 404 and wood class.
    """
    prompt = audit_prompt(object_name, provisional_class, sources)
    body = {
        "prompt": prompt,
        "called": False,
        "model_id": None,
        "http_status": None,
        "text": "",
        "decision": "pending",
        "historical": [dict(item) for item in HISTORICAL_VLM],
        "paper_model_id": PAPER_MODEL_ID,
    }
    if not live or not api_key:
        return body
    from sceneagent.gemini_client import GeminiClient

    reply = GeminiClient(api_key).generate_flash(prompt)
    body["called"] = True
    body["model_id"] = reply.get("model_id")
    body["http_status"] = reply.get("http_status")
    body["text"] = reply.get("text") or ""
    body["decision"] = "replied" if reply.get("http_status") == 200 and body["text"] else "pending"
    body["used_paper_model"] = bool(reply.get("used_paper_model"))
    return body


def part_inspection_prompt(parts):
    """Ask whether hidden geometry is missing. Does not call a model."""
    listed = ", ".join(str(part) for part in parts) if parts else "(no parts)"
    return (
        "Section 3.4 part inspection. A VLM looks at these PartSAM parts and "
        f"decides whether hidden geometry is missing: {listed}. "
        "For each part, say whether hidden geometry is missing, such as a tip "
        "under a cap. Reply yes only if completion is needed. "
        "CubePart must not run unless completion is needed."
    )


def inspect_parts(parts, model_reply=None):
    """Write the part prompt. Without a model reply the decision stays pending.

    CubePart is requested only when the reply says completion is needed.
    """
    prompt = part_inspection_prompt(parts)
    if not model_reply:
        return {
            "prompt": prompt,
            "decision": "pending",
            "completion_needed": None,
            "call_cubepart": False,
            "inspected": False,
            "model_id": None,
        }
    needed = _completion_needed(model_reply)
    return {
        "prompt": prompt,
        "decision": "replied",
        "completion_needed": needed,
        "call_cubepart": needed is True,
        "inspected": True,
        "model_reply": model_reply,
    }


def _completion_needed(text):
    raw = (text or "").strip()
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and "completion_needed" in payload:
            return bool(payload["completion_needed"])
    lowered = raw.lower()
    if "completion is needed" in lowered or lowered.startswith("yes"):
        return True
    if "not needed" in lowered or "no missing" in lowered or lowered.startswith("no"):
        return False
    return None


def material_properties_prompt(object_name, notes=""):
    """Ask for the section 3.4 properties the paper names.

    The reply is a class plus numeric friction, mass, and density. This prompt
    does not call the network.
    """
    note = f" {notes}" if notes else ""
    return (
        "Section 3.4 physical properties. "
        f"Object: {object_name}.{note} "
        "Reply with one JSON object and no other text. "
        "Keys: material_class, friction, mass_kg, density_kg_m3. "
        f"material_class must be one of: {', '.join(MATERIAL_CLASSES)}. "
        "friction is one numeric coefficient. mass_kg is kilograms. "
        "density_kg_m3 is kilograms per cubic meter. "
        "Do not sign the reply as a coding agent. "
        "The historical gemini-2.5-flash HTTP 404 and the gemini-3.8-flash class wood "
        "are previous calls, not this reply."
    )


def parse_material_properties(text):
    """Pull a class and three numbers out of a VLM reply. Missing stays None."""
    raw = text or ""
    payload = {}
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            loaded = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            payload = loaded

    def number(*keys):
        for key in keys:
            if key not in payload:
                continue
            try:
                value = float(payload[key])
            except (TypeError, ValueError):
                continue
            return value
        return None

    material = payload.get("material_class")
    if material not in MATERIAL_CLASSES:
        material = parse_material_class(str(material or raw))
    friction = number("friction", "friction_static")
    mass = number("mass_kg", "mass")
    density = number("density_kg_m3", "density")
    return {
        "material_class": material,
        "friction": friction,
        "mass_kg": mass,
        "density_kg_m3": density,
        "complete": material is not None and None not in (friction, mass, density),
    }


def request_material_properties(object_name, notes="", api_key=None, image_bytes=None, live=False):
    """Ask Gemini for class, friction, mass, and density. Default is not to call.

    A 404 on gemini-2.5-flash stays the paper attempt. gemini-3.8-flash, when
    chosen by the client, is recorded as the fallback and is not labeled 2.5.
    """
    prompt = material_properties_prompt(object_name, notes)
    body = {
        "prompt": prompt,
        "called": False,
        "model_id": None,
        "http_status": None,
        "paper_model_id": PAPER_MODEL_ID,
        "used_paper_model": False,
        "model_role": None,
        "fallback_model_id": None,
        "text": "",
        "properties": None,
        "historical": [dict(item) for item in HISTORICAL_VLM],
    }
    if not live or not api_key:
        return body
    from sceneagent.gemini_client import GeminiClient

    reply = GeminiClient(api_key).generate_flash(prompt, image_bytes=image_bytes)
    body["called"] = True
    body["model_id"] = reply.get("model_id")
    body["http_status"] = reply.get("http_status")
    body["paper_http_status"] = reply.get("paper_http_status")
    body["used_paper_model"] = bool(reply.get("used_paper_model"))
    body["fallback_model_id"] = reply.get("fallback_model_id")
    body["model_role"] = "paper" if reply.get("used_paper_model") else "fallback"
    body["text"] = reply.get("text") or ""
    body["properties"] = parse_material_properties(body["text"])
    if api_key:
        body["prompt"] = body["prompt"].replace(api_key, "[redacted]")
        body["text"] = body["text"].replace(api_key, "[redacted]")
    return body


def partsam_summary_prompt(masks, parts_after_nms, parts):
    """Text summary of PartSAM parts. Does not call CubePart or a model."""
    listed = []
    for part in parts or []:
        if isinstance(part, dict):
            listed.append(
                f"index {part.get('index')} rgb {part.get('rgb')} faces {part.get('faces')}"
            )
        else:
            listed.append(str(part))
    body = (
        f"PartSAM summary: {int(masks)} masks, {int(parts_after_nms)} parts after NMS. "
        + ("; ".join(listed) if listed else "no per-part colors")
    )
    return part_inspection_prompt([body])


def choose_qwen_embed_rope_call(parameter_names):
    """Pick the call shape from the live ``forward`` parameter names.

    ``('video_fhw', 'device')`` is diffusers 0.39.
    ``('img_shapes', 'txt_seq_lens', 'device')`` is the CubePart repo call.
    """
    names = tuple(parameter_names)
    if names and names[0] == "video_fhw":
        return {
            "signature": DIFFUSERS_039_FORWARD,
            "positional": ("video_fhw",),
            "keyword": ("device",),
            "call": "forward(video_fhw, device=device)",
        }
    if "img_shapes" in names and "txt_seq_lens" in names:
        return {
            "signature": REPO_POS_EMBED_CALL,
            "positional": ("img_shapes", "txt_seq_lens"),
            "keyword": ("device",),
            "call": "pos_embed(img_shapes, txt_seq_lens, device=device)",
        }
    raise ValueError(f"unrecognized QwenEmbedRope parameters: {names}")


def cubepart_status(part_mesh_path=None):
    """Fixed only when a part mesh file exists. The RoPE note is not a mesh."""
    mesh = Path(part_mesh_path) if part_mesh_path else None
    has_mesh = bool(mesh and mesh.is_file() and mesh.stat().st_size > 0)
    return {
        "status": "fixed" if has_mesh else "still_open",
        "part_mesh": str(mesh) if has_mesh else None,
        "gpu_rerun": False,
        "compat_note": (
            f"{DIFFUSERS_039_FORWARD} versus the repo call {REPO_POS_EMBED_CALL}. "
            f"The Quebec failure was {ROPE_CRASH}."
        ),
    }


def articulate_download_allowed(gpu_up, free_bytes, uncompressed_bytes=ARTICULATE_UNCOMPRESSED_BYTES):
    """True only when a GPU is up and free space exceeds the extract plus 8 GiB."""
    if not gpu_up:
        return False
    return int(free_bytes) > int(uncompressed_bytes) + ARTICULATE_SPARE_BYTES


def articulate_row(gpu_up, free_bytes, urdf_path=None):
    """Leave the row open until a URDF file exists. Does not download the zip."""
    urdf = Path(urdf_path) if urdf_path else None
    has_urdf = bool(urdf and urdf.is_file() and urdf.stat().st_size > 0)
    allowed = articulate_download_allowed(gpu_up, free_bytes)
    return {
        "status": "fixed" if has_urdf else "still_open",
        "download_started": False,
        "download_allowed": allowed,
        "urdf": str(urdf) if has_urdf else None,
        "reason": "URDF exists." if has_urdf else "No URDF. The Articulate Anything download was not started.",
    }
