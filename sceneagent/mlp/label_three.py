"""Label the three checked meshes with one gemini-3.8-flash prompt.

The schema is the seven fields in ``labels.REQUIRED_LABEL_KEYS``. Class
strings are the names in ``taxonomy.py``. Mass and density are logged once
by ``parse_label``. Slots 28–32 are not requested and are not stored.

Gemini 3.8 Flash does not accept thinking off or ``thinkingLevel``
``minimal``. This script sends ``thinkingLevel`` ``low``, the lowest level
that model accepts, and temperature 0. It does not call ``gemini-2.5-flash``.

Records are written beside each object as ``label.json`` and together as
``data/mlp/label_three.jsonl``.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from sceneagent.gemini_client import API_ROOT, redact
from sceneagent.mlp.labels import REQUIRED_LABEL_KEYS, _BESIDE_KEYS, parse_label
from sceneagent.taxonomy import BODY_TYPES, COLLISION_PRIMITIVES, MATERIAL_CLASSES

MODEL_ID = "gemini-3.8-flash"
TEMPERATURE = 0
THINKING_LEVEL = "low"
THINKING_NOTE = (
    "gemini-3.8-flash does not accept thinking off or thinkingLevel minimal. "
    "The request uses thinkingLevel low, the lowest level that model accepts, "
    "and temperature 0."
)
VIEW_NAMES = ("front", "side", "top")
EXPECTED = (
    ("ycb", "002_master_chef_can"),
    ("gso", "Weisshai_Great_White_Shark"),
    ("abo", "B079TXJCR5"),
)
KEY_PATH = Path(r"C:\Users\sande\cursor\SimFoundary\.env.local")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "mlp" / "samples_manifest.json"
LABEL_THREE_PATH = PROJECT_ROOT / "data" / "mlp" / "label_three.jsonl"
OBJECT_LABEL_NAME = "label.json"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_PROMPT = """Label this one object for a physics prior.

The attached images are rendered camera views of that same object, in this order: front, side, top. The real texture is bound on the mesh. They are not a UV texture atlas.

Return one JSON object and no other text. Use exactly these keys:
material, body_type, collision, mass_kg, density_kg_m3, rigidity, deformability.

material is one of: {materials}
body_type is one of: {bodies}
collision is one of: {collisions}

mass_kg is a positive number, in kilograms.
density_kg_m3 is a positive number, in kilograms per cubic meter.
rigidity is a number from 0 to 1.
deformability is a number from 0 to 1.

Copy the class strings exactly. Do not add Young's modulus, Poisson's ratio, a rendering material, or any unnamed numeric field.

Object id: {obj_id}
Catalog name: {name}
Source: {source}
Measured extent in meters (x, y, z): {extent}
{view_note}"""


def load_api_key(path=KEY_PATH):
    """Read GEMINI_API_KEY. The value is not returned in error text."""

    if not path.is_file():
        raise RuntimeError("GEMINI_API_KEY file is missing")
    found = None
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        name, value = text.split("=", 1)
        if name.strip() != "GEMINI_API_KEY":
            continue
        found = value.strip().strip('"').strip("'")
        break
    if not found:
        raise RuntimeError("GEMINI_API_KEY is missing")
    return found


def prompt_for(obj):
    """One prompt. Front is the largest face, from ``choose_front_view``."""

    view_note = (
        "The front view shows the largest face of the object and fills the frame. "
        "The side view is about a quarter turn from the front. "
        "The top view is the remaining axis."
    )
    extent = ", ".join(f"{float(value):.6f}" for value in obj["extent_m"])
    return _PROMPT.format(
        materials="; ".join(MATERIAL_CLASSES),
        bodies="; ".join(BODY_TYPES),
        collisions="; ".join(COLLISION_PRIMITIVES),
        obj_id=obj["id"],
        name=obj["name"],
        source=obj["source"],
        extent=extent,
        view_note=view_note,
    )


def response_schema():
    """The seven training fields. Slots 28–32 are absent. Friction is absent."""

    return {
        "type": "OBJECT",
        "properties": {
            "material": {"type": "STRING", "enum": list(MATERIAL_CLASSES)},
            "body_type": {"type": "STRING", "enum": list(BODY_TYPES)},
            "collision": {"type": "STRING", "enum": list(COLLISION_PRIMITIVES)},
            "mass_kg": {"type": "NUMBER"},
            "density_kg_m3": {"type": "NUMBER"},
            "rigidity": {"type": "NUMBER"},
            "deformability": {"type": "NUMBER"},
        },
        "required": list(REQUIRED_LABEL_KEYS),
    }


def generation_config():
    return {
        "temperature": TEMPERATURE,
        "responseMimeType": "application/json",
        "responseSchema": response_schema(),
        "maxOutputTokens": 2048,
        "thinkingConfig": {
            "thinkingLevel": THINKING_LEVEL,
            "includeThoughts": False,
        },
    }


def load_objects(manifest_path=MANIFEST_PATH):
    """The three objects that already have views and prior_input.npy."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in manifest["samples"]}
    if [row["id"] for row in manifest["samples"]] != [item[1] for item in EXPECTED]:
        raise RuntimeError("manifest is not the three checked meshes")
    objects = []
    for source, obj_id in EXPECTED:
        row = by_id[obj_id]
        if row["source"] != source:
            raise RuntimeError(f"{obj_id} source is {row['source']}")
        obj_dir = PROJECT_ROOT / "data" / "mlp" / source / obj_id
        representation = json.loads((obj_dir / "representation.json").read_text(encoding="utf-8"))
        views = _view_paths(obj_dir)
        prior = obj_dir / "prior_input.npy"
        shape = tuple(int(value) for value in np.load(prior, mmap_mode="r").shape)
        if len(shape) != 2 or shape[1] != 912:
            raise RuntimeError(f"{obj_id} prior_input shape is {shape}")
        objects.append({
            "id": obj_id,
            "source": source,
            "name": row["name"],
            "extent_m": row["extent_m"],
            "directory": obj_dir,
            "views": views,
            "prior_input": str(prior),
            "prior_input_shape": list(shape),
            "representation_views": [
                representation["view_front"],
                representation["view_side"],
                representation["view_top"],
            ],
        })
    return objects


def _view_paths(obj_dir):
    views_dir = obj_dir / "views"
    paths = []
    for name in VIEW_NAMES:
        path = views_dir / f"{name}.png"
        if path.parent.name != "views" or not path.is_file():
            raise RuntimeError(f"missing camera view {name}.png")
        if path.read_bytes()[:8] != PNG_SIGNATURE:
            raise RuntimeError(f"{name}.png is not a png camera view")
        paths.append(path)
    return paths


def label_three(api_key, objects=None):
    """Call the model once per object, or twice if the first text does not parse."""

    objects = load_objects() if objects is None else objects
    records = []
    for obj in objects:
        destination = obj["directory"] / OBJECT_LABEL_NAME
        existing = _existing_record(destination)
        if existing is not None:
            records.append(existing)
            continue
        record = _label_object(api_key, obj)
        destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        records.append(record)
    _write_jsonl(records)
    return records


def _existing_record(path):
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if record.get("status") != "ok" or record.get("model_id") != MODEL_ID:
        return None
    parse_label(record)
    return record


def _label_object(api_key, obj):
    prompt = prompt_for(obj)
    outcome = _generate(api_key, obj, prompt)
    parsed, error = _accept(outcome)
    calls = 1
    first_error = None
    if parsed is None:
        first_error = error
        retry_prompt = (
            prompt
            + "\n\nThe previous reply was rejected: "
            + error
            + "\nReturn only the JSON object with the seven keys and the exact class strings."
        )
        outcome = _generate(api_key, obj, retry_prompt)
        parsed, error = _accept(outcome)
        calls = 2
    base = {
        "id": obj["id"],
        "source": obj["source"],
        "name": obj["name"],
        "model_id": MODEL_ID,
        "temperature": TEMPERATURE,
        "thinking_level": THINKING_LEVEL,
        "thinking_note": THINKING_NOTE,
        "calls": calls,
        "views": [str(path) for path in obj["views"]],
        "view_bytes": [path.stat().st_size for path in obj["views"]],
        "prior_input": obj["prior_input"],
        "prior_input_shape": obj["prior_input_shape"],
        "http_status": outcome["http_status"],
    }
    if first_error:
        base["first_response_error"] = first_error
    if parsed is None:
        base["status"] = "rejected"
        base["error"] = error
        return base
    base["status"] = "ok"
    base.update(parsed)
    return base


def _generate(api_key, obj, prompt):
    captions = {
        "front": "front view, the largest face",
        "side": "side view, a quarter turn from the front",
        "top": "top view, the remaining axis",
    }
    parts = [{"text": prompt}]
    for name, path in zip(VIEW_NAMES, obj["views"]):
        parts.append({"text": captions[name]})
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        })
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": generation_config(),
    }
    request = urllib.request.Request(
        f"{API_ROOT}/models/{MODEL_ID}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    raw = redact(raw, api_key)
    text = _answer_text(raw) if status == 200 else ""
    error_body = "" if status == 200 else _clip(raw)
    return {"http_status": status, "text": text, "error_body": error_body}


def _answer_text(raw):
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    chunks = []
    for part in parts:
        if part.get("thought"):
            continue
        text = part.get("text")
        if text:
            chunks.append(text)
    return "".join(chunks).strip()


def _accept(outcome):
    if outcome["http_status"] != 200:
        return None, f"HTTP {outcome['http_status']}: {outcome['error_body']}"
    if not outcome["text"]:
        return None, "empty model text"
    try:
        payload = _extract_json(outcome["text"])
    except (json.JSONDecodeError, ValueError) as exc:
        return None, f"JSON parse failed: {exc}. Text: {_clip(outcome['text'])}"
    if not isinstance(payload, dict):
        return None, "model JSON was not an object"
    try:
        parsed = parse_label(_normalize_beside(payload))
    except (TypeError, ValueError) as exc:
        return None, f"schema rejected the reply: {exc}. Text: {_clip(outcome['text'])}"
    record = {
        "material": parsed.material,
        "body_type": parsed.body_type,
        "collision": parsed.collision,
        "mass_kg": parsed.mass_kg,
        "density_kg_m3": parsed.density_kg_m3,
        "log_mass": parsed.log_mass,
        "log_density": parsed.log_density,
        "rigidity": parsed.rigidity,
        "deformability": parsed.deformability,
        "friction_static": parsed.friction_static,
        "friction_dynamic": parsed.friction_dynamic,
        "restitution": parsed.restitution,
    }
    extras = _friction_extras(payload)
    if extras:
        record["friction_restitution_extras"] = extras
    return record, ""


def _normalize_beside(payload):
    data = dict(payload)
    aliases = {
        "friction": "friction_static",
        "static_friction": "friction_static",
        "dynamic_friction": "friction_dynamic",
        "coefficient_of_restitution": "restitution",
    }
    for source, dest in aliases.items():
        if source in data and data.get(dest) is None:
            data[dest] = data[source]
    return data


def _friction_extras(payload):
    extras = {}
    known = set(REQUIRED_LABEL_KEYS) | set(_BESIDE_KEYS)
    for key, value in payload.items():
        if key in known:
            continue
        lowered = str(key).lower()
        if "friction" in lowered or "restitution" in lowered:
            extras[key] = value
    return extras


def _extract_json(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(stripped[start:end + 1])


def _clip(text, limit=800):
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _write_jsonl(records, path=LABEL_THREE_PATH):
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    key = load_api_key()
    try:
        records = label_three(key)
    except Exception as exc:
        raise SystemExit(redact(str(exc), key)) from None
    for record in records:
        if record.get("status") == "ok":
            print(
                f"{record['id']} ok {record['material']} calls={record['calls']}",
                flush=True,
            )
        else:
            print(
                f"{record['id']} rejected calls={record.get('calls')} {record.get('error')}",
                flush=True,
            )
    print(f"wrote {LABEL_THREE_PATH}", flush=True)


if __name__ == "__main__":
    main()
