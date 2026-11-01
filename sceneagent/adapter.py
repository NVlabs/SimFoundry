"""Read a phase-1 capture or a SimFoundry scene JSON.

The SimFoundry path only reads object names out of ``objects_info.init_info``.
It does not load USD, OmniGibson, or meshes. Names become physics-prior inputs.
Geometry sisters wait until a mesh or a box size is present.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_document(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "objects" in data:
        return normalize_capture(data)
    if "objects_info" in data:
        return from_simfoundry_scene(data)
    raise ValueError("document needs 'objects' or SimFoundry 'objects_info'")


def normalize_capture(data):
    if "scene_name" not in data:
        raise ValueError("capture needs scene_name")
    if "task" not in data or "objects" not in data:
        raise ValueError("capture needs task and objects")
    objects = []
    for obj in data["objects"]:
        if "name" not in obj:
            raise ValueError("each object needs a name")
        objects.append(obj)
    return {
        "format": "sceneagent_capture_v1",
        "scene_name": data["scene_name"],
        "task": data["task"],
        "objects": objects,
        "evaluation": data.get("evaluation") or {},
        "source": "sceneagent_capture",
    }


def from_simfoundry_scene(data, scene_name=None):
    init = (data.get("objects_info") or {}).get("init_info") or {}
    if not init:
        raise ValueError("SimFoundry scene has no objects_info.init_info")
    names = list(init)
    objects = [{"name": name, "movable": True, "from_simfoundry": True} for name in names]
    target = names[-1]
    manipulated = names[0]
    return {
        "format": "sceneagent_capture_v1",
        "scene_name": scene_name or data.get("scene_name") or "simfoundry_scene",
        "task": {
            "instruction": f"place the {manipulated} in the {target}",
            "manipulated": manipulated,
            "target": target,
            "goal": "inside",
        },
        "objects": objects,
        "evaluation": {},
        "source": "simfoundry_objects_info",
        "note": (
            "Names only. SimFoundry A/B/C artifacts are not modified. "
            "Mesh sisters require a box size that this scene JSON does not carry."
        ),
    }
