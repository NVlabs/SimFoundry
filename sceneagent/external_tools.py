"""Install paths for the tools SceneAgent names.

``probe`` looks for a command or a Python module. A miss is recorded. It is
not treated as a successful run of that tool.
"""

from __future__ import annotations

import importlib.util
import shutil

from sceneagent.catalog import rows


def _module_present(name):
    if not name:
        return False
    return importlib.util.find_spec(name) is not None


def probe_environment():
    """Return one record per catalog row that names an external install."""
    found = []
    for row in rows():
        if row["relation"] == "in_tree":
            continue
        command = None
        module = None
        if row["tool"] == "colmap":
            command = shutil.which("colmap")
        elif row["tool"] == "ns-train splatfacto":
            command = shutil.which("ns-train")
        elif row["tool"] in {"segment_anything", "open_clip", "groundingdino"}:
            module = _module_present(row["tool"])
        available = bool(command or module)
        found.append({
            "sub_block": row["sub_block"],
            "repo": row.get("repo") or "",
            "available": available,
            "status": row["status"],
            "install": row.get("install") or row.get("next") or "",
        })
    return found
