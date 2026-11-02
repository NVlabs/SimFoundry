"""Reviewer view manifest for SceneAgent section 3.1.

The paper says the reviewer inspects object registration overlays, eight
isolated views (six axis-aligned and two oblique), and two views of the
object in the assembled scene. One overlay file plus those ten roles is
eleven paths. The task text said "ten"; that count is the eight isolated
views plus the two assembled views, and it leaves the overlay uncounted.
This manifest includes the overlay so a missing overlay is reported missing.

The module writes the names and paths. It does not render images. Status
stays missing until those files exist.
"""

from __future__ import annotations

import json
from pathlib import Path

# (file stem, group). Order is the paper's order: overlay, six axis,
# two oblique, two assembled.
REQUIRED_VIEWS = (
    ("registration_overlay", "registration_overlay"),
    ("axis_pos_x", "axis"),
    ("axis_neg_x", "axis"),
    ("axis_pos_y", "axis"),
    ("axis_neg_y", "axis"),
    ("axis_pos_z", "axis"),
    ("axis_neg_z", "axis"),
    ("oblique_1", "oblique"),
    ("oblique_2", "oblique"),
    ("assembled_1", "assembled"),
    ("assembled_2", "assembled"),
)


def write_view_manifest(out_dir, object_name="object"):
    """Write the reviewer view manifest. Missing image files stay missing."""
    out = Path(out_dir)
    view_dir = out / "reviewer_views"
    view_dir.mkdir(parents=True, exist_ok=True)
    views = []
    missing = []
    for name, group in REQUIRED_VIEWS:
        path = view_dir / f"{object_name}-{name}.png"
        present = path.is_file() and path.stat().st_size > 0
        views.append({
            "name": name,
            "group": group,
            "path": str(path),
            "present": present,
        })
        if not present:
            missing.append(name)
    body = {
        "section": "3.1",
        "object": object_name,
        "required": [name for name, _group in REQUIRED_VIEWS],
        "views": views,
        "missing": missing,
        "status": "missing" if missing else "present",
        "images_generated": False,
        "note": (
            "Manifest only. Registration overlay, six axis views, two oblique "
            "views, and two assembled views are named. Image files are not synthesized."
        ),
    }
    path = out / "reviewer_views.json"
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    body["manifest"] = str(path)
    return body
