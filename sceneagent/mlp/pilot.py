"""Finish the 20-object pilot and label it. Do not start the stratified 240.

Eight YCB, six GSO, and six ABO, including the can, the shark, and the pillow.
New files use the mesh suitability check. Labeling views come from
``choose_front_view`` and ``render_views``. The three existing ``prior_input.npy``
files and ``label_three.jsonl`` stay. New objects are packed at ``(N, 912)``
when that path succeeds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from sceneagent.gemini_client import redact
from sceneagent.mlp.check import check_sample
from sceneagent.mlp.label_three import (
    LABEL_THREE_PATH,
    MODEL_ID,
    _label_object,
    _write_jsonl,
    load_api_key,
)
from sceneagent.mlp.represent import MANIFEST, FeatureEncoders, _object_dir, represent_sample
from sceneagent.mlp.sources import (
    MAX_PILOT_FACES,
    PROTECTED_IDS,
    download_pilot,
)
from sceneagent.mlp.splat_from_mesh import load_textured_mesh, render_views, save_views

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "data" / "mlp"
PILOT_MANIFEST = ROOT / "pilot_manifest.json"
PILOT_LABELS = ROOT / "pilot_labels.jsonl"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
VIEW_RENDERER = "choose_front_view"
VIEW_NAMES = ("front", "side", "top")


def run_pilot(stage="all"):
    """``views`` rerenders the original three. ``acquire`` adds the other 17.

    ``label`` calls Gemini. ``all`` does those stages in order and stops.
    """

    if stage not in {"views", "acquire", "label", "all"}:
        raise SystemExit(f"unknown pilot stage {stage}")
    protected = _protected_rows()
    if stage in {"views", "acquire", "all"}:
        _render_protected(protected)
        if stage == "views":
            if not PILOT_MANIFEST.is_file():
                _write_manifest(_manifest_rows(protected, {}), [])
            print("protected views ready", flush=True)
            return
    reports = {}
    if stage in {"acquire", "all"}:
        encoders = FeatureEncoders()

        def accept(sample, _encoders=encoders, _reports=reports):
            return _accept_new(sample, _encoders, _reports)

        acquired = download_pilot(
            root=ROOT,
            accept=accept,
            protected_samples=_resume_rows(protected),
        )
        rejected = acquired["rejected"]
        rows = _manifest_rows(acquired["samples"], reports)
        _write_manifest(rows, rejected)
        if len(rows) != 20:
            raise SystemExit(f"pilot has {len(rows)} objects, not 20")
    if stage in {"label", "all"}:
        rows = json.loads(PILOT_MANIFEST.read_text(encoding="utf-8"))["samples"]
        if len(rows) != 20:
            raise SystemExit(f"pilot manifest has {len(rows)} objects, not 20")
        _label_rows(rows)
    if LABEL_THREE_PATH.is_file():
        print(f"kept {LABEL_THREE_PATH}", flush=True)


def _protected_rows():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for row in manifest["samples"]:
        if row["id"] not in PROTECTED_IDS:
            continue
        checked = _recheck(row)
        if checked is None:
            raise RuntimeError(f"protected object {row['id']} failed the mesh check")
        rows.append(checked)
    if [row["id"] for row in rows] != [
        "002_master_chef_can",
        "Weisshai_Great_White_Shark",
        "B079TXJCR5",
    ]:
        raise RuntimeError("the three checked meshes are not the protected set")
    return rows


def _resume_rows(protected):
    """Protected objects plus packed objects already on disk."""

    rows = list(protected)
    seen = {row["id"] for row in rows}
    for row in _completed_on_disk():
        if row["id"] in seen:
            continue
        checked = _recheck(row)
        if checked is None:
            print(f"skip incomplete {row['id']}", flush=True)
            continue
        rows.append(checked)
        seen.add(checked["id"])
    if not PILOT_MANIFEST.is_file():
        return rows
    manifest = json.loads(PILOT_MANIFEST.read_text(encoding="utf-8"))
    for row in manifest.get("samples") or []:
        if row.get("id") in seen or row.get("id") in PROTECTED_IDS:
            continue
        checked = _recheck(row)
        if checked is None or not _views_current(_object_dir(checked)):
            continue
        rows.append(checked)
        seen.add(checked["id"])
    return rows


def _completed_on_disk():
    rows = []
    for source in ("ycb", "gso", "abo"):
        base = ROOT / source
        if not base.is_dir():
            continue
        for directory in sorted(base.iterdir(), key=lambda path: path.name):
            if not directory.is_dir() or directory.name in PROTECTED_IDS:
                continue
            report_path = directory / "representation.json"
            prior = directory / "prior_input.npy"
            if not report_path.is_file() or not prior.is_file() or not _views_current(directory):
                continue
            shape = tuple(int(value) for value in np.load(prior, mmap_mode="r").shape)
            if len(shape) != 2 or int(shape[1]) != 912 or int(shape[0]) < 1:
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            mesh = report.get("mesh_path")
            if not mesh or not Path(mesh).is_file():
                continue
            rows.append({
                "id": report.get("id") or directory.name,
                "source": source,
                "name": report.get("name") or directory.name,
                "mesh_path": mesh,
                "image_path": _find_image(directory),
                "source_url": "",
                "extent_m": None,
            })
    return rows


def _find_image(directory):
    for pattern in ("**/texture_map.png", "**/textures/texture.png", "**/texture.jpg", "**/texture.jpeg", "**/texture.png"):
        found = sorted(directory.glob(pattern))
        if found:
            return str(found[0])
    front = directory / "views" / "front.png"
    return str(front) if front.is_file() else ""


def _recheck(row):
    sample = {
        "id": row["id"],
        "source": row["source"],
        "name": row.get("name") or row["id"],
        "source_url": row.get("source_url") or "",
        "mesh_path": row["mesh_path"],
        "image_path": row.get("image_path") or "",
        "extent_m": row.get("extent_m"),
    }
    result = check_sample(sample, {"id": row["id"], "source": row["source"]})
    if not result.ok:
        return None
    sample["check_status"] = "pass"
    if not sample.get("extent_m") and result.extent_m is not None:
        scale = 0.001 if sample["source"] == "ycb" and max(result.extent_m) > 5 else 1.0
        sample["extent_m"] = [round(float(value) * scale, 6) for value in result.extent_m]
    return sample


def _render_protected(rows):
    for row in rows:
        directory = _object_dir(row)
        if _views_current(directory):
            print(f"views current {row['id']}", flush=True)
            continue
        print(f"render protected {row['id']}", flush=True)
        mesh, image, uv = load_textured_mesh(row["mesh_path"], row.get("image_path"))
        views = render_views(mesh, image, uv)
        save_views(views, directory / "views")
        _mark_views(directory / "views")
        _require_views(directory)


def _accept_new(sample, encoders, reports):
    """Pack one new object. A mesh that cannot be viewed is replaced.

    A pack failure after a good textured render is kept and labeled.
    """

    try:
        preview, _image, _uv = load_textured_mesh(sample["mesh_path"], sample.get("image_path"))
    except Exception as exc:
        return f"mesh load failed: {type(exc).__name__}: {exc}"
    if len(preview.faces) > MAX_PILOT_FACES:
        return f"{len(preview.faces)} triangles is too many to pack a pilot row"
    try:
        report = represent_sample(sample, encoders)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        directory = _object_dir(sample)
        front = directory / "views" / "front.png"
        if not _png_file(front):
            try:
                mesh, image, uv = load_textured_mesh(sample["mesh_path"], sample.get("image_path"))
                views = render_views(mesh, image, uv)
                save_views(views, directory / "views")
            except Exception as view_exc:
                return f"{reason}; views failed: {type(view_exc).__name__}: {view_exc}"
        if not _views_current(directory) and _png_file(front):
            _mark_views(directory / "views")
        if not _views_current(directory):
            return reason
        reports[sample["id"]] = {"pack_status": "failed", "pack_error": reason}
        print(f"pack failed {sample['id']}: {reason}", flush=True)
        return None
    reports[sample["id"]] = report
    directory = _object_dir(sample)
    _mark_views(directory / "views")
    return None


def _manifest_rows(samples, reports):
    rows = []
    for sample in samples:
        directory = _object_dir(sample)
        _require_views(directory)
        report = reports.get(sample["id"]) or _report_on_disk(sample)
        view_paths = {name: str((directory / "views" / f"{name}.png").resolve()) for name in VIEW_NAMES}
        pack_error = None if report.get("pack_status", "ok") == "ok" else report.get("pack_error")
        if report.get("prior_input_shape") is None and pack_error is None:
            prior = directory / "prior_input.npy"
            if prior.is_file():
                shape = tuple(int(value) for value in np.load(prior, mmap_mode="r").shape)
                report["prior_input"] = str(prior.resolve())
                report["prior_input_shape"] = list(shape)
                report["pack_status"] = "ok" if len(shape) == 2 and shape[1] == 912 else "failed"
                if report["pack_status"] != "ok":
                    pack_error = f"prior_input shape is {shape}"
            else:
                report["pack_status"] = "failed"
                pack_error = report.get("pack_error") or "prior_input.npy is missing"
        rows.append({
            "id": sample["id"],
            "source": sample["source"],
            "name": sample["name"],
            "mesh_path": str(Path(sample["mesh_path"]).resolve()),
            "image_path": sample.get("image_path") or "",
            "source_url": sample.get("source_url") or "",
            "extent_m": sample.get("extent_m"),
            "view_paths": view_paths,
            "check_status": "pass",
            "ply_path": report.get("ply") or report.get("ply_path"),
            "prior_input": report.get("prior_input"),
            "prior_input_shape": report.get("prior_input_shape"),
            "gaussians_kept": report.get("gaussians_kept"),
            "gaussians_dropped": report.get("gaussians_dropped"),
            "pack_status": report.get("pack_status") or ("failed" if pack_error else "ok"),
            "pack_error": pack_error,
            "prior_matches_views": sample["id"] not in PROTECTED_IDS and not pack_error,
        })
    return rows


def _report_on_disk(sample):
    directory = _object_dir(sample)
    path = directory / "representation.json"
    report = {}
    if path.is_file():
        report.update(json.loads(path.read_text(encoding="utf-8")))
        report["pack_status"] = "ok"
    prior = directory / "prior_input.npy"
    if sample["id"] in PROTECTED_IDS:
        report["prior_matches_views"] = False
        report["pack_status"] = "ok" if prior.is_file() else "failed"
    return report


def _label_rows(rows):
    existing = _existing_labels()
    key = load_api_key()
    records = []
    try:
        for row in rows:
            views = [Path(row["view_paths"][name]) for name in VIEW_NAMES]
            previous = existing.get(row["id"])
            if previous is not None and _same_views(previous, views):
                records.append(previous)
                print(f"{row['id']} label kept", flush=True)
                continue
            obj = {
                "id": row["id"],
                "source": row["source"],
                "name": row["name"],
                "extent_m": row["extent_m"],
                "directory": _object_dir(row),
                "views": views,
                "prior_input": row.get("prior_input"),
                "prior_input_shape": row.get("prior_input_shape"),
            }
            record = _label_object(key, obj)
            if row["id"] not in PROTECTED_IDS:
                (obj["directory"] / "label.json").write_text(
                    json.dumps(record, indent=2) + "\n",
                    encoding="utf-8",
                )
            records.append(record)
            _write_jsonl(records, PILOT_LABELS)
            if record.get("status") == "ok":
                print(
                    f"{record['id']} {record['material']} calls={record['calls']}",
                    flush=True,
                )
            else:
                print(
                    f"{record['id']} rejected calls={record.get('calls')} {record.get('error')}",
                    flush=True,
                )
    except Exception as exc:
        raise SystemExit(redact(str(exc), key)) from None
    _write_jsonl(records, PILOT_LABELS)
    print(f"wrote {PILOT_LABELS}", flush=True)
    return records


def _existing_labels():
    if not PILOT_LABELS.is_file():
        return {}
    found = {}
    for line in PILOT_LABELS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "ok" and record.get("model_id") == MODEL_ID:
            found[record["id"]] = record
    return found


def _same_views(record, views):
    sizes = [path.stat().st_size for path in views]
    return list(record.get("view_bytes") or []) == sizes


def _views_current(directory):
    meta_path = directory / "views" / "meta.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if meta.get("renderer") != VIEW_RENDERER or meta.get("textured") is not True:
        return False
    return all(_png_file(directory / "views" / f"{name}.png") for name in VIEW_NAMES)


def _mark_views(directory):
    payload = {
        "renderer": VIEW_RENDERER,
        "textured": True,
        "rule": (
            "largest textured face fills the frame; "
            "side is a quarter turn; top is the remaining axis"
        ),
    }
    (directory / "meta.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _require_views(directory):
    if not _views_current(directory):
        raise RuntimeError(f"labeling views are missing in {directory}")


def _png_file(path):
    path = Path(path)
    return path.is_file() and path.read_bytes()[:8] == PNG_SIGNATURE


def _write_manifest(rows, rejected):
    ROOT.mkdir(parents=True, exist_ok=True)
    payload = {"samples": rows, "rejected": rejected}
    PILOT_MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {PILOT_MANIFEST} ({len(rows)} objects)", flush=True)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    run_pilot(stage)


if __name__ == "__main__":
    main()
