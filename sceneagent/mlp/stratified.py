"""Stratified YCB, GSO, and ABO labels for the physics prior.

About 80 object models from each source. GSO and ABO are balanced across the
ten material classes, with a small and a large object in each class and no
near-duplicate names. A mesh that fails the check, is too large to render,
or cannot be packed at (N, 912) is replaced from the same source and material
and is not labeled.

The original three view folders are not written. Object dumps stay in
data/mlp on this machine.
"""

from __future__ import annotations

import csv
import gc
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

from sceneagent.mlp.represent import FeatureEncoders, _object_dir, finish_object, render_object
from sceneagent.mlp.sources import (
    ABO_ASSET_ROOT,
    ABO_LISTING_SHARD,
    ABO_MODELS_CSV,
    GSO_COLLECTION,
    MAX_ABO_BYTES,
    PROTECTED_IDS,
    SKIP_IDS,
    Candidate,
    _content_length,
    _english_name,
    _read_bytes,
    _safe_id,
    _ycb_candidates,
    attempt_download,
    math_finite_positive,
)
from sceneagent.mlp.splat_from_mesh import _triangle_mesh_path, load_textured_mesh
from sceneagent.mlp.stratum import (
    MATERIAL_CLASSES,
    guess_material,
    labels_should_stop,
    order_bucket,
)
from sceneagent.taxonomy import MATERIAL_CLASSES as TAXONOMY_MATERIALS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "data" / "mlp"
MANIFEST_PATH = ROOT / "stratified_manifest.json"
LABELS_PATH = ROOT / "stratified_labels.jsonl"
QUEUE_PATH = ROOT / "stratified_queue.json"
SUMMARY_PATH = ROOT / "stratified_summary.json"
PILOT_MANIFEST = ROOT / "pilot_manifest.json"
PILOT_LABELS = ROOT / "pilot_labels.jsonl"
SOURCE_TARGET = 80
PER_MATERIAL = 8
MATERIAL_CAP = 12
QUEUE_PER_MATERIAL = 40
MAX_RENDER_FACES = 48000
SOFT_FACES = 20000
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
VIEW_NAMES = ("front", "side", "top")
SOURCES = ("ycb", "gso", "abo")
PROTECTED_DIRS = tuple(
    (ROOT / source / obj_id).resolve()
    for source, obj_id in (
        ("ycb", "002_master_chef_can"),
        ("gso", "Weisshai_Great_White_Shark"),
        ("abo", "B079TXJCR5"),
    )
)

# One DINOv2 + OpenCLIP copy peaked at 2446 MB private on this PC.
# A second copy is what killed the two-worker runs. Workers only render.
_ENCODER_PEAK_PRIVATE_MB = 2446
_RENDER_SLOT_FLOOR_MB = 700
_WORKER_RSS_LOGS = 0


def _limit_numeric_threads(load_torch=False):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    if not load_torch:
        return
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


def _mem_snapshot():
    try:
        import psutil

        proc = psutil.Process().memory_info()
        vm = psutil.virtual_memory()
        return {
            "pid": os.getpid(),
            "rss_mb": round(proc.rss / 1048576, 1),
            "private_mb": round(proc.private / 1048576, 1),
            "avail_mb": round(vm.available / 1048576, 1),
        }
    except Exception as exc:
        return {"pid": os.getpid(), "error": type(exc).__name__}


def _init_worker():
    global _WORKER_RSS_LOGS
    # Render workers must not import torch. The parent holds the only models.
    _limit_numeric_threads(load_torch=False)
    _WORKER_RSS_LOGS = 0
    # region agent log
    _agent_log(
        "H6",
        "stratified.py:_init_worker",
        "worker started without encoders",
        {**_mem_snapshot(), "encoders": False},
    )
    # endregion


def acquire_candidate(payload):
    """Download, check, and pack one object. A failure is not labeled."""

    obj_id = payload["id"]
    source = payload["source"]
    try:
        if obj_id in PROTECTED_IDS or obj_id in SKIP_IDS or obj_id in _blocked_ids():
            return _failure(payload, "blocked id")
        destination = ROOT / source / _safe_id(obj_id)
        if _is_protected(destination):
            return _failure(payload, "refusing to write a protected object")
        if _packed_row(destination, source) is not None:
            return {"ok": True, "row": _packed_row(destination, source), "reused": True}
        candidate = Candidate(
            source=source,
            id=obj_id,
            name=payload["name"],
            source_url=payload["source_url"],
            download_url=payload["download_url"],
            kind=payload["kind"],
        )
        sample, rejection = attempt_download(candidate, ROOT)
        if rejection is not None:
            return _failure(payload, rejection["reason"])
        mesh_path = _triangle_mesh_path(sample["mesh_path"])
        fast_faces = _obj_face_lines(mesh_path)
        if fast_faces is not None and fast_faces > MAX_RENDER_FACES:
            _delete_unprotected(source, obj_id)
            return _failure(payload, f"{fast_faces} triangles is too many to render")
        mesh, _image, _uv = load_textured_mesh(sample["mesh_path"], sample.get("image_path"))
        faces = int(len(mesh.faces))
        del mesh, _image, _uv
        if faces > MAX_RENDER_FACES:
            _delete_unprotected(source, obj_id)
            return _failure(payload, f"{faces} triangles is too many to render")
        directory = _object_dir(sample)
        if _is_protected(directory):
            raise RuntimeError("refusing to write a protected view folder")
        rendered = render_object(sample)
        gc.collect()
        global _WORKER_RSS_LOGS
        if _WORKER_RSS_LOGS < 2:
            _WORKER_RSS_LOGS += 1
            # region agent log
            _agent_log(
                "H6",
                "stratified.py:acquire_candidate",
                "render finished without encoders",
                {
                    **_mem_snapshot(),
                    "id": obj_id,
                    "encoders": False,
                    "gaussians": int(len(rendered["gaussians"]["xyz"])),
                },
            )
            # endregion
        return {
            "ok": True,
            "rendered": rendered,
            "faces": faces,
            "material": payload.get("material") or "unknown",
            "sample": sample,
        }
    except Exception as exc:
        if obj_id not in PROTECTED_IDS and not _is_protected(ROOT / source / _safe_id(obj_id)):
            _delete_unprotected(source, obj_id)
        return _failure(payload, f"{type(exc).__name__}: {exc}")


def build_queue(occupied_names):
    """Catalog rows in keep order. This does not download meshes."""

    if tuple(MATERIAL_CLASSES) != tuple(TAXONOMY_MATERIALS):
        raise RuntimeError("sampler material order drifted from taxonomy.py")
    blocked = _blocked_ids()
    print("listing YCB google_16k models", flush=True)
    ycb = []
    for candidate in _ycb_candidates():
        if candidate.id in blocked:
            continue
        ycb.append(_job(candidate, guess_material(candidate.name), None, None))
    print(f"ycb candidates {len(ycb)}", flush=True)
    print("listing GSO scans", flush=True)
    gso_jobs = _gso_jobs(blocked, occupied_names)
    print("listing ABO models", flush=True)
    abo_jobs = _abo_jobs(blocked, occupied_names)
    jobs = ycb + gso_jobs + abo_jobs
    _print_queue(jobs)
    return jobs


_SLOT_HELD_LOGGED = False


def _submission_slots(pool_workers):
    """How many renders may run at once given free RAM.

    Two encoder copies are not allowed. A second render starts only while at
    least 700 MB is free, which is the headroom this machine did not have when
    the second model load was killed.
    """

    global _SLOT_HELD_LOGGED
    slots = max(1, min(int(pool_workers), 2))
    snap = _mem_snapshot()
    avail = snap.get("avail_mb")
    if slots > 1 and avail is not None and avail < _RENDER_SLOT_FLOOR_MB:
        slots = 1
        if not _SLOT_HELD_LOGGED:
            _SLOT_HELD_LOGGED = True
            # region agent log
            _agent_log(
                "H6",
                "stratified.py:_submission_slots",
                "held second render worker",
                {"slots": slots, **snap},
            )
            # endregion
    return slots


_PARENT_FEATURE_LOGS = 0


def _finish_rendered(result, encoders):
    """Encode one rendered object in the parent. Workers do not hold models."""

    global _PARENT_FEATURE_LOGS
    sample = result["sample"]
    rendered = result.pop("rendered")
    source = sample["source"]
    obj_id = sample["id"]
    material = result.get("material") or "unknown"
    faces = result["faces"]
    try:
        directory = _object_dir(sample)
        if _is_protected(directory):
            raise RuntimeError("refusing to write a protected view folder")
        report = finish_object(sample, rendered, encoders)
        _mark_views(directory / "views")
        if not _views_ready(directory):
            _delete_unprotected(source, obj_id)
            return _failure({**sample, "material": material}, "labeling views were not written")
        shape = report.get("prior_input_shape")
        if not shape or int(shape[1]) != 912 or int(shape[0]) < 1:
            _delete_unprotected(source, obj_id)
            return _failure({**sample, "material": material}, f"prior_input shape is {shape}")
        row = _manifest_row(sample, report, material, faces)
        (directory / "object_record.json").write_text(
            json.dumps(row, indent=2) + "\n",
            encoding="utf-8",
        )
        if _PARENT_FEATURE_LOGS < 2:
            _PARENT_FEATURE_LOGS += 1
            # region agent log
            _agent_log(
                "H6",
                "stratified.py:_finish_rendered",
                "features finished in parent",
                {**_mem_snapshot(), "id": obj_id, "encoders": True},
            )
            # endregion
        return {"ok": True, "row": row, "reused": False}
    except Exception as exc:
        if obj_id not in PROTECTED_IDS and not _is_protected(ROOT / source / _safe_id(obj_id)):
            _delete_unprotected(source, obj_id)
        return _failure({**sample, "material": material}, f"{type(exc).__name__}: {exc}")
    finally:
        del rendered
        gc.collect()


def run_acquire(jobs, workers):
    """Pack and label until each source is full or its queue is empty."""

    signature = _view_signature()
    blocked = _blocked_ids()
    rejected, memory_retries = _drop_retryable(_load_rejected())
    if memory_retries:
        print(f"retry memory and thin-object failures {len(memory_retries)}", flush=True)
        # region agent log
        _agent_log(
            "H5",
            "stratified.py:run_acquire",
            "requeued failures",
            {"count": len(memory_retries), "ids": [item.get("id") for item in memory_retries]},
        )
        # endregion
    packed = {row["id"]: row for row in _scan_packed() if row["id"] not in blocked}
    labels = _load_labels()
    kept_ids = set()
    for row in list(packed.values()):
        record = labels.get(row["id"])
        if record is None:
            record = _label_with_retries(row)
            if record.get("status") == "ok":
                labels[row["id"]] = record
                _write_labels(labels)
            else:
                rejected.append(_failure(row, record.get("error") or "label rejected"))
                continue
        if record.get("status") != "ok":
            continue
        kept_ids.add(row["id"])
    plan = _Plan(jobs, blocked, rejected, skip_ids=kept_ids)
    for row in packed.values():
        if row["id"] in kept_ids:
            plan.note_kept(row)
    _write_state(packed, labels, rejected, kept_ids)
    print(
        "resume "
        + " ".join(f"{source}={plan.source_kept[source]}" for source in SOURCES),
        flush=True,
    )
    if plan.finished():
        _write_state(packed, labels, rejected, kept_ids)
        _check_views(signature)
        return labels
    _limit_numeric_threads(load_torch=True)
    before = _mem_snapshot()
    encoders = FeatureEncoders().load()
    after = _mem_snapshot()
    # region agent log
    _agent_log(
        "H6",
        "stratified.py:run_acquire",
        "encoder loaded once in parent",
        {
            "before": before,
            "after": after,
            "copies": 1,
            "prior_encoder_peak_private_mb": _ENCODER_PEAK_PRIVATE_MB,
        },
    )
    # endregion
    pool_workers = min(max(1, int(workers)), 2)
    print(
        f"encoder once, render workers {pool_workers} "
        f"(requested {int(workers)}, free {after.get('avail_mb')} MB)",
        flush=True,
    )
    # region agent log
    _agent_log(
        "H4",
        "stratified.py:run_acquire",
        "worker pool",
        {"requested": int(workers), "workers": pool_workers, **_mem_snapshot()},
    )
    # endregion
    pending = {}
    with ProcessPoolExecutor(max_workers=pool_workers, initializer=_init_worker) as pool:
        while True:
            slots = _submission_slots(pool_workers)
            while len(pending) < slots:
                job = plan.next_job()
                if job is None:
                    break
                pending[pool.submit(acquire_candidate, job)] = job
            if not pending:
                break
            done, _pending_left = wait(pending, return_when=FIRST_COMPLETED, timeout=120)
            if not done:
                print(
                    "still rendering "
                    + " ".join(item["id"] for item in list(pending.values())[:6]),
                    flush=True,
                )
                continue
            for future in done:
                job = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = _failure(job, f"{type(exc).__name__}: {exc}")
                plan.finish_job(job)
                if result.get("ok") and "rendered" in result:
                    result = _finish_rendered(result, encoders)
                if not result.get("ok"):
                    rejected.append(result)
                    print(f"replace {job['source']} {job['id']}: {result.get('reason')}", flush=True)
                    _write_state(packed, labels, rejected, kept_ids)
                    continue
                row = result["row"]
                packed[row["id"]] = row
                record = _label_with_retries(row)
                if record.get("status") != "ok":
                    rejected.append(_failure(row, record.get("error") or "label rejected"))
                    print(f"unlabeled {row['id']}: {record.get('error')}", flush=True)
                else:
                    labels[row["id"]] = record
                    kept_ids.add(row["id"])
                    plan.note_kept(row)
                    print(
                        f"kept {row['source']} {row['id']} {record['material']} "
                        + " ".join(f"{source}={plan.source_kept[source]}" for source in SOURCES),
                        flush=True,
                    )
                _write_state(packed, labels, rejected, kept_ids)
    _write_state(packed, labels, rejected, kept_ids)
    _check_views(signature)
    return labels


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    if stage not in {"catalog", "acquire", "all"}:
        raise SystemExit(f"unknown stage {stage}")
    ROOT.mkdir(parents=True, exist_ok=True)
    signature = _view_signature()
    if stage == "catalog" or not QUEUE_PATH.is_file():
        jobs = build_queue(_occupied_names())
        QUEUE_PATH.write_text(json.dumps(jobs) + "\n", encoding="utf-8")
        print(f"wrote {QUEUE_PATH}", flush=True)
    else:
        jobs = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
        print(f"reused {QUEUE_PATH} ({len(jobs)} jobs)", flush=True)
    if stage == "catalog":
        _check_views(signature)
        return
    labels = run_acquire(jobs, workers)
    ok = [row for row in labels.values() if row.get("status") == "ok"]
    reason = labels_should_stop(ok)
    summary = _summary(ok, reason)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _print_summary(summary)
    _check_views(signature)
    if reason:
        print(f"stopped before training: {reason}", flush=True)
        raise SystemExit(2)
    if stage == "all":
        from sceneagent.mlp.train_prior import train_and_report

        train_and_report()
    _check_views(signature)


class _Plan:
    def __init__(self, jobs, blocked, rejected, skip_ids=()):
        self.ycb = deque()
        self.buckets = {
            source: {material: deque() for material in MATERIAL_CLASSES}
            for source in ("gso", "abo")
        }
        self.source_kept = {source: 0 for source in SOURCES}
        self.material_kept = {
            source: {material: 0 for material in MATERIAL_CLASSES}
            for source in ("gso", "abo")
        }
        self.inflight = {source: 0 for source in SOURCES}
        self.inflight_material = {
            source: {material: 0 for material in MATERIAL_CLASSES}
            for source in ("gso", "abo")
        }
        seen = set(blocked) | set(skip_ids)
        seen.update(item.get("id") for item in rejected if item.get("id"))
        for job in jobs:
            if job["id"] in seen:
                continue
            seen.add(job["id"])
            if job["source"] == "ycb":
                self.ycb.append(job)
            elif job["source"] in self.buckets:
                material = job.get("material") or "unknown"
                if material not in self.buckets[job["source"]]:
                    material = "unknown"
                self.buckets[job["source"]][material].append(job)

    def note_kept(self, row):
        source = row["source"]
        self.source_kept[source] += 1
        if source in self.material_kept:
            material = row.get("guessed_material") or row.get("material") or "unknown"
            if material not in self.material_kept[source]:
                material = "unknown"
            self.material_kept[source][material] += 1

    def next_job(self):
        ranked = []
        for source in SOURCES:
            if self.source_kept[source] + self.inflight[source] >= SOURCE_TARGET:
                continue
            deficit = SOURCE_TARGET - self.source_kept[source] - self.inflight[source]
            ranked.append((-deficit, source))
        ranked.sort()
        for _deficit, source in ranked:
            job = self._pop(source)
            if job is not None:
                self.inflight[source] += 1
                if source in self.inflight_material:
                    material = job.get("material") or "unknown"
                    self.inflight_material[source][material] += 1
                return job
        return None

    def finish_job(self, job):
        source = job["source"]
        self.inflight[source] = max(0, self.inflight[source] - 1)
        if source in self.inflight_material:
            material = job.get("material") or "unknown"
            if material not in self.inflight_material[source]:
                material = "unknown"
            self.inflight_material[source][material] = max(
                0, self.inflight_material[source][material] - 1
            )

    def finished(self):
        if any(self.inflight.values()):
            return False
        if self.ycb and self.source_kept["ycb"] < SOURCE_TARGET:
            return False
        for source in ("gso", "abo"):
            if self.source_kept[source] >= SOURCE_TARGET:
                continue
            for material, queue in self.buckets[source].items():
                count = self.material_kept[source][material]
                if queue and count < MATERIAL_CAP:
                    return False
        return True

    def _pop(self, source):
        if source == "ycb":
            return self.ycb.popleft() if self.ycb else None
        ranked = []
        for material, queue in self.buckets[source].items():
            if not queue:
                continue
            count = self.material_kept[source][material] + self.inflight_material[source][material]
            if count >= MATERIAL_CAP:
                continue
            rank = 0 if count < PER_MATERIAL else 1
            ranked.append((rank, count, material))
        if not ranked:
            return None
        ranked.sort()
        return self.buckets[source][ranked[0][2]].popleft()


def _gso_jobs(blocked, occupied_names):
    grouped = {material: [] for material in MATERIAL_CLASSES}
    seen = set()
    consecutive_failures = 0
    for page in range(1, 81):
        url = f"{GSO_COLLECTION}?page={page}&per_page=20"
        try:
            models = json.loads(_read_bytes(url))
        except Exception as exc:
            consecutive_failures += 1
            print(f"gso page {page} failed: {exc}", flush=True)
            if consecutive_failures >= 2:
                break
            continue
        consecutive_failures = 0
        if not isinstance(models, list) or not models:
            break
        added = 0
        for model in models:
            if model.get("owner") != "GoogleResearch":
                continue
            name = model.get("name")
            if not name or name in seen or name in blocked:
                continue
            seen.add(name)
            added += 1
            description = (model.get("description") or "").strip().splitlines()
            label = description[0].strip() if description and description[0].strip() else name.replace("_", " ")
            material = guess_material(label)
            grouped[material].append({
                "id": name,
                "name": label,
                "material": material,
                "extent": None,
                "faces": None,
                "source_url": f"https://app.gazebosim.org/GoogleResearch/fuel/models/{name}",
                "download_url": f"https://fuel.gazebosim.org/1.0/GoogleResearch/models/{name}.zip",
                "kind": "archive",
            })
        print(f"gso page {page} models {added}", flush=True)
        if added == 0:
            break
    jobs = []
    for material, items in grouped.items():
        jobs.extend(_queue_material("gso", items, occupied_names))
    return jobs


def _abo_jobs(blocked, occupied_names):
    raw = _read_bytes(ABO_MODELS_CSV)
    if raw[:2] == b"\x1f\x8b":
        text = gzip.decompress(raw).decode("utf-8")
    else:
        text = raw.decode("utf-8")
    grouped = {material: [] for material in MATERIAL_CLASSES}
    wanted = {}
    for row in csv.DictReader(io.StringIO(text)):
        rel = row.get("path") or ""
        if not rel.lower().endswith((".glb", ".gltf")):
            continue
        model_id = row.get("3dmodel_id") or ""
        if not model_id or model_id in blocked:
            continue
        try:
            faces = int(row["faces"])
            images = int(row["images"])
            textures = int(row["textures"])
            side = max(int(row["image_width_max"]), int(row["image_height_max"]))
            extent = max(float(row[key]) for key in ("extent_x", "extent_y", "extent_z"))
        except (KeyError, TypeError, ValueError):
            continue
        if faces < 100 or faces > MAX_RENDER_FACES or images < 1 or textures < 1 or side > 4096:
            continue
        if not math_finite_positive(extent) or extent >= 5:
            continue
        wanted[model_id] = {
            "id": model_id,
            "faces": faces,
            "extent": extent,
            "download_url": ABO_ASSET_ROOT + rel,
        }
    names = _abo_names(set(wanted))
    named = 0
    for model_id, item in wanted.items():
        name = names.get(model_id) or model_id
        if name != model_id:
            named += 1
        material = guess_material(name)
        grouped[material].append({
            "id": model_id,
            "name": name,
            "material": material,
            "extent": item["extent"],
            "faces": item["faces"],
            "source_url": item["download_url"],
            "download_url": item["download_url"],
            "kind": "mesh",
        })
    print(f"abo named {named} of {len(wanted)}", flush=True)
    if wanted and named / len(wanted) < 0.5:
        raise SystemExit("ABO product names were missing for more than half of the models")
    jobs = []
    for material, items in grouped.items():
        preferred = _prefer_faces(items, QUEUE_PER_MATERIAL + 20)
        ordered = _queue_material("abo", preferred, occupied_names)
        kept = []
        for job in ordered:
            length = _content_length(job["download_url"])
            if length is None or length < 1 or length > MAX_ABO_BYTES:
                continue
            kept.append(job)
            if len(kept) >= QUEUE_PER_MATERIAL:
                break
        jobs.extend(kept)
    return jobs


def _queue_material(source, items, occupied_names):
    primary_names = list(occupied_names)
    ordered = order_bucket(items, QUEUE_PER_MATERIAL, primary_names)
    jobs = []
    for item in ordered:
        jobs.append(_job_from_item(source, item))
    return jobs


def _prefer_faces(items, need):
    soft = [row for row in items if row.get("faces") is None or row["faces"] <= SOFT_FACES]
    if len(soft) >= need:
        return soft
    return [row for row in items if row.get("faces") is None or row["faces"] <= MAX_RENDER_FACES]


def _abo_names(wanted):
    found = {}
    shards = [str(index) for index in range(10)] + list("abcdef")
    for shard in shards:
        if len(found) >= len(wanted):
            break
        url = ABO_LISTING_SHARD.format(shard=shard)
        print(f"abo names listings_{shard}", flush=True)
        try:
            raw = _read_bytes(url)
        except Exception as exc:
            print(f"abo listings_{shard} failed: {exc}", flush=True)
            continue
        if raw[:2] == b"\x1f\x8b":
            stream = gzip.GzipFile(fileobj=io.BytesIO(raw))
        else:
            stream = io.BytesIO(raw)
        for line in stream:
            if b"3dmodel_id" not in line and b"item_id" not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys = [record.get("3dmodel_id"), record.get("item_id")]
            if not any(key in wanted for key in keys):
                continue
            name = _english_name(record.get("item_name"))
            if not name:
                continue
            for key in keys:
                if key in wanted and key not in found:
                    found[key] = name
    return found


def _job(candidate, material, extent, faces):
    return {
        "id": candidate.id,
        "source": candidate.source,
        "name": candidate.name,
        "material": material,
        "extent": extent,
        "faces": faces,
        "source_url": candidate.source_url,
        "download_url": candidate.download_url,
        "kind": candidate.kind,
    }


def _job_from_item(source, item):
    return {
        "id": item["id"],
        "source": source,
        "name": item["name"],
        "material": item["material"],
        "extent": item.get("extent"),
        "faces": item.get("faces"),
        "source_url": item["source_url"],
        "download_url": item["download_url"],
        "kind": item["kind"],
    }


def _existing_label(row):
    path = Path(row["directory"]) / "label.json"
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if record.get("status") != "ok" or record.get("model_id") != "gemini-3.8-flash":
        return None
    sizes = []
    for name in VIEW_NAMES:
        view = Path(row["view_paths"][name])
        if not view.is_file():
            return None
        sizes.append(view.stat().st_size)
    if list(record.get("view_bytes") or []) != sizes:
        return None
    return record


def _label_with_retries(row):
    from sceneagent.gemini_client import redact
    from sceneagent.mlp.label_three import _label_object, load_api_key

    key = load_api_key()
    existing = _existing_label(row)
    if existing is not None:
        return existing
    obj = {
        "id": row["id"],
        "source": row["source"],
        "name": row["name"],
        "extent_m": row["extent_m"],
        "directory": Path(row["directory"]),
        "views": [Path(row["view_paths"][name]) for name in VIEW_NAMES],
        "prior_input": row["prior_input"],
        "prior_input_shape": row["prior_input_shape"],
    }
    record = None
    try:
        for attempt in range(4):
            record = _label_object(key, obj)
            error = record.get("error") or ""
            if record.get("status") == "ok":
                break
            if not any(code in error for code in ("HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503")):
                break
            time.sleep(20 * (attempt + 1))
    except Exception as exc:
        raise SystemExit(redact(str(exc), key)) from None
    if record.get("error"):
        record["error"] = redact(record["error"], key)
    if record.get("status") == "ok":
        (Path(row["directory"]) / "label.json").write_text(
            json.dumps(record, indent=2) + "\n",
            encoding="utf-8",
        )
    return record


def _manifest_row(sample, report, material, faces):
    directory = _object_dir(sample)
    extent = sample.get("extent_m")
    if isinstance(extent, (list, tuple)) and len(extent) == 3:
        extent = [float(value) for value in extent]
    else:
        raise RuntimeError(f"{sample['id']} has no extent")
    return {
        "id": sample["id"],
        "source": sample["source"],
        "name": sample["name"],
        "mesh_path": str(Path(sample["mesh_path"]).resolve()),
        "image_path": sample.get("image_path") or "",
        "source_url": sample.get("source_url") or "",
        "extent_m": extent,
        "view_paths": {
            name: str((directory / "views" / f"{name}.png").resolve()) for name in VIEW_NAMES
        },
        "directory": str(directory.resolve()),
        "check_status": "pass",
        "ply_path": report.get("ply"),
        "prior_input": report.get("prior_input"),
        "prior_input_shape": [int(report["prior_input_shape"][0]), int(report["prior_input_shape"][1])],
        "gaussians_kept": report.get("gaussians_kept"),
        "gaussians_dropped": report.get("gaussians_dropped"),
        "pack_status": "ok",
        "pack_error": None,
        "guessed_material": material,
        "faces": int(faces),
    }


def _scan_packed():
    rows = []
    blocked = _blocked_ids()
    for source in SOURCES:
        base = ROOT / source
        if not base.is_dir():
            continue
        for directory in sorted(base.iterdir(), key=lambda path: path.name):
            if not directory.is_dir() or directory.name in blocked or _is_protected(directory):
                continue
            row = _packed_row(directory, source)
            if row is not None:
                rows.append(row)
    return rows


def _packed_row(directory, source):
    record_path = directory / "object_record.json"
    prior = directory / "prior_input.npy"
    if not record_path.is_file() or not prior.is_file() or not _views_ready(directory):
        return None
    try:
        row = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    shape = tuple(int(value) for value in np.load(prior, mmap_mode="r").shape)
    if len(shape) != 2 or shape[1] != 912 or shape[0] < 1:
        return None
    if row.get("id") != directory.name or row.get("source") != source:
        return None
    row["prior_input_shape"] = [shape[0], shape[1]]
    row["prior_input"] = str(prior.resolve())
    return row


def _write_state(packed, labels, rejected, kept_ids):
    samples = []
    for obj_id in kept_ids:
        row = packed.get(obj_id)
        record = labels.get(obj_id)
        if row is None or record is None or record.get("status") != "ok":
            continue
        item = dict(row)
        item["material"] = record["material"]
        samples.append(item)
    samples.sort(key=lambda row: (row["source"], row["id"]))
    MANIFEST_PATH.write_text(
        json.dumps({"samples": samples, "rejected": rejected}, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_labels(labels)


def _write_labels(labels):
    rows = [labels[key] for key in sorted(labels) if labels[key].get("status") == "ok"]
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    LABELS_PATH.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _load_labels():
    found = {}
    if not LABELS_PATH.is_file():
        return found
    for line in LABELS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "ok" and record.get("id"):
            found[record["id"]] = record
    for row in _scan_packed():
        path = Path(row["directory"]) / "label.json"
        if row["id"] in found or not path.is_file():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if record.get("status") == "ok":
            found[record["id"]] = record
    return found


def _drop_retryable(rejected):
    """Return memory and edge-on failures to the queue. Gray stays rejected."""

    kept = []
    retry = []
    for item in rejected:
        reason = str(item.get("reason") or "")
        if reason.startswith("MemoryError") or "front view is edge-on" in reason:
            retry.append(item)
        else:
            kept.append(item)
    return kept, retry


def _agent_log(hypothesis_id, location, message, data, run_id="post-fix"):
    try:
        import json as _json
        import time as _time

        line = _json.dumps({
            "sessionId": "18246b",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(_time.time() * 1000),
        })
        with open(r"C:\Users\sande\cursor\SimFoundary\debug-18246b.log", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        return


def _load_rejected():
    if not MANIFEST_PATH.is_file():
        return []
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return list(payload.get("rejected") or [])


def _summary(ok, reason):
    by_source = {source: 0 for source in SOURCES}
    by_material = {material: 0 for material in MATERIAL_CLASSES}
    for row in ok:
        if row.get("source") in by_source:
            by_source[row["source"]] += 1
        material = row.get("material")
        if material in by_material:
            by_material[material] += 1
    ordered = sorted(ok, key=lambda row: (row.get("source") or "", row.get("id") or ""))
    if len(ordered) <= 20:
        chosen = ordered
    else:
        chosen = [ordered[int(index * len(ordered) / 20)] for index in range(20)]
    spot = [
        {
            "name": row.get("name"),
            "material": row.get("material"),
            "body_type": row.get("body_type"),
            "mass_kg": row.get("mass_kg"),
            "source": row.get("source"),
            "id": row.get("id"),
        }
        for row in chosen
    ]
    return {
        "per_source": by_source,
        "per_material": by_material,
        "ok": len(ok),
        "stop": reason,
        "spot_check": spot,
    }


def _print_summary(summary):
    print("stratified counts " + " ".join(f"{key}={value}" for key, value in summary["per_source"].items()), flush=True)
    print("materials " + " ".join(f"{key}={value}" for key, value in summary["per_material"].items()), flush=True)
    for row in summary["spot_check"]:
        print(
            f"spot {row['name']} | {row['material']} | {row['body_type']} | {row['mass_kg']}",
            flush=True,
        )


def _print_queue(jobs):
    counts = {source: {material: 0 for material in MATERIAL_CLASSES} for source in SOURCES}
    totals = {source: 0 for source in SOURCES}
    for job in jobs:
        totals[job["source"]] = totals.get(job["source"], 0) + 1
        material = job.get("material") or "unknown"
        if job["source"] in counts and material in counts[job["source"]]:
            counts[job["source"]][material] += 1
    print("queue " + " ".join(f"{source}={totals.get(source, 0)}" for source in SOURCES), flush=True)
    for source in ("gso", "abo"):
        print(
            source + " " + " ".join(f"{material}={counts[source][material]}" for material in MATERIAL_CLASSES),
            flush=True,
        )


def _blocked_ids():
    found = set(PROTECTED_IDS) | set(SKIP_IDS)
    for path in (PILOT_LABELS, PILOT_MANIFEST):
        if not path.is_file():
            continue
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    found.add(json.loads(line).get("id"))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("samples") or []:
            found.add(row.get("id"))
        for row in payload.get("rejected") or []:
            found.add(row.get("id"))
    found.discard(None)
    return found


def _occupied_names():
    names = []
    if PILOT_LABELS.is_file():
        for line in PILOT_LABELS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                names.append(json.loads(line).get("name") or "")
    return [name for name in names if name]


def _failure(payload, reason):
    return {
        "ok": False,
        "id": payload.get("id"),
        "source": payload.get("source"),
        "name": payload.get("name"),
        "material": payload.get("material") or payload.get("guessed_material"),
        "reason": str(reason),
    }


def _delete_unprotected(source, obj_id):
    if obj_id in PROTECTED_IDS:
        raise RuntimeError(f"refusing to delete protected object {obj_id}")
    path = ROOT / source / _safe_id(obj_id)
    if _is_protected(path):
        raise RuntimeError(f"refusing to delete protected object {obj_id}")
    shutil.rmtree(path, ignore_errors=True)


def _is_protected(path):
    resolved = Path(path).resolve()
    for folder in PROTECTED_DIRS:
        if resolved == folder or folder in resolved.parents:
            return True
    return False


def _views_ready(directory):
    meta_path = directory / "views" / "meta.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if meta.get("renderer") != "choose_front_view" or meta.get("textured") is not True:
        return False
    return all(_png_file(directory / "views" / f"{name}.png") for name in VIEW_NAMES)


def _mark_views(directory):
    if _is_protected(directory):
        raise RuntimeError("refusing to write a protected view folder")
    payload = {
        "renderer": "choose_front_view",
        "textured": True,
        "rule": (
            "largest textured face fills the frame; "
            "side is a quarter turn; top is the remaining axis"
        ),
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _png_file(path):
    path = Path(path)
    return path.is_file() and path.read_bytes()[:8] == PNG_SIGNATURE


def _obj_face_lines(path):
    path = Path(path)
    if path.suffix.lower() != ".obj":
        return None
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.startswith(b"f ") or line.startswith(b"f\t"):
                count += 1
    return count


def _view_signature():
    signature = {}
    for folder in PROTECTED_DIRS:
        for name in VIEW_NAMES:
            path = folder / "views" / f"{name}.png"
            if not path.is_file():
                raise RuntimeError(f"protected view is missing: {path}")
            data = path.read_bytes()
            signature[str(path.resolve())] = hashlib.sha256(data).hexdigest()
    return signature


def _check_views(signature):
    current = _view_signature()
    if current != signature:
        raise RuntimeError("a protected view folder changed")


if __name__ == "__main__":
    main()
