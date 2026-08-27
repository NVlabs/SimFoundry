# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Find the scenes on this machine, and remember which ones were opened.

A scene lives in a directory of ``<name>_scene_state_<tag>.json`` files. The
``latest`` tag is what downstream stages load; the rest are history. The
catalog is one row per directory, with the other saves offered as variants.
Recently opened scenes are recorded separately, because opening a file does
not change its mtime.
"""

import json
import os
import re
import time
from pathlib import Path

from scene_io import (
    SceneEditError,
    atomic_write_text,
    background_id_from_records,
    iter_objects,
)

#: What every downstream stage loads by default.
CANONICAL_TAG = "latest"

#: Scene state documents, whatever their tag.
STATE_GLOB = "*_scene_state_*.json"

_STATE_RE = re.compile(r"^(?P<scene>.+)_scene_state_(?P<tag>.+)$")

# A trailing "20260814_091423" or "20260814_091423_478750_cab729d9" is the
# save's timestamp, not part of its tag. The leading underscore is optional
# because a bare timestamp is itself a common tag.
_STAMP_RE = re.compile(r"(?:^|_)(?P<stamp>\d{8}_\d{6})(?:_\d+)?(?:_[0-9a-f]{6,})?$")

# A file manager's duplicate: "..._scene_state_latest (Copy).json", "... copy".
# Recognised so it can be offered last -- its mtime is the copy's, not the
# save's -- but never dropped.
_DUPLICATE_RE = re.compile(r"[ _]\(?(?:another )?copy\)?(?: \d+)?$", re.IGNORECASE)

#: Directories under a scenes root that are never scenes.
_SKIP_DIRS = frozenset({"objects", "mesh_backgrounds", "cousins", "__pycache__"})


def parse_state_filename(path):
    """Split ``<scene>_scene_state_<tag>.json`` into its parts.

    Args:
        path (str or Path): A scene state document.

    Returns:
        tuple[str, str] or None: ``(scene_name, tag)``, or None if the name is
        not a scene state document at all.
    """
    match = _STATE_RE.match(Path(path).stem)
    if not match:
        return None
    return match.group("scene"), match.group("tag")


#: The keys ``og.sim.save()`` writes; enough to tell a scene document from the
#: other JSON that lives beside one.
_SCENE_KEYS = ("objects_info", "state")

#: JSON beside a scene that is never a scene; skipped without parsing.
_NOT_SCENE_SUFFIXES = ("_export.json", ".background.json")


def looks_like_scene(path):
    """Whether *path* is a scene document, judged by content rather than name.

    Openable scenes do not all follow the ``*_scene_state_*.json`` naming --
    the pipeline's ``reconstructed_og_scene.json`` loads directly.
    """
    path = Path(path)
    if any(path.name.endswith(suffix) for suffix in _NOT_SCENE_SUFFIXES):
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        return False
    return isinstance(document, dict) and all(key in document for key in _SCENE_KEYS)


def variant_label(tag):
    """A tag as something worth reading in a list.

    ``light_edit_20260814_091423_478750_cab729d9`` -> ``light edit · 2026-08-14 09:14``.
    """
    if tag == CANONICAL_TAG:
        return "latest"
    stamp = _STAMP_RE.search(tag)
    if not stamp:
        return tag.replace("_", " ")
    name = tag[: stamp.start()].replace("_", " ").strip()
    raw = stamp.group("stamp")
    when = f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]} {raw[9:11]}:{raw[11:13]}"
    return f"{name} · {when}" if name else f"saved · {when}"


def describe_scene(scene_json, *, dataset_dir=None, robot_asset_dir=None):
    """Summarise one scene document without loading any geometry.

    A document that will not parse becomes a row with an error rather than an
    exception, so one broken file cannot empty the launcher.

    Args:
        scene_json (str or Path): Path to a scene state document.
        dataset_dir (str or Path or None): OmniGibson's ``gm.DATA_PATH``.
            ``DatasetObject`` assets name geometry by category/model rather
            than by path; without this root they report as unchecked.
        robot_asset_dir (str or Path or None): Root of
            ``omnigibson-robot-assets``, for the same reason.

    Returns:
        dict: ``name``, ``path``, ``tag``, ``label``, ``mtime``, ``bytes``, and
        either ``objects``/``props``/``missing``/``unchecked``/``background``/
        ``robot`` or ``error``.
    """
    path = Path(scene_json)
    parsed = parse_state_filename(path)
    name, tag = parsed if parsed else (path.stem, "")
    try:
        stat = path.stat()
    except OSError as e:
        return {"name": name, "path": str(path), "tag": tag,
                "label": variant_label(tag), "error": str(e)}

    summary = {
        "name": name,
        "path": str(path),
        "tag": tag,
        "label": variant_label(tag),
        "mtime": stat.st_mtime,
        "bytes": stat.st_size,
    }
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("not a JSON object")
        # usd_facts=False keeps this from opening every asset in the scene.
        records = list(iter_objects(document, path, robot_asset_dir=robot_asset_dir,
                                    dataset_dir=dataset_dir, usd_facts=False))
        summary.update({
            "objects": len(records),
            "props": sum(1 for r in records if r["editable"]),
            # Assets looked for and not found, including class-named ones.
            "missing": sum(1 for r in records if r["usd_status"] == "missing"),
            # Assets that could not be checked, kept apart from "missing".
            "unchecked": sum(1 for r in records if r["usd_status"] == "unchecked"),
            "background": background_id_from_records(records),
            "robot": next((r["category"] for r in records if r["kind"] == "robot"), None),
            "composed_from": (document.get("metadata") or {}).get("composed_from"),
        })
    # AttributeError covers a JSON null where an object should be, e.g.
    # `"objects_info": null` reaching `iter_objects` as `None.get(...)`.
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
        summary["error"] = f"{type(e).__name__}: {e}"
    return summary


def _state_files(directory):
    """Scene state documents in one directory, newest first."""
    try:
        found = [p for p in Path(directory).glob(STATE_GLOB) if p.is_file()]
        # Also admit scene documents that do not follow the naming rule; only
        # those strays are parsed.
        named = set(found)
        for path in sorted(Path(directory).glob("*.json")):
            if path in named or not path.is_file():
                continue
            if looks_like_scene(path):
                found.append(path)
    except OSError:
        return []
    # Name is a stable secondary key for files written in the same second.
    return sorted(found, key=lambda p: (-p.stat().st_mtime, p.name))


def _canonical(directory, files, open_scene=None):
    """The file a row opens: the open scene if it is among *files*, else
    ``<dir>_scene_state_latest.json``, else any ``latest`` tag, else the newest.
    """
    if open_scene is not None:
        opened = Path(open_scene)
        if opened in files:
            return opened
    preferred = Path(directory) / f"{Path(directory).name}_scene_state_{CANONICAL_TAG}.json"
    if preferred in files:
        return preferred
    for path in files:
        parsed = parse_state_filename(path)
        if parsed and parsed[1] == CANONICAL_TAG:
            return path
    return files[0] if files else None


def _is_duplicate(path):
    """Whether *path* is a file manager's copy of some other save."""
    return bool(_DUPLICATE_RE.search((parse_state_filename(path) or ("", ""))[1]))


def _pick_variants(files, canonical, limit):
    """Which of *files* a row offers: at most *limit*, always including
    *canonical*, with file-manager duplicates ranked last."""
    ranked = sorted(files, key=lambda path: (path != canonical, _is_duplicate(path)))
    kept = set(ranked[:limit])
    return [path for path in files if path in kept]


def _scene_row(directory, *, variant_limit=12, dataset_dir=None, robot_asset_dir=None,
               open_scene=None):
    """One catalog row for one scene directory, or None if it holds no scenes."""
    files = _state_files(directory)
    if not files:
        return None
    canonical = _canonical(directory, files, open_scene=open_scene)
    row = describe_scene(canonical, dataset_dir=dataset_dir,
                         robot_asset_dir=robot_asset_dir)
    row["dir"] = str(Path(directory))
    variants = _pick_variants(files, canonical, variant_limit)
    row["variants"] = [
        {
            "path": str(path),
            "label": variant_label((parse_state_filename(path) or ("", ""))[1]),
            "mtime": path.stat().st_mtime,
            "canonical": path == canonical,
        }
        for path in variants
    ]
    # Count what the row offers, not what the directory holds.
    row["variant_count"] = len(variants)
    row["variant_hidden"] = len(files) - len(variants)
    return row


#: Where SimFoundry scenes live, relative to the repository root, and the
#: launcher label for each. ``assets/scenes`` ships with the checkout; ``Data``
#: holds scenes reconstructed from a user's own videos.
SCENE_SOURCES = (
    ("assets/scenes", "preset", "SimFoundry scenes"),
    ("Data", "generated", "Generated from your videos"),
)

#: How deep below each source root a scene directory sits: presets are
#: `assets/scenes/<Group>/<scene>/` (e.g. `DROID/`, `YAM/`), generated scenes
_SOURCE_DEPTH = {"assets/scenes": 2, "Data": 2}

#: The two source roots as paths, for the naming logic below.
_PRESET_ROOT, _GENERATED_ROOT = (
    next(Path(relative) for relative, key, _ in SCENE_SOURCES if key == wanted)
    for wanted in ("preset", "generated")
)


def scene_roots(scene_json=None, repo_root=None, extra=()):
    """Directories that hold scene directories, in the order they are searched.

    The open scene's own root comes first, then *extra*, then the checkout's
    two source roots.
    """
    roots, seen = [], set()

    def add(candidate):
        if candidate is None:
            return
        resolved = Path(candidate).expanduser()
        try:
            resolved = resolved.resolve()
        except OSError:
            return
        if resolved in seen or not resolved.is_dir():
            return
        seen.add(resolved)
        roots.append(resolved)

    if scene_json:
        add(Path(scene_json).resolve().parent.parent)
    for candidate in extra:
        add(candidate)
    for relative, _, _ in SCENE_SOURCES if repo_root else ():
        source = Path(repo_root) / relative
        add(source)
        # Sources two levels deep contribute their children as roots too. The
        # source itself stays listed: it is the confinement boundary that
        # `resolve_scene_path` checks against.
        if _SOURCE_DEPTH.get(relative, 1) > 1:
            try:
                for child in sorted(p for p in source.iterdir() if p.is_dir()):
                    add(child)
            except OSError:
                pass
    return roots


def _preset_twin(run):
    """Whether a preset scene in the same checkout is also called ``run.name``."""
    repo = run.parent
    for _ in _GENERATED_ROOT.parts:
        repo = repo.parent
    if repo / _GENERATED_ROOT != run.parent:
        return False            # not the layout SCENE_SOURCES describes
    return (repo / _PRESET_ROOT / run.name).is_dir()


def generated_scene_name(path):
    """What to call a reconstructed scene, or None to keep its filename's name.

    Reconstructed scenes are all saved as
    ``reconstructed_og_scene_scene_state_<tag>.json``, so the run directory
    under ``Data`` supplies the name. The stage directory (e.g. ``s14_og``) is
    appended only when needed to tell rows apart: when a run holds more than
    one scene, or shares its name with a preset scene. Only applied to the
    generated tree; a preset scene's filename is already its own name.

    Args:
        path (str or Path): A scene state document under ``Data``.

    Returns:
        str or None: The name to show, or None to leave it alone.
    """
    directory = Path(path).parent
    run = directory.parent
    if not run.name:
        return None
    if run.name == _GENERATED_ROOT.name:
        # A scene directly in the generated root has no run above it; its own
        # directory supplies the name.
        return directory.name or None
    try:
        siblings = [
            child for child in run.iterdir()
            if child.is_dir() and any(child.glob("*_scene_state_*.json"))
        ]
    except OSError:
        siblings = []
    if len(siblings) > 1 or _preset_twin(run):
        return f"{run.name} · {directory.name}"
    return run.name


def scene_source(path, repo_root):
    """Which of `SCENE_SOURCES` *path* came from: its key, or None.

    Read off the path, not the root that discovered it, so the answer does not
    depend on which root found the scene first.
    """
    if not repo_root:
        return None
    try:
        resolved = Path(path).resolve()
    except OSError:
        return None
    for relative, key, _ in SCENE_SOURCES:
        try:
            root = (Path(repo_root) / relative).resolve()
        except OSError:
            continue
        if resolved == root or root in resolved.parents:
            return key
    return None


def discover_scenes(roots, *, limit=400, dataset_dir=None, robot_asset_dir=None,
                    open_scene=None):
    """Every scene directory under *roots*, newest save first.

    A root may itself be a scene directory.

    Args:
        roots (iterable): Directories to search, one level deep.
        limit (int): Stop after this many rows, so a misconfigured root cannot
            hang the launcher.
        dataset_dir (str or Path or None): Passed to :func:`describe_scene`.
        robot_asset_dir (str or Path or None): Likewise.
        open_scene (str or Path or None): The scene the editor holds open; its
            directory's row opens it rather than the newest sibling save.

    Returns:
        list[dict]: Rows from :func:`_scene_row`, deduplicated by directory.
    """
    rows, seen = [], set()
    for root in roots:
        root = Path(root)
        candidates = [root]
        try:
            candidates += sorted(p for p in root.iterdir() if p.is_dir())
        except OSError:
            continue
        for directory in candidates:
            resolved = str(directory.resolve())
            if resolved in seen or directory.name in _SKIP_DIRS:
                continue
            seen.add(resolved)
            row = _scene_row(directory, dataset_dir=dataset_dir,
                             robot_asset_dir=robot_asset_dir, open_scene=open_scene)
            if row is None:
                continue
            row["root"] = str(root)
            rows.append(row)
            if len(rows) >= limit:
                return sorted(rows, key=lambda r: -(r.get("mtime") or 0))
    return sorted(rows, key=lambda r: -(r.get("mtime") or 0))


def resolve_scene_path(candidate, allowed_roots):
    """Check a path the browser asked to open, before anything acts on it.

    The path is confined to the roots the catalog searched, so "open a scene"
    cannot become "read any JSON on this machine".

    Args:
        candidate (str): Path from the client.
        allowed_roots (iterable): Directories the catalog covers.

    Returns:
        Path: The resolved, existing scene document.

    Raises:
        SceneEditError: If the path is malformed, missing, not a scene state
            document, or outside every allowed root.
    """
    if not isinstance(candidate, str) or not candidate.strip():
        raise SceneEditError("a scene path is required")
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        raise SceneEditError(f"scene path must be absolute: {candidate}")
    try:
        path = path.resolve(strict=True)
    except OSError as e:
        raise SceneEditError(f"no such scene: {candidate} ({e.strerror})") from None
    if not path.is_file():
        raise SceneEditError(f"not a file: {path}")
    # Confinement before content: `looks_like_scene` reads the file, so the
    # root check must run first.
    roots = [Path(r).resolve() for r in allowed_roots]
    if not any(path == root or root in path.parents for root in roots):
        listed = ", ".join(str(r) for r in roots) or "none"
        raise SceneEditError(
            f"{path} is outside the scene roots this server searches ({listed}); "
            "restart with --scene-root to include it"
        )
    if not looks_like_scene(path):
        raise SceneEditError(
            f"{path.name} is not a scene document "
            f"(no {' or '.join(_SCENE_KEYS)} block in it)"
        )
    return path


# --- the recents list --------------------------------------------------------
# Opening a scene does not change its mtime, so recency is recorded here --
# outside the repository, since it is one person's history on one machine.

def default_recents_path():
    """Where the recents list lives, honouring ``XDG_STATE_HOME``."""
    base = os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state")
    return Path(base) / "simfoundry" / "light_editor" / "recent_scenes.json"


class RecentScenes:
    """A most-recently-opened list of scene paths, persisted as JSON.

    An unreadable or corrupt file is treated as an empty history and a failed
    write is dropped: the launcher must start even if this cache is broken.
    """

    VERSION = 1

    def __init__(self, path=None, limit=25):
        self.path = Path(path) if path else default_recents_path()
        self.limit = limit
        self._entries = self._read()

    def _read(self):
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(document, dict) or document.get("version") != self.VERSION:
            return []
        entries = document.get("scenes")
        if not isinstance(entries, list):
            return []
        clean = []
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                clean.append({
                    "path": entry["path"],
                    "opened": float(entry.get("opened") or 0.0),
                })
        return clean[: self.limit]

    def _write(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                self.path,
                json.dumps({"version": self.VERSION, "scenes": self._entries}, indent=1),
            )
        except OSError:
            pass        # history is a convenience; losing it is not an error

    def record(self, scene_json, when=None):
        """Move *scene_json* to the front of the list."""
        path = str(Path(scene_json).resolve())
        self._entries = [e for e in self._entries if e["path"] != path]
        self._entries.insert(0, {"path": path, "opened": float(when or time.time())})
        self._entries = self._entries[: self.limit]
        self._write()
        return self._entries

    def forget(self, scene_json):
        path = str(Path(scene_json).resolve())
        before = len(self._entries)
        self._entries = [e for e in self._entries if e["path"] != path]
        if len(self._entries) != before:
            self._write()
        return before - len(self._entries)

    def entries(self, *, describe=True, dataset_dir=None, robot_asset_dir=None):
        """Recent scenes that still exist, newest first.

        Moved or deleted scenes are dropped rather than listed as dead rows.
        """
        rows, live = [], []
        for entry in self._entries:
            path = Path(entry["path"])
            if not path.is_file():
                continue
            live.append(entry)
            row = describe_scene(path, dataset_dir=dataset_dir,
                                 robot_asset_dir=robot_asset_dir) if describe else {
                "name": (parse_state_filename(path) or (path.stem, ""))[0],
                "path": str(path),
            }
            row["opened"] = entry["opened"]
            row["dir"] = str(path.parent)
            rows.append(row)
        if len(live) != len(self._entries):
            self._entries = live
            self._write()
        return rows
