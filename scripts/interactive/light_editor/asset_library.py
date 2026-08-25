# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find USD assets the editor can add to a scene, and copy them in.

Pipeline scenes keep their geometry beside the scene JSON in one of two shapes::

    objects/<category>/<asset_id>/usd/<asset_id>.usd     + material/  misc/
    objects/cousins/<Task>/<variant>/<asset_id>.usd      (textures embedded)

The *bundle directory* -- the one holding the USD and everything it references
relatively -- is the USD's parent, or its grandparent when the parent is
literally ``usd``. Imports copy the whole bundle so relative references keep
resolving, and always copy it into the target scene so ``usd_path`` stays
relative to the scene JSON and the scene remains portable.
"""

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asset_import import MESH_SUFFIXES, USD_SUFFIXES  # noqa: E402

#: Directories that never hold importable props: room-background libraries,
#: Omniverse layer/configuration directories, and pure-data directories.
#: Matched on exact names, never on substrings, so a prop whose filename merely
#: contains "background" still lists.
EXCLUDED_DIR_NAMES = frozenset({
    "background", "backgrounds", "gs_background",
    "mesh_backgrounds", "gs_backgrounds", "hdr_backgrounds",
    "configuration", "material", "materials", "textures", "misc",
    "normalized_obj", "urdf",
})

# Payload USDs recognised by name. Deliberately narrow: some real assets end in
# "input_mesh", so the mesh filter below must not be applied to USDs.
_NOT_A_USD_ASSET = re.compile(r"instanceable_meshes", re.I)

# A scanned room saved loose in a scene directory. Matched on the room naming
# convention -- a stem that is or ends in ``mesh_background``/``gs_background``
# -- never on "background" as a substring.
_ROOM_STEM = re.compile(r"(?:^|_)(?:mesh|gs)_background$", re.I)

# Cap on discovery, so walking a huge root cannot hang the server.
MAX_ASSETS = 4000


def md5(path):
    """MD5 of a file, matching what OmniGibson stores as ``expected_file_hash``."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_dir(usd_path):
    """Directory holding *usd_path* and everything it references relatively."""
    usd_path = Path(usd_path)
    parent = usd_path.parent
    return parent.parent if parent.name == "usd" else parent


def default_roots(scene_json):
    """Directories to search for importable assets: the scene's own
    ``objects/`` tree first, then its sibling scenes."""
    scene_dir = Path(scene_json).resolve().parent
    roots = []
    own = scene_dir / "objects"
    if own.is_dir():
        roots.append(own)
    siblings = scene_dir.parent
    if siblings.is_dir() and siblings != scene_dir:
        roots.append(siblings)
    return roots


def _slug(text):
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", str(text)).strip("_-").lower()
    return cleaned or "object"


def _category_for(usd_path, root):
    """Grouping label for an asset.

    ``objects/blue_cup/gtmlvp/usd/gtmlvp.usd`` -> ``blue_cup``;
    ``objects/cousins/PutOnTop/cousin_013/vyeaba.usd`` -> ``putontop``.
    """
    return _slug(bundle_dir(usd_path).parent.name) or _slug(Path(root).name)


def discover_assets(roots, scene_dir=None):
    """List importable USD assets under *roots*.

    Args:
        roots (iterable[Path]): Directories to walk.
        scene_dir (Path or None): The scene being edited; assets already inside
            it are flagged ``local`` so the browser can say an import needs no
            copy.

    Returns:
        list[dict]: One entry per asset, sorted by category then id. Each has
            ``key`` (a stable opaque id), ``usd`` (absolute path), ``category``,
            ``asset_id``, ``source`` (the scene or root it came from),
            ``size_mb`` and ``local``.
    """
    scene_dir = Path(scene_dir).resolve() if scene_dir else None
    found = {}

    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            here = Path(dirpath)
            # Prune excluded subtrees instead of walking them.
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".") and _slug(d) not in EXCLUDED_DIR_NAMES
            ]
            for filename in filenames:
                suffix = Path(filename).suffix.lower()
                kind = "usd" if suffix in USD_SUFFIXES else (
                    "mesh" if suffix in MESH_SUFFIXES else None)
                if kind is None:
                    continue
                usd = here / filename
                if _ROOM_STEM.search(usd.stem):
                    continue
                if kind == "usd" and _NOT_A_USD_ASSET.search(usd.stem):
                    continue
                key = str(usd)
                if key in found:
                    continue
                try:
                    size = usd.stat().st_size
                except OSError:
                    continue
                bundle = bundle_dir(usd)
                local = scene_dir is not None and _is_within(usd, scene_dir)
                found[key] = {
                    "key": hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
                    "usd": key,
                    "kind": kind,
                    "asset_id": usd.stem,
                    # A loose mesh has no bundle layout; its own name is the category.
                    "category": (_slug(usd.stem) if kind == "mesh"
                                 else _category_for(usd, root)),
                    "variant": usd.stem if kind == "mesh" else bundle.name,
                    "format": suffix.lstrip("."),
                    # Local assets are labelled with the scene's name.
                    "source": (scene_dir.name if local else _source_label(usd, root)),
                    "size_bytes": size,
                    "size_mb": round(size / 1e6, 2),
                    "local": local,
                }
                if len(found) >= MAX_ASSETS:
                    break
            if len(found) >= MAX_ASSETS:
                break

    assets = _deduplicate(_without_source_meshes(found.values()))
    return sorted(assets, key=lambda a: (a["category"], a["variant"], a["asset_id"]))


def _content_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deduplicate(assets):
    """Collapse byte-identical copies of the same asset into one entry.

    The pipeline copies each asset into every scene that uses it, so the same
    file appears many times. Deduplication is by content, not asset id -- the
    same id can name different geometry in different scenes. Only files that
    share an exact size are hashed.
    """
    by_size = {}
    for asset in assets:
        by_size.setdefault(asset.get("size_bytes"), []).append(asset)

    unique = []
    for group in by_size.values():
        if len(group) == 1:
            unique.append(group[0])
            continue
        by_hash = {}
        for asset in group:
            try:
                digest = _content_hash(asset["usd"])
            except OSError:
                # Unreadable: keep it; the import attempt will report the problem.
                unique.append(asset)
                continue
            by_hash.setdefault(digest, []).append(asset)
        for copies in by_hash.values():
            unique.append(_representative(copies))

    return _disambiguate(unique)


def _representative(copies):
    """Pick which copy of an asset to offer, and record the others.

    A copy already inside the scene being edited wins: it needs no copying.
    """
    chosen = next((c for c in copies if c["local"]), None) or min(
        copies, key=lambda c: c["usd"])
    sources = sorted({c["source"] for c in copies})
    chosen = dict(chosen)
    chosen["copies"] = len(copies)
    chosen["sources"] = sources
    return chosen


def _disambiguate(assets):
    """Append the source scene to entries that would otherwise share a label."""
    counts = {}
    for asset in assets:
        counts[(asset["category"], asset["variant"])] = counts.get(
            (asset["category"], asset["variant"]), 0) + 1
    for asset in assets:
        if counts[(asset["category"], asset["variant"])] > 1:
            asset["variant"] = f"{asset['variant']} ({asset['source']})"
    return assets


# Meshes that are build inputs or colliders for an asset, not assets themselves.
# No `\b`: these names run into underscores, which are word characters.
_NOT_AN_ASSET = re.compile(r"collision|convex|decomp|_hull|input_mesh", re.I)


def _without_source_meshes(assets):
    """Drop raw meshes that are inputs to, or parts of, a USD in the same bundle."""
    usd_bundles = {str(bundle_dir(a["usd"])) for a in assets if a["kind"] == "usd"}
    kept = []
    for asset in assets:
        if asset["kind"] == "mesh":
            if _NOT_AN_ASSET.search(Path(asset["usd"]).stem):
                continue
            # A mesh sitting where a USD was already found is that USD's source.
            if str(bundle_dir(asset["usd"])) in usd_bundles:
                continue
        kept.append(asset)
    return kept


def _source_label(usd, root):
    """Which scene (or root) an asset came from, for display."""
    try:
        relative = Path(usd).resolve().relative_to(root)
    except ValueError:
        return root.name
    return relative.parts[0] if len(relative.parts) > 1 else root.name


def _is_within(path, directory):
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
        return True
    except ValueError:
        return False


def resolve_key(key, assets):
    """Look up a discovered asset by its opaque key.

    The browser only sends keys, never paths, so an add request cannot name an
    arbitrary file.
    """
    for asset in assets:
        if asset["key"] == key:
            return asset
    return None


def unique_object_name(existing, category):
    """First free ``<category>_<n>``, matching how the pipeline names objects."""
    base = _slug(category)
    index = 0
    while f"{base}_{index}" in existing:
        index += 1
    return f"{base}_{index}"


def _copytree_atomic(src, dst):
    """Copy a directory into place, leaving nothing behind on failure."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dst.name}.", dir=dst.parent))
    try:
        payload = staging / "payload"
        shutil.copytree(src, payload, symlinks=False)
        os.replace(payload, dst)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def import_asset(asset, scene_json):
    """Make *asset* available to a scene and return its scene-relative usd_path.

    An asset already inside the scene directory is referenced where it lies.
    Anything else has its whole bundle directory copied to
    ``<scene>/objects/<category>/<variant>/``, so the imported scene stays
    self-contained and portable.

    Args:
        asset (dict): An entry from :func:`discover_assets`.
        scene_json (str or Path): The scene being edited.

    Returns:
        tuple[str, Path]: The POSIX relative ``usd_path`` to store in
        ``init_info``, and the absolute path of the USD that path resolves to.
    """
    scene_dir = Path(scene_json).resolve().parent
    usd = Path(asset["usd"]).resolve()

    if _is_within(usd, scene_dir):
        return usd.relative_to(scene_dir).as_posix(), usd

    bundle = bundle_dir(usd)
    target = scene_dir / "objects" / asset["category"] / bundle.name
    imported = target / usd.relative_to(bundle)

    if imported.exists():
        # Same asset re-imported (two apples): the bundle on disk is already right.
        return imported.relative_to(scene_dir).as_posix(), imported
    if target.exists():
        # Same variant name, different asset: use a fresh directory.
        suffix = 1
        while (scene_dir / "objects" / asset["category"] / f"{bundle.name}_{suffix}").exists():
            suffix += 1
        target = scene_dir / "objects" / asset["category"] / f"{bundle.name}_{suffix}"
        imported = target / usd.relative_to(bundle)

    _copytree_atomic(bundle, target)
    return imported.relative_to(scene_dir).as_posix(), imported


def object_spec(name, usd_relative, usd_absolute, category, *, position, orientation, scale):
    """Build the ``init_info`` and registry state for one added object.

    ``init_info`` mirrors what ``USDObject`` serializes; OmniGibson recomputes
    ``expected_file_hash`` and compares it on load. Velocities are zero: the
    object has never been simulated.
    """
    return {
        "name": name,
        "category": category,
        "usd_path": usd_relative,
        "usd_absolute": str(usd_absolute),
        "init_info": {
            "class_module": "omnigibson.objects.usd_object",
            "class_name": "USDObject",
            "args": {
                "name": name,
                "usd_path": usd_relative,
                "category": category,
                "scale": [float(v) for v in scale],
                "expected_file_hash": md5(usd_absolute),
            },
        },
        "registry": {
            "is_asleep": False,
            "root_link": {
                "pos": [float(v) for v in position],
                "ori": [float(v) for v in orientation],
                "lin_vel": [0.0, 0.0, 0.0],
                "ang_vel": [0.0, 0.0, 0.0],
            },
            # One zero per degree of freedom. Required: OmniGibson reads
            # `state["joint_pos"]` for any object with joints, so an articulated
            # asset without these keys fails to load.
            **_joint_state(usd_absolute),
            "non_kin": {},
        },
    }


def _joint_state(usd_absolute):
    """Rest joint state for an articulated asset, or {} for a rigid one."""
    from scene_io import read_usd_joints

    count = len(read_usd_joints(usd_absolute))
    if not count:
        return {}
    return {"joint_pos": [0.0] * count, "joint_vel": [0.0] * count}


def main():
    """List what would be importable for a scene -- a debugging aid."""
    import argparse

    parser = argparse.ArgumentParser(description="List importable USD assets")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--root", action="append", default=None)
    args = parser.parse_args()

    scene_json = Path(args.scene).resolve()
    roots = [Path(r) for r in args.root] if args.root else default_roots(scene_json)
    assets = discover_assets(roots, scene_json.parent)
    for asset in assets:
        flag = "local" if asset["local"] else asset["source"]
        print(f"{asset['category']:24s} {asset['variant']:22s} "
              f"{asset['size_mb']:7.2f} MB  {flag}")
    print(f"\n{len(assets)} asset(s) from {', '.join(str(r) for r in roots)}")


if __name__ == "__main__":
    main()
