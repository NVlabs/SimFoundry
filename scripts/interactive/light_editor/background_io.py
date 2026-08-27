# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Attach a scanned room (mesh background) to a scene that has none.

Rooms form a registry: each is ``assets/mesh_backgrounds/<id>.usd`` with a
``<id>.background.json`` sidecar recording the pose the room was registered at,
in a frame whose support surface sits at z = 0 — the same convention as the
pipeline's world frame, so a registered pose transfers to a pipeline scene
directly. Height lands right; yaw and origin within the room are arbitrary, so
check the result (``--check``) and adjust with ``--position``/``--orientation``.

Usage:
    python background_io.py --scene <scene.json> --list
    python background_io.py --scene <scene.json> --background droid_desk_mesh
    python background_io.py --scene <scene.json> --check
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scene_io import (  # noqa: E402
    SceneEditError,
    atomic_write_text,
    iter_objects,
    load_scene,
    save_scene,
)

#: Default directory of scanned rooms, under the shared ``assets`` tree.
DEFAULT_BACKGROUND_SUBDIR = "assets/backgrounds/mesh_backgrounds"

#: The object name consumers expect. ``scene_io.iter_objects`` recognises a
#: room only by exact name (``mesh_background``/``mesh_background_<n>``/
#: ``gs_background``) or its stamped ``mesh_background`` category, and saved
#: camera rigs are keyed off a recognised background.
BACKGROUND_OBJECT_NAME = "mesh_background_0"

#: Radius of the column searched for mesh beneath a prop, in metres.
SURFACE_PROBE_RADIUS = 0.03


def background_roots(scene_json=None, repo_root=None, extra=()):
    """Directories that hold scanned rooms, in the order they are searched.

    The scene's own asset tree is searched first: a scene references its room
    relatively (``../../mesh_backgrounds/<id>.usd``), which only resolves
    within one tree.

    Args:
        scene_json (str or Path or None): Scene being edited.
        repo_root (str or Path or None): This checkout, for ``assets/``.
        extra (iterable): Additional roots, searched after the scene's own.

    Returns:
        list[Path]: Existing directories, deduplicated, most specific first.
    """
    roots, seen = [], []

    def add(candidate):
        if candidate is None:
            return
        path = Path(candidate).expanduser()
        if not path.is_dir():
            return
        # Deduplicate by resolved path, but keep it unresolved so paths written
        # into a scene stay inside the checkout rather than through a symlink.
        key = os.path.realpath(path)
        if key in seen:
            return
        seen.append(key)
        roots.append(path)

    if scene_json:
        scene_dir = Path(scene_json).resolve().parent
        # A scene sits at <root>/<scene>/..., so mesh_backgrounds/ is a sibling
        # of the scene directory or of the root above it.
        for parent in (scene_dir.parent, scene_dir.parent.parent):
            add(parent / "mesh_backgrounds")
    for candidate in extra:
        add(candidate)
    if repo_root:
        add(Path(repo_root) / DEFAULT_BACKGROUND_SUBDIR)
    return roots


def _read_sidecar(path):
    """Parse a ``<id>.background.json``, or return None if it is unusable."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _sidecar_pose(document):
    """Pull (position, orientation) out of a sidecar, or None.

    The published ``pose`` wins over ``registration.pose``: it is what every
    scene using the room was authored against.
    """
    for key in ("pose", "registration"):
        block = document.get(key) or {}
        pose = block.get("pose") if key == "registration" else block
        if not isinstance(pose, dict):
            continue
        position, orientation = pose.get("position"), pose.get("orientation")
        if (
            isinstance(position, list) and len(position) == 3
            and isinstance(orientation, list) and len(orientation) == 4
        ):
            return [float(v) for v in position], [float(v) for v in orientation]
    return None


def discover_backgrounds(roots):
    """Every scanned room found under *roots*.

    A room is a ``.usd`` with a ``.background.json`` beside it; the sidecar is
    required because it carries the registered pose.

    Args:
        roots (iterable): Directories from :func:`background_roots`.

    Returns:
        list[dict]: ``id``, ``label``, ``usd`` (Path), ``sidecar`` (Path),
        ``position``, ``orientation``, ``root`` (Path), and ``surface_height``
        (the z the registration puts the support surface at, or None).
    """
    found, seen = [], set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for sidecar in sorted(root.glob("*.background.json")):
            document = _read_sidecar(sidecar)
            if document is None:
                continue
            identifier = str(document.get("id") or sidecar.name[: -len(".background.json")])
            if identifier in seen:
                continue
            usd = root / str(document.get("usd_path") or f"{identifier}.usd")
            pose = _sidecar_pose(document)
            if not usd.is_file() or pose is None:
                continue
            support = (
                (document.get("registration") or {}).get("support_surface") or {}
            )
            seen.add(identifier)
            found.append({
                "id": identifier,
                "label": str(document.get("label") or identifier),
                "usd": usd,
                "sidecar": sidecar,
                "position": pose[0],
                "orientation": pose[1],
                "root": root,
                "surface_height": support.get("canonical_height_m"),
            })
    return found


def resolve_background(key, roots):
    """Find one room by id, or by a path to its USD.

    Args:
        key (str): A registry id such as ``droid_desk_mesh``, or a path.
        roots (iterable): Directories to search.

    Returns:
        dict: As :func:`discover_backgrounds`.

    Raises:
        SceneEditError: When nothing matches, naming what is available.
    """
    available = discover_backgrounds(roots)
    for background in available:
        if background["id"] == key:
            return background

    candidate = Path(str(key)).expanduser()
    if candidate.suffix in (".usd", ".usda", ".usdc") and candidate.is_file():
        for background in available:
            if os.path.realpath(background["usd"]) == os.path.realpath(candidate):
                return background
        raise SceneEditError(
            f"{candidate} has no {candidate.stem}.background.json beside it, so there "
            "is no registered pose to place it at. Register the room first, or pass "
            "--position/--orientation explicitly."
        )

    listed = ", ".join(b["id"] for b in available) or "none found"
    raise SceneEditError(f"unknown background {key!r}; available: {listed}")


def _reference_path(usd, scene_json):
    """How a room should be spelled inside a scene document.

    Rooms are referenced relatively (``../../mesh_backgrounds/<id>.usd``) so an
    asset tree can move without rewriting its scenes. Symlinks are not
    resolved: resolving a linked ``assets`` tree would write a path that leaves
    this repository.
    """
    scene_dir = Path(scene_json).parent.absolute()
    usd = Path(usd).absolute()
    try:
        relative = os.path.relpath(usd, scene_dir)
    except ValueError:  # different drive on Windows
        return str(usd)
    # Fall back to absolute when the relative path climbs too far.
    if relative.count("..") > 4:
        return str(usd)
    return relative


def attach_background(scene, scene_json, background, *, name=BACKGROUND_OBJECT_NAME,
                      position=None, orientation=None):
    """Add a scanned room to *scene*, in place.

    Purely additive: one new entry in each half of the document, nothing
    existing is touched, and no prop moves.

    Args:
        scene (dict): Parsed scene document, modified in place.
        scene_json (str or Path): Path the document came from; the room's
            ``usd_path`` is written relative to its parent.
        background (dict): A row from :func:`discover_backgrounds`.
        name (str): Object name. Leave it alone unless a scene already has one.
        position (list or None): Override the registered position.
        orientation (list or None): Override the registered orientation, XYZW.

    Returns:
        dict: What was added — ``name``, ``usd_path``, ``position``,
        ``orientation`` and ``expected_file_hash``.

    Raises:
        SceneEditError: If the scene already has an object by that name, or
            already has a background under any name.
    """
    init_info = scene.setdefault("objects_info", {}).setdefault("init_info", {})
    registry = (
        scene.setdefault("state", {})
        .setdefault("registry", {})
        .setdefault("object_registry", {})
    )
    if name in init_info or name in registry:
        raise SceneEditError(f"{name} is already in this scene")
    existing = [
        record["name"]
        for record in iter_objects(scene, scene_json, robot_asset_dir=None,
                                   usd_facts=False)
        if record["kind"] == "background"
    ]
    if existing:
        raise SceneEditError(
            f"this scene already has a background: {', '.join(existing)}. "
            "Remove it before attaching another, or the two will interpenetrate."
        )

    spec = background_spec(background, scene_json, name=name,
                           position=position, orientation=orientation)
    init_info[name] = spec["init_info"]
    registry[name] = spec["registry"]
    return {
        "name": name,
        "usd_path": spec["init_info"]["args"]["usd_path"],
        "position": spec["registry"]["root_link"]["pos"],
        "orientation": spec["registry"]["root_link"]["ori"],
        "expected_file_hash": spec["init_info"]["args"]["expected_file_hash"],
    }


def background_spec(background, scene_json, *, name=BACKGROUND_OBJECT_NAME,
                    position=None, orientation=None):
    """The two document halves a room needs, without touching a scene.

    Used by both :func:`attach_background` and the editor's pending-add path,
    so a room attached either way produces identical entries in the file.

    Args:
        background (dict): A row from :func:`discover_backgrounds`.
        scene_json (str or Path): Scene the ``usd_path`` will be relative to.
        name (str): Object name.
        position, orientation: Override the registered pose.

    Returns:
        dict: ``init_info`` and ``registry``, shaped as the scene document wants
        them, plus ``usd_absolute`` for whoever has to build a proxy from it.
    """
    usd = Path(background["usd"])
    position = list(position if position is not None else background["position"])
    orientation = list(orientation if orientation is not None else background["orientation"])
    return {
        "init_info": {
            "class_module": "omnigibson.objects.usd_object",
            "class_name": "USDObject",
            "args": {
                "name": name,
                "usd_path": _reference_path(usd, scene_json),
                "category": "mesh_background",
                # Scenery: without fixed_base the room falls when physics starts.
                "fixed_base": True,
                # OmniGibson recomputes this hash on load and refuses a mismatch.
                "expected_file_hash": hashlib.md5(usd.read_bytes()).hexdigest(),
            },
        },
        # No lin_vel/ang_vel: a fixed-base room has no velocity to replay.
        "registry": {
            "is_asleep": False,
            "root_link": {"pos": position, "ori": orientation},
            "non_kin": {},
        },
        "usd_absolute": str(usd),
    }


def table_centre(background):
    """The registered table centre for a room, or None if nobody has set one.

    Args:
        background (dict): A row from :func:`discover_backgrounds`.

    Returns:
        list[float] or None: ``[x, y, z]`` in scene coordinates.
    """
    document = _read_sidecar(background["sidecar"]) or {}
    support = (document.get("registration") or {}).get("support_surface") or {}
    centre = support.get("centre")
    if isinstance(centre, list) and len(centre) == 3:
        try:
            return [float(v) for v in centre]
        except (TypeError, ValueError):
            return None
    return None


def default_robot(background):
    """The robot(s) a room's sidecar prescribes, or None.

    Args:
        background (dict): A row from :func:`discover_backgrounds`.

    Returns:
        list[dict] or None: One entry per robot -- a bimanual room like the
        YAM workstation prescribes two. Each is ``{"init_info": ..., "registry":
        ...}``, a verbatim robot-shaped spec captured from a real settled
        scene -- not synthesized, since a valid ``joint_pos``/``controllers``
        layout for an arbitrary robot isn't derivable outside a live
        OmniGibson run.
    """
    document = _read_sidecar(background["sidecar"]) or {}
    specs = document.get("default_robot")
    if not isinstance(specs, list) or not specs:
        return None
    if not all(isinstance(s, dict) and "init_info" in s and "registry" in s for s in specs):
        return None
    return specs


def write_table_centre(background, centre, *, estimate=None):
    """Record where a room's table centre is, in its sidecar.

    The centre is a property of the furniture, so it lives with the room and
    every scene shot there shares it. The write is additive; the sidecar's
    other blocks are preserved untouched.

    Args:
        background (dict): A row from :func:`discover_backgrounds`.
        centre (sequence[float]): ``[x, y, z]``, scene coordinates.
        estimate (dict or None): What :func:`estimate_table` reported, stored
            alongside so a hand-placed point is distinguishable from a guess.

    Returns:
        Path: The sidecar that was written.

    Raises:
        SceneEditError: If the centre is not three finite numbers, or the
            sidecar cannot be read back.
    """
    if not isinstance(centre, (list, tuple)) or len(centre) != 3:
        raise SceneEditError("table centre must be three numbers")
    try:
        point = [float(v) for v in centre]
    except (TypeError, ValueError):
        raise SceneEditError("table centre must be three numbers") from None
    if not all(v == v and abs(v) != float("inf") for v in point):
        raise SceneEditError("table centre must be finite")

    path = Path(background["sidecar"])
    document = _read_sidecar(path)
    if document is None:
        raise SceneEditError(f"could not read {path}")

    registration = document.setdefault("registration", {})
    if not isinstance(registration, dict):
        registration = document["registration"] = {}
    support = registration.setdefault("support_surface", {})
    if not isinstance(support, dict):
        support = registration["support_surface"] = {}
    support["centre"] = [round(v, 6) for v in point]
    support["centre_source"] = "operator_placed"
    if estimate:
        support["centre_estimate"] = {
            k: estimate[k] for k in ("centre", "extent", "yaw_deg", "area_m2")
            if k in estimate
        }

    atomic_write_text(path, json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def estimate_table(background, props, *, cell=0.05, slab=0.06, up=0.85, near=0.70):
    """Guess a room's table centre from its scan. A seed, not a measurement.

    Up-facing faces near the registered support height, bounded to the
    neighbourhood of the props, grouped into 4-connected cells; the largest
    component wins. Horizontal geometry adjoining the table inflates the
    returned ``extent``, so treat it as a caveat, not a size.

    Args:
        background (dict): A row from :func:`discover_backgrounds`.
        props (sequence): XY positions of the props, used to bound the search.
        cell (float): Grid size for connectivity, metres.
        slab (float): Half-thickness of the horizontal band searched, metres.
        up (float): Minimum face-normal z to count as up-facing.
        near (float): How far from a prop a face may be, metres.

    Returns:
        dict: ``centre`` ([x, y, z]), ``extent`` ([long, short]), ``yaw_deg``,
        ``area_m2`` and ``caveat``; or None when nothing horizontal was found.
    """
    import numpy as np

    import splat_io
    from extract import load_visual_scene

    props = np.asarray(props, dtype=np.float64).reshape(-1, 2)
    if not len(props):
        return None

    if splat_io.is_nurec_usd(background["usd"]):
        # A Gaussian splat has no faces or normals; the marker is placed by hand.
        return None

    geometry, _, _, _, failures = load_visual_scene(str(background["usd"]), allow_texture=False)
    if geometry is None:
        raise SceneEditError(f"could not read {background['usd']}: {'; '.join(failures)}")
    mesh = geometry if hasattr(geometry, "faces") else geometry.to_geometry()

    x, y, z, w = background["orientation"]
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    verts = np.asarray(mesh.vertices) @ rotation.T + np.asarray(background["position"])
    tri = verts[np.asarray(mesh.faces)]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    usable = lengths > 1e-12
    normals[usable] /= lengths[usable, None]
    centres, areas = tri.mean(axis=1), 0.5 * lengths

    horizontal = usable & (normals[:, 2] > up) & (np.abs(centres[:, 2]) < slab)
    if not horizontal.any():
        return None
    points, weights = centres[horizontal], areas[horizontal]

    # Bounding to the props separates the meant surface from adjoining furniture.
    close = np.linalg.norm(points[:, None, :2] - props[None, :, :], axis=2).min(axis=1) < near
    if not close.any():
        return None
    points, weights = points[close], weights[close]

    cells = {}
    for index, point in enumerate(points):
        cells.setdefault((int(point[0] // cell), int(point[1] // cell)), []).append(index)
    seen, components = set(), []
    for start in cells:
        if start in seen:
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            cx, cy = stack.pop()
            group.append((cx, cy))
            # 4-connected: diagonal contact reconnects what this is separating.
            for neighbour in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if neighbour in cells and neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(group)

    group = max(components, key=lambda g: sum(weights[i] for c in g for i in cells[c]))
    index = [i for c in group for i in cells[c]]
    patch, area = points[index], weights[index]

    mean = np.average(patch[:, :2], axis=0, weights=area)
    delta = patch[:, :2] - mean
    covariance = (delta * area[:, None]).T @ delta / area.sum()
    values, vectors = np.linalg.eigh(covariance)
    long_axis = vectors[:, int(np.argmax(values))]
    if long_axis[1] < 0:
        long_axis = -long_axis
    short_axis = np.array([long_axis[1], -long_axis[0]])
    along, across = delta @ long_axis, delta @ short_axis
    # Area-weighted centroid, not mid-extent: thin horizontal bridges into
    # neighbouring furniture stretch the extent while carrying almost no area.
    # `extent` stays min/max on purpose -- it reports how much to disbelieve.
    centre = mean

    return {
        "centre": [float(centre[0]), float(centre[1]), float(np.median(patch[:, 2]))],
        "extent": [float(along.max() - along.min()), float(across.max() - across.min())],
        "yaw_deg": float(np.degrees(np.arctan2(long_axis[1], long_axis[0]))),
        "area_m2": float(area.sum()),
        "caveat": "scan estimate; horizontal geometry beside the table inflates it",
    }


def _background_vertices(usd, position, orientation):
    """World-space points of a room placed at a pose.

    Mesh vertices for a scanned room; gaussian centres for a Gaussian-splat
    one, which has no mesh prims. Imports lazily: only this function needs
    OpenUSD and trimesh, and reading a room is slow and memory-heavy.
    """
    import numpy as np

    import splat_io

    if splat_io.is_nurec_usd(usd):
        vertices = np.asarray(splat_io.load_centres(usd), dtype=np.float64)
    else:
        from extract import load_visual_scene

        geometry, _, _, _, failures = load_visual_scene(str(usd), allow_texture=False)
        if geometry is None:
            raise SceneEditError(
                f"could not read {usd}: {'; '.join(failures) or 'no geometry'}")
        if hasattr(geometry, "vertices"):
            vertices = np.asarray(geometry.vertices, dtype=np.float64)
        else:  # a trimesh Scene, which is what a multi-mesh room loads as
            vertices = np.asarray(geometry.to_geometry().vertices, dtype=np.float64)

    x, y, z, w = orientation
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return vertices @ rotation.T + np.asarray(position, dtype=np.float64)


def surface_report(scene, scene_json, name=BACKGROUND_OBJECT_NAME,
                   radius=SURFACE_PROBE_RADIUS):
    """Is there room geometry under each prop, and how far below?

    Answers numerically, without OmniGibson, whether the room actually sits
    under the props. A gap is measured from the prop's *origin*, not its base,
    so a healthy result is a small positive number of centimetres rather than
    zero.

    Args:
        scene (dict): Scene document carrying the background.
        scene_json (str or Path): Path it came from.
        name (str): Background object name.
        radius (float): Half-width of the column searched under each prop.

    Returns:
        dict: ``background``, ``bounds``, ``rows`` (one per editable prop, with
        ``name``, ``prop_z``, ``surface_z`` and ``gap``; ``surface_z`` is None
        where the column found nothing) and ``covered``/``total`` counts.

    Raises:
        SceneEditError: If the scene has no such background.
    """
    import numpy as np

    records = {r["name"]: r for r in
               iter_objects(scene, scene_json, robot_asset_dir=None, usd_facts=False)}
    if name not in records:
        raise SceneEditError(f"{name} is not in this scene")
    background = records[name]
    if background["usd"] is None:
        raise SceneEditError(
            f"{name} references {background['usd_reference']}, which is not on this machine"
        )

    world = _background_vertices(
        background["usd"], background["position"], background["orientation"]
    )
    rows = []
    for record in records.values():
        if not record["editable"]:
            continue
        prop = np.asarray(record["position"][:2], dtype=np.float64)
        column = world[(np.abs(world[:, :2] - prop) < radius).all(axis=1)]
        # Nearest surface within ±0.30 m, not nearest below: scans are noisy
        # and a prop's origin is not its base, so a thin object can sit
        # slightly under the scanned surface without the room being misplaced.
        window = column[np.abs(column[:, 2] - record["position"][2]) < 0.30]
        surface = (
            float(window[np.argmin(np.abs(window[:, 2] - record["position"][2])), 2])
            if len(window) else None
        )
        rows.append({
            "name": record["name"],
            "category": record["category"],
            "prop_z": float(record["position"][2]),
            "surface_z": surface,
            "gap": None if surface is None else float(record["position"][2] - surface),
        })
    rows.sort(key=lambda row: row["name"])
    return {
        "background": name,
        "bounds": {
            "min": [float(v) for v in world.min(axis=0)],
            "max": [float(v) for v in world.max(axis=0)],
        },
        "rows": rows,
        "covered": sum(1 for row in rows if row["surface_z"] is not None),
        "total": len(rows),
    }


def placement_verdict(report, plausible=0.25):
    """Decide whether a room is placed, from where its surface fell.

    A misplaced room strands *every* prop together, because they all move as
    one; a gappy scan (or a prop past the desk edge) strands one or two while
    the rest agree. The verdict is the majority, and the strays are named.

    Args:
        report (dict): From :func:`surface_report`.
        plausible (float): Metres of gap still consistent with resting on the
            surface. Generous, because the gap is measured from the prop's
            *origin* height.

    Returns:
        dict: ``placed`` (bool), ``resting`` and ``stray`` (lists of names), and
        ``summary`` — one sentence fit to print.
    """
    rows = report["rows"]
    resting = [r["name"] for r in rows if r["gap"] is not None and abs(r["gap"]) < plausible]
    stray = [r["name"] for r in rows if r["name"] not in resting]
    total = len(rows)

    if not total:
        return {"placed": True, "resting": [], "stray": [],
                "summary": "no props to check this room against."}
    if not stray:
        return {"placed": True, "resting": resting, "stray": [],
                "summary": "every prop has room geometry just beneath it."}
    if len(resting) > len(stray):
        listed = ", ".join(stray)
        return {
            "placed": True, "resting": resting, "stray": stray,
            "summary": (
                f"{len(resting)} of {total} props rest on this room; {listed} "
                f"do{'es' if len(stray) == 1 else ''} not. With the rest agreeing, "
                "that is a hole in the scan or a prop past the table edge rather "
                "than a misplaced room — worth a look, not a blocker."
            ),
        }
    return {
        "placed": False, "resting": resting, "stray": stray,
        "summary": (
            f"only {len(resting)} of {total} props rest on this room. They move "
            "together, so this is the room being in the wrong place rather than "
            "the scan being gappy."
        ),
    }


def _print_report(report, plausible=0.25):
    """Print a surface report, and return whether the room looks placed."""
    print(f"\nSurface under each prop ({report['covered']}/{report['total']} covered):")
    verdict = placement_verdict(report, plausible)
    for row in report["rows"]:
        flagged = "   <-- far" if row["name"] in verdict["stray"] else ""
        if row["surface_z"] is None:
            print(f"  {row['name']:12s} {row['category']:22s} prop z={row['prop_z']:+.3f}"
                  f"   nothing beneath it{flagged}")
            continue
        print(f"  {row['name']:12s} {row['category']:22s} prop z={row['prop_z']:+.3f}"
              f"   surface z={row['surface_z']:+.3f}   gap={row['gap']:+.3f} m{flagged}")

    low = report["bounds"]["min"]
    high = report["bounds"]["max"]
    print(f"  room bounds: [{low[0]:.2f} {low[1]:.2f} {low[2]:.2f}]"
          f" .. [{high[0]:.2f} {high[1]:.2f} {high[2]:.2f}]")
    print(f"\n  {verdict['summary']}")
    return verdict["placed"]


def main():
    parser = argparse.ArgumentParser(
        description="Attach a scanned room to a scene that has none, or check the one it has",
    )
    parser.add_argument("--scene", required=True, help="Path to a scene JSON")
    parser.add_argument(
        "--background",
        default=None,
        help="Room id (e.g. droid_desk_mesh) or a path to its USD. Omit with --list "
             "to see what is available, or with --check to test the scene's own.",
    )
    parser.add_argument("--list", action="store_true", help="List available rooms and exit")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report where the room sits relative to the props without writing anything",
    )
    parser.add_argument(
        "--background-root",
        action="append",
        default=None,
        help="Directory of scanned rooms. Repeatable. Defaults to the scene's sibling "
             f"mesh_backgrounds/ plus <repo>/{DEFAULT_BACKGROUND_SUBDIR}.",
    )
    parser.add_argument("--name", default=BACKGROUND_OBJECT_NAME, help="Object name to add")
    # Space-separated rather than "x,y,z": argparse reads a lone
    # "-0.14,-0.69,-0.955" as an unknown option, but separate values parse.
    parser.add_argument(
        "--position", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
        help="Override the registered position, e.g. --position -0.14 -0.69 -0.955",
    )
    parser.add_argument(
        "--orientation", nargs=4, type=float, default=None, metavar=("X", "Y", "Z", "W"),
        help="Override the registered orientation, XYZW",
    )
    parser.add_argument(
        "--no-check", action="store_true",
        help="Skip the surface check after attaching (it reads the whole room mesh)",
    )
    args = parser.parse_args()

    scene_json = Path(args.scene).expanduser().resolve()
    if not scene_json.is_file():
        sys.exit(f"ERROR: scene JSON not found: {scene_json}")

    repo_root = HERE.parents[2]
    roots = background_roots(scene_json, repo_root, extra=args.background_root or [])
    if not roots:
        sys.exit("ERROR: no mesh_backgrounds directory found; pass --background-root")

    if args.list:
        print("Rooms found in " + ", ".join(str(r) for r in roots) + ":")
        for background in discover_backgrounds(roots):
            height = background["surface_height"]
            note = "" if height is None else f"  support surface z={height}"
            print(f"  {background['id']:22s} {background['label']:20s} {background['usd']}{note}")
        return 0

    scene = load_scene(scene_json)

    if args.check and not args.background:
        try:
            report = surface_report(scene, scene_json, args.name)
        except SceneEditError as e:
            sys.exit(f"ERROR: {e}")
        return 0 if _print_report(report) else 1

    if not args.background:
        sys.exit("ERROR: --background is required (or pass --list)")

    try:
        background = resolve_background(args.background, roots)
        added = attach_background(
            scene, scene_json, background,
            name=args.name,
            position=args.position,
            orientation=args.orientation,
        )
    except SceneEditError as e:
        sys.exit(f"ERROR: {e}")

    print(f"Attaching {background['label']} ({background['id']}) as {added['name']}")
    print(f"  usd_path  {added['usd_path']}")
    print(f"  position  {[round(v, 4) for v in added['position']]}")
    print(f"  orient    {[round(v, 4) for v in added['orientation']]}")
    if background["surface_height"] is not None:
        print(f"  registered with its support surface at z={background['surface_height']}")

    ok = True
    if not args.no_check:
        try:
            ok = _print_report(surface_report(scene, scene_json, added["name"]))
        except SceneEditError as e:
            print(f"  (surface check skipped: {e})")

    out = save_scene(scene, scene_json, suffix="background")
    print(f"\nWrote {out}")
    print("Open it with:\n"
          f"  mamba run -n simfoundry-editor python server.py --scene {out}")
    if not ok:
        print("\nFix the pose before using this scene — re-run with --position/--orientation.")
    else:
        print("\nTo nudge the room, re-run with --position/--orientation — the browser "
              "shows a background but will not move it (scene_io.iter_objects marks it "
              "non-editable).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
