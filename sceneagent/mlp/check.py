"""Suitability check for one downloaded YCB, GSO, or ABO object.

A download is usable only when every condition holds:

- the mesh file is an OBJ, a glTF (``.gltf`` or ``.glb``), or an SDF that
  references one of those meshes, and it is not an HTML page, a zip, or empty
- at least one RGB image exists (texture photo or rendered view)
- the mesh bounding box has a non-zero finite extent, and the volume or the
  surface area is finite and positive, so later geometric features can be
  computed
- the source tag is ``ycb``, ``gso``, or ``abo``, and the id matches the
  manifest row that will keep the object

The extent is reported in the mesh file's units. YCB, GSO, and ABO publish
these meshes in meters. Callers that discover a different unit scale the
reported extent before writing the manifest.
"""

from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import trimesh
from PIL import Image, UnidentifiedImageError

VALID_SOURCES = ("ycb", "gso", "abo")
MODEL_SUFFIXES = (".obj", ".gltf", ".glb", ".sdf")
REFERENCED_MESH_SUFFIXES = (".obj", ".gltf", ".glb")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
# A bounding-box side shorter than this is treated as zero.
_EXTENT_EPSILON = 1e-8


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one suitability check.

    ``extent_m`` is the axis-aligned bounding-box size (x, y, z) when the
    mesh could be measured, in file units. It is ``None`` when measurement
    did not succeed. ``status`` is ``pass`` only when ``reasons`` is empty.
    """

    ok: bool
    status: str
    reasons: tuple[str, ...]
    extent_m: tuple[float, float, float] | None


def check_sample(sample: Mapping, manifest: Mapping) -> CheckResult:
    """Return whether ``sample`` may stay in the usable set.

    ``manifest`` is the row the caller intends to keep. The sample id and
    source must equal that row. ``sample`` supplies ``id``, ``source``,
    ``mesh_path``, and ``image_path``.
    """

    reasons: list[str] = []
    _check_identity(sample, manifest, reasons)

    mesh_path = _as_path(sample.get("mesh_path"))
    image_path = _as_path(sample.get("image_path"))
    mesh_targets: list[Path] = []

    if mesh_path is None:
        reasons.append("mesh path is missing")
    else:
        container = _container_problem(mesh_path, kind="mesh")
        if container:
            reasons.append(container)
        else:
            suffix = mesh_path.suffix.lower()
            if suffix not in MODEL_SUFFIXES:
                reasons.append(
                    f"mesh suffix {suffix or '(none)'} is not OBJ, glTF, or SDF"
                )
            elif suffix == ".sdf":
                mesh_targets, sdf_reasons = _sdf_meshes(mesh_path)
                reasons.extend(sdf_reasons)
            elif suffix == ".glb" and not _has_glb_magic(mesh_path):
                reasons.append("glTF binary header is missing")
            else:
                mesh_targets = [mesh_path]

    if image_path is None:
        reasons.append("image path is missing")
    else:
        image_reason = rgb_image_problem(image_path)
        if image_reason:
            reasons.append(image_reason)

    extent = None
    if mesh_targets and not any(r.startswith("SDF") or "mesh uri" in r or "referenced mesh" in r for r in reasons):
        try:
            extent, area, volume = _measure(mesh_targets)
        except Exception as exc:
            reasons.append(f"mesh could not be measured: {exc}")
        else:
            _check_geometry(extent, area, volume, reasons)

    ok = not reasons
    return CheckResult(
        ok=ok,
        status="pass" if ok else "fail",
        reasons=tuple(reasons),
        extent_m=extent,
    )


def rgb_image_problem(path: Path) -> str | None:
    """Return a reason the file is not a usable RGB image, or ``None``."""

    problem = _container_problem(path, kind="image")
    if problem:
        return problem
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return f"image suffix {path.suffix or '(none)'} is not a photo"
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
            if mode == "P":
                mode = "RGB" if _palette_has_color(image) else "L"
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return f"image is not a readable RGB file: {exc}"
    if width < 1 or height < 1:
        return "image has no pixels"
    if mode not in ("RGB", "RGBA"):
        return f"image mode {mode} is not RGB"
    return None


def _check_identity(sample: Mapping, manifest: Mapping, reasons: list[str]) -> None:
    sample_id = sample.get("id")
    manifest_id = manifest.get("id")
    if not isinstance(manifest_id, str) or not manifest_id.strip():
        reasons.append("manifest id is missing")
    elif sample_id != manifest_id:
        reasons.append("id does not match the manifest")

    sample_source = sample.get("source")
    manifest_source = manifest.get("source")
    if sample_source != manifest_source:
        reasons.append("source does not match the manifest")
    if sample_source not in VALID_SOURCES:
        reasons.append("source tag must be ycb, gso, or abo")


def _check_geometry(extent, area: float, volume: float, reasons: list[str]) -> None:
    if extent is None or len(extent) != 3 or not all(math.isfinite(v) for v in extent):
        reasons.append("extent is not finite")
        return
    if max(extent) <= _EXTENT_EPSILON:
        reasons.append("extent is zero")
    positive_area = math.isfinite(area) and area > _EXTENT_EPSILON
    positive_volume = math.isfinite(volume) and abs(volume) > _EXTENT_EPSILON
    if not positive_area and not positive_volume:
        if not math.isfinite(area) and not math.isfinite(volume):
            reasons.append("volume and surface area are not finite")
        else:
            reasons.append("volume and surface area are zero")


def _measure(paths: list[Path]):
    bounds = []
    area = 0.0
    volume = 0.0
    volume_finite = True
    triangles = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for path in paths:
            loaded = trimesh.load(path, process=False, allow_remote=False)
            meshes = _as_meshes(loaded)
            if not meshes:
                raise ValueError(f"no triangles in {path.name}")
            for mesh in meshes:
                if len(mesh.faces) == 0:
                    continue
                vertices = np.asarray(mesh.vertices, dtype=float)
                if vertices.size == 0 or not np.isfinite(vertices).all():
                    raise ValueError("mesh coordinates are not finite")
                triangles += len(mesh.faces)
                lo, hi = mesh.bounds
                bounds.append((tuple(float(v) for v in lo), tuple(float(v) for v in hi)))
                area += float(mesh.area)
                # Open scans often have no enclosed volume. Surface area still
                # counts. Asking trimesh for volume on those meshes divides by zero.
                if not bool(mesh.is_watertight):
                    continue
                try:
                    piece = float(mesh.volume)
                except Exception:
                    volume_finite = False
                    continue
                if math.isfinite(piece):
                    volume += abs(piece)
                else:
                    volume_finite = False
    if triangles == 0 or not bounds:
        raise ValueError("mesh has no triangles")
    extent = _union_extent(bounds)
    if not volume_finite:
        volume = float("nan")
    return extent, area, volume


def _as_meshes(loaded):
    if isinstance(loaded, trimesh.Trimesh):
        return [loaded]
    if isinstance(loaded, trimesh.Scene):
        return [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
    return []


def _union_extent(bounds):
    mins = [lo for lo, _hi in bounds]
    maxs = [hi for _lo, hi in bounds]
    low = [min(row[axis] for row in mins) for axis in range(3)]
    high = [max(row[axis] for row in maxs) for axis in range(3)]
    return tuple(high[axis] - low[axis] for axis in range(3))


def _sdf_meshes(sdf_path: Path) -> tuple[list[Path], list[str]]:
    try:
        tree = ET.parse(sdf_path)
    except ET.ParseError as exc:
        return [], [f"SDF is not readable XML: {exc}"]
    uris = []
    seen = set()
    for mesh in tree.getroot().iter("mesh"):
        for uri in mesh.iter("uri"):
            text = uri.text.strip() if uri.text else ""
            if text and text not in seen:
                seen.add(text)
                uris.append(text)
    if not uris:
        return [], ["SDF does not reference a mesh"]

    targets = []
    reasons = []
    for uri in uris:
        resolved, problem = _resolve_mesh_uri(sdf_path, uri)
        if problem:
            reasons.append(problem)
            continue
        container = _container_problem(resolved, kind="referenced mesh")
        if container:
            reasons.append(container)
            continue
        suffix = resolved.suffix.lower()
        if suffix not in REFERENCED_MESH_SUFFIXES:
            reasons.append(
                f"referenced mesh suffix {suffix or '(none)'} is not OBJ or glTF"
            )
            continue
        if suffix == ".glb" and not _has_glb_magic(resolved):
            reasons.append("referenced glTF binary header is missing")
            continue
        targets.append(resolved)
    if not targets and not reasons:
        reasons.append("SDF does not reference a mesh")
    return targets, reasons


def _resolve_mesh_uri(sdf_path: Path, uri: str) -> tuple[Path | None, str | None]:
    text = uri.strip().replace("\\", "/")
    lowered = text.lower()
    if lowered.startswith(("http://", "https://")):
        return None, "mesh uri is remote"
    if lowered.startswith("model://"):
        rest = text.split("://", 1)[1]
        parts = rest.split("/", 1)
        text = parts[1] if len(parts) == 2 else ""
        if not text:
            return None, "mesh uri has no path"
    elif lowered.startswith("file://"):
        text = text[7:]
        if text.startswith("/") or (len(text) > 2 and text[1] == ":"):
            return None, "mesh uri escapes the model directory"
        text = text.lstrip("/")
    if text.startswith(("/", "\\")) or _has_drive(text) or ".." in Path(text).parts:
        return None, "mesh uri escapes the model directory"
    root = sdf_path.parent.resolve()
    resolved = (root / text).resolve()
    if not resolved.is_relative_to(root):
        return None, "mesh uri escapes the model directory"
    if not resolved.is_file():
        return None, f"referenced mesh is missing: {text}"
    return resolved, None


def _container_problem(path: Path, *, kind: str) -> str | None:
    if not path.is_file():
        return f"{kind} file is missing"
    size = path.stat().st_size
    if size == 0:
        return f"{kind} file is empty"
    with path.open("rb") as handle:
        head = handle.read(512)
    if not head.strip():
        return f"{kind} file is empty"
    stripped = head.lstrip().lower()
    if stripped.startswith((b"<!doctype", b"<html", b"<head")):
        return f"{kind} file is HTML, not a mesh" if kind != "image" else "image file is HTML, not a photo"
    if head.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return f"{kind} file is a zip archive, not a mesh" if kind != "image" else "image file is a zip archive, not a photo"
    return None


def _has_glb_magic(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(4) == b"glTF"


def _palette_has_color(image: Image.Image) -> bool:
    rgb = image.convert("RGB")
    colors = rgb.getcolors(maxcolors=256 * 256)
    if not colors:
        return True
    return any(red != green or green != blue for _count, (red, green, blue) in colors)


def _as_path(value) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _has_drive(text: str) -> bool:
    return len(text) >= 2 and text[0].isalpha() and text[1] == ":"
