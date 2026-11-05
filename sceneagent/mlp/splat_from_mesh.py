"""Camera views and a Gaussian ply from one textured mesh.

Labeling views are textured renders: the publisher texture is bound, a light
is on, and the background is plain. ``choose_front_view`` picks the front as
the direction where the largest textured face fills the frame. Side is a
quarter turn from that front. Top is the remaining axis. An edge-on, gray, or
faceted front is not returned; the view is rendered again with the texture bound.


Each kept triangle becomes one Gaussian. Position is the triangle centroid.
Scales and rotation are the covariance of the uniform distribution on that
triangle, plus a thickness along the normal so the third scale is positive.
``f_dc`` is the spherical-harmonic DC term of the texture color at the
triangle. Opacity is the logit of an opaque surface. A triangle with a
degenerate covariance or no texture color is dropped. Nothing is filled in.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from sceneagent.mlp.contract import PriorContractError, geometry_from_ply, write_gaussian_ply


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

SH_C0 = 0.28209479177387814
OPAQUE_OPACITY = 0.99
# Std along the normal, as a fraction of the smaller in-plane std.
NORMAL_THICKNESS = 0.2
# DINOv2 ViT-S/14 is published at 518. OpenCLIP ViT-B-16 is published at 224.
VIEW_SIZE = 518
CLIP_SIZE = 224
VIEW_NAMES = ("front", "side", "top")
# A 4096 float64 texture is 384 MB. Two of those, plus the mesh, exhausted this PC.
TEXTURE_MAX_SIDE = 1024
_BACKGROUND = np.array([232.0, 236.0, 240.0], dtype=np.float64)
# Ambient keeps the bound texture readable. The rest is one light.
_LIGHT_AMBIENT = 0.78
_LIGHT_DIFFUSE = 0.22


def _triangle_mesh_path(mesh_path):
    """Return the OBJ or glTF a file points at. An SDF is not a triangle mesh."""

    path = Path(mesh_path)
    if path.suffix.lower() != ".sdf":
        return path
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise PriorContractError(f"SDF is not readable XML: {exc}") from exc
    uris = []
    for mesh in root.iter("mesh"):
        for uri in mesh.iter("uri"):
            text = uri.text.strip() if uri.text else ""
            if text and text not in uris:
                uris.append(text)
    if not uris:
        raise PriorContractError("SDF does not reference a mesh")
    base = path.parent.resolve()
    for uri in uris:
        lowered = uri.lower()
        if lowered.startswith(("http://", "https://")):
            raise PriorContractError("mesh uri is remote")
        text = uri.strip().replace("\\", "/")
        if lowered.startswith("model://"):
            rest = text.split("://", 1)[1]
            parts = rest.split("/", 1)
            text = parts[1] if len(parts) == 2 else ""
        if not text or ".." in Path(text).parts:
            raise PriorContractError("mesh uri escapes the model directory")
        resolved = (base / text).resolve()
        if not resolved.is_relative_to(base) or not resolved.is_file():
            raise PriorContractError(f"referenced mesh is missing: {text}")
        if resolved.suffix.lower() in {".obj", ".glb", ".gltf"}:
            return resolved
    raise PriorContractError("SDF does not reference an OBJ or glTF mesh")


def load_textured_mesh(mesh_path, texture_path=None):
    """Load one mesh and the publisher texture that colors it."""

    mesh_path = _triangle_mesh_path(mesh_path)
    loaded = trimesh.load(mesh_path, process=False, force="scene")
    meshes = []
    if isinstance(loaded, trimesh.Trimesh):
        meshes = [loaded]
    else:
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
    if len(meshes) != 1:
        raise PriorContractError(f"expected one mesh in {mesh_path}, found {len(meshes)}")
    mesh = meshes[0]
    if len(mesh.faces) < 1 or len(mesh.vertices) < 3:
        raise PriorContractError("mesh has no triangles")
    image = _texture_image(mesh, texture_path)
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) != len(mesh.vertices):
        raise PriorContractError("mesh has no per-vertex UV; texture color is not filled")
    return mesh, np.asarray(image, dtype=np.uint8), np.asarray(uv, dtype=np.float64)


def gaussians_from_mesh(mesh, image, uv):
    """Build Gaussian parameters for every triangle that can fill its columns."""

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    uv_triangles = uv[faces]
    corners = triangles - triangles[:, 0:1, :]
    edge_1 = corners[:, 1, :]
    edge_2 = corners[:, 2, :]
    covariance = (
        (1.0 / 18.0) * _outer(edge_1, edge_1)
        + (1.0 / 18.0) * _outer(edge_2, edge_2)
        - (1.0 / 36.0) * (_outer(edge_1, edge_2) + _outer(edge_2, edge_1))
    )
    covariance = 0.5 * (covariance + np.swapaxes(covariance, 1, 2))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    in_plane_ok = (eigenvalues[:, 1] > 1e-12) & (eigenvalues[:, 2] > 1e-12)
    thickness = (NORMAL_THICKNESS * np.sqrt(np.clip(eigenvalues[:, 1], 0.0, None))) ** 2
    variances = np.stack([thickness, eigenvalues[:, 1], eigenvalues[:, 2]], axis=1)
    finite_variance = np.isfinite(variances).all(axis=1) & (variances > 0.0).all(axis=1)
    rgb = _sample_texture(image, uv_triangles.mean(axis=1))
    colored = np.isfinite(rgb).all(axis=1)
    keep = in_plane_ok & finite_variance & colored
    dropped = int((~keep).sum())
    if int(keep.sum()) < 1:
        raise PriorContractError(
            f"every triangle was dropped ({dropped}); missing values are not filled",
            kept=0,
            dropped=dropped,
        )
    rotation = eigenvectors[keep].copy()
    flip = np.linalg.det(rotation) < 0.0
    rotation[flip, :, 0] *= -1.0
    log_scales = 0.5 * np.log(variances[keep])
    means = triangles[keep].mean(axis=1)
    quaternions = _quaternion_from_matrices(rotation)
    f_dc = (rgb[keep] - 0.5) / SH_C0
    opacity = np.full(int(keep.sum()), math.log(OPAQUE_OPACITY / (1.0 - OPAQUE_OPACITY)), dtype=np.float64)
    if not (
        np.isfinite(means).all()
        and np.isfinite(log_scales).all()
        and np.isfinite(quaternions).all()
        and np.isfinite(f_dc).all()
        and np.isfinite(opacity).all()
    ):
        raise PriorContractError("Gaussian columns are not finite; missing values are not filled")
    return {
        "xyz": means,
        "log_scales": log_scales,
        "opacity": opacity,
        "rotation": quaternions,
        "f_dc": f_dc,
        "triangles_in": int(len(faces)),
        "dropped_geometry": dropped,
    }


def choose_front_view(vertices, faces, uv=None, image=None, exclude=()):
    """Pick front, side, and top camera directions.

    Front is the candidate where the largest textured face fills the frame.
    Side is a quarter turn from that front. Top is the remaining axis.
    ``exclude`` drops directions along an axis that was already rejected.
    Each value is ``(center_to_camera, image_up)``, both unit vectors.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) < 1 or len(vertices) < 3:
        raise PriorContractError("mesh has no triangles")
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(cross, axis=1)
    valid = length > 1e-12
    normals = np.zeros_like(cross)
    normals[valid] = cross[valid] / length[valid, None]
    areas = 0.5 * length
    weights = np.where(valid, areas, 0.0)
    if uv is not None and image is not None:
        centroids = np.asarray(uv, dtype=np.float64)[faces].mean(axis=1)
        sampled = _sample_texture(image, centroids)
        missing = ~np.isfinite(sampled).all(axis=1)
        weights = weights.copy()
        weights[missing] = 0.0
        if float(weights.sum()) <= 0.0:
            weights = np.where(valid, areas, 0.0)
    ranked = []
    for direction in _candidate_directions(vertices):
        if _direction_excluded(direction, exclude):
            continue
        facing = np.clip(normals @ direction, 0.0, None)
        projected = float(np.sum(weights * facing))
        aspect = _planar_aspect(vertices, direction)
        # A thin edge has little projected area and a skinny frame. The broad
        # face is large in both directions, so it wins.
        score = projected * (0.2 + 0.8 * aspect)
        ranked.append((score, aspect, projected, direction))
    if not ranked:
        raise PriorContractError("no candidate view for the largest textured face")
    ranked.sort(key=lambda item: _direction_rank(item[0], item[1], item[3]))
    best_score, best_aspect, best_projected, front_dir = ranked[0]
    if best_score <= 0.0 or not np.isfinite(best_score):
        raise PriorContractError("no textured face can fill a view")
    alternative = 0.0
    alternative_projected = 0.0
    for _score, aspect, projected, direction in ranked[1:]:
        if abs(float(np.dot(direction, front_dir))) > 0.85:
            continue
        if aspect > alternative:
            alternative = aspect
        if projected > alternative_projected:
            alternative_projected = projected
    axes = _axes_from_front(front_dir)
    axes["front_aspect"] = float(best_aspect)
    axes["alternative_aspect"] = float(alternative)
    axes["front_score"] = float(best_score)
    axes["front_projected"] = float(best_projected)
    axes["alternative_projected"] = float(alternative_projected)
    return axes


def render_textured_view(vertices, faces, image, uv, forward, up, vertex_normals, center, radius):
    """Rasterize one camera with the publisher texture and a light.

    The UV atlas is sampled onto the mesh. It is not the returned image.
    """

    if image is None or uv is None:
        raise PriorContractError("texture is not bound; refusing an untextured view")
    camera = _fit_camera(vertices, center, forward, up, radius)
    rgb, depth = _rasterize(
        vertices,
        faces,
        None,
        camera,
        uv=uv,
        image=image,
        vertex_normals=vertex_normals,
    )
    camera["rgb"] = rgb
    camera["depth"] = depth
    camera["textured"] = True
    return camera


def render_views(mesh, image, uv):
    """Render front, side, and top at ``VIEW_SIZE`` square.

    The front view is chosen by ``choose_front_view``. If that render is
    edge-on, gray, or so faceted that the surface label is gone, it is not
    kept. The three views are rendered again with the texture bound.
    """

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if image is None or uv is None:
        raise PriorContractError("texture is not bound; refusing an untextured view")
    vertex_normals = _vertex_normals(vertices, faces)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    radius = max(float(np.linalg.norm(extent)) * 0.5, 1e-4)
    choice = choose_front_view(vertices, faces, uv, image)
    views = _render_choice(
        vertices, faces, image, uv, vertex_normals, center, radius, choice
    )
    reason = front_view_unusable(
        views["front"]["rgb"],
        views["front"]["depth"],
        image=image,
        choice=choice,
        vertices=vertices,
        faces=faces,
        uv=uv,
        camera=views["front"],
    )
    if reason is not None:
        print(f"front rejected ({reason}); rendering again with the texture bound", flush=True)
        exclude = (choice["front"][0],) if reason == "edge-on" else ()
        choice = choose_front_view(vertices, faces, uv, image, exclude=exclude)
        views = _render_choice(
            vertices, faces, image, uv, vertex_normals, center, radius, choice
        )
        reason = front_view_unusable(
            views["front"]["rgb"],
            views["front"]["depth"],
            image=image,
            choice=choice,
            vertices=vertices,
            faces=faces,
            uv=uv,
            camera=views["front"],
        )
        if reason is not None:
            raise PriorContractError(
                f"front view is {reason} after rendering again with the texture bound"
            )
    return views


def _render_choice(vertices, faces, image, uv, vertex_normals, center, radius, choice):
    views = {}
    for name in VIEW_NAMES:
        forward, up = choice[name]
        print(
            f"render {name} axis={np.round(np.asarray(forward), 3).tolist()}",
            flush=True,
        )
        views[name] = render_textured_view(
            vertices, faces, image, uv, forward, up, vertex_normals, center, radius
        )
    return views


def front_view_unusable(
    rgb,
    depth,
    image=None,
    choice=None,
    vertices=None,
    faces=None,
    uv=None,
    camera=None,
):
    """Return ``edge-on``, ``gray``, or ``faceted`` when the front must not be sent."""

    mask = np.isfinite(depth) & (depth < 1e8)
    if int(mask.sum()) < 32:
        return "edge-on"
    ys, xs = np.nonzero(mask)
    height = int(ys.max() - ys.min() + 1)
    width = int(xs.max() - xs.min() + 1)
    aspect = min(height, width) / float(max(height, width, 1))
    short_fill = min(height, width) / float(rgb.shape[0])
    alternative = 0.0 if not choice else float(choice.get("alternative_aspect") or 0.0)
    # A pillow edge is a sliver while the broad face is another candidate.
    # The largest visible face of a thin object is kept even when that face is narrow.
    skinny = aspect < 0.30 or short_fill < 0.22
    old_edge = alternative >= 0.55 and skinny
    largest_face = False
    if choice and "front_projected" in choice and "alternative_projected" in choice:
        largest_face = float(choice["front_projected"]) + 1e-9 >= float(choice["alternative_projected"])
    if old_edge and largest_face:
        # region agent log
        _agent_log(
            "H3",
            "splat_from_mesh.py:front_view_unusable",
            "kept narrow largest face",
            {
                "aspect": aspect,
                "short_fill": short_fill,
                "alternative_aspect": alternative,
                "front_projected": float(choice["front_projected"]),
                "alternative_projected": float(choice["alternative_projected"]),
            },
        )
        # endregion
    elif old_edge:
        # region agent log
        _agent_log(
            "H3",
            "splat_from_mesh.py:front_view_unusable",
            "rejected edge-on sliver",
            {
                "aspect": aspect,
                "short_fill": short_fill,
                "alternative_aspect": alternative,
                "largest_face": largest_face,
            },
        )
        # endregion
        return "edge-on"
    foreground = rgb[mask].astype(np.float32)
    if _foreground_is_gray(foreground, image):
        return "gray"
    if (
        vertices is not None
        and faces is not None
        and uv is not None
        and camera is not None
        and _surface_label_gone(vertices, faces, image, uv, camera, rgb)
    ):
        return "faceted"
    return None


def _candidate_directions(vertices):
    directions = []
    for axis in range(3):
        base = np.zeros(3, dtype=np.float64)
        base[axis] = 1.0
        directions.append(base)
        directions.append(-base)
    centered = vertices - vertices.mean(axis=0)
    _values, vectors = np.linalg.eigh(centered.T @ centered)
    for axis in range(3):
        directions.append(vectors[:, axis])
        directions.append(-vectors[:, axis])
    return _unique_directions(directions)


def _unique_directions(directions):
    kept = []
    for direction in directions:
        direction = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            continue
        direction = direction / norm
        if any(abs(float(np.dot(direction, other))) > 0.999 for other in kept):
            continue
        kept.append(direction)
    return kept


def _direction_excluded(direction, exclude):
    for item in exclude:
        other = np.asarray(item, dtype=np.float64)
        norm = float(np.linalg.norm(other))
        if norm < 1e-8:
            continue
        if abs(float(np.dot(direction, other / norm))) > 0.85:
            return True
    return False


def _direction_rank(score, aspect, direction):
    dominant = int(np.argmax(np.abs(direction)))
    positive = 0 if float(direction[dominant]) > 0.0 else 1
    return (-float(score), -float(aspect), positive, dominant)


def _planar_aspect(vertices, direction):
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    centered = vertices - vertices.mean(axis=0)
    planar = centered - np.outer(centered @ direction, direction)
    evals = np.sort(np.linalg.eigvalsh(planar.T @ planar))
    major = math.sqrt(max(float(evals[-1]), 0.0))
    minor = math.sqrt(max(float(evals[-2]), 0.0)) if len(evals) > 1 else 0.0
    if major < 1e-8:
        return 0.0
    return float(np.clip(minor / major, 0.0, 1.0))


def _axes_from_front(front_dir):
    """Quarter turn around the front image's up, then the remaining axis."""

    front_dir = np.asarray(front_dir, dtype=np.float64)
    front_dir = front_dir / np.linalg.norm(front_dir)
    up = _camera_up(front_dir)
    side_dir = np.cross(front_dir, up)
    side_dir = side_dir / np.linalg.norm(side_dir)
    top_dir = up
    return {
        "front": (front_dir, up),
        "side": (side_dir, _image_up(side_dir, up, front_dir)),
        "top": (top_dir, _image_up(top_dir, front_dir, side_dir)),
    }


def _camera_up(view_dir):
    world = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(view_dir, world))) > 0.85:
        world = np.array([0.0, 1.0, 0.0])
    up = world - view_dir * float(np.dot(world, view_dir))
    return up / np.linalg.norm(up)


def _image_up(view_dir, preferred, fallback):
    view_dir = np.asarray(view_dir, dtype=np.float64)
    view_dir = view_dir / np.linalg.norm(view_dir)
    candidates = (
        preferred,
        fallback,
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    for candidate in candidates:
        candidate = np.asarray(candidate, dtype=np.float64)
        up = candidate - view_dir * float(np.dot(view_dir, candidate))
        norm = float(np.linalg.norm(up))
        if norm > 1e-6:
            return up / norm
    raise PriorContractError("camera up is parallel to the view direction")


def _vertex_normals(vertices, faces):
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], cross)
    lengths = np.linalg.norm(normals, axis=1)
    ok = lengths > 1e-12
    normals[ok] /= lengths[ok, None]
    return normals


def _foreground_is_gray(foreground, image):
    spread = float(foreground.std())
    peak = foreground.max(axis=1)
    floor = foreground.min(axis=1)
    saturation = float(np.mean((peak - floor) / np.maximum(peak, 1.0)))
    flat = (spread < 8.0 and saturation < 0.08) or (saturation < 0.045 and spread < 16.0)
    if image is None:
        return flat
    texture = np.asarray(image, dtype=np.float32)
    if texture.ndim != 3 or texture.shape[2] != 3:
        return flat
    if float(np.nanmax(texture)) <= 1.5:
        texture = texture * 255.0
    luma = texture.mean(axis=2)
    active = luma > 10.0
    if int(active.sum()) < 32:
        return flat
    pixels = texture[active]
    tex_peak = pixels.max(axis=1)
    tex_floor = pixels.min(axis=1)
    tex_sat = float(np.mean((tex_peak - tex_floor) / np.maximum(tex_peak, 1.0)))
    # A white plate is low saturation in both the atlas and the render.
    # A colorful atlas that renders flat gray is still unbound.
    if flat and tex_sat < 0.12:
        # region agent log
        _agent_log(
            "H2",
            "splat_from_mesh.py:_foreground_is_gray",
            "kept textured white",
            {"spread": spread, "saturation": saturation, "tex_sat": tex_sat},
        )
        # endregion
        return False
    if flat:
        return True
    return tex_sat > 0.22 and saturation < 0.08


def _surface_label_gone(vertices, faces, image, uv, camera, rgb):
    """True when large triangles flatten a detailed texture into facets."""

    if image is None or uv is None or len(faces) < 1:
        return False
    image = _texture_unit(image)
    count = len(faces)
    sample_n = min(count, 240)
    if count > sample_n:
        pick = np.random.default_rng(0).choice(count, size=sample_n, replace=False)
    else:
        pick = np.arange(count)
    px, py, depth = _project(vertices, camera)
    zbuf = camera.get("depth")
    uv = np.asarray(uv, dtype=np.float64)
    lost = 0
    compared = 0
    grid = _barycentric_grid()
    for index in pick:
        face = faces[int(index)]
        z = depth[face]
        if np.any(z <= 1e-6) or not np.isfinite(z).all():
            continue
        xs = px[face]
        ys = py[face]
        area = abs(float(_edge(xs[0], ys[0], xs[1], ys[1], xs[2], ys[2]))) * 0.5
        if area < 24.0:
            continue
        uvs = uv[face]
        samples_uv = grid @ uvs
        texture_rgb = _sample_texture(image, samples_uv) * 255.0
        render_rgb = []
        screen = grid @ np.stack([xs, ys], axis=1)
        for sx, sy in screen:
            xi = int(round(float(sx)))
            yi = int(round(float(sy)))
            if xi < 0 or yi < 0 or xi >= rgb.shape[1] or yi >= rgb.shape[0]:
                continue
            if zbuf is not None and not np.isfinite(zbuf[yi, xi]):
                continue
            render_rgb.append(rgb[yi, xi].astype(np.float32))
        if len(render_rgb) < 8 or not np.isfinite(texture_rgb).all():
            continue
        render_rgb = np.asarray(render_rgb, dtype=np.float32)
        tex_std = float(texture_rgb.std())
        render_std = float(render_rgb.std())
        compared += 1
        if tex_std > 22.0 and render_std < 0.40 * tex_std:
            lost += 1
    if compared < 12:
        return False
    return (lost / compared) >= 0.55


def _barycentric_grid():
    samples = []
    for i in range(4):
        for j in range(4):
            alpha = 0.25 + 0.5 * (i / 3.0)
            beta = (1.0 - alpha) * (0.25 + 0.5 * (j / 3.0))
            gamma = 1.0 - alpha - beta
            if gamma < 0.02:
                continue
            samples.append((alpha, beta, gamma))
    return np.asarray(samples, dtype=np.float64)


def save_views(views, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in VIEW_NAMES:
        path = directory / f"{name}.png"
        Image.fromarray(views[name]["rgb"], mode="RGB").save(path)
        paths[name] = path
    return paths


def filter_visible(gaussians, views, features_by_view, clip_size=CLIP_SIZE):
    """Keep Gaussians that land on a rendered surface in at least one view.

    ``features_by_view[name]`` is ``(dino_grid, clip_grid)``. The DINOv2 grid
    is in the rendered view. The OpenCLIP grid is in a ``clip_size`` resize of
    that same view. A Gaussian with no finite feature is dropped. Rows are
    not zero-filled.
    """

    means = gaussians["xyz"]
    count = len(means)
    dino_sum = np.zeros((count, 384), dtype=np.float64)
    clip_sum = np.zeros((count, 512), dtype=np.float64)
    hits = np.zeros(count, dtype=np.int32)
    radius = max(float(np.linalg.norm(means.max(axis=0) - means.min(axis=0))) * 0.5, 1e-4)
    tolerance = max(1e-4, 0.03 * radius)
    for name, camera in views.items():
        dino_grid, clip_grid = features_by_view[name]
        px, py, depth = _project(means, camera)
        xi = np.rint(px).astype(np.int64)
        yi = np.rint(py).astype(np.int64)
        height, width = camera["depth"].shape
        inside = (depth > 1e-6) & (xi >= 1) & (yi >= 1) & (xi < width - 1) & (yi < height - 1)
        visible = np.zeros(count, dtype=bool)
        surface = np.full(count, np.nan)
        idx = np.flatnonzero(inside)
        for shift_y in (-1, 0, 1):
            for shift_x in (-1, 0, 1):
                ys = yi[idx] + shift_y
                xs = xi[idx] + shift_x
                sampled = camera["depth"][ys, xs]
                close = np.isfinite(sampled) & (np.abs(sampled - depth[idx]) <= tolerance)
                better = close & (np.isnan(surface[idx]) | (np.abs(sampled - depth[idx]) < np.abs(surface[idx] - depth[idx])))
                surface[idx[better]] = sampled[better]
                visible[idx[better]] = True
        if not visible.any():
            continue
        dino_patch = _patch_index(py[visible], px[visible], dino_grid.shape[0], dino_grid.shape[1], height, width)
        clip_py = py[visible] * (clip_size / height)
        clip_px = px[visible] * (clip_size / width)
        clip_patch = _patch_index(clip_py, clip_px, clip_grid.shape[0], clip_grid.shape[1], clip_size, clip_size)
        dino_rows = dino_grid[dino_patch[:, 0], dino_patch[:, 1]]
        clip_rows = clip_grid[clip_patch[:, 0], clip_patch[:, 1]]
        usable = np.isfinite(dino_rows).all(axis=1) & np.isfinite(clip_rows).all(axis=1)
        usable &= np.linalg.norm(dino_rows, axis=1) > 0.0
        usable &= np.linalg.norm(clip_rows, axis=1) > 0.0
        chosen = np.flatnonzero(visible)[usable]
        dino_sum[chosen] += dino_rows[usable]
        clip_sum[chosen] += clip_rows[usable]
        hits[chosen] += 1
    keep = hits > 0
    if not keep.any():
        raise PriorContractError(
            "no Gaussian is visible in a view with a finite feature; missing values are not filled",
            kept=0,
            dropped=count,
        )
    return {
        "xyz": means[keep],
        "log_scales": gaussians["log_scales"][keep],
        "opacity": gaussians["opacity"][keep],
        "rotation": gaussians["rotation"][keep],
        "f_dc": gaussians["f_dc"][keep],
        "dino": dino_sum[keep] / hits[keep, None],
        "clip": clip_sum[keep] / hits[keep, None],
        "dropped_unseen": int((~keep).sum()),
        "kept": int(keep.sum()),
    }


def write_splat(path, gaussians):
    """Write the ply and read it back through the frozen geometry contract."""

    write_gaussian_ply(
        path,
        xyz=gaussians["xyz"],
        log_scales=gaussians["log_scales"],
        opacity=gaussians["opacity"],
        rotation=gaussians["rotation"],
        f_dc=gaussians["f_dc"],
    )
    geometry = geometry_from_ply(path)
    if geometry.dropped != 0 or geometry.kept != len(gaussians["xyz"]):
        raise PriorContractError("written ply did not round-trip through the geometry contract")
    return geometry


def _texture_image(mesh, texture_path):
    material = getattr(mesh.visual, "material", None)
    image = None
    if material is not None:
        image = getattr(material, "image", None)
        if image is None:
            image = getattr(material, "baseColorTexture", None)
    if image is None and texture_path:
        image = Image.open(texture_path)
    if image is None:
        raise PriorContractError("mesh has no texture image; color is not filled")
    rgb = image.convert("RGB")
    orig_w, orig_h = rgb.size
    # A 4096 texture is 384 MiB as float64. This machine has about 12 GiB,
    # and that allocation failed while four objects rendered together.
    # 1024 is still finer than the 518 labeling view.
    if max(orig_w, orig_h) > TEXTURE_MAX_SIDE:
        rgb.thumbnail((TEXTURE_MAX_SIDE, TEXTURE_MAX_SIDE), Image.Resampling.BOX)
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or array.shape[0] < 1 or array.shape[1] < 1:
        raise PriorContractError("texture image is not RGB")
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    # region agent log
    _agent_log(
        "H1",
        "splat_from_mesh.py:_texture_image",
        "texture loaded",
        {
            "orig_w": int(orig_w),
            "orig_h": int(orig_h),
            "stored_shape": [int(array.shape[0]), int(array.shape[1]), int(array.shape[2])],
            "dtype": str(array.dtype),
            "nbytes": int(array.nbytes),
        },
    )
    # endregion
    return array


def _texture_unit(image):
    """RGB texture in 0–1 float32. A floating image is already in that range."""

    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating):
        if array.dtype == np.float32:
            return array
        return array.astype(np.float32)
    return array.astype(np.float32) * np.float32(1.0 / 255.0)


def _sample_texture(image, uv):
    uv = np.asarray(uv, dtype=np.float64)
    if not np.isfinite(uv).all():
        bad = ~np.isfinite(uv).all(axis=-1)
        colors = np.full(uv.shape[:-1] + (3,), np.nan, dtype=np.float64)
        if bad.shape != colors.shape[:-1]:
            return colors
        good = ~bad
        if not np.any(good):
            return colors
        sampled = _sample_texture(image, uv[good])
        colors[good] = sampled
        return colors
    if not isinstance(image, np.ndarray) or image.dtype != np.float32:
        image = _texture_unit(image)
    height, width = image.shape[:2]
    x = np.mod(uv[..., 0], 1.0) * (width - 1)
    y = np.mod(1.0 - uv[..., 1], 1.0) * (height - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x0 = np.clip(x0, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]
    c00 = image[y0, x0]
    c10 = image[y0, x1]
    c01 = image[y1, x0]
    c11 = image[y1, x1]
    return c00 * (1.0 - wx) * (1.0 - wy) + c10 * wx * (1.0 - wy) + c01 * (1.0 - wx) * wy + c11 * wx * wy


def _outer(left, right):
    return left[:, :, None] * right[:, None, :]


def _quaternion_from_matrices(rotation):
    matrices = np.asarray(rotation, dtype=np.float64)
    trace = matrices[:, 0, 0] + matrices[:, 1, 1] + matrices[:, 2, 2]
    quaternions = np.zeros((len(matrices), 4), dtype=np.float64)
    primary = trace > 0.0
    if np.any(primary):
        scale = np.sqrt(trace[primary] + 1.0) * 2.0
        block = matrices[primary]
        quaternions[primary, 0] = 0.25 * scale
        quaternions[primary, 1] = (block[:, 2, 1] - block[:, 1, 2]) / scale
        quaternions[primary, 2] = (block[:, 0, 2] - block[:, 2, 0]) / scale
        quaternions[primary, 3] = (block[:, 1, 0] - block[:, 0, 1]) / scale
    rest = ~primary
    if np.any(rest):
        block = matrices[rest]
        use_x = (block[:, 0, 0] >= block[:, 1, 1]) & (block[:, 0, 0] >= block[:, 2, 2])
        use_y = (~use_x) & (block[:, 1, 1] >= block[:, 2, 2])
        use_z = (~use_x) & (~use_y)
        _fill_quaternion(quaternions, rest, block, use_x, 0)
        _fill_quaternion(quaternions, rest, block, use_y, 1)
        _fill_quaternion(quaternions, rest, block, use_z, 2)
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms <= 0.0) or not np.isfinite(quaternions).all():
        raise PriorContractError("rotation quaternion is not finite")
    return quaternions / norms[:, None]


def _fill_quaternion(quaternions, rest_mask, block, which, axis):
    if not np.any(which):
        return
    chosen = block[which]
    parent = np.flatnonzero(rest_mask)[which]
    if axis == 0:
        scale = np.sqrt(1.0 + chosen[:, 0, 0] - chosen[:, 1, 1] - chosen[:, 2, 2]) * 2.0
        quaternions[parent, 0] = (chosen[:, 2, 1] - chosen[:, 1, 2]) / scale
        quaternions[parent, 1] = 0.25 * scale
        quaternions[parent, 2] = (chosen[:, 0, 1] + chosen[:, 1, 0]) / scale
        quaternions[parent, 3] = (chosen[:, 0, 2] + chosen[:, 2, 0]) / scale
    elif axis == 1:
        scale = np.sqrt(1.0 + chosen[:, 1, 1] - chosen[:, 0, 0] - chosen[:, 2, 2]) * 2.0
        quaternions[parent, 0] = (chosen[:, 0, 2] - chosen[:, 2, 0]) / scale
        quaternions[parent, 1] = (chosen[:, 0, 1] + chosen[:, 1, 0]) / scale
        quaternions[parent, 2] = 0.25 * scale
        quaternions[parent, 3] = (chosen[:, 1, 2] + chosen[:, 2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + chosen[:, 2, 2] - chosen[:, 0, 0] - chosen[:, 1, 1]) * 2.0
        quaternions[parent, 0] = (chosen[:, 1, 0] - chosen[:, 0, 1]) / scale
        quaternions[parent, 1] = (chosen[:, 0, 2] + chosen[:, 2, 0]) / scale
        quaternions[parent, 2] = (chosen[:, 1, 2] + chosen[:, 2, 1]) / scale
        quaternions[parent, 3] = 0.25 * scale


def _fit_camera(vertices, center, forward, up, radius):
    """Place the camera so the mesh fills the frame and stays inside the margin."""

    forward = np.asarray(forward, dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    fov = math.radians(40.0)
    focal = ((VIEW_SIZE - 1) * 0.5) / math.tan(fov * 0.5)
    margin = 8
    near = max(float(radius) * 0.02, 1e-4)
    low = max(float(radius) * 0.05, 1e-3)
    high = max(float(radius) * 40.0, low * 8.0)
    best = None
    best_span = -1.0
    for _ in range(28):
        distance = 0.5 * (low + high)
        eye = center + forward * distance
        camera = _camera(eye, center, up, focal)
        px, py, depth = _project(vertices, camera)
        finite = np.isfinite(px) & np.isfinite(py) & np.isfinite(depth)
        if not finite.all() or np.any(depth[finite] <= near) or np.any(depth <= near):
            low = distance
            continue
        inside = (
            float(px.min()) >= margin
            and float(py.min()) >= margin
            and float(px.max()) <= VIEW_SIZE - 1 - margin
            and float(py.max()) <= VIEW_SIZE - 1 - margin
        )
        span = max(float(px.max() - px.min()), float(py.max() - py.min()))
        if not inside:
            low = distance
            continue
        if span > best_span:
            best = camera
            best_span = span
        high = distance
    if best is None or best_span < 8.0:
        raise PriorContractError("camera could not frame the mesh")
    best["forward"] = forward
    look = -forward
    right = np.cross(look, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-8:
        raise PriorContractError("camera up is parallel to the view direction")
    right = right / right_norm
    light = forward * 0.70 + up * 0.58 + right * 0.30
    best["light"] = light / np.linalg.norm(light)
    return best


def _camera(eye, target, up, focal):
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        raise PriorContractError("camera up is parallel to the view direction")
    right = right / right_norm
    cam_up = np.cross(right, forward)
    return {
        "eye": np.asarray(eye, dtype=np.float64),
        "rot": np.stack([right, cam_up, forward], axis=0),
        "focal": float(focal),
        "width": VIEW_SIZE,
        "height": VIEW_SIZE,
    }


def _project(points, camera):
    cam = (np.asarray(points, dtype=np.float64) - camera["eye"]) @ camera["rot"].T
    depth = cam[:, 2]
    safe = np.where(np.abs(depth) < 1e-8, 1e-8, depth)
    center = (camera["width"] - 1) * 0.5
    px = camera["focal"] * cam[:, 0] / safe + center
    py = center - camera["focal"] * cam[:, 1] / safe
    return px, py, depth


def _rasterize(vertices, faces, vertex_rgb, camera, uv=None, image=None, vertex_normals=None):
    if image is not None:
        image = _texture_unit(image)
    px, py, depth = _project(vertices, camera)
    height = camera["height"]
    width = camera["width"]
    color = np.empty((height, width, 3), dtype=np.float64)
    color[:] = _BACKGROUND / 255.0
    zbuf = np.full((height, width), np.inf, dtype=np.float64)
    for face in faces:
        z = depth[face]
        if np.any(z <= 1e-6) or not np.isfinite(z).all():
            continue
        xs = px[face]
        ys = py[face]
        minx = max(int(math.floor(xs.min())), 0)
        maxx = min(int(math.ceil(xs.max())), width - 1)
        miny = max(int(math.floor(ys.min())), 0)
        maxy = min(int(math.ceil(ys.max())), height - 1)
        if minx > maxx or miny > maxy:
            continue
        grid_y, grid_x = np.mgrid[miny : maxy + 1, minx : maxx + 1]
        area = _edge(xs[0], ys[0], xs[1], ys[1], xs[2], ys[2])
        if area == 0.0 or not math.isfinite(area) or abs(area) < 0.8:
            continue
        w0 = _edge(xs[1], ys[1], xs[2], ys[2], grid_x, grid_y)
        w1 = _edge(xs[2], ys[2], xs[0], ys[0], grid_x, grid_y)
        w2 = _edge(xs[0], ys[0], xs[1], ys[1], grid_x, grid_y)
        if area < 0.0:
            w0, w1, w2, area = -w0, -w1, -w2, -area
        mask = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
        if not np.any(mask):
            continue
        b0 = w0[mask] / area
        b1 = w1[mask] / area
        b2 = w2[mask] / area
        inv = b0 / z[0] + b1 / z[1] + b2 / z[2]
        if np.any(inv <= 0.0):
            positive = inv > 0.0
            if not np.any(positive):
                continue
            b0, b1, b2, inv = b0[positive], b1[positive], b2[positive], inv[positive]
            mask_idx = np.flatnonzero(mask)[positive]
        else:
            mask_idx = np.flatnonzero(mask)
        if image is not None:
            if uv is None:
                raise PriorContractError("texture is not bound; refusing an untextured view")
            uvs = uv[face]
            interp_uv = (
                (b0 / z[0])[:, None] * uvs[0]
                + (b1 / z[1])[:, None] * uvs[1]
                + (b2 / z[2])[:, None] * uvs[2]
            ) / inv[:, None]
            albedo = _sample_texture(image, interp_uv)
        else:
            albedo = (
                (b0 / z[0])[:, None] * vertex_rgb[face[0]]
                + (b1 / z[1])[:, None] * vertex_rgb[face[1]]
                + (b2 / z[2])[:, None] * vertex_rgb[face[2]]
            ) / inv[:, None]
        if vertex_normals is not None and camera.get("light") is not None:
            normals = vertex_normals[face]
            interp_n = (
                (b0 / z[0])[:, None] * normals[0]
                + (b1 / z[1])[:, None] * normals[1]
                + (b2 / z[2])[:, None] * normals[2]
            ) / inv[:, None]
            lengths = np.linalg.norm(interp_n, axis=1)
            lit = lengths > 1e-8
            ndot = np.full(len(interp_n), 0.65, dtype=np.float64)
            if np.any(lit):
                unit = interp_n[lit] / lengths[lit, None]
                ndot[lit] = np.abs(unit @ camera["light"])
            albedo = albedo * (_LIGHT_AMBIENT + _LIGHT_DIFFUSE * ndot)[:, None]
        shaded = albedo
        pixel_z = 1.0 / inv
        ys = grid_y.ravel()[mask_idx]
        xs = grid_x.ravel()[mask_idx]
        closer = pixel_z < zbuf[ys, xs]
        if not np.any(closer):
            continue
        zbuf[ys[closer], xs[closer]] = pixel_z[closer]
        color[ys[closer], xs[closer]] = np.clip(shaded[closer], 0.0, 1.0)
    rgb = np.clip(np.rint(color * 255.0), 0, 255).astype(np.uint8)
    zbuf[~np.isfinite(zbuf)] = np.inf
    return rgb, zbuf


def _edge(ax, ay, bx, by, px, py):
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax)


def _patch_index(py, px, grid_h, grid_w, height, width):
    row = np.clip(np.floor(py / height * grid_h).astype(np.int64), 0, grid_h - 1)
    col = np.clip(np.floor(px / width * grid_w).astype(np.int64), 0, grid_w - 1)
    return np.stack([row, col], axis=1)
