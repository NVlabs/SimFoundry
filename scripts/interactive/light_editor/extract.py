# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Extract a scene's visual geometry to glTF for the browser editor.

Runs on ``usd-core`` alone — no OmniGibson, no Isaac Sim.

Usage:
    python extract.py --scene <scene_state.json> [--out web/data]
"""

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

try:
    from pxr import Sdf, Usd, UsdGeom, UsdShade
except ImportError:  # pragma: no cover - environment guard
    sys.exit(
        "pxr not found. This tool needs standalone OpenUSD, not Isaac Sim's copy:\n"
        "    pip install -r requirements.txt"
    )

import trimesh

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robot_pose  # noqa: E402
import splat_io  # noqa: E402
import usd_cache  # noqa: E402
from scene_io import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    DEFAULT_ROBOT_ASSET_DIR,
    iter_objects,
    load_scene,
    node_name_for,
    scene_sha256,
)

# Purposes that exist for physics or debugging rather than display.
SKIP_PURPOSES = {"guide", "proxy"}

# Most vertices a face-varying mesh may grow to when split so glTF can carry
# its UVs; past this the object is emitted untextured. Counted against the
# distinct (point, uv) triples actually needed.
UNWELD_VERT_BUDGET = 600_000

# Textures are downscaled to this edge length; placement does not need more.
MAX_TEXTURE_PX = 2048

_texture_cache = {}


def _fan_slots(counts):
    """Fan-triangulate faces, returning indices into the flat face-vertex array.

    Slots index face corners rather than points, so the same triangulation
    drives both positions and face-varying UVs.

    Args:
        counts (list[int]): Vertex count per face.

    Returns:
        np.ndarray: (n, 3) indices into the flat face-vertex array.
    """
    slots = []
    offset = 0
    for count in counts:
        for i in range(1, count - 1):
            slots.append((offset, offset + i, offset + i + 1))
        offset += count
    return np.asarray(slots, dtype=np.int64).reshape(-1, 3)


def _mtime_ns(path):
    """Modification time of an asset, or None when it cannot be read."""
    try:
        return Path(path).stat().st_mtime_ns
    except (OSError, TypeError):
        return None


def _archive_member(reference, usd_dir):
    """Split ``archive.usdz[inner/path.jpeg]`` into an archive path and a member.

    Args:
        reference (str): An asset reference, resolved or as authored.
        usd_dir (Path): Directory to anchor a relative archive against.

    Returns:
        tuple[Path, str] or None: None when this is not an archive reference.
    """
    if not reference or "[" not in reference or not reference.endswith("]"):
        return None
    archive, member = reference[:-1].split("[", 1)
    archive_path = Path(archive)
    if not archive_path.is_absolute():
        archive_path = (usd_dir / archive_path).resolve()
    return archive_path, member


def _open_asset_image(asset, usd_dir):
    """Open a texture asset as a PIL image.

    Handles both loose files and textures packaged inside a USDZ, which USD
    spells ``archive.usdz[inner/path.jpeg]``. USDZ is an uncompressed zip, so
    the member can be read directly.

    Args:
        asset (Sdf.AssetPath): Asset reference from a shader input.
        usd_dir (Path): Directory of the USD, for relative paths.

    Returns:
        PIL.Image or None
    """
    if Image is None or asset is None:
        return None

    raw = str(getattr(asset, "path", "") or "")
    resolved = str(getattr(asset, "resolvedPath", "") or "")

    # Check the resolved path first: a texture inside a USDZ is authored
    # relatively ("0/tex.jpeg") and only the resolved path names the archive.
    archive = _archive_member(resolved, usd_dir) or _archive_member(raw, usd_dir)

    if archive:
        # The archive's mtime/size stands in for the member's: a member's own
        # timestamp isn't visible without extracting it, and any edit to the
        # zip is a new zip anyway.
        stat_path, key = archive[0], f"{archive[0]}[{archive[1]}]"
    elif resolved:
        stat_path = key = resolved
    else:
        # Anchor a bare relative reference to the USD that names it, so two
        # objects using the same filename do not collide in the cache.
        stat_path = key = str((usd_dir / raw).resolve())
    # Keyed on file identity, not just path -- otherwise an edited texture
    # keeps serving its old pixels for the life of the server process, same
    # as `usd_cache`'s stage cache would without this.
    key = (key, usd_cache.cache_key(stat_path))
    if key in _texture_cache:
        return _texture_cache[key]

    img = None
    try:
        if archive:
            with zipfile.ZipFile(archive[0]) as zf:
                with zf.open(archive[1]) as fh:
                    img = Image.open(io.BytesIO(fh.read())).convert("RGB")
        else:
            img = Image.open(resolved or str((usd_dir / raw).resolve())).convert("RGB")
    except Exception as e:
        print(f"      ! texture {raw}: {type(e).__name__}: {e}")
        img = None

    if img is not None:
        if max(img.size) > MAX_TEXTURE_PX:
            scale = MAX_TEXTURE_PX / max(img.size)
            img = img.resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                Image.LANCZOS,
            )
        # Re-encode as JPEG; trimesh would otherwise embed photographic scans
        # as very large PNGs.
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=88)
        buffer.seek(0)
        img = Image.open(buffer)
        # PIL opens lazily; force the pixels in so the image does not depend on
        # this buffer surviving until trimesh writes the glTF.
        img.load()

    _texture_cache[key] = img
    return img


def _texture_from_shader(shader, usd_dir):
    """Pull a diffuse texture out of a shader, covering both conventions here.

    OmniGibson's exported objects use MDL shaders with a ``diffuse_texture``
    input; the scanned mesh backgrounds use UsdPreviewSurface with
    ``diffuseColor`` connected to a UsdUVTexture.
    """
    direct = shader.GetInput("diffuse_texture")
    if direct:
        img = _open_asset_image(direct.Get(), usd_dir)
        if img is not None:
            return img

    diffuse = shader.GetInput("diffuseColor")
    if diffuse:
        source = diffuse.GetConnectedSource()
        if source:
            upstream = UsdShade.Shader(source[0].GetPrim())
            file_input = upstream.GetInput("file")
            if file_input:
                return _open_asset_image(file_input.Get(), usd_dir)

    file_input = shader.GetInput("file")
    if file_input:
        return _open_asset_image(file_input.Get(), usd_dir)
    return None


def _bound_texture(prim, usd_dir):
    """Find the diffuse texture (or, failing that, a flat colour) bound to a mesh prim.

    Returns (image, color) — either may be None. A constant colour is a fallback
    for shaders authored with no texture at all (e.g. MDL-derived materials that
    set a color3f like ``diffuse_color_constant`` instead of a texture file).
    """
    try:
        # Arity of ComputeBoundMaterial has varied across USD releases; take the
        # material positionally rather than unpacking a fixed-width tuple.
        result = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        material = result[0] if isinstance(result, tuple) else result
    except Exception as e:
        print(f"      ! material lookup failed on {prim.GetPath()}: {type(e).__name__}: {e}")
        return None, None
    if not material:
        return None, None

    # Prefer the surface shader; fall back to any shader in the network.
    shaders = [c for c in Usd.PrimRange(material.GetPrim()) if c.IsA(UsdShade.Shader)]
    shaders.sort(key=lambda c: UsdShade.Shader(c).GetShaderId() != "UsdPreviewSurface")
    color = None
    for child in shaders:
        shader = UsdShade.Shader(child)
        img = _texture_from_shader(shader, usd_dir)
        if img is not None:
            return img, None
        if color is None:
            # Parameter names vary by material (UsdPreviewSurface diffuseColor,
            # vMaterials color_front, etc.); take any color3f input, preferring
            # one that looks diffuse-ish over e.g. a specular tint.
            color3f_inputs = [i for i in shader.GetInputs()
                               if i.GetTypeName() == Sdf.ValueTypeNames.Color3f and i.Get() is not None]
            color3f_inputs.sort(key=lambda i: "diffuse" not in i.GetBaseName().lower()
                                 and "color" not in i.GetBaseName().lower())
            if color3f_inputs:
                color = tuple(float(c) for c in color3f_inputs[0].Get())
    return None, color


_extractor_version = None


#: Every module whose source decides what a proxy contains. All are hashed
#: into the extractor version so a change to any of them invalidates cached
#: proxies; a test asserts this list matches what an extraction imports.
_VERSIONED_MODULES = ("extract.py", "robot_pose.py", "scene_io.py", "splat_io.py",
                      "usd_cache.py")


def extractor_version():
    """A digest of the extractor's own source, stamped into every manifest.

    The proxy cache is validated against the scene hash and the mtime of every
    USD read, which covers the inputs changing but not the extractor; this
    stamp invalidates cached proxies when the extractor itself changes.
    """
    global _extractor_version
    if _extractor_version is None:
        digest = hashlib.sha256()
        here = Path(__file__).resolve().parent
        for name in _VERSIONED_MODULES:
            try:
                digest.update((here / name).read_bytes())
            except OSError:
                return "unknown"
        _extractor_version = digest.hexdigest()[:12]
    return _extractor_version


def _unweld_face_varying(points, indices, slots, st):
    """Split a face-varying mesh into vertices a glTF can carry, or decline.

    A face-varying UV belongs to a face *corner*, not to a point, so a point on
    a UV seam needs one copy per distinct UV it is given. glTF has one UV per
    vertex, so the mesh must be split before it can be written. The cost is the
    number of distinct ``(point, u, v)`` triples, which is what this measures.

    Args:
        points (np.ndarray): (n, 3) positions, already in stage-root space.
        indices (np.ndarray): Face-vertex indices into *points*.
        slots (np.ndarray): (t, 3) fan-triangulated indices into *indices*.
        st (np.ndarray): (len(indices), 2) UVs, one per face corner.

    Returns:
        tuple or None: ``(vertices, faces, uvs)``, or None when the split would
        cost more vertices than :data:`UNWELD_VERT_BUDGET` — the caller then
        emits the mesh untextured.
    """
    corners = slots.reshape(-1)
    # One row per corner: its point index and UV. Identical rows are the same
    # vertex; rows differing only in UV are seam copies.
    rows = np.empty((corners.size, 3), dtype=np.float64)
    rows[:, 0] = indices[corners]        # exact: a point index is far below 2^53
    rows[:, 1:] = st[corners]
    unique, inverse = np.unique(rows, axis=0, return_inverse=True)
    if len(unique) > UNWELD_VERT_BUDGET:
        return None
    vertices = points[unique[:, 0].astype(np.int64)]
    uvs = unique[:, 1:]
    # `inverse` is the new index of each corner, in corner order, so the
    # triangles fall straight out with winding preserved.
    faces = inverse.reshape(-1, 3).astype(np.int64)
    return vertices, faces, uvs


def _implicit_prim_geometry(prim):
    """Local-frame (points, faces) for a USD implicit gprim, or None.

    ``UsdGeom.Cube`` (and friends) have no explicit points -- they are defined
    by a size/radius attribute -- so a workstation built from them would
    otherwise render as nothing but its `Mesh`-typed trim.
    """
    if not prim.IsA(UsdGeom.Cube):
        return None
    size = UsdGeom.Cube(prim).GetSizeAttr().Get()
    box = trimesh.creation.box(extents=[size or 2.0] * 3)
    return np.asarray(box.vertices, dtype=np.float64), np.asarray(box.faces, dtype=np.int64)


def _prim_to_trimesh(prim, cache, usd_dir, allow_texture, correction=None):
    """Convert one UsdGeom.Mesh or implicit gprim into a trimesh, baked into
    stage-root space.

    Args:
        prim (Usd.Prim): Mesh or implicit-gprim (e.g. Cube) prim.
        cache (UsdGeom.XformCache): Shared transform cache.
        usd_dir (Path): Directory of the USD file.
        allow_texture (bool): Whether to attempt texture extraction.
        correction (np.ndarray or None): (4, 4) row-vector transform applied
            after the prim's own local-to-world, moving this mesh's link from the
            USD's rest configuration to a saved joint configuration.

    Returns:
        trimesh.Trimesh or None
    """
    implicit = _implicit_prim_geometry(prim)
    if implicit is not None:
        points, faces = implicit
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        if correction is not None:
            matrix = matrix @ correction
        points = points @ matrix[:3, :3] + matrix[3, :3]
        return _build(points, faces, None, prim, usd_dir, allow_texture)

    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not points or not counts or not indices:
        return None

    slots = _fan_slots(list(counts))
    if slots.size == 0:
        return None

    points = np.asarray(points, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)

    # USD is row-vector: world = local * M, translation in the last row.
    matrix = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
    # Row-vector, so the joint correction composes on the right.
    if correction is not None:
        matrix = matrix @ correction
    points = points @ matrix[:3, :3] + matrix[3, :3]

    uvs = None
    if allow_texture:
        primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("primvars:st")
        if primvar:
            # ComputeFlattened applies authored primvar indices; Get() alone
            # returns indexed face-varying UVs out of face-vertex order.
            try:
                st = primvar.ComputeFlattened()
            except Exception:
                st = primvar.Get()
            interp = primvar.GetInterpolation()
            if st is not None:
                st = np.asarray(st, dtype=np.float64)
                if interp == "faceVarying" and len(st) == len(indices):
                    unwelded = _unweld_face_varying(points, indices, slots, st)
                    if unwelded is not None:
                        verts, faces, uvs = unwelded
                        return _build(verts, faces, uvs, prim, usd_dir, allow_texture)
                if interp in ("vertex", "varying") and len(st) == len(points):
                    uvs = st

    faces = indices[slots]
    return _build(points, faces, uvs, prim, usd_dir, allow_texture)


def _build(vertices, faces, uvs, prim, usd_dir, allow_texture):
    """Assemble a trimesh, attaching a texture when UVs and an image exist."""
    visual = None
    if allow_texture:
        img, color = _bound_texture(prim, usd_dir)
        if img is not None and uvs is not None:
            # Build the PBR material directly; passing image= would make a
            # SimpleMaterial whose grey default diffuse darkens every texture.
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=img,
                baseColorFactor=[255, 255, 255, 255],
                metallicFactor=0.0,
                roughnessFactor=0.9,
            )
            visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
        elif color is not None:
            material = trimesh.visual.material.PBRMaterial(
                baseColorFactor=[*(round(c * 255) for c in color), 255],
                metallicFactor=0.0,
                roughnessFactor=0.9,
            )
            visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    return trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)


def load_visual_scene(usd_path, allow_texture=True, joint_pose=None):
    """Load the visible render geometry of a USD as a trimesh Scene.

    Each mesh prim stays a separate geometry so per-mesh materials survive into
    glTF, and each node is named for its prim via `scene_io.node_name_for`.
    Everything is baked into the stage's root frame, so the result is
    positioned by the scene JSON's pose and scale exactly as OmniGibson does.

    Args:
        usd_path (str or Path): USD file to read.
        allow_texture (bool): Extract diffuse textures and UVs.
        joint_pose (robot_pose.RobotJointPose or None): Solved articulation. When
            given, each mesh is baked at its link's saved joint configuration
            instead of the USD's rest pose.

    Returns:
        tuple[trimesh.Scene or None, int, int, bool, list[str]]: scene, verts,
            faces, textured, and one diagnostic per prim that failed to convert.
            A non-empty diagnostic list means the proxy is incomplete even when
            a scene came back.
    """
    stage = usd_cache.open_stage(usd_path)
    if stage is None:
        return None, 0, 0, False, [f"could not open stage: {usd_path}"]

    usd_dir = Path(usd_path).parent
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = trimesh.Scene()
    verts = faces = 0
    textured = False
    failures = []

    # Assets processed with make_instanceable=True (common for BEHAVIOR-1K-style
    # imports) carry their real geometry only under instance proxies; plain
    # Traverse() skips into instances at all and silently sees nothing there.
    # Restricting to the default prim's own subtree also skips the file's
    # uninstantiated prototype/library prims (e.g. a root-level `/meshes`
    # scope) sitting outside it, which would otherwise double the geometry.
    default_prim_path = stage.GetDefaultPrim().GetPath() if stage.HasDefaultPrim() else None
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if default_prim_path and not prim.GetPath().HasPrefix(default_prim_path):
            continue
        if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube)):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputePurpose() in SKIP_PURPOSES:
            continue
        if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue

        correction = None if joint_pose is None else joint_pose.correction_for(prim.GetPath())
        try:
            geom = _prim_to_trimesh(prim, cache, usd_dir, allow_texture, correction)
        except Exception as e:
            # Recorded so a partially converted object reports as degraded.
            detail = f"{prim.GetPath()}: {type(e).__name__}: {e}"
            print(f"      ! {detail}")
            failures.append(detail)
            continue
        if geom is None or len(geom.faces) == 0:
            continue

        if isinstance(geom.visual, trimesh.visual.TextureVisuals):
            textured = True
        verts += len(geom.vertices)
        faces += len(geom.faces)
        # Encoded, not raw: three.js strips slashes from node names.
        # `scene_io.node_name_for` defines the encoding shared with
        # `read_usd_joints`'s ``child_node``.
        out.add_geometry(geom, node_name=node_name_for(prim.GetPath()))

    if not out.geometry:
        return None, 0, 0, False, failures
    return out, verts, faces, textured, failures


def native_extent(geom):
    """The asset's bounding box at scale 1, in its own frame.

    The editor solves ``scale = measured_size / native_size``, so this is
    measured on exactly the geometry the browser draws: the visible render
    prims baked into the stage's root frame by :func:`load_visual_scene`, the
    same frame OmniGibson's ``scale`` multiplies.

    Args:
        geom (trimesh.Scene): The scene :func:`load_visual_scene` returned.

    Returns:
        tuple[list[float], list[float]]: ``(size, centre)``, each three metres.
            The centre is not assumed to be the origin; a resize that keeps an
            object standing needs the pivot's offset from the box.
    """
    low, high = np.asarray(geom.bounds, dtype=float)
    size = high - low
    centre = (high + low) / 2.0
    # Round to 6 dp (a micrometre) so the manifest stays diffable.
    return [round(float(v), 6) for v in size], [round(float(v), 6) for v in centre]


# How faithfully a typed size survives into OmniGibson; both failure modes are
# authored facts about the USD, so this is measured on the stage.
#
#   linear       one rigid body, or several with no offset between them: the
#                box is exactly native x scale, in the browser and in the sim.
#   link-offset  a link sits at a non-zero xformOp:translate. OmniGibson scales
#                each link about its own origin and leaves the translate alone,
#                so gaps between parts do not scale; the browser scales the
#                whole baked proxy, and the two diverge.
#   root-scale   the default prim authors a non-unit xformOp:scale. OmniGibson
#                rewrites every link's scale from that authored value and
#                discards the requested scale outright.
SCALE_LINEAR = "linear"
SCALE_LINK_OFFSET = "link-offset"
SCALE_ROOT_SCALE = "root-scale"

# Below this a translate or a scale deviation is authoring noise, not intent.
_SCALE_EPS = 1e-6


def scale_fidelity(stage):
    """Classify how OmniGibson will treat a scale applied to this asset.

    ``scale = target_size / native_size`` only holds where the box is linear in
    the scale. Recorded per asset in the manifest so the browser can warn
    without opening a USD.

    Args:
        stage (Usd.Stage): An open stage for the asset.

    Returns:
        dict: ``kind`` (one of the three constants), ``rootScale`` (the authored
            default-prim scale, or None) and ``offsetLinks`` (names of links
            sitting away from the object's origin), so a warning can name the
            cause.
    """
    result = {"kind": SCALE_LINEAR, "rootScale": None, "offsetLinks": []}
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsA(UsdGeom.Xformable):
        return result

    for op in UsdGeom.Xformable(default_prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            value = op.Get()
            if value is not None:
                result["rootScale"] = [round(float(v), 6) for v in value]

    links = []
    for child in default_prim.GetChildren():
        if not child.IsA(UsdGeom.Xformable):
            continue
        # Only links carrying geometry matter; a metadata-only Scope
        # contributes nothing to the box.
        if not any(prim.IsA(UsdGeom.Mesh) for prim in Usd.PrimRange(child)):
            continue
        translate = None
        for op in UsdGeom.Xformable(child).GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate = op.Get()
        offset = (translate is not None
                  and max(abs(float(v)) for v in translate) > _SCALE_EPS)
        links.append((child.GetName(), offset))

    # root-scale wins: such an asset loses its scale entirely, so offset links
    # never get the chance to matter.
    root = result["rootScale"]
    if root is not None and any(abs(v - 1.0) > _SCALE_EPS for v in root):
        result["kind"] = SCALE_ROOT_SCALE
        return result

    # A single link scales about its own origin exactly, whatever its
    # translate, so only multi-link assets can be offset.
    if len(links) > 1:
        result["offsetLinks"] = [name for name, offset in links if offset]
        if result["offsetLinks"]:
            result["kind"] = SCALE_LINK_OFFSET
    return result


def _write_atomic(path, payload, mode="wb"):
    """Write bytes or text to *path* via a sibling temporary file."""
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, mode, **({} if "b" in mode else {"encoding": "utf-8"})) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            Path(tmp_name).unlink()
        except FileNotFoundError:
            pass
        raise


def build_proxy(usd_path, out_dir, glb_name, textures=True, joint_pose=None,
                splat_budget=splat_io.DEFAULT_SPLAT_BUDGET, on_progress=None):
    """Convert one USD to a browser proxy and describe the result.

    The returned fields are the asset-derived half of a manifest entry; the
    caller supplies the scene-derived half (name, pose, editability).

    Ordinary geometry becomes a ``.glb``. A NuRec Gaussian-splat room has no
    mesh prims and becomes a ``.splat`` instead — same stem, different suffix —
    with ``splat`` set in its entry where a mesh carries ``glb``.

    Args:
        usd_path (Path): USD to convert.
        out_dir (Path): Directory the proxy is written to.
        glb_name (str): Filename for the proxy, relative to *out_dir*. A splat
            room uses the same stem with a ``.splat`` suffix.
        textures (bool): Extract diffuse textures.
        joint_pose (robot_pose.RobotJointPose or None): Solved articulation to
            bake, for a robot whose saved ``joint_pos`` could be mapped.
        splat_budget (int or None): Gaussian budget for a splat room.
        on_progress (callable or None): Receives status strings while a large
            room is read.

    Returns:
        dict: ``glb``, ``splat``, ``status``, ``error``, ``textured``, ``verts``,
        ``faces``, ``upAxis``, ``metersPerUnit``, ``nativeSize``,
        ``nativeCentre``, ``scaleFidelity``, ``prim_failures``,
        ``sourceMtimeNs``. ``glb`` is None and ``status`` is ``"error"`` when
        the USD has no convertible visible geometry.
    """
    usd_path = Path(usd_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if splat_io.is_nurec_usd(usd_path):
        splat_name = str(Path(glb_name).with_suffix(".splat"))
        try:
            return splat_io.build_splat_proxy(
                usd_path, out_dir, splat_name,
                budget=splat_budget, on_progress=on_progress,
            )
        except splat_io.SplatReadError as e:
            # An unreadable room is an error row, exactly as a USD with no
            # visible geometry is.
            return {
                "glb": None, "splat": None, "status": "error", "error": str(e),
                "prim_failures": [], "textured": False, "verts": 0, "faces": 0,
                "nativeSize": None, "nativeCentre": None, "scaleFidelity": None,
            }

    geom, nverts, nfaces, textured, failures = load_visual_scene(usd_path, textures, joint_pose)
    if geom is None:
        error = "USD contains no supported visible mesh geometry"
        if failures:
            error += f" ({len(failures)} prim(s) failed to convert)"
        return {
            "glb": None, "splat": None, "status": "error", "error": error,
            "prim_failures": failures, "textured": False, "verts": 0, "faces": 0,
            # Every key the docstring promises, on the failure path too.
            "nativeSize": None, "nativeCentre": None, "scaleFidelity": None,
        }

    _write_atomic(out_dir / glb_name, geom.export(file_type="glb"))

    native_size, native_centre = native_extent(geom)

    stage = usd_cache.open_stage(usd_path)
    return {
        "glb": glb_name,
        # Present-but-null so a browser can branch on which kind of proxy an
        # entry carries.
        "splat": None,
        # Recorded, not applied: USD references do not rescale authored points
        # for a referenced layer's stage metadata, and leaving geometry raw
        # matches OmniGibson's USD reference path.
        "upAxis": str(UsdGeom.GetStageUpAxis(stage)),
        "metersPerUnit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        # A proxy that lost prims is not "ok": the user would place against
        # part of an object.
        "status": "degraded" if failures else "ok",
        "error": (
            f"{len(failures)} prim(s) failed to convert; proxy is incomplete"
            if failures else None
        ),
        "prim_failures": failures,
        "textured": textured,
        "verts": int(nverts),
        "faces": int(nfaces),
        # The asset's own size, measured on the exported geometry so the number
        # the panel shows is the box the viewer draws.
        "nativeSize": native_size,
        # Where the pivot sits relative to the box; recorded for headless or
        # server-side resizes, not applied here.
        "nativeCentre": native_centre,
        # Whether "scale = target / native" will actually hold in OmniGibson.
        "scaleFidelity": scale_fidelity(stage),
        # Lets --no-extract notice a USD edited since this proxy was built.
        "sourceMtimeNs": _mtime_ns(usd_path),
    }


def _joint_pose_for(scene, record, on_warning):
    """Solve one object's saved articulation, and describe the outcome.

    Args:
        scene (dict): Parsed scene JSON.
        record (dict): One record from ``scene_io.iter_objects``.
        on_warning (callable): Receives one string per reason a pose was skipped.

    Returns:
        tuple: ``(pose, manifest)``. *pose* is a ``robot_pose.RobotJointPose`` or
            None; *manifest* is the ``jointPose`` entry, or None when the object
            has no saved articulation worth reporting.
    """
    registry = scene.get("state", {}).get("registry", {}).get("object_registry", {})
    if record["name"] not in registry:
        return None, None
    joint_pos = robot_pose.saved_joint_pos(scene, record["name"])

    if record["kind"] != "robot":
        # Articulated props are drawn at rest: their joint order is PhysX
        # metadata no pinned table covers. Report that rather than pose them
        # wrong.
        if not robot_pose.is_articulated(joint_pos):
            return None, None
        return None, {
            "applied": False,
            "reason": "articulated props are not posed; only the robots pinned in "
                      "robot_pose.ROBOT_ARM_JOINTS have a verified joint ordering",
            "values": [float(v) for v in joint_pos],
        }

    info = scene.get("objects_info", {}).get("init_info", {}).get(record["name"], {})
    class_name = info.get("class_name")
    end_effector = info.get("args", {}).get("end_effector")
    if joint_pos is None:
        return None, {"applied": False, "reason": "the saved state records no joint_pos"}

    reasons = []
    pose = robot_pose.solve_arm_pose(
        record["usd"], class_name, end_effector, joint_pos, on_warning=reasons.append
    )
    for reason in reasons:
        on_warning(reason)
    if pose is None:
        return None, {
            "applied": False,
            "reason": reasons[0] if reasons else "joint pose could not be solved",
        }
    return pose, pose.as_manifest()


def extract(scene_json, out_dir, robot_asset_dir, textures=True, dataset_dir=None,
            splat_budget=splat_io.DEFAULT_SPLAT_BUDGET):
    """Write per-object proxies plus a manifest describing the scene.

    Args:
        scene_json (Path): Scene-state JSON to read.
        out_dir (Path): Directory for ``manifest.json``, ``*.glb`` and
            ``*.splat``.
        robot_asset_dir (Path): Root of ``omnigibson-robot-assets``.
        textures (bool): Extract diffuse textures.
        dataset_dir (Path or None): OmniGibson's ``gm.DATA_PATH``, holding the
            dataset trees that ``DatasetObject`` entries are resolved against.
        splat_budget (int or None): Gaussians a NuRec room may keep; 0 or None
            keeps all of them.

    Returns:
        dict: The manifest that was written.
    """
    scene = load_scene(scene_json)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for index, record in enumerate(
        iter_objects(scene, scene_json, robot_asset_dir, dataset_dir)
    ):
        name = record["name"]

        entry = {
            "name": name,
            "category": record["category"],
            "kind": record["kind"],
            "editable": record["editable"],
            # The browser gates *selection* on `posable`, not `editable`.
            "posable": record["posable"],
            # Distinguishes "cannot be scaled" (the robot) from "cannot be
            # touched at all" (the room).
            "scalable": record["scalable"],
            # Mass, friction, and the link name a friction material addresses;
            # read here because only this side can open the object's USD.
            "physics": record["physics"],
            # None for anything with no degrees of freedom; the browser shows
            # the joints panel exactly when this is present.
            "joints": record["joints"],
            "glb": None,
            "splat": None,
            "position": record["position"],
            "orientation": record["orientation"],
            "scale": record["scale"],
            "sourceUsd": str(record["usd"]) if record["usd"] is not None else record["usd_reference"],
            "status": "error",
            "error": None,
            "textured": False,
            "verts": 0,
            "faces": 0,
            # Present-but-null so a consumer can tell "no measurable geometry"
            # from "manifest predates the field".
            "nativeSize": None,
            "nativeCentre": None,
            "scaleFidelity": None,
        }

        if record["usd"] is None:
            entry["error"] = "no resolvable USD visual asset"
            entries.append(entry)
            print(f"  {name:24s} ERROR no resolvable USD")
            continue

        joint_pose, joint_manifest = _joint_pose_for(
            scene, record, lambda msg: print(f"  {name:24s} ! {msg}")
        )

        # Robots are context, not editing subjects; skip their textures.
        allow_texture = textures and record["kind"] != "robot"
        # Never use a scene-controlled object name as a path; opaque IDs also
        # avoid platform filename and Unicode normalization collisions.
        proxy = build_proxy(
            record["usd"], out_dir, f"asset_{index:04d}.glb", allow_texture, joint_pose,
            splat_budget=splat_budget,
            on_progress=lambda msg: print(f"  {name:24s} {msg}"),
        )
        entry.update(proxy)
        if joint_manifest is not None:
            entry["jointPose"] = joint_manifest
        entries.append(entry)

        if proxy.get("splat"):
            kept, total = proxy["splatCount"], proxy["splatTotal"]
            thinned = "" if kept == total else f" of {total:,}"
            size = " x ".join(f"{v:.3f}" for v in proxy["nativeSize"])
            print(
                f"  {name:24s} {kept:9,d} gaussians{thinned}  "
                f"{proxy['splatBytes'] / 1e6:6.1f} MB  splat    "
                f"up={proxy['upAxis']} m/unit={proxy['metersPerUnit']:g}  native {size} m"
            )
            continue

        if proxy["glb"] is None:
            print(f"  {name:24s} ERROR {proxy['error'] or 'no visible render geometry'}")
            continue

        mb = (out_dir / proxy["glb"]).stat().st_size / 1e6
        tag = "textured" if proxy["textured"] else "flat"
        failures = proxy["prim_failures"]
        flag = f"  DEGRADED ({len(failures)} prim failures)" if failures else ""
        if joint_pose is not None:
            flag += f"  joints={len(joint_pose.joint_names)} ({joint_pose.robot})"
        size = " x ".join(f"{v:.3f}" for v in proxy["nativeSize"])
        fidelity = proxy["scaleFidelity"]["kind"]
        if fidelity != SCALE_LINEAR:
            flag += f"  SCALE {fidelity}"
        print(
            f"  {name:24s} {proxy['verts']:7d} verts {proxy['faces']:7d} tris  "
            f"{mb:6.1f} MB  {tag:8s} up={proxy['upAxis']} "
            f"m/unit={proxy['metersPerUnit']:g}  native {size} m{flag}"
        )

    errors = [
        {"name": entry["name"], "error": entry["error"], "status": entry["status"]}
        for entry in entries
        if entry["status"] != "ok"
    ]
    manifest = {
        "schema": "simfoundry.light_editor.manifest.v1",
        "scene_json": str(Path(scene_json).resolve()),
        # Two digests of the same bytes at extraction time; they diverge once a
        # promotion rewrites the scene file:
        #   base_scene_sha256    identity — echoed by the browser on every save
        #                        and checked by the server.
        #   source_scene_sha256  currency — which on-disk revision this
        #                        extraction is good for; --no-extract validates
        #                        against it.
        "base_scene_sha256": scene_sha256(scene_json),
        "source_scene_sha256": scene_sha256(scene_json),
        "extractor_version": extractor_version(),
        "complete": not errors,
        "errors": errors,
        "objects": entries,
    }
    write_manifest(manifest, out_dir)
    return manifest


def write_manifest(manifest, out_dir):
    """Persist a manifest, recomputing ``complete``/``errors`` from its entries.

    The server rewrites the manifest whenever an object is imported, so the
    derived fields must follow.
    """
    manifest["errors"] = [
        {"name": entry["name"], "error": entry["error"], "status": entry["status"]}
        for entry in manifest["objects"]
        if entry["status"] != "ok"
    ]
    manifest["complete"] = not manifest["errors"]
    _write_atomic(
        Path(out_dir) / "manifest.json",
        json.dumps(manifest, indent=2, allow_nan=False),
        mode="w",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Extract scene geometry to glTF")
    parser.add_argument("--scene", required=True, help="Path to a scene_state JSON")
    parser.add_argument("--out", default=None, help="Output dir (default: web/data)")
    parser.add_argument("--no_textures", action="store_true", help="Geometry only")
    parser.add_argument(
        "--robot_asset_dir",
        default=None,
        help=f"Root of omnigibson-robot-assets (default: <repo>/{DEFAULT_ROBOT_ASSET_DIR})",
    )
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Directory holding the dataset trees that DatasetObject entries name, i.e. "
             f"OmniGibson's gm.DATA_PATH (default: <repo>/{DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--splat_budget",
        type=int,
        default=splat_io.DEFAULT_SPLAT_BUDGET,
        help="Most gaussians a NuRec background may keep; 0 keeps all "
             f"(default: {splat_io.DEFAULT_SPLAT_BUDGET})",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]
    out_dir = Path(args.out) if args.out else here / "web" / "data"
    robot_dir = Path(args.robot_asset_dir) if args.robot_asset_dir else repo_root / DEFAULT_ROBOT_ASSET_DIR
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else repo_root / DEFAULT_DATASET_DIR

    scene_json = Path(args.scene).resolve()
    if not scene_json.exists():
        sys.exit(f"ERROR: scene JSON not found: {scene_json}")

    print(f"Extracting {scene_json.name}")
    manifest = extract(
        scene_json, out_dir, robot_dir, textures=not args.no_textures,
        dataset_dir=dataset_dir, splat_budget=args.splat_budget,
    )
    print(f"\n{len(manifest['objects'])} objects -> {out_dir}")


if __name__ == "__main__":
    main()
