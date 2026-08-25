# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bring a USD or mesh asset from anywhere on this machine into a scene.

A USD is copied in with its full dependency closure. A mesh (glTF, OBJ, PLY,
STL, ...) is converted to a USD laid out the way the reconstruction pipeline
emits and OmniGibson loads::

    /<asset_id>                        Xform, default prim
      /Looks/material_<i>              Material  (UsdPreviewSurface + OmniPBR MDL)
      /link_geometry_0                 Xform, PhysicsRigidBodyAPI + PhysicsMassAPI
        /visuals/mesh_<i>              Mesh, MaterialBindingAPI
        /collisions/collision_0        Mesh, PhysicsCollisionAPI + PhysicsMeshCollisionAPI

Each material carries both shaders: OmniPBR MDL for Omniverse renderers,
UsdPreviewSurface for everything else.

Scenes are Z-up in metres. glTF sources are Y-up and are rotated to Z-up by
default; OBJ, PLY and STL carry no up convention and are left alone. Converted
meshes get a generated collider: ``convexHull`` suits most props,
``convexDecomposition`` is needed for hollow shapes like bowls or mugs.
"""

import re
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scene_io import MASS_BOUNDS, SceneEditError  # noqa: E402

USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})
MESH_SUFFIXES = frozenset({".glb", ".gltf", ".obj", ".ply", ".stl", ".off", ".dae"})

# Formats that are Y-up by specification; the others have no up convention.
Y_UP_SUFFIXES = frozenset({".glb", ".gltf", ".dae"})

COLLISION_APPROXIMATIONS = ("convexHull", "convexDecomposition", "boundingCube", "none")

# Fallback density in kg/m^3 for the mass estimate; mass can be overridden.
DEFAULT_DENSITY = 500.0

# Imported textures are downscaled to at most this many pixels on the long edge.
MAX_TEXTURE_PX = 2048


def slug(text):
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", str(text)).strip("_-").lower()
    return cleaned or "asset"


def classify(path):
    """Return ``'usd'``, ``'mesh'``, ``'directory'`` or None for *path*."""
    path = Path(path)
    if path.is_dir():
        return "directory"
    suffix = path.suffix.lower()
    if suffix in USD_SUFFIXES:
        return "usd"
    if suffix in MESH_SUFFIXES:
        return "mesh"
    return None


def resolve_user_path(raw):
    """Resolve a user-pasted path, accepting quotes, ``file://`` URLs and ``~``."""
    if not isinstance(raw, str) or not raw.strip():
        raise SceneEditError("enter a path to a USD or mesh file")
    text = raw.strip().strip("'\"").strip()
    if text.startswith("file://"):
        from urllib.parse import unquote, urlsplit
        text = unquote(urlsplit(text).path)
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise SceneEditError(
            f"give an absolute path (got {text!r}) — the server's working "
            "directory is not where you are looking"
        )
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise SceneEditError(f"no such file or directory: {path}") from None
    if classify(resolved) is None:
        supported = ", ".join(sorted(USD_SUFFIXES | MESH_SUFFIXES))
        raise SceneEditError(
            f"{resolved.name}: unsupported file type. Supported: {supported}"
        )
    return resolved


# --- importing a USD -------------------------------------------------------

def _dependency_closure(usd_path):
    """Every file the stage references (sublayers, references, payloads, assets),
    plus the unresolved reference strings."""
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(usd_path))
    files = {Path(usd_path).resolve()}
    for layer in layers:
        if layer and layer.realPath:
            files.add(Path(layer.realPath).resolve())
    for asset in assets:
        candidate = Path(str(asset))
        if candidate.exists():
            files.add(candidate.resolve())
    return sorted(files), [str(u) for u in unresolved]


def import_usd_file(usd_path, dest_dir):
    """Copy a USD and everything it references into *dest_dir*.

    Relative layout among the copied files is preserved, so references that
    resolved before still resolve after.

    Returns:
        tuple[Path, list[str]]: The copied root USD, and human-readable notes
        (unresolved references, and how many extra files came along).
    """
    usd_path = Path(usd_path).resolve()
    dest_dir = Path(dest_dir)
    notes = []

    if usd_path.suffix.lower() == ".usdz":
        # USDZ is a self-contained archive; copy it alone.
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / usd_path.name
        shutil.copy2(usd_path, target)
        return target, notes

    files, unresolved = _dependency_closure(usd_path)
    if unresolved:
        # Unresolved references (usually missing textures) are reported, not fatal.
        shown = ", ".join(Path(u).name for u in unresolved[:4])
        notes.append(f"{len(unresolved)} unresolved reference(s) not copied: {shown}")

    # Root the copy at the deepest common directory so relative references survive.
    import os
    root = Path(os.path.commonpath([str(f.parent) for f in files]))
    dest_dir.mkdir(parents=True, exist_ok=True)
    for source in files:
        target = dest_dir / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    if len(files) > 1:
        notes.append(f"copied {len(files)} file(s) from {root}")
    return dest_dir / usd_path.relative_to(root), notes


# --- converting a mesh -----------------------------------------------------

def _load_mesh_scene(mesh_path):
    loaded = trimesh.load(str(mesh_path), force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        loaded = trimesh.Scene(loaded)
    if not getattr(loaded, "geometry", None):
        raise SceneEditError(f"{Path(mesh_path).name} contains no mesh geometry")
    return loaded


def _world_meshes(scene):
    """Yield (name, mesh) with each geometry's scene-graph transform baked in."""
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        geometry = scene.geometry.get(geometry_name)
        if geometry is None or not isinstance(geometry, trimesh.Trimesh):
            continue
        if len(geometry.faces) == 0:
            continue
        mesh = geometry.copy()
        mesh.apply_transform(transform)
        yield geometry_name, mesh


def _texture_image(mesh):
    """The diffuse image for a mesh, or None."""
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    if material is None:
        return None
    for attribute in ("baseColorTexture", "image", "emissiveTexture"):
        image = getattr(material, attribute, None)
        if image is not None:
            return image
    return None


def _base_color(mesh):
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    for attribute in ("baseColorFactor", "diffuse"):
        value = getattr(material, attribute, None)
        if value is not None and len(value) >= 3:
            scale = 255.0 if max(value[:3]) > 1.0 else 1.0
            return tuple(float(c) / scale for c in value[:3])
    return (1.0, 1.0, 1.0)


def _write_texture(image, material_dir, name):
    from PIL import Image

    material_dir.mkdir(parents=True, exist_ok=True)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    if max(image.size) > MAX_TEXTURE_PX:
        ratio = MAX_TEXTURE_PX / max(image.size)
        image = image.resize(
            (max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
            Image.LANCZOS,
        )
    target = material_dir / f"{name}.png"
    image.save(target)
    return target


def _define_material(stage, looks_path, index, texture_relative, base_color):
    """Author one material carrying both a preview and an MDL shader."""
    material_path = looks_path.AppendChild(f"material_{index}")
    material = UsdShade.Material.Define(stage, material_path)

    preview = UsdShade.Shader.Define(stage, material_path.AppendChild("preview"))
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    mdl = UsdShade.Shader.Define(stage, material_path.AppendChild("mdl_shader"))
    mdl.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    mdl.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    mdl.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.85)
    mdl.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)

    if texture_relative:
        reader = UsdShade.Shader.Define(stage, material_path.AppendChild("st_reader"))
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        texture = UsdShade.Shader.Define(stage, material_path.AppendChild("diffuse_tex"))
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(texture_relative))
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result")
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture.ConnectableAPI(), "rgb")
        # The texture carries the colour; a tint would double-apply it.
        mdl.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(1.0, 1.0, 1.0))
        mdl.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(texture_relative))
    else:
        colour = Gf.Vec3f(*base_color)
        preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(colour)
        mdl.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(colour)

    material.CreateSurfaceOutput().ConnectToSource(
        preview.ConnectableAPI(), "surface")
    material.CreateSurfaceOutput("mdl").ConnectToSource(
        mdl.ConnectableAPI(), "out")
    return material


def _author_mesh(stage, path, mesh, *, uv=None):
    """Write a trimesh as a UsdGeom.Mesh and return the prim."""
    usd_mesh = UsdGeom.Mesh.Define(stage, path)
    points = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    usd_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    usd_mesh.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    # Without an explicit scheme a renderer may subdivide the mesh.
    usd_mesh.CreateSubdivisionSchemeAttr("none")
    # Gf.Vec3f rejects numpy scalars, so hand it plain floats.
    lo = [float(v) for v in points.min(axis=0)]
    hi = [float(v) for v in points.max(axis=0)]
    usd_mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))

    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if normals.shape == points.shape:
        usd_mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
        usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    if uv is not None and len(uv) == len(points):
        primvar = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        primvar.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, dtype=np.float32)))
    return usd_mesh


def _estimate_mass(mesh, density):
    """Estimate a mass in kg from volume, clamped to MASS_BOUNDS.

    ``volume`` is unreliable for a mesh that is not watertight, so those fall
    back to a fraction of the bounding-box volume.
    """
    volume = 0.0
    try:
        if mesh.is_watertight:
            volume = float(abs(mesh.volume))
    except Exception:  # noqa: BLE001 - any geometry failure means "unknown"
        volume = 0.0
    if volume <= 0.0:
        # Not watertight: a third of the bounding-box volume is the fallback.
        volume = float(np.prod(mesh.extents)) / 3.0
    return float(np.clip(volume * density, *MASS_BOUNDS))


def prepare_mesh(mesh_path, *, scale=1.0, up_axis="auto"):
    """Load a mesh file and transform it into scene coordinates (Z-up, metres).

    Args:
        mesh_path (Path): Source mesh.
        scale (float): Uniform scale.
        up_axis (str): ``'auto'``, ``'y'`` or ``'z'``.

    Returns:
        tuple[list[tuple[int, str, trimesh.Trimesh]], bool]: The transformed
        geometries with their index and name, and whether a Y-up correction was
        applied.

    Raises:
        SceneEditError: If the file holds no triangulated geometry.
    """
    mesh_path = Path(mesh_path)
    scene = _load_mesh_scene(mesh_path)
    rotate = (up_axis == "y") or (
        up_axis == "auto" and mesh_path.suffix.lower() in Y_UP_SUFFIXES)

    transform = np.eye(4)
    if rotate:
        # +Y up becomes +Z up: rotate +90 degrees about X.
        transform = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    if float(scale) != 1.0:
        transform = trimesh.transformations.scale_matrix(float(scale)) @ transform

    meshes = []
    for index, (name, mesh) in enumerate(_world_meshes(scene)):
        mesh.apply_transform(transform)
        meshes.append((index, name, mesh))
    if not meshes:
        raise SceneEditError(f"{mesh_path.name} contains no triangulated mesh geometry")
    return meshes, rotate


def describe_mesh(mesh_path, *, scale=1.0, up_axis="auto", collision="convexHull",
                  mass=None, density=DEFAULT_DENSITY):
    """Report what importing this mesh would produce, without writing anything.

    Takes the same arguments as :func:`convert_mesh` and measures the same
    geometry the import would write.

    Returns:
        dict: ``size`` in metres, ``mass``, ``rotated``, ``verts``, ``faces``,
        ``meshes``, ``textured``, ``collision`` and ``caveats``.
    """
    if collision not in COLLISION_APPROXIMATIONS:
        raise SceneEditError(
            f"collision must be one of {', '.join(COLLISION_APPROXIMATIONS)}")
    if not (0.0 < float(scale) <= 1000.0):
        raise SceneEditError("scale must be greater than zero and at most 1000")

    meshes, rotated = prepare_mesh(mesh_path, scale=scale, up_axis=up_axis)
    combined = trimesh.util.concatenate([m for _, _, m in meshes])
    resolved_mass = float(mass) if mass else _estimate_mass(combined, density)
    size = [round(float(v), 4) for v in combined.extents]
    return {
        "size": size,
        "mass": round(float(np.clip(resolved_mass, *MASS_BOUNDS)), 4),
        "rotated": bool(rotated),
        "verts": int(sum(len(m.vertices) for _, _, m in meshes)),
        "faces": int(sum(len(m.faces) for _, _, m in meshes)),
        "meshes": len(meshes),
        "textured": any(_texture_image(m) is not None for _, _, m in meshes),
        "collision": collision,
        "caveats": mesh_caveats(size, collision),
        "watertight": bool(getattr(combined, "is_watertight", False)),
    }


def mesh_caveats(size, collision):
    """Warnings to show for a converted mesh."""
    caveats = []
    if collision == "none":
        caveats.append("no collider authored; this object will not collide")
    if max(size) > 5.0:
        caveats.append(
            f"bounding box is {max(size):.1f} m across — check the scale field "
            "if the source was authored in centimetres or millimetres")
    if collision == "convexHull":
        caveats.append(
            "convex-hull collider: a bowl or mug will behave as a solid lump. "
            "Re-import with convexDecomposition if it needs to be hollow")
    caveats.append(
        "collision is approximated here, not cooked by the pipeline's USD-import "
        "stage — settle the scene before relying on it")
    return caveats


def describe_usd(usd_path):
    """Read size, mass and collision info from a USD.

    Returns:
        dict: ``size`` (metres, from the render-purpose bounds), ``mass``,
        ``collision`` (the distinct approximations found), ``collision_prims``,
        ``up_axis``, ``meters_per_unit`` and ``caveats``.
    """
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise SceneEditError(f"could not open {Path(usd_path).name} as a USD stage")

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_,
                                                      UsdGeom.Tokens.render])
    default_prim = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    box = cache.ComputeWorldBound(default_prim).ComputeAlignedRange()
    size = ([0.0, 0.0, 0.0] if box.IsEmpty()
            else [round(float(v), 4) for v in box.GetSize()])

    masses, approximations, collision_prims = [], [], 0
    for prim in stage.Traverse():
        schemas = prim.GetAppliedSchemas()
        if "PhysicsMassAPI" in schemas:
            attribute = prim.GetAttribute("physics:mass")
            if attribute and attribute.HasAuthoredValue():
                masses.append(float(attribute.Get()))
        if "PhysicsCollisionAPI" in schemas:
            collision_prims += 1
            attribute = prim.GetAttribute("physics:approximation")
            value = attribute.Get() if attribute else None
            approximations.append(str(value) if value else "mesh")

    caveats = []
    if not collision_prims:
        caveats.append(
            "no collision geometry in this USD — it will fall through surfaces "
            "until the pipeline's USD-import stage gives it a collider")
    if not masses:
        caveats.append("no authored mass; PhysX will derive one from the collider")
    if max(size or [0]) > 5.0:
        caveats.append(f"bounding box is {max(size):.1f} m across")

    return {
        "size": size,
        "mass": round(sum(masses), 4) if masses else None,
        "collision": sorted(set(approximations)),
        "collision_prims": collision_prims,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "caveats": caveats,
    }


def convert_mesh(
    mesh_path, dest_dir, asset_id, *,
    scale=1.0, up_axis="auto", collision="convexHull", mass=None,
    density=DEFAULT_DENSITY,
):
    """Convert a mesh file into a scene-ready USD.

    Args:
        mesh_path (Path): Source ``.glb``/``.obj``/``.ply``/``.stl``/...
        dest_dir (Path): Bundle directory to create; the USD lands in
            ``<dest_dir>/usd/`` and textures in ``<dest_dir>/material/``.
        asset_id (str): Name for the USD and its default prim.
        scale (float): Uniform scale applied to the geometry, for a source
            authored in centimetres or inches.
        up_axis (str): ``'auto'`` (Y-up for glTF, unchanged otherwise),
            ``'y'`` to rotate, or ``'z'`` to leave alone.
        collision (str): One of :data:`COLLISION_APPROXIMATIONS`.
        mass (float or None): Kilograms. Estimated from volume when omitted.
        density (float): kg/m^3 used for that estimate.

    Returns:
        tuple[Path, dict]: The written USD, and a report carrying the final
        ``size`` in metres, ``mass``, ``rotated``, ``textured``, ``verts``,
        ``faces`` and a ``caveats`` list.

    Raises:
        SceneEditError: If the file holds no geometry or an argument is invalid.
    """
    mesh_path = Path(mesh_path)
    dest_dir = Path(dest_dir)
    if collision not in COLLISION_APPROXIMATIONS:
        raise SceneEditError(
            f"collision must be one of {', '.join(COLLISION_APPROXIMATIONS)}")
    if not (0.0 < float(scale) <= 1000.0):
        raise SceneEditError("scale must be greater than zero and at most 1000")

    meshes, rotate = prepare_mesh(mesh_path, scale=scale, up_axis=up_axis)

    usd_dir = dest_dir / "usd"
    material_dir = dest_dir / "material"
    usd_dir.mkdir(parents=True, exist_ok=True)
    usd_path = usd_dir / f"{asset_id}.usd"

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, Sdf.Path(f"/{asset_id}"))
    stage.SetDefaultPrim(root.GetPrim())
    looks = UsdGeom.Scope.Define(stage, root.GetPath().AppendChild("Looks")).GetPath()

    link = UsdGeom.Xform.Define(stage, root.GetPath().AppendChild("link_geometry_0"))
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    visuals = UsdGeom.Xform.Define(stage, link.GetPath().AppendChild("visuals"))
    collisions = UsdGeom.Xform.Define(stage, link.GetPath().AppendChild("collisions"))

    textured = False
    total_verts = total_faces = 0
    for index, name, mesh in meshes:
        image = _texture_image(mesh)
        texture_relative = None
        if image is not None:
            written = _write_texture(image, material_dir, f"material_{index}")
            # Relative to the USD, which sits one directory deeper.
            texture_relative = f"../material/{written.name}"
            textured = True
        material = _define_material(
            stage, looks, index, texture_relative, _base_color(mesh))

        uv = getattr(getattr(mesh, "visual", None), "uv", None)
        prim = _author_mesh(
            stage, visuals.GetPath().AppendChild(f"mesh_{index}"), mesh,
            uv=uv if texture_relative else None)
        # Apply the schema so consumers' ComputeBoundMaterial sees the binding.
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material)
        total_verts += len(mesh.vertices)
        total_faces += len(mesh.faces)

    combined = trimesh.util.concatenate([m for _, _, m in meshes])

    caveats = []
    if collision != "none":
        source = combined
        if collision == "convexHull":
            try:
                source = combined.convex_hull
            except Exception:  # noqa: BLE001 - degenerate input, fall back
                caveats.append("convex hull failed; the full mesh is the collider")
        collider = _author_mesh(
            stage, collisions.GetPath().AppendChild("collision_0"), source)
        # Guide purpose keeps colliders out of renders and the browser proxy.
        collider.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim())
        collider.GetPrim().CreateAttribute(
            "physics:approximation", Sdf.ValueTypeNames.Token).Set(collision)
    else:
        caveats.append("no collider authored; this object will not collide")

    resolved_mass = float(mass) if mass else _estimate_mass(combined, density)
    if not (MASS_BOUNDS[0] <= resolved_mass <= MASS_BOUNDS[1]):
        raise SceneEditError(
            f"mass must be between {MASS_BOUNDS[0]} and {MASS_BOUNDS[1]} kg")
    mass_api = UsdPhysics.MassAPI.Apply(link.GetPrim())
    mass_api.CreateMassAttr(resolved_mass)
    # Inertia is not authored; PhysX derives it from the collision geometry.

    stage.GetRootLayer().Save()

    size = [round(float(v), 4) for v in combined.extents]
    if max(size) > 5.0:
        caveats.append(
            f"bounding box is {max(size):.1f} m across — check the scale field "
            "if the source was authored in centimetres or millimetres")
    if collision == "convexHull":
        caveats.append(
            "convex-hull collider: a bowl or mug will behave as a solid lump. "
            "Re-import with convexDecomposition if it needs to be hollow")
    caveats.append(
        "collision is approximated here, not cooked by the pipeline's USD-import "
        "stage — settle the scene before relying on it")

    return usd_path, {
        "size": size,
        "mass": round(resolved_mass, 4),
        "rotated": bool(rotate),
        "textured": textured,
        "verts": int(total_verts),
        "faces": int(total_faces),
        "meshes": len(meshes),
        "collision": collision,
        "caveats": caveats,
    }


def main():
    """Convert or inspect one asset from the command line."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Import an asset into a scene bundle")
    parser.add_argument("source", help="USD or mesh file")
    parser.add_argument("--dest", required=True, help="Bundle directory to create")
    parser.add_argument("--id", default=None, help="Asset id (default: the file stem)")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--up-axis", default="auto", choices=("auto", "y", "z"))
    parser.add_argument("--collision", default="convexHull",
                        choices=COLLISION_APPROXIMATIONS)
    parser.add_argument("--mass", type=float, default=None)
    args = parser.parse_args()

    source = resolve_user_path(args.source)
    asset_id = slug(args.id or source.stem)
    kind = classify(source)
    if kind == "usd":
        out, notes = import_usd_file(source, Path(args.dest))
        print(json.dumps({"usd": str(out), "notes": notes}, indent=2))
    elif kind == "mesh":
        out, report = convert_mesh(
            source, Path(args.dest), asset_id, scale=args.scale,
            up_axis=args.up_axis, collision=args.collision, mass=args.mass)
        print(json.dumps({"usd": str(out), **report}, indent=2))
    else:
        sys.exit(f"ERROR: {source} is a directory, not an asset")


if __name__ == "__main__":
    main()
