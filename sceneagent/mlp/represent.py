"""Views, Gaussian ply, and prior_input.npy for the three checked meshes.

DINOv2 ViT-S/14 and OpenCLIP ViT-B-16 run on the rendered views. A Gaussian
keeps the patch token of each view that sees it. Views that do not see it
are skipped. A Gaussian with no finite feature is dropped. This module does
not fill a missing column and does not call Gemini.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sceneagent.mlp.contract import PriorContractError, pack_prior_input, save_prior_input
from PIL import Image

from sceneagent.mlp.splat_from_mesh import (
    CLIP_SIZE,
    VIEW_SIZE,
    _triangle_mesh_path,
    filter_visible,
    gaussians_from_mesh,
    load_textured_mesh,
    render_views,
    save_views,
    write_splat,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "data" / "mlp" / "samples_manifest.json"
DINO_NAME = "vit_small_patch14_dinov2.lvd142m"
CLIP_NAME = "ViT-B-16"
CLIP_PRETRAINED = "openai"


class FeatureEncoders:
    """Patch features for one rendered view. Models load on first use."""

    def __init__(self):
        self._dino = None
        self._clip = None
        self._dino_mean = None
        self._dino_std = None
        self._clip_mean = None
        self._clip_std = None
        self._torch = None

    def load(self):
        """Load both networks once. A worker process must not call this."""

        self._load_dino()
        self._load_clip()
        return self

    def patches(self, rgb):
        """Return ``(dino_grid, clip_grid)`` for one uint8 RGB view."""

        if rgb.shape != (VIEW_SIZE, VIEW_SIZE, 3):
            raise PriorContractError(f"view must be {VIEW_SIZE} square, got {rgb.shape}")
        dino = self._dino_patches(rgb)
        clip_rgb = np.asarray(
            Image.fromarray(rgb, mode="RGB").resize((CLIP_SIZE, CLIP_SIZE), Image.Resampling.BICUBIC),
            dtype=np.uint8,
        )
        clip = self._clip_patches(clip_rgb)
        if dino.shape[-1] != 384 or clip.shape[-1] != 512:
            raise PriorContractError(
                f"feature widths are dino {dino.shape[-1]} and clip {clip.shape[-1]}, not 384 and 512"
            )
        if dino.shape[0] * dino.shape[1] == 0 or clip.shape[0] * clip.shape[1] == 0:
            raise PriorContractError("feature grid is empty")
        if not np.isfinite(dino).all() or not np.isfinite(clip).all():
            raise PriorContractError("a view feature is not finite; missing values are not filled")
        if _all_zero(dino) or _all_zero(clip):
            raise PriorContractError("a view feature is all zeros; refusing to substitute zeros")
        return dino, clip

    def _dino_patches(self, rgb):
        model, mean, std = self._load_dino()
        torch = self._torch
        batch = _normalize(torch, rgb, mean, std)
        with torch.inference_mode():
            tokens = model.forward_features(batch)
        array = tokens.detach().float().cpu().numpy()
        if array.ndim != 3 or array.shape[-1] != 384:
            raise PriorContractError(f"DINOv2 ViT-S/14 tokens have shape {array.shape}, expected (*, *, 384)")
        prefix = int(getattr(model, "num_prefix_tokens", 1))
        patches = array[:, prefix:, :]
        grid = _square_grid(patches, VIEW_SIZE, 14, "DINOv2 ViT-S/14")
        return grid[0]

    def _clip_patches(self, rgb):
        model, mean, std = self._load_clip()
        torch = self._torch
        if rgb.shape != (CLIP_SIZE, CLIP_SIZE, 3):
            raise PriorContractError(f"OpenCLIP view must be {CLIP_SIZE} square, got {rgb.shape}")
        batch = _normalize(torch, rgb, mean, std)
        with torch.inference_mode():
            patches = _openclip_patch_tokens(model, batch)
        array = patches.detach().float().cpu().numpy()
        if array.ndim != 3 or array.shape[-1] != 512:
            raise PriorContractError(
                f"OpenCLIP ViT-B-16 tokens have shape {getattr(array, 'shape', None)}, expected (*, *, 512)"
            )
        grid = _square_grid(array, CLIP_SIZE, 16, "OpenCLIP ViT-B-16")
        return grid[0]

    def _load_dino(self):
        if self._dino is not None:
            return self._dino, self._dino_mean, self._dino_std
        torch, timm = _import_feature_stack()
        self._torch = torch
        model = timm.create_model(DINO_NAME, pretrained=True, num_classes=0)
        model.eval()
        config = timm.data.resolve_model_data_config(model)
        embed = int(getattr(model, "embed_dim", 0) or getattr(model, "num_features", 0))
        patch = int(getattr(getattr(model, "patch_embed", None), "patch_size", (0,))[0])
        if embed != 384 or patch != 14:
            raise PriorContractError(
                f"loaded {DINO_NAME} has embed {embed} and patch {patch}, not ViT-S/14"
            )
        self._dino = model
        self._dino_mean = np.asarray(config["mean"], dtype=np.float32)
        self._dino_std = np.asarray(config["std"], dtype=np.float32)
        return self._dino, self._dino_mean, self._dino_std

    def _load_clip(self):
        if self._clip is not None:
            return self._clip, self._clip_mean, self._clip_std
        torch, _timm = _import_feature_stack()
        self._torch = torch
        import open_clip

        created = open_clip.create_model_and_transforms(
            CLIP_NAME,
            pretrained=CLIP_PRETRAINED,
            force_quick_gelu=True,
        )
        model = created[0]
        preprocess = created[-1]
        model.eval()
        if int(getattr(model.visual, "output_dim", 512)) != 512:
            raise PriorContractError("OpenCLIP visual output is not 512")
        mean, std = _preprocess_norm(preprocess)
        self._clip = model
        self._clip_mean = mean
        self._clip_std = std
        return self._clip, self._clip_mean, self._clip_std


def render_object(sample):
    """Rasterize one object and save its views. Does not load feature models."""

    root = _triangle_mesh_path(sample["mesh_path"]).resolve()
    object_dir = _object_dir(sample)
    print(f"object {sample['id']}", flush=True)
    mesh, image, uv = load_textured_mesh(sample["mesh_path"], sample.get("image_path"))
    gaussians = gaussians_from_mesh(mesh, image, uv)
    print(
        f"geometry kept {len(gaussians['xyz'])} dropped {gaussians['dropped_geometry']}",
        flush=True,
    )
    views = render_views(mesh, image, uv)
    view_paths = save_views(views, object_dir / "views")
    del mesh, image, uv
    return {
        "mesh_path": str(root),
        "view_paths": {name: str(path) for name, path in view_paths.items()},
        "gaussians": gaussians,
        "views": views,
    }


def finish_object(sample, rendered, encoders):
    """Run the one loaded feature stack and write ply plus prior_input.npy."""

    object_dir = _object_dir(sample)
    gaussians = rendered["gaussians"]
    views = rendered["views"]
    print("features", flush=True)
    features = {name: encoders.patches(camera["rgb"]) for name, camera in views.items()}
    for camera in views.values():
        camera.pop("rgb", None)
    kept = filter_visible(gaussians, views, features)
    ply_path = object_dir / "splat.ply"
    npy_path = object_dir / "prior_input.npy"
    try:
        geometry = write_splat(ply_path, kept)
        if np.any(np.linalg.norm(kept["dino"], axis=1) <= 0.0) or np.any(np.linalg.norm(kept["clip"], axis=1) <= 0.0):
            raise PriorContractError("refusing to pack an all-zero feature row")
        packed = pack_prior_input(kept["dino"], kept["clip"], geometry.rows)
        save_prior_input(npy_path, packed)
        loaded = np.load(npy_path)
        if loaded.shape != packed.shape or loaded.shape[1] != 912:
            raise PriorContractError(f"saved prior_input has shape {loaded.shape}")
    except Exception:
        ply_path.unlink(missing_ok=True)
        npy_path.unlink(missing_ok=True)
        raise
    view_paths = rendered["view_paths"]
    report = {
        "id": sample["id"],
        "source": sample["source"],
        "name": sample["name"],
        "view_front": str(view_paths["front"]),
        "view_side": str(view_paths["side"]),
        "view_top": str(view_paths["top"]),
        "ply": str(ply_path),
        "prior_input": str(npy_path),
        "prior_input_shape": [int(loaded.shape[0]), int(loaded.shape[1])],
        "gaussians_kept": int(kept["kept"]),
        "gaussians_dropped": int(gaussians["triangles_in"] - kept["kept"]),
        "dropped_geometry": int(gaussians["dropped_geometry"]),
        "dropped_unseen_or_feature": int(kept["dropped_unseen"]),
        "mesh_path": str(rendered["mesh_path"]),
        "dino": DINO_NAME,
        "clip": f"{CLIP_NAME} {CLIP_PRETRAINED}",
        "view_size": VIEW_SIZE,
        "clip_size": CLIP_SIZE,
    }
    (object_dir / "representation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def represent_sample(sample, encoders):
    """Render one manifest object and write its ply and prior_input.npy."""

    return finish_object(sample, render_object(sample), encoders)


def represent_manifest(manifest_path=MANIFEST):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if [row["id"] for row in samples] != [
        "002_master_chef_can",
        "Weisshai_Great_White_Shark",
        "B079TXJCR5",
    ]:
        raise PriorContractError("manifest is not the three checked meshes")
    encoders = FeatureEncoders()
    return [represent_sample(sample, encoders) for sample in samples]


def _object_dir(sample):
    mesh = Path(sample["mesh_path"])
    if sample["source"] == "ycb":
        # .../id/google_16k/textured.obj
        return mesh.parents[1]
    if sample["source"] == "gso":
        # .../id/model.sdf, or .../id/meshes/model.obj
        if mesh.suffix.lower() == ".sdf":
            return mesh.parent
        return mesh.parents[1]
    if sample["source"] == "abo":
        return mesh.parent
    raise PriorContractError(f"unknown source {sample['source']}")


def _import_feature_stack():
    try:
        import torch
        import timm
    except ImportError as exc:
        raise PriorContractError(
            "DINOv2 and OpenCLIP are not installed on this machine; "
            "feature columns are not filled"
        ) from exc
    return torch, timm


def _normalize(torch, rgb, mean, std):
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    array = (array - mean) / std
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _square_grid(patches, image_size, patch_size, name):
    count = int(patches.shape[1])
    side = int(round(count ** 0.5))
    if side * side != count or image_size % patch_size != 0 or side != image_size // patch_size:
        raise PriorContractError(
            f"{name} patch count {count} does not match a {image_size}/{patch_size} grid"
        )
    return patches.reshape(patches.shape[0], side, side, patches.shape[-1])


def _preprocess_norm(preprocess):
    for step in getattr(preprocess, "transforms", []):
        mean = getattr(step, "mean", None)
        std = getattr(step, "std", None)
        if mean is not None and std is not None:
            return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    raise PriorContractError("OpenCLIP preprocess does not publish a mean and std")


def _openclip_patch_tokens(model, batch):
    visual = model.visual
    if hasattr(visual, "output_tokens"):
        visual.output_tokens = True
        result = visual(batch)
        if isinstance(result, tuple) and len(result) == 2 and result[1] is not None:
            _pooled, tokens = result
            return _drop_prefix_and_project(visual, tokens)
    return _manual_openclip_tokens(visual, batch)


def _drop_prefix_and_project(visual, tokens):
    """Keep patch tokens in the 512-d projected space. Do not slice a 768-d vector."""

    projected = tokens
    if tokens.shape[-1] != 512:
        proj = getattr(visual, "proj", None)
        if proj is None or tokens.shape[-1] != proj.shape[0] or proj.shape[-1] != 512:
            raise PriorContractError(
                f"OpenCLIP patch width {tuple(tokens.shape)} has no 512-d projection"
            )
        projected = tokens @ proj
    if projected.shape[-1] != 512:
        raise PriorContractError(f"OpenCLIP projected width is {projected.shape[-1]}")
    patch_count = (CLIP_SIZE // 16) ** 2
    prefix = 1 if projected.shape[1] == patch_count + 1 else 0
    if projected.shape[1] - prefix != patch_count:
        raise PriorContractError(f"OpenCLIP token count {projected.shape[1]} is not a 14 by 14 grid")
    return projected[:, prefix:, :]


def _manual_openclip_tokens(visual, batch):
    import torch

    if not hasattr(visual, "conv1") or not hasattr(visual, "transformer"):
        raise PriorContractError("this OpenCLIP visual has no patch transformer to read")
    x = visual.conv1(batch)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    class_embed = visual.class_embedding.to(x.dtype)
    cls = class_embed + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    x = visual.ln_post(x)
    return _drop_prefix_and_project(visual, x)


def _all_zero(array):
    return bool(np.all(array == 0.0))


def main():
    reports = represent_manifest()
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
