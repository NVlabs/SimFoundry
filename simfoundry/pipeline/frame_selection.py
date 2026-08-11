# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical-frame selection for the A_reconstruction pipeline.

Stages 3-13 all reconstruct the scene from a *single* frame of the input capture: stage 3
fits the support plane in it, stage 4 turns that frame's camera into the world frame, and
stage 5 crops every object out of it, which is what stages 6-8 turn into meshes and poses.
That frame used to be hard-coded (`s3_ground.img_idx: 0`), so a first frame that happened to
be blurry, shot from far away, or full of mutually-occluding objects silently capped the
quality of everything downstream -- e.g. a whiteboard marker only ~470 px in frame 0 of the
PutMarkerInCup capture, versus ~1700 px a few frames later.

Setting `s3_ground.img_idx: auto` makes stage 3 score every candidate frame and pick one.
Scoring runs on stage-2 depth output, so it needs no extra models beyond the SAM3 instance
stage 3 already builds. Per candidate we fit the support plane exactly the way stage 3 does,
then measure:

- `object_coverage`: image fraction taken up by geometry sitting on the support plane. This is
  the direct proxy for "the objects are big in this frame".
- `object_separation`: how many distinct blobs that geometry splits into. Objects that occlude
  or touch each other from a given viewpoint merge into one blob, so more blobs means a
  viewpoint that separates the scene better -- which is what stage 5 needs to crop them apart.
- `sharpness`: variance of the Laplacian inside the object region of the *full-resolution*
  source frame, which is the image stage 5 actually crops from.
- `support_coverage` / `plane_quality`: how much of the frame the support surface fills and how
  well it fits a plane, both of which stages 3-4 depend on.
- `clipped_frac`: penalty for object mass running off the edge of the frame.

Frames that fail a hard gate (no support mask, camera rolled past `floor_tilt_threshold`, bad
plane fit, no objects found) are dropped before ranking, which also stops stage 3 from
asserting its way out of a run that a different frame would have handled fine.

With `mode: hybrid` (the default) the heuristic short-lists the top few frames and a VLM makes
the final call, since "is this object occluded" reads much better to a VLM than to a blob
counter. The VLM step is fail-soft: no credentials or a failed call just keeps the heuristic
winner.

The chosen index is written to `<s3_ground.out_dir>/frame_selection.json`; every downstream
stage reads it back through `resolve_img_idx` so the whole pipeline stays on one frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import glob
import logging
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

SELECTION_FILENAME = "frame_selection.json"
SELECTION_SHEET_FILENAME = "frame_selection.png"

#: `img_idx` values that mean "let stage 3 choose" rather than naming a frame.
AUTO_TOKENS = {"auto", "automatic", "best", "none", "null", ""}

SELECTION_MODES = {"heuristic", "vlm", "hybrid"}

DEFAULT_SELECTION_CFG = {
    "mode": "hybrid",
    "max_candidates": 24,
    "min_object_height": 0.015,
    "max_object_height": 0.6,
    "min_component_px": 50,
    "max_tilt": None,             # None -> inherit s3_ground.floor_tilt_threshold
    "min_plane_inlier_ratio": 0.6,
    "min_support_coverage": 0.05,
    "max_clipped_frac": 0.5,
    "clipped_penalty": 0.35,
    "vlm_top_k": 4,
    "vlm_max_side": 1024,
    "vlm_model": None,            # None -> inherit s3_ground.detection_model
    "write_debug_sheet": True,
    "weights": {
        "object_coverage": 0.40,
        "object_separation": 0.20,
        "sharpness": 0.20,
        "support_coverage": 0.10,
        "plane_quality": 0.10,
    },
}


class FrameSelectionError(RuntimeError):
    """Raised when no candidate frame is usable, or a persisted selection is missing."""


@dataclass
class FrameScore:
    """Per-candidate metrics and the score they combine into."""

    idx: int
    eligible: bool = True
    reject_reason: str | None = None
    floor_category: str | None = None
    floor_logit: float = 0.0
    tilt: float = float("nan")
    plane_inlier_ratio: float = 0.0
    support_coverage: float = 0.0
    object_coverage: float = 0.0
    smallest_object: float = 0.0
    n_objects: int = 0
    clipped_frac: float = 0.0
    sharpness: float = 0.0
    score: float = 0.0
    terms: dict[str, float] = field(default_factory=dict)


@dataclass
class FrameSelection:
    """Outcome of a selection run, mirrored into `frame_selection.json`."""

    selected_idx: int
    mode: str
    decided_by: str
    n_candidates: int
    scores: list[FrameScore]
    vlm_shortlist: list[int] = field(default_factory=list)
    vlm_note: str | None = None

    def to_payload(self) -> dict[str, Any]:
        def sanitize(value):
            # Frames rejected before the plane fit carry NaN metrics, which is not valid JSON.
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        return {
            "selected_idx": int(self.selected_idx),
            "mode": self.mode,
            "decided_by": self.decided_by,
            "n_candidates": int(self.n_candidates),
            "vlm_shortlist": [int(i) for i in self.vlm_shortlist],
            "vlm_note": self.vlm_note,
            "scores": [{k: sanitize(v) for k, v in asdict(s).items()} for s in self.scores],
        }


# --------------------------------------------------------------------------------------
# Config / index resolution
# --------------------------------------------------------------------------------------

def is_auto_img_idx(value: Any) -> bool:
    """True when `img_idx` asks stage 3 to choose the frame instead of naming one."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in AUTO_TOKENS
    return False


def selection_cfg(cfg) -> dict[str, Any]:
    """Merge `s3_ground.frame_selection` over the defaults, so older configs still load."""
    merged = OmegaConf.create(DEFAULT_SELECTION_CFG)
    user = OmegaConf.select(cfg, "s3_ground.frame_selection")
    if user is not None:
        # Resolve against the full config first; the merged node is detached from its parent.
        merged = OmegaConf.merge(merged, OmegaConf.to_container(user, resolve=True))
    out = OmegaConf.to_object(merged)

    mode = str(out["mode"]).lower()
    if mode not in SELECTION_MODES:
        raise ValueError(f"Unknown frame_selection.mode '{mode}'. Valid modes: {sorted(SELECTION_MODES)}")
    out["mode"] = mode

    if out["max_tilt"] is None:
        out["max_tilt"] = float(OmegaConf.select(cfg, "s3_ground.floor_tilt_threshold") or 0.125)
    if out["vlm_model"] is None:
        out["vlm_model"] = OmegaConf.select(cfg, "s3_ground.detection_model")
    return out


def selection_fpath(cfg) -> str:
    return f"{cfg.s3_ground.out_dir}/{SELECTION_FILENAME}"


def load_selection(cfg) -> dict[str, Any] | None:
    """Read back the frame stage 3 committed to, or None if it has not run in auto mode."""
    import json

    fpath = selection_fpath(cfg)
    if not os.path.isfile(fpath):
        return None
    with open(fpath, "r") as f:
        return json.load(f)


def write_selection(cfg, selection: FrameSelection) -> str:
    import json

    fpath = selection_fpath(cfg)
    Path(fpath).parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, "w") as f:
        json.dump(selection.to_payload(), f, indent=4)
    return fpath


def resolve_img_idx(cfg, stage_key: str = "s3_ground") -> int:
    """Return the frame index this run is built on.

    An explicit integer in `<stage_key>.img_idx` always wins, so pinning a frame keeps
    working. `auto`/null defers to the index stage 3 recorded in `frame_selection.json`.
    """
    value = OmegaConf.select(cfg, f"{stage_key}.img_idx")
    if not is_auto_img_idx(value):
        return int(value)

    selection = load_selection(cfg)
    if selection is None:
        raise FrameSelectionError(
            f"{stage_key}.img_idx is '{value}' (automatic) but no frame selection was found at "
            f"{selection_fpath(cfg)}. Run stage 3 (3_segment_ground_plane.py) first, or pin a "
            f"frame with {stage_key}.img_idx=<int>."
        )
    idx = int(selection["selected_idx"])
    logger.info("Using auto-selected canonical frame %s (from %s)", idx, selection_fpath(cfg))
    return idx


# --------------------------------------------------------------------------------------
# Candidate frame loading
# --------------------------------------------------------------------------------------

@dataclass
class FrameBundle:
    """Per-frame depth-stage output plus the full-resolution source frame it came from."""

    rgbs: Sequence[np.ndarray]
    depths: Sequence[np.ndarray]
    intrinsics: Sequence[np.ndarray]
    source_fpaths: list[str | None]
    #: `img_idx` value each position corresponds to. These match positions for the DA3 backend
    #: (one npz row per frame) but not necessarily for per-file backends, where the index is
    #: baked into the filename downstream stages reload.
    frame_ids: list[int] | None = None

    def __post_init__(self):
        if self.frame_ids is None:
            self.frame_ids = list(range(len(self.depths)))

    def __len__(self) -> int:
        return len(self.depths)

    def position_of(self, frame_id: int) -> int:
        return self.frame_ids.index(int(frame_id))


def _source_frame_fpaths(cfg) -> list[str]:
    """Full-resolution frames the depth stage consumed, in the same order it consumed them."""
    frames_dir = OmegaConf.select(cfg, "s2_da.frames_dir")
    if not frames_dir:
        n = OmegaConf.select(cfg, "s1_video.n_subsampled_frames")
        frames_dir = f"{cfg.s1_video.out_dir}/frames_subsampled_{n}"
    if not os.path.isdir(frames_dir):
        return []
    return sorted(str(p) for p in glob.glob(f"{frames_dir}/*.png"))


def load_frame_bundle(cfg) -> FrameBundle:
    """Load every candidate frame from whichever stage-2 backend produced the depth."""
    use_fs = bool(OmegaConf.select(cfg, "s3_ground.use_fs") or False)

    if use_fs:
        # Per-file backend: the index lives in the filename, so carry it explicitly rather than
        # assuming the frames are numbered 0..N-1 with no gaps.
        fs_dir = cfg.s2_fs.out_dir
        stems = {}
        for path in glob.glob(f"{fs_dir}/*_rgb.npy"):
            prefix, _, index = Path(path).name.removesuffix("_rgb.npy").rpartition("_")
            if index.isdigit():
                stems[int(index)] = f"{prefix}_{index}"
        if not stems:
            raise FrameSelectionError(f"No FoundationStereo frames found under {fs_dir}")
        ids = sorted(stems)
        rgbs = [np.load(f"{fs_dir}/{stems[i]}_rgb.npy") for i in ids]
        depths = [np.load(f"{fs_dir}/{stems[i]}_depth_meter.npy") for i in ids]
        Ks = [np.load(f"{fs_dir}/{stems[i]}_K.npy") for i in ids]
        # Stereo captures carry no separate full-res frame set; score the rectified RGB.
        return FrameBundle(rgbs, depths, Ks, [None] * len(ids), frame_ids=ids)

    results = np.load(f"{cfg.s2_da.out_dir}/da/exports/npz/results.npz")
    n = results["depth"].shape[0]
    sources = _source_frame_fpaths(cfg)
    if len(sources) != n:
        # Any mismatch means we cannot trust the pairing, so fall back to the depth-stage RGB.
        if sources:
            logger.warning(
                "Found %s source frames but %s depth frames; scoring sharpness on the "
                "depth-stage RGB instead.", len(sources), n,
            )
        sources = [None] * n
    return FrameBundle(results["image"], results["depth"], results["intrinsics"], list(sources))


# --------------------------------------------------------------------------------------
# Per-frame scoring
# --------------------------------------------------------------------------------------

def predict_support_mask(rgb: np.ndarray, sam3, floor_categories: Sequence[str], floor_threshold: float):
    """First floor category SAM3 finds above threshold, as a single (H, W) boolean mask."""
    from PIL import Image

    pil_img = Image.fromarray(rgb)
    for category in floor_categories:
        masks, _boxes, logits = sam3.predict_segmentation(pil_img=pil_img, text_prompt=category)
        if len(masks) == 0 or logits.max() < floor_threshold:
            continue
        if len(masks) > 1:
            # Same tie-break as stage 3: the largest mask is the support surface.
            areas = masks.reshape(len(masks), -1).sum(axis=1)
            masks = masks[[int(np.argmax(areas))]]
            logits = logits[[int(np.argmax(areas))]]
        return masks[0][0].astype(bool), category, float(logits.max())
    return None, None, 0.0


def _fit_support_plane(points: np.ndarray):
    """RANSAC plane fit with the normal flipped to point up out of the surface."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # More iterations than stage 3 uses for its single fit: here the fits are compared against
    # each other, so RANSAC scatter shows up directly as noise in the ranking.
    (a, b, c, d), inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=2000)
    normal = np.array([a, b, c], dtype=np.float64)
    # Camera Z points into the scene and towards the surface, so an upward normal has a
    # negative dot product with it (same convention as stage 3).
    if float(normal @ np.array([0.0, 0.0, 1.0])) > 0.0:
        normal, d = -normal, -d
    return normal, float(d), np.asarray(inliers, dtype=int)


def _object_mask_above_plane(points_flat, normal, d, shape, valid, sel_cfg, plane_points):
    """Pixels holding geometry that rests on the support plane.

    Restricted to the plane's own footprint (the convex hull of its inliers, taken in plane
    coordinates) so background walls, chairs and floor never count as tabletop objects.
    """
    from scipy.spatial import ConvexHull, Delaunay, QhullError

    heights = points_flat @ normal + d

    e1 = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    norm_e1 = np.linalg.norm(e1)
    if norm_e1 < 1e-8:
        e1 = np.cross(normal, np.array([0.0, 1.0, 0.0]))
        norm_e1 = np.linalg.norm(e1)
    e1 = e1 / norm_e1
    e2 = np.cross(normal, e1)

    hull_uv = np.stack([plane_points @ e1, plane_points @ e2], axis=-1)
    try:
        hull = Delaunay(hull_uv[ConvexHull(hull_uv).vertices])
    except (QhullError, ValueError):
        return None
    all_uv = np.stack([points_flat @ e1, points_flat @ e2], axis=-1)
    inside_footprint = hull.find_simplex(all_uv) >= 0

    above = (heights > sel_cfg["min_object_height"]) & (heights < sel_cfg["max_object_height"])
    return (valid & inside_footprint & above).reshape(shape)


def _sharpness(source_fpath: str | None, rgb: np.ndarray, object_mask: np.ndarray) -> float:
    """Laplacian variance around the objects, measured on the image stage 5 will crop from."""
    import cv2

    if source_fpath is not None and os.path.isfile(source_fpath):
        gray = cv2.cvtColor(cv2.imread(source_fpath), cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    region = cv2.resize(object_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    # Dilate so object silhouettes -- the edges where blur shows up most clearly -- are included.
    k = max(3, (min(h, w) // 60) | 1)
    region = cv2.dilate(region, np.ones((k, k), np.uint8)).astype(bool)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    if region.sum() < 100:
        return float(laplacian.var())
    return float(laplacian[region].var())


def score_frame(
    position: int,
    bundle: FrameBundle,
    sam3,
    cfg,
    sel_cfg: dict[str, Any],
) -> FrameScore:
    """Measure the candidate at `position`. One that cannot be measured comes back ineligible.

    `FrameScore.idx` carries the bundle's frame id -- the value downstream stages use as
    `img_idx` -- which is not always the position within the bundle.
    """
    import cv2

    from simfoundry.utils.processing_utils import compute_point_cloud_from_depth

    score = FrameScore(idx=bundle.frame_ids[position])
    rgb = np.asarray(bundle.rgbs[position])
    # Negative depth is physically meaningless and trips an assert downstream; clamping drops
    # those pixels through the validity mask instead of failing the whole frame.
    depth = np.clip(np.asarray(bundle.depths[position]).astype(np.float64), 0.0, None)
    K = np.asarray(bundle.intrinsics[position]).astype(np.float64)

    mask, category, logit = predict_support_mask(
        rgb, sam3, cfg.s3_ground.floor_categories, cfg.s3_ground.floor_threshold,
    )
    if mask is None:
        score.eligible = False
        score.reject_reason = "no support surface detected"
        return score
    score.floor_category = category
    score.floor_logit = logit
    score.support_coverage = float(mask.mean())

    points = compute_point_cloud_from_depth(depth=depth, K=K).reshape(-1, 3)
    support_points = points[mask.reshape(-1)]
    if len(support_points) < 100:
        score.eligible = False
        score.reject_reason = "support mask too small to fit a plane"
        return score

    try:
        normal, d, inliers = _fit_support_plane(support_points)
    except Exception as exc:  # open3d raises a bare RuntimeError on degenerate input
        score.eligible = False
        score.reject_reason = f"plane fit failed: {exc}"
        return score

    score.tilt = abs(float(normal[0]))
    score.plane_inlier_ratio = len(inliers) / max(1, len(support_points))

    z_far = float(OmegaConf.select(cfg, "s4_frame.z_far") or 5.0)
    valid = (points[:, 2] > 0.05) & (points[:, 2] < z_far)
    object_mask = _object_mask_above_plane(
        points, normal, d, depth.shape, valid, sel_cfg, support_points[inliers],
    )
    if object_mask is None:
        score.eligible = False
        score.reject_reason = "degenerate support footprint"
        return score

    # Open first: depth noise along the surface edges leaves slivers that would otherwise read
    # as extra objects.
    object_u8 = cv2.morphologyEx(object_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(object_u8, 8)
    kept = [i for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= sel_cfg["min_component_px"]]
    areas = sorted(int(stats[i, cv2.CC_STAT_AREA]) for i in kept)

    n_pixels = float(object_u8.size)
    score.n_objects = len(areas)
    score.object_coverage = float(sum(areas) / n_pixels)
    # Recorded for diagnostics only: objects that occlude each other merge into one blob, so
    # this over-reports when the viewpoint is bad -- exactly when it would matter most.
    score.smallest_object = float(areas[0] / n_pixels) if areas else 0.0

    # Object mass that runs off the edge of the frame. Measured per component rather than by
    # counting edge pixels, because an object half out of frame contributes very few pixels to
    # the boundary ring while being entirely unusable to stage 5.
    edge_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    clipped_area = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in kept if i in edge_labels)
    score.clipped_frac = clipped_area / max(1, sum(areas))

    score.sharpness = _sharpness(bundle.source_fpaths[position], rgb, object_u8.astype(bool))

    if score.tilt > sel_cfg["max_tilt"]:
        score.eligible = False
        score.reject_reason = f"roll {score.tilt:.3f} > max_tilt {sel_cfg['max_tilt']:.3f}"
    elif score.plane_inlier_ratio < sel_cfg["min_plane_inlier_ratio"]:
        score.eligible = False
        score.reject_reason = f"plane inliers {score.plane_inlier_ratio:.2f} too low"
    elif score.support_coverage < sel_cfg["min_support_coverage"]:
        score.eligible = False
        score.reject_reason = f"support fills only {score.support_coverage:.1%} of frame"
    elif score.n_objects == 0:
        score.eligible = False
        score.reject_reason = "no objects on the support surface"
    elif score.clipped_frac > sel_cfg["max_clipped_frac"]:
        score.eligible = False
        score.reject_reason = f"{score.clipped_frac:.0%} of object area runs off the frame"
    return score


def _normalize(values: Sequence[float]) -> list[float]:
    """Min-max to [0, 1]; an all-equal metric contributes nothing to the ranking."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def rank_frames(scores: Sequence[FrameScore], sel_cfg: dict[str, Any]) -> list[FrameScore]:
    """Assign each eligible frame a combined score, best first.

    Metrics are normalized across the eligible candidates rather than against absolute
    thresholds, because what counts as "close enough" or "sharp enough" depends entirely on
    how the capture was shot.
    """
    eligible = [s for s in scores if s.eligible]
    if not eligible:
        return []

    weights = sel_cfg["weights"]
    metrics = {
        "object_coverage": _normalize([s.object_coverage for s in eligible]),
        "object_separation": _normalize([float(s.n_objects) for s in eligible]),
        "sharpness": _normalize([s.sharpness for s in eligible]),
        "support_coverage": _normalize([s.support_coverage for s in eligible]),
        "plane_quality": _normalize([s.plane_inlier_ratio for s in eligible]),
    }
    for i, s in enumerate(eligible):
        s.terms = {name: round(float(values[i]), 4) for name, values in metrics.items()}
        total = sum(weights.get(name, 0.0) * values[i] for name, values in metrics.items())
        total -= sel_cfg["clipped_penalty"] * s.clipped_frac
        s.score = float(total)
    return sorted(eligible, key=lambda s: s.score, reverse=True)


# --------------------------------------------------------------------------------------
# VLM refinement
# --------------------------------------------------------------------------------------

def _write_vlm_candidate_images(bundle: FrameBundle, shortlist: Sequence[int], out_dir: str, max_side: int) -> list[str]:
    """Write one downscaled, corner-labelled image per shortlisted frame."""
    from PIL import Image, ImageDraw

    tmp_dir = Path(out_dir) / "frame_selection_candidates"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fpaths = []
    for rank, frame_id in enumerate(shortlist, start=1):
        position = bundle.position_of(frame_id)
        source = bundle.source_fpaths[position]
        img = Image.open(source).convert("RGB") if source else Image.fromarray(np.asarray(bundle.rgbs[position]))
        scale = max_side / max(img.size)
        if scale < 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        draw = ImageDraw.Draw(img)
        label = f"OPTION {rank}"
        draw.rectangle([0, 0, 190, 44], fill=(0, 0, 0))
        draw.text((12, 14), label, fill=(0, 255, 0))
        fpath = str(tmp_dir / f"option_{rank}_frame_{frame_id}.png")
        img.save(fpath)
        fpaths.append(fpath)
    return fpaths


def refine_with_vlm(cfg, bundle: FrameBundle, shortlist: Sequence[int], sel_cfg: dict[str, Any]) -> tuple[int | None, str]:
    """Ask a VLM which shortlisted frame reconstructs best. Returns (frame index, note).

    Fail-soft by design: the heuristic ranking is already a usable answer, so a missing
    credential or a flaky remote call must not take down stage 3.
    """
    from simfoundry.models.vlm import Gemini
    from simfoundry.utils.prompt_utils import prompt_canonical_frame_select

    if len(shortlist) < 2:
        return None, "shortlist too short to need a VLM"

    try:
        image_fpaths = _write_vlm_candidate_images(
            bundle, shortlist, cfg.s3_ground.out_dir, int(sel_cfg["vlm_max_side"]),
        )
        vlm = Gemini(project=cfg.gcloud_project, location="global", model=sel_cfg["vlm_model"])
        result = vlm(
            prompt=prompt_canonical_frame_select(len(shortlist)),
            image_paths=image_fpaths,
            temperature=0,
            top_p=0,
            seed=0,
            print_results=bool(OmegaConf.select(cfg, "visualize")),
        )
        text = vlm.get_result_text(result=result)
    except Exception as exc:
        logger.warning("VLM frame selection failed (%s); keeping the heuristic winner.", exc)
        return None, f"vlm call failed: {exc}"

    answer = text.rsplit("ANSWER:", 1)[-1] if "ANSWER:" in text else ""
    digits = "".join(ch for ch in answer if ch.isdigit() or ch == " ").split()
    if not digits:
        logger.warning("Could not parse a VLM frame choice from: %r", text[-200:])
        return None, "vlm response had no parseable ANSWER"
    option = int(digits[0])
    if not 1 <= option <= len(shortlist):
        logger.warning("VLM picked option %s, outside 1..%s; keeping the heuristic winner.", option, len(shortlist))
        return None, f"vlm option {option} out of range"
    return int(shortlist[option - 1]), f"vlm picked option {option}"


# --------------------------------------------------------------------------------------
# Debug sheet
# --------------------------------------------------------------------------------------

def write_debug_sheet(bundle: FrameBundle, scores: Sequence[FrameScore], selected_idx: int, out_dir: str) -> str | None:
    """Contact sheet of every candidate with its score, so a bad pick is obvious at a glance."""
    from PIL import Image, ImageDraw

    try:
        n = len(scores)
        cols = min(5, n)
        rows = (n + cols - 1) // cols
        tile_w, tile_h = 320, 200
        sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + 34)), (16, 16, 16))
        draw = ImageDraw.Draw(sheet)
        for i, s in enumerate(scores):
            position = bundle.position_of(s.idx)
            source = bundle.source_fpaths[position]
            img = Image.open(source).convert("RGB") if source else Image.fromarray(np.asarray(bundle.rgbs[position]))
            img = img.resize((tile_w, tile_h), Image.LANCZOS)
            x, y = (i % cols) * tile_w, (i // cols) * (tile_h + 34)
            sheet.paste(img, (x, y))
            if s.idx == selected_idx:
                draw.rectangle([x, y, x + tile_w - 1, y + tile_h - 1], outline=(0, 255, 0), width=5)
            if s.eligible:
                caption = f"idx {s.idx}  score {s.score:.3f}  obj {s.object_coverage:.3%}  sharp {s.sharpness:.0f}"
                color = (0, 255, 0) if s.idx == selected_idx else (220, 220, 220)
            else:
                caption = f"idx {s.idx}  REJECTED: {s.reject_reason}"
                color = (255, 120, 120)
            # Truncate rather than let a long reason bleed into the neighbouring tile.
            draw.text((x + 6, y + tile_h + 10), caption[:52], fill=color)
        fpath = f"{out_dir}/{SELECTION_SHEET_FILENAME}"
        sheet.save(fpath)
        return fpath
    except Exception as exc:
        logger.warning("Could not write the frame-selection debug sheet: %s", exc)
        return None


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def select_canonical_frame(
    cfg,
    sam3,
    bundle: FrameBundle | None = None,
    refine_fn: Callable[..., tuple[int | None, str]] = refine_with_vlm,
) -> FrameSelection:
    """Score the candidate frames and choose the one the reconstruction is built from."""
    import open3d as o3d

    sel_cfg = selection_cfg(cfg)
    bundle = bundle if bundle is not None else load_frame_bundle(cfg)

    # The plane fits are RANSAC, so without seeding the same capture can pick different frames
    # on different runs -- and the whole reconstruction hangs off that choice. Open3D fits in
    # parallel, so this narrows the run-to-run scatter rather than eliminating it.
    o3d.utility.random.seed(int(OmegaConf.select(cfg, "seed") or 0))

    n_frames = len(bundle)
    if n_frames == 0:
        raise FrameSelectionError("Stage 2 produced no frames to select from.")

    # Cap the SAM3 calls on long captures by sampling evenly across the whole sequence.
    max_candidates = int(sel_cfg["max_candidates"])
    stride = max(1, -(-n_frames // max_candidates))
    candidates = list(range(0, n_frames, stride))[:max_candidates]

    logger.info("Scoring %s of %s frames for the canonical reconstruction frame...", len(candidates), n_frames)

    def score_one(position: int) -> FrameScore:
        try:
            return score_frame(position, bundle, sam3, cfg, sel_cfg)
        except Exception as exc:
            # One unscoreable frame should cost us that frame, not the run.
            frame_id = bundle.frame_ids[position]
            logger.warning("Could not score frame %s: %s", frame_id, exc)
            return FrameScore(idx=frame_id, eligible=False, reject_reason=f"scoring failed: {exc}")

    scores = [score_one(i) for i in candidates]
    for s in scores:
        if not s.eligible:
            logger.info("  frame %-3s rejected: %s", s.idx, s.reject_reason)

    ranked = rank_frames(scores, sel_cfg)
    if not ranked:
        reasons = ", ".join(f"{s.idx}: {s.reject_reason}" for s in scores)
        raise FrameSelectionError(
            "No frame passed the selection gates. Loosen s3_ground.frame_selection (e.g. raise "
            f"floor_tilt_threshold) or pin s3_ground.img_idx to an integer. Reasons -- {reasons}"
        )

    for s in ranked:
        logger.info(
            "  frame %-3s score %.3f  objects %s (coverage %.2f%%)  sharpness %.0f  tilt %.3f",
            s.idx, s.score, s.n_objects, 100 * s.object_coverage, s.sharpness, s.tilt,
        )

    # The tilt gate protects stage 3's floor_tilt_threshold assertion, but a capture shot with a
    # rolled camera can have all its best-framed views on the wrong side of it. Say so rather
    # than silently settling for a more distant frame.
    rolled_out = [
        s for s in scores
        if not s.eligible and s.reject_reason and s.reject_reason.startswith("roll ")
        and s.object_coverage > ranked[0].object_coverage
    ]
    if rolled_out:
        logger.warning(
            "Frames %s show the objects larger than the chosen frame but were rejected for camera "
            "roll. Raise s3_ground.floor_tilt_threshold (or frame_selection.max_tilt) to allow them.",
            [s.idx for s in rolled_out],
        )

    mode = sel_cfg["mode"]
    selected_idx = ranked[0].idx
    decided_by = "heuristic"
    shortlist: list[int] = []
    note = None

    if mode in ("vlm", "hybrid"):
        top_k = len(ranked) if mode == "vlm" else int(sel_cfg["vlm_top_k"])
        shortlist = [s.idx for s in ranked[:max(1, top_k)]]
        vlm_idx, note = refine_fn(cfg, bundle, shortlist, sel_cfg)
        if vlm_idx is not None:
            selected_idx = vlm_idx
            decided_by = "vlm"

    logger.info("Selected canonical frame %s (mode=%s, decided_by=%s)", selected_idx, mode, decided_by)
    selection = FrameSelection(
        selected_idx=selected_idx,
        mode=mode,
        decided_by=decided_by,
        n_candidates=len(candidates),
        scores=scores,
        vlm_shortlist=shortlist,
        vlm_note=note,
    )
    if sel_cfg["write_debug_sheet"]:
        Path(cfg.s3_ground.out_dir).mkdir(parents=True, exist_ok=True)
        write_debug_sheet(bundle, scores, selected_idx, cfg.s3_ground.out_dir)
    return selection
