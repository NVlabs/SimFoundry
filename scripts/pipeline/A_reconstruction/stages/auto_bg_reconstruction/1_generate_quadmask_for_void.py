# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Step 1 of the auto_bg background pipeline: quadmask generation for VOID.

Generates a binary quadmask video from a keyframe → all-frames pipeline:
  1. Pick a keyframe (default: the first subsampled frame, index 0)
  2. Gemini object proposal on the keyframe (reuses prompt_object_setlist_specific_floorplane_v3)
  3. Per-object SAM3 segmentation on the keyframe (reuses sam_v3_gmask.SAM3)
  4. SAM2 video propagation from the keyframe forward + backward to cover all frames
  5. Union per-object masks per frame → binary mask
  6. Encode as grayscale mp4 (0=object, 255=bg) named quadmask_0.mp4

VOID's loader at videox_fun/utils/utils.py:386-393 quantizes the input mask into
the 4-value scheme {0, 63, 127, 255}. Binary {0, 255} → {0_obj, 255_bg}, which
is what we want for static-scene object removal (no causally-affected zones).

Reads config from scripts/cfg/auto_bg.yaml (Hydra), section `s1_quadmask`.
Run from simfoundry env (Hydra override syntax):
  mamba run -n simfoundry python \\
      scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/1_generate_quadmask_for_void.py \\
      scene_name=quillen_table floor_category="desk, table, or counter"
Per-stage values can be overridden directly, e.g. `s1_quadmask.gcloud_project=...`.

(Default s1_quadmask.keyframe_idx is 0; override only if frame 0 doesn't show
every object cleanly.)
"""
import json
import logging
import subprocess
import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from PIL import Image

from simfoundry.pipeline.stage_utils import bootstrap_hydra_workdir

# cd to scripts/cfg so the cfg's `${root_dir}` (= ../../Data) resolves to repo/Data,
# matching the sibling A_reconstruction stages.
bootstrap_hydra_workdir(__file__)

from simfoundry import CFG_DIR  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("generate_quadmask_for_void")


def _gemini_propose(keyframe_path: Path, gcloud_project: str, floor_category: str,
                    model_name: str):
    from simfoundry.models.vlm import Gemini
    from simfoundry.utils.prompt_utils import (
        prompt_object_setlist_specific_floorplane_v3,
        parse_json_response,
    )

    vlm = Gemini(project=gcloud_project, location="global", model=model_name)
    result = vlm(
        prompt=prompt_object_setlist_specific_floorplane_v3(floor_category),
        image_paths=str(keyframe_path),
        temperature=0,
        top_p=0,
        seed=0,
        print_results=False,
    )
    if result is None:
        raise SystemExit("Gemini returned no response")
    text = vlm.get_result_text(result)
    parsed = parse_json_response(text)
    objects = parsed.get("objects", [])
    detections = parsed.get("detections", [])
    if not objects and detections:
        objects = [d.get("name", "") for d in detections if d.get("name")]
    if not objects:
        raise SystemExit(f"Gemini returned no objects. Raw text:\n{text}")
    # Stage 5 stores the slash-separated full string and segments using the primary name only
    primary_names = [o.split(" / ")[0].strip() for o in objects]
    return primary_names, objects, detections


def _bbox_center_from_box2d(box_2d, img_w, img_h):
    """Gemini box_2d = [ymin, xmin, ymax, xmax] normalized 0-1000 → pixel center (x, y).
    Returns None if invalid. Mirrors stage 5's convert_gemini_box2d_to_xyxy."""
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
        return None
    try:
        y1n, x1n, y2n, x2n = [float(v) for v in box_2d]
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite([y1n, x1n, y2n, x2n])):
        return None
    y1, x1, y2, x2 = (int(y1n / 1000 * img_h), int(x1n / 1000 * img_w),
                      int(y2n / 1000 * img_h), int(x2n / 1000 * img_w))
    x_min, x_max = sorted((x1, x2))
    y_min, y_max = sorted((y1, y2))
    if x_max <= x_min or y_max <= y_min:
        return None
    cx = int(np.clip((x_min + x_max) // 2, 0, img_w - 1))
    cy = int(np.clip((y_min + y_max) // 2, 0, img_h - 1))
    return (cx, cy)


def _sam3_keyframe_masks(keyframe_pil: Image.Image, primary_names, detections, sam_conf=0.30):
    """For each primary_name, try SAM3 text-prompt; if empty, fall back to bbox-center
    point-prompt using the matching Gemini detection. Mirrors stage 5's pattern.
    """
    from simfoundry.models.sam_v3_gmask import SAM3
    sam3 = SAM3(confidence_threshold=sam_conf, device="cuda", video=False)
    W, H = keyframe_pil.size
    # Map primary name -> bbox center (if Gemini provided one)
    name_to_pt = {}
    for det in detections:
        full = det.get("name", "")
        if not full:
            continue
        primary = full.split(" / ")[0].strip()
        center = _bbox_center_from_box2d(det.get("box_2d"), W, H)
        if center is not None:
            name_to_pt[primary] = center

    obj_to_mask = {}
    for name in primary_names:
        masks, boxes, scores = sam3.predict_segmentation(pil_img=keyframe_pil, text_prompt=name)
        used_fallback = False
        if masks is None or len(masks) == 0 or (masks.shape[0] == 1 and masks.sum() == 0):
            # Fallback: point prompt at Gemini bbox center
            pt = name_to_pt.get(name)
            if pt is not None:
                logger.info("  SAM3 text empty for '%s' — falling back to point prompt at %s", name, pt)
                masks, boxes, scores = sam3.predict_segmentation_with_point(
                    pil_img=keyframe_pil, point_coords=pt, point_labels=[1], multimask_output=False,
                )
                used_fallback = True

        if masks is None or len(masks) == 0:
            logger.warning("  SAM3 found nothing for '%s' (text + point both failed)", name)
            continue
        best = int(np.argmax(scores))
        m = masks[best][0].astype(np.uint8)
        if m.sum() == 0:
            logger.warning("  '%s': empty mask, skipping", name)
            continue
        obj_to_mask[name] = m
        tag = " (fallback point)" if used_fallback else ""
        logger.info("  '%s'%s: mask area %d px (score %.2f)", name, tag, int(m.sum()), float(scores[best]))

    del sam3
    torch.cuda.empty_cache()
    return obj_to_mask


def _build_sam2_video_predictor():
    """Build SAM2 video predictor from local hydra config + checkpoint."""
    sam2_root = REPO_ROOT / "deps" / "Any6D" / "sam2"
    ckpt = sam2_root / "checkpoints" / "sam2.1_hiera_large.pt"
    if not ckpt.exists():
        ckpt = REPO_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
    if not ckpt.exists():
        raise SystemExit(f"sam2.1_hiera_large.pt not found at {ckpt}")
    sys.path.insert(0, str(sam2_root))
    # SAM2 builds its model via hydra.compose against its OWN packaged configs, but its
    # __init__ registers them (initialize_config_module("sam2")) only if Hydra isn't already
    # initialized. This stage is a @hydra.main app (global Hydra pinned to scripts/cfg), so
    # clear it first — otherwise sam2 skips registration and compose() can't find
    # configs/sam2.1/sam2.1_hiera_l.yaml. The auto_bg cfg values are already read into local
    # vars by now, so clearing the singleton is safe.
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor(
        config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
        ckpt_path=str(ckpt),
        device="cuda",
    )


def _sam2_propagate(video_mp4: Path, keyframe_idx: int, obj_to_mask, num_frames: int,
                    out_h: int, out_w: int):
    """
    For each object, init SAM2 with its keyframe mask, propagate forward
    (keyframe -> end) and reverse (keyframe -> 0). Union all per-frame masks.

    Returns: (num_frames, H, W) uint8 array — 255 where any object, else 0.
    """
    predictor = _build_sam2_video_predictor()
    union = np.zeros((num_frames, out_h, out_w), dtype=np.uint8)

    for obj_id, (name, mask) in enumerate(obj_to_mask.items(), start=1):
        logger.info("[SAM2 propagate] obj %d/%d: '%s'", obj_id, len(obj_to_mask), name)
        # New inference state per object — simpler than batching, and SAM2 video
        # state is small enough that we can re-init per-object cheaply.
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(
                video_path=str(video_mp4),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )
            # SAM2's load_video_frames may resize — confirm we know the resolution
            n = state["num_frames"]
            assert n == num_frames, f"SAM2 saw {n} frames, expected {num_frames}"

            # Resize the keyframe mask to SAM2's video_height x video_width if needed
            vh, vw = state["video_height"], state["video_width"]
            if mask.shape != (vh, vw):
                mask_rs = cv2.resize(mask, (vw, vh), interpolation=cv2.INTER_NEAREST)
            else:
                mask_rs = mask
            mask_t = torch.from_numpy(mask_rs.astype(bool))

            predictor.add_new_mask(state, frame_idx=keyframe_idx, obj_id=obj_id, mask=mask_t)

            # Forward (keyframe -> end)
            for fidx, _, mlogits in predictor.propagate_in_video(state, start_frame_idx=keyframe_idx, reverse=False):
                # mlogits: (n_obj, 1, H, W) float
                m = (mlogits[0, 0] > 0).cpu().numpy().astype(np.uint8) * 255
                if m.shape != (out_h, out_w):
                    m = cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                union[fidx] = np.maximum(union[fidx], m)

            # Reverse (keyframe -> 0)
            for fidx, _, mlogits in predictor.propagate_in_video(state, start_frame_idx=keyframe_idx, reverse=True):
                m = (mlogits[0, 0] > 0).cpu().numpy().astype(np.uint8) * 255
                if m.shape != (out_h, out_w):
                    m = cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                union[fidx] = np.maximum(union[fidx], m)

        torch.cuda.empty_cache()

    del predictor
    torch.cuda.empty_cache()
    return union


def _save_quadmask_mp4(union_uint8: np.ndarray, out_path: Path, fps: int):
    """Encode (T, H, W) uint8 (0/255) as a grayscale mp4 (yuv420p, 3-channel replicated)."""
    T, H, W = union_uint8.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Voids loader reads with media.read_video which yields RGB; replicate gray to 3 channels.
    rgb = np.repeat(union_uint8[..., None], 3, axis=-1)  # (T, H, W, 3)

    # Use ffmpeg to encode at near-lossless quality so 0/255 quantization survives.
    tmp_dir = out_path.parent / "_quadmask_frames_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i in range(T):
        Image.fromarray(rgb[i], mode="RGB").save(tmp_dir / f"{i:04d}.png")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(tmp_dir / "%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "0",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    # Cleanup tmp PNGs
    for p in tmp_dir.glob("*.png"):
        p.unlink()
    tmp_dir.rmdir()


@hydra.main(config_name="auto_bg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    sec = cfg.s1_quadmask

    in_dir = Path(sec.in_dir)
    sub_dir = in_dir / "subsampled"
    video_mp4 = in_dir / "input_video.mp4"
    if not video_mp4.exists():
        raise SystemExit(
            f"missing {video_mp4} — the orchestrator stages void/input from the canonical "
            f"s1_video (frames_subsampled_<N> + input_video.mp4). Run the canonical "
            f"reconstruction in splat-prep mode (s1_video.splat_prep=true) first."
        )

    frames = sorted(sub_dir.glob("*.png"))
    if not frames:
        raise SystemExit(f"no PNGs in {sub_dir}")
    num_frames = len(frames)
    keyframe_idx = max(0, min(sec.keyframe_idx, num_frames - 1))
    keyframe_path = frames[keyframe_idx]  # naming-agnostic (frames are frame_<i>.png)
    keyframe_pil = Image.open(keyframe_path).convert("RGB")
    out_w, out_h = keyframe_pil.size  # PIL: (W, H)
    logger.info("Loaded %d frames; keyframe = %s (%dx%d)",
                num_frames, keyframe_path.name, out_w, out_h)

    # Resolve gcloud project
    gcloud_project = sec.gcloud_project
    if gcloud_project is None:
        gcloud_project = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True, text=True
        ).stdout.strip()
        logger.info("Resolved gcloud project: %s", gcloud_project)

    # 1) Gemini propose object names
    logger.info("Running Gemini object proposal on keyframe...")
    primary_names, full_names, detections = _gemini_propose(
        keyframe_path, gcloud_project, sec.floor_category, sec.gemini_model
    )
    skip_set = set(sec.skip)
    primary_names = [n for n in primary_names if n not in skip_set]
    logger.info("Detected %d objects: %s", len(primary_names), primary_names)

    # 2) SAM3 segment each object on keyframe (with bbox-center point-prompt fallback)
    logger.info("Running SAM3 keyframe segmentation (text + point fallback)...")
    obj_to_mask = _sam3_keyframe_masks(
        keyframe_pil, primary_names, detections, sam_conf=sec.sam_conf
    )
    if not obj_to_mask:
        raise SystemExit("No objects segmented — quadmask would be empty. Aborting.")

    # 3) Save debug overlay
    debug_dir = in_dir / "keyframe_masks_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    overlay = np.array(keyframe_pil).copy()
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
    ]
    for i, (name, m) in enumerate(obj_to_mask.items()):
        color = np.array(palette[i % len(palette)], dtype=np.uint8)
        bool_m = m.astype(bool)
        overlay[bool_m] = (0.5 * overlay[bool_m] + 0.5 * color).astype(np.uint8)
        Image.fromarray((m * 255).astype(np.uint8), mode="L").save(
            debug_dir / f"obj{i:02d}_{name.replace(' ', '_')}.png"
        )
    Image.fromarray(overlay).save(debug_dir / "all_objects_overlay.png")
    logger.info("Wrote debug masks -> %s", debug_dir)

    # 4) SAM2 video propagation
    logger.info("Running SAM2 video propagation across %d frames (forward + reverse)...", num_frames)
    union_uint8 = _sam2_propagate(
        video_mp4=video_mp4,
        keyframe_idx=keyframe_idx,
        obj_to_mask=obj_to_mask,
        num_frames=num_frames,
        out_h=out_h,
        out_w=out_w,
    )

    # 5) Dilate per-frame BEFORE inversion. VOID's loader skips dilation in
    # quadmask mode (only mask_*.mp4 path runs read_mask_video_binary). Tight
    # SAM-precision boundaries → object-colored halo pixels survive → bad fills.
    if sec.mask_dilate_px > 0:
        k = sec.mask_dilate_px * 2 + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        # Per-frame cv2.dilate (vectorized would need scipy; loop is fast — sub-second even at 400 frames)
        for i in range(num_frames):
            union_uint8[i] = cv2.dilate(union_uint8[i], kernel)
        logger.info("Dilated obj mask by %d px before encoding (kernel %dx%d)",
                    sec.mask_dilate_px, k, k)

    # For VOID quadmask: 0 = obj, 255 = bg. We currently have 255 = obj → invert.
    quadmask = 255 - union_uint8
    n_pix_obj = int((quadmask == 0).sum())
    logger.info("Quadmask: %d total obj pixels across all frames (mean %.2f%% per frame)",
                n_pix_obj, n_pix_obj / (num_frames * out_h * out_w) * 100)

    out_mp4 = in_dir / "quadmask_0.mp4"
    _save_quadmask_mp4(quadmask, out_mp4, fps=sec.mp4_fps)
    logger.info("Wrote quadmask -> %s (%.1f MB)", out_mp4, out_mp4.stat().st_size / 1e6)

    # 6) Persist metadata
    meta = {
        "keyframe_idx": keyframe_idx,
        "floor_category": sec.floor_category,
        "gemini_model": sec.gemini_model,
        "sam_conf_threshold": sec.sam_conf,
        "mask_dilate_px": sec.mask_dilate_px,
        "objects": list(obj_to_mask.keys()),
        "objects_full": full_names,
        "skipped": list(skip_set),
        "num_frames": num_frames,
        "resolution_hw": [out_h, out_w],
    }
    (in_dir / "quadmask_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote quadmask_meta.json")

    # VOID's data loader (deps/void-model/videox_fun/utils/utils.py:337) reads
    # prompt.json for the post-removal background description. Write a default
    # derived from --floor-category if --bg-prompt isn't given.
    bg_prompt = sec.bg_prompt or (
        f"An empty {sec.floor_category}. No objects on top of the {sec.floor_category}."
    )
    prompt_path = in_dir / "prompt.json"
    prompt_path.write_text(json.dumps({"bg": bg_prompt}, indent=2))
    logger.info("Wrote prompt.json -> %r", bg_prompt)


if __name__ == "__main__":
    main()
