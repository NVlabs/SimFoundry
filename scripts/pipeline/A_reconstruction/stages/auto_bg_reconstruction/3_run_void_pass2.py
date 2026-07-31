# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Chunked VOID Pass 2 (warped-noise refinement) for videos longer than the
85-frame temporal window — canonical Pass 2 driver for auto-bg.

Pass 2's `inference_with_pass1_warped_noise.py` doesn't multidiffuse — it
expects `max_video_length == temporal_window_size` (85). For longer clips we
slice both the input video and its Pass 1 output (the stitched mp4 written by
`run_void_pass1.py`) into overlapping 85-frame chunks, run Pass 2 per
chunk in the `void` env, and linearly cross-fade the chunk outputs into one
full-length mp4 plus a `cleaned_frames/` PNG directory the rest of the
pipeline consumes.

Chunk plan example: 400-frame input → 5 chunks at 85 frames each with
linear cross-fade in the overlap zones; coverage 0..399 ✓.

Outputs:
  <out_dir>/<scene>_pass2_full.mp4              — stitched mp4
  <out_dir>/cleaned_frames/frame_<i:04d>.png    — per-frame PNGs (downstream input)
  <out_dir>/_chunks/                            — per-chunk Pass 2 intermediates

Reads config from scripts/cfg/auto_bg.yaml (Hydra), section `s3_pass2`.
Run from simfoundry env (subprocesses into void env per chunk; Hydra override syntax):
  mamba run -n simfoundry python \\
      scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/3_run_void_pass2.py \\
      scene_name=<scene>
Per-stage values can be overridden directly, e.g. `s3_pass2.in_dir=...`.
"""
import json
import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import hydra
import numpy as np

from simfoundry.pipeline.stage_utils import bootstrap_hydra_workdir

bootstrap_hydra_workdir(__file__)

from simfoundry import CFG_DIR  # noqa: E402
from simfoundry.utils.video_utils import read_video_frames, slice_video, slice_video_lossless, write_video  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[5]
VOID_ROOT = REPO_ROOT / "deps" / "void-model"

# The `void` env must already exist (`mamba env list`). We shell into it per
# chunk via `mamba run -n void python …`, so this script can stay in the simfoundry env.
VOID_ENV_NAME = "void"
PASS2_SCRIPT = VOID_ROOT / "inference" / "cogvideox_fun" / "inference_with_pass1_warped_noise.py"

DEFAULT_W = 672
DEFAULT_H = 384
DEFAULT_FPS = 12
DEFAULT_TEMPORAL_WINDOW = 85


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("void_pass2_chunked")


def _stitch_with_crossfade(chunk_outputs, chunk_starts, chunk_lengths, total_frames,
                           warmup_k: int = 5):
    """Linear cross-fade over overlap regions, with chunk-warm-up suppression.

    For each output frame t, sum_i w_i(t) * chunk_i[t - start_i] over chunks
    that contain t. Weights are 0 outside the chunk and ramp linearly to 1 in
    the *non-overlap interior* of each chunk.

    In an overlap [a, b] between chunks i (prev) and i+1 (curr):
      - default linear: w_curr(t) = (t - a) / (b - a)
      - with warm-up suppression: w_curr(t) = 0 for the first `warmup_k`
        frames of chunk i+1 inside the overlap, then a linear ramp from 0 to 1
        across the remaining (overlap_len - warmup_k) frames.

    The warm-up suppression exists because CogVideoX-Fun video-diffusion has
    weak temporal anchoring in the first ~5 frames of every clip (warm-up
    artifacts; the model invents object-like content because its temporal
    attention has nothing to attend to). Without suppression, those warm-up
    frames bleed into the cross-fade at up to 80% weight and produce visible
    hallucinations at every chunk boundary.

    Edge case: if `overlap_len <= warmup_k`, w_curr stays 0 across the whole
    overlap (chunk i wins the entire overlap) — safer than letting warm-up
    frames dominate.

    Set `warmup_k=0` to recover the original pure-linear cross-fade.
    """
    H, W = chunk_outputs[0].shape[1:3]
    accum = np.zeros((total_frames, H, W, 3), dtype=np.float64)
    weight_sum = np.zeros((total_frames,), dtype=np.float64)

    starts = list(chunk_starts)
    ends = [s + n - 1 for s, n in zip(chunk_starts, chunk_lengths)]

    def _curr_chunk_weight_in_prev_overlap(k_curr, overlap_max_k):
        """Weight for the CURRENT chunk inside its overlap with the previous chunk.

        k_curr ∈ [0, overlap_max_k]. The first warmup_k frames of the current
        chunk are suppressed (w=0); after that, weight ramps linearly to 1.
        If the overlap is too short to fit any ramp past warm-up
        (warmup_k >= overlap_max_k), the current chunk is fully suppressed
        across the overlap — the previous chunk will carry full weight.
        """
        if k_curr < warmup_k or warmup_k >= overlap_max_k:
            return 0.0
        return (k_curr - warmup_k) / (overlap_max_k - warmup_k)

    for ci, (start, length, frames) in enumerate(zip(chunk_starts, chunk_lengths, chunk_outputs)):
        end = start + length - 1
        # Identify overlap with previous chunk (i-1) and next chunk (i+1)
        prev_end = ends[ci - 1] if ci > 0 else -1
        next_start = starts[ci + 1] if ci + 1 < len(starts) else total_frames

        for t in range(start, end + 1):
            if t >= total_frames:
                break
            # Determine weight within this chunk based on where t lies.
            if t <= prev_end:
                # overlap with the previous chunk; current chunk is the "new"
                # one — apply warm-up suppression directly.
                k_curr = t - start
                overlap_max_k = prev_end - start
                w = _curr_chunk_weight_in_prev_overlap(k_curr, overlap_max_k)
            elif t >= next_start:
                # overlap with the next chunk; current chunk is the "old" one.
                # Compute the next chunk's weight using the SAME function (its
                # k=0 is at next_start), then take 1 - that so the two chunks
                # always sum to 1.0. This is what guarantees full coverage even
                # when warmup_k absorbs the entire overlap window.
                k_next = t - next_start
                overlap_max_k_next = end - next_start
                w_next = _curr_chunk_weight_in_prev_overlap(k_next, overlap_max_k_next)
                w = 1.0 - w_next
            else:
                # interior — full weight
                w = 1.0
            local_idx = t - start
            accum[t] += w * frames[local_idx].astype(np.float64)
            weight_sum[t] += w

    # Normalize. Any frame should have weight_sum > 0 (covered by ≥1 chunk).
    if (weight_sum <= 0).any():
        bad = np.where(weight_sum <= 0)[0]
        raise SystemExit(f"Frames with no chunk coverage: {bad.tolist()}")
    out = (accum / weight_sum[:, None, None, None]).clip(0, 255).astype(np.uint8)
    return out


def _run_pass2_chunk(in_dir: Path, video_name: str, chunk_id: int, start: int, count: int,
                     out_dir: Path, height: int, width: int, fps: int,
                     temporal_window: int, num_steps: int, guidance_scale: float, seed: int,
                     model_name: Path, model_checkpoint: Path):
    """Stage chunk inputs, invoke Pass 2 in void env, return chunk output mp4 path."""
    chunk_root = out_dir / "_chunks" / f"chunk_{chunk_id:02d}"
    chunk_data = chunk_root / "data" / video_name
    chunk_pass1 = chunk_root / "pass1"
    chunk_out = chunk_root / "out"
    chunk_noise = chunk_root / "noise_cache"
    chunk_data.mkdir(parents=True, exist_ok=True)
    chunk_pass1.mkdir(parents=True, exist_ok=True)
    chunk_out.mkdir(parents=True, exist_ok=True)
    chunk_noise.mkdir(parents=True, exist_ok=True)

    src_input = in_dir / "input" / "input_video.mp4"
    src_mask = in_dir / "input" / "quadmask_0.mp4"
    src_prompt = in_dir / "input" / "prompt.json"
    src_pass1 = in_dir / "pass1" / "pass1.mp4"
    for p in [src_input, src_mask, src_prompt, src_pass1]:
        if not p.exists():
            raise SystemExit(f"missing {p}")

    logger.info("[chunk %d] slicing inputs: start=%d count=%d", chunk_id, start, count)
    slice_video(src_input, chunk_data / "input_video.mp4", start, count, fps)
    slice_video_lossless(src_mask, chunk_data / "quadmask_0.mp4", start, count, fps)
    shutil.copy2(src_prompt, chunk_data / "prompt.json")
    slice_video(src_pass1, chunk_pass1 / f"{video_name}-fg=-1-0001.mp4", start, count, fps)

    cmd = [
        "mamba", "run", "-n", VOID_ENV_NAME, "python", str(PASS2_SCRIPT),
        "--video_name", video_name,
        "--data_rootdir", str(chunk_root / "data"),
        "--pass1_dir", str(chunk_pass1),
        "--output_dir", str(chunk_out),
        "--warped_noise_cache_dir", str(chunk_noise),
        "--model_name", str(model_name),
        "--model_checkpoint", str(model_checkpoint),
        "--max_video_length", str(temporal_window),
        "--temporal_window_size", str(temporal_window),
        "--height", str(height),
        "--width", str(width),
        "--guidance_scale", str(guidance_scale),
        "--num_inference_steps", str(num_steps),
        "--seed", str(seed),
        "--use_quadmask",
    ]
    logger.info("[chunk %d] running Pass 2", chunk_id)
    proc = subprocess.run(cmd, cwd=str(VOID_ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"Pass 2 chunk {chunk_id} exited {proc.returncode}")

    out_mp4 = chunk_out / f"{video_name}_warped_noise_inference.mp4"
    if not out_mp4.exists():
        raise SystemExit(f"chunk {chunk_id} did not produce {out_mp4}")
    logger.info("[chunk %d] done -> %s", chunk_id, out_mp4)
    return out_mp4


@hydra.main(config_name="auto_bg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    sec = cfg.s3_pass2

    # Resolve VOID checkpoint paths if relative.
    model_name = sec.model_name
    if model_name and not Path(model_name).is_absolute():
        model_name = str(VOID_ROOT / model_name)
    model_checkpoint = sec.model_checkpoint
    if model_checkpoint and not Path(model_checkpoint).is_absolute():
        model_checkpoint = str(VOID_ROOT / model_checkpoint)

    in_dir = Path(sec.in_dir)
    out_dir = Path(sec.out_dir) if sec.out_dir else (in_dir / "pass2")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Probe total frame count from the source input mp4
    src_input = in_dir / "input" / "input_video.mp4"
    if not src_input.exists():
        raise SystemExit(f"missing {src_input}")
    cap = cv2.VideoCapture(str(src_input))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    logger.info("Source has %d frames", n_total)

    W = sec.temporal_window
    if n_total <= W:
        logger.warning("Source ≤ temporal_window (%d ≤ %d). One chunk is enough — fall back to "
                       "running inference_with_pass1_warped_noise.py directly.", n_total, W)

    # Chunk plan. Spread n_chunks chunks of size W evenly across [0, n_total-W],
    # with at least 30% overlap between adjacent chunks for clean cross-fade.
    # For 173 frames, W=85 → 3 chunks at starts [0, 44, 88], 41-frame overlaps.
    if n_total <= W:
        starts = [0]
    else:
        # Smallest number of chunks needed for the W=85 windows to cover n_total
        n_chunks = max(2, (n_total + W - 1) // W)
        starts = np.linspace(0, n_total - W, n_chunks).round().astype(int).tolist()
        starts = sorted(set(starts))
    counts = [W] * len(starts)
    logger.info("Chunk plan: %d chunks, starts=%s, ends=%s",
                len(starts), starts, [s + W - 1 for s in starts])

    chunk_outputs = []
    for ci, (s, c) in enumerate(zip(starts, counts)):
        if sec.restitch_only:
            # Look up the existing per-chunk output mp4 (written by a prior run).
            existing = out_dir / "_chunks" / f"chunk_{ci:02d}" / "out" / f"{sec.video_name}_warped_noise_inference.mp4"
            if not existing.exists():
                raise SystemExit(
                    f"--restitch-only set but per-chunk output missing: {existing}. "
                    f"Did you previously run Pass 2 for this scene?"
                )
            logger.info("[chunk %d] RESTITCH-ONLY: loading existing %s", ci, existing)
            out_mp4 = existing
        else:
            out_mp4 = _run_pass2_chunk(
                in_dir=in_dir, video_name=sec.video_name, chunk_id=ci,
                start=s, count=c, out_dir=out_dir,
                height=sec.height, width=sec.width, fps=sec.fps,
                temporal_window=sec.temporal_window,
                num_steps=sec.num_inference_steps,
                guidance_scale=sec.guidance_scale, seed=sec.seed,
                model_name=Path(model_name),
                model_checkpoint=Path(model_checkpoint),
            )
        chunk_outputs.append(read_video_frames(out_mp4))

    # Chunk 0 has no prior chunk to substitute for its own warm-up frames.
    # Flag this so the operator knows the first --chunk-warmup-frames frames
    # of the stitched mp4 still contain warm-up artifacts (e.g. the kitchen
    # scene's frame-0 hallucinations). Subsequent chunk boundaries ARE fixed.
    if sec.chunk_warmup_frames > 0 and len(starts) > 1:
        logger.warning(
            "Chunk-0 warm-up note: the first ~%d frames of the stitched mp4 still "
            "carry CogVideoX-Fun warm-up artifacts (chunk 0 has no predecessor to "
            "blend with). All later chunk boundaries are now warm-up-suppressed.",
            sec.chunk_warmup_frames,
        )

    logger.info("Stitching %d chunks via linear cross-fade (warmup_k=%d)...",
                len(chunk_outputs), sec.chunk_warmup_frames)
    stitched = _stitch_with_crossfade(
        chunk_outputs, starts, counts, total_frames=n_total,
        warmup_k=sec.chunk_warmup_frames,
    )

    final_mp4 = out_dir / f"{sec.video_name}_pass2_full.mp4"
    write_video(stitched, final_mp4, fps=sec.fps)
    logger.info("Wrote final stitched mp4 -> %s (%d frames, %.1f MB)",
                final_mp4, stitched.shape[0], final_mp4.stat().st_size / 1e6)

    # Publish canonical Pass 2 path (orchestrator's run_or_skip marker). Mirrors
    # run_void_pass1.py's canonical-publish step.
    canonical_path = out_dir / "pass2.mp4"
    if canonical_path.exists():
        backup = canonical_path.with_suffix(".preChunked.mp4")
        canonical_path.rename(backup)
        logger.info("Renamed pre-existing %s -> %s", canonical_path, backup)
    shutil.copy2(final_mp4, canonical_path)
    logger.info("Published canonical Pass 2 mp4 -> %s", canonical_path)

    # Decompose into PNGs for downstream DA3
    cleaned_frames = out_dir / "cleaned_frames"
    cleaned_frames.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(stitched):
        cv2.imwrite(str(cleaned_frames / f"frame_{i:04d}.png"),
                    cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    logger.info("Wrote %d frame PNGs -> %s", stitched.shape[0], cleaned_frames)

    (out_dir / "pass2_run_meta.json").write_text(json.dumps({
        "video_name": sec.video_name,
        "n_total_frames": int(n_total),
        "chunks": [{"start": int(s), "count": int(c)} for s, c in zip(starts, counts)],
        "temporal_window": sec.temporal_window,
        "sample_size_hw": [sec.height, sec.width],
        "num_inference_steps": sec.num_inference_steps,
        "guidance_scale": sec.guidance_scale,
        "seed": sec.seed,
        "chunk_warmup_frames": int(sec.chunk_warmup_frames),
        "restitched_from_existing": bool(sec.restitch_only),
    }, indent=2))


if __name__ == "__main__":
    main()
