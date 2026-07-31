# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Chunked VOID Pass 1 inpainting (canonical Pass 1 driver for auto-bg).

VOID's VAE temporal cap is 197 frames per call (`videox_fun/utils/utils.py:
temporal_padding` silently truncates beyond that). To support the canonical
400-frame subsamples we slice the input video and its quadmask into
overlapping ≤173-frame chunks (173 keeps a margin under 197 while satisfying
the VAE's `(L-1) % 4 == 0` constraint), run `predict_v2v.py` per chunk in
the `void` env, and linearly cross-fade the chunk outputs into one long
Pass 1 mp4 — written to `<in-dir>/pass1/pass1.mp4` which is exactly what
`run_void_pass2.py` reads next.

Reads config from scripts/cfg/auto_bg.yaml (Hydra), section `s2_pass1`.
Run from simfoundry env (Hydra override syntax):
  python scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/2_run_void_pass1.py \\
      scene_name=<scene>
Per-stage values can be overridden directly, e.g. `s2_pass1.in_dir=...`.

`s2_pass1.in_dir` resolves to an absolute path: the `void` env subprocess
starts in a different working directory, so relative paths break.
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

# cd to scripts/cfg so the cfg's `${root_dir}` (= ../../Data) resolves to repo/Data,
# matching the sibling A_reconstruction stages.
bootstrap_hydra_workdir(__file__)

from simfoundry import CFG_DIR  # noqa: E402
from simfoundry.utils.video_utils import read_video_frames, slice_video, slice_video_lossless, write_video  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[5]
VOID_ROOT = REPO_ROOT / "deps" / "void-model"

# The `void` env must already exist (`mamba env list`). We shell into it per
# chunk via `mamba run -n void python …`, so this script can stay in the simfoundry env.
VOID_ENV_NAME = "void"
PASS1_SCRIPT = VOID_ROOT / "inference" / "cogvideox_fun" / "predict_v2v.py"
PASS1_CONFIG = VOID_ROOT / "config" / "quadmask_cogvideox.py"

DEFAULT_W = 672
DEFAULT_H = 384
DEFAULT_FPS = 12
DEFAULT_CHUNK_SIZE = 173        # VAE cap is 197; 173 satisfies (L-1)%4==0 with margin.
DEFAULT_TEMPORAL_WINDOW = 85    # The transformer's training-time latent window.
DEFAULT_MIN_OVERLAP = 30        # Force more chunks if a linspace plan ends up thinner.


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("void_pass1_chunked")


def _stitch_with_crossfade(chunk_outputs, chunk_starts, chunk_lengths, total_frames):
    """Linear cross-fade over overlap regions. Same math as the Pass 2 driver."""
    H, W = chunk_outputs[0].shape[1:3]
    accum = np.zeros((total_frames, H, W, 3), dtype=np.float64)
    weight_sum = np.zeros((total_frames,), dtype=np.float64)

    starts = list(chunk_starts)
    ends = [s + n - 1 for s, n in zip(chunk_starts, chunk_lengths)]

    for ci, (start, length, frames) in enumerate(zip(chunk_starts, chunk_lengths, chunk_outputs)):
        end = start + length - 1
        prev_end = ends[ci - 1] if ci > 0 else -1
        next_start = starts[ci + 1] if ci + 1 < len(starts) else total_frames

        for t in range(start, end + 1):
            if t >= total_frames:
                break
            if t <= prev_end:
                num = t - start
                den = prev_end - start
                w = num / max(den, 1)
            elif t >= next_start:
                num = end - t
                den = end - next_start
                w = num / max(den, 1)
            else:
                w = 1.0
            local_idx = t - start
            accum[t] += w * frames[local_idx].astype(np.float64)
            weight_sum[t] += w

    if (weight_sum <= 0).any():
        bad = np.where(weight_sum <= 0)[0]
        raise SystemExit(f"Frames with no chunk coverage: {bad.tolist()}")
    out = (accum / weight_sum[:, None, None, None]).clip(0, 255).astype(np.uint8)
    return out


def _plan_chunks(n_total: int, chunk_size: int, min_overlap: int) -> list[int]:
    """np.linspace plan, but bump n_chunks until min adjacent overlap >= min_overlap."""
    if n_total <= chunk_size:
        return [0]
    n_chunks = max(2, (n_total + chunk_size - 1) // chunk_size)
    while True:
        starts = np.linspace(0, n_total - chunk_size, n_chunks).round().astype(int).tolist()
        starts = sorted(set(starts))
        if len(starts) < n_chunks:
            n_chunks += 1
            continue
        overlaps = [(s_prev + chunk_size) - s_curr for s_prev, s_curr in zip(starts[:-1], starts[1:])]
        if min(overlaps) >= min_overlap or n_chunks >= n_total // 2 + 1:
            return starts
        n_chunks += 1


def _run_pass1_chunk(in_dir: Path, video_name: str, chunk_id: int, start: int, count: int,
                     out_dir: Path, height: int, width: int, fps: int,
                     temporal_window: int, num_steps: int, guidance_scale: float, seed: int,
                     model_name: Path, model_checkpoint: Path) -> Path:
    """Stage chunk inputs, invoke Pass 1 (predict_v2v.py) in void env, return chunk output mp4."""
    chunk_seq = f"{video_name}_p1c{chunk_id:02d}"
    chunk_root = out_dir / "_chunks" / f"chunk_{chunk_id:02d}"
    chunk_data = chunk_root / "data" / chunk_seq
    chunk_out = chunk_root / "out"
    chunk_data.mkdir(parents=True, exist_ok=True)
    # Clean prior outputs so predict_v2v's index suffix (-0001, -0002, ...) doesn't drift.
    if chunk_out.exists():
        shutil.rmtree(chunk_out)
    chunk_out.mkdir(parents=True, exist_ok=True)

    src_input = in_dir / "input" / "input_video.mp4"
    src_mask = in_dir / "input" / "quadmask_0.mp4"
    src_prompt = in_dir / "input" / "prompt.json"
    for p in [src_input, src_mask, src_prompt]:
        if not p.exists():
            raise SystemExit(f"missing {p}")

    logger.info("[chunk %d] slicing inputs: start=%d count=%d", chunk_id, start, count)
    slice_video(src_input, chunk_data / "input_video.mp4", start, count, fps)
    slice_video_lossless(src_mask, chunk_data / "quadmask_0.mp4", start, count, fps)
    shutil.copy2(src_prompt, chunk_data / "prompt.json")

    cmd = [
        "mamba", "run", "-n", VOID_ENV_NAME, "python", str(PASS1_SCRIPT),
        f"--config={PASS1_CONFIG}",
        "--config.system.gpu_memory_mode=sequential_cpu_offload",
        f"--config.data.data_rootdir={chunk_root / 'data'}",
        f"--config.data.sample_size={height}x{width}",
        f"--config.data.fps={fps}",
        f"--config.data.max_video_length={count}",
        f"--config.experiment.run_seqs={chunk_seq}",
        f"--config.experiment.save_path={chunk_out}",
        f"--config.video_model.model_name={model_name}",
        f"--config.video_model.transformer_path={model_checkpoint}",
        f"--config.video_model.temporal_window_size={temporal_window}",
        f"--config.video_model.guidance_scale={guidance_scale}",
        f"--config.system.seed={seed}",
        f"--config.video_model.num_inference_steps={num_steps}",
    ]
    logger.info("[chunk %d] running Pass 1", chunk_id)
    proc = subprocess.run(cmd, cwd=str(VOID_ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"Pass 1 chunk {chunk_id} exited {proc.returncode}")

    # predict_v2v writes <seq>-fg=-1-NNNN.mp4; we cleaned chunk_out so 0001 is the one.
    candidates = sorted(
        p for p in chunk_out.glob(f"{chunk_seq}-fg=-1-*.mp4")
        if not p.name.endswith("_tuple.mp4")
    )
    if not candidates:
        raise SystemExit(f"chunk {chunk_id} produced no Pass 1 output under {chunk_out}")
    out_mp4 = candidates[-1]
    logger.info("[chunk %d] done -> %s", chunk_id, out_mp4)
    return out_mp4


@hydra.main(config_name="auto_bg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    sec = cfg.s2_pass1

    # Resolve VOID checkpoint paths if relative (YAML stores them VOID-root-relative).
    model_name = sec.model_name
    if model_name and not Path(model_name).is_absolute():
        model_name = str(VOID_ROOT / model_name)
    model_checkpoint = sec.model_checkpoint
    if model_checkpoint and not Path(model_checkpoint).is_absolute():
        model_checkpoint = str(VOID_ROOT / model_checkpoint)

    if sec.chunk_size > 197:
        raise SystemExit(f"chunk_size {sec.chunk_size} exceeds VOID's 197-frame VAE cap.")
    if (sec.chunk_size - 1) % 4 != 0:
        raise SystemExit(f"chunk_size must satisfy (L-1) %% 4 == 0 (e.g. 85, 173, 197); got {sec.chunk_size}.")

    in_dir = Path(sec.in_dir)
    # New canonical layout: <in-dir> is auto_bg/void/, with input/, pass1/, pass2/
    # underneath. _chunks/ + pass1.mp4 live in pass1/.
    pass1_dir = in_dir / "pass1"
    out_dir = Path(sec.out_dir) if sec.out_dir else (pass1_dir / "_chunks")
    out_dir.mkdir(parents=True, exist_ok=True)

    pass1_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = pass1_dir / "pass1.mp4"

    if sec.skip_if_exists and canonical_path.exists():
        logger.info("Found %s and skip_if_exists set; nothing to do.", canonical_path)
        return

    src_input = in_dir / "input" / "input_video.mp4"
    if not src_input.exists():
        raise SystemExit(f"missing {src_input}")
    cap = cv2.VideoCapture(str(src_input))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    logger.info("Source has %d frames", n_total)

    starts = _plan_chunks(n_total, sec.chunk_size, sec.min_overlap)
    counts = [min(sec.chunk_size, n_total)] * len(starts) if len(starts) == 1 else [sec.chunk_size] * len(starts)
    ends = [s + c - 1 for s, c in zip(starts, counts)]
    overlaps = [(starts[i] + counts[i]) - starts[i + 1] for i in range(len(starts) - 1)]
    logger.info("Chunk plan: %d chunks of %d frames; starts=%s; ends=%s; min overlap=%s",
                len(starts), sec.chunk_size, starts, ends, min(overlaps) if overlaps else "n/a")

    chunk_outputs = []
    for ci, (s, c) in enumerate(zip(starts, counts)):
        out_mp4 = _run_pass1_chunk(
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

    if len(starts) == 1:
        # Single-chunk degenerate case: chunk_outputs[0] may be padded by temporal_padding;
        # truncate to n_total before publishing.
        stitched = chunk_outputs[0][:n_total]
    else:
        logger.info("Stitching %d Pass 1 chunks via linear cross-fade...", len(chunk_outputs))
        stitched = _stitch_with_crossfade(chunk_outputs, starts, counts, total_frames=n_total)

    # Convenience copy + meta alongside the per-chunk outputs in pass1/_chunks/
    full_mp4 = out_dir / f"{sec.video_name}_pass1_full.mp4"
    write_video(stitched, full_mp4, fps=sec.fps)
    logger.info("Wrote %s (%d frames, %.1f MB)", full_mp4, stitched.shape[0],
                full_mp4.stat().st_size / 1e6)

    # Publish to the canonical Pass 1 path that Pass 2 chunked reads. Never silently overwrite.
    if canonical_path.exists():
        backup = canonical_path.with_suffix(".preChunked.mp4")
        canonical_path.rename(backup)
        logger.info("Renamed pre-existing %s -> %s", canonical_path, backup)
    shutil.copy2(full_mp4, canonical_path)
    logger.info("Published canonical Pass 1 mp4 -> %s", canonical_path)

    (out_dir / "pass1_run_meta.json").write_text(json.dumps({
        "video_name": sec.video_name,
        "n_total_frames": int(n_total),
        "chunks": [{"start": int(s), "count": int(c)} for s, c in zip(starts, counts)],
        "min_overlap": int(min(overlaps)) if overlaps else None,
        "chunk_size": int(sec.chunk_size),
        "temporal_window": int(sec.temporal_window),
        "sample_size_hw": [int(sec.height), int(sec.width)],
        "num_inference_steps": int(sec.num_inference_steps),
        "guidance_scale": float(sec.guidance_scale),
        "seed": int(sec.seed),
    }, indent=2))
    logger.info("Wrote pass1_run_meta.json")


if __name__ == "__main__":
    main()
