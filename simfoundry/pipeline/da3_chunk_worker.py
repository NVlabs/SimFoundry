# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-chunk DA3 worker.

Runs DepthAnything-V3 once on a directory of ``frame_*.png`` and writes the
standard ``results.npz`` (extrinsics/intrinsics/depth/conf/image) under
``<out-dir>/exports/npz/``.

Invoked per chunk by the chunk-capable ``DepthAnythingV3Backend`` in
``simfoundry/pipeline/depth_backends.py`` as a fresh subprocess
(``python -m simfoundry.pipeline.da3_chunk_worker ...``) so each chunk's
GPU memory is released on process exit — DA3-NESTED-GIANT-LARGE's per-layer
attention buffer otherwise saturates ~22 GiB on a 24 GB GPU.

Run from the ``da3`` env.
"""
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("da3_chunk_worker")

MODEL_NAME = "DA3NESTED-GIANT-LARGE-1.1"


def run_da3(frames_dir: Path, out_dir: Path, resolution: int, heavy_exports: bool = False) -> Path:
    """Run DA3 on every ``frame_*.png`` in @frames_dir; return the results.npz path."""
    image_fpaths = sorted(
        str(p) for p in frames_dir.iterdir()
        if p.suffix == ".png" and p.stem.startswith("frame_")
    )
    if not image_fpaths:
        raise SystemExit(f"No frame_*.png under {frames_dir}")

    import pycolmap  # noqa: F401 — must precede torch
    from simfoundry.models.depth_anything_v3 import DepthAnythingV3

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("DA3 on %d frames @ res %d -> %s", len(image_fpaths), resolution, out_dir)
    da = DepthAnythingV3(model_name=MODEL_NAME, device="cuda")
    da.infer(
        image_fpaths,
        str(out_dir),
        resolution=resolution,
        extrinsics=None,
        intrinsics=None,
        include_mesh=heavy_exports,
        include_gs=heavy_exports,
        include_gs_video=heavy_exports,
        include_colmap=heavy_exports,
    )
    return out_dir / "exports" / "npz" / "results.npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", required=True, help="Dir of frame_*.png to run DA3 on")
    ap.add_argument("--out-dir", required=True, help="DA3 output dir; results.npz lands under exports/npz/")
    ap.add_argument("--resolution", type=int, default=448, help="DA3 internal resolution (multiple of 14)")
    ap.add_argument("--heavy-exports", action="store_true",
                    help="Also export mesh/gs/colmap; off by default (not consumed downstream)")
    args = ap.parse_args()

    npz = run_da3(Path(args.frames_dir).resolve(), Path(args.out_dir).resolve(),
                  args.resolution, heavy_exports=args.heavy_exports)
    if not npz.exists():
        raise SystemExit(f"DA3 produced no results.npz at {npz}")
    logger.info("DA3 done -> %s", npz)


if __name__ == "__main__":
    main()
