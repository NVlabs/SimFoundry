# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified depth backends used by stage-2 scripts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import logging
import os
import subprocess
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


def _plan_chunks(n: int, chunk_size: int, overlap: int):
    """Split N frames into overlapping chunks. Returns (starts, chunk_sizes)."""
    if n <= chunk_size:
        return [0], [n]
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
    starts = list(range(0, n - chunk_size + 1, step))
    if starts[-1] + chunk_size < n:
        starts.append(n - chunk_size)
    return starts, [chunk_size] * len(starts)


def _stage_chunk_frames(src_frames, dst_dir: Path):
    """Symlink @src_frames as frame_0000..frame_{k-1}.png in @dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in list(dst_dir.glob("frame_*.png")):
        f.unlink()
    for i, src in enumerate(src_frames):
        dst = dst_dir / f"frame_{i:04d}.png"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(Path(src).resolve())


def _merge_chunk_npzs(chunk_npzs, starts, chunk_sizes, n: int) -> dict:
    """Sim(3)-merge per-chunk DA3 NPZs into one N-frame world (chunk 0 = anchor).

    Aligns each later chunk into chunk 0's world via Umeyama on the overlap camera
    centers, applies the Sim(3) to non-overlap frames, and scales their metric depth
    by the fitted scale. Returns a dict of merged arrays (extrinsics/intrinsics/
    depth/conf/image).
    """
    from simfoundry.utils.transform_utils import (
        camera_centers_from_world2cam,
        umeyama_alignment,
    )

    ext_0 = chunk_npzs[0]["extrinsics"]
    intr_0 = chunk_npzs[0]["intrinsics"]
    img_0 = chunk_npzs[0]["image"]
    depth_0 = chunk_npzs[0]["depth"]
    conf_0 = chunk_npzs[0]["conf"]
    H_img, W_img = img_0.shape[1], img_0.shape[2]
    H_d, W_d = depth_0.shape[1], depth_0.shape[2]
    merged_ext = np.zeros((n, 3, 4), dtype=ext_0.dtype)
    merged_intr = np.zeros((n, 3, 3), dtype=intr_0.dtype)
    merged_image = np.zeros((n, H_img, W_img, 3), dtype=img_0.dtype)
    merged_depth = np.zeros((n, H_d, W_d), dtype=depth_0.dtype)
    merged_conf = np.zeros((n, H_d, W_d), dtype=conf_0.dtype)
    filled = np.zeros(n, dtype=bool)

    # Chunk 0 is the anchor world.
    for j in range(chunk_sizes[0]):
        g = starts[0] + j
        merged_ext[g] = ext_0[j]
        merged_intr[g] = intr_0[j]
        merged_image[g] = img_0[j]
        merged_depth[g] = depth_0[j]
        merged_conf[g] = conf_0[j]
        filled[g] = True

    # Later chunks: fit Sim(3) chunk-i world -> merged world over overlap cam centers.
    for i in range(1, len(starts)):
        si = starts[i]
        ci = chunk_npzs[i]
        ext_i, intr_i, img_i, depth_i, conf_i = (
            ci["extrinsics"], ci["intrinsics"], ci["image"], ci["depth"], ci["conf"]
        )
        prev_end = starts[i - 1] + chunk_sizes[i - 1] - 1
        ov_start_global = si
        ov_end_global = min(prev_end, si + chunk_sizes[i] - 1)
        ov_count = ov_end_global - ov_start_global + 1
        if ov_count < 3:
            raise SystemExit(f"Chunk {i} has only {ov_count} overlap frames with chunk {i-1}; need >=3 for Sim(3).")

        P = camera_centers_from_world2cam(ext_i[0:ov_count].astype(np.float64))
        Q = camera_centers_from_world2cam(merged_ext[ov_start_global:ov_end_global + 1].astype(np.float64))
        T, scale = umeyama_alignment(P, Q)
        pred = (T[:3, :3] @ P.T + T[:3, 3:4]).T
        res_norms = np.linalg.norm(pred - Q, axis=1)
        logger.info("Chunk %d -> merged: ov=%d, scale=%.6f, mean res=%.4f m, max res=%.4f m",
                    i, ov_count, scale, res_norms.mean(), res_norms.max())

        # Pure rotation (re-orthogonalized) + scale + translation from T.
        R_pure = T[:3, :3] / scale
        U, _, Vt = np.linalg.svd(R_pure)
        R_pure = U @ Vt
        if np.linalg.det(R_pure) < 0:
            U[:, -1] *= -1
            R_pure = U @ Vt
        t_T = T[:3, 3]

        for j in range(ov_count, chunk_sizes[i]):
            g = si + j
            if g >= n:
                break
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :4] = ext_i[j].astype(np.float64)
            c2w = np.linalg.inv(w2c)
            c2w_new = np.eye(4, dtype=np.float64)
            c2w_new[:3, :3] = R_pure @ c2w[:3, :3]
            c2w_new[:3, 3] = scale * (R_pure @ c2w[:3, 3]) + t_T
            w2c_new = np.linalg.inv(c2w_new)
            merged_ext[g] = w2c_new[:3, :4].astype(ext_0.dtype)
            merged_intr[g] = intr_i[j]
            merged_image[g] = img_i[j]
            # Depth is a metric distance; scale by the Sim(3) scale factor.
            merged_depth[g] = (depth_i[j].astype(np.float64) * scale).astype(depth_0.dtype)
            merged_conf[g] = conf_i[j]
            filled[g] = True

    missing = np.where(~filled)[0]
    if len(missing):
        raise SystemExit(f"Frames not filled after merge: {missing.tolist()}")

    return dict(
        extrinsics=merged_ext,
        intrinsics=merged_intr,
        depth=merged_depth,
        conf=merged_conf,
        image=merged_image,
    )


@dataclass
class DepthRunResult:
    backend: str
    output_dir: str
    n_inputs: int


class DepthBackend(ABC):
    """Model backend contract for depth/disparity stage."""

    @abstractmethod
    def run(self, cfg) -> DepthRunResult:
        raise NotImplementedError


class DepthAnythingV3Backend(DepthBackend):
    """DA3 backend (video RGB input). Single-pass for small N; chunked + Sim(3)-merged
    for long sequences (configurable via ``s2_da.chunk_*``)."""

    MODEL_NAME = "DA3NESTED-GIANT-LARGE-1.1"

    def run(self, cfg) -> DepthRunResult:
        s2 = cfg.s2_da
        # Frame source: explicit override (e.g. the auto_bg void run points at
        # cleaned_frames) else the canonical subsampled dir.
        frames_dir = OmegaConf.select(s2, "frames_dir")
        if frames_dir:
            img_dir = str(frames_dir)
        else:
            img_dir = f"{cfg.s1_video.out_dir}/frames_subsampled_{cfg.s1_video.n_subsampled_frames}"
        image_fpaths = sorted(
            f"{img_dir}/{fname}" for fname in os.listdir(img_dir) if fname.endswith(".png")
        )
        n = len(image_fpaths)
        if n == 0:
            raise SystemExit(f"No .png frames under {img_dir}")

        da_dir = Path(s2.out_dir) / "da"
        da_dir.mkdir(parents=True, exist_ok=True)
        resolution = s2.resolution
        heavy = bool(OmegaConf.select(s2, "heavy_exports") or False)
        chunk_size = int(OmegaConf.select(s2, "chunk_size") or 220)
        overlap = int(OmegaConf.select(s2, "overlap") or 40)
        threshold = OmegaConf.select(s2, "chunk_threshold")
        threshold = int(threshold) if threshold is not None else chunk_size

        if n <= threshold:
            # Single-pass, in-process (legacy path; identical recipe, just no chunking).
            import pycolmap  # noqa: F401 - must happen before torch
            from simfoundry.models.depth_anything_v3 import DepthAnythingV3

            logger.info("Running DA3 (single-pass) on %d frames @ res %d", n, resolution)
            da = DepthAnythingV3(model_name=self.MODEL_NAME, device="cuda")
            da.infer(
                image_fpaths,
                str(da_dir),
                resolution=resolution,
                extrinsics=None,
                intrinsics=None,
                include_mesh=heavy,
                include_gs=heavy,
                include_gs_video=heavy,
                include_colmap=heavy,
            )
            return DepthRunResult(backend="da3", output_dir=str(da_dir), n_inputs=n)

        # Chunked path: per-chunk fresh subprocess (frees GPU VRAM on exit) + Sim(3) merge.
        starts, sizes = _plan_chunks(n, chunk_size, overlap)
        logger.info("DA3 chunked: %d frames -> %d chunks (size=%d, overlap=%d) @ res %d",
                    n, len(starts), chunk_size, overlap, resolution)
        rerun_chunks = bool(OmegaConf.select(s2, "rerun_chunks") or False)
        work = da_dir / "_chunks"
        all_frames = [Path(p) for p in image_fpaths]
        chunk_npzs = []
        for i, (s, c) in enumerate(zip(starts, sizes)):
            cdir = work / f"chunk_{i:02d}" / "frames"
            cout = work / f"chunk_{i:02d}" / "da"
            result_npz = cout / "exports" / "npz" / "results.npz"
            if result_npz.exists() and not rerun_chunks:
                logger.info("[chunk %d] reusing %s", i, result_npz)
            else:
                _stage_chunk_frames(all_frames[s:s + c], cdir)
                logger.info("[chunk %d] DA3 on frames %d..%d", i, s, s + c - 1)
                env = os.environ.copy()
                env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                cmd = [
                    sys.executable, "-m", "simfoundry.pipeline.da3_chunk_worker",
                    "--frames-dir", str(cdir.resolve()),
                    "--out-dir", str(cout.resolve()),
                    "--resolution", str(resolution),
                ]
                if heavy:
                    cmd.append("--heavy-exports")
                subprocess.run(cmd, check=True, env=env)
                if not result_npz.exists():
                    raise SystemExit(f"[chunk {i}] DA3 produced no results.npz at {result_npz}")
            chunk_npzs.append(np.load(str(result_npz)))

        merged = _merge_chunk_npzs(chunk_npzs, starts, sizes, n)
        out_npz = da_dir / "exports" / "npz" / "results.npz"
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(out_npz), **merged)
        logger.info("Wrote merged %d-frame DA3 NPZ -> %s (image %s, depth %s)",
                    n, out_npz, merged["image"].shape, merged["depth"].shape)
        return DepthRunResult(backend="da3", output_dir=str(da_dir), n_inputs=n)


class FoundationStereoBackend(DepthBackend):
    """FoundationStereo backend (stereo RGB input)."""

    def run(self, cfg) -> DepthRunResult:
        import sys
        import imageio
        import cv2
        import numpy as np
        import open3d as o3d
        from PIL import Image
        import torch

        fs_dir = "../../deps/FoundationStereo"
        sys.path.append(fs_dir)
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder
        from Utils import set_logging_format, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud

        img_dir = cfg.s1_zed.out_dir
        names = sorted(
            {
                file.split(suffix)[0]
                for file in os.listdir(img_dir)
                for suffix in ["_l.png", "_r.png"]
                if file.endswith(suffix)
            }
        )

        out_dir = Path(cfg.s2_fs.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        set_logging_format()
        set_seed(cfg.seed)
        torch.autograd.set_grad_enabled(False)
        ckpt_dir = f"{fs_dir}/pretrained_models/23-51-11/model_best_bp2.pth"

        fs_cfg = OmegaConf.load(f"{os.path.dirname(ckpt_dir)}/cfg.yaml")
        if "vit_size" not in fs_cfg:
            fs_cfg["vit_size"] = "vitl"

        user_fs_cfg = OmegaConf.to_container(cfg.s2_fs.fs, resolve=True)
        fs_cfg.update(user_fs_cfg)
        fs_cfg["intrinsic_file"] = f"{img_dir}/intrinsic.txt"
        fs_cfg["ckpt_dir"] = ckpt_dir

        args = OmegaConf.create(fs_cfg)
        model = FoundationStereo(args)
        ckpt = torch.load(ckpt_dir, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.cuda()
        model.eval()

        for name in names:
            img0 = imageio.imread(f"{img_dir}/{name}_l.png")
            img1 = imageio.imread(f"{img_dir}/{name}_r.png")
            scale = fs_cfg["scale"]
            assert scale <= 1, "scale must be <=1"
            img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
            img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
            h, w = img0.shape[:2]
            img0_ori = img0.copy()

            img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
            img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
            padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
            img0_t, img1_t = padder.pad(img0_t, img1_t)

            with torch.cuda.amp.autocast(True):
                if not args.hiera:
                    disp = model.forward(img0_t, img1_t, iters=args.valid_iters, test_mode=True)
                else:
                    disp = model.run_hierachical(
                        img0_t,
                        img1_t,
                        iters=args.valid_iters,
                        test_mode=True,
                        small_ratio=0.5,
                    )
            disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(h, w)
            vis_disp = vis_disparity(disp)
            vis = np.concatenate([img0_ori, vis_disp], axis=1)
            imageio.imwrite(f"{out_dir}/{name}_vis.png", vis)
            imageio.imwrite(f"{out_dir}/{name}_disp.png", vis_disp)

            if args.remove_invisible:
                yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing="ij")
                us_right = xx - disp
                invalid = us_right < 0
                disp[invalid] = np.inf

            if args.get_pc:
                with open(args.intrinsic_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    k = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
                    baseline = float(lines[1])
                k[:2] *= scale
                depth = k[0, 0] * baseline / disp
                depth[(depth < 0) | (depth > 1e2)] = 0
                depth = depth.clip(0, 1e2)
                Image.fromarray(img0_ori).save(f"{out_dir}/{name}_rgb.png")
                np.save(f"{out_dir}/{name}_rgb.npy", img0_ori)
                np.save(f"{out_dir}/{name}_depth_meter.npy", depth)
                np.save(f"{out_dir}/{name}_K.npy", k)
                xyz_map = depth2xyzmap(depth, k)
                pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3))
                o3d.io.write_point_cloud(f"{out_dir}/{name}_pc_raw.ply", pcd)
                keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (np.asarray(pcd.points)[:, 2] <= args.z_far)
                keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
                pcd = pcd.select_by_index(keep_ids)
                o3d.io.write_point_cloud(f"{out_dir}/{name}_pc_1_pruned.ply", pcd)
                if args.denoise_cloud:
                    _, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
                    inlier_cloud = pcd.select_by_index(ind)
                    o3d.io.write_point_cloud(f"{out_dir}/{name}_pc_2_denoised.ply", inlier_cloud)

        return DepthRunResult(backend="fs", output_dir=str(out_dir), n_inputs=len(names))


BACKENDS = {
    "fs": FoundationStereoBackend,
    "da3": DepthAnythingV3Backend,
}


def create_backend(name: str) -> DepthBackend:
    if name not in BACKENDS:
        raise ValueError(f"Unknown depth backend '{name}'. Valid backends: {sorted(BACKENDS)}")
    return BACKENDS[name]()
