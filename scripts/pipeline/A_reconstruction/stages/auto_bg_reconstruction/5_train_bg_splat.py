# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train splatfacto on the VOID-cleaned background sequence using DA3-supplied
poses + intrinsics (no COLMAP). Exports the trained splat to a PLY in DA3's
metric-meter world frame, ready for the bridge step.

Two phases run in sequence:

Phase 1 — build a nerfstudio transforms.json from:
  * DA3 results.npz: extrinsics (N,3,4) OpenCV world2cam, intrinsics (N,3,3) in
    DA3's internal pixel space (typically 252x448 at resolution 448).
  * VOID Pass 2 cleaned frames at 672x384.
  * Optionally SAM3 object masks (omitted in the canonical recipe — VOID
    frames have no foreground objects, so masks aren't needed).
  Intrinsics are rescaled from DA3's 252x448 pixel space to the inpainted
  frames' 672x384 space; DA3 world2cam extrinsics are inverted to nerfstudio
  cam2world (OpenGL) on write.

Phase 2 — `ns-train splatfacto` with dataparser flags that preserve DA3's
metric scale and disable any auto-reorientation (`auto_scale_poses=False,
center_method=none, orientation_method=none, scale_factor=1.0`), then
`ns-export gaussian-splat`.

Phase 1 runs in `simfoundry` (just numpy / cv2 / plyfile). Phase 2 is shelled out
via `mamba run -n nerfstudio_simfoundry ns-{train,export,viewer}` — the env
must already exist on `mamba env list` (see auto_bg_pipeline_setup_README.md
§4). The subprocess env is curated in `_ns_env()` for CUDA toolchain
correctness.

Canonical invocation (see auto_bg_pipeline_README.md step 3):
  mamba run -n simfoundry python \\
      scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/5_train_bg_splat.py \\
      scene_name=<scene> \\
      s5_train_bg_splat.no_masks=True s5_train_bg_splat.max_num_iterations=80000 \\
      s5_train_bg_splat.method=splatfacto-big \\
      s5_train_bg_splat.camera_optimizer_mode=SO3xR3
Reads config from scripts/cfg/auto_bg.yaml (Hydra), section `s5_train_bg_splat`.
"""
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
from plyfile import PlyData, PlyElement

from simfoundry.pipeline.stage_utils import bootstrap_hydra_workdir

bootstrap_hydra_workdir(__file__)

from simfoundry import CFG_DIR  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_bg_splat")

REPO_ROOT = Path(__file__).resolve().parents[5]

# Name of the nerfstudio env (assumed already created and on `mamba env list`).
# Overridable via NERFSTUDIO_ENV_NAME for non-standard installs.
NS_ENV_NAME = os.environ.get("NERFSTUDIO_ENV_NAME", "nerfstudio_simfoundry")


def build_seed_pointcloud(
    da3_npz: Path,
    masks_dir: Path | None,
    out_ply: Path,
    conf_min: float = 2.0,
    max_points: int = 200_000,
    rng_seed: int = 0,
    use_masks: bool = True,
) -> int:
    """Backproject DA3 depth maps into a world-frame seed point cloud.

    Splatfacto without a seed PLY random-inits gaussians in a cube whose default size
    (~5 m) dwarfs the cm-scale DA3 trajectory of nv_desk, so most random gaussians are
    never seen by any camera and hang around as low-opacity floaters. Seeding from
    DA3's own depth maps puts gaussians at real scene geometry from the start.

    Drops object pixels (SAM3 mask=255) so the seed is background-only and keeps
    points where DA3's per-pixel confidence is at least conf_min.

    Returns the number of points written.
    """
    d = np.load(da3_npz)
    ext = d["extrinsics"]    # (N, 3, 4) DA3 world2cam OpenCV (NOT cam2world; DA3 source confirms w2c)
    intr = d["intrinsics"]   # (N, 3, 3) at DA3 processed resolution
    depth = d["depth"]       # (N, H, W)
    img = d["image"]         # (N, H, W, 3) uint8 — DA3's resampled view of the input frame
    conf = d["conf"]         # (N, H, W) DA3 confidence (1 .. ~9, higher = better)

    N, H, W = depth.shape
    mask_paths = None
    if use_masks:
        if masks_dir is None:
            sys.exit("use_masks=True but masks_dir is None")
        mask_paths = sorted(p for p in masks_dir.iterdir() if p.suffix == ".png" and p.stem.startswith("frame_"))
        if len(mask_paths) != N:
            sys.exit(f"Mask count {len(mask_paths)} != DA3 frames {N}")

    # Pixel grid (shared across frames; DA3 is the same resolution for every frame).
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    chunks_xyz = []
    chunks_rgb = []
    for i in range(N):
        K = intr[i]
        # DA3 returns world2cam; invert to get cam2world for the backprojection below.
        T_w2c = np.eye(4, dtype=np.float64)
        T_w2c[:3, :4] = ext[i]
        T = np.linalg.inv(T_w2c)
        if use_masks:
            m = cv2.imread(str(mask_paths[i]), cv2.IMREAD_GRAYSCALE)
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_keep = (m <= 127)
        else:
            mask_keep = np.ones((H, W), dtype=bool)
        keep = mask_keep & (depth[i] > 0) & (conf[i] >= conf_min)
        if not keep.any():
            continue
        z = depth[i][keep].astype(np.float64)
        uu = u[keep].astype(np.float64)
        vv = v[keep].astype(np.float64)
        x_cam = (uu - K[0, 2]) * z / K[0, 0]
        y_cam = (vv - K[1, 2]) * z / K[1, 1]
        cam_pts = np.stack([x_cam, y_cam, z], axis=1)
        world_pts = cam_pts @ T[:3, :3].T + T[:3, 3]
        chunks_xyz.append(world_pts.astype(np.float32))
        chunks_rgb.append(img[i][keep])

    xyz = np.concatenate(chunks_xyz, axis=0)
    rgb = np.concatenate(chunks_rgb, axis=0)
    logger.info("Backprojected %d depth points (conf>=%.1f, background only)", len(xyz), conf_min)

    if len(xyz) > max_points:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(xyz), size=max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]
        logger.info("Subsampled to %d points", max_points)

    arr = np.empty(len(xyz), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    arr["x"] = xyz[:, 0]
    arr["y"] = xyz[:, 1]
    arr["z"] = xyz[:, 2]
    arr["red"] = rgb[:, 0]
    arr["green"] = rgb[:, 1]
    arr["blue"] = rgb[:, 2]
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(out_ply))
    logger.info("Wrote seed PLY %d points -> %s", len(xyz), out_ply)
    return len(xyz)


def opencv_to_opengl(ext_3x4: np.ndarray) -> np.ndarray:
    """DA3 extrinsic (OpenCV, x-right, y-down, z-forward) -> OpenGL (x-right, y-up, z-back).
    Flips the y and z columns of the rotation block. Camera center unchanged.
    ext_3x4: (3, 4) cam2world in OpenCV.
    Returns: (4, 4) cam2world in OpenGL.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :4] = ext_3x4.astype(np.float64)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    T[:, :4] = T[:, :4] @ flip  # flips the y and z columns of the rotation (and leaves t alone via the 4th column)
    return T


def opencv4x4_to_opengl(T_cv: np.ndarray) -> np.ndarray:
    """Same convention flip as opencv_to_opengl but for an already-4x4 cam2world matrix."""
    T = T_cv.astype(np.float64).copy()
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    T[:, :4] = T[:, :4] @ flip
    return T


def _write_per_frame_depth_npy(
    depth_npz: Path,
    image_hw: tuple[int, int],
    out_dir: Path,
    conf_min: float,
    n_expected: int,
) -> list[Path]:
    """Write per-frame depth NPYs (float32, training-image res) from a DA3 NPZ.

    Used by the dn-splatter codepath to supply depth GT to the depth-loss term.
    Pixels with DA3 conf < `conf_min` are set to 0.0 (not NaN) — dn-splatter masks
    pixels via `gt_depth > depth_tolerance` (default 0.1m), so 0-marked invalid
    pixels are excluded from the loss without producing NaN gradients.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(depth_npz)
    depth = d["depth"]  # (N, dH, dW)
    conf = d["conf"]    # (N, dH, dW)
    if depth.shape[0] != n_expected:
        sys.exit(f"depth NPZ has {depth.shape[0]} frames, expected {n_expected}")
    img_h, img_w = image_hw
    written = []
    for i in range(depth.shape[0]):
        d_i = depth[i].astype(np.float32).copy()
        d_i[conf[i] < conf_min] = 0.0
        if d_i.shape != (img_h, img_w):
            d_i = cv2.resize(d_i, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        out_path = out_dir / f"frame_{i:03d}.npy"
        np.save(out_path, d_i)
        written.append(out_path)
    logger.info("Wrote %d depth NPYs (conf>=%.1f, 0=invalid) -> %s", len(written), conf_min, out_dir)
    return written


def _write_transforms_common(
    frame_paths,
    mask_paths,
    cam2world_per_frame: np.ndarray,
    K_per_frame: np.ndarray,
    out_dir: Path,
    seed_ply_filename: str | None,
    use_masks: bool = True,
) -> Path:
    """Generic transforms.json builder for the DA3 path.

    cam2world_per_frame: (N, 4, 4) OpenCV cam2world.
    K_per_frame: (N, 3, 3) intrinsics in the inpainted-image pixel coords.
    use_masks: when False, skip writing mask_path entries (splatfacto trains on
        the full inpainted frames including ProPainter fills).
    """
    sample = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    img_h, img_w = sample.shape[:2]

    processed_images_dir = out_dir / "images"
    processed_images_dir.mkdir(parents=True, exist_ok=True)
    if use_masks:
        processed_masks_dir = out_dir / "masks"
        processed_masks_dir.mkdir(parents=True, exist_ok=True)

    frames_entries = []
    for i, fpath in enumerate(frame_paths):
        img_out = processed_images_dir / fpath.name
        if img_out.exists() or img_out.is_symlink():
            img_out.unlink()
        img_out.symlink_to(fpath.resolve())

        entry = {
            "file_path": f"images/{fpath.name}",
        }

        if use_masks:
            mpath = mask_paths[i]
            mask_out = processed_masks_dir / fpath.name
            m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            if m.shape[:2] != (img_h, img_w):
                m = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            m_ns = np.where(m > 127, np.uint8(0), np.uint8(255))
            cv2.imwrite(str(mask_out), m_ns)
            entry["mask_path"] = f"masks/{fpath.name}"

        K = K_per_frame[i]
        T = opencv4x4_to_opengl(cam2world_per_frame[i])
        entry.update({
            "transform_matrix": T.tolist(),
            "fl_x": float(K[0, 0]),
            "fl_y": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "w": img_w,
            "h": img_h,
        })
        frames_entries.append(entry)

    transforms = {
        "camera_model": "OPENCV",
        "orientation_override": "none",
        "frames": frames_entries,
    }
    if seed_ply_filename is not None:
        transforms["ply_file_path"] = seed_ply_filename
    (out_dir / "transforms.json").write_text(json.dumps(transforms, indent=2))
    logger.info("Wrote transforms.json with %d frames to %s", len(frames_entries), out_dir)
    return out_dir


def build_transforms(
    frames_dir: Path,
    masks_dir: Path,
    da3_npz: Path,
    out_dir: Path,
    seed_ply_filename: str = None,
    use_masks: bool = True,
) -> Path:
    """DA3 path: read NPZ, scale intrinsics from DA3 res to inpainted res, run common writer."""
    d = np.load(da3_npz)
    ext = d["extrinsics"]  # (N, 3, 4) OpenCV world2cam (DA3 convention; will invert to cam2world below)
    intr = d["intrinsics"]  # (N, 3, 3), for DA3 processed resolution (504x280 for nv_desk)
    da3_h, da3_w = d["image"].shape[1], d["image"].shape[2]

    frame_paths = sorted(p for p in frames_dir.iterdir() if p.suffix == ".png" and p.stem.startswith("frame_"))
    mask_paths = []
    if use_masks:
        mask_paths = sorted(p for p in masks_dir.iterdir() if p.suffix == ".png" and p.stem.startswith("frame_"))
    if len(frame_paths) != ext.shape[0]:
        sys.exit(f"Frame count {len(frame_paths)} != extrinsics {ext.shape[0]}")
    if use_masks and len(mask_paths) != ext.shape[0]:
        sys.exit(f"Mask count {len(mask_paths)} != extrinsics {ext.shape[0]}")

    sample = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    img_h, img_w = sample.shape[:2]
    sx = img_w / da3_w
    sy = img_h / da3_h
    logger.info("Image res %dx%d, DA3 res %dx%d, scale (%.3f, %.3f)", img_w, img_h, da3_w, da3_h, sx, sy)

    cam2world = np.zeros((ext.shape[0], 4, 4), dtype=np.float64)
    K_per_frame = np.zeros((ext.shape[0], 3, 3), dtype=np.float64)
    for i in range(ext.shape[0]):
        # DA3 returns world2cam; invert to get cam2world for transforms.json.
        T_w2c = np.eye(4, dtype=np.float64)
        T_w2c[:3, :4] = ext[i]
        cam2world[i] = np.linalg.inv(T_w2c)
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = intr[i, 0, 0] * sx
        K[1, 1] = intr[i, 1, 1] * sy
        K[0, 2] = intr[i, 0, 2] * sx
        K[1, 2] = intr[i, 1, 2] * sy
        K_per_frame[i] = K

    return _write_transforms_common(frame_paths, mask_paths, cam2world, K_per_frame, out_dir,
                                    seed_ply_filename, use_masks=use_masks)


def _ns_env(env: dict) -> dict:
    """Curate the environment for the ns-train / ns-export subprocess.

    Why this function exists: invoking ns-train from `simfoundry` leaks several env
    vars set by simfoundry's `cuda-nvcc` activate script (TORCH_CUDA_ARCH_LIST,
    NVCC_PREPEND_FLAGS, CC/CXX, GCC*, CFLAGS, ...). They point at simfoundry's gcc-13
    toolchain and arch list 10.0/10.1/12.0. The fixes below are empirically
    necessary for nerfstudio_simfoundry + torch 2.7.1+cu128 + gsplat 1.5.3:

      1. TORCHDYNAMO_DISABLE / TORCH_COMPILE_DISABLE — avoid splatfacto JIT
         issues.
      2. CPLUS_INCLUDE_PATH — conda's cuda-toolkit 12.8 puts CUDA headers in
         targets/x86_64-linux/include/ (not in include/), so the C++ host
         compiler can't find cuda_runtime_api.h without this path.
      3. TORCH_CUDA_ARCH_LIST=7.0..9.0;12.0 — include sm_120 (RTX 5090 /
         Blackwell) which gsplat 1.5.3 supports with torch 2.7.1+cu128.
      4. Strip every simfoundry-side compiler/toolchain var so gsplat's JIT picks
         up the nerfstudio env's own x86_64-conda-linux-gnu-cc toolchain.
    """
    import shutil as _shutil
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["TORCH_COMPILE_DISABLE"] = "1"
    # conda's cuda-toolkit puts CUDA headers in targets/x86_64-linux/include/
    # rather than the standard include/. The C++ host compiler needs this path
    # to find cuda_runtime_api.h when compiling gsplat's .cpp extension files.
    mamba_bin = _shutil.which("mamba") or _shutil.which("conda") or ""
    conda_base = os.path.dirname(os.path.dirname(mamba_bin)) if mamba_bin else ""
    ns_cuda_include = os.path.join(conda_base, "envs", NS_ENV_NAME,
                                   "targets", "x86_64-linux", "include")
    if os.path.isdir(ns_cuda_include):
        env["CPLUS_INCLUDE_PATH"] = f"{ns_cuda_include}:{env.get('CPLUS_INCLUDE_PATH', '')}"
    # torch 2.7.1+cu128 + gsplat 1.5.3 supports sm_120 (RTX 5090 / Blackwell).
    env["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;8.9;9.0;12.0"
    # Strip simfoundry-side compiler/toolchain vars and any stale CUDA_HOME so gsplat's
    # JIT uses the nerfstudio env's own nvcc and x86_64-conda-linux-gnu-cc.
    for k in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
              "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS",
              "CC", "CXX", "CC_FOR_BUILD", "CXX_FOR_BUILD",
              "GCC", "GCC_AR", "GCC_NM", "GCC_RANLIB", "CXXFILT",
              "CUDAARCHS", "CMAKE_ARGS", "CUDA_HOME"):
        env.pop(k, None)
    return env


@hydra.main(config_name="auto_bg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    sec = cfg.s5_train_bg_splat

    scene = cfg.scene_name
    # Canonical auto_bg layout defaults (Data/<scene>/auto_bg/...). All values come
    # from the cfg section, but the defaults match the canonical layout so a bare
    # `scene_name=<scene>` invocation works.
    ab = REPO_ROOT / "Data" / scene / "auto_bg"
    inpainted_dir = Path(sec.inpainted_dir) if sec.inpainted_dir else ab / "void" / "pass2" / "cleaned_frames"
    masks_dir = Path(sec.masks_dir) if sec.masks_dir else ab / "void" / "pass2" / "masks"
    da3_npz = Path(sec.da3_npz) if sec.da3_npz else ab / "da3" / "orig" / "results.npz"
    processed_dir = Path(sec.processed_dir) if sec.processed_dir else ab / "splat" / "ns_data"
    base_dir = Path(sec.base_dir) if sec.base_dir else ab / "splat"

    if sec.view_only:
        env = _ns_env(os.environ.copy())
        configs = sorted((base_dir / "outputs").glob("**/config.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not configs:
            sys.exit(f"No training config found under {base_dir}/outputs")
        viewer_cmd = [
            "mamba", "run", "-n", NS_ENV_NAME, "ns-viewer",
            "--load-config", str(configs[0]),
            "--viewer.websocket-port", str(sec.viewer_port),
        ]
        logger.info("Launching ns-viewer on %s. Open http://<host>:%d in a browser.",
                    configs[0], sec.viewer_port)
        logger.info("ns-viewer: %s", " ".join(viewer_cmd))
        subprocess.run(viewer_cmd, env=env, check=True)
        return

    required = (inpainted_dir, da3_npz)
    if not sec.no_masks:
        required = (*required, masks_dir)
    for p in required:
        if not p.exists():
            sys.exit(f"Missing input: {p}")

    if sec.no_seed_ply and sec.seed_ply_path is not None:
        sys.exit("--no-seed-ply and --seed-ply-path are mutually exclusive.")

    # E.1 Build transforms.json (and seed PLY).
    seed_ply_filename = None
    if not sec.no_seed_ply:
        seed_ply_filename = "points3d.ply"
        processed_dir.mkdir(parents=True, exist_ok=True)
        if sec.seed_ply_path is not None:
            seed_src = Path(sec.seed_ply_path)
            if not seed_src.exists():
                sys.exit(f"--seed-ply-path missing: {seed_src}")
            shutil.copy2(seed_src, processed_dir / seed_ply_filename)
            logger.info("Copied pre-built seed PLY %s -> %s", seed_src, processed_dir / seed_ply_filename)
        else:
            build_seed_pointcloud(
                da3_npz=da3_npz,
                masks_dir=masks_dir,
                out_ply=processed_dir / seed_ply_filename,
                conf_min=sec.seed_conf_min,
                max_points=sec.seed_max_points,
                use_masks=not sec.no_masks,
            )

    # Depth supervision codepath (env-var gate, see populate_modules patch in
    # nerfstudio's splatfacto.py). Writes per-frame depth NPYs ahead of training;
    # splatfacto.py side-loads them via NERFSTUDIO_DEPTH_DIR.
    depth_dir = None
    if sec.use_depth_loss:
        depth_npz_path = (
            Path(sec.depth_da3_npz) if sec.depth_da3_npz
            else REPO_ROOT / "Data" / scene / "auto_bg" / "da3" / "void" / "results.npz"
        )
        if not depth_npz_path.exists():
            sys.exit(f"--depth-da3-npz missing: {depth_npz_path}")
        # Need image res to resize depth — peek at the first inpainted frame.
        first_frame = next(p for p in sorted(inpainted_dir.iterdir())
                           if p.suffix == ".png" and p.stem.startswith("frame_"))
        _img = cv2.imread(str(first_frame), cv2.IMREAD_COLOR)
        img_h, img_w = _img.shape[:2]
        n_frames = sum(1 for p in inpainted_dir.iterdir()
                       if p.suffix == ".png" and p.stem.startswith("frame_"))
        _write_per_frame_depth_npy(
            depth_npz=depth_npz_path,
            image_hw=(img_h, img_w),
            out_dir=processed_dir / "depths",
            conf_min=sec.depth_loss_conf_min,
            n_expected=n_frames,
        )
        depth_dir = (processed_dir / "depths").resolve()

    build_transforms(inpainted_dir, masks_dir, da3_npz, processed_dir,
                     seed_ply_filename=seed_ply_filename,
                     use_masks=not sec.no_masks)
    if sec.skip_train:
        logger.info("Skip-train flag set; stopping after transforms.json.")
        return

    # E.2 Train + export
    base_dir.mkdir(parents=True, exist_ok=True)
    env = _ns_env(os.environ.copy())

    # Ensure no leftover gate from a prior shell sets MCMC unintentionally
    # (the splatfacto.py patch still reads NERFSTUDIO_MCMC; clear it here).
    env.pop("NERFSTUDIO_MCMC", None)
    env.pop("NERFSTUDIO_MCMC_CAP_MAX", None)
    env.pop("NERFSTUDIO_MCMC_NOISE_LR", None)
    logger.info("Strategy: DefaultStrategy (split/clone/prune).")

    if sec.use_depth_loss:
        env["NERFSTUDIO_DEPTH_LOSS"] = "1"
        env["NERFSTUDIO_DEPTH_LOSS_MULT"] = str(sec.depth_loss_mult)
        env["NERFSTUDIO_DEPTH_DIR"] = str(depth_dir)
        logger.info("Depth loss: ON (mult=%g, dir=%s)", sec.depth_loss_mult, depth_dir)
    else:
        for _k in ("NERFSTUDIO_DEPTH_LOSS", "NERFSTUDIO_DEPTH_LOSS_MULT", "NERFSTUDIO_DEPTH_DIR",
                   "NERFSTUDIO_DEPTH_LOSS_MIN"):
            env.pop(_k, None)

    # Use `mamba run -n <env>` so env discovery is mamba's job, not ours.
    # The env must already exist (see auto_bg_pipeline_setup_README.md §4).
    quit_on_done = "False" if sec.keep_viewer else "True"
    train_cmd = [
        "mamba", "run", "-n", NS_ENV_NAME, "ns-train", sec.method,
        "--max-num-iterations", str(sec.max_num_iterations),
        "--data", str(processed_dir.resolve()),
        "--vis", "viewer",
        "--viewer.websocket-port", str(sec.viewer_port),
        "--viewer.quit-on-train-completion", quit_on_done,
        # camera-optimizer.mode is model-side config; must precede the
        # dataparser subcommand boundary.
        f"--pipeline.model.camera-optimizer.mode={sec.camera_optimizer_mode}",
    ]
    train_cmd += [
        "nerfstudio-data",
        "--auto-scale-poses", "False",
        "--center-method", "none",
        "--orientation-method", "none",
        "--scale-factor", "1.0",
        # Use the seed PLY (if build_transforms wrote one) for clean init; only force-off
        # when --no-seed-ply was requested (so splatfacto falls back to random init).
        "--load-3D-points", "False" if sec.no_seed_ply else "True",
    ]
    logger.info("Web viewer: http://<host>:%d  (or http://localhost:%d if local). "
                "Viser binds to 0.0.0.0; for SSH use `ssh -L %d:localhost:%d <host>`.",
                sec.viewer_port, sec.viewer_port, sec.viewer_port, sec.viewer_port)
    logger.info("ns-train: %s", " ".join(train_cmd))
    subprocess.run(train_cmd, cwd=str(base_dir), env=env, check=True)

    if sec.no_export:
        logger.info("--no-export set; skipping ns-export.")
        return

    outputs_dir = base_dir / "outputs"
    configs = sorted(outputs_dir.glob("**/config.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not configs:
        sys.exit(f"No training config found under {outputs_dir}")
    load_config = configs[0]

    export_dir = base_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    # torch >=2.6 changed weights_only default to True, breaking nerfstudio's
    # torch.load call. Patch via a shim script instead of editing site-packages.
    shim = (
        "import torch, sys;"
        " _orig=torch.load;"
        " torch.load=lambda *a,**kw: _orig(*a,**{**{'weights_only':False},**kw});"
        f" sys.argv=['ns-export','gaussian-splat','--load-config',{str(load_config.resolve())!r},'--output-dir',{str(export_dir.resolve())!r}];"
        " from nerfstudio.scripts.exporter import entrypoint; entrypoint()"
    )
    export_cmd = ["mamba", "run", "-n", NS_ENV_NAME, "python", "-c", shim]
    logger.info("ns-export (shim): load_config=%s output_dir=%s", load_config, export_dir)
    subprocess.run(export_cmd, cwd=str(base_dir), env=env, check=True)

    # ns-export hardcodes `splat.ply` as the output filename. Rename it to
    # <scene>_bg.ply so the canonical artifact is scene-named (the bridge step
    # writes its pose sidecar next to this file).
    default_out = export_dir / "splat.ply"
    scene_named = export_dir / f"{scene}_bg.ply"
    if default_out.exists():
        if scene_named.exists():
            scene_named.unlink()
        default_out.rename(scene_named)
        logger.info("Renamed splat.ply -> %s", scene_named.name)
    logger.info("Done. trained splat at %s", scene_named)


if __name__ == "__main__":
    main()
